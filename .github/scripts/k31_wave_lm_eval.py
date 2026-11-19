"""Run the pinned GSM8K workload in waves, then capture a separate replay probe."""

import asyncio
import copy
import gzip
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import time
from pathlib import Path

DEADLINE = time.monotonic() + 900
RUN_DIR = Path("k31-wave-diagnostic").resolve()
TRACE_DIR = Path("/app/trace")
SERVER_LOG = Path("/tmp/atom_server.log")
INPUT_HASH = "73caa1949cd6d00eed1e0c4b0dccadef04f273556400bcab86ed38638e2dc190"
TASK_HASH = "3d03c05f5358b3d498e6dafc7845060bc3865730b6a01eb055bb23bbbb9e8e9e"
CLIENT_HASHES = {
    "lm_eval/models/api_models.py": "8da25fde90c9a14ee7c603177a08814a3ac57aed7f18f6a00e040771335212cf",
    "lm_eval/models/openai_completions.py": "0afb2771209f9261a2ec0146cbd01928f22747e994a25225c24fd046c5c138ff",
    "lm_eval/models/utils.py": "bf4c2b9690c653f40319b8e9e92c4da5905b70d3662de4eff81377b223a04739",
}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_hash(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def check_runtime():
    distribution = importlib.metadata.distribution("lm_eval")
    assert distribution.version == "0.4.13"
    for name, expected in CLIENT_HASHES.items():
        assert file_hash(Path(distribution.locate_file(name))) == expected, name
    print("Pinned lm_eval client modules match", flush=True)


def save(name, value):
    path = RUN_DIR / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def remaining(cap=900):
    seconds = min(cap, DEADLINE - time.monotonic())
    if seconds <= 0:
        raise TimeoutError("whole-client 900-second deadline")
    return seconds


if sys.argv[1:] == ["--check-runtime"]:
    check_runtime()
    raise SystemExit(0)

from aiohttp import ClientError
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI


@register_model("k31-wave-local-completions")
class WaveClient(LocalCompletionsAPI):
    instance = None

    async def get_batched_requests(
        self, requests, cache_keys, *, generate=True, ctxlens=None, **kwargs
    ):
        assert generate and len(requests) == len(cache_keys) == 1319
        assert self._batch_size == 1 and self._concurrent == 65
        assert not self.tokenized_requests and self._seed == 1234
        assert self._max_gen_toks == 256 and self.max_retries == 3
        assert list(requests) == [key[0] for key in cache_keys]
        assert input_hash(cache_keys) == INPUT_HASH, "full ordered input mismatch"
        WaveClient.instance = self
        self.probe = (
            list(requests[:65]),
            copy.deepcopy(cache_keys[:65]),
            copy.deepcopy(kwargs),
        )
        results = []
        for offset in range(0, len(requests), 65):
            keys = cache_keys[offset : offset + 65]
            event = {
                "wave": offset // 65,
                "count": len(keys),
                "input_hash": input_hash(keys),
                "started_at": time.time(),
            }
            save("active-wave.json", event)
            budget = remaining()
            batch = await asyncio.wait_for(
                super().get_batched_requests(
                    requests[offset : offset + 65],
                    keys,
                    generate=generate,
                    ctxlens=None if ctxlens is None else ctxlens[offset : offset + 65],
                    **kwargs,
                ),
                timeout=budget,
            )
            assert len(batch) == len(keys), "incomplete wave"
            results.extend(batch)
            event["completed_at"] = time.time()
            with (RUN_DIR / "waves.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            print(f"Completed wave {event['wave']}: {len(keys)} requests", flush=True)
        return results


def preserve_main(output_path):
    root = Path(output_path)
    results = list(root.rglob("results_*.json"))
    samples = list(root.rglob("samples_gsm8k_*.jsonl"))
    assert len(results) == len(samples) == 1, "missing/ambiguous main artifacts"
    result = json.loads(results[0].read_text(encoding="utf-8"))
    assert result["n-samples"]["gsm8k"]["effective"] == 1319
    assert result["task_hashes"]["gsm8k"] == TASK_HASH
    filters = {}
    with samples[0].open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            filters.setdefault(record["doc_id"], []).append(record["filter"])
    assert set(filters) == set(range(1319))
    assert all(
        sorted(items) == ["flexible-extract", "strict-match"]
        for items in filters.values()
    )
    shutil.copyfile(SERVER_LOG, RUN_DIR / "main-server.log")
    log = (RUN_DIR / "main-server.log").read_text(encoding="utf-8")
    rows = [
        int(x)
        for x in re.findall(r"Scheduled prefill batch: \d+ reqs, (\d+) new tokens", log)
    ]
    save(
        "main-completed.json",
        {
            "status": "completed",
            "completed_at": time.time(),
            "evaluated": 1319,
            "filter_records": 2638,
            "input_hash": INPUT_HASH,
            "task_hash": TASK_HASH,
            "scores": result["results"]["gsm8k"],
            "large_prefills": sum(x >= 8192 for x in rows),
            "artifacts": {str(p): file_hash(p) for p in [results[0], samples[0]]},
        },
    )


async def profile_request(session, action, cap):
    async with session.post(
        f"http://localhost:8000/{action}", timeout=remaining(cap)
    ) as response:
        response.raise_for_status()
        body = await response.json()
        assert body.get("status") == "success", body
        return body


async def stop_after_delay(session, log_offset):
    await asyncio.sleep(5)
    started = time.monotonic()
    response = await profile_request(session, "stop_profile", 300)
    stopped_at = time.time()
    save("probe-stop.json", {"response": response, "response_at": stopped_at})
    traces = await asyncio.to_thread(
        preserve_traces, log_offset, min(DEADLINE, started + 300)
    )
    return {
        "stop_response": response,
        "stop_response_at": stopped_at,
        "rank_traces": traces,
    }


def preserve_traces(log_offset, stop_deadline):
    expected = set(range(8))
    while True:
        with SERVER_LOG.open("rb") as stream:
            stream.seek(log_offset)
            log = stream.read().decode("utf-8", errors="replace")
        (RUN_DIR / "probe-server.log").write_text(log, encoding="utf-8")
        assert not re.search(
            r"profiler stop failed|failed to export trace", log
        ), "profiler error"
        stopped = {
            int(x) for x in re.findall(r"Rank (\d+): profiler stop completed", log)
        }
        if stopped == expected:
            break
        if time.monotonic() >= stop_deadline:
            raise TimeoutError(
                f"missing rank stop acknowledgments: {expected - stopped}"
            )
        time.sleep(min(0.2, remaining()))
    manifests = {}
    for rank in sorted(expected):
        remaining()
        directory = TRACE_DIR / f"rank_{rank}"
        files = list(directory.glob("*.pt.trace.json.gz"))
        assert len(files) == 1 and not list(
            directory.glob("*.tmp*")
        ), f"rank {rank} trace incomplete"
        path = files[0]
        assert (
            directory.resolve().parent == TRACE_DIR.resolve()
            and not directory.is_symlink()
        )
        assert path.resolve().parent == directory.resolve() and not path.is_symlink()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
        assert trace.get("traceEvents"), f"rank {rank} empty trace"
        target = RUN_DIR / "traces" / f"rank_{rank}" / path.name
        target.parent.mkdir(parents=True)
        shutil.copyfile(path, target)
        digest = file_hash(path)
        assert digest == file_hash(target), f"rank {rank} copied trace mismatch"
        manifests[str(rank)] = {
            "path": str(target.relative_to(RUN_DIR)),
            "sha256": digest,
            "events": len(trace["traceEvents"]),
        }
        del trace
        if time.monotonic() > stop_deadline:
            raise TimeoutError("trace validation/copy exceeded stop/export deadline")
    return manifests


async def replay_probe(client):
    import aiohttp

    assert not list(TRACE_DIR.rglob("*.pt.trace.json*")), "pre-existing trace files"
    log_offset = SERVER_LOG.stat().st_size
    requests, keys, kwargs = client.probe
    record = {
        "status": "started",
        "started_at": time.time(),
        "logical_requests": 65,
        "input_hash": input_hash(keys),
    }
    save("probe.json", record)
    async with aiohttp.ClientSession() as session:
        await profile_request(session, "start_profile", 30)
        stopping = asyncio.create_task(stop_after_delay(session, log_offset))
        budget = remaining()
        try:
            responses = await asyncio.wait_for(
                LocalCompletionsAPI.get_batched_requests(
                    client, requests, keys, generate=True, **kwargs
                ),
                timeout=budget,
            )
            save("probe-responses.json", responses)
        finally:
            record.update(await stopping)
            save("probe.json", record)
    assert len(responses) == 65, "incomplete probe responses"
    record.update(
        status="captured",
        completed_at=time.time(),
        branch_witness="pending trace analysis",
    )
    save("probe.json", record)


def main():
    from lm_eval._cli import HarnessCLI
    from lm_eval.utils import setup_logging

    check_runtime()
    RUN_DIR.mkdir()
    setup_logging()
    cli = HarnessCLI()
    args = cli.parse_args()
    cli.execute(args)
    preserve_main(args.output_path)
    assert WaveClient.instance is not None
    try:
        asyncio.run(replay_probe(WaveClient.instance))
    except (
        AssertionError,
        OSError,
        EOFError,
        ValueError,
        RuntimeError,
        ClientError,
    ) as error:
        save(
            "probe-inconclusive.json",
            {"reason": f"{type(error).__name__}: {error}", "at": time.time()},
        )
        print(
            f"Probe inconclusive; saved main accuracy is unchanged: {error}", flush=True
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
