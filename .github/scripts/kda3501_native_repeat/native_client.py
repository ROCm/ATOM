"""One baseline instance, three fixed requests, public native client calls only."""

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path


def save(output, name, data):
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    assert len(payload.encode()) <= 1024 * 1024, "client record exceeds 1 MiB"
    with (output / name).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def receive(sample, records):
    if not sample.output_tokens:
        assert not sample.logprobs
        return
    assert sample.logprobs is not None
    assert len(sample.output_tokens) == len(sample.logprobs)
    for token, row in zip(sample.output_tokens, sample.logprobs, strict=True):
        assert len(row) == 42 and len(set(row[2::2])) == 20
        assert all(math.isfinite(float(value)) for value in row)
        top = list(zip(map(int, row[2::2]), map(float, row[3::2]), strict=True))
        records.append(
            {
                "token": int(token),
                "selected": float(row[0]),
                "rank": int(row[1]),
                "top": top,
            }
        )
        assert len(records) <= 32, "duplicate or excess streamed output"


def main():
    data = json.loads(Path(__file__).with_name("inputs.json").read_text())
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    assert len(data["tokens"]) == 1024
    raw_tokens = json.dumps(data["tokens"], separators=(",", ":")).encode()
    assert hashlib.sha256(raw_tokens).hexdigest() == data["token_sha256"]
    proof_path = Path(sys.argv[2])
    proof = json.loads(proof_path.read_text())
    assert proof["tokens"] == data["tokens"] and proof["revisions"] == data["revisions"]
    draft = data["engine_args"]["draft_model"]
    assert proof["model_files"][draft]["resolved_path"] == draft
    assert all(
        os.environ.get(key) == value for key, value in data["environment"].items()
    )
    import torch
    import triton

    import atom
    from atom import SamplingParams
    from atom.model_engine.arg_utils import EngineArgs

    source = Path(os.environ["PR3501_SOURCE"]).resolve()
    assert Path(atom.__file__).resolve() == source / "atom/__init__.py"
    runtime = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "aiter": importlib.metadata.version("amd-aiter"),
    }
    assert runtime == {key: data["runtime"][key] for key in runtime}
    save(
        output,
        "client-start.json",
        {
            "source": data["source"],
            "atom_path": atom.__file__,
            "runtime": runtime,
            "args": data["engine_args"],
            "environment": data["environment"],
            "preflight_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        },
    )
    engine, phase = None, "initializing"
    records, outputs = [], {}
    try:
        start = time.monotonic()
        engine = EngineArgs(**data["engine_args"]).create_engine()
        save(output, "initialized.json", {"elapsed_seconds": time.monotonic() - start})
        assert not engine.config.kv_transfer_config
        assert engine.config.speculative_config.synthetic_acceptance_rates is None
        completions = []
        for phase in data["phases"]:
            before = {
                "cache": engine.get_cache_statistics(timeout=10),
                "mtp": engine.get_mtp_statistics(timeout=10),
            }
            records, outputs = [], {}
            params = SamplingParams(**data["sampling"])
            engine.core_mgr.reset_dp_router()
            start_request = time.monotonic()
            print(f"PR3501 phase {phase} start", flush=True)
            with engine.core_mgr.offline_backpressure_batch():
                engine.add_request(
                    [list(data["tokens"])],
                    params,
                    stream_callback=lambda sample, target=records: receive(
                        sample, target
                    ),
                )
            while not engine.is_finished() and (
                engine.core_mgr.is_alive() or engine.core_mgr.is_rest()
            ):
                assert (
                    time.monotonic() - start_request < 300
                ), "request exceeded 300 seconds"
                outputs.update(engine.io_processor.postprocess(engine.step()))
            after = {
                "cache": engine.get_cache_statistics(timeout=10),
                "mtp": engine.get_mtp_statistics(timeout=10),
            }
            rows = list(outputs.values())
            save(
                output,
                phase + ".json",
                {
                    "phase": phase,
                    "before": before,
                    "after": after,
                    "prompt_sha256": data["token_sha256"],
                    "outputs": rows,
                    "records": records,
                },
            )
            assert len(rows) == 1
            row = rows[0]
            assert (
                row["num_tokens_input"] == 1024 and row["finish_reason"] == "max_tokens"
            )
            assert [
                row["num_tokens_output"],
                len(records),
                len(row["token_ids"]),
                len(row["logprobs"]),
            ] == [32] * 4
            assert [record["token"] for record in records] == row["token_ids"]
            assert [record["selected"] for record in records] == row["logprobs"]
            assert (
                after["cache"]["offload_tokens"] - before["cache"]["offload_tokens"]
                == 0
            )
            expected_cache = 0 if phase == "C0" else 512
            assert (
                after["cache"]["cached_tokens"] - before["cache"]["cached_tokens"]
                == expected_cache
            )
            completions.append(row["token_ids"])
        save(
            output,
            "collection.json",
            {
                "collected": True,
                "qualification": "not_assessed",
                "C0_W0_same_ids": completions[0] == completions[1],
                "W0_W1_same_ids": completions[1] == completions[2],
            },
        )
    except BaseException as error:
        save(
            output,
            "failure.json",
            {
                "phase": phase,
                "error": repr(error),
                "outputs": list(outputs.values()),
                "records": records,
            },
        )
        raise
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
