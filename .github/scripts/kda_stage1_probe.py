"""#3501 only: a cold/resumed continuation pair without CPU-tier offload."""

import argparse
import ast
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path

BASE = "fb1304722a9f79cd16e7383b8b504ccb29851d77"
CANDIDATE = "05ce5a39e10fcf3605fbd798870a0a5ab2fe1107"
TREES = {
    "baseline": "89b9807886ba635ff254dde89c5b8af15df29f3c",
    "candidate": "35db1231c2ec0c1b9628446fd396f36233ab6a50",
}
TARGET = "/models/moonshotai/Kimi-K3.1-Pro"
DRAFT = "/models/Inferact/Kimi-K3.1-DSpark"
REVISIONS = {
    TARGET: "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e",
    DRAFT: "187f8f6f792d931af93f96c8d2271a084e00280d",
}
ARGS = {
    "model": TARGET,
    "draft_model": DRAFT,
    "trust_remote_code": True,
    "tensor_parallel_size": 8,
    "decode_context_parallel_size": 8,
    "pipeline_parallel_size": 1,
    "kv_cache_dtype": "bf16",
    "index_cache_dtype": "fp8",
    "block_size": 64,
    "enable_prefix_caching": True,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 16384,
    "max_model_len": 131072,
    "gpu_memory_utilization": 0.9,
    "state_checkpoint_interval_tokens": -1,
    "method": "dspark",
    "num_speculative_tokens": 7,
    "level": 3,
    "cudagraph_mode": "FULL",
    "kv_transfer_config": "{}",
}
ENV = {
    "ATOM_ENABLE_REPLAYSSM": "1",
    "ATOM_MONO_ENABLE": "0",
    "ATOM_GDN_SSM_DTYPE": "fp32",
    "ATOM_STATE_CHECKPOINT_DEMAND": "0",
    "ATOM_STATE_CHECKPOINT_END_MARGIN": "0",
}
PROMPT_TOKENS, OUTPUT_TOKENS = 1024, 32
PHASES = ("cold", "warm")
STEP = re.compile(
    r"Scheduled prefill batch: \d+ reqs, \d+ new tokens \(done: (\[.*?\]), new: (\[.*?\])\), req_ids: (\(.*\))"
)


class CoverageError(RuntimeError):
    pass


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def token_hash(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def parse_step(message):
    match = STEP.fullmatch(message)
    if not match:
        raise CoverageError("unrecognized scheduled-prefill record")
    done, new, ids = map(ast.literal_eval, match.groups())
    if len(done) != 1 or len(new) != 1 or len(ids) != 1:
        raise CoverageError("probe request was mixed with another request")
    if any(type(n) is not int or n < 0 for n in (*done, *new)) or new[0] == 0:
        raise CoverageError("invalid prefill extent")
    return {"done": done[0], "new": new[0], "sequence": str(ids[0])}


def emit(kind, **values):
    root = Path(os.environ["STAGE1_EVENTS"])
    phase = root / "phase"
    if not phase.exists():
        return
    row = dict(kind=kind, phase=phase.read_text().strip(), pid=os.getpid(), **values)
    with (root / f"events-{os.getpid()}.jsonl").open("a") as out:
        out.write(json.dumps(row, allow_nan=False) + "\n")


def operation(op):
    return {
        "units": [int(unit) for unit in op.unit_ids],
        "slot": int(op.dst_slot),
        "layout": op.layout_id,
        "bytes": int(op.total_bytes),
    }


def install_observers():
    # Spawned workers import this file too; hooks stay outside compiled model code.
    import torch

    import atom
    from atom.model_engine import scheduler
    from atom.model_engine.page_unit_checkpoint import PageUnitCheckpointStore
    from atom.model_ops.attentions.gdn_attn import GDNStateMixin

    expected = Path(os.environ["STAGE1_SOURCE"]).resolve() / "atom/__init__.py"
    if Path(atom.__file__).resolve() != expected:
        raise RuntimeError("ATOM imported from the wrong source tree")
    if getattr(GDNStateMixin.execute_paged_state_copies, "stage1_observer", False):
        return
    original_info = scheduler.logger.info

    def info(msg, *args, **kwargs):
        original_info(msg, *args, **kwargs)
        if (
            isinstance(msg, str)
            and msg.startswith("Scheduled prefill batch:")
            and (Path(os.environ["STAGE1_EVENTS"]) / "phase").exists()
        ):
            emit("prefill", **parse_step(msg))

    scheduler.logger.info = info
    begin = PageUnitCheckpointStore.begin_restore

    @functools.wraps(begin)
    def begin_restore(self, prefix_hash, dst_slot):
        result = begin(self, prefix_hash, dst_slot)
        if result is not None:
            emit(
                "queued_restore",
                prefix_hash=str(prefix_hash),
                operation=operation(result),
            )
        return result

    PageUnitCheckpointStore.begin_restore = begin_restore
    copy = GDNStateMixin.execute_paged_state_copies

    @functools.wraps(copy)
    def copied(self, store_ops, restore_ops, descriptor_slot=0):
        result = copy(self, store_ops, restore_ops, descriptor_slot)
        if restore_ops and (Path(os.environ["STAGE1_EVENTS"]) / "phase").exists():
            torch.cuda.current_stream().synchronize()
            emit(
                "restore_complete",
                rank=torch.distributed.get_rank(),
                replayssm=bool(self.replayssm),
                operations=[operation(op) for op in restore_ops],
            )
        return result

    copied.stage1_observer = True
    GDNStateMixin.execute_paged_state_copies = copied


if os.environ.get("STAGE1_EVENTS"):
    install_observers()


def check_models():
    fingerprints = {}
    for folder, revision in REVISIONS.items():
        root = Path(folder)
        if (root / ".hf-revision").read_text().strip() != revision:
            raise RuntimeError(f"missing or wrong cached revision: {folder}")
        if not any(
            (root / marker).is_file()
            for marker in (".atom_download_complete", ".download-complete")
        ):
            raise RuntimeError(f"model cache is not marked complete: {folder}")
        if not (root / "config.json").is_file() or not any(
            p.is_file() and p.stat().st_size > 0 for p in root.glob("*.safetensors")
        ):
            raise RuntimeError(f"model payload missing: {folder}")
        index = root / "model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards or any(
                not (root / shard).is_file() or (root / shard).stat().st_size == 0
                for shard in shards
            ):
                raise RuntimeError(f"incomplete model shards: {folder}")
        fingerprints[folder] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "config.json", index)
            if p.exists()
        }
    return fingerprints


def assets(output):
    import pydoc_data.topics

    from transformers import AutoTokenizer

    fingerprints = check_models()
    text = Path(pydoc_data.topics.__file__).read_bytes()
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET, trust_remote_code=True, local_files_only=True
    )
    tokens = tokenizer.encode(text.decode(), add_special_tokens=False)[:PROMPT_TOKENS]
    if len(tokens) != PROMPT_TOKENS:
        raise CoverageError("pydoc corpus cannot supply the controlled prompt")
    save(
        output,
        {
            "text_sha256": hashlib.sha256(text).hexdigest(),
            "revisions": REVISIONS,
            "model_files": fingerprints,
            "tokens": tokens,
            "sha256": token_hash(tokens),
        },
    )


def check_response(row):
    ids, scalars = row["token_ids"], row["logprobs"]
    counts = [len(ids), len(scalars), row["num_tokens_output"]]
    if row["finish_reason"] != "max_tokens" or counts != [OUTPUT_TOKENS] * 3:
        raise CoverageError("request did not finish with the requested output length")
    if any(type(token) is not int or token < 0 for token in ids):
        raise CoverageError("invalid completion token IDs")
    if not all(math.isfinite(p) and p <= 1e-6 for p in scalars):
        raise FloatingPointError("invalid target-model logprobs")


def first_token(row, streamed):
    check_response(row)
    if len(streamed) != 1:
        raise CoverageError("missing first streamed logprobs")
    first_id, record = streamed[0]
    if len(record) != 42:
        raise CoverageError("requested top-20 record was not returned")
    top = {int(t): float(p) for t, p in zip(record[2::2], record[3::2])}
    if len(top) != 20:
        raise CoverageError("duplicate top-token IDs")
    if not all(math.isfinite(value) for value in top.values()):
        raise FloatingPointError("non-finite first-token probability")
    if first_id != row["token_ids"][0] or record[0] != row["logprobs"][0]:
        raise CoverageError("streamed first record disagrees with final SDK output")
    return dict(row, token=first_id, logprob=float(record[0]), top=top)


def compare_responses(left, right, threshold=0.01):
    """Check complete continuations and first-position top-20 probability bounds."""
    values = [
        left["logprob"],
        right["logprob"],
        *left["top"].values(),
        *right["top"].values(),
    ]
    if (
        not left["top"]
        or not right["top"]
        or not all(math.isfinite(p) and p <= 1e-6 for p in values)
    ):
        return {"status": "fail", "reason": "invalid target-model logprobs"}
    a = {int(t): math.exp(p) for t, p in left["top"].items()}
    b = {int(t): math.exp(p) for t, p in right["top"].items()}
    lower, upper = [], []
    for token in a.keys() | b.keys():
        if token in a and token in b:
            lower.append(abs(a[token] - b[token]))
            upper.append(lower[-1])
        else:
            known, tail = (
                (a[token], min(b.values()))
                if token in a
                else (b[token], min(a.values()))
            )
            lower.append(max(0.0, known - tail))
            upper.append(max(known, abs(known - tail)))
    lo, hi = max(lower), max(upper)
    same = left["token_ids"] == right["token_ids"]
    selected, delta = {}, 0.0
    if same:
        pairs = list(zip(left["logprobs"], right["logprobs"], strict=True))
        delta = max(abs(math.exp(a) - math.exp(b)) for a, b in pairs)
        selected = {
            "max_selected_probability_delta": delta,
            "max_selected_logprob_delta": max(abs(a - b) for a, b in pairs),
        }
    status = "pass"
    if lo > threshold or delta > threshold:
        status = "fail"
    elif hi > threshold or not same or left["token"] != right["token"]:
        status = "inconclusive"
    return {
        "status": status,
        "max_top20_union_probability_delta_lower": lo,
        "max_top20_union_probability_delta_upper": hi,
        "same_token": left["token"] == right["token"],
        "top_overlap": len(a.keys() & b.keys()),
        "same_completion": same,
        **selected,
    }


def check_trace(arm, stage):
    check_response(stage["first"])
    steps = [event for event in stage["events"] if event["kind"] == "prefill"]
    if not steps or len({event["sequence"] for event in steps}) != 1:
        raise CoverageError("missing or mixed prefill steps")
    cursor = steps[0]["done"]
    for step in steps:
        if step["done"] != cursor or not 0 < step["new"] <= 16384:
            raise CoverageError("non-contiguous or oversized prefill step")
        cursor += step["new"]
    if cursor != PROMPT_TOKENS or stage["offload_delta"] != 0:
        raise CoverageError("incomplete prefill or unexpected offload")
    if stage["phase"] == "cold":
        expected = [512, 512] if arm == "baseline" else [1024]
        if (
            steps[0]["done"] != 0
            or stage["cached_delta"] != 0
            or [step["new"] for step in steps] != expected
        ):
            raise CoverageError(
                "cold step count did not exercise the intended checkpoint cut"
            )
        return
    if steps[0]["done"] != 512 or stage["cached_delta"] != 512:
        raise CoverageError("warm request did not resume an HBM prefix")
    queued = [
        event["operation"]
        for event in stage["events"]
        if event["kind"] == "queued_restore"
    ]
    completed = [
        event for event in stage["events"] if event["kind"] == "restore_complete"
    ]
    if not queued or {event["rank"] for event in completed} != set(range(8)):
        raise CoverageError("HBM restore was not observed on all TP ranks")
    for op in queued:
        if {event["rank"] for event in completed if op in event["operations"]} != set(
            range(8)
        ):
            raise CoverageError("an HBM image was not restored on every TP rank")
    for event in completed:
        if not event["replayssm"] or not event["operations"]:
            raise CoverageError("wrong ReplaySSM path or empty restore")
        for op in event["operations"]:
            if op not in queued or not op["units"] or min(op["units"]) <= 0:
                raise CoverageError("GPU restore does not match the queued HBM image")


def check_capture(result):
    arm = result["arm"]
    if (
        arm not in TREES
        or not result["complete"]
        or result["atom_tree"] != TREES[arm]
        or result["source"] != {"baseline": BASE, "candidate": CANDIDATE}[arm]
    ):
        raise CoverageError("incomplete or wrong-source capture")
    if [stage["phase"] for stage in result["stages"]] != list(PHASES):
        raise CoverageError("missing, duplicate or reordered probe requests")
    for stage in result["stages"]:
        check_trace(arm, stage)


def require_stable_baseline(arm, comparison):
    if arm == "baseline" and comparison["status"] != "pass":
        error = FloatingPointError if comparison["status"] == "fail" else CoverageError
        raise error("baseline cold/warm numerical control was not stable")


def capture(arm, inputs, output):
    import aiter
    import torch
    import triton

    import atom
    from atom import SamplingParams
    from atom.model_engine.arg_utils import EngineArgs

    for key, value in ENV.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"runtime setting differs: {key}")
    if "gd17ee8dfe" not in importlib.metadata.version("amd-aiter"):
        raise RuntimeError("wrong AITER wheel")
    data = json.loads(Path(inputs).read_text())
    if data["revisions"] != REVISIONS or data["model_files"] != check_models():
        raise RuntimeError("model cache changed after preflight")
    if (
        len(data["tokens"]) != PROMPT_TOKENS
        or token_hash(data["tokens"]) != data["sha256"]
    ):
        raise CoverageError("invalid prompt manifest")
    root = Path(os.environ["STAGE1_EVENTS"])
    result = {
        "arm": arm,
        "source": BASE if arm == "baseline" else CANDIDATE,
        "atom_tree": TREES[arm],
        "actual_atom_path": atom.__file__,
        "args": ARGS,
        "env": ENV,
        "inputs_sha256": hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "aiter": importlib.metadata.version("amd-aiter"),
            "aiter_path": aiter.__file__,
        },
        "stages": [],
        "complete": False,
    }
    save(output, result)
    engine = None
    try:
        engine = EngineArgs(**ARGS).create_engine()
        resolved = {
            key: getattr(engine.config, key)
            for key in (
                "tensor_parallel_size",
                "decode_context_parallel_size",
                "pipeline_parallel_size",
                "enable_prefix_caching",
                "kv_cache_dtype",
                "index_cache_dtype",
                "max_num_seqs",
                "max_num_batched_tokens",
                "state_checkpoint_interval_tokens",
            )
        }
        resolved["block_size"] = engine.config.kv_cache_block_size
        resolved["level"] = engine.config.compilation_config.level
        resolved["cudagraph_mode"] = (
            engine.config.compilation_config.cudagraph_mode.name
        )
        if resolved != {key: ARGS[key] for key in resolved}:
            raise RuntimeError(
                "resolved engine configuration differs from the probe contract"
            )
        result["resolved"] = resolved
        if (
            engine.config.kv_transfer_config
            or engine.config.speculative_config.synthetic_acceptance_rates is not None
            or engine.config.speculative_config.num_speculative_tokens != 7
        ):
            raise RuntimeError("offload or forced acceptance enabled")
        for phase in PHASES:
            temporary = root / "next-phase"
            temporary.write_text(phase)
            temporary.replace(root / "phase")
            before = engine.get_cache_statistics()
            streamed, outputs = [], {}

            def receive(sample, target=streamed):
                if not target and sample.logprobs is not None and len(sample.logprobs):
                    target.append(
                        (int(sample.output_tokens[0]), tuple(sample.logprobs[0]))
                    )

            params = SamplingParams(
                temperature=0,
                seed=0,
                max_tokens=OUTPUT_TOKENS,
                ignore_eos=True,
                logprobs=True,
                top_logprobs=20,
            )
            engine.core_mgr.reset_dp_router()
            with engine.core_mgr.offline_backpressure_batch():
                engine.add_request([data["tokens"]], params, stream_callback=[receive])
            while not engine.is_finished() and (
                engine.core_mgr.is_alive() or engine.core_mgr.is_rest()
            ):
                outputs.update(engine.io_processor.postprocess(engine.step()))
            rows = list(outputs.values())
            after = engine.get_cache_statistics()
            if len(rows) != 1 or rows[0]["num_tokens_input"] != PROMPT_TOKENS:
                raise CoverageError("response does not match the requested prompt")
            events = [
                json.loads(line)
                for path in root.glob("events-*.jsonl")
                for line in path.read_text().splitlines()
            ]
            stage = {
                "phase": phase,
                "prompt_sha256": data["sha256"],
                "first": first_token(rows[0], streamed),
                "events": [event for event in events if event["phase"] == phase],
                "cached_delta": after["cached_tokens"] - before["cached_tokens"],
                "offload_delta": after["offload_tokens"] - before["offload_tokens"],
            }
            result["stages"].append(stage)
            save(output, result)
            check_trace(arm, stage)
        engine.close()
        engine = None
        result["complete"] = True
        if data["model_files"] != check_models():
            raise RuntimeError("model cache changed during capture")
        result["within_arm"] = compare(result)
        save(output, result)
        require_stable_baseline(arm, result["within_arm"])
    finally:
        if engine is not None:
            engine.close()


def compare(base, candidate=None):
    pairs = []
    for result in (base,) if candidate is None else (base, candidate):
        check_capture(result)
        pairs.append((*result["stages"], result["arm"] + "/resume"))
    if candidate is not None:
        if (base["arm"], candidate["arm"]) != ("baseline", "candidate"):
            raise CoverageError("wrong capture arm")
        for key in ("args", "env", "inputs_sha256", "runtime", "resolved"):
            if base[key] != candidate[key]:
                raise CoverageError(f"unmatched pair: {key}")
        pairs.extend(
            (left, right, left["phase"])
            for left, right in zip(base["stages"], candidate["stages"])
        )
    rows = []
    for left, right, phase in pairs:
        if left["prompt_sha256"] != right["prompt_sha256"]:
            raise CoverageError("mismatched prompt identity")
        comparison = compare_responses(left["first"], right["first"])
        rows.append(dict(phase=phase, **comparison))
    status = (
        "fail"
        if any(row["status"] == "fail" for row in rows)
        else "inconclusive" if any(row["status"] != "pass" for row in rows) else "pass"
    )
    return {"status": status, "threshold": 0.01, "comparisons": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("assets", "capture", "compare"))
    parser.add_argument("--arm", choices=("baseline", "candidate"))
    parser.add_argument("--inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    try:
        if args.command == "assets":
            assets(args.output)
        elif args.command == "capture":
            capture(args.arm, args.inputs, args.output)
        else:
            result = compare(
                json.loads(Path(args.baseline).read_text()),
                json.loads(Path(args.candidate).read_text()),
            )
            save(args.output, result)
            return {"pass": 0, "fail": 1, "inconclusive": 3}[result["status"]]
    except Exception as exc:
        kind = "setup_or_runtime"
        if isinstance(exc, CoverageError):
            kind = "coverage"
        elif isinstance(exc, FloatingPointError):
            kind = "numerical"
        save(
            str(args.output) + ".failure.json",
            {"classification": kind, "error": repr(exc)},
        )
        if kind == "setup_or_runtime":
            raise
        return 3 if kind == "coverage" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
