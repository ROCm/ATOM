"""Validation-only receipt checks; no model launch, signaling or lease logic."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def response_check(request, response):
    assert response["prompt_token_ids"] == request["prompt_token_ids"], "wrong input IDs"
    assert response["usage"]["prompt_tokens"] == len(request["prompt_token_ids"])
    assert response["usage"]["completion_tokens"] == request["max_tokens"]
    assert len(response["choices"]) == 1
    assert response["choices"][0]["finish_reason"] == "length"
    assert isinstance(response["choices"][0]["text"], str)
    assert isinstance(response["id"], str) and response["id"], "missing native ID"


def compare(request, reference, response):
    response_check(request, reference)
    response_check(request, response)
    assert reference["choices"][0]["text"] == response["choices"][0]["text"], "output text differs"


def events(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    assert isinstance(data.get("traceEvents"), list), "invalid exported trace"
    return data["traceEvents"]


def inside(event, parent):
    return (event.get("pid"), event.get("tid")) == (parent.get("pid"), parent.get("tid")) and (
        parent["ts"] <= event.get("ts", -1)
        and event.get("ts", -1) + event.get("dur", 0) <= parent["ts"] + parent["dur"]
    )


def zero_inventory(site, trace):
    args = site.get("args", {})
    dims = args.get("Input Dims", [])
    types = args.get("Input type", [])
    shape = dims[0] if dims and isinstance(dims[0], list) else None
    dtype = types[0] if types and isinstance(types[0], str) else None
    sizes = {"c10::BFloat16": 2, "BFloat16": 2, "bfloat16": 2, "float": 4,
             "float32": 4, "c10::Half": 2, "Half": 2, "half": 2}
    logical_bytes = math.prod(shape) * sizes[dtype] if shape and dtype in sizes else None
    identifiers = set()
    for event in trace:
        if event is site or inside(event, site):
            for key in ("External id", "correlation"):
                value = event.get("args", {}).get(key)
                if isinstance(value, int) and value > 0:
                    identifiers.add((key, value))
    kernels = [e for e in trace if e.get("cat") in ("kernel", "gpu_memset") and e.get("dur", 0) > 0
               and re.search(r"fill|memset|zero", e.get("name", ""), re.I)
               and any((key, e.get("args", {}).get(key)) in identifiers for key in ("External id", "correlation"))]
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    union, end = 0, float("-inf")
    for start, finish in intervals:
        union += max(0, finish - max(start, end))
        end = max(end, finish)
    return {"ts": site["ts"], "name": site["name"], "cpu_duration_us": site.get("dur"),
            "shape": shape, "dtype": dtype, "logical_initialized_bytes": logical_bytes, "args": args,
            "correlated_fill_events": [{k: e.get(k) for k in ("name", "ts", "dur", "args")} for e in kernels],
            "gpu_duration_sum_us": sum(e["dur"] for e in kernels) if kernels else None,
            "gpu_interval_union_us": union if kernels else None,
            "unresolved_attribution": logical_bytes is None or not kernels}


def inspect_traces(paths, scheduled, *, replay=False, zeros=False, native_summary=None):
    """Keep raw traces authoritative; the zero report is an attribution inventory."""
    ranks, records = set(), []
    for path in sorted(map(Path, paths)):
        match = re.search(r"(?:^|/)rank_(\d+)(?:/|$)", path.as_posix())
        assert match, f"unrecognized native rank directory: {path}"
        rank = int(match[1])
        assert rank not in ranks, f"duplicate rank {rank}"
        ranks.add(rank)
        trace = events(path)
        gpu_by_correlation = {}
        if replay:
            for index, event in enumerate(trace):
                correlation = event.get("args", {}).get("correlation")
                if (event.get("ph") == "X" and event.get("cat") == "kernel"
                        and event.get("dur", 0) > 0 and math.isfinite(event["dur"])
                        and type(correlation) is int and correlation > 0):
                    gpu_by_correlation.setdefault(correlation, []).append(index)
        prefills = sorted((e for e in trace if e.get("ph") == "X"
                           and e.get("cat") == "user_annotation"
                           and e.get("name", "").startswith("prefill[")), key=lambda e: e["ts"])
        assert len(prefills) == len(scheduled), (rank, len(prefills), scheduled)
        rank_steps = []
        for step, n in zip(prefills, scheduled):
            tokens = re.search(r"\btok=(\d+)(?:/(\d+))?(?:\s|\])", step["name"])
            assert tokens and int(tokens[1]) == n, step["name"]
            running = int(tokens[2] or tokens[1])
            assert re.search(r"\bbs=1(?:\s|\])", step["name"]), step["name"]
            assert math.isfinite(step["dur"]) and step["dur"] > 0
            launches = [(i, e) for i, e in enumerate(trace) if inside(e, step)
                        and e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
                        and e.get("name") in ("hipGraphLaunch", "cudaGraphLaunch")]
            graph_work = []
            if replay:
                assert running == next(b for b in (256, 512, 1024, 2048, 3072) if b >= n)
                assert launches, f"rank {rank}: no graph launch inside real prefill"
                for index, launch in launches:
                    correlation = launch.get("args", {}).get("correlation")
                    assert type(correlation) is int and correlation > 0, "unproven prefill replay correlation"
                    attached = gpu_by_correlation.get(correlation, [])
                    assert attached, f"rank {rank}: no GPU work attached to prefill graph correlation {correlation}"
                    graph_work.append({"api_event_index": index, "correlation": correlation,
                                       "gpu_event_indices": attached})
            item = {"label": step["name"], "host_duration_us": step["dur"],
                    "scheduled": n, "running": running, "graph_launches": len(launches)}
            if replay:
                item["prefill_graph_work"] = graph_work
            if zeros:
                assert running == n and n in (8192, 8193)
                compiled = [e for e in trace if inside(e, step) and re.search(
                    r"Torch-Compiled|CompiledFunction", e.get("name", ""))]
                ops = [e for e in trace if inside(e, step) and
                       "kimi_k3_post_attention_forward" in e.get("name", "")]
                assert compiled and ops, "missing actual compiled post-attention witness"
                sites = [e for e in trace if e.get("name") in ("aten::zeros_like", "aten::zeros")
                         and any(inside(e, op) for op in ops)]
                assert sites, "no attributable real zero initializer; do not call this zero cost"
                item["zero_initializers"] = [zero_inventory(e, trace) for e in sites]
                item["compiled_regions"] = len(compiled)
                item["post_attention_calls"] = len(ops)
                item["cost_status"] = "raw shape/stack/correlation attribution requires independent review"
            rank_steps.append(item)
        kernels = [e for e in trace if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
        assert kernels, f"rank {rank}: no executed GPU kernels"
        item = {"rank": rank, "path": str(path), "sha256": sha(path), "steps": rank_steps,
                "gpu_kernel_count": len(kernels)}
        if native_summary is not None:
            item["native_summary_markdown"] = native_summary.generate_report(
                native_summary.extract_labeled_events(trace), str(path))
        records.append(item)
    assert ranks == set(range(8)), ranks
    return records


def host_summary(windows):
    per_rank = {}
    for rank in range(8):
        values = [step["host_duration_us"] for window in windows for item in window
                  if item["rank"] == rank for step in item["steps"]]
        assert len(values) == 30
        per_rank[str(rank)] = {"samples_us": values, "median_us": statistics.median(values),
                               "p90_us": sorted(values)[math.ceil(.9 * len(values)) - 1]}
    return {"metric": "CPU prefill forward user_annotation duration; not TTFT or GPU time",
            "per_rank": per_rank}


def accuracy_check(output, population, server_log, start_offset, end_offset):
    """Accept only the current fresh output and every expected GSM8K document."""
    output = Path(output).resolve()
    assert read(output / "lm-eval-exit.json")["exit_code"] == 0
    receipt = read(output / "client-wait.json")
    assert receipt["wrapper_returncode"] == receipt["drain_returncode"] == 0
    root = output / "lm-eval"
    candidates = []
    for path in root.rglob("*.json"):
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        value = read(path)
        if isinstance(value, dict) and "gsm8k" in value.get("results", {}):
            candidates.append((path, value))
    assert len(candidates) == 1, "missing or ambiguous current result"
    result_path, result = candidates[0]
    assert result["n-samples"]["gsm8k"]["effective"] == 1319
    assert result["n-shot"]["gsm8k"] == 3 and result["config"].get("limit") is None
    cfg = result["config"]
    assert cfg["model"] == "local-completions"
    model_args = cfg["model_args"]
    assert model_args["model"] == "/input/model-target"
    assert model_args["base_url"] == "http://localhost:8000/v1/completions"
    assert model_args["num_concurrent"] == 65 and model_args["max_retries"] == 3
    assert model_args["tokenized_requests"] is False
    task = result["configs"]["gsm8k"]
    assert task["dataset_path"] == "parquet" and task.get("dataset_name") is None
    files = task["dataset_kwargs"]["data_files"]
    assert set(files) == {"train", "test"}
    for split in files:
        paths = [files[split]] if isinstance(files[split], str) else files[split]
        assert paths == [f"/input/data/{split}.parquet"]
    assert [cfg[k] for k in ("random_seed", "numpy_seed", "torch_seed", "fewshot_seed")] == [0, 1234, 1234, 1234]
    score = result["results"]["gsm8k"]["exact_match,flexible-extract"]
    assert math.isfinite(score) and .96 <= score <= 1
    samples = list(root.rglob("samples_gsm8k*.jsonl"))
    assert len(samples) == 1 and not samples[0].is_symlink()
    rows = [json.loads(line) for line in samples[0].read_text().splitlines()]
    expected = {r["doc_id"]: r for r in population}
    assert len(expected) == 1319
    seen, by_doc = set(), {}
    for row in rows:
        key = row["doc_id"], row["filter"]
        assert key not in seen and row["doc_id"] in expected
        seen.add(key)
        doc_id = row["doc_id"]
        assert row["doc"] == expected[doc_id]["doc"], "dataset row differs"
        assert hashlib.sha256(json.dumps(row["doc"], indent=2, ensure_ascii=False).encode()).hexdigest() == row["doc_hash"]
        assert row["target"] == row["doc"]["answer"]
        assert hashlib.sha256(str(row["target"]).encode()).hexdigest() == row["target_hash"]
        prompt = row["arguments"]["gen_args_0"]["arg_0"]
        assert isinstance(prompt, str) and row["doc"]["question"] in prompt
        assert hashlib.sha256(prompt.encode()).hexdigest() == row["prompt_hash"]
        assert row["resps"] and all(group and all(isinstance(t, str) and t.strip() for t in group)
                                    for group in row["resps"]), "missing response"
        identity = {k: row[k] for k in ("doc_hash", "prompt_hash", "target_hash")}
        assert doc_id not in by_doc or by_doc[doc_id] == identity
        by_doc[doc_id] = identity
    assert set(by_doc) == set(expected), "incomplete prompt population"
    # The pinned evaluator emits per-filter rows; strict-only coverage is incomplete.
    for doc_id in expected:
        matches = [r for r in rows if r["doc_id"] == doc_id]
        assert sum(r["filter"] == "flexible-extract" for r in matches) == 1
    with Path(server_log).open("rb") as stream:
        stream.seek(start_offset)
        log = stream.read(end_offset - start_offset).decode(errors="replace")
    rates = [float(m[1]) for line in log.splitlines() if "[MTP Stats " in line and "Interval" not in line
             if (m := re.search(r"Acceptance rate: ([\d.]+)%", line))]
    assert rates and math.isfinite(rates[-1]) and 25 <= rates[-1] <= 100
    return {"result": str(result_path), "result_sha256": sha(result_path), "score": score,
            "sample_file": str(samples[0]), "sample_sha256": sha(samples[0]),
            "unique_documents": len(by_doc), "prompt_coverage": by_doc,
            "acceptance_percent": rates[-1], "acceptance_fraction": rates[-1] / 100,
            "acceptance_scope": "current server cumulative stats during accuracy, including prior probes",
            "server_log_start": start_offset, "server_log_end": end_offset}


if __name__ == "__main__":
    import importlib.util
    import sys
    mode, spec_path, output_path = sys.argv[1:]
    spec = read(spec_path)
    if mode == "traces":
        native_spec = importlib.util.spec_from_file_location("native_trace_summary", spec["native_summary"])
        native = importlib.util.module_from_spec(native_spec)
        sys.modules[native_spec.name] = native
        native_spec.loader.exec_module(native)
        try:
            result = inspect_traces(spec["paths"], spec["scheduled"], replay=spec["replay"],
                                    zeros=spec["zeros"], native_summary=native)
        except (EOFError, OSError, json.JSONDecodeError):
            raise SystemExit(75)  # export not complete; semantic witness failures are not retried
    elif mode == "accuracy":
        result = accuracy_check(spec["output"], read(spec["population"]), spec["server_log"],
                                spec["start_offset"], spec["end_offset"])
    else:
        raise SystemExit("unknown fixed evidence mode")
    write(output_path, result)
