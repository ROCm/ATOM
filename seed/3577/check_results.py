"""Check receipts after the common supervisor proves cleanup. No model execution."""
import json
from pathlib import Path
import sys

evidence = Path(sys.argv[1]).resolve()
read = lambda name: json.loads((evidence / name).read_text())
assert read("entry-exit.json")["entry_exit"] == 0
termination = read("supervisor-exit.json")
assert termination["status"] == "completed"
assert termination["kernel_children_empty"] and not termination["owned_remaining"]
assert termination["lease_held"] is False and termination["lease_released_utc"]
assert read("verified-inputs.json")
if evidence.name.endswith("pp2"):
    results = read("benchmark-detailed.json")
    prompts = [json.loads(line) for line in (evidence / "benchmark-prompts.jsonl").read_text().splitlines()]
    assert len(prompts) == results["completed"] == 8
    assert len(results["generated_texts"]) == len(results["errors"]) == 8
    assert all(not error for error in results["errors"])
    assert results["p3577_cached_tokens"] == [0] * 8
    assert results["input_lens"] == [p["input_tokens"] for p in prompts]
    assert results["output_lens"] == [64] * 8
    assert all(text for text in results["generated_texts"])
    assert read("exact-exit.json") == dict(start=0, request=0, stop=0)
    request, response = read("exact-request.json"), read("exact-response.json")
    assert response["prompt_token_ids"] == request["prompt"]
    assert response["usage"]["prompt_tokens"] == 49152
    assert response["usage"]["completion_tokens"] == 64
    files = sorted((evidence / "pp-observer").glob("pp*-rank*.jsonl"))
    assert len(files) == 8
    ranks = set()
    for path in files:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records[0]["event"] == "begin" and records[-1]["event"] == "end"
        ranks.add(records[0]["rank"])
        steps = [r for r in records if r["event"] == "batch" and r["prefill_tokens"] > 0 and not r["dummy"]]
        assert steps and all(r["prefill_seqs"] == 1 and len(r["req_ids"]) == 1 for r in steps)
        cursor = 0
        for step in steps:
            assert step["cached"] == [cursor], step
            assert 0 < step["scheduled"][0] <= 16384
            cursor += step["scheduled"][0]
        assert cursor == 49152
        events = {r["event"] for r in records}
        assert {"layer-bank", "close-before", "close-after", "model-partition", "wire-buffer"} <= events
        if records[0]["pp"] == 0:
            assert {"pp-send-logical", "clone-buffer", "send-wait-complete"} <= events
        else:
            assert {"pp-receive-logical", "allgather-before", "allgather-after"} <= events
        if evidence.name == "B-pp2":
            assert {"bank-allocate-before", "bank-allocate-after"} <= events
            if records[0]["pp"] == 0:
                assert {"prefix-before", "prefix-after"} <= events
        partitions = {r["step"]: r for r in records if r["event"] == "model-partition"}
        logical = {}
        for item in records:
            if item["event"] not in ("pp-send-logical", "pp-receive-logical"):
                continue
            bank = item["tensors"]["block_residual"]
            partition = partitions[item["step"]]
            boundary = partition["end_layer"] if item["event"] == "pp-send-logical" else partition["start_layer"]
            assert bank["shape"][1:] == [
                -(-boundary // partition["block_size"]), partition["hidden"]]
            logical[item["step"]] = bank
        for item in records:
            if item["event"] != "wire-buffer" or item["key"] != "block_residual":
                continue
            info = item["tensor"]
            assert info["logical_bytes"] > 0 and info["element_size"] > 0
            assert len(info["shape"]) == len(info["stride"])
            assert info["storage_offset"] >= 0
            assert info["storage_bytes"] >= info["logical_bytes"]
            assert info["logical_bytes"] * 4 == logical[item["step"]]["logical_bytes"]
        if evidence.name == "B-pp2":
            for item in records:
                if item["event"] == "bank-allocate-after":
                    partition = partitions[item["step"]]
                    assert item["bank"]["shape"][1] == -(-partition["layers"] // partition["block_size"])
    assert ranks == set(range(8)), ranks
    traces = list((evidence / "traces").rglob("*.pt.trace.json.gz"))
    assert len(traces) == 8 and all(p.stat().st_size > 0 for p in traces)
else:
    assert read("lm-eval-exit.json")["exit_code"] == 0
    results = list((evidence / "accuracy_test_results").glob("*.json"))
    assert len(results) == 1
    data = json.loads(results[0].read_text())
    assert data["n-samples"]["gsm8k"]["effective"] == 1319
    assert data["n-shot"]["gsm8k"] == 3 and data["config"].get("limit") is None
    assert data["results"]["gsm8k"]["exact_match,flexible-extract"] >= 0.96
    assert data["atom_ci_metadata"]["mtp_acceptance_rate"] / 100 >= 0.25
    samples = list((evidence / "accuracy_test_results").rglob("samples_gsm8k*.jsonl"))
    assert len(samples) == 1
    rows = [json.loads(line) for line in samples[0].read_text().splitlines()]
    assert len(rows) == len({row["doc_id"] for row in rows}) == 1319
    assert {row["doc_id"] for row in rows} == set(range(1319))
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from strings(child)
    for row in rows:
        assert any(row["doc"]["question"] in prompt for prompt in strings(row["arguments"]))
        assert row["resps"] and all(group for group in row["resps"])
        assert all(isinstance(text, str) and text.strip() for group in row["resps"] for text in group)
    trace_requested = read("compiled-trace-request.json")["requested"]
    assert trace_requested in (0, 1)
    if trace_requested:
        assert read("compiled-trace-outcome.json")["status"] == "completed"
        assert read("exact-exit.json") == dict(start=0, request=0, stop=0)
        request, response = read("exact-request.json"), read("exact-response.json")
        assert response["prompt_token_ids"] == request["prompt"]
        assert response["usage"]["prompt_tokens"] == 49152
        assert response["usage"]["completion_tokens"] == 64
        traces = list((evidence / "traces").rglob("*.pt.trace.json.gz"))
        assert len(traces) == 8 and all(path.stat().st_size > 0 for path in traces)
    else:
        assert read("compiled-trace-outcome.json")["status"] == "not_requested"
print("Required raw receipts are present and basic completeness checks passed.")
print("Independent trace attribution/layout/byte reconciliation and A/B output comparison remain required.")
