#!/usr/bin/env python3
"""Check complete per-request span trees against a successful ON benchmark."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def check(rows, expected):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["trace_id"]].append(row)
    failures = []
    for trace_id in expected:
        spans = grouped[trace_id]
        by_name = {span["name"]: span for span in spans}
        names = {
            "atom.request",
            "engine.request",
            "prefill_waiting",
            "prefill_forward",
            "decode_forward",
        }
        if len(spans) != 5 or set(by_name) != names:
            failures.append(
                {
                    "trace_id": trace_id,
                    "error": "missing_or_duplicate_spans",
                    "names": [s["name"] for s in spans],
                }
            )
            continue
        http = by_name["atom.request"]
        engine = by_name["engine.request"]
        if http["parent_span_id"] != "00f067aa0ba902b7":
            failures.append({"trace_id": trace_id, "error": "unjoined_upstream_parent"})
        if isinstance(expected, dict):
            request = expected[trace_id]
            public_id = request.get("response_id")
            if public_id and (
                http["attributes"].get("gen_ai.request.id") != public_id
                or engine["attributes"].get("gen_ai.request.id") != public_id
            ):
                failures.append(
                    {"trace_id": trace_id, "error": "wrong_public_request_id"}
                )
            if engine["attributes"].get("gen_ai.usage.cached_tokens") != 0:
                failures.append(
                    {"trace_id": trace_id, "error": "unexpected_timed_cache_hit"}
                )
            usage = request["usage"]
            if (
                engine["attributes"].get("gen_ai.usage.prompt_tokens")
                != usage["prompt_tokens"]
                or engine["attributes"].get("gen_ai.usage.completion_tokens")
                != usage["completion_tokens"]
            ):
                failures.append({"trace_id": trace_id, "error": "wrong_engine_usage"})
        if engine["parent_span_id"] != http["span_id"]:
            failures.append({"trace_id": trace_id, "error": "unjoined_http_engine"})
        if (
            not http["start_ns"]
            <= engine["start_ns"]
            <= engine["end_ns"]
            <= http["end_ns"]
        ):
            failures.append(
                {"trace_id": trace_id, "error": "engine_outside_http_lifetime"}
            )
        for name in names - {"atom.request", "engine.request"}:
            span = by_name[name]
            if (
                span["parent_span_id"] != engine["span_id"]
                or not engine["start_ns"]
                <= span["start_ns"]
                <= span["end_ns"]
                <= engine["end_ns"]
            ):
                failures.append(
                    {"trace_id": trace_id, "error": "invalid_phase_tree", "phase": name}
                )
        if (
            by_name["prefill_waiting"]["end_ns"]
            != by_name["prefill_forward"]["start_ns"]
            or by_name["prefill_forward"]["end_ns"]
            != by_name["decode_forward"]["start_ns"]
        ):
            failures.append({"trace_id": trace_id, "error": "noncontiguous_phases"})
        if any(
            span["status_code"] == 2 or "nsdye=1" not in span["trace_state"]
            for span in spans
        ):
            failures.append({"trace_id": trace_id, "error": "error_or_lost_dye"})
    return {
        "expected_requests": len(expected),
        "valid": bool(expected) and not failures,
        "failures": failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans", required=True)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.spans).read_text().splitlines()]
    expected = {}
    for path in args.benchmark:
        benchmark = json.loads(Path(path).read_text())
        if not benchmark["valid"]:
            raise ValueError("Cannot accept traces against failed benchmark")
        for row in benchmark["requests"]:
            if row["trace_id"] in expected:
                raise ValueError("Duplicate trace ID across benchmark inputs")
            expected[row["trace_id"]] = row
    result = check(rows, expected)
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["valid"] else 1)
