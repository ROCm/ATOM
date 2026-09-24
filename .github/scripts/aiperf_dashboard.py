# SPDX-License-Identifier: MIT
"""Shared legacy AIPerf summary for single-node and ATOMesh CI."""

import argparse
import json
import os
import shlex
from pathlib import Path


def dashboard_summary(data, src, conc, env, *, single_node=False):
    def avg(name):
        value = data.get(name)
        if isinstance(value, dict):
            return value.get("avg")
        return value

    def pct(name, key):
        value = data.get(name)
        if isinstance(value, dict):
            return value.get(key)
        return None

    def total_tokens(name):
        """Return one of AIPerf's profiling-only aggregate token counters."""
        value = avg(name)
        return int(value) if isinstance(value, (int, float)) else None

    # These aggregates contain successful profiling records only: AIPerf excludes
    # its internal warmup and requests cancelled during grace-period draining.
    cache_hit_tokens = total_tokens("total_usage_prompt_cache_read_tokens")
    cache_total_tokens = total_tokens("total_usage_prompt_tokens")

    payload = {
        "benchmark_backend": "atom",
        # Directory holding this run's profile_export.jsonl, so process_result.py can
        # find the per-request records both interactivity definitions are computed
        # from, without reconstructing the directory name.
        "aiperf_artifact_dir": src.parent.name,
        "benchmark_model_name": env.get("MODEL_NAME")
        or data.get("model")
        or data.get("model_id"),
        "backend": "atom",
        "benchmark_kind": env.get("BENCHMARK_KIND") or "aiperf_agentic",
        "scenario": env.get("AIPERF_SCENARIO"),
        "public_dataset": env.get("AIPERF_PUBLIC_DATASET"),
        "topology": env.get("TOPOLOGY") or data.get("topology"),
        "display_topology": env.get("DISPLAY_TOPOLOGY") or data.get("display_topology"),
        "precision": env.get("PRECISION") or data.get("precision"),
        "random_input_len": int(
            data.get("max_context_length") or env.get("AIPERF_MAX_CONTEXT_LENGTH") or 0
        ),
        "random_output_len": 1024,
        "max_concurrency": conc,
        "random_range_ratio": "",
        "request_throughput": avg("request_throughput"),
        "mean_ttft_ms": avg("time_to_first_token"),
        "median_ttft_ms": pct("time_to_first_token", "p50"),
        "p90_ttft_ms": pct("time_to_first_token", "p90"),
        "p99_ttft_ms": pct("time_to_first_token", "p99"),
        "mean_itl_ms": avg("inter_token_latency"),
        "median_itl_ms": pct("inter_token_latency", "p50"),
        "p90_itl_ms": pct("inter_token_latency", "p90"),
        "p99_itl_ms": pct("inter_token_latency", "p99"),
        "mean_e2el_ms": avg("request_latency"),
        "median_e2el_ms": pct("request_latency", "p50"),
        "p90_e2el_ms": pct("request_latency", "p90"),
        "p99_e2el_ms": pct("request_latency", "p99"),
        "input_throughput": avg("input_token_throughput"),
        "output_throughput": avg("output_token_throughput"),
        "total_token_throughput": avg("total_token_throughput"),
        "successful_requests": avg("request_count"),
        "completed": avg("request_count"),
        "benchmark_duration_s": avg("benchmark_duration")
        or data.get("benchmark_duration_s"),
        "total_input_tokens": avg("total_usage_prompt_tokens"),
        "total_output_tokens": avg("total_usage_completion_tokens"),
        "cache_hit_tokens": cache_hit_tokens,
        "cache_total_tokens": cache_total_tokens,
        "cache_hit_rate": (
            round(cache_hit_tokens / cache_total_tokens, 4)
            if cache_hit_tokens is not None and cache_total_tokens
            else None
        ),
    }

    if single_node:
        from atom.benchmarks.results.metadata import synthetic_settings

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
        parallel, _ = parser.parse_known_args(shlex.split(env.get("SERVER_ARGS", "")))
        validity = data.get("metadata", data)
        synthetic = synthetic_settings(shlex.split(env.get("SERVER_ARGS", "")), env)
        payload.update(
            tensor_parallel_size=parallel.tensor_parallel_size,
            output_sequence_length=avg("output_sequence_length"),
            input_sequence_length=avg("input_sequence_length"),
            dashboard_publish_allowed=(
                validity.get("submission_valid") is not False
                and not synthetic["synthetic"]
                and env.get("ENABLE_TORCH_PROFILER") != "1"
                and env.get("ENABLE_RTL_PROFILER") != "1"
            ),
            synthetic=synthetic["synthetic"],
        )
    return {key: value for key, value in payload.items() if value is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("concurrency", type=int)
    parser.add_argument("--single-node", action="store_true")
    args = parser.parse_args()
    payload = dashboard_summary(
        json.loads(args.source.read_text()),
        args.source,
        args.concurrency,
        os.environ,
        single_node=args.single_node,
    )
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    hit, total = payload.get("cache_hit_tokens"), payload.get("cache_total_tokens")
    if hit is not None and total:
        print(f"[aiperf] prefix cache hit: {hit}/{total} tokens ({hit / total:.2%})")
    else:
        print(
            "[aiperf] prefix cache hit: unavailable (AIPerf profiling cache-read counters were not produced)"
        )
    print(f"[aiperf] dashboard json: {args.output}")


if __name__ == "__main__":
    main()
