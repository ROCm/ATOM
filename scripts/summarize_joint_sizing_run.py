"""Summarize one aiperf trace-replay round into the joint-sizing results row.

Reads the round's ``driver.log`` (aiperf's periodic stats), ``profile_export.jsonl``
(per-request records) and the copied ``decode.log``, and prints the numbers the
run table in ``.claude/plans/overnight_sparsekv_joint_sizing_runs.md`` wants:
latency percentiles, output-token throughput over the profiling window, the
scheduler's high-water marks, and whether the round survived.

  python scripts/summarize_joint_sizing_run.py results/js_c32_m48_r14
"""

import json
import re
import sys
from pathlib import Path

STAT_LINE = re.compile(
    r"\s(ttft|itl|e2e|intvty|isl|osl)\s+"
    r"p50=\s*([\d,]+)(?:ms)?\s+p(?:75)=\s*([\d,]+)(?:ms)?\s+"
    r"p(?:90|95)=\s*([\d,]+)(?:ms)?\s+p99=\s*([\d,]+)(?:ms)?"
)
TICK = re.compile(
    r"Cache Pool Tick\] used (\d+) \((\d+)%\), free \d+, retained-cache \d+, "
    r"evicted total \d+, running (\d+), peak-decode-batch (\d+)"
)
DEFER = re.compile(
    r"SparseKV admit DEFER .*host_used=(\d+)/(\d+) promoted_to_gpu=(\d+)/(\d+)"
)
CRASH = re.compile(
    r"Memory access fault|a prefill forward reached MLAAttention|"
    r"proc died unexpectedly|SparseKV host pool exhausted|no free request slots"
)


def num(s: str) -> int:
    return int(s.replace(",", ""))


def latency_stats(driver_log: Path) -> dict:
    """Last periodic stats block aiperf printed (the run's final state)."""
    stats: dict[str, tuple[int, int, int, int]] = {}
    if not driver_log.exists():
        return stats
    for line in driver_log.read_text(errors="replace").splitlines():
        m = STAT_LINE.search(line)
        if m:
            stats[m.group(1)] = tuple(num(m.group(i)) for i in (2, 3, 4, 5))
    return stats


def throughput(jsonl: Path) -> dict:
    """Completed requests and output-token rate over the profiling window."""
    if not jsonl.exists():
        return {}
    n = 0
    out_tokens = 0
    first_ns = None
    last_ns = 0
    for line in jsonl.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        meta = rec.get("metadata", {})
        if meta.get("benchmark_phase") != "profiling" or meta.get("was_cancelled"):
            continue
        start = meta.get("request_start_ns")
        end = meta.get("request_end_ns")
        if not start or not end:
            continue
        n += 1
        out_tokens += int(
            rec.get("metrics", {}).get("output_token_count", {}).get("value", 0)
        )
        first_ns = start if first_ns is None else min(first_ns, start)
        last_ns = max(last_ns, end)
    if not n:
        return {}
    window_s = (last_ns - first_ns) / 1e9
    return {
        "requests": n,
        "out_tokens": out_tokens,
        "window_s": window_s,
        "out_tok_per_s": out_tokens / window_s if window_s else 0.0,
    }


def failures(jsonl: Path, phase: str = "profiling") -> list:
    """Every request in ``phase`` that did not return a complete response.

    Any request failure counts as a defect, so this is reported per round rather
    than left in the CSV's aggregate error rate.

    ``acked`` is what makes the duration readable. aiperf stamps
    ``request_end_ns == request_start_ns`` when no response header ever arrived,
    so an unacked failure's duration is 0 s no matter how long the request
    actually sat there — reading that as "rejected instantly" is wrong. Only an
    acked failure's duration means anything (a stream that started and was then
    cut, e.g. pinned at the proxy's request timeout).
    """
    if not jsonl.exists():
        return []
    out = []
    for line in jsonl.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        meta = rec.get("metadata", {})
        if meta.get("benchmark_phase") != phase:
            continue
        err = rec.get("error")
        if not (err or meta.get("was_cancelled") or meta.get("context_overflow_skip")):
            continue
        start, end = meta.get("request_start_ns"), meta.get("request_end_ns")
        acked = meta.get("request_ack_ns") is not None
        out.append(
            {
                "kind": (err or {}).get("type")
                or ("cancelled" if meta.get("was_cancelled") else "context_overflow"),
                "duration_s": (end - start) / 1e9 if start and end else None,
                "acked": acked,
                "start_ns": start,
            }
        )
    return out


def official(csv: Path) -> dict:
    """aiperf's own end-of-run summary. Only written when the round completed.

    Preferred over the per-record recomputation: it is what the tool reports,
    and it carries the prefix-cache hit that the records do not aggregate.
    """
    if not csv.exists():
        return {}
    wanted = {
        "Benchmark Duration (sec)": "duration_s",
        "Output Token Throughput (tokens/sec)": "out_tok_per_s",
        "Total Token Throughput (tokens/sec)": "total_tok_per_s",
        "Request Count": "requests",
        "Request Throughput (requests/sec)": "req_per_s",
        "Total Usage Prompt Tokens (tokens)": "prompt_tokens",
        "Total Usage Prompt Cache Read Tokens (tokens)": "cache_read_tokens",
    }
    out: dict[str, float] = {}
    for line in csv.read_text(errors="replace").splitlines():
        name, _, rest = line.partition(",")
        if name in wanted and rest:
            out[wanted[name]] = float(rest.split(",")[0])
    if out.get("prompt_tokens"):
        out["cache_hit_pct"] = 100.0 * out["cache_read_tokens"] / out["prompt_tokens"]
    return out


def server_marks(decode_log: Path) -> dict:
    """Index-pool / host-pool high-water marks and the crash verdict."""
    if not decode_log.exists():
        return {}
    peak_used = peak_pct = peak_batch = 0
    host_used = host_total = promoted = gpu_total = 0
    defers = 0
    crash = ""
    for line in decode_log.read_text(errors="replace").splitlines():
        m = TICK.search(line)
        if m:
            peak_used = max(peak_used, int(m.group(1)))
            peak_pct = max(peak_pct, int(m.group(2)))
            peak_batch = max(peak_batch, int(m.group(4)))
            continue
        m = DEFER.search(line)
        if m:
            defers += 1
            host_used = max(host_used, int(m.group(1)))
            host_total = int(m.group(2))
            promoted = max(promoted, int(m.group(3)))
            gpu_total = int(m.group(4))
            continue
        if not crash and CRASH.search(line):
            crash = line.strip()[:160]
    # The host / GPU-cold high-water marks are only ever printed on a DEFER
    # line, so a round that never deferred has no sample at all — which is not
    # the same as "the pools stayed empty".
    pools = (
        {
            "host_peak": f"{host_used}/{host_total}",
            "promoted_peak": f"{promoted}/{gpu_total}",
        }
        if defers
        else {
            "host_peak": "unsampled (no DEFER)",
            "promoted_peak": "unsampled (no DEFER)",
        }
    )
    return {
        "index_peak": peak_used,
        "index_peak_pct": peak_pct,
        "peak_decode_batch": peak_batch,
        **pools,
        "defers": defers,
        "crash": crash or "none",
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1])
    print(f"=== {d.name} ===")
    for k, v in latency_stats(d / "driver.log").items():
        unit = "ms" if k in ("ttft", "itl", "e2e") else ""
        print(
            f"  {k:<7} p50={v[0]:,}{unit} p75={v[1]:,}{unit} p95={v[2]:,}{unit} p99={v[3]:,}{unit}"
        )
    tp = throughput(d / "aiperf_artifacts" / "profile_export.jsonl")
    if tp:
        print(
            f"  profiling: {tp['requests']} requests, {tp['out_tokens']:,} output "
            f"tokens over {tp['window_s']:.0f}s = {tp['out_tok_per_s']:.1f} tok/s"
        )
    else:
        print("  profiling: no completed profiling records")
    off = official(d / "aiperf_artifacts" / "profile_export_aiperf.csv")
    if off:
        print(
            f"  aiperf:    {off['requests']:.0f} requests over {off['duration_s']:.0f}s, "
            f"{off['out_tok_per_s']:.2f} out tok/s, "
            f"{off['total_tok_per_s']:.0f} total tok/s, "
            f"{off['req_per_s']:.2f} req/s, "
            f"prefix-cache {off['cache_hit_pct']:.1f}%"
        )
    else:
        print("  aiperf:    no end-of-run CSV (round did not complete)")
    jsonl = d / "aiperf_artifacts" / "profile_export.jsonl"
    for phase in ("warmup", "profiling"):
        fails = failures(jsonl, phase)
        label = f"FAIL/{phase[:4]}:"
        if not fails:
            print(f"  {label:<10} none")
            continue
        by_kind: dict[tuple, list] = {}
        for f in fails:
            by_kind.setdefault((f["kind"], f["acked"]), []).append(f["duration_s"])
        print(f"  {label:<10} {len(fails)} requests")
        for (kind, acked), durs in by_kind.items():
            if acked:
                shown = "streamed then cut at " + ", ".join(
                    f"{x:.0f}s" for x in sorted(x for x in durs if x)
                )
            else:
                shown = "never acked (duration not meaningful)"
            print(f"               {len(durs)}x {kind} — {shown}")
    for k, v in server_marks(d / "decode.log").items():
        print(f"  {k:<18} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
