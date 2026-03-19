#!/usr/bin/env python3
"""Analyze profiler trace to generate a human-readable performance summary.

Extracts prefill/decode/draft step annotations and produces:
- Per-phase timing breakdown (prefill, decode, draft)
- Draft step comparison across MTP iterations
- Decode iteration statistics

Usage:
    python analyze_trace_summary.py <trace.json.gz> [--output report.md]
"""

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class StepEvent:
    name: str
    dur_us: float
    ts: float
    cat: str


@dataclass
class DecodeIteration:
    decode: Optional[StepEvent] = None
    draft_steps: list[StepEvent] = field(default_factory=list)


def load_trace_events(filepath: str) -> list[dict]:
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def extract_labeled_events(events: list[dict]) -> list[StepEvent]:
    """Extract prefill/decode/draft events from trace."""
    labeled = []
    for e in events:
        name = e.get("name", "")
        if e.get("ph") != "X":
            continue
        if name.startswith(("prefill[", "decode[", "draft[")):
            labeled.append(
                StepEvent(
                    name=name,
                    dur_us=e.get("dur", 0),
                    ts=e.get("ts", 0),
                    cat=e.get("cat", ""),
                )
            )
    labeled.sort(key=lambda x: x.ts)
    return labeled


def parse_label(name: str) -> dict[str, str]:
    """Parse label like 'decode[bs=4 tok=16 d=4 spec=3]' into dict."""
    m = re.match(r"(\w+)\[(.+)\]", name)
    if not m:
        return {"type": name}
    result = {"type": m.group(1)}
    for pair in m.group(2).split():
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def group_decode_iterations(events: list[StepEvent]) -> list[DecodeIteration]:
    """Group sequential decode + draft events into iterations."""
    iterations: list[DecodeIteration] = []
    current: Optional[DecodeIteration] = None

    for ev in events:
        if ev.name.startswith("decode["):
            current = DecodeIteration(decode=ev)
            iterations.append(current)
        elif ev.name.startswith("draft[") and current is not None:
            current.draft_steps.append(ev)

    return iterations


def format_duration(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.1f} us"


def generate_report(events: list[StepEvent], filepath: str) -> str:
    """Generate markdown performance summary report."""
    lines: list[str] = []
    lines.append("# Trace Performance Summary")
    lines.append(f"\n**File:** `{Path(filepath).name}`\n")

    # Separate by phase
    prefills = [
        e
        for e in events
        if e.name.startswith("prefill[") and e.cat == "user_annotation"
    ]
    decodes = [
        e for e in events if e.name.startswith("decode[") and e.cat == "user_annotation"
    ]
    # draft[i/k bs=N] — each MTP speculative step
    draft_steps = [
        e for e in events if e.name.startswith("draft[") and e.cat == "user_annotation"
    ]

    # --- Prefill Summary ---
    if prefills:
        lines.append("## Prefill")
        lines.append("")
        lines.append("| # | Label | Duration |")
        lines.append("|---|-------|----------|")
        for i, ev in enumerate(prefills):
            lines.append(f"| {i} | `{ev.name}` | {format_duration(ev.dur_us)} |")
        total_prefill = sum(e.dur_us for e in prefills)
        lines.append(f"\n**Total prefill:** {format_duration(total_prefill)}\n")

    # --- Decode Summary ---
    if decodes:
        lines.append("## Decode")
        lines.append("")
        durs = [e.dur_us for e in decodes]
        lines.append(f"- **Iterations:** {len(durs)}")
        lines.append(f"- **Mean:** {format_duration(sum(durs) / len(durs))}")
        lines.append(f"- **Min:** {format_duration(min(durs))}")
        lines.append(f"- **Max:** {format_duration(max(durs))}")
        lines.append(f"- **Total:** {format_duration(sum(durs))}")
        lines.append("")

    # --- Draft Step Comparison ---
    if draft_steps:
        lines.append("## Draft (MTP) Step Breakdown")
        lines.append("")
        durs = [e.dur_us for e in draft_steps]
        lines.append(f"- **Total steps:** {len(durs)}")
        lines.append(f"- **Mean per step:** {format_duration(sum(durs) / len(durs))}")
        lines.append(f"- **Total:** {format_duration(sum(durs))}")
        lines.append("")

        # Group by step index
        step_durations: dict[str, list[float]] = defaultdict(list)
        for ev in draft_steps:
            step_key = ev.name.split(" bs=")[0] + "]"  # e.g. "draft[0/3]"
            step_durations[step_key].append(ev.dur_us)

        lines.append("| Step | Calls | Mean | Min | Max | Δ vs step 0 |")
        lines.append("|------|-------|------|-----|-----|-------------|")
        step_means: list[float] = []
        for step_key in sorted(step_durations.keys()):
            durs = step_durations[step_key]
            mean = sum(durs) / len(durs)
            step_means.append(mean)
            delta = ""
            if len(step_means) > 1:
                pct = (mean - step_means[0]) / step_means[0] * 100
                delta = f"{pct:+.1f}%"
            lines.append(
                f"| `{step_key}` | {len(durs)} | {format_duration(mean)} "
                f"| {format_duration(min(durs))} | {format_duration(max(durs))} | {delta} |"
            )
        lines.append("")

        if len(step_means) >= 2 and step_means[0] > step_means[1] and step_means[0] > 0:
            overhead = step_means[0] - step_means[1]
            lines.append(
                f"> **Note:** Step 0 is {format_duration(overhead)} slower than step 1 "
                f"({overhead / step_means[0] * 100:.0f}% overhead from `index_select` on first draft step)."
            )
            lines.append("")

    # --- Decode + Draft Combined Iteration View ---
    iterations = group_decode_iterations(
        [e for e in events if e.cat == "user_annotation"]
    )
    if iterations and iterations[0].draft_steps:
        lines.append("## Iteration Timing (decode + draft)")
        lines.append("")
        lines.append("| # | Decode | Draft (sum) | Total | Draft/Decode |")
        lines.append("|---|--------|-------------|-------|--------------|")
        for i, it in enumerate(iterations[:10]):
            d_dur = it.decode.dur_us if it.decode else 0
            dr_dur = sum(s.dur_us for s in it.draft_steps)
            total = d_dur + dr_dur
            ratio = f"{dr_dur / d_dur:.1f}x" if d_dur > 0 else "N/A"
            lines.append(
                f"| {i} | {format_duration(d_dur)} | {format_duration(dr_dur)} "
                f"| {format_duration(total)} | {ratio} |"
            )
        if len(iterations) > 10:
            lines.append(f"| ... | ({len(iterations) - 10} more) | | | |")
        lines.append("")

        # Averages
        d_durs = [it.decode.dur_us for it in iterations if it.decode]
        dr_durs = [
            sum(s.dur_us for s in it.draft_steps) for it in iterations if it.draft_steps
        ]
        if d_durs and dr_durs:
            avg_d = sum(d_durs) / len(d_durs)
            avg_dr = sum(dr_durs) / len(dr_durs)
            lines.append(
                f"**Average:** decode={format_duration(avg_d)}, "
                f"draft={format_duration(avg_dr)}, "
                f"ratio={avg_dr / avg_d:.1f}x"
            )
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze profiler trace and generate performance summary."
    )
    parser.add_argument("filepath", help="Path to trace JSON or JSON.GZ file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown file (default: print to stdout)",
    )
    args = parser.parse_args()

    events = load_trace_events(args.filepath)
    labeled = extract_labeled_events(events)

    if not labeled:
        print("No prefill/decode/draft annotations found in trace.", file=sys.stderr)
        sys.exit(1)

    report = generate_report(labeled, args.filepath)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
