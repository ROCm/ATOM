"""Summarize an ATOM_COMPASS_DIR: per-rank forward stats and top GPU ops.

python -m atom.metrics.compass_summary <ATOM_COMPASS_DIR> [--top 15]
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import median


def _read_jsonl(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def _pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def summarize(root, top):
    for rank_dir in sorted(glob.glob(os.path.join(root, "dp*_pp*_tp*"))):
        rank = os.path.basename(rank_dir)
        forwards = _read_jsonl(os.path.join(rank_dir, "forward.*.jsonl"))
        ops = _read_jsonl(os.path.join(rank_dir, "ops.*.jsonl"))
        print(f"== {rank}: {len(forwards)} forward rows, {len(ops)} op rows")

        kinds = defaultdict(list)
        for r in forwards:
            if r.get("phase", "target") != "target":
                kind = r["phase"]
            elif r["is_dummy"]:
                kind = "dummy"
            elif r["is_prefill"]:
                kind = "prefill"
            else:
                kind = "decode_graph" if r["use_cudagraph"] else "decode_eager"
            kinds[kind].append(r)
        print(
            f"  {'kind':<13}{'n':>6}{'gpu_ms p50':>12}{'p90':>9}{'max':>9}"
            f"{'tokens p50':>12}{'bs p50':>8}"
        )
        for kind, rows in sorted(kinds.items()):
            gpu = [r["gpu_ms"] for r in rows]
            tokens = [r.get("num_tokens", r["scheduled_tokens"]) for r in rows]
            bs = [r["scheduled_bs"] for r in rows]
            print(
                f"  {kind:<13}{len(rows):>6}{median(gpu):>12.3f}{_pct(gpu, 0.9):>9.3f}"
                f"{max(gpu):>9.3f}{median(tokens):>12.0f}{median(bs):>8.0f}"
            )
        bad = [
            r
            for r in forwards
            if not r["gpu_start_ns"] <= r["gpu_end_ns"]
            or r["gpu_start_ns"] < r["cpu_start_ns"] - 1_000_000
        ]
        print(f"  timestamp sanity violations: {len(bad)}")
        if forwards:
            print("  sample forward row:", json.dumps(_trim(forwards[-1])))

        if not ops:
            continue
        by_op = defaultdict(lambda: [0, 0, set()])
        for r in ops:
            entry = by_op[r["op"]]
            entry[0] += 1
            entry[1] += r["gpu_kernel_ns"]
            entry[2].add(json.dumps(r["shapes"]))
        total = sum(e[1] for e in by_op.values())
        print(f"  top {top} ops by GPU kernel time ({total / 1e6:.1f} ms total):")
        for name, (n, ns, shapes) in sorted(by_op.items(), key=lambda kv: -kv[1][1])[
            :top
        ]:
            print(
                f"    {ns / total:6.1%} {ns / 1e6:9.2f} ms  n={n:<6} "
                f"shapes={len(shapes):<4} {name[:70]}"
            )
        print("  sample op row:", json.dumps(_trim(ops[len(ops) // 2])))


def _trim(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, list) and len(v) > 4:
            out[k] = v[:4] + [f"...+{len(v) - 4}"]
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    summarize(args.root, args.top)


if __name__ == "__main__":
    main()
