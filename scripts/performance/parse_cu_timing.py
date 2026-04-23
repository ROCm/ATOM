#!/usr/bin/env python3
"""Parse CU-mask timing logs from disaggregated prefill/decode runs.

Each log file contains lines like:
  [atom HH:MM:SS] prefill iter 231.08ms | reqs=1 | tokens=992
  [atom HH:MM:SS] iter 12.36ms | reqs=1 (prefill=0 decode=1) | tokens=1 (prefill=0 decode=1)

Usage:
    python parse_cu_timing.py iter_times_50cu.log [iter_times_70cu.log ...]

Outputs a CSV to stdout with columns:
    file, type, iter_ms, total_reqs, prefill_reqs, decode_reqs,
    total_tokens, prefill_tokens, decode_tokens
"""

import argparse
import csv
import re
import sys

PREFILL_RE = re.compile(r"prefill iter ([\d.]+)ms \| reqs=(\d+) \| tokens=(\d+)")
DECODE_RE = re.compile(
    r"(?<!prefill )iter ([\d.]+)ms \| reqs=(\d+) "
    r"\(prefill=(\d+) decode=(\d+)\) \| "
    r"tokens=(\d+) \(prefill=(\d+) decode=(\d+)\)"
)


def parse_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = PREFILL_RE.search(line)
            if m:
                ms, reqs, tokens = float(m.group(1)), int(m.group(2)), int(m.group(3))
                rows.append(
                    {
                        "file": path,
                        "type": "prefill",
                        "iter_ms": ms,
                        "total_reqs": reqs,
                        "prefill_reqs": reqs,
                        "decode_reqs": 0,
                        "total_tokens": tokens,
                        "prefill_tokens": tokens,
                        "decode_tokens": 0,
                    }
                )
                continue
            m = DECODE_RE.search(line)
            if m:
                ms = float(m.group(1))
                total_r, pf_r, dc_r = int(m.group(2)), int(m.group(3)), int(m.group(4))
                total_t, pf_t, dc_t = int(m.group(5)), int(m.group(6)), int(m.group(7))
                rows.append(
                    {
                        "file": path,
                        "type": "decode",
                        "iter_ms": ms,
                        "total_reqs": total_r,
                        "prefill_reqs": pf_r,
                        "decode_reqs": dc_r,
                        "total_tokens": total_t,
                        "prefill_tokens": pf_t,
                        "decode_tokens": dc_t,
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Log file paths")
    args = parser.parse_args()

    fields = [
        "file",
        "type",
        "iter_ms",
        "total_reqs",
        "prefill_reqs",
        "decode_reqs",
        "total_tokens",
        "prefill_tokens",
        "decode_tokens",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()

    for path in args.logs:
        for row in parse_file(path):
            writer.writerow(row)


if __name__ == "__main__":
    main()
