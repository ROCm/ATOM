#!/usr/bin/env python3
"""Check that every worker of one engine built its KV pool at one size.

Each ModelRunner logs `Memory budget: ... num_kvcache_blocks=N` after the
block count is reduced across its TP and PCP groups. The engine serves with
rank 0's count, so a worker that logged a different N would be asked for a pool
it did not budget for.

Usage: check_kv_block_agreement.py SERVER_LOG --ranks N
"""

import argparse
import re
import sys

PATTERN = re.compile(r"Memory budget: .*\bnum_kvcache_blocks=(\d+)")


def block_counts(lines) -> list[int]:
    return [int(m.group(1)) for line in lines if (m := PATTERN.search(line))]


def disagreement(counts: list[int], ranks: int) -> str | None:
    if len(counts) != ranks:
        return f"expected {ranks} 'Memory budget' lines, found {len(counts)}"
    if len(set(counts)) != 1:
        return f"ranks disagree on num_kvcache_blocks: {sorted(set(counts))}"
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log")
    parser.add_argument("--ranks", type=int, required=True)
    args = parser.parse_args(argv)
    with open(args.log, encoding="utf-8", errors="replace") as f:
        counts = block_counts(f)
    problem = disagreement(counts, args.ranks)
    if problem:
        print(f"::error title=KV block counts::{problem}")
        return 1
    print(f"All {args.ranks} ranks built {counts[0]} KV blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
