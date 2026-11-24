#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Did two runs feed the model the same data in the same order?

Reads the "[lumen] batch_hash rank=R call=N ... md5=H" lines train_dit_lumen.py
prints with LUMEN_BATCH_HASH=1 from the condition model's entry points, and
compares them per rank and per call. Every rank is checked, since each reads its
own shard of the data.

    compare_batch_hash.py <log A> <log B> [get_condition|process_condition]
"""

import re
import sys

LINE = re.compile(
    r"\[(?:lumen|probe)\] batch_hash rank=(\d+) call=(\d+) (\S+) md5=([0-9a-f]{32})"
)


def load(path, method):
    seen = {}
    with open(path, errors="ignore") as fh:
        for line in fh:
            for m in LINE.finditer(line):
                if (
                    method
                    and not m.group(3).endswith("." + method)
                    and m.group(3).count(".") == 1
                ):
                    continue
                if method and "." not in m.group(3) and method != "get_condition":
                    continue  # lines from before the per-method tags were all get_condition
                seen[(int(m.group(1)), int(m.group(2)))] = m.group(4)
    return seen


def main():
    method = sys.argv[3] if len(sys.argv) > 3 else "get_condition"
    print(f"comparing {method} inputs")
    a, b = load(sys.argv[1], method), load(sys.argv[2], method)
    keys = sorted(set(a) | set(b))
    same = sum(1 for k in keys if a.get(k) == b.get(k) and a.get(k) is not None)
    missing = [k for k in keys if k not in a or k not in b]
    differ = [k for k in keys if k in a and k in b and a[k] != b[k]]
    ranks = sorted({r for r, _ in keys})
    print(f"A: {len(a)} hashed samples   B: {len(b)}   ranks: {ranks}")
    print(
        f"identical: {same}   differ: {len(differ)}   present in only one run: {len(missing)}"
    )
    for k in differ[:10]:
        print(f"  DIFFER rank={k[0]} call={k[1]}  {a[k]}  vs  {b[k]}")
    for k in missing[:10]:
        print(f"  MISSING rank={k[0]} call={k[1]}  A={a.get(k)}  B={b.get(k)}")
    print(
        "=> same data, same order"
        if not differ and not missing and same
        else "=> NOT identical"
    )
    if differ or missing:
        from collections import Counter

        ca, cb = Counter(a.values()), Counter(b.values())
        print(
            f"as multisets of samples: A has {len(ca)} distinct, B has {len(cb)} distinct, "
            f"shared {len(set(ca) & set(cb))}; "
            + (
                "=> the SAME samples, only reordered"
                if ca == cb
                else "=> the sample sets differ"
            )
        )


if __name__ == "__main__":
    main()
