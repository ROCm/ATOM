#!/usr/bin/env python3
"""Generate observed Kimi-K3 BF16 GEMM shapes for gfx1250 tuning."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

# Recurred in the B0 DP16/EP16 AgentX server logs. The first five families
# dominate the 16k chunked-prefill path; the rest are secondary projections.
HOT_FAMILIES = (
    (67584, 7168),
    (7168, 33792),
    (12288, 7168),
    (7168, 12288),
    (7168, 6144),
)
SECONDARY_FAMILIES = (
    (49376, 7168),
    (18432, 1536),
    (24576, 512),
    (12288, 128),
    (10240, 7168),
)

# The LM head is observed at small decode M. Sweeping it to the prefill budget
# would allocate a huge output shape that the serving path does not request.
LM_HEAD_FAMILY = (163840, 7168)
LM_HEAD_M_VALUES = (1, 2, 4, 8, 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--stage",
        choices=("hot", "all"),
        default="hot",
        help="hot: five dominant families; all: include secondary and LM head",
    )
    args = parser.parse_args()

    families = list(HOT_FAMILIES)
    if args.stage == "all":
        families.extend(SECONDARY_FAMILIES)

    rows = []
    for n, k in families:
        rows.extend(
            {
                "M": m,
                "N": n,
                "K": k,
                "bias": False,
                "dtype": "torch.bfloat16",
                "outdtype": "torch.bfloat16",
                "scaleAB": False,
                "bpreshuffle": False,
            }
            for m in M_VALUES
        )
    if args.stage == "all":
        n, k = LM_HEAD_FAMILY
        rows.extend(
            {
                "M": m,
                "N": n,
                "K": k,
                "bias": False,
                "dtype": "torch.bfloat16",
                "outdtype": "torch.bfloat16",
                "scaleAB": False,
                "bpreshuffle": False,
            }
            for m in LM_HEAD_M_VALUES
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} shapes to {args.output}")


if __name__ == "__main__":
    main()
