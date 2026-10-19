# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``python -m atom.compass`` — DeepSeek-V4-Pro, TP8, DSpark."""

from __future__ import annotations

import argparse
import json

from atom.compass.dsv4_tp8_dspark import estimate
from atom.compass.price_list import PriceList


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analytic Compass estimate for DeepSeek-V4-Pro TP8 + DSpark. "
        "No GPU. The seconds are a peak roofline, not a calibrated prediction."
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prefill-tokens", type=int, default=1024)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument(
        "--acceptance-length",
        type=float,
        default=5.45,
        help="tokens emitted per decode step, including the target token",
    )
    parser.add_argument("--num-slots", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument(
        "--price-list",
        type=str,
        default=None,
        help="JSON of CUDA-graph GEMM prices; GEMMs then use measured seconds",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    prices = PriceList.load(args.price_list) if args.price_list else None
    result = estimate(
        batch=args.batch,
        prefill_tokens=args.prefill_tokens,
        context=args.context,
        num_speculative_tokens=args.num_speculative_tokens,
        acceptance_length=args.acceptance_length,
        num_slots=args.num_slots,
        num_blocks=args.num_blocks,
        tp=args.tp,
        price_list=prices,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "provenance": result.provenance,
                    "tp": result.tp,
                    "fits": result.fits,
                    "bytes_per_rank": result.bytes_per_rank,
                    "prefill_s": result.prefill.seconds,
                    "decode_s": result.decode_s,
                    "tpot_s": result.tpot_s,
                    "output_tokens_per_s": result.output_tokens_per_s,
                    "limits": list(result.limits),
                },
                indent=2,
            )
        )
        return
    print(result.summary())


if __name__ == "__main__":
    main()
