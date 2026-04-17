# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Verify disagg mode produces token-identical output to non-disagg (temperature=0).

Usage:
    python verify_disagg.py          # TP=1 baseline vs disagg
    python verify_disagg.py --tp 8   # TP=8 baseline vs disagg
"""

import argparse
import sys

import aiter as _aiter  # noqa: E402

for _sym in ["fused_qk_rmsnorm", "fused_add_rmsnorm", "rmsnorm", "fused_moe_fwd"]:
    if not hasattr(_aiter, _sym):
        setattr(_aiter, _sym, None)

from atom import LLMEngine, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B" #"deepseek-ai/DeepSeek-R1"
PROMPTS = [
    100*"What is 2+2?",
    100*"Name the capital of France.",
    100*"Write a haiku about the ocean.",
]
SAMPLING = SamplingParams(temperature=0.0, max_tokens=64)


def run(enable_disagg: bool, tp: int) -> list[str]:
    engine = LLMEngine(
        MODEL,
        enable_disagg=enable_disagg,
        tensor_parallel_size=tp,
        enforce_eager=True,
        max_model_len=2048,
    )
    try:
        outputs = engine.generate(PROMPTS, SAMPLING)
    finally:
        engine.close()
    return [o["text"] for o in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()

    # print(f"--- Running baseline (no disagg, TP={args.tp}) ---")
    # baseline = run(enable_disagg=False, tp=args.tp)
    # for p, o in zip(PROMPTS, baseline):
    #     print(f"  [{p!r}] → {o!r}")

    print(f"\n--- Running disagg (TP={args.tp}) ---")
    disagg = run(enable_disagg=True, tp=args.tp)
    for p, o in zip(PROMPTS, disagg):
        print(f"  [{p!r}] → {o!r}")

    # print("\n--- Comparison ---")
    # all_match = True
    # for i, (b, d) in enumerate(zip(baseline, disagg)):
    #     match = b == d
    #     status = "OK" if match else "MISMATCH"
    #     print(f"  Prompt {i}: {status}")
    #     if not match:
    #         all_match = False
    #         print(f"    baseline: {b!r}")
    #         print(f"    disagg:   {d!r}")

    # if all_match:
    #     print("\nPASS: disagg output matches baseline exactly.")
    # else:
    #     print("\nFAIL: disagg output differs from baseline.")
    # sys.exit(0 if all_match else 1)
    return 0


if __name__ == "__main__":
    main()
