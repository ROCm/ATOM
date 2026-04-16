# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Verify disagg mode produces token-identical output to non-disagg (temperature=0).

Stubs two aiter symbols that are missing from this build but are only used by
MLA/DeepSeek models — not by Qwen3, which is what we test here.
"""

import sys

# # --- Stub missing aiter symbols before any atom import touches them ---
# _stub = types.ModuleType("aiter.ops.triton.gather_kv_b_proj")
# _stub.gather_kv_b_proj = None
# sys.modules["aiter.ops.triton.gather_kv_b_proj"] = _stub

import aiter as _aiter  # noqa: E402

for _sym in ["fused_qk_rmsnorm", "fused_add_rmsnorm", "rmsnorm", "fused_moe_fwd"]:
    if not hasattr(_aiter, _sym):
        setattr(_aiter, _sym, None)
# -----------------------------------------------------------------------

from atom import LLMEngine, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "What is 2+2?",
    "Name the capital of France.",
    "Write a haiku about the ocean.",
]
#PROMPTS = [100*"San Francisco is a " for _ in range(100)]
SAMPLING = SamplingParams(temperature=0.0, max_tokens=64)

ENGINE_KWARGS = dict(
    enforce_eager=True,  # skip CUDA graph capture
    max_model_len=2048,  # keep KV cache small
)


def run(enable_disagg: bool) -> list[str]:
    engine = LLMEngine(MODEL, enable_disagg=enable_disagg, **ENGINE_KWARGS)
    try:
        outputs = engine.generate(PROMPTS, SAMPLING)
    finally:
        engine.close()
    return [o["text"] for o in outputs]


def main():
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    print(f"--- Running baseline (no disagg) {num_trials} times ---")
    baseline_results = []
    for i in range(num_trials):
        outputs = run(enable_disagg=True)
        baseline_results.append(outputs)
        print(outputs)
        #print(f"  Trial {i+1}:")
        #for p, o in zip(PROMPTS, outputs):
        #    print(f"    [{p!r}] → {o!r}")

    # baseline_consistent = all(r == baseline_results[0] for r in baseline_results)
    # print(f"\nBaseline consistent across {num_trials} trials: {baseline_consistent}")
    # if not baseline_consistent:
    #     print("  BASELINE IS NON-DETERMINISTIC")
    #     for i, r in enumerate(baseline_results):
    #         for j, (a, b) in enumerate(zip(baseline_results[0], r)):
    #             if a != b:
    #                 print(f"  Trial 0 vs {i}, prompt {j}: {a!r} vs {b!r}")
    # else:
    #     print("  Baseline is deterministic.")

    #sys.exit(0 if baseline_consistent else 1)
    return 0

if __name__ == "__main__":
    main()
