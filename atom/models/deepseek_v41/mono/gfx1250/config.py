# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The one deployment the single-launch step is written for, and why a deployment cannot run it.

TP4, DSpark with a block of 5 (a verify of 6 rows a request), BF16 KV cache, FP8 index cache, up to 8 requests (48
target rows, 40 draft rows), the 40 target layers and 3 draft stages of DeepSeek-V4.1-Flash.
"""

from atom.mono.runtime.deployment import common_refusals, first_refusal

TP = 4
SUPPORTED_TP = (TP,)
SPEC_TOKENS = 5
ROWS_PER_REQ = SPEC_TOKENS + 1  # a request's verify rows: the anchor and its 5 drafts
MAX_REQUESTS = 8
MAX_ROWS = MAX_REQUESTS * ROWS_PER_REQ
NL, NDR = 40, 3  # target layers, draft stages


def config_refusal(atom_config, dims) -> str | None:
    """Why this deployment cannot run the single-launch step, or None."""
    spec = atom_config.speculative_config
    method = None if spec is None else spec.method
    # the shared refusals but their arch check (gfx950 only: the per-layer kernels' wave64 code)
    shared = [
        c
        for c in common_refusals(atom_config, dims, SUPPORTED_TP)
        if not (isinstance(c[1], str) and c[1].startswith("arch "))
    ]
    checks = shared + [
        (
            atom_config.kv_cache_dtype == "bf16",
            f"kv cache {atom_config.kv_cache_dtype}",
        ),
        (
            atom_config.index_cache_dtype == "fp8",
            f"index cache {atom_config.index_cache_dtype}",
        ),
        (method == "dspark", f"speculative method {method}"),
        (
            spec is None or spec.num_speculative_tokens == SPEC_TOKENS,
            "DSpark block size",
        ),
        # a request's verify is all 6 rows: the confidence schedule and ragged verify give it fewer
        (not atom_config.dspark.confidence_schedule, "DSpark confidence schedule"),
        (not atom_config.dspark.ragged, "DSpark ragged verify"),
    ]
    return first_refusal(checks)
