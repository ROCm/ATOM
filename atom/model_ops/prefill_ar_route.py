# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""When a fused all-reduce + Gemma RMSNorm takes quick-reduce + a separate norm.

Kept free of AITER imports so the decision can be checked on a CPU runner; the
ops that use it live in `atom.model_ops.layernorm`
(`ATOM_PREFILL_AR_QR_MIN_TOKENS`).
"""

import inspect
from collections.abc import Callable


def prefill_ar_takes_split(
    num_tokens: int, min_tokens: int, qr_takes: Callable[[], bool]
) -> bool:
    """Whether an all-reduce + norm of `num_tokens` rows takes quick-reduce and
    a separate norm instead of the fused reduce-scatter kernel.

    `min_tokens` <= 0 means off. `qr_takes` is a zero-argument callable that
    says whether quick-reduce is on and accepts this tensor; it is asked only
    once the size qualifies, so decode-sized calls never look up the
    communicator.
    """
    return min_tokens > 0 and num_tokens >= min_tokens and bool(qr_takes())


def takes_gemma_norm(op, fn: Callable) -> bool:
    """Whether aiter's `rmsnorm2d_fwd_with_add` takes `gemma_norm`.

    `op` is the registered `torch.ops.aiter.rmsnorm2d_fwd_with_add` or None;
    `fn` is the Python function, read only when the op is not registered,
    because the registered op's Python wrapper is `(*args, **kwargs)`. aiter
    before ROCm/aiter#3872 has `use_model_sensitive_rmsnorm` where newer aiter
    has `gemma_norm`, so the split norm passes the flag by keyword and is
    turned off when neither the schema nor the signature names it.
    """
    if op is not None:
        params = str(op.default._schema)
    else:
        params = str(inspect.signature(fn).parameters)
    return "gemma_norm" in params
