# SPDX-License-Identifier: MIT
"""Check explicit FP8 drafting before loading weights or capturing graphs."""

from importlib import import_module


def validate_fp8_draft_support():
    gfx = import_module("aiter.jit.utils.chip_info").get_gfx_runtime()
    reason = None
    if gfx not in ("gfx950", "gfx1250"):
        reason = f"AITER V4.1 A8W8 attention does not support {gfx}"
    else:
        try:
            dispatcher = import_module("aiter.ops.mla_sparse_prefill")
            kernels = import_module("aiter.ops.opus.mla")
            if not callable(
                getattr(dispatcher, "mla_sparse_prefill_v41", None)
            ) or not callable(
                getattr(kernels, f"opus_mla_v41_prefill_a8w8_{gfx}_fwd", None)
            ):
                reason = f"AITER is missing the V4.1 A8W8 interface for {gfx}"
        except ImportError:
            reason = f"AITER's V4.1 A8W8 interface for {gfx} cannot be imported"
    if reason is not None:
        raise ValueError(
            f"draft_kv_dtype=fp8 requires supported AITER V4.1 A8W8 attention: {reason}. "
            "Use draft_kv_dtype=auto for BF16 drafting."
        )
