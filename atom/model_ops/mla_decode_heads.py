# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Query-head floor of a plain MLA decode.

Kept free of the AITER runtime so the rule is importable (and unit-tested) on a
CPU-only runner.
"""

from atom.utils import envs

MLA_MIN_HEADS = 16  # AITER MLA kernels require at least 16 attention heads
_NARROW_MIN_HEADS = 12


def mla_decode_min_heads(
    kv_cache_dtype: str, is_sparse: bool, dcp_world_size: int
) -> int:
    """The query-head floor a decode pads its query heads up to.

    16, unless ``ATOM_MLA_MIN_HEADS=12`` and the decode is non-sparse, has no DCP
    and reads an FP8 KV cache. aiter's gfx950 FP8 asm decode also dispatches 12
    heads natively (12 * qlen <= 128), so a model with 12 heads a rank (Kimi-K3
    at TP8) can skip the zero padding to 16. Every other case keeps 16.

    The layer (``MLAAttention``) and the metadata builder
    (``AiterMLAMetadataBuilder``) both call this so their padded head counts
    match. That holds only while they pass the same answer for each argument:
    ``is_sparse`` must be True whenever the model carries any sparse/indexed MLA
    (the layer derives it from ``mla_modules``, the builder from
    ``model_runner.has_mla_indexer``), and ``kv_cache_dtype`` is compared only by
    its ``fp8`` prefix, so "fp8" and "fp8_e4m3" are treated alike.
    """
    floor = envs.ATOM_MLA_MIN_HEADS
    if floor not in (_NARROW_MIN_HEADS, MLA_MIN_HEADS):
        raise ValueError(f"ATOM_MLA_MIN_HEADS must be 12 or 16, got {floor}")
    if (
        floor == MLA_MIN_HEADS
        or is_sparse
        or dcp_world_size > 1
        or not str(kv_cache_dtype).startswith("fp8")
    ):
        return MLA_MIN_HEADS
    return floor
