# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which fields the Qwen3.8-Flash-Next (`qwen4_exp`) QSA arena holds.

The arena (`Qwen4ExpMetadataBuilder._qsa_fields`), the layers that bind its
rows (`Qwen4ExpAttention`), the PAGE namespace (`build_page_namespace`) and the
offload checks all decide the field set from these rules, so a change to one
reaches every reader. Kept free of AITER so the namespace and the checks run on
a plain runner.
"""

from __future__ import annotations

QSA_CACHE_FORMATS = ("bf16", "fp8", "fp4")
# The QSA layers' row space. Named rather than derived from
# `layer_id // full_attention_interval`, because the row a layer holds is the
# order it was bound in: on a non-first PP stage the global layer numbers do
# not start at zero, and that arithmetic would index past the pool.
QSA_ROWS = "qsa"
# MTP draft QSA layers that keep BF16 caches while the target's are quantized
# (`ATOM_QSA_DRAFT_KV_BF16=1`): their own rows, their own `*_draft` fields.
QSA_DRAFT_ROWS = "qsa_draft"
QSA_MODEL_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})


def qsa_cache_formats(atom_config) -> tuple[str, str]:
    """`(K/V format, compressed index-key format)` this engine serves.

    The raw index keys stay BF16 whatever is asked: they are mean-pooled
    before normalization, and a group can straddle a prefill chunk.
    An unset index dtype follows the KV cache's, as `Config` resolves it.
    """
    kv = getattr(atom_config, "kv_cache_dtype", "bf16") or "bf16"
    kv = "bf16" if kv in ("auto", "bfloat16") else kv
    index = getattr(atom_config, "index_cache_dtype", None) or kv
    index = "bf16" if index in ("auto", "bfloat16") else index
    if kv not in ("bf16", "fp8") or index not in QSA_CACHE_FORMATS:
        raise ValueError(
            f"Qwen QSA serves bf16/fp8 KV and bf16/fp8/fp4 index caches, got "
            f"kv_cache_dtype={kv!r} index_cache_dtype={index!r}"
        )
    return kv, index


def qsa_caches_quantized(config) -> bool:
    """Whether the QSA K/V or compressed index-key cache is stored quantized."""

    return qsa_cache_formats(config) != ("bf16", "bf16")


def qsa_draft_rows(config) -> bool:
    """Whether an MTP draft QSA layer keeps BF16 caches in rows of its own:
    under quantized target caches, with `ATOM_QSA_DRAFT_KV_BF16`."""

    from atom.utils import envs

    return bool(envs.ATOM_QSA_DRAFT_KV_BF16) and qsa_caches_quantized(config)


def qsa_row_space(config, layer_num: int, num_target_layers: int) -> str:
    """The row space of QSA layer `layer_num`. The MTP draft layer sits past
    the `num_target_layers` target stack."""

    if layer_num >= num_target_layers and qsa_draft_rows(config):
        return QSA_DRAFT_ROWS
    return QSA_ROWS


def qsa_caches_rope_positions(config) -> bool:
    """Whether the arena caches per-token mRoPE positions.

    Image/video requests give mRoPE three genuinely different position rows,
    and then a compressed key's group position cannot be recomputed
    arithmetically. Cached only when the engine can actually be handed an
    image: it costs ~1.4 GB of the paged budget.
    """

    return getattr(config, "multimodal_config", None) is not None


def qsa_page_segments(config) -> dict | None:
    """The facts besides the namespace's own fields that change the PAGE
    segment set, for `build_page_namespace`; None for any other model.

    Image input adds `rope_positions` after the index keys, and BF16 draft
    rows add `*_draft` fields, so a block holds other bytes and the GDN image
    starts elsewhere.
    """

    from atom.utils import get_hf_text_config

    text_config = get_hf_text_config(config.hf_config)
    if getattr(text_config, "model_type", None) not in QSA_MODEL_TYPES:
        return None
    spec = getattr(config, "speculative_config", None)
    draft = getattr(spec, "method", None) == "mtp" and qsa_draft_rows(config)
    return {
        "image_input": qsa_caches_rope_positions(config),
        "draft_kv": "bf16" if draft else "target",
    }
