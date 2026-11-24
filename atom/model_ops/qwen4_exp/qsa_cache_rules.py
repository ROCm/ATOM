# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which QSA caches a Qwen3.8 engine keeps and how the PAGE layout id spells
them.

The arena, the layers binding its rows, the PAGE namespace, the layout id and
`Qwen4ExpBackend.validate_config` all read these. Kept out of `qwen4_exp_attn`
and `ops/qsa.py`, which import Triton and AITER, so `kv_transfer` can import
them and CPU CI runs their tests.
"""

from collections.abc import Iterable

import torch

from atom.utils import envs

QSA_CACHE_FORMATS = ("bf16", "fp8", "fp4")

# The QSA layers' row space. Named rather than derived from
# `layer_id // full_attention_interval`, because the row a layer holds is the
# order it was bound in: on a non-first PP stage the global layer numbers do
# not start at zero, and that arithmetic would index past the pool.
QSA_ROWS = "qsa"
# MTP draft QSA layers that keep BF16 caches while the target's are quantized
# (`ATOM_QSA_DRAFT_KV_BF16=1`): their own rows, their own `*_draft` fields.
QSA_DRAFT_ROWS = "qsa_draft"
# Side caches in `KVCacheTensor.extra_planes` order. The two index caches
# are always present; rope positions follow the multimodal config, and only
# the compressed-index scale depends on the index format. The vLLM MP
# publisher names its PAGE regions after these.
QSA_SIDE_PLANES = (
    "index_raw",
    "index_compressed",
    "rope_positions",
    "index_compressed_scale",
)


def normalize_qsa_cache_formats(atom_config) -> tuple[str, str]:
    """`(K/V format, compressed index-key format)` as asked, served or not.

    An unset index dtype follows the KV cache's, as `Config` resolves it.
    """
    kv = atom_config.kv_cache_dtype or "bf16"
    kv = "bf16" if kv in ("auto", "bfloat16") else kv
    index = atom_config.index_cache_dtype or kv
    return kv, "bf16" if index in ("auto", "bfloat16") else index


def qsa_cache_formats(atom_config) -> tuple[str, str]:
    """`(K/V format, compressed index-key format)` this engine serves.

    The raw index keys stay BF16 whatever is asked: they are mean-pooled
    before normalization, and a group can straddle a prefill chunk.
    """
    kv, index = normalize_qsa_cache_formats(atom_config)
    if kv not in ("bf16", "fp8") or index not in QSA_CACHE_FORMATS:
        raise ValueError(
            f"Qwen QSA serves bf16/fp8 KV and bf16/fp8/fp4 index caches, got "
            f"kv_cache_dtype={kv!r} index_cache_dtype={index!r}"
        )
    return kv, index


def qsa_draft_rows(formats: tuple[str, str]) -> bool:
    """Whether the MTP draft layer keeps BF16 rows of its own under the
    target's `qsa_cache_formats`."""
    return envs.ATOM_QSA_DRAFT_KV_BF16 and formats != ("bf16", "bf16")


def qsa_row_space(
    formats: tuple[str, str], layer_num: int, num_target_layers: int
) -> str:
    """The MTP draft layer sits past the `num_target_layers` target stack."""
    if layer_num >= num_target_layers and qsa_draft_rows(formats):
        return QSA_DRAFT_ROWS
    return QSA_ROWS


def qsa_caches_rope_positions(config) -> bool:
    # Image/video requests give mRoPE three genuinely different position rows,
    # and then a compressed key's group position cannot be recomputed
    # arithmetically. Cache the per-token axes only when the engine can
    # actually be handed an image: they take paged budget.
    return config.multimodal_config is not None


def qsa_page_segments(config) -> dict:
    """What besides the namespace's own fields changes the PAGE segment set."""
    spec = config.speculative_config
    draft = getattr(spec, "method", None) == "mtp" and qsa_draft_rows(
        qsa_cache_formats(config)
    )
    return {
        "image_input": qsa_caches_rope_positions(config),
        "draft_kv": "bf16" if draft else "target",
    }


def _shape_and_dtype(shape, dtype) -> str:
    """`(shape),dtype` of one slot or entry, as the layout id spells it."""
    return f"{tuple(shape)},{dtype}".replace(" ", "")


def qsa_paged_layout_id(
    base: str,
    ssm_storage: str,
    page_fields: Iterable[tuple[str, tuple[int, ...], torch.dtype]],
    ple_planes: Iterable[tuple[str, tuple[int, ...], torch.dtype]],
) -> str:
    """`base` plus what it cannot see: the SSM's physical order, the PAGE
    fields `(name, shape, dtype)` and the PLE planes `(tag, shape, dtype)`,
    each in image order."""
    return (
        base
        + f":ssm_storage={ssm_storage}"
        + ":page="
        + ";".join(f"{n}={_shape_and_dtype(s, d)}" for n, s, d in page_fields)
        + "".join(f":ple_{t}={_shape_and_dtype(s, d)}" for t, s, d in ple_planes)
    )
