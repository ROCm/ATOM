# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which QSA caches a Qwen3.8 engine keeps, which KV transfer may move them,
and how the PAGE layout id spells them.

The arena, the layers binding its rows, the PAGE namespace, the layout id and
`Qwen4ExpBackend.validate_config` all read these. Kept out of `qwen4_exp_attn`
and `ops/qsa.py`, which import Triton and AITER, so `kv_transfer` can import
them and CPU CI runs their tests.
"""

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
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


def qsa_draft_rows(config) -> bool:
    if not envs.ATOM_QSA_DRAFT_KV_BF16:
        return False
    return qsa_cache_formats(config) != ("bf16", "bf16")


def qsa_row_space(config, layer_num: int, num_target_layers: int) -> str:
    """The MTP draft layer sits past the `num_target_layers` target stack."""
    if layer_num >= num_target_layers and qsa_draft_rows(config):
        return QSA_DRAFT_ROWS
    return QSA_ROWS


def qsa_caches_rope_positions(config) -> bool:
    # Image/video requests give mRoPE three genuinely different position rows,
    # and then a compressed key's group position cannot be recomputed
    # arithmetically. Cache the per-token axes only when the engine can
    # actually be handed an image: they take paged budget.
    return getattr(config, "multimodal_config", None) is not None


def qsa_page_segments(config) -> dict:
    """What besides the namespace's own fields changes the PAGE segment set."""
    spec = getattr(config, "speculative_config", None)
    draft = getattr(spec, "method", None) == "mtp" and qsa_draft_rows(config)
    return {
        "image_input": qsa_caches_rope_positions(config),
        "draft_kv": "bf16" if draft else "target",
    }


def _shape_and_dtype(shape, dtype) -> str:
    """`(shape),dtype` of one slot or entry, as the layout id spells it."""
    return f"{tuple(shape)},{dtype}".replace(" ", "")


def qsa_paged_layout_id(base: str, ssm_storage: str, page_fields, ple_planes) -> str:
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


def kv_transfer_leaves(config) -> list[str]:
    return [name for name, _ in KVConnectorFactory.leaf_connectors(config) or ()]


def check_kv_transfer(config) -> None:
    """Qwen3.8's GDN and PLE state ride the PAGE checkpoint image, so only an
    offload tier registered with `reads_page_images` may run."""
    for name in kv_transfer_leaves(config):
        if not KVConnectorFactory.is_offload_backend(name):
            # A P/D connector moves the QSA pages alone, and decode would
            # continue from a zero recurrent state: wrong output, no error.
            raise ValueError(
                "Qwen3.8 disaggregated serving is not supported: "
                f"kv_connector={name!r} would transfer the QSA pages without "
                "the GDN and PLE state. Use an offload connector "
                "(lmcache_offload or lmcache_mp) or run without "
                "kv_transfer_config."
            )
        if not KVConnectorFactory.reads_page_images(name):
            raise ValueError(
                "Qwen3.8 offload supports lmcache_offload and lmcache_mp; "
                f"kv_connector={name!r} does not read the QSA/GDN/PLE PAGE "
                "checkpoint image."
            )
