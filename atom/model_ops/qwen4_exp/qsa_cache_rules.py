# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which QSA caches a Qwen3.8 engine keeps, and which KV transfer may move them.

The arena, the layers binding its rows, the PAGE namespace and
`Qwen4ExpBackend.validate_config` all read these. Kept out of `qwen4_exp_attn`
and `ops/qsa.py`, which import Triton and AITER, so CPU CI runs them.
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


def qsa_caches_quantized(config) -> bool:
    return qsa_cache_formats(config) != ("bf16", "bf16")


def qsa_draft_rows(config) -> bool:
    return bool(envs.ATOM_QSA_DRAFT_KV_BF16) and qsa_caches_quantized(config)


def qsa_row_space(config, layer_num: int, num_target_layers: int) -> str:
    """The MTP draft layer sits past the `num_target_layers` target stack."""
    if layer_num >= num_target_layers and qsa_draft_rows(config):
        return QSA_DRAFT_ROWS
    return QSA_ROWS


def qsa_caches_rope_positions(config) -> bool:
    # Image/video requests give mRoPE three genuinely different position rows,
    # and then a compressed key's group position cannot be recomputed
    # arithmetically. Cached only when the engine can actually be handed an
    # image: it costs ~1.4 GB of the paged budget.
    return getattr(config, "multimodal_config", None) is not None


def qsa_page_segments(config) -> dict:
    """What besides the namespace's own fields changes the PAGE segment set."""
    spec = getattr(config, "speculative_config", None)
    draft = getattr(spec, "method", None) == "mtp" and qsa_draft_rows(config)
    return {
        "image_input": qsa_caches_rope_positions(config),
        "draft_kv": "bf16" if draft else "target",
    }


def kv_transfer_leaves(config) -> list[str]:
    return [name for name, _ in KVConnectorFactory.leaf_connectors(config) or ()]


def check_kv_transfer(config) -> None:
    """Qwen3.8's GDN and PLE state ride the PAGE checkpoint image, so only an
    offload tier registered with `reads_page_images` may run."""
    leaves = kv_transfer_leaves(config)
    for name in leaves:
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
                "Qwen3.8 offload supports "
                f"{', '.join(KVConnectorFactory.page_image_readers())}; "
                f"kv_connector={name!r} does not read the QSA/GDN/PLE PAGE "
                "checkpoint image."
            )
    if "lmcache_offload" in leaves and qsa_caches_quantized(config):
        # The in-process codec copies what `build_kv_cache_tensor` hands it:
        # no scale planes, so a CPU hit restores rows under stale scales.
        raise ValueError(
            "Qwen3.8 lmcache_offload does not carry the scale planes of a "
            "quantized QSA cache yet (kv_cache_dtype="
            f"{getattr(config, 'kv_cache_dtype', None)!r}, index_cache_dtype="
            f"{getattr(config, 'index_cache_dtype', None)!r}). Use "
            "--kv-cache-dtype bf16 --index-cache-dtype bf16, or lmcache_mp, "
            "which stores every QSA field; ROCm/ATOM#2655 lifts this."
        )
