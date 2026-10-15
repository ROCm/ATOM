# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 PAGE/SLOT transfer regions for the ATOM vLLM plugin.

Port of ``deepseek_v4_attn.CommonAttentionBuilder_DSV4.get_kv_transfer_tensors``
onto the plugin's proxy allocation, carved from
``slice_deepseek_v4_proxy_cache_views`` since there is no ATOM runner here.

The proxy's ``FullAttentionSpec`` is a memory request, not a description of
contents: ATOM re-carves it into planes, so block N's bytes are not at
``N * page_size_bytes`` and a connector handed the raw tensor would copy
correctly-sized ranges of the wrong data. ``KVTransferRegion`` restates the
layout as base pointer plus per-unit stride.

Only PAGE (``block_regions``) is described, block-keyed and forward-indexed.
The per-request SLOT state is deliberately not advertised: it is recomputed by
re-forwarding the prefix tail rather than transferred, and naming its regions
is what makes the worker allocate a SLOT codec and staging pool for it.
"""

from __future__ import annotations

import logging

import torch

from atom.kv_transfer.disaggregation.types import (
    KVTransferRegion,
    KVTransferTensors,
)
from atom.plugin.vllm.deepseek_v4_bridge import (
    ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    _index_row_bytes,
    _layer_counts,
    _v4_kv_fp8,
    _v4_rope_head_dim,
    _v4_state_layout,
    _v4_win_with_spec,
    slice_deepseek_v4_proxy_cache_views,
)

logger = logging.getLogger(__name__)


def _plane_roles_and_row_bytes(
    head_dim: int, rope_head_dim: int, kv_fp8: bool
) -> list[tuple[str, int]]:
    """Return ``(semantic_role, row_bytes)`` per KV plane, in carve order.

    Order is ``[NoPE][RoPE?][CSA indexers]``; the RoPE plane exists only under
    the fp8 2buff layout.
    """
    planes = [("dsv4.main_kv.nope", head_dim * (1 if kv_fp8 else 2))]
    if kv_fp8:
        planes.append(("dsv4.main_kv.rope", rope_head_dim * 2))
    return planes


def build_deepseek_v4_transfer_tensors(
    proxy_kv_cache: torch.Tensor,
    vllm_config,
) -> KVTransferTensors:
    """Describe the proxy arena as PAGE and SLOT transfer regions.

    Every parameter is derived the way ``_proxy_page_bytes`` derived it when it
    sized the spec, so the carve agrees with the allocation by construction.
    Regions alias the allocation; nothing is copied.
    """
    # Every PAGE and SLOT region spans the full model-layer plane, so a
    # stage-local worker would register the global carve and transfer bytes
    # belonging to another stage. Native refuses the same configuration in
    # `deepseek_v4_attn.get_kv_transfer_tensors`; fail closed here too rather
    # than transfer mismatched bytes.
    pp_size = int(
        getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) or 1
    )
    if pp_size > 1:
        raise NotImplementedError(
            "ATOM V4 offload does not support pipeline parallelism "
            f"(pipeline_parallel_size={pp_size}): the PAGE/SLOT regions describe "
            "every layer's plane, not a stage's slice of it"
        )

    hf = vllm_config.model_config.hf_config
    ratios, _dense, n_csa, _n_hca = _layer_counts(hf)
    head_dim = int(getattr(hf, "head_dim", 512))
    index_head_dim = int(getattr(hf, "index_head_dim", 128))
    rope_head_dim = _v4_rope_head_dim(hf)
    kv_fp8 = _v4_kv_fp8(vllm_config)
    # The CSA regions below describe one fp8 pool per layer. An fp4 indexer
    # splits into a data pool plus an e8m0 scale pool, which this carve does
    # not enumerate, so a load would restore half the indexer state and the
    # sparse attention would read scales that were never written.
    index_dtype = str(
        getattr(hf, "index_cache_dtype", None)
        or getattr(vllm_config.cache_config, "index_cache_dtype", None)
        or "fp8"
    ).lower()
    if "fp4" in index_dtype:
        raise NotImplementedError(
            "ATOM V4 offload does not support an fp4 CSA indexer "
            f"(index_cache_dtype={index_dtype!r}): its e8m0 scale pool is a "
            "separate region this carve does not describe"
        )
    num_slots = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
    win_with_spec = _v4_win_with_spec(
        vllm_config, int(getattr(hf, "sliding_window", 128))
    )

    # `arena_planes=None` / `row_widths=None` skips StateArena construction: a
    # second arena over the same storage would have no reader, and the slot
    # regions below are described by geometry alone.
    _arena_planes, arena_rows, _row_widths = _v4_state_layout(vllm_config, kv_fp8)
    views = slice_deepseek_v4_proxy_cache_views(
        proxy_kv_cache,
        compress_ratios=ratios,
        num_slots=num_slots,
        window_size=win_with_spec,
        head_dim=head_dim,
        index_head_dim=index_head_dim,
        kv_fp8=kv_fp8,
        rope_head_dim=rope_head_dim,
        arena_planes=None,
        arena_rows=arena_rows,
        row_widths=None,
    )

    geo = views["geometry"]
    num_blocks = int(views["num_blocks"])
    plane_specs = _plane_roles_and_row_bytes(head_dim, rope_head_dim, kv_fp8)
    planes = [views["kv_plane"]]
    if kv_fp8:
        planes.append(views["kv_plane_rope"])

    block_regions: list[KVTransferRegion] = []

    # PAGE: one region per plane, not per layer. A block's rows are one
    # envelope, so `base + id * unit_bytes` describes it exactly.
    for plane, (role, row_bytes) in zip(planes, plane_specs, strict=True):
        if plane is None:
            raise ValueError(f"V4 proxy plane {role!r} missing under kv_fp8={kv_fp8}")
        block_bytes = geo.block_bytes(row_bytes)
        if plane.shape[0] < num_blocks * geo.envelope_rows:
            raise ValueError(
                f"V4 proxy plane {role!r} has {plane.shape[0]} rows, need "
                f"{num_blocks * geo.envelope_rows} for {num_blocks} blocks of "
                f"{geo.envelope_rows}"
            )
        block_regions.append(
            KVTransferRegion(
                plane.data_ptr(),
                num_blocks * block_bytes,
                block_bytes,
                semantic_role=role,
            )
        )

    # PAGE: CSA indexer, already block-major per layer.
    csa_layer_ids = [i for i, r in enumerate(ratios) if r == 4]
    csa_rows_per_block = ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4
    index_row_bytes = _index_row_bytes(index_head_dim)
    indexers = views["csa_indexer"]
    if len(indexers) != n_csa or len(csa_layer_ids) != n_csa:
        raise ValueError(
            f"V4 proxy CSA indexer count mismatch: {len(indexers)} views, "
            f"{len(csa_layer_ids)} layer ids, {n_csa} expected"
        )
    for layer_id, idx in zip(csa_layer_ids, indexers, strict=True):
        block_regions.append(
            KVTransferRegion(
                idx.data_ptr(),
                idx.numel() * idx.element_size(),
                csa_rows_per_block * index_row_bytes,
                semantic_role=f"dsv4.csa_indexer.layer_{layer_id}",
            )
        )

    logger.info(
        "ATOM V4 (vLLM plugin): %d PAGE regions (%d blocks x %d B/block total), "
        "no SLOT regions (PAGE-only offload)",
        len(block_regions),
        num_blocks,
        sum(r.unit_bytes for r in block_regions),
    )

    # PAGE-only, and every stateful field is left at its default to say so.
    # `_validate_stateful_page_slot_geometry` calls a layout stateful when any
    # one of `swa_block_regions`, `num_slots` or
    # `expected_full_slot_region_count` is set, and then the worker allocates
    # the SLOT codec and its staging pool and initializes checkpoint storage --
    # GPU memory nothing here would ever read, since the per-request ring is
    # recomputed rather than transferred. Describing the regions and declining
    # to use them was the earlier spelling and it bought exactly that cost.
    #
    # `num_slots` still shapes the carve above, because the PAGE planes are cut
    # around the slot rows; it just is not advertised.
    tensors = KVTransferTensors(
        block_regions=block_regions,
        slot_regions=[],
    )
    # `num_blocks` is `init=False` on the dataclass, so it cannot be passed to
    # the constructor.
    tensors.set_block_count(num_blocks)
    return tensors
