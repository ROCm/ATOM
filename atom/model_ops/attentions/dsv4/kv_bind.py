"""DeepSeek-V4 KV-cache tensor binding.

The functions in this module only bind runner-owned storage to model modules.
They deliberately keep model imports lazy so importing the attention backend
does not recurse through ``atom.models.deepseek_v4``.
"""

from __future__ import annotations

from typing import Any

import torch

from atom.kv_cache.dsv4.arena import ArenaGroupSpec, group_of_ratio


def _prepare_bind_layout(builder: Any) -> tuple[Any, int, bool]:
    """Return runner/SWA split state and persist arena row strides."""
    runner = builder.model_runner
    swa_pages = runner.num_swa_blocks * builder.block_size
    arena_on = bool(getattr(builder, "_arena_on", False))
    if arena_on:
        specs = [
            ArenaGroupSpec.coerce(spec) for spec in runner.config.v4_arena_group_specs
        ]
        swa_pages = 0
        # Consumers that build SWA indices run after module binding.  Persisting
        # this map is therefore part of the binding contract, not scratch state.
        builder._arena_group_rows = {spec.name: int(spec.chunk_rows) for spec in specs}
    return runner, swa_pages, arena_on


def _swa_row_stride(builder: Any, layer_id: int) -> int:
    """Physical row stride for one SWA block in a layer's unified pool."""
    if not bool(getattr(builder, "_arena_on", False)):
        return builder.block_size
    group = group_of_ratio(builder.compress_ratios[layer_id])
    return getattr(builder, "_arena_group_rows", {}).get(group, builder.block_size)


def _bind_v4_attention(
    builder: Any,
    module: Any,
    *,
    runner: Any,
    swa_pages: int,
    arena_on: bool,
) -> None:
    """Bind normal and DSpark-draft V4 attention storage."""
    if getattr(module, "dspark_draft", False):
        module.swa_kv = torch.zeros(
            (swa_pages, builder.head_dim),
            dtype=torch.bfloat16,
            device=runner.device,
        )
        module.swa_block_size = builder.block_size
        module.swa_row_stride = builder.block_size
        module.kv_fp8 = False
        module.unified_kv = None
        module.unified_kv_rope = None
        module.swa_kv_rope = None
        return

    unified = runner.v4_unified_kv[module.layer_id]
    module.unified_kv = unified
    module.swa_kv = unified if arena_on else unified[:swa_pages]
    module.swa_block_size = builder.block_size
    module.swa_row_stride = _swa_row_stride(builder, module.layer_id)
    module.kv_fp8 = builder._kv_fp8
    if builder._kv_fp8:
        rope = runner.v4_unified_kv_rope[module.layer_id]
        module.unified_kv_rope = rope
        module.swa_kv_rope = rope if arena_on else rope[:swa_pages]
    else:
        module.unified_kv_rope = None
        module.swa_kv_rope = None


def _bind_v4_indexer(builder: Any, module: Any, *, runner: Any) -> None:
    """Bind the 3-D FP8 indexer cache for one CSA layer."""
    layer_id = int(module.prefix.split(".")[1])
    pos = builder.layer_id_to_csa_pos[layer_id]
    # Keep [num_blocks, k1_csa, aligned_dim].  The gather kernel infers its
    # block size from dimension 1.
    module.kv_cache = runner.v4_csa_idx_kv[pos]


def _bind_indexer_inner_compressor(
    builder: Any,
    module: Any,
    *,
    runner: Any,
    layer_id: int,
) -> None:
    """Bind the CSA indexer's compressor state and inline FP8 scales."""
    assert module.compress_ratio == 4, "Indexer-inner Compressor only on CSA layers"
    pos = builder.layer_id_to_csa_pos[layer_id]
    module.kv_state = runner.v4_csa_idx_kv_state[pos]
    module.score_state = runner.v4_csa_idx_score_state[pos]
    if builder.enable_csa_prefix_state_cache:
        module.boundary_kv = runner.v4_csa_idx_boundary_kv[pos]
        module.boundary_score = runner.v4_csa_idx_boundary_score[pos]
        module._csa_owner = "idx"

    idx_kv = runner.v4_csa_idx_kv[pos]
    module.kv_cache = idx_kv
    nb, k1, aligned_dim = idx_kv.shape
    head_dim = builder.index_head_dim
    assert (
        k1 * aligned_dim
    ) % 4 == 0, f"per-block bytes ({k1 * aligned_dim}) must be 4-aligned"
    block_fp32_stride = (k1 * aligned_dim) // 4
    scale_fp32_offset = (k1 * head_dim) // 4
    idx_kv_f32 = idx_kv.view(torch.float32)
    # as_strided's storage_offset is absolute.  Include the layer slice offset
    # or every indexer layer aliases layer zero's scale rows.
    module.cache_scale = idx_kv_f32.view(-1).as_strided(
        size=(nb, k1),
        stride=(block_fp32_stride, 1),
        storage_offset=idx_kv_f32.storage_offset() + scale_fp32_offset,
    )
    module.write_mode = "indexer_fp8"
    module.kv_cache_rope = None


def _bind_main_compressor_c4(
    builder: Any,
    module: Any,
    *,
    runner: Any,
    layer_id: int,
    swa_pages: int,
) -> None:
    """Bind a ratio-4 CSA main compressor."""
    pos = builder.layer_id_to_csa_pos[layer_id]
    module.kv_state = runner.v4_csa_main_kv_state[pos]
    module.score_state = runner.v4_csa_main_score_state[pos]
    if builder.enable_csa_prefix_state_cache:
        module.boundary_kv = runner.v4_csa_main_boundary_kv[pos]
        module.boundary_score = runner.v4_csa_main_boundary_score[pos]
        module._csa_owner = "main"

    num_blocks = runner.num_physical_kvcache_blocks
    unified = runner.v4_unified_kv[layer_id]
    module.kv_cache = unified[swa_pages:].view(
        num_blocks, builder.k1_csa, builder.head_dim
    )
    if builder._kv_fp8:
        rope = runner.v4_unified_kv_rope[layer_id]
        module.kv_cache_rope = rope[swa_pages:].view(
            num_blocks, builder.k1_csa, builder.rope_head_dim
        )
        module.write_mode = "main_2buff_fp8"
    else:
        module.kv_cache_rope = None
        module.write_mode = "bf16"


def _bind_main_compressor_c128(
    builder: Any,
    module: Any,
    *,
    runner: Any,
    layer_id: int,
    swa_pages: int,
    arena_on: bool,
) -> None:
    """Bind a ratio-128 HCA main compressor."""
    pos = builder.layer_id_to_hca_pos[layer_id]
    module.kv_state = runner.v4_hca_main_kv_state[pos]
    module.score_state = runner.v4_hca_main_score_state[pos]
    num_blocks = runner.num_physical_kvcache_blocks
    unified = runner.v4_unified_kv[layer_id]
    # C128 arena rows do not track the fatter c4 group's physical block count.
    # Derive the page count from this layer's actual storage when arena-backed.
    hca_pages = (
        unified[swa_pages:].shape[0] // builder.k2_hca if arena_on else num_blocks
    )
    module.kv_cache = unified[swa_pages:].view(
        hca_pages, builder.k2_hca, builder.head_dim
    )
    if builder._kv_fp8:
        rope = runner.v4_unified_kv_rope[layer_id]
        module.kv_cache_rope = rope[swa_pages:].view(
            hca_pages, builder.k2_hca, builder.rope_head_dim
        )
        module.write_mode = "main_2buff_fp8"
    else:
        module.kv_cache_rope = None
        module.write_mode = "bf16"


def bind_kv_cache_tensor(builder: Any, layer_id: int, module: Any) -> bool:
    """Bind a recognized V4 module and return whether it was handled."""
    # Local imports avoid a backend -> model -> backend import cycle.
    from atom.models.deepseek_v4 import Compressor as V4Compressor
    from atom.models.deepseek_v4 import DeepseekV4Attention as V4Attention
    from atom.models.deepseek_v4 import Indexer as V4Indexer

    runner, swa_pages, arena_on = _prepare_bind_layout(builder)

    if isinstance(module, V4Attention):
        _bind_v4_attention(
            builder,
            module,
            runner=runner,
            swa_pages=swa_pages,
            arena_on=arena_on,
        )
        return True

    if isinstance(module, V4Indexer):
        _bind_v4_indexer(builder, module, runner=runner)
        return True

    if not isinstance(module, V4Compressor):
        return False

    parts = module.prefix.split(".")
    compressor_layer_id = int(parts[1])
    if "indexer" in parts:
        _bind_indexer_inner_compressor(
            builder,
            module,
            runner=runner,
            layer_id=compressor_layer_id,
        )
    elif module.compress_ratio == 4:
        _bind_main_compressor_c4(
            builder,
            module,
            runner=runner,
            layer_id=compressor_layer_id,
            swa_pages=swa_pages,
        )
    elif module.compress_ratio == 128:
        _bind_main_compressor_c128(
            builder,
            module,
            runner=runner,
            layer_id=compressor_layer_id,
            swa_pages=swa_pages,
            arena_on=arena_on,
        )
    else:
        raise ValueError(
            f"Unknown V4 compress_ratio={module.compress_ratio} on Compressor at "
            f"prefix={module.prefix!r}"
        )
    return True
