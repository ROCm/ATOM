from __future__ import annotations

import logging
from collections.abc import Hashable
from contextlib import contextmanager
from functools import cache
from types import SimpleNamespace

import numpy as np
import torch
from aiter import dtypes
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    visible_csa,
    visible_hca,
)

ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME = "model.layers.0.atom_deepseek_v4_proxy"
ATOM_DEEPSEEK_V4_DRAFT_PROXY_LAYER_PREFIX = "atom_deepseek_v4_draft_proxy"
ATOM_DEEPSEEK_V4_BLOCK_SIZE = 128
ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT = 256

logger = logging.getLogger(__name__)

# aiter's V4 native 2buff fp8 prefill (op4) / decode (op5) kernels exist only on
# gfx950 / gfx1250. Mirror native's guard (deepseek_v4_attn.py): a request for an
# fp8 KV cache on any other arch degrades to a bf16 cache instead of hard-failing.
_V4_FP8_SUPPORTED_GFX = ("gfx950", "gfx1250")
_V4_SWA_DEST_RATIOS = (0, 4, 128)


@cache
def _warn_once(msg: str, *args: Hashable) -> None:
    """Log once per distinct message+args; callers fire from per-step paths."""
    logger.warning(msg, *args)


def _v4_kv_fp8(vllm_config) -> bool:
    """Whether the V4 KV cache runs the native 2buff fp8 layout.

    True iff vLLM's ``--kv-cache-dtype`` is an fp8 spelling AND the arch is one
    aiter op4/op5 support. Any ``fp8*`` spelling (``fp8`` / ``fp8_e4m3`` / ...)
    maps to ATOM's single ``"fp8"`` 2buff path. This is the single authority for
    the fp8 decision -- the proxy KV-cache sizing (``_proxy_page_bytes`` /
    ``get_kv_cache_spec``), the view slicing, and the per-module bind all key off
    it, so pool geometry and the runtime dispatch never disagree.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    cache_dtype = getattr(cache_config, "cache_dtype", None) if cache_config else None
    if not (isinstance(cache_dtype, str) and cache_dtype.startswith("fp8")):
        return False
    try:
        from aiter.jit.utils.chip_info import get_gfx

        gfx = get_gfx()
    except Exception:
        gfx = None
    if gfx not in _V4_FP8_SUPPORTED_GFX:
        _warn_once(
            "DeepSeek-V4 --kv-cache-dtype %r (2buff fp8) is only supported on "
            "%s (aiter op4/op5); got gfx=%r. Falling back to a bf16 KV cache.",
            cache_dtype,
            "/".join(_V4_FP8_SUPPORTED_GFX),
            gfx,
        )
        return False
    return True


def _v4_rope_head_dim(hf_config) -> int:
    return int(getattr(hf_config, "qk_rope_head_dim", 64))


def deepseek_v4_draft_proxy_layer_name(hf_config) -> str:
    return (
        f"model.layers.{int(hf_config.num_hidden_layers)}."
        f"{ATOM_DEEPSEEK_V4_DRAFT_PROXY_LAYER_PREFIX}"
    )


def _index_row_bytes(index_head_dim: int) -> int:
    # `index_head_dim` FP8 bytes plus a 4-byte fp32 scale. Data and scale are
    # two REGIONS inside a block, not interleaved per row, so the alignment
    # that matters is the block stride (`rows_per_block * this`), not this
    # value. Rounding it up to a multiple of 16 pays the rounding once per row
    # instead of once per block — 1.5% of the KV pool for nothing.
    return index_head_dim + 4


def _layer_counts(hf_config) -> tuple[list[int], int, int, int]:
    ratios = [int(r) for r in (getattr(hf_config, "compress_ratios", []) or [])]
    csa = sum(1 for r in ratios if r == 4)
    hca = sum(1 for r in ratios if r == 128)
    dense = sum(1 for r in ratios if r == 0)
    return ratios, dense, csa, hca


def _v4_spec_steps(vllm_config) -> int:
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return 0
    n = getattr(spec, "num_speculative_tokens", None)
    return int(n) if n else 0


def _v4_win_with_spec(vllm_config, window_size: int) -> int:
    return int(window_size) + _v4_spec_steps(vllm_config)


def _v4_state_layout(vllm_config, kv_fp8: bool):
    from atom.model_ops.attentions.pool_layout.state_arena import (
        StateField,
        plan_field_planes,
    )

    hf = vllm_config.model_config.hf_config
    _ratios, _dense, n_csa, n_hca = _layer_counts(hf)
    head_dim = int(getattr(hf, "head_dim", 512))
    index_head_dim = int(getattr(hf, "index_head_dim", 128))
    ring_extra = _v4_spec_steps(vllm_config)
    fields = [
        StateField("csa_main_kv", n_csa, (8 + ring_extra, 2 * head_dim), torch.float32),
        StateField(
            "csa_main_score",
            n_csa,
            (8 + ring_extra, 2 * head_dim),
            torch.float32,
            float("-inf"),
        ),
        StateField(
            "csa_idx_kv",
            n_csa,
            (8 + ring_extra, 2 * index_head_dim),
            torch.float32,
        ),
        StateField(
            "csa_idx_score",
            n_csa,
            (8 + ring_extra, 2 * index_head_dim),
            torch.float32,
            float("-inf"),
        ),
        # `in_checkpoint=False` for the same reason the native list says so
        # (`deepseek_v4_attn._state_fields`): HCA pools `[P, P + 128)` with no
        # overlap, so a checkpoint boundary owes it nothing. This list does not
        # price an image today — only `plan_field_planes`, which ignores the
        # flag, reads it — but two declarations of one layout disagreeing about
        # the one rule `layout_id` fences is exactly what that fence cannot
        # catch, since the id is derived from the native list alone.
        StateField(
            "hca_main_kv",
            n_hca,
            (128 + ring_extra, head_dim),
            torch.float32,
            in_checkpoint=False,
        ),
        StateField(
            "hca_main_score",
            n_hca,
            (128 + ring_extra, head_dim),
            torch.float32,
            float("-inf"),
            in_checkpoint=False,
        ),
    ]
    row_widths = [head_dim * (1 if kv_fp8 else 2)]
    if kv_fp8:
        row_widths.append(_v4_rope_head_dim(hf) * 2)
    planes, arena_rows = plan_field_planes(fields, row_widths)
    return planes, arena_rows, row_widths


def _proxy_region_byte_sizes(
    *,
    geometry,
    csa_layers: int,
    num_blocks: int,
    head_dim: int,
    rope_head_dim: int,
    index_head_dim: int,
    kv_fp8: bool,
) -> list[int]:
    """Linear proxy layout: [NoPE plane][RoPE plane?][CSA indexers].

    Matches native ``allocate_per_req_cache``: both KV planes are adjacent and
    ``plan_regions``-aligned; indexer bytes follow. Inserting indexers between
    the planes breaks ``StateArena``'s 256 B retype boundary on the RoPE plane.
    """
    nope_row_bytes = head_dim * (1 if kv_fp8 else 2)
    regions = [geometry.plane_bytes(nope_row_bytes)]
    if kv_fp8:
        regions.append(geometry.plane_bytes(rope_head_dim * 2))
    regions.append(
        csa_layers
        * num_blocks
        * (ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4)
        * _index_row_bytes(index_head_dim)
    )
    return regions


def _v4_proxy_min_blocks(vllm_config) -> int:
    """Block count ``_proxy_page_bytes`` amortizes the fixed per-slot regions
    over. Anything less and the page is short. See
    ``apply_vllm_v4_profiling_min_blocks_patch``.
    """
    max_model_len = int(vllm_config.model_config.max_model_len)
    return max(
        1,
        (max_model_len + ATOM_DEEPSEEK_V4_BLOCK_SIZE - 1)
        // ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    )


def _proxy_page_bytes(vllm_config) -> int:
    from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
        UnifiedPoolGeometry,
    )

    hf = vllm_config.model_config.hf_config
    ratios, _dense, csa_layers, _hca = _layer_counts(hf)
    head_dim = int(getattr(hf, "head_dim", 512))
    rope_head_dim = _v4_rope_head_dim(hf)
    index_head_dim = int(getattr(hf, "index_head_dim", 128))
    kv_fp8 = _v4_kv_fp8(vllm_config)
    _arena_planes, arena_rows, _row_widths = _v4_state_layout(vllm_config, kv_fp8)
    win = _v4_win_with_spec(vllm_config, int(getattr(hf, "sliding_window", 128)))
    max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
    min_blocks = _v4_proxy_min_blocks(vllm_config)
    geometry = UnifiedPoolGeometry(
        ratios,
        num_blocks=min_blocks,
        num_slots=max_num_seqs,
        ring_slots=win,
        block_size=ATOM_DEEPSEEK_V4_BLOCK_SIZE,
        arena_rows=arena_rows,
    )
    from atom.model_ops.attentions.pool_layout.state_arena import plan_regions

    regions = _proxy_region_byte_sizes(
        geometry=geometry,
        csa_layers=csa_layers,
        num_blocks=min_blocks,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        index_head_dim=index_head_dim,
        kv_fp8=kv_fp8,
    )
    _, total = plan_regions(regions)
    # vLLM 0.26 may pack this cache after another layer at a non-aligned
    # storage offset. Budget one-time leading slack so the runtime carve can
    # move its first plane to the 256B boundary StateArena requires.
    total += ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT - 1
    page_bytes = (total + min_blocks - 1) // min_blocks
    if page_bytes % ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT:
        page_bytes += ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT - (
            page_bytes % ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT
        )
    return page_bytes


def slice_deepseek_v4_proxy_cache_views(
    proxy_kv_cache: torch.Tensor,
    *,
    compress_ratios: list[int] | tuple[int, ...] | None = None,
    csa_layer_count: int | None = None,
    hca_layer_count: int | None = None,
    num_slots: int = 1,
    window_size: int = 128,
    head_dim: int = 512,
    index_head_dim: int = 128,
    kv_fp8: bool = False,
    rope_head_dim: int = 64,
    arena_planes=None,
    arena_rows: int = 0,
    row_widths: list[int] | None = None,
) -> dict[str, object]:
    """Carve native-equivalent unified V4 planes from vLLM proxy storage."""
    from atom.model_ops.attentions.pool_layout.state_arena import (
        SplitStateArena,
        StateArena,
        plan_regions,
    )
    from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
        UnifiedPoolGeometry,
    )

    if compress_ratios is None:
        assert csa_layer_count is not None and hca_layer_count is not None
        compress_ratios = [4] * csa_layer_count + [128] * hca_layer_count
    ratios = [int(r) for r in compress_ratios]
    index_dim = _index_row_bytes(index_head_dim)
    physical = proxy_kv_cache.permute(1, 0, 2, 3, 4)
    if not physical.is_contiguous():
        raise ValueError("DeepSeek V4 proxy cache must be block-major contiguous")
    num_blocks = int(physical.shape[0])
    raw = physical.reshape(-1)
    if raw.dtype is not torch.uint8:
        raise ValueError(f"DeepSeek V4 proxy cache must be uint8, got {raw.dtype}")
    # Packed vLLM KV allocations can start an individual layer on only a 128B
    # boundary. Consume the sizing slack above so every plane and embedded
    # StateArena starts on the 256B boundary its retyped field views require.
    alignment_pad = (-raw.storage_offset()) % ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT
    raw = raw[alignment_pad:]
    offset = 0
    geometry = UnifiedPoolGeometry(
        ratios,
        num_blocks=num_blocks,
        num_slots=num_slots,
        ring_slots=window_size,
        block_size=ATOM_DEEPSEEK_V4_BLOCK_SIZE,
        arena_rows=arena_rows,
    )
    csa_indexer: list[torch.Tensor] = []
    csa_rows = ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4
    n_csa = sum(1 for r in ratios if r == 4)

    nope_dtype = dtypes.fp8 if kv_fp8 else torch.bfloat16

    def take_bytes(n: int) -> torch.Tensor:
        nonlocal offset
        if offset + n > raw.numel():
            raise ValueError(
                f"DeepSeek V4 proxy cache too small: need {offset+n}, have {raw.numel()}"
            )
        out = raw[offset : offset + n]
        offset += n
        return out

    regions = _proxy_region_byte_sizes(
        geometry=geometry,
        csa_layers=n_csa,
        num_blocks=num_blocks,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        index_head_dim=index_head_dim,
        kv_fp8=kv_fp8,
    )
    region_offsets, layout_total = plan_regions(regions)

    def take_region(region_idx: int) -> torch.Tensor:
        nonlocal offset
        start = region_offsets[region_idx]
        offset = max(offset, start)
        return take_bytes(regions[region_idx])

    kv_plane = take_region(0).view(nope_dtype).view(geometry.plane_rows, head_dim)
    region_idx = 1
    kv_plane_rope = None
    if kv_fp8:
        kv_plane_rope = (
            take_region(region_idx)
            .view(torch.bfloat16)
            .view(geometry.plane_rows, rope_head_dim)
        )
        region_idx += 1
    if n_csa:
        per_layer_indexer = num_blocks * csa_rows * index_dim
        indexer_blob = take_region(region_idx)
        for layer_idx in range(n_csa):
            sl = indexer_blob[
                layer_idx * per_layer_indexer : (layer_idx + 1) * per_layer_indexer
            ]
            csa_indexer.append(
                sl.view(dtypes.fp8).view(num_blocks, csa_rows, index_dim)
            )
    offset = max(offset, layout_total)
    planes = [kv_plane]
    if kv_plane_rope is not None:
        planes.append(kv_plane_rope)
    if arena_planes is None or row_widths is None:
        arena = None
    else:
        arena = SplitStateArena(
            [
                StateArena(
                    fields,
                    geometry.slot_positions,
                    proxy_kv_cache.device,
                    buf=plane.view(torch.uint8).reshape(-1),
                    slot_stride=geometry.slot_rows * row_bytes,
                    live_entries=num_slots,
                )
                for fields, plane, row_bytes in zip(
                    arena_planes, planes, row_widths, strict=True
                )
            ]
        )

    def layer_views(plane):
        if plane is None:
            return [None] * len(ratios)
        return [plane[geometry.layer_base_row(i) :] for i in range(len(ratios))]

    unified = layer_views(kv_plane)
    unified_rope = layer_views(kv_plane_rope)

    def compress_view(plane, layer_id: int):
        cls = geometry.layer_class(layer_id)
        envelopes = plane[: num_blocks * geometry.envelope_rows].view(
            num_blocks, geometry.envelope_rows, plane.shape[1]
        )
        base = geometry.layer_base_row(layer_id)
        return envelopes[:, base : base + cls.block_rows]

    csa_main = [
        compress_view(kv_plane, i) for i, ratio in enumerate(ratios) if ratio == 4
    ]
    hca_main = [
        compress_view(kv_plane, i) for i, ratio in enumerate(ratios) if ratio == 128
    ]
    csa_main_rope = [
        compress_view(kv_plane_rope, i)
        for i, ratio in enumerate(ratios)
        if ratio == 4 and kv_plane_rope is not None
    ]
    hca_main_rope = [
        compress_view(kv_plane_rope, i)
        for i, ratio in enumerate(ratios)
        if ratio == 128 and kv_plane_rope is not None
    ]

    return {
        "geometry": geometry,
        "state_arena": arena,
        "unified": unified,
        "csa_main": csa_main,
        "csa_indexer": csa_indexer,
        "hca_main": hca_main,
        "unified_rope": unified_rope,
        "csa_main_rope": csa_main_rope,
        "hca_main_rope": hca_main_rope,
    }


def _is_runtime_dummy_decode(common_attn_metadata, req_ids, num_spec_tokens) -> bool:
    """Whether this build is a dummy decode batch that reached the ordinary path.

    An idle DP rank steps via ``execute_dummy_batch()``, whose batch is the same
    degenerate ``InputBatch.make_dummy`` a capture uses but arrives with
    ``capturing`` False -- so a capture-only repair skips it and the rank faults.

    Gated on ``req_ids == []`` ("no real requests") plus a decode shape, so
    prefill-shaped profiling dummies, whose context is coherent, are left alone.
    """
    if req_ids is None or len(req_ids) != 0:
        return False
    max_q = int(getattr(common_attn_metadata, "max_query_len", 0) or 0)
    return 0 < max_q <= 1 + int(num_spec_tokens)


def _synthesize_v4_decode_context(common_attn_metadata, meta_params) -> None:
    """Give a dummy decode batch a realistic decode context.

    ``InputBatch.make_dummy`` uses ``seq_len == query_len`` and zeroed
    ``positions``, i.e. decoding at ``start_pos == 0``. V4 decode is undefined
    there -- the SWA gather, the committed counts and the sparse top-k all
    derive from ``seq_len - query_len``, so the paged gather addresses outside
    the pool. Mirrors native's ``build_for_cudagraph_capture``.

    In-place value rewrite only: replays refill ``seq_lens``/``positions`` from
    the real batch and shapes stay as vLLM chose them.
    """
    if meta_params is None:
        return
    num_reqs = int(common_attn_metadata.num_reqs)
    q_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
    if num_reqs <= 0 or q_cpu is None:
        return
    q_np = q_cpu[: num_reqs + 1].numpy().astype(np.int32)
    lens = np.diff(q_np).astype(np.int32)
    total = int(lens.sum())
    # A prefill capture already carries a coherent context.
    if total <= 0 or lens.max() <= 0:
        return
    start_pos = int(meta_params.window_size)

    # context_len = start_pos + query_len, matching native's synthetic batch.
    # Zero-query CG-padding rows keep seq_len 0 so they stay inactive.
    seq_np = np.where(lens > 0, lens + start_pos, 0).astype(np.int32)
    seq_lens = common_attn_metadata.seq_lens
    seq_lens[:num_reqs].copy_(
        torch.from_numpy(seq_np).to(seq_lens.dtype), non_blocking=True
    )
    # Keep every host mirror in sync with the device tensor, or the metadata
    # describes a different batch than the kernels see.
    seq_cpu = torch.from_numpy(seq_np).to(torch.int32)
    ub = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
    if ub is not None:
        ub[:num_reqs].copy_(seq_cpu.to(ub.dtype))
    if getattr(common_attn_metadata, "_seq_lens_cpu", None) is not None:
        common_attn_metadata._seq_lens_cpu[:num_reqs].copy_(
            seq_cpu.to(common_attn_metadata._seq_lens_cpu.dtype)
        )
    else:
        common_attn_metadata._seq_lens_cpu = seq_cpu
    common_attn_metadata.max_seq_len = int(seq_np.max())

    # positions = start_pos + within-sequence offset.
    positions = getattr(common_attn_metadata, "positions", None)
    if positions is not None and positions.numel() >= total:
        batch_np = np.repeat(np.arange(num_reqs, dtype=np.int32), lens)
        within = np.arange(total, dtype=np.int32) - q_np[batch_np]
        pos_np = (within + start_pos).astype(np.int64)
        positions[:total].copy_(
            torch.from_numpy(pos_np).to(positions.dtype), non_blocking=True
        )


class AtomDeepseekV4ProxyMetadataBuilder(AttentionMetadataBuilder):
    # Decode is full-graph safe for uniform query batches, including speculative
    # decode where each request contributes 1 + num_speculative_tokens queries.
    # The per-fwd index/indptr/slot/compress-plan tensors are staged into
    # persistent fixed-address buffers here in build() (outside the captured
    # region), so replay re-reads the same addresses. Prefill/mixed batches stay
    # on the piecewise/eager path.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.device = device
        # Decodes get pulled to the front of the batch so vLLM can classify a
        # uniform-decode batch and dispatch the captured decode graph. The plugin
        # reads the (reordered) CommonAttentionMetadata transparently; the
        # per-request state slot is keyed on the req id, so it is invariant to
        # reordering.
        #
        # With speculative decoding (MTP) each request's verify step carries
        # ``1 + num_speculative_tokens`` query rows, so the decode threshold must
        # be raised accordingly -- otherwise the verify batch is misclassified as
        # prefill/mixed and the captured uniform-decode graph is fed mismatched
        # metadata (garbage outputs, and an illegal access once the V4 compressor
        # side-stream is captured). Use vLLM's own spec-as-decode computation
        # (``1 + num_speculative_tokens``, doubled for parallel drafting) so this
        # tracks the upstream contract for spec-decode-capable backends.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        # Number of MTP draft tokens per step (0 when spec decode is off). The
        # spec-verify batch carries ``1 + num_spec_tokens`` uniform query rows;
        # the metadata builder must classify that as DECODE (not PREFILL) so it
        # stages into the persistent fixed-address decode buffers the captured
        # FULL graph replays against. Kept consistent with the reorder threshold.
        self._num_spec_tokens = _v4_spec_steps(vllm_config)
        # CUDA/HIP-graph capture token-bucket sizes. vLLM pads the decode model
        # forward (``positions``/hidden states) up to one of these even under
        # PIECEWISE, where it leaves the attention-metadata token count
        # unpadded. The decode metadata token-pad is rounded up to the same
        # bucket so the ``batch_id == -1`` sentinel tail covers the padded rows
        # the fused decode ``qk_norm_rope`` iterates over.
        cc = getattr(vllm_config, "compilation_config", None)
        self._cg_token_sizes = sorted(
            {int(s) for s in (getattr(cc, "cudagraph_capture_sizes", None) or [])}
        )

    def build(
        self, common_prefix_len: int, common_attn_metadata, fast_build: bool = False
    ):
        if common_prefix_len:
            raise ValueError(
                "ATOM DeepSeek V4 proxy does not support cascade attention"
            )
        return self._build_and_attach_atom_v4_md(common_attn_metadata, capturing=False)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        # vLLM builds the metadata for a synthetic uniform-decode batch here,
        # OUTSIDE the captured region, then captures the model forward. Stage
        # into the persistent decode buffers with arange slots (the dummy
        # batch's NULL block ids must not pollute the real slot allocator).
        return self._build_and_attach_atom_v4_md(common_attn_metadata, capturing=True)

    def _build_and_attach_atom_v4_md(self, common_attn_metadata, *, capturing):
        """Build the ATOM V4 attention metadata OUTSIDE the captured graph and
        attach it to the vLLM ``CommonAttentionMetadata`` the model forward reads.

        vLLM calls ``builder.build()`` / ``build_for_cudagraph_capture()`` once
        per step, before ``set_forward_context`` + the (possibly
        CUDA/HIP-graph-wrapped) model forward. Building here -- rather than
        inside the forward -- is what makes a captured decode graph correct:
        for decode this refreshes the per-fwd index/indptr/slot/compress-plan
        tensors *in place* in persistent fixed-address buffers (allocated at
        cache-bind time), so the captured kernels replay against stable
        addresses. The per-request selective state reset also runs here
        (outside any capture). Prefill stays on the eager fresh-tensor path and
        is never captured.

        Returns the same ``common_attn_metadata`` (now carrying ``atom_v4_md``)
        so it flows through vLLM's per-layer attn-metadata dict to the forward,
        which consumes the prebuilt metadata instead of rebuilding it.
        """
        if common_attn_metadata is None:
            return common_attn_metadata
        sfc = self.vllm_config.compilation_config.static_forward_context
        proxy_layer_name = self.layer_names[0]
        proxy = sfc.get(proxy_layer_name)
        model = getattr(proxy, "_atom_v4_model", None) if proxy is not None else None
        meta_params = getattr(model, "_atom_v4_meta_params", None)
        if model is None or meta_params is None:
            # Pre-bind (profiling / first warmup forward, before the proxy cache is
            # bound): leave common untouched. The forward detects the missing
            # atom_v4_md and falls back to an inline eager build (force_dummy).
            _warn_once(
                "ATOM V4: metadata builder ran unbound (layer=%r, proxy=%s, "
                "proxy_kv_numel=%s, profiling_cache=%s, model=%s, "
                "meta_params=%s, capturing=%s)",
                proxy_layer_name,
                type(proxy).__name__ if proxy is not None else None,
                getattr(getattr(proxy, "kv_cache", None), "numel", lambda: None)(),
                getattr(proxy, "_atom_v4_profiling_kv_cache", None),
                model is not None,
                meta_params is not None,
                capturing,
            )
            return common_attn_metadata
        # Batch-ordered req_ids exposed by the ATOM vLLM patch for this step;
        # used as the host-resident state-slot key (no block-table D2H). None
        # when the patch isn't applied (standalone/tests) -> build falls back.
        req_ids = None
        if not capturing:
            try:
                from atom.plugin.vllm.req_id_passthrough_patch import (
                    get_current_req_ids,
                )

                req_ids = get_current_req_ids()
            except Exception:  # noqa: BLE001 - passthrough patch is optional
                req_ids = None
        if capturing or _is_runtime_dummy_decode(
            common_attn_metadata, req_ids, self._num_spec_tokens
        ):
            _synthesize_v4_decode_context(common_attn_metadata, meta_params)
        slot_allocator = (
            None if capturing else getattr(model, "_atom_v4_slot_allocator", None)
        )
        decode_bufs = getattr(model, "_atom_v4_decode_bufs", None)
        md = build_atom_v4_attention_metadata(
            common_attn_metadata,
            meta_params=meta_params,
            slot_allocator=slot_allocator,
            decode_bufs=decode_bufs,
            capturing=capturing,
            req_ids=req_ids,
            num_spec_tokens=self._num_spec_tokens,
            cudagraph_token_sizes=self._cg_token_sizes,
        )
        # Native ATOM enables V4 compressor side-stream launches only while the
        # forward is being captured into a HIP/CUDA graph. vLLM builds this metadata
        # on the capture path, so carry the signal into ATOM's forward context.
        md.in_hipgraph = bool(capturing)
        # Selective per-slot reset OUTSIDE the captured region. For decode this
        # is empty (no fresh slots are bound mid-generation); it fires for the
        # prefill chunk that first allocates a request's slot, which is eager.
        reset_slots = getattr(md, "reset_slots", None)
        if reset_slots:
            reset_deepseek_v4_state_slots(model, reset_slots)
        common_attn_metadata.atom_v4_md = md
        return common_attn_metadata


class AtomDeepseekV4ProxyBackend(AttentionBackend):
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "ATOM_DEEPSEEK_V4_PROXY"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [ATOM_DEEPSEEK_V4_BLOCK_SIZE]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return ATOM_DEEPSEEK_V4_BLOCK_SIZE

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (
            (1, 0, 2, 3, 4) if not include_num_layers_dimension else (1, 0, 2, 3, 4, 5)
        )

    @staticmethod
    def get_impl_cls():
        return nn.Identity

    @staticmethod
    def get_builder_cls():
        return AtomDeepseekV4ProxyMetadataBuilder

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AtomDeepseekV4ProxyAttention(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str = ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME):
        super().__init__()
        self.prefix = prefix
        self._atom_v4_proxy_layer = True
        self._atom_v4_profiling_kv_cache = False
        self.kv_cache = torch.tensor([])
        self.impl = nn.Identity()

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AtomDeepseekV4ProxyBackend

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        page_bytes = _proxy_page_bytes(vllm_config)
        head_size = (page_bytes + 2 * ATOM_DEEPSEEK_V4_BLOCK_SIZE - 1) // (
            2 * ATOM_DEEPSEEK_V4_BLOCK_SIZE
        )
        return FullAttentionSpec(
            block_size=ATOM_DEEPSEEK_V4_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.uint8,
        )


def register_deepseek_v4_proxy_layer(
    vllm_config,
    layer_name: str = ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME,
) -> AtomDeepseekV4ProxyAttention:
    from atom.plugin.vllm.deepseek_v4_profiling_patch import (
        apply_vllm_v4_profile_cache_patch,
    )

    apply_vllm_v4_profile_cache_patch()
    sfc = vllm_config.compilation_config.static_forward_context
    existing = sfc.get(layer_name)
    if isinstance(existing, AtomDeepseekV4ProxyAttention):
        return existing
    if existing is not None:
        raise ValueError(f"Duplicate layer name: {layer_name}")
    proxy = AtomDeepseekV4ProxyAttention(prefix=layer_name)
    sfc[layer_name] = proxy
    return proxy


def _bind_compressor_state(
    compressor,
    kv_cache: torch.Tensor,
    head_dim: int,
    *,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    is_indexer: bool = False,
    write_mode: str = "bf16",
    kv_cache_rope: torch.Tensor | None = None,
) -> None:
    compressor.kv_state = kv_state
    compressor.score_state = score_state
    compressor.kv_cache = kv_cache
    if is_indexer:
        nb, csa_rows_per_block, index_row_bytes = kv_cache.shape
        block_fp32_stride = (csa_rows_per_block * index_row_bytes) // 4
        scale_fp32_offset = (csa_rows_per_block * head_dim) // 4
        # storage_offset is ABSOLUTE and `kv_cache` is a view into vLLM's proxy
        # pool, so its own offset has to be added or every layer's cache_scale
        # aliases the first layer's. Mirrors `deepseek_v4_attn.py`.
        flat_f32 = kv_cache.view(torch.float32).view(-1)
        compressor.cache_scale = flat_f32.as_strided(
            size=(nb, csa_rows_per_block),
            stride=(block_fp32_stride, 1),
            storage_offset=flat_f32.storage_offset() + scale_fp32_offset,
        )
    else:
        compressor.cache_scale = None
    # Compressor.forward selects its scatter path from `quant_mode`. Keep the
    # legacy write_mode attribute too for integrations that still consume it.
    # The indexer-inner cache is per-row fp8; CSA/HCA Main is unquantized or,
    # under the fp8 2buff layout, group-fp8 with a parallel bf16 RoPE pool.
    compressor.kv_cache_rope = kv_cache_rope
    compressor.write_mode = write_mode
    if is_indexer:
        compressor.quant_mode = "per_row_fp8"
    else:
        compressor.quant_mode = "group_fp8" if kv_cache_rope is not None else "none"


def _v4_max_spec_steps(vllm_config) -> int:
    """Speculative draft length per decode step (0 when spec decode is off).

    Sets the decode CG bucket: the per-fwd token count for a uniform decode
    batch of ``bs`` requests is ``bs * (1 + max_spec_steps)``.
    """
    return _v4_spec_steps(vllm_config)


def _deepseek_v4_blocks(model):
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is not None:
        return layers
    mtp = getattr(inner, "mtp", None)
    if mtp is not None:
        return mtp
    return []


def _compressed_layer_cache_index(ratios, layer_id: int, ratio: int) -> int:
    return sum(1 for r in ratios[:layer_id] if int(r) == int(ratio))


def _v4_padded_token_count(common_attn_metadata, total: int) -> int:
    num_actual = int(getattr(common_attn_metadata, "num_actual_tokens", total) or total)
    slot_mapping = getattr(common_attn_metadata, "slot_mapping", None)
    slot_tokens = (
        int(slot_mapping.numel()) if isinstance(slot_mapping, torch.Tensor) else 0
    )
    return max(total, num_actual, slot_tokens)


def _v4_round_to_cudagraph_bucket(n: int, sizes) -> int:
    """Round ``n`` up to the smallest CUDA/HIP-graph capture size >= ``n``.

    vLLM only pads the *attention metadata* token count to a capture-size
    bucket when the decode dispatches the FULL graph (``pad_attn`` is
    ``cudagraph_mode == FULL``). Under PIECEWISE, the metadata is left at the
    real (unpadded) token count, yet the model forward still receives
    ``positions``/hidden states padded to a piecewise capture-size bucket
    (fixed shapes are required to replay the captured piecewise regions). The
    fused decode ``qk_norm_rope`` reads ``T = positions.shape[0]`` (the padded
    forward width) and asserts ``len(batch_id_per_token) >= T``; if the V4
    decode metadata is sized to the unpadded count the padded tail tokens read
    ``batch_id_per_token`` out of bounds (illegal access / launch failure under
    load). Rounding the decode token-pad count up to the same bucket vLLM pads
    the forward to keeps the ``batch_id == -1`` sentinel tail long enough to
    cover those padded rows. Batches larger than the max capture size run eager
    (no padding), so ``n`` passes through unchanged.
    """
    if not sizes:
        return n
    if n > sizes[-1]:
        return n
    for s in sizes:
        if s >= n:
            return s
    return n


class _V4DecodeMetaBuffers:
    """Persistent, fixed-address scratch for the V4 *decode* attention metadata.

    A captured decode HIP/CUDA graph re-runs recorded kernels that read these
    tensors by address; the per-step ``build()`` refreshes their *contents* in
    place (numpy -> ``copy_to_gpu`` / ``write_v4_paged_decode_indices``) before
    replay -- mirroring native ATOM's ``forward_vars`` decode buffers. Sized
    once to the worst-case decode shape (``num_slots`` seqs,
    ``num_slots * (1 + max_spec_steps)`` tokens, ``max_committed_hca`` HCA
    entries per seq for the native ``max_model_len``). Index/indptr views are
    always sliced from the buffer base so their data pointer is stable across
    builds even as the logical length changes. Prefill never touches these.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        max_decode_tokens: int,
        window: int,
        index_topk: int,
        max_committed_hca: int,
        ratios_overlap,
        device: torch.device,
        max_blocks: int = 1,
    ):
        from atom.utils import CpuGpuBuffer

        self.device = device
        self.window = int(window)
        S = max(1, int(num_slots))
        T = max(1, int(max_decode_tokens))
        win = int(window)
        topk = int(index_topk)
        hca = int(max_committed_hca)
        self.num_slots = S
        self.max_decode_tokens = T
        self.max_blocks = max(1, int(max_blocks))

        def i32(*shape):
            return CpuGpuBuffer(*shape, dtype=torch.int32, device=device)

        # Per-seq scalars (sized to padded request count == num_slots).
        # Core DeepSeek-V4 metadata distinguishes the state slot read by the
        # compressor from the slot written by this forward. vLLM does not fork
        # state today, so both buffers carry the same values, but they must have
        # independent addresses for CUDA/HIP graph replay.
        self.state_slot_in = i32(S)
        self.state_slot_out = i32(S)
        self.n_csa = i32(S)
        # Per-token mapping (sized to padded token count). int32: accepted by
        # torch advanced-indexing AND by the fused flydsl SWA scatter (which
        # loads batch_id as int32); matches the in-tree model_runner path.
        self.batch_id = CpuGpuBuffer(T, dtype=torch.int32, device=device)
        self.swa_dest_rows = {ratio: i32(T) for ratio in _V4_SWA_DEST_RATIOS}
        self.csa_n_committed_per_token = i32(T)
        self.block_tables_per_token = i32(T, self.max_blocks)
        # Ragged cumsums (T + 1) and ragged index pools (worst-case per-token
        # slot counts): SWA = win, CSA = win + index_topk, HCA = win + hca.
        self.indptr_swa = i32(T + 1)
        self.indptr_csa = i32(T + 1)
        self.indptr_hca = i32(T + 1)
        self.idx_swa = i32(T * max(1, win))
        self.idx_csa = i32(T * max(1, win + topk))
        self.idx_hca = i32(T * max(1, win + hca))
        # Per-token paged-decode index tensors for the fp8 asm decode kernel
        # (aiter op5 `mla_decode_fwd_v4_nm`, page_size=1). Mirrors native
        # deepseek_v4_attn.py: values depend ONLY on the (padded) decode token
        # count, never the batch content -- qo_indptr == arange(N+1) (with the
        # CG-padded tail repeating the real token count so padded slots are
        # 0-length queries the asm kernel skips), kv_last_page_lens == ones(N).
        # Re-staged into these persistent buffers every fwd (H2D into a stable
        # address) => CUDAGraph-safe. Allocated unconditionally (tiny); only
        # populated on the fp8 path. bf16 leaves md.qo_indptr / kv_last_page_lens
        # None (the native Triton decode path ignores them).
        self.qo_indptr = i32(T + 1)
        self.kv_last_page_lens = i32(T)
        self._qo_indptr_np = np.arange(T + 1, dtype=np.int32)
        # Native compress-plan buffers (one pair per compress ratio present).
        # Decode worst case: each seq contributes ceil((1 + spec) / ratio)
        # compression boundaries. The write plan is a subset of the per-fwd
        # ragged tokens (a token is written iff its position falls in the per-seq
        # "last K_pool" window), so for decode it has at most `total` rows
        # (<= T == max_decode_tokens). Sizing the write buffer to T instead of
        # the prefill-style S*K_pool worst case keeps the per-step sentinel fill,
        # the H2D copy, AND the write-kernel grid (== write_plan.shape[0]) bounded
        # to the decode token count -- the prior S*K_pool sizing filled/copied an
        # almost-entirely-sentinel buffer every decode step (up to 128x for the
        # HCA ratio). CUDAGraph-safe: shape[0]==T is fixed across capture/replay.
        from atom.model_ops.v4_kernels.compress_plan import (  # noqa: F401
            make_compress_plans as _mcp,
        )

        # Decode CG plan slicing is `running_bs * per_seq_bound` (computed inside
        # make_compress_plans from these two scalars). running_bs == num_slots
        # (padded decode batch); max_q_len == 1 + max_spec_steps.
        self.decode_running_bs = S
        self.decode_q_len = max(1, T // S)
        # Completion of the previous build's staging H2Ds; see
        # `gate_staging_reuse`.
        self._h2d_done = torch.cuda.Event()
        self.plan_buffers: dict[int, dict] = {}
        for ratio, is_overlap in ratios_overlap:
            ratio = int(ratio)
            per_seq = (self.decode_q_len + ratio - 1) // ratio
            cap = max(1, S * per_seq)
            self.plan_buffers[ratio] = {
                "compress": i32(cap, 4),
                "write": i32(max(1, T), 4),
            }

    def gate_staging_reuse(self):
        """Block until the previous build's staging H2Ds have executed.

        ``stage`` copies non-blocking out of one pinned host buffer per name.
        Stream ordering protects the GPU side, not the host side, so the next
        build's ``buf.np[:n] = arr`` races the previous DMA. Async scheduling
        removes the sampled-token sync that used to drain it, and a torn
        ``kv_indptr`` faults the decode kernel.

        Depth-1, as in native's ``model_runner._gate_staging_reuse``: one host
        buffer admits one build of lead. Skipped while capturing -- events are
        illegal in a capture region and there is no CPU run-ahead to guard.
        """
        if not torch.cuda.is_current_stream_capturing():
            self._h2d_done.synchronize()

    def mark_h2d_enqueued(self):
        """Close the window ``gate_staging_reuse`` waits on. Called once per
        build, after the last staging copy, so one event covers them all."""
        if not torch.cuda.is_current_stream_capturing():
            self._h2d_done.record()

    def stage(self, buf, arr_np):
        """Copy ``arr_np`` into the head of CpuGpuBuffer ``buf`` and return the
        from-base GPU view (stable data pointer)."""
        n = int(arr_np.shape[0]) if getattr(arr_np, "ndim", 1) else 1
        assert (
            n <= buf.np.shape[0]
        ), f"V4 decode buffer too small: need {n}, have {buf.np.shape[0]}"
        if n:
            buf.np[:n] = arr_np
        return buf.copy_to_gpu(n)


def bind_deepseek_v4_proxy_cache_views(
    model,
    vllm_config,
    layer_name: str = ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME,
) -> bool:
    sfc = vllm_config.compilation_config.static_forward_context
    proxy = sfc.get(layer_name)
    if proxy is None or not isinstance(proxy, AtomDeepseekV4ProxyAttention):
        return False
    if not isinstance(proxy.kv_cache, torch.Tensor) or proxy.kv_cache.numel() == 0:
        return False
    if proxy._atom_v4_profiling_kv_cache:
        return False
    ptr = proxy.kv_cache.untyped_storage().data_ptr()
    if getattr(model, "_atom_vllm_v4_proxy_cache_ptr", None) == ptr:
        return True
    ratios = [int(r) for r in model.args.compress_ratios]
    num_slots = max(1, int(vllm_config.scheduler_config.max_num_seqs))
    # Stash the per-request state-slot allocator and the geometry parameters
    # the bridge cannot read from common_attn_metadata.
    if not hasattr(model, "_atom_v4_slot_allocator"):
        model._atom_v4_slot_allocator = _V4StateSlotAllocator(num_slots)
    window_size = int(model.args.window_size)
    win_with_spec = _v4_win_with_spec(vllm_config, window_size)
    # Single fp8 authority for the whole bind (must agree with the _proxy_page_bytes
    # sizing vLLM already used to allocate proxy.kv_cache). Threaded onto
    # meta_params so the metadata builder stages the fp8 op5 per-token decode
    # index tensors (qo_indptr / kv_last_page_lens) only under fp8.
    kv_fp8 = _v4_kv_fp8(vllm_config)
    arena_planes, arena_rows, row_widths = _v4_state_layout(vllm_config, kv_fp8)
    rope_head_dim = _v4_rope_head_dim(vllm_config.model_config.hf_config)
    # CSA/HCA Main scatter mode: fp8 2buff -> "main_2buff_fp8" (nope fp8 + parallel
    # bf16 rope), else the plain bf16 scatter. The indexer is always "indexer_fp8".
    main_write_mode = "main_2buff_fp8" if kv_fp8 else "bf16"
    physical = proxy.kv_cache.permute(1, 0, 2, 3, 4)
    if not physical.is_contiguous():
        raise ValueError("DeepSeek V4 proxy cache must be block-major contiguous")
    raw = physical.reshape(-1)
    if raw.storage_offset() % 256:
        raise RuntimeError(
            f"DeepSeek V4 proxy KV storage offset {raw.storage_offset()} is not "
            "256B-aligned; StateArena cannot retype carved planes safely"
        )
    views = slice_deepseek_v4_proxy_cache_views(
        proxy.kv_cache,
        compress_ratios=ratios,
        num_slots=num_slots,
        window_size=win_with_spec,
        head_dim=int(model.args.head_dim),
        index_head_dim=int(model.args.index_head_dim),
        kv_fp8=kv_fp8,
        rope_head_dim=rope_head_dim,
        arena_planes=arena_planes,
        arena_rows=arena_rows,
        row_widths=row_widths,
    )
    geometry = views["geometry"]
    model._atom_v4_geometry = geometry
    arena = views["state_arena"]
    model._atom_v4_meta_params = SimpleNamespace(
        num_slots=num_slots,
        window_size=window_size,
        cs=win_with_spec,
        index_topk=int(getattr(model.args, "index_topk", 1024)),
        kv_fp8=kv_fp8,
        geometry=geometry,
    )
    for fallback_layer_id, block in enumerate(_deepseek_v4_blocks(model)):
        attn = block.attn
        layer_id = int(getattr(attn, "layer_id", fallback_layer_id))
        ratio = int(attn.compress_ratio)
        attn.unified_kv = views["unified"][layer_id]
        attn.swa_kv = attn.unified_kv
        attn.swa_plane = attn.unified_kv
        attn.swa_window = geometry.window_params(ratio)
        attn.swa_cache_size = int(win_with_spec)
        # #1600 contract: DeepseekV4Attention.forward reads unified_kv_rope /
        # swa_kv_rope unconditionally (the parallel bf16 rope pool of the fp8
        # 2buff layout). bf16 -> both None (RoPE stays inline in unified_kv,
        # matching the native builder's bf16 branch). fp8 2buff -> the parallel
        # bf16 RoPE pool, paged identically to unified_kv / swa_kv. kv_fp8 is
        # (re)asserted here as the bind-time authority so the module geometry
        # agrees with the proxy pool (incl. the gfx950/1250 -> bf16 fallback).
        attn.kv_fp8 = kv_fp8
        if kv_fp8:
            attn.unified_kv_rope = views["unified_rope"][layer_id]
            attn.swa_kv_rope = attn.unified_kv_rope
            attn.swa_plane_rope = attn.unified_kv_rope
        else:
            attn.unified_kv_rope = None
            attn.swa_kv_rope = None
            attn.swa_plane_rope = None
        if ratio == 4:
            csa_i = _compressed_layer_cache_index(ratios, layer_id, ratio)
            _bind_compressor_state(
                attn.compressor,
                views["csa_main"][csa_i],
                int(model.args.head_dim),
                kv_state=arena.view("csa_main_kv")[csa_i],
                score_state=arena.view("csa_main_score")[csa_i],
                write_mode=main_write_mode,
                kv_cache_rope=views["csa_main_rope"][csa_i],
            )
            attn.indexer.kv_cache = views["csa_indexer"][csa_i]
            attn.indexer._max_model_len_idx = max(
                1, int(vllm_config.model_config.max_model_len) // 4
            )
            _bind_compressor_state(
                attn.indexer.compressor,
                views["csa_indexer"][csa_i],
                int(model.args.index_head_dim),
                kv_state=arena.view("csa_idx_kv")[csa_i],
                score_state=arena.view("csa_idx_score")[csa_i],
                is_indexer=True,
                write_mode="indexer_fp8",
            )
        elif ratio == 128:
            hca_i = _compressed_layer_cache_index(ratios, layer_id, ratio)
            _bind_compressor_state(
                attn.compressor,
                views["hca_main"][hca_i],
                int(model.args.head_dim),
                kv_state=arena.view("hca_main_kv")[hca_i],
                score_state=arena.view("hca_main_score")[hca_i],
                write_mode=main_write_mode,
                kv_cache_rope=views["hca_main_rope"][hca_i],
            )
    # Persistent decode-metadata buffers for the FULL decode CUDA/HIP graph.
    # Allocated once (sized to the worst-case decode shape) so build() can
    # refresh them in place each step; the captured kernels read stable
    # addresses. Eager prefill never touches them. Stash the model on the proxy
    # so the metadata builder (which only sees vllm_config) can reach it.
    proxy._atom_v4_model = model
    if not hasattr(model, "_atom_v4_decode_bufs"):
        max_spec = _v4_max_spec_steps(vllm_config)
        max_model_len = int(vllm_config.model_config.max_model_len)
        max_committed_hca = max(1, (max_model_len + 127) // 128)
        # CSA (ratio 4) compress windows overlap; HCA (ratio 128) does not.
        ratios_overlap = [(r, r == 4) for r in sorted(set(ratios)) if r > 0]
        model._atom_v4_decode_bufs = _V4DecodeMetaBuffers(
            num_slots=num_slots,
            max_decode_tokens=num_slots * (1 + max_spec),
            window=int(model.args.window_size),
            index_topk=int(getattr(model.args, "index_topk", 1024)),
            max_committed_hca=max_committed_hca,
            ratios_overlap=ratios_overlap,
            device=proxy.kv_cache.device,
            max_blocks=max_committed_hca,
        )
    model._atom_vllm_v4_proxy_cache_ptr = ptr
    return True


def reset_deepseek_v4_state_slots(model, slots) -> None:
    """Clear V4 per-request SWA + compressor state for specific state slots.

    Chunk-aware analogue of `reset_deepseek_v4_state_caches`: rather than wiping
    every slot whenever a batch happens to start at position 0, reset only the
    slots the allocator just (re)assigned to a fresh request. This preserves a
    long prompt's accumulated SWA window and compressor state across its prefill
    chunks while still guaranteeing a brand-new request (or a slot left dirty by
    a finished request or a profiling forward) starts from clean state.
    """
    if not slots:
        return
    layers = getattr(getattr(model, "model", None), "layers", [])
    if not layers:
        return
    device = None
    for block in layers:
        swa = getattr(getattr(block, "attn", None), "swa_kv", None)
        if isinstance(swa, torch.Tensor):
            device = swa.device
            break
    if device is None:
        return
    physical_slots = sorted(int(s) for s in slots)
    idx = torch.as_tensor(physical_slots, dtype=torch.long, device=device)
    for block in layers:
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        swa = getattr(attn, "swa_plane", None)
        window = getattr(attn, "swa_window", None)
        if isinstance(swa, torch.Tensor) and window is not None:
            rows = [
                window.index(slot, pos)
                for slot in physical_slots
                for pos in range(window.ring_slots)
            ]
            row_idx = torch.as_tensor(rows, dtype=torch.long, device=device)
            swa[row_idx] = 0
            swa_rope = getattr(attn, "swa_plane_rope", None)
            if isinstance(swa_rope, torch.Tensor):
                swa_rope[row_idx] = 0
        for compressor in (
            getattr(attn, "compressor", None),
            getattr(getattr(attn, "indexer", None), "compressor", None),
        ):
            if compressor is None:
                continue
            if isinstance(getattr(compressor, "kv_state", None), torch.Tensor):
                compressor.kv_state[idx] = 0
            if isinstance(getattr(compressor, "score_state", None), torch.Tensor):
                compressor.score_state[idx] = float("-inf")


def _infer_atom_attn_state(common_attn_metadata, num_spec_tokens: int = 0):
    """Classify the batch as DECODE / PREFILL_PREFIX / PREFILL_NATIVE.

    A speculative-decode verify (MTP) step contributes a *uniform*
    ``1 + num_speculative_tokens`` query rows per request, so its
    ``max_query_len`` is ``1 + num_spec_tokens`` (== 2 for the default MTP
    k=1), NOT 1. vLLM batches exactly these query lengths as a uniform decode
    (its ``reorder_batch_threshold`` is ``1 + num_speculative_tokens`` for
    spec-as-decode backends) and dispatches the captured FULL decode graph for
    them. ATOM must agree: a batch whose longest query fits inside the spec
    decode block is DECODE, so the build stages the per-fwd metadata into the
    persistent fixed-address decode buffers that the captured graph replays
    against. The old ``max_query_len == 1`` test classified the verify batch as
    PREFILL, which builds fresh per-step tensors at new addresses; the captured
    decode graph then replays against stale addresses -> garbage outputs under
    cudagraph (eager is unaffected because nothing is captured). Mirrors native
    ATOM, where the verify step is ``AttnState.DECODE`` with
    ``max_seqlen_q == num_spec_step + 1`` (deepseek_v4_attn.py:prepare_decode).

    ``max_query_len`` alone is NOT sufficient, though. vLLM's batch reorder
    (``reorder_batch_to_split_decodes_and_prefills``) groups by *scheduled token
    count*, so a request whose current step has ``<= 1 + num_spec`` query tokens
    but is still PREFILLING -- a fresh <=decode_q-token prompt (``num_computed ==
    0``) or a chunked-prefill tail whose remaining tokens are ``<= decode_q``
    (a "short extend") -- can sit in the same batch as real decodes. If the
    batch has no longer prefill row to push ``max_query_len`` past ``decode_q``,
    the ``max_q``-only test would misclassify it as DECODE and feed those
    prefill rows through the fixed-shape paged decode path: ``_score_topk_decode``
    reshapes ``[bs, next_n]`` assuming ONE uniform decode length (a ragged
    ``[4,4,4,1]`` breaks the view) and a fresh row (no committed K) needs the
    prefill index build, not the decode one. So gate DECODE on vLLM's own
    uniform-decode predicate plus a prefill guard (see ``_is_pure_uniform_decode``),
    using CPU-resident metadata only (NO H2D/D2H sync). A non-uniform or
    prefill-containing batch cannot be a FULL-captured decode anyway (capture is
    gated on ``batch_descriptor.uniform``), so routing it to PREFILL -- where
    ``_populate_indexer``'s leading-uniform-decode split already peels the decode
    prefix onto the paged path and folds the rest onto the dense path -- is both
    correct and capture-safe.
    """
    from atom.utils.forward_context import AttnState

    if common_attn_metadata is None:
        return AttnState.PREFILL_NATIVE
    decode_q = 1 + max(0, int(num_spec_tokens))
    if _is_pure_uniform_decode(common_attn_metadata, decode_q):
        return AttnState.DECODE
    num_computed = getattr(common_attn_metadata, "_num_computed_tokens_cpu", None)
    if num_computed is not None and bool((num_computed > 0).any().item()):
        return AttnState.PREFILL_PREFIX
    return AttnState.PREFILL_NATIVE


def _is_pure_uniform_decode(common_attn_metadata, decode_q: int) -> bool:
    """True iff the batch is the exact uniform decode vLLM captures a graph for.

    Reuses vLLM'sown uniform-decode definition (``GPUModelRunner._is_uniform_decode``):
    ``max_query_len == decode_q`` and ``num_actual_tokens == decode_q * num_reqs``
    -- pure scalars already on ``common_attn_metadata`` (padded decode rows are
    also ``decode_q`` long, so the product identity still holds). That aligns our
    DECODE state exactly with when vLLM dispatches its captured FULL decode graph.
    Since the predicate is purely shape-based, it is paired with ``is_prefilling``
    (a CPU bool ``num_computed < num_prompt``, padded rows zeroed) to reject a
    fresh ``decode_q``-length prompt / extend that is uniform yet not a real
    decode (no committed K -> needs the prefill index build, not paged decode).
    """
    num_reqs = int(getattr(common_attn_metadata, "num_reqs", 0) or 0)
    if num_reqs <= 0:
        return False
    is_pref = getattr(common_attn_metadata, "is_prefilling", None)
    if is_pref is not None and bool(is_pref[:num_reqs].any()):
        return False
    max_q = int(getattr(common_attn_metadata, "max_query_len", 0) or 0)
    num_tokens = int(getattr(common_attn_metadata, "num_actual_tokens", 0) or 0)
    return max_q == decode_q and num_tokens == decode_q * num_reqs


def _counts_to_indptr(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(counts) + 1, dtype=np.int32)
    out[1:] = np.cumsum(counts, dtype=np.int32)
    return out


def _make_compress_plans(
    extend_lens_cpu, context_lens_cpu, ratios, device, decode: bool
):
    total = int(extend_lens_cpu.sum())
    from atom.model_ops.v4_kernels import make_compress_plans
    from atom.utils import CpuGpuBuffer

    capacity = max(1, total)
    plan_buffers = {
        int(ratio): {
            "compress": CpuGpuBuffer(capacity, 4, dtype=torch.int32, device=device),
            "write": CpuGpuBuffer(capacity, 4, dtype=torch.int32, device=device),
        }
        for ratio, _ in ratios
    }
    plans = make_compress_plans(
        extend_lens_cpu,
        context_lens_cpu,
        ratios,
        plan_buffers=plan_buffers,
    )
    # Eager path (running_bs unset): make_compress_plans returns a full-buffer
    # write slice (sentinel-padded). The eager bridge launches
    # update_compressor_states with exactly num_write rows, so re-slice down.
    # Graph decode uses _make_decode_compress_plans (fixed running_bs slice).
    for plan in plans.values():
        plan.write_plan_gpu = plan.write_plan_gpu[: plan.num_write]
    return plans


class _V4StateSlotAllocator:
    """Stable per-request state-slot allocator over ``[0, num_slots)``.

    Keyed by each request's id (``req_id``), the canonical, host-resident
    request identity from vLLM's ``InputBatch``. This hands back the same state
    slot for every chunked-prefill step and every decode step of a request, so
    its SWA ring and compressor state accumulate in one place -- matching native
    ATOM's per-request cache slots.

    Keying on ``req_id`` (rather than the first KV block id, which lived on the
    GPU block table) removes the per-step D2H copy + host<->device sync that the
    block-id key required, and is immune to vLLM recycling a finished request's
    blocks to a new request within the same step.

    A slot is reported as freshly allocated (caller resets it) when it is newly
    bound to an unseen ``req_id``, or when a known ``req_id`` reappears with
    ``num_computed == 0`` -- vLLM recomputes preempted requests from scratch
    under the same id, so the slot's accumulated state must be cleared on resume.

    Slots are reclaimed lazily on exhaustion by evicting the least-recently-seen
    slot whose ``req_id`` is absent from the current step (its request finished
    or was preempted). vLLM caps concurrency at ``num_slots`` (max_num_seqs), so
    a request that is live this step never has its slot evicted.
    """

    def __init__(self, num_slots: int):
        self.num_slots = max(1, int(num_slots))
        self._key_to_slot: dict[object, int] = {}
        self._slot_to_key: list[object] = [None] * self.num_slots
        self._free: list[int] = list(range(self.num_slots - 1, -1, -1))
        self._last_seen: list[int] = [-1] * self.num_slots
        self._step = 0

    def assign(self, req_keys, num_computed):
        """Return ``(slots: np.int32[num_reqs], reset_slots: set[int])``.

        ``req_keys`` is a per-request sequence of stable, hashable keys (the
        ``req_id`` strings), aligned with the batch rows.
        """
        self._step += 1
        # Pull num_computed to a Python list in one C call (per-element
        # numpy-scalar -> int was the dominant cost of this per-decode-step
        # loop). req_keys is already a host-side list[str]. Local-bind the
        # dict/list fields too -- attribute lookups inside the bs-length loop
        # add up at large batch (profiled #1 build cost).
        keys = list(req_keys)
        nc = (
            num_computed.tolist()
            if hasattr(num_computed, "tolist")
            else list(num_computed)
        )
        n = len(keys)
        active = set(keys)
        key_to_slot = self._key_to_slot
        slot_to_key = self._slot_to_key
        last_seen = self._last_seen
        step = self._step
        slots = [0] * n
        reset: set[int] = set()
        for i in range(n):
            k = keys[i]
            slot = key_to_slot.get(k)
            if slot is None:
                slot = self._acquire(active)
                key_to_slot[k] = slot
                slot_to_key[slot] = k
                reset.add(slot)
            elif nc[i] == 0:
                # Known request recomputed from scratch (preemption resume).
                reset.add(slot)
            slots[i] = slot
            last_seen[slot] = step
        return np.asarray(slots, dtype=np.int32), reset

    def _acquire(self, active: set) -> int:
        if self._free:
            return self._free.pop()
        victim = -1
        victim_seen = None
        for s in range(self.num_slots):
            if self._slot_to_key[s] in active:
                continue
            if victim_seen is None or self._last_seen[s] < victim_seen:
                victim = s
                victim_seen = self._last_seen[s]
        if victim < 0:
            # All slots belong to requests active this step: only possible if
            # concurrency exceeds num_slots, which vLLM forbids. Fall back to
            # slot 0 rather than crash.
            victim = 0
        old = self._slot_to_key[victim]
        if old is not None:
            self._key_to_slot.pop(old, None)
        self._slot_to_key[victim] = None
        return victim


def build_atom_v4_attention_metadata(
    common_attn_metadata,
    *,
    meta_params=None,
    slot_allocator=None,
    decode_bufs=None,
    capturing=False,
    req_ids=None,
    num_spec_tokens=0,
    cudagraph_token_sizes=None,
):
    """Translate a vLLM ``CommonAttentionMetadata`` into ATOM's V4
    ``AttentionMetaData``.

    When ``decode_bufs`` is provided and the batch is a pure decode, the per-fwd
    index/indptr/slot/compress-plan tensors are staged into those persistent
    fixed-address buffers (CUDA/HIP-graph replay safety) and the token count is
    padded to the captured bucket (``num_actual_tokens``) with a ``batch_id ==
    -1`` sentinel tail + repeating indptr tail for the padded slots. Otherwise
    (prefill, or decode without buffers) it falls back to fresh per-fwd tensors
    (eager-only). ``capturing`` forces ``arange`` state slots so a CUDA-graph
    capture dummy batch (whose block ids are NULL) does not pollute the real
    per-request slot allocator.

    ``req_ids`` (batch-ordered, host-resident) is the slot-allocation key,
    threaded in by the req_id passthrough patch with no device sync. The decode
    slot-assignment path requires it: if it is missing/short there (patch not
    applied or out of sync) the build raises rather than reading the device
    block table.
    """
    from atom.model_ops.attentions.deepseek_v4_attn import AttentionMetaData_DSV4

    if common_attn_metadata is None:
        return AttentionMetaData_DSV4()
    state = _infer_atom_attn_state(common_attn_metadata, num_spec_tokens)
    is_decode = state.value == "decode"
    device = common_attn_metadata.seq_lens.device
    num_reqs = int(common_attn_metadata.num_reqs)
    q_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
    if q_cpu is None:
        q_cpu = common_attn_metadata.query_start_loc.cpu()
    q_np = q_cpu[: num_reqs + 1].numpy().astype(np.int32)
    lens = np.diff(q_np).astype(np.int32)
    total = int(lens.sum())  # real tokens (CG-padded reqs contribute 0)
    # Per-seq lengths on the HOST without a device sync. This vLLM build does
    # not expose an eager `seq_lens_cpu`, so `seq_lens.cpu()` is a blocking D2H
    # that drains the prior decode step's GPU work -> a large per-step bubble.
    # Prefer, in order: a future `seq_lens_cpu`; the (deprecated but exact)
    # `_seq_lens_cpu`; vLLM's CPU-resident `seq_lens_cpu_upper_bound` (exact for
    # prefill and for every decode row outside async spec-decode, which this
    # integration does not use). Fall back to the D2H only if none exist.
    # NOTE: test each for None explicitly -- `a or b` on a multi-element tensor
    # raises "Boolean value of Tensor ... is ambiguous" (e.g. CG-capture warmup).
    # IMPORTANT: read the RAW backing attributes, never the `seq_lens_cpu`
    # property -- that property lazily does `seq_lens.to("cpu")` (a blocking
    # D2H) whenever `_seq_lens_cpu` is unset, which is exactly the bubble we are
    # removing. `_seq_lens_cpu` is the exact CPU tensor when present;
    # `seq_lens_cpu_upper_bound` is a CPU tensor that is always populated and is
    # exact for prefill and every decode row outside async spec-decode (which
    # this integration does not use). Only as a last resort do the D2H.
    seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
    seq_np = seq_lens_cpu[:num_reqs].numpy().astype(np.int32)
    batch_np = np.repeat(np.arange(num_reqs, dtype=np.int32), lens)
    md = AttentionMetaData_DSV4(
        cu_seqlens_q=common_attn_metadata.query_start_loc,
        cu_seqlens_k=common_attn_metadata.query_start_loc,
        max_seqlen_q=int(common_attn_metadata.max_query_len),
        max_seqlen_k=int(common_attn_metadata.max_seq_len),
        slot_mapping=getattr(common_attn_metadata, "slot_mapping", None),
        context_lens=common_attn_metadata.seq_lens,
        block_tables=common_attn_metadata.block_table_tensor,
        state=state,
    )
    # #1600 decode contract: DeepseekV4Attention._attn_core unconditionally reads
    # `attn_md.qo_indptr` / `attn_md.kv_last_page_lens` for the paged decode call.
    # The base AttentionMetaData declares `kv_last_page_lens` (default None) but
    # NOT `qo_indptr`, so the attribute must be set explicitly or decode raises
    # AttributeError (even in bf16). Default both to None -> the native bf16
    # Triton decode path ignores them; the fp8 op5 asm path gets real per-token
    # tensors staged below.
    md.qo_indptr = None
    md.kv_last_page_lens = None
    # fp8 2buff KV cache => stage the op5 per-token decode index tensors. False
    # for bf16 and for standalone/no-meta_params builds (safe: stays None).
    kv_fp8 = bool(getattr(meta_params, "kv_fp8", False)) if meta_params else False
    if meta_params is not None:
        md.swa_num_slots = int(meta_params.num_slots)
        md.swa_window = int(meta_params.window_size)
        md.swa_cs = int(meta_params.cs)
        md.index_topk = int(meta_params.index_topk)
        md.pool_geometry = meta_params.geometry
        md.envelope_rows = int(meta_params.geometry.envelope_rows)
    else:
        # Standalone/test fallback. Production always passes meta_params bound
        # from the proxy cache before metadata reaches the model forward.
        md.swa_num_slots = num_reqs
        md.swa_window = 128
        md.swa_cs = 128
        md.index_topk = 512
    # chunk_start == num_computed_tokens (== global position of each seq's first
    # token this forward); 0 for a fresh prompt / single-shot prefill.
    chunk_start_np = np.maximum(seq_np - lens, 0).astype(np.int32)
    md.chunk_start_per_seq_cpu = chunk_start_np

    decode_persistent = is_decode and decode_bufs is not None
    # Real reqs are contiguous at the front of a (reordered) decode batch; CG
    # padding appends zero-query-len reqs at the tail.
    scheduled_bs = int((lens > 0).sum()) if is_decode else num_reqs
    # T_pad: the per-fwd token count seen by the model. For draft decode, vLLM
    # keeps `num_actual_tokens` at the real batch size but passes padded
    # input/slot tensors; use slot_mapping length so ATOM metadata also carries
    # a sentinel tail for padded draft slots.
    T_pad = _v4_padded_token_count(common_attn_metadata, total)
    # The model forward runs over ``positions`` padded to a CUDA/HIP-graph
    # capture-size bucket even when vLLM leaves the *attention metadata* token
    # count unpadded (PIECEWISE decode: ``pad_attn`` is FULL-only). Size the
    # decode token-pad (and thus the ``batch_id == -1`` sentinel tail) to that
    # same bucket so the fused decode ``qk_norm_rope`` never reads
    # ``batch_id_per_token`` past its end for the padded tail rows.
    if is_decode:
        T_pad = _v4_round_to_cudagraph_bucket(T_pad, cudagraph_token_sizes)

    # ---- per-request state slot ----
    # Real per-request state slots are assigned only for genuine (non-capture)
    # builds that carry a live allocator and real scheduled rows. The slot key
    # is vLLM's batch-ordered req_ids (the canonical, host-resident request
    # identity), threaded in by the ATOM req_id passthrough patch with no device
    # sync (installed at register.apply_vllm_req_id_passthrough_patch).
    real_slots = not capturing and slot_allocator is not None and scheduled_bs > 0
    if real_slots and req_ids is None:
        # Patch contract violated: a real build with a live allocator must
        # receive batch-ordered req_ids. None means the passthrough patch did
        # not run (not installed / out of sync) -> fail fast rather than
        # silently degrading to the old block-id key, which needed a per-step
        # D2H sync and was not immune to vLLM recycling a finished request's
        # blocks to a new request within the same step.
        raise RuntimeError(
            "ATOM V4 decode slot assignment requires batch-ordered req_ids "
            f"from the vLLM passthrough patch (scheduled_bs={scheduled_bs}), "
            "but none were threaded in. Ensure "
            "apply_vllm_req_id_passthrough_patch() ran at model registration "
            "and is still active."
        )
    if not real_slots or len(req_ids) < scheduled_bs:
        # Capture / profiling / warmup / empty synthetic batch (patch ran but
        # there are no -- or too few -- real request ids): throwaway arange
        # slots. The batch's results are discarded, and its NULL block ids /
        # absent req ids must not pollute the real per-request slot allocator.
        slot_arr = np.arange(num_reqs, dtype=np.int32)
        reset_slots: set = set()
    else:
        slot_real, reset_slots = slot_allocator.assign(
            req_ids[:scheduled_bs], chunk_start_np[:scheduled_bs]
        )
        # Padded reqs get slot 0 (a valid slot); their tokens carry batch_id ==
        # -1 so the per-token decode kernels never read them.
        slot_arr = np.zeros(num_reqs, dtype=np.int32)
        slot_arr[:scheduled_bs] = slot_real
    physical_slot_arr = np.asarray(
        [md.pool_geometry.physical_slot(int(slot)) for slot in slot_arr],
        dtype=np.int32,
    )
    md.reset_slots = {md.pool_geometry.physical_slot(int(slot)) for slot in reset_slots}

    n_csa_cpu = (seq_np // 4).astype(np.int32)
    md.n_committed_csa_per_seq_cpu = n_csa_cpu
    index_topk = int(md.index_topk)

    if decode_persistent:
        bufs = decode_bufs
        # Make the pinned host buffers safe to overwrite before staging below.
        bufs.gate_staging_reuse()
        md.state_slot_out_cpu = physical_slot_arr
        md.state_slot_out = bufs.stage(bufs.state_slot_out, physical_slot_arr)
        md.state_slot_in = bufs.stage(bufs.state_slot_in, physical_slot_arr)
        # Bridge-local helpers still use the legacy single-slot spelling. It is
        # the write slot because paged-index generation addresses the state
        # populated by this forward.
        md.state_slot_mapping_cpu = physical_slot_arr
        md.state_slot_mapping = md.state_slot_out
        # Per-token seq map padded to T_pad with the -1 sentinel tail.
        if total:
            bufs.batch_id.np[:total] = batch_np
        if T_pad > total:
            bufs.batch_id.np[total:T_pad] = -1
        md.batch_id_per_token = bufs.batch_id.copy_to_gpu(T_pad)
        # Pad CSA committed count with index_topk (aiter top_k_per_row_decode
        # derives a per-row length from this for the whole captured grid; a
        # stale/zero value on a pad row can make that length negative -> hang).
        bufs.n_csa.np[:num_reqs] = n_csa_cpu
        if num_reqs > scheduled_bs:
            bufs.n_csa.np[scheduled_bs:num_reqs] = index_topk
        md.n_committed_csa_per_seq = bufs.n_csa.copy_to_gpu(num_reqs)
        md.compress_plans = _make_decode_compress_plans(
            lens[:scheduled_bs], seq_np[:scheduled_bs], bufs
        )
        positions = getattr(common_attn_metadata, "positions", None)
        if positions is None:
            positions = torch.arange(max(total, 1), dtype=torch.int64, device=device)
        # Per-token global position from chunk_start + within-seq offset (no
        # D2H; equals vLLM's `positions[:total]` for a decode batch).
        if total:
            cu_real = np.zeros(scheduled_bs + 1, dtype=np.int32)
            np.cumsum(lens[:scheduled_bs], out=cu_real[1:], dtype=np.int32)
            within = np.arange(total, dtype=np.int32) - cu_real[batch_np]
            pos_np = (chunk_start_np[batch_np] + within).astype(np.int32)
        else:
            pos_np = np.zeros(0, dtype=np.int32)
        visible_np = np.zeros(T_pad, dtype=np.int32)
        if total:
            visible_np[:total] = visible_csa(pos_np).astype(np.int32)
        md.csa_n_committed_per_token = bufs.stage(
            bufs.csa_n_committed_per_token, visible_np
        )
        block_cols = int(common_attn_metadata.block_table_tensor.shape[1])
        block_rows = bufs.block_tables_per_token.gpu[:T_pad, :block_cols]
        safe_batch_ids = md.batch_id_per_token[:T_pad].clamp_min(0).long()
        torch.index_select(
            common_attn_metadata.block_table_tensor,
            0,
            safe_batch_ids,
            out=block_rows,
        )
        md.block_tables_per_token = block_rows
        _populate_decode_persistent(
            md,
            common_attn_metadata,
            pos_np,
            bufs,
            total,
            T_pad,
            positions,
        )
        # Decode indexer (CUDAGraph-friendly path) reads only the per-seq
        # committed count; the prefill-only fields stay unset.
        md.indexer_meta = {
            "total_committed": 0,
            "cu_committed_gpu": None,
            "n_committed_per_seq_gpu": md.n_committed_csa_per_seq,
            "batch_id_per_token_gpu": md.batch_id_per_token,
            "seq_base_per_token_gpu": None,
            "cu_starts_gpu": None,
            "cu_ends_gpu": None,
        }
        # fp8 op5 asm decode per-token index tensors (page_size=1), staged into
        # persistent buffers so the captured decode graph replays a stable
        # address. Real region [0..total] == arange; the CG-padded tail
        # [total+1..T_pad] repeats `total` (0-length queries the asm kernel skips
        # -- same trick as the kv_indptr pad tail). kv_last_page_lens == ones.
        # Matches native deepseek_v4_attn.py `_build_paged_decode_meta`.
        if kv_fp8:
            qob = bufs.qo_indptr.np
            qob[: total + 1] = bufs._qo_indptr_np[: total + 1]
            if T_pad > total:
                qob[total + 1 : T_pad + 1] = total
            md.qo_indptr = bufs.qo_indptr.copy_to_gpu(T_pad + 1)
            bufs.kv_last_page_lens.np[:T_pad] = 1
            md.kv_last_page_lens = bufs.kv_last_page_lens.copy_to_gpu(T_pad)
        # Last staging copy of this build is enqueued.
        bufs.mark_h2d_enqueued()
        return md

    # ---- eager path: prefill, or decode without persistent buffers ----
    md.state_slot_out = torch.from_numpy(physical_slot_arr).to(device)
    md.state_slot_in = md.state_slot_out.clone()
    md.state_slot_out_cpu = physical_slot_arr
    md.state_slot_mapping = md.state_slot_out
    md.state_slot_mapping_cpu = physical_slot_arr
    md.batch_id_per_token = torch.from_numpy(batch_np).to(device)
    md.n_committed_csa_per_seq = torch.from_numpy(n_csa_cpu).to(device)
    md.compress_plans = _make_compress_plans(
        lens, seq_np, [(4, True), (128, False)], device, is_decode
    )
    positions = getattr(common_attn_metadata, "positions", None)
    if positions is None:
        positions = torch.arange(total, dtype=torch.int64, device=device)
    pos_np = positions[:total].detach().cpu().numpy().astype(np.int32)
    if is_decode:
        visible_np = visible_csa(pos_np).astype(np.int32)
        md.csa_n_committed_per_token = torch.from_numpy(visible_np).to(device)
        batch_ids = torch.from_numpy(batch_np).to(device=device, dtype=torch.long)
        md.block_tables_per_token = common_attn_metadata.block_table_tensor[batch_ids]
        _populate_decode(md, common_attn_metadata, batch_np, pos_np, positions)
        # Eager fp8 decode (no persistent buffers -- standalone/tests, or a
        # batch that skips CG capture): fresh per-token op5 index tensors. Not
        # CUDAGraph-captured here, so plain arange/ones over the real token count
        # is correct (no pad tail). Production decode uses the persistent path.
        if kv_fp8:
            md.qo_indptr = torch.arange(total + 1, dtype=torch.int32, device=device)
            md.kv_last_page_lens = torch.ones(total, dtype=torch.int32, device=device)
    else:
        _populate_prefill(md, common_attn_metadata, batch_np, pos_np, q_np, positions)
    # Decode/prefill split for the indexer. vLLM reorders rows with
    # query_len <= (1 + num_spec_tokens) — i.e. decode / spec-verify / 1-token
    # rows — to the FRONT of the batch (reorder_batch_threshold). Grouping by
    # query length is exactly what the indexer needs: those rows only require
    # committed K from the paged cache, so they take the fixed-shape paged
    # `_score_topk_decode` path and are kept out of the dense prefill logits.
    decode_q = 1 + max(0, int(num_spec_tokens))
    if is_decode:
        idx_num_decodes = num_reqs
        idx_num_decode_tokens = total
        idx_decode_next_n = int(lens[0]) if num_reqs > 0 else 1
    else:
        # The paged `_score_topk_decode` reshapes the decode tokens to
        # `[num_decode_tokens // decode_next_n, decode_next_n]`, so the paged
        # decode group must be a contiguous batch prefix that shares ONE query
        # length (`decode_next_n`). vLLM usually reorders the short decode /
        # spec-verify rows (query_len <= decode_q) ahead of the longer prefill
        # rows, but that layout is NOT guaranteed under MTP + chunked prefill:
        #   * the decode prefix is not uniform -- a steady-state verify row is
        #     `1 + num_spec_tokens` long, but a request on its first decode step
        #     (or one whose drafts were all rejected) contributes a shorter row,
        #     e.g. `[4, 4, ..., 4, 1]`; and
        #   * a prefill chunk can land AMONG the verify rows, e.g. `[4, 4, 10,
        #     4]`, so decode-length rows are not always a clean leading prefix.
        # Rather than assert a layout vLLM does not promise, take the paged
        # decode group as the leading contiguous run of rows that share the
        # FIRST row's length (only when that is a decode/verify length,
        # `<= decode_q`), and end it at the first row that differs -- whether a
        # short decode straggler or a longer prefill row. Everything from there
        # flows to the dense prefill path below, which handles arbitrary per-row
        # query lengths. This is memory-safe: the long-context verify rows are
        # emitted first, so they stay on the paged path, and only the (few)
        # rows after the first mismatch reach the dense `total_committed` -- so
        # its `[total_tokens, total_committed]` logits never explode the way
        # they would if every running seq were summed in. `lens` is a CPU numpy
        # array (from `query_start_loc.cpu()`), so this adds no H2D/D2H sync.
        if num_reqs > 0 and int(lens[0]) <= decode_q:
            idx_decode_next_n = int(lens[0])
            mismatch = np.nonzero(lens != idx_decode_next_n)[0]
            idx_num_decodes = int(mismatch[0]) if mismatch.size else num_reqs
        else:
            idx_decode_next_n = 1
            idx_num_decodes = 0
        idx_num_decode_tokens = int(q_np[idx_num_decodes]) if idx_num_decodes > 0 else 0
    _populate_indexer(
        md,
        common_attn_metadata,
        batch_np,
        positions[:total],
        device,
        num_decodes=idx_num_decodes,
        num_decode_tokens=idx_num_decode_tokens,
        decode_next_n=idx_decode_next_n,
    )
    return md


def _make_decode_compress_plans(extend_lens_cpu, context_lens_cpu, bufs):
    """Decode compress plans via native ``make_compress_plans`` into the
    persistent per-ratio plan buffers (fixed capacity per ratio so capture and
    replay dispatch identically shaped compress kernels)."""
    from atom.model_ops.v4_kernels.compress_plan import make_compress_plans

    ratios_overlap = [(int(r), int(r) == 4) for r in bufs.plan_buffers]
    return make_compress_plans(
        np.ascontiguousarray(extend_lens_cpu, dtype=np.int32),
        np.ascontiguousarray(context_lens_cpu, dtype=np.int32),
        ratios_overlap,
        plan_buffers=bufs.plan_buffers,
        running_bs=bufs.decode_running_bs,
        max_q_len=bufs.decode_q_len,
    )


def _populate_decode_persistent(md, common, pos_np, bufs, total, T_pad, positions_gpu):
    """Decode index/indptr build into persistent fixed-address buffers.

    Faithful port of ATOM's ``_attach_v4_paged_decode_meta`` for plugin mode:
    three ragged ``indptr`` cumsums sized to the captured (padded) token count
    with a repeating tail (kv_len == 0 for padded slots), the HCA compress tail
    scattered on CPU, then ``write_v4_paged_decode_indices`` fills the SWA / CSA
    / HCA window-prefix offsets. All index/indptr views are sliced from the
    buffer base so their data pointers are stable across builds (the captured
    decode-attention kernels read these addresses on replay).
    """
    from atom.model_ops.v4_kernels import write_v4_paged_decode_indices
    from atom.plugin.vllm.deepseek_v4_ops import (
        write_v4_decode_hca_compress_tail,
    )

    win = int(md.swa_window)
    index_topk = int(md.index_topk)

    # Per-token slot counts over the real tokens [0:total].
    actual_swa = np.minimum(pos_np + 1, win).astype(np.int32)
    csa_valid_k = np.minimum(visible_csa(pos_np), index_topk).astype(np.int32)
    n_h_per_token = visible_hca(pos_np).astype(np.int32)

    def _indptr(counts):
        out = np.zeros(T_pad + 1, dtype=np.int32)
        out[1 : total + 1] = np.cumsum(counts, dtype=np.int32)
        if T_pad > total:
            out[total + 1 :] = out[total]
        return out

    swa_indptr = _indptr(actual_swa)
    csa_indptr = _indptr(actual_swa + csa_valid_k)
    hca_indptr = _indptr(actual_swa + n_h_per_token)
    swa_indptr_gpu = bufs.stage(bufs.indptr_swa, swa_indptr)
    csa_indptr_gpu = bufs.stage(bufs.indptr_csa, csa_indptr)
    hca_indptr_gpu = bufs.stage(bufs.indptr_hca, hca_indptr)
    hca_total = int(hca_indptr[total]) if total else 0

    # Build the decode index set directly in persistent GPU buffers. The core
    # geometry kernel writes the SWA window prefix at each slice tail; the HCA
    # helper writes compressed rows at each slice head from the GPU block table.
    # The two segments are disjoint and together cover
    # the full HCA segment `[hca_indptr[t], hca_indptr[t+1])`, so no `-1`
    # pre-fill is needed. This replaces the prior CPU HCA-tail scatter (a
    # per-step block-table D2H + numpy repeat/cumsum/fancy-index + H2D). T ==
    # real tokens; the `-1` batch_id pad tail is skipped natively by the kernel.
    swa_indices_gpu = bufs.idx_swa.gpu
    csa_indices_gpu = bufs.idx_csa.gpu
    dest_rows = {ratio: buf.gpu for ratio, buf in bufs.swa_dest_rows.items()}
    write_v4_paged_decode_indices(
        state_slot_per_seq=md.state_slot_out,
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        swa_indptr=swa_indptr_gpu,
        csa_indptr=csa_indptr_gpu,
        hca_indptr=hca_indptr_gpu,
        swa_indices=swa_indices_gpu,
        csa_indices=csa_indices_gpu,
        hca_indices=bufs.idx_hca.gpu,
        dest_rows=dest_rows,
        T=total,
        win=win,
        geometry=md.pool_geometry,
    )
    write_v4_decode_hca_compress_tail(
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        hca_indptr=hca_indptr_gpu,
        block_tables=common.block_table_tensor,
        hca_indices=bufs.idx_hca.gpu,
        T=total,
        win=win,
        envelope_rows=md.pool_geometry.envelope_rows,
    )
    md.swa_dest_rows = dest_rows
    md.kv_indices_swa = swa_indices_gpu[: int(swa_indptr[total])]
    md.kv_indices_csa = csa_indices_gpu[: int(csa_indptr[total])]
    md.kv_indices_hca = bufs.idx_hca.gpu[: max(hca_total, 0)]
    md.kv_indptr_swa = swa_indptr_gpu
    md.kv_indptr_csa = csa_indptr_gpu
    md.kv_indptr_hca = hca_indptr_gpu


def _populate_indexer(
    md,
    common,
    batch_np,
    positions,
    device,
    num_decodes=0,
    num_decode_tokens=0,
    decode_next_n=1,
):
    # In a MIXED (chunked-prefill + decode) batch vLLM orders the decode rows
    # first (reorder threshold == 1[+spec]); only the prefill rows feed the
    # dense `fp8_mqa_logits` indexer. Build the committed cumsum / per-token
    # start-end offsets over the PREFILL SEQUENCES ONLY (seqs [num_decodes:],
    # tokens [num_decode_tokens:]) so the long-context decode seqs do NOT get
    # summed into `total_committed` (the O(total_tokens x total_committed) dense
    # logits that OOMs at high concurrency). The decode rows are scored by the
    # fixed-shape paged path (`_score_topk_decode`), which reads only the full
    # per-seq committed tensor kept below. Pure prefill => num_decodes == 0, so
    # this reduces to the whole batch (unchanged).
    n_csa = md.n_committed_csa_per_seq_cpu[num_decodes:]
    cu = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(n_csa, dtype=np.int32)])
    cu[-1] = max(int(cu[-1]), 1)
    cu_gpu = torch.from_numpy(cu).to(device)
    # Per-prefill-token batch id, rebased to 0-based prefill-seq indexing.
    bid = (md.batch_id_per_token[num_decode_tokens:] - num_decodes).to(
        md.batch_id_per_token.dtype
    )
    pos_pref = positions[num_decode_tokens:]
    base = cu_gpu[bid].to(torch.int32)
    end = base + visible_csa(pos_pref).to(torch.int32)
    md.indexer_meta = {
        "total_committed": int(cu[-1]),
        "cu_committed_gpu": cu_gpu,
        # FULL per-seq committed (decode-first order): the decode sub-call
        # slices [:num_decodes]; the pure-decode path reads it whole.
        "n_committed_per_seq_gpu": md.n_committed_csa_per_seq,
        "batch_id_per_token_gpu": md.batch_id_per_token,
        "seq_base_per_token_gpu": base,
        "cu_starts_gpu": base,
        "cu_ends_gpu": end,
        # Decode/prefill split (consumed by `indexer_score_topk` for mixed
        # batches; harmless for pure prefill where num_decode_tokens == 0).
        "num_decodes": int(num_decodes),
        "num_decode_tokens": int(num_decode_tokens),
        "decode_next_n": int(decode_next_n),
        # Largest committed compressed-KV length among the DECODE sub-batch
        # seqs (host int from the CPU committed array -- no D2H sync). The
        # mixed-batch paged `_score_topk_decode` sizes its per-fwd logits width
        # to this instead of the model-max `_max_model_len_idx` (= max_seq_len
        # // compress_ratio). The deepgemm kernel guards every store by
        # `col < max_model_len_arg` and only writes [0, n_committed) per row,
        # so a width >= this max is exact, while the model-max width would
        # allocate a ~GB transient per CSA layer and OOM at high concurrency.
        "decode_max_committed": (
            int(md.n_committed_csa_per_seq_cpu[:num_decodes].max())
            if num_decodes > 0
            else 0
        ),
    }


def _populate_prefill(md, common, batch_np, pos_np, q_np, positions_gpu):
    """Chunk-aware paged-prefill index build.

    Mirrors native ATOM's ``_build_prefill_paged_indices`` (deepseek_v4_attn.py):
    per-token counts/indptrs on CPU, then one ``write_v4_paged_prefill_indices``
    Triton kernel scatters the SWA-prefix, extend, and HCA-compress index
    segments. Handles both a single full prefill (chunk_start == 0, reduces to
    the old behavior) and any later chunk (chunk_start > 0): each token's SWA
    window splits into a paged "prefix" part (positions before this chunk, read
    from the ring) and an "extend" part (positions in this chunk, read from the
    freshly written K/V). The per-layer ``csa_translate_pack`` later fills the
    CSA topk section of the prefix_csa buffer (sized exactly to match, so no
    ``-1`` sentinel fill is needed).
    """
    device = md.state_slot_mapping.device
    T = len(batch_np)
    num_reqs = int(common.num_reqs)
    win = int(md.swa_window)
    index_topk = int(md.index_topk)
    if T == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        zero1 = torch.zeros(1, dtype=torch.int32, device=device)
        md.kv_indices_extend = empty
        md.kv_indptr_extend = zero1
        md.kv_indices_prefix_swa = empty
        md.kv_indptr_prefix_swa = zero1.clone()
        md.kv_indices_prefix_csa = empty.clone()
        md.kv_indptr_prefix_csa = zero1.clone()
        md.kv_indices_prefix_hca = empty.clone()
        md.kv_indptr_prefix_hca = zero1.clone()
        md.skip_prefix_len_csa = empty.clone()
        return

    # ----- Exact per-seq chunk start (sync-free; robust to spec-decode) ------
    # chunk_start == the global position of each sequence's FIRST token this
    # forward (== num_computed_tokens by definition). Derive it from the real
    # per-token ``positions`` rather than ``seq_len - query_len``: in
    # speculative-decode (MTP) mixed prefill+verify batches
    # ``seq_lens_cpu_upper_bound`` can OVERESTIMATE seq_len, so
    # ``seq_len - query_len`` exceeds a verify token's true position. That makes
    # ``prefix_swa_count = chunk_start - swa_low`` exceed the window ``win``, and
    # the index kernel (``BLOCK_N = next_pow2(win) == win``) only writes the
    # first ``win`` slots, leaving the overflow slot UNINITIALIZED -> a garbage
    # paged offset -> illegal memory access in sparse_attn_v4_paged_prefill.
    # The first-token position is exact and guarantees token_pos_in_chunk >= 0
    # for every token, capping prefix_swa_count at ``win``. Both the CPU indptr
    # sizing below and the Triton scatter kernel read this same array, so they
    # stay consistent.
    lens_cs = np.diff(q_np[: num_reqs + 1]).astype(np.int64)
    first_tok = q_np[:num_reqs].astype(np.int64)
    chunk_start_seq = md.chunk_start_per_seq_cpu[:num_reqs].astype(np.int32).copy()
    has_tok = lens_cs > 0
    chunk_start_seq[has_tok] = pos_np[first_tok[has_tok]].astype(np.int32)
    md.chunk_start_per_seq_cpu = chunk_start_seq

    # ----- Per-token counts (CPU numpy; cumsum gives indptr totals w/o D2H) ---
    chunk_start_pt = md.chunk_start_per_seq_cpu[batch_np]
    token_pos_in_chunk = pos_np - chunk_start_pt
    swa_low = np.maximum(pos_np - win + 1, 0)
    extend_count = np.minimum(token_pos_in_chunk + 1, win).astype(np.int32)
    prefix_swa_count = np.maximum(chunk_start_pt - swa_low, 0).astype(np.int32)
    # These SIZE each slice; `write_v4_paged_prefill_indices` FILLS it from the
    # same two counts. Reserve exactly what it writes or the tail is garbage.
    csa_valid_k = np.minimum(visible_csa(pos_np), index_topk).astype(np.int32)
    n_hca_pt = visible_hca(pos_np).astype(np.int32)

    ext_indptr_np = _counts_to_indptr(extend_count)
    swa_indptr_np = _counts_to_indptr(prefix_swa_count)
    csa_indptr_np = _counts_to_indptr(prefix_swa_count + csa_valid_k)
    hca_indptr_np = _counts_to_indptr(prefix_swa_count + n_hca_pt)
    ext_total = int(ext_indptr_np[-1])
    swa_total = int(swa_indptr_np[-1])
    csa_total = int(csa_indptr_np[-1])
    hca_total = int(hca_indptr_np[-1])

    ext_indptr = torch.from_numpy(ext_indptr_np).to(device)
    swa_indptr = torch.from_numpy(swa_indptr_np).to(device)
    csa_indptr = torch.from_numpy(csa_indptr_np).to(device)
    hca_indptr = torch.from_numpy(hca_indptr_np).to(device)

    # Scatter on-GPU with the plugin-local kernel: vLLM's proxy layout keeps
    # each layer's SWA ring before its compressed pages, unlike native ATOM's
    # interleaved unified-pool geometry.
    from atom.model_ops.v4_kernels import write_v4_paged_prefill_indices

    ext_indices = torch.empty(max(ext_total, 1), dtype=torch.int32, device=device)
    swa_indices = torch.empty(max(swa_total, 1), dtype=torch.int32, device=device)
    csa_indices = torch.empty(max(csa_total, 1), dtype=torch.int32, device=device)
    hca_indices = torch.empty(max(hca_total, 1), dtype=torch.int32, device=device)
    chunk_start_g = torch.from_numpy(
        np.ascontiguousarray(md.chunk_start_per_seq_cpu[:num_reqs])
    ).to(device)
    cu_q_g = torch.from_numpy(np.ascontiguousarray(q_np[:num_reqs])).to(device)
    write_v4_paged_prefill_indices(
        positions=positions_gpu[:T].to(torch.int32),
        bid_per_token=md.batch_id_per_token[:T],
        chunk_start_per_seq=chunk_start_g,
        cu_seqlens_q_per_seq=cu_q_g,
        state_slot_per_seq=md.state_slot_out[:num_reqs],
        block_tables=common.block_table_tensor[:num_reqs],
        extend_indptr=ext_indptr,
        prefix_swa_indptr=swa_indptr,
        prefix_csa_indptr=csa_indptr,
        prefix_hca_indptr=hca_indptr,
        extend_indices=ext_indices,
        prefix_swa_indices=swa_indices,
        prefix_csa_indices=csa_indices,
        prefix_hca_indices=hca_indices,
        T=T,
        win=win,
        geometry=md.pool_geometry,
        hca_rows_per_block=ATOM_DEEPSEEK_V4_BLOCK_SIZE // 128,
    )
    md.kv_indices_extend = ext_indices[:ext_total]
    md.kv_indices_prefix_swa = swa_indices[:swa_total]
    md.kv_indices_prefix_csa = csa_indices[:csa_total]
    md.kv_indices_prefix_hca = hca_indices[:hca_total]

    md.kv_indptr_extend = ext_indptr
    md.kv_indptr_prefix_swa = swa_indptr
    md.kv_indptr_prefix_csa = csa_indptr
    md.kv_indptr_prefix_hca = hca_indptr
    md.skip_prefix_len_csa = torch.from_numpy(prefix_swa_count).to(device)


def _populate_decode(md, common, batch_np, pos_np, positions_gpu):
    device = md.state_slot_mapping.device
    win = int(md.swa_window)
    index_topk = int(md.index_topk)
    swa_counts = np.minimum(pos_np + 1, win).astype(np.int32)
    csa_counts = np.minimum(visible_csa(pos_np), index_topk).astype(np.int32)
    hca_counts = visible_hca(pos_np).astype(np.int32)
    swa_indptr_np = _counts_to_indptr(swa_counts)
    csa_indptr_np = _counts_to_indptr(swa_counts + csa_counts)
    hca_indptr_np = _counts_to_indptr(swa_counts + hca_counts)
    swa_total = int(swa_indptr_np[-1])
    csa_total = int(csa_indptr_np[-1])
    hca_total = int(hca_indptr_np[-1])
    swa_indptr = torch.from_numpy(swa_indptr_np).to(device)
    csa_indptr = torch.from_numpy(csa_indptr_np).to(device)
    hca_indptr = torch.from_numpy(hca_indptr_np).to(device)
    T = len(batch_np)

    # On-GPU build mirrors the persistent path and uses vLLM's proxy layout.
    from atom.model_ops.v4_kernels import write_v4_paged_decode_indices
    from atom.plugin.vllm.deepseek_v4_ops import (
        write_v4_decode_hca_compress_tail,
    )

    swa_indices = torch.empty(max(swa_total, 1), dtype=torch.int32, device=device)
    csa_indices = torch.empty(max(csa_total, 1), dtype=torch.int32, device=device)
    hca_indices = torch.empty(max(hca_total, 1), dtype=torch.int32, device=device)
    dest_rows = {
        ratio: torch.empty(max(T, 1), dtype=torch.int32, device=device)
        for ratio in _V4_SWA_DEST_RATIOS
    }
    write_v4_paged_decode_indices(
        state_slot_per_seq=md.state_slot_out,
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        swa_indptr=swa_indptr,
        csa_indptr=csa_indptr,
        hca_indptr=hca_indptr,
        swa_indices=swa_indices,
        csa_indices=csa_indices,
        hca_indices=hca_indices,
        dest_rows=dest_rows,
        T=T,
        win=win,
        geometry=md.pool_geometry,
    )
    write_v4_decode_hca_compress_tail(
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        hca_indptr=hca_indptr,
        block_tables=common.block_table_tensor,
        hca_indices=hca_indices,
        T=T,
        win=win,
        envelope_rows=md.pool_geometry.envelope_rows,
    )
    md.swa_dest_rows = dest_rows
    md.kv_indices_swa = swa_indices[:swa_total]
    md.kv_indices_csa = csa_indices[:csa_total]
    md.kv_indices_hca = hca_indices[:hca_total]

    md.kv_indptr_swa = swa_indptr
    md.kv_indptr_csa = csa_indptr
    md.kv_indptr_hca = hca_indptr


def get_deepseek_v4_proxy_metadata_from_vllm_context(
    layer_name: str = ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME,
):
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    meta = get_forward_context().attn_metadata
    if isinstance(meta, dict):
        return meta.get(layer_name)
    if isinstance(meta, list) and meta and isinstance(meta[0], dict):
        return meta[0].get(layer_name)
    return None


def _is_vllm_decode_graph_phase(attn_metadata, atom_config) -> bool:
    """True when vLLM is inside its CUDA-graph capture window for V4 decode.

    vLLM sets ``cudagraph_capturing_enabled=True`` around both the eager warmup
    and the actual capture. The flag is global and defaults to True, so narrow
    it to real V4 decode-shaped forwards before mapping it to ATOM's
    ``in_hipgraph``.
    """
    if getattr(getattr(attn_metadata, "state", None), "value", None) != "decode":
        return False
    try:
        import vllm.compilation.monitor as vllm_monitor
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        vllm_config = getattr(
            getattr(atom_config, "plugin_config", None), "vllm_config", None
        )
        if vllm_config is None:
            return False
        if getattr(getattr(vllm_config, "model_config", None), "enforce_eager", False):
            return False
        compilation_config = getattr(vllm_config, "compilation_config", None)
        if getattr(compilation_config, "cudagraph_mode", None) == CUDAGraphMode.NONE:
            return False
        if not is_forward_context_available():
            return False
        vllm_ctx = get_forward_context()
        batch_descriptor = getattr(vllm_ctx, "batch_descriptor", None)
        is_uniform_decode_bucket = bool(
            batch_descriptor is not None and getattr(batch_descriptor, "uniform", False)
        )
        is_single_query_decode = int(getattr(attn_metadata, "max_seqlen_q", 0)) == 1
        if not (is_uniform_decode_bucket or is_single_query_decode):
            return False
        return bool(getattr(vllm_monitor, "cudagraph_capturing_enabled", False))
    except Exception:  # noqa: BLE001 - vLLM capture state is best-effort
        return False


def _warn_inline_v4_build(common_attn_metadata, proxy_layer_name) -> None:
    """Diagnose a bound forward that still had to build metadata inline.

    Once the proxy cache is bound the builder should attach ``atom_v4_md`` and
    the forward should take the fast path, the only graph-safe one. Reaching the
    inline fallback with a live allocator means the builder never ran, or ran
    for a different object. Log it rather than degrade silently.
    """
    from vllm.forward_context import get_forward_context, is_forward_context_available

    keys = None
    if is_forward_context_available():
        meta = get_forward_context().attn_metadata
        if isinstance(meta, dict):
            keys = tuple(sorted(meta))
        elif isinstance(meta, list) and meta and isinstance(meta[0], dict):
            keys = tuple(sorted(meta[0]))
        else:
            keys = f"<{type(meta).__name__}>"
    _warn_once(
        "ATOM V4: inline attention-metadata build on a bound forward "
        "(proxy_layer=%r, common_attn_metadata=%s, forward-context attn keys=%s). "
        "The prebuilt fast path was missed; decode graph capture is unsafe here.",
        proxy_layer_name,
        type(common_attn_metadata).__name__ if common_attn_metadata else None,
        keys,
    )


@contextmanager
def atom_deepseek_v4_forward_context(
    *,
    atom_config,
    input_ids,
    positions,
    common_attn_metadata=None,
    force_dummy: bool = False,
    state_model=None,
    meta_params=None,
    slot_allocator=None,
    proxy_layer_name: str = ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME,
):
    from atom.utils.forward_context import (
        Context,
        reset_forward_context,
        running_tokens_from_bs,
        set_forward_context,
    )

    if common_attn_metadata is None:
        common_attn_metadata = get_deepseek_v4_proxy_metadata_from_vllm_context(
            proxy_layer_name
        )
    if force_dummy:
        # vLLM 0.26 captures throwaway graphs while profiling memory with a
        # deliberately minimal KV cache. V4 attention short-circuits when
        # Context.is_dummy_run is set, so avoid building real metadata here:
        # that fallback allocates and copies tensors, which HIP forbids inside
        # stream capture.
        attn_metadata = SimpleNamespace(
            state=SimpleNamespace(value="profile"),
            in_hipgraph=False,
        )
    else:
        # Fast path: the proxy metadata builder already built the ATOM metadata
        # into persistent buffers (outside any captured region) and attached it.
        # This is the only path that is CUDA/HIP-graph safe -- the captured
        # forward merely reads it. The per-slot reset was already applied in
        # build().
        attn_metadata = getattr(common_attn_metadata, "atom_v4_md", None)
    if attn_metadata is None:
        # Fallback (profiling / dummy / standalone, before the proxy cache is
        # bound): build inline with fresh tensors. Never captured.
        if slot_allocator is not None:
            _warn_inline_v4_build(common_attn_metadata, proxy_layer_name)
        if common_attn_metadata is not None:
            common_attn_metadata.positions = positions
        _spec_cfg = getattr(atom_config, "speculative_config", None)
        _num_spec = (
            int(getattr(_spec_cfg, "num_speculative_tokens", 0) or 0)
            if _spec_cfg is not None
            else 0
        )
        # With a live allocator this fallback is a *real* build (the first bound
        # forward lands here), so it needs the same host-resident slot key the
        # fast path uses. Without it the allocator's fail-fast contract trips,
        # or it silently uses throwaway arange slots and scribbles over live
        # per-request state.
        _req_ids = None
        try:
            from atom.plugin.vllm.req_id_passthrough_patch import get_current_req_ids

            _req_ids = get_current_req_ids()
        except Exception:  # noqa: BLE001 - passthrough patch is optional
            _req_ids = None
        attn_metadata = build_atom_v4_attention_metadata(
            common_attn_metadata,
            meta_params=meta_params,
            slot_allocator=slot_allocator,
            decode_bufs=(
                getattr(state_model, "_atom_v4_decode_bufs", None)
                if state_model is not None
                else None
            ),
            req_ids=_req_ids,
            num_spec_tokens=_num_spec,
        )
        # Selective per-slot reset: clear only the slots the allocator just
        # bound to a fresh request (replaces the old global position-0 reset,
        # which corrupted in-flight requests in a mixed prefill/decode batch).
        if state_model is not None:
            reset_slots = getattr(attn_metadata, "reset_slots", None)
            if reset_slots:
                reset_deepseek_v4_state_slots(state_model, reset_slots)
    in_hipgraph = not force_dummy and (
        bool(getattr(attn_metadata, "in_hipgraph", False))
        or _is_vllm_decode_graph_phase(attn_metadata, atom_config)
    )
    is_prefill = not force_dummy and attn_metadata.state.value.startswith("prefill")
    batch_size = int(
        getattr(common_attn_metadata, "num_reqs", 0)
        or (input_ids.shape[0] if input_ids is not None else 0)
    )
    context = Context(
        positions=positions,
        is_prefill=is_prefill,
        is_dummy_run=force_dummy or common_attn_metadata is None,
        scheduled_bs=batch_size,
        # The rows this forward really runs -- what MoE will be handed.
        scheduled_tokens=int(input_ids.shape[0]) if input_ids is not None else 0,
        running_bs=batch_size,
        running_tokens=running_tokens_from_bs(
            batch_size,
            is_prefill=is_prefill,
            # A `force_dummy` forward profiles memory on a throwaway batch, and
            # its `attn_metadata` is the bare namespace built above -- `state`
            # and `in_hipgraph` only, deliberately, because building real
            # metadata here allocates and copies, which HIP forbids under
            # stream capture. It therefore carries no sequence length, and
            # there is no height to derive from it. Hand the helper the `None`
            # it already documents as "count only", which is exactly the
            # `graph_bs=batch_size` this call replaced.
            attn_metadata=None if force_dummy else attn_metadata,
        ),
        input_ids=input_ids,
    )
    set_forward_context(
        attn_metadata=attn_metadata,
        atom_config=atom_config,
        context=context,
        num_tokens=int(positions.numel()),
        in_hipgraph=in_hipgraph,
    )
    try:
        yield
    finally:
        reset_forward_context()
