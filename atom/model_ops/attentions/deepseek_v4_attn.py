# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek V4 hybrid-attention backend.

Per paper §3.6.1, V4 splits cache into two parts:

  1. State cache (per-request, fixed-size pool, dynamically assigned)
     - SWA segment: most recent n_win tokens KV per layer (every layer)
     - Compressor tail buffers: uncompressed pending tokens + scores
       (CSA Main / CSA Indexer / HCA Main, fp32 for softmax-pool stability)

  2. Classical KV cache (PagedAttention-style, multi-block per request,
     block_size = lcm(m, m'))
     - CSA Main compressed KV
     - CSA Indexer compressed KV
     - HCA Main compressed KV

PR3-pre2a  (done): Compressor state buffers (kv_state + score_state ×3 owners)
                   migrated to per_req_cache pool.
PR3-pre2c-A (done): SWA buffer migration to per_req_cache pool.
PR3-pre2c-B (this revision): classical KV cache (compressed entries) moved
                   under the block_table per paper §3.6.1. Three pools allocated
                   (csa_main_kv / csa_idx_kv / hca_main_kv), shape
                   `[num_blocks, n_layers_of_type, k, head_dim]`. block_size =
                   2*lcm(m, m') = 256 original tokens, so a CSA layer keeps 64
                   rows per block and the FP4 indexer kernels run N_PHYS=1.
                   Compressor + Indexer
                   .kv_cache attributes bound to per-layer pool slices.
PR3-main:   multi-sequence dispatch (slot=0 -> per-seq slot).

Per-slot cost (V4-Pro, BF16 SWA + fp32 tail buffers, 30 CSA + 31 HCA + 1 dense):
  SWA:         62 layers * 128 * 512 * 2B  =  8.0 MB
  CSA Main:    30 * 2 * (8 * 1024)  * 4B   =  1.875 MB
  CSA Indexer: 30 * 2 * (8 * 256)   * 4B   =  0.469 MB
  HCA Main:    31 * 2 * (128 * 512) * 4B   = 16.0 MB
  Total                                      = ~26.5 MB / slot
"""

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

from atom.distributed.pcp_utils import (
    get_pcp_world_size,
    pcp_is_enabled,
    pcp_pad_dense,
    pcp_pad_indptr,
    pcp_pad_len,
    pcp_reindex_ragged,
    pcp_round_robin_query_indices,
)
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
)
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attentions.backends import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionBuilder,
)
from atom.model_ops.attentions.paged_state_copy import (
    SegmentedCopyPlan,
    launch_copy_descriptor,
    plan_segmented_copy,
)
from atom.model_ops.attentions.state_arena import (
    SplitStateArena,
    StateArena,
    StateField,
    checkpoint_ranges_for,
    plan_field_planes,
    plan_regions,
)
from atom.model_ops.attentions.sub_pool_spec import (
    SubPoolSpec,
    page_pool,
    state_pool,
)
from atom.model_ops.attentions.v4_pool_geometry import (
    ABSENT_RATIO,
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
    UnifiedPoolGeometry,
    WindowParams,
    merge_abutting,
    require_step_within_full_q,
    visible_csa,
    visible_hca,
)
from atom.model_ops.v4_kernels import (
    FP4_MQA_BLOCK_K,
    FP4_MQA_PARALLEL_UNIT_NUM,
    build_v4_paged_decode_indptr,
    fp4_indexer_enabled,
    write_v4_paged_decode_indices,
    write_v4_paged_prefill_indices,
)
from atom.utils import CpuGpuBuffer, upload_numpy
from atom.utils.forward_context import (
    AttentionMetaData,
    AttnState,
    Context,
    get_forward_context,
)

logger = logging.getLogger("atom")


def _uses_pd_staging(kv_transfer_config: dict | None) -> bool:
    """Whether this transfer topology needs compressor-only P/D staging."""

    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    return KVConnectorFactory.topology_uses_pd_staging(kv_transfer_config)


# AF_PIECEWISE: attn-core capture/replay (keyed layer, bucket_bs, q_eff, nt_pad).
# Owns its isolated graph pool + per-key graph cache + output buffers.
# State field carrying the windows of layers whose KV dtype is not the pool's.
# One field for all of them: they share a dtype (see `_discover_field_windows`)
# and a ring length, so they differ only in the field's layer dimension.
STATE_WINDOW_FIELD = "state_window"

# Per-compress-class buffer holding, for each decode token, the plane row its
# own KV goes to. One per class because a class interleaves its layers' windows
# by its own stride, so the same token lands somewhere different in each.
_DEST_ROW_BUFFERS = {
    DENSE_RATIO: "v4_swa_dest_dense",
    CSA_RATIO: "v4_swa_dest_csa",
    HCA_RATIO: "v4_swa_dest_hca",
}

# ---------------------------------------------------------------------------
# Typed metadata surface for V4. The base AttentionMetaData class is shared
# across all backends; carrying V4-specific dynamic attributes there would
# pollute it. Subclassing here gives pyright/pylance a typed surface so
# `attn_metadata.v4_kv_indices_csa` etc. don't trigger
# reportAttributeAccessIssue, while runtime behaviour stays identical
# (V4 builder constructs / promotes instances to this subclass).
# ---------------------------------------------------------------------------


@dataclass
class AttentionMetaData_DSV4(AttentionMetaData):
    """DeepSeek-V4 attention metadata.

    Extends the shared `AttentionMetaData` with V4-specific per-fwd
    metadata that `DeepseekV4AttentionMetadataBuilder` populates. The
    base class is shared across backends; carrying V4 fields there would
    pollute it. Subclassing gives pyright/pylance a typed surface so
    `attn_metadata.kv_indices_csa` etc. don't trip
    `reportAttributeAccessIssue`.

    Lifecycle: built per fwd by `prepare_decode` / `prepare_prefill` /
    `build_for_cudagraph_capture`. `is_pure_decode`-gated fields are only
    populated when the builder confirms a uniform-tokens-per-seq +
    non-fresh-prefill batch (doc §7.4); other paths leave them at
    defaults.

    Shape symbols used below:
      bs         = scheduled_bs            the seqs the scheduler gave us
      running_bs  = running_bs              what the step runs (= bs when eager)
      T          = total_tokens this fwd   (= sum of token_num_per_seq)
      padded_T   = running_bs * max_q_len   (>= T; captured kernels iterate this)
      win        = self.window_size        (128 for V4-Pro)
      index_topk = self.index_topk         (1024 for V4-Pro)
    """

    # ----- CPU mirrors (avoid GPU→CPU `.item()` / `.tolist()` syncs) -----
    state_slot_out_cpu: Any | None = None
    """[bs] np.int32 — per-seq state cache slot id (host copy)."""
    n_committed_csa_per_seq_cpu: Any | None = None
    """[bs] np.int32 — `ctx_len // 4` (CSA committed K per seq). Built once in
    `_attach_v4_per_fwd_meta` from `var["context_lens"].np`; the one consumer is
    `_attach_v4_indexer_meta`'s `cu_committed` cumsum, which concatenates the
    sequences' committed K and so needs the per-SEQUENCE total. Per-token counts
    do not come from here — see `visible_csa` in `v4_pool_geometry`."""

    # ----- Per-seq GPU scalars (single-source-of-truth, shared by kernels) -----
    state_slot_out: torch.Tensor | None = None
    """[bs] int32 GPU — per-seq state cache slot this fwd WRITES. Shared by
    swa_write + Compressor + paged-decode kernels (looked up via
    batch_id_per_token)."""
    state_slot_in: torch.Tensor | None = None
    """[bs] int32 GPU — per-seq state cache slot this fwd READS its incoming
    compressor ring from. Equal in value to `state_slot_out` except on the one
    forward carrying a state fork, but always its OWN buffer: the decode path
    replays a CUDAGraph with this pointer baked in at capture, so aliasing it to
    `state_slot_out` when nothing forked would make the distinction unreplayable
    the moment one does."""
    n_committed_csa_per_seq: torch.Tensor | None = None
    """[bs] int32 GPU — RAW `ctx_len // 4` per-seq committed count. Consumed by
    the FP4 ragged window build (`compute_varqlen_windows`), which lays out one
    window per SEQUENCE. `csa_translate_pack` does not read it: it recovers each
    token's `valid_k` from its own indptr delta."""

    # DSpark RAGGED (paper §5.2): per-request ragged verify lengths [bs] int32
    # (len_i = ell_i+1). None => regular rectangular decode. Set by
    # prepare_decode's ragged branch; consumed by `_score_topk_decode` to pad Q
    # back to a [bs, full_q] rectangle for the (rectangular-only) decode indexer
    # kernel, then gather results back to the ragged layout.
    dspark_ragged_lens_gpu: torch.Tensor | None = None
    dspark_full_q: int = 0

    # ----- Per-fwd hoisted (built in `_attach_v4_per_fwd_meta`) -----
    batch_id_per_token: torch.Tensor | None = None
    """[padded_T] int32 GPU — the SINGLE per-token mapping
    (token_idx → seq_idx). int32 indices are accepted by PyTorch
    advanced-indexing (used in the indexer); triton kernels (swa_write,
    csa_translate_pack) and the fused flydsl SWA scatter read int32. Padded
    tail [T:padded_T] = -1 sentinel; consumer kernels skip on `bid < 0`. All
    other per-token quantities resolved as `per_seq_data[batch_id_per_token[t]]`
    — no [T] aliases of seq data."""
    compress_plans: dict[int, Any] | None = None
    """dict[ratio:int -> CompressPlan] — packed plan tensors per
    compress_ratio (4=CSA, 128=HCA)."""

    # ----- Phase B paged-decode metadata (set when state is DECODE) -----
    # `state` lives on the base AttentionMetaData; every V4 `prepare_*` path
    # overrides it. Below buffers are populated only when state is DECODE
    # (built by `_attach_v4_paged_decode_meta`).
    kv_indices_swa: torch.Tensor | None = None
    """[swa_indptr[T]] int32 GPU — ragged-packed paged offsets into `unified_kv`
    for the SWA path (per-token length `min(positions[t]+1, win)`)."""
    kv_indices_csa: torch.Tensor | None = None
    """[csa_indptr[T]] int32 GPU — packed paged offsets for CSA layers
    (CSA topk compress at slice head + SWA window prefix at tail; topk section
    filled per-layer by csa_translate_pack)."""
    csa_n_committed_per_token: torch.Tensor | None = None
    """int32 GPU — uncapped CSA visibility end for each decode query row.
    Flat for rectangular decode and FP4 ragged; right-aligned `[bs*full_q]`
    for FP8 ragged. Shared by MQA logits and top-k with `next_n=1`; unlike
    the output allocation length, this is NOT capped by `index_topk`. Named
    for its class: HCA has a per-token count of its own, and the bare name read
    as if it covered both."""
    block_tables_per_token: torch.Tensor | None = None
    """int32 GPU `[decode_rows, block_table_cols]` — compressed-cache block
    table expanded from sequences to query rows by a device-side gather. It
    lets the existing aiter MQA kernels treat every query row as an
    independent batch with `next_n=1`."""
    kv_indices_hca: torch.Tensor | None = None
    """[hca_indptr[T]] int32 GPU — packed paged offsets for HCA layers
    (HCA compress at slice head + SWA window prefix at tail; layer-invariant)."""
    kv_indptr_swa: torch.Tensor | None = None
    """[padded_T+1] int32 GPU — ragged cumsum of per-token SWA length
    `min(positions[t]+1, win)`. Padded tail repeats last value → kv_len=0
    sentinel for CG-padded slots."""
    kv_indptr_csa: torch.Tensor | None = None
    """[padded_T+1] int32 GPU — packed cumsum of per-token CSA kv_len
    (= `min(positions[t]+1, win) + min((positions[t]+1)//4, index_topk)`).
    Padded tail = last value."""
    kv_indptr_hca: torch.Tensor | None = None
    """[padded_T+1] int32 GPU — packed cumsum of per-token HCA kv_len
    (= `min(positions[t]+1, win) + (positions[t]+1)//128`). Padded tail = last
    value."""
    envelope_rows: int = 0
    """Rows one V4 block takes across every layer of the pool — the stride from
    one block's compressed rows to the next in a layer's view of a plane. What
    `csa_translate_pack` turns a physical block id into a row with."""
    swa_dest_rows: dict[int, torch.Tensor] | None = None
    """{compress ratio: [padded_T] int32 GPU} — the plane row each decode token's
    own KV goes to, in a layer of that class. Handed to the fused SWA write so
    the row formula lives in one place; decode-only (prefill scatters its window
    tail through `swa_write`, which derives the row itself)."""

    # ----- Native 2buff fp8 per-token paged-decode index tensors -----
    # Feed the aiter asm decode kernel `mla_decode_fwd_v4_nm` (op5), which treats
    # each decode token as a 1-token page (page_size=1). Both depend ONLY on the
    # padded decode token count N (the captured kernel grid), never on batch
    # content — the values are always arange(N+1) / ones(N). Staged every fwd via
    # the SAME forward_vars path as `kv_indptr_*` (CpuGpuBuffer H2D), which is
    # what makes them CUDAGraph-safe. Only populated on the fp8 path.
    qo_indptr: torch.Tensor | None = None
    """[padded_T+1] int32 GPU — per-token q indptr `arange(N+1)` (page_size=1,
    max_seqlen_q=1). NOT `cu_seqlens_q` (which is per-seq and differs under
    MTP); this is the per-token indptr the decode kernel consumes."""
    kv_last_page_lens: torch.Tensor | None = None
    """[padded_T] int32 GPU — per-token last-page length `ones(N)` (page_size=1
    → every page is full)."""

    # ----- Indexer / sparse-layout side metadata -----
    indexer_meta: dict[str, Any] | None = None
    """dict — `Indexer.forward_batched` per-fwd GPU tensors. Notable keys:
      cu_committed_gpu              [bs+1] int32  per-seq committed cumsum
      seq_base_per_token_gpu        [T] int32  prefill subtract base (also
                                                aliased as cu_starts_gpu for
                                                fp8_mqa_logits)
      cu_ends_gpu                   [T] int32  per-token end offset for
                                                fp8_mqa_logits (causal cap)
      total_committed               int  sum of n_committed_csa_per_seq

    Note: decode logits / topk-indices scratch are allocated per-fwd inside
    `Indexer._score_topk_decode` (write-once, no CPU mirror, CG-stable via
    the captured graph's private memory pool).

    The indexer's downstream contract: `_score_topk_*` returns RAW seq-local
    `[T, index_topk] int32` with kernel-native -1 in tail cols (cells past
    the per-token visibility cap). `csa_translate_pack` consumes this layout
    directly — no separate width-mask / offset / future-threshold staging
    needed.
    """
    skip_prefix_len_csa: torch.Tensor | None = None
    """[padded_T] int32 GPU — per-token SWA prefix length within each token's
    region. Decode path: filled with `window_size`; csa_translate_pack uses it
    to recover the CSA topk length (`valid_k = slice_len - skip`) and writes
    the topk section at the slice head (SWA prefix occupies the tail). Prefill
    path: equals `prefix_swa_count_per_token[t]` — 0 for pure prefill (no prior
    chunk), or the `< chunk_start` portion of the SWA window for chunked
    prefill (prefill keeps the SWA prefix at the head). CG-padded tail slots:
    0 (kernel bails on `bid<0` so the value is irrelevant)."""

    # ----- Prefill-only paged-prefill index buffers (set in `_build_paged_prefill_meta`) -----
    # Two-source paged_prefill kernel reads:
    #   prefix region from `unified_kv` (SWA history + CSA/HCA compress)
    #   extend region from per-fwd `kv` tensor (in-chunk SWA tail)
    # Per-ratio prefix buffers (SWA-only stride for Dense, SWA + compress
    # for CSA/HCA). Extend buffer is layer-invariant, shared by all 3.
    kv_indices_prefix_swa: torch.Tensor | None = None
    """[sum(prefix_swa_count)] int32 GPU — flat paged offsets into
    `unified_kv` for Dense (ratio==0) layers' prefix region (SWA history
    only)."""
    kv_indptr_prefix_swa: torch.Tensor | None = None
    """[total_tokens + 1] int32 GPU — packed cumsum of `prefix_swa_count`."""
    kv_indices_prefix_csa: torch.Tensor | None = None
    """[sum(prefix_swa_count + min((positions[t]+1)//4, index_topk))] int32 GPU
    — CSA topk (head) + SWA history (tail) per token. Filled per-layer by
    `csa_translate_pack`; SWA prefix section is filled by builder at the slice
    tail (head-CSA / tail-SWA convention, matching decode, #1116)."""
    kv_indptr_prefix_csa: torch.Tensor | None = None
    """[total_tokens + 1] int32 GPU — packed cumsum of
    `prefix_swa_count + min((positions[t]+1)//4, index_topk)`."""
    kv_indices_prefix_hca: torch.Tensor | None = None
    """[sum(prefix_swa_count + (positions[t]+1)//128)] int32 GPU — SWA history
    (head) + the HCA groups closed at or before the token's own position
    (tail). Layer-invariant, fully filled by builder."""
    kv_indptr_prefix_hca: torch.Tensor | None = None
    """[total_tokens + 1] int32 GPU — packed cumsum of
    `prefix_swa_count + (positions[t]+1)//128`."""
    kv_indices_extend: torch.Tensor | None = None
    """[sum(extend_count)] int32 GPU — flat row offsets into the per-fwd
    `kv` tensor (in-chunk SWA tail) for the extend region. Layer-invariant
    (same `kv` shared by all 3 ratios; one builder pass)."""
    kv_indptr_extend: torch.Tensor | None = None
    """[total_tokens + 1] int32 GPU — packed cumsum of `extend_count`."""


class DeepseekV4Backend(AttentionBackend):
    """Backend selector entry for V4 hybrid attention.

    V4 forward is custom (does not go through ATOM's standard AttentionImpl);
    this backend exists primarily so the metadata builder is reachable from
    `ModelRunner.attn_metadata_builder` and the per-request cache abstraction
    can size + own V4's state caches.
    """

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4"

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return DeepseekV4AttentionMetadataBuilder


class DeepseekV4AttentionMetadataBuilder(CommonAttentionBuilder):
    """Per-request cache owner for V4's state-cache buffers.

    Inherits CommonAttentionBuilder for the standard prefill/decode prep
    (slot_mapping, block_tables, cu_seqlens). `block_size` is 2*lcm(m, m') =
    256 (V4-Pro: m=4 CSA, m'=128 HCA), a multiple of lcm so each classical KV
    cache block still holds an integral number of compressed rows per layer
    (block_size/m = 64 for CSA, block_size/m' = 2 for HCA). We use 2*lcm (not
    lcm) because the FP4 paged-MQA-logits indexer kernels require the indexer
    kv_block_size (= the CSA row count) to be 64 for N_PHYS=1. Must equal
    `config.kv_cache_block_size` (config.py forces the same value for V4).
    """

    block_size = 256

    # Number of micro-batches for Two-Batch Overlap (TBO).
    _NUM_TBO_UBATCHES = 2

    def __init__(self, model_runner):
        super().__init__(model_runner)
        hf = model_runner.config.hf_config
        ratios = list(getattr(hf, "compress_ratios", ()))
        assert ratios, "deepseek_v4 hf_config must define compress_ratios"
        self.compress_ratios = ratios
        self.num_layers = len(ratios)
        # Per-buffer-type layer indexing.
        # Buffers are layer-major: shape [num_layers_of_type, num_slots, *state_shape].
        self.csa_layers = [i for i, r in enumerate(ratios) if r == 4]
        self.hca_layers = [i for i, r in enumerate(ratios) if r == 128]
        self.dense_layers = [i for i, r in enumerate(ratios) if r == 0]
        self.layer_id_to_csa_pos = {lid: p for p, lid in enumerate(self.csa_layers)}
        self.layer_id_to_hca_pos = {lid: p for p, lid in enumerate(self.hca_layers)}
        # Unique (ratio, is_overlap) pairs needed for compress-plan generation.
        # CSA ratio=4 has overlap=True; HCA ratio=128 has overlap=False.
        unique = []
        if self.csa_layers:
            unique.append((4, True))
        if self.hca_layers:
            unique.append((128, False))
        self._unique_compress_ratios_overlap = unique

        # Geometry from HF config.
        self.head_dim = getattr(hf, "kv_head_dim", 512)
        self.index_head_dim = getattr(hf, "index_head_dim", 128)
        self.window_size = getattr(hf, "sliding_window", 128)
        self.index_topk = getattr(hf, "index_topk", 1024)
        self.rope_head_dim = getattr(hf, "qk_rope_head_dim", 64)
        # MTP-portion of compress_ratios. `prepare_mtp_decode`'s direct-kernel
        # fast path only handles SWA (ratio=0) draft layers; non-zero ratios
        # would also need n_committed_{csa,hca} + HCA compress tail rebuilt.
        # V4-Pro currently ships all-zero MTP ratios; assert keeps future
        # configs honest.
        n_main = int(getattr(hf, "num_hidden_layers", len(ratios)))
        self._n_main_layers = n_main
        self._mtp_layers_are_swa_only = all(r == 0 for r in ratios[n_main:])
        # `deepgemm_fp8_paged_mqa_logits` decode-path output column count
        # = max compressed K positions per seq. CSA ratio=4 is the
        # max-density ratio (1 indexer slot per 4 source tokens).
        self.max_model_len_idx = model_runner.config.max_model_len // 4

        # Classical KV pool geometry. block_size=256 original tokens means one
        # V4 block compresses to 256/4=64 rows in a CSA layer and 256/128=2 rows
        # in an HCA layer (paper §3.6.1; block_size is a multiple of lcm).
        self.csa_rows_per_block = self.block_size // 4  # = 64
        self.hca_rows_per_block = self.block_size // 128  # = 2
        self._rows_per_block = {
            4: self.csa_rows_per_block,
            128: self.hca_rows_per_block,
        }

        self._state_dtype = torch.float32  # fp32 required for softmax-pool
        # KV cache dtype gate. fp8 → 2buff native layout (nope fp8 in a 512B
        # entry with inline e8m0 scale; parallel bf16 rope pool). bf16 →
        # unchanged. SWA and classical (CSA/HCA Main) share the nope dtype; the
        # rope pool is always bf16.
        self._kv_fp8 = model_runner.kv_cache_dtype == "fp8"
        # aiter prefill (op4) / decode (op5) implement the fp8 (2buff) path only
        # on gfx950 / gfx1250. On any other arch, transparently fall back to a
        # bf16 KV cache instead of hard-failing. Flipping self._kv_fp8 here (before
        # the *_dtype attrs are read) keeps the whole V4 path consistent: pool
        # sizing (sub_pool_specs / swa_block_bytes_per_layer), quant_mode,
        # and module.kv_fp8 (build_kv_cache_tensor) all key off self._kv_fp8 /
        # these dtype attrs. Sync model_runner.kv_cache_dtype (and the shared
        # config) so any generic reader / log line agrees.
        if self._kv_fp8 and get_gfx() not in ("gfx950", "gfx1250"):
            logger.warning(
                "DeepSeek-V4 --kv_cache_dtype fp8 (2buff) is only supported on "
                "gfx950 / gfx1250 (aiter op4/op5); got %r. Falling back to a "
                "bf16 KV cache.",
                get_gfx(),
            )
            self._kv_fp8 = False
            model_runner.kv_cache_dtype = "bf16"
            cfg = getattr(model_runner, "config", None)
            if cfg is not None and getattr(cfg, "kv_cache_dtype", None) == "fp8":
                cfg.kv_cache_dtype = "bf16"
        if self._kv_fp8:
            self._swa_dtype = dtypes.fp8
            self._classical_dtype = dtypes.fp8
            self._rope_dtype = torch.bfloat16  # rope pool is always bf16
        else:
            self._swa_dtype = torch.bfloat16  # SWA window matches KV dtype
            self._classical_dtype = torch.bfloat16  # CSA / HCA Main KV is BF16
            self._rope_dtype = torch.bfloat16  # unused in bf16 path (symmetry)
        # CSA Indexer cache: `index_head_dim` FP8 bytes plus a 4-byte fp32
        # scale per row. Data and scale sit in two REGIONS inside a block, NOT
        # interleaved per row: `[rows*index_head_dim data][rows*4 scale]`. All
        # three consumers address it that way — fused_compress.py (write),
        # cache_kernels.cu:1638/1651 (cp_gather_indexer_k_quant_cache), and
        # pa_mqa_logits.py:493-500 (deepgemm_fp8_paged_mqa_logits).
        #
        # So the alignment that matters is the BLOCK stride, and 64 * 132 =
        # 8448 = 16 * 528 is already 16-byte aligned. Rounding this per-row
        # value up to a multiple of 16 instead pays the 12-byte rounding once
        # per row rather than once per block: 768 B per block per CSA layer,
        # 1.5% of the whole KV pool.
        self._index_row_bytes = self.index_head_dim + 4
        indexer_block_bytes = self.csa_rows_per_block * self._index_row_bytes
        assert indexer_block_bytes % 16 == 0, (
            f"indexer block stride {indexer_block_bytes} B "
            f"({self.csa_rows_per_block} rows x {self._index_row_bytes} B) must "
            "be 16-byte aligned: the FP8 data region is read with dwordx4 loads"
        )

        # FP4 indexer cache (the native single-node default except on gfx942).
        # When enabled, the CSA Indexer KV is
        # stored as packed FP4 E2M1 + per-group(32) e8m0 scale in the
        # `pa_mqa_logits_fp4` preshuffle layout (data
        # [NB, k_tiles, 4, rows, 16] uint8 + scale [NB, k_tiles, 4, rows] uint8)
        # written by `fused_compress_attn(quant_mode="fp4")`. The scoring path
        # auto-detects FP4 via `kv_cache.dtype == uint8`. Explicit fp8 keeps the
        # existing FP8 (+fp32 scale) path byte-identical.
        # `--index_cache_dtype` remains an explicit override. This is the
        # authoritative decision re-asserted onto every Indexer in
        # `build_kv_cache_tensor`.
        # `warn=True`: the gfx942 fallback message is emitted here only, since the
        # builder is constructed once while `Indexer.__init__` runs per CSA layer.
        # Shared predicate (see `fp4_indexer_enabled`) so the builder and
        # `Indexer.__init__` cannot drift apart.
        self._indexer_fp4 = fp4_indexer_enabled(
            getattr(model_runner.config, "index_cache_dtype", None), warn=True
        )
        # FP4 KV tile geometry (group_size 32; 16 packed bytes per group).
        self._idx_k_tiles = self.index_head_dim // 128

        # MTP token-per-fwd factor for paged-decode buffer sizing. V4-Pro
        # `num_nextn_predict_layers = 1` → mtp_k = 1 → max_q_len = 2 per req.
        # `model_runner.drafter` is created BEFORE `attn_metadata_builder`
        # (model_runner.__init__ ordering), so this hasattr is reliable.
        self.max_spec_steps = (
            int(model_runner.drafter.mtp_k) if hasattr(model_runner, "drafter") else 0
        )

        # Compressor state shape: [ring_size, coff * head_dim], fp32.
        # ring_size = K_pool + max_spec_steps, where K_pool = coff * ratio.
        #
        # Per spec round we write up to (1 + max_spec_steps) consecutive token
        # positions; if some draft tokens are rejected, round R+1 re-commits
        # those slots starting from a later offset. The aliasing concern is:
        # at round R+1, while we read the K_pool committed entries that R+1's
        # attention needs, can a position R already wrote (and we'd be about
        # to overwrite) collide with one of those reads?
        #
        # Slot index = (compressed_K_id) % ring_size, where
        # compressed_K_id = pos // ratio. Round R+1 reads
        # `K_pool` consecutive ids ending at its own commit head; round R's
        # rejected writes sit `<= max_spec_steps` ids beyond that head. With
        # `ring_size = K_pool + max_spec_steps`, R's stale ids are guaranteed
        # to fall outside R+1's K_pool-wide read window — no collision.
        # Adding a further +1 (the old layout) was unnecessary slack.
        # CSA: ratio=4, overlap=True  → K_pool=8;  ring_size=8 + mtp_k
        # HCA: ratio=128, overlap=False → K_pool=128; ring_size=128 + mtp_k
        # Non-spec (max_spec_steps=0) → ring_size = K_pool: no rejections ever
        # happen, so the bare commit pool is sufficient (causal writes mean
        # the alias slot is never read before being overwritten).
        # `ring_extra` is slack beyond K_pool for the compressor ring buffer.
        # Validated via `ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL=1` (100% reject =
        # worst case for aliasing): even at ring_extra=0, decode commits the
        # correct next token, confirming no read-from-stale slot collision.
        # See `Adding a further +1 (the old layout) was unnecessary slack` below.
        ring_extra = self.max_spec_steps
        self.csa_main_state_shape = (2 * 4 + ring_extra, 2 * self.head_dim)
        self.csa_idx_state_shape = (2 * 4 + ring_extra, 2 * self.index_head_dim)
        self.hca_main_state_shape = (128 + ring_extra, self.head_dim)
        self.max_decode_tokens = self.max_bs * (1 + self.max_spec_steps)
        # SWA ring-buffer slots per req. Distinct from `window_size`:
        #   * `window_size`  = SWA attention window = topk count per token
        #     (each query attends to W consecutive K/V positions).
        #   * `win_with_spec` = `window_size + max_spec_steps` = ring-buffer
        #     slot count per req. With MTP-k the per-fwd writes the verified
        #     token + k draft tokens at positions [p_0..p_k]; if the cache
        #     were only sized W, draft slots `p_(i+1)..p_k` would alias into
        #     [p_0-W+1..p_0] and the verified query at `p_0` would read
        #     future tokens (silent correctness bug). MTP off → max_spec_steps
        #     == 0 → win_with_spec == window_size, identical bytes layout.
        # Used as: SWA `unified_kv` per-slot stride, `swa_kv` ring-buffer dim,
        # `swa_write` modulo, and the ring-index modulo `cs` in the V4
        # paged-decode index-write kernel.
        self.win_with_spec = self.window_size + self.max_spec_steps
        # Layers whose window the planes cannot hold at their own row width.
        # Asked of the modules rather than derived from the config: the width
        # is a property of what a layer's attention kernel can consume, and the
        # layer is the only thing that knows it. This runs after the model and
        # the drafter are built (`model_runner.py` builds them, then us).
        self._field_window_layers, self._field_window_dtype = (
            self._discover_field_windows()
        )
        # A layer whose window moved into a state field is not served by the
        # dense class any more — it is not in the row space at all — so the
        # SWA-only MTP fast path cannot address it, whatever its config ratio
        # says. Narrowed here rather than where the ratios are read, because
        # which layers those are is only known once the modules exist.
        self._mtp_layers_are_swa_only = self._mtp_layers_are_swa_only and not any(
            layer_id >= self._n_main_layers for layer_id in self._field_window_layers
        )
        # How the compressor state divides between the planes, and what that
        # costs a slot in rows. Settled here because the row space is built
        # from it and sizing prices a slot before either count exists.
        self._arena_planes, arena_rows = plan_field_planes(
            self._state_fields(), self._plane_row_widths()
        )
        # The row space both KV planes materialize (see `v4_pool_geometry`).
        # Built at capacity zero: sizing has to price a block and a request
        # before either count exists, and both prices are counts of rows, which
        # the split does not change. `allocate_per_req_cache` replaces this with
        # the same layout at the split sizing chose. Nothing dereferences an
        # address before then — warmup runs first and `DeepseekV4Attention`
        # short-circuits on `is_dummy_run`.
        self.pool_geometry = UnifiedPoolGeometry(
            self._geometry_ratios(),
            num_blocks=0,
            num_slots=0,
            ring_slots=self.win_with_spec,
            block_size=self.block_size,
            arena_rows=arena_rows,
            slot_align_rows=self._slot_align_rows(),
        )
        # Worst-case HCA per-token committed compress count
        # (= max_model_len // 128 for V4-Pro = 8192 at 1M context).
        self.max_committed_hca = model_runner.config.max_model_len // 128

        # Sparse-attn + per-fwd metadata buffers (CG-A: pre-allocate for fixed
        # GPU pointers, prerequisite for CUDAGraph capture). All H2D copies in
        # the V4 metadata builder go through these buffers via the
        # `np[:n] = arr; copy_to_gpu(n)` pattern instead of per-call
        # `torch.as_tensor(arr)` allocations.
        self._alloc_v4_metadata_buffers()

        self._ubatch_decode_meta: list | None = None
        # Filled on the first checkpoint copy — the pools do not exist yet
        # here. Four of the five hold raw addresses read out of the pool
        # tensors, so a re-carve invalidates them all; that is what
        # `_invalidate_pool_caches` is for, and `allocate_per_req_cache`
        # calls it.
        self._slot_view_cache: list[list[torch.Tensor]] | None = None
        self._checkpoint_range_cache: list[list[tuple[int, int]]] | None = None
        self._page_unit_region_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._page_unit_region_owners: tuple[int, ...] = ()
        self._checkpoint_plan_cache: SegmentedCopyPlan | None = None
        self._checkpoint_slot_base_cache: np.ndarray | None = None
        self._checkpoint_descriptor: CpuGpuBuffer | None = None

    @property
    def prep_stream(self):
        return self.model_runner.async_execute_stream

    # ------------------------------------------------------------------ #
    # AttentionMetadataBuilder hooks (per-request cache abstraction).    #
    # ------------------------------------------------------------------ #

    # ---- Windows the planes cannot hold at their own row width ---------- #
    # A DSpark draft layer's block attention consumes unquantized KV, so under
    # a packed pool its ring cannot be rows of a plane. Rather than let the
    # pool guess, the layer declares the width and the pool reserves it as a
    # state field — which is what makes a future fused fp8 draft kernel a
    # one-line change on the layer instead of a layout decision here.

    def _discover_field_windows(self) -> tuple[tuple[int, ...], torch.dtype | None]:
        """The layers that declared a window KV dtype, and that dtype.

        Read off the modules, not the config, because the width follows from
        which kernel the layer runs. Safe at this point: `ModelRunner` builds
        the model and the drafter before it builds us.

        Layers come back sorted, which fixes their order in the field's layer
        dimension — the one place that order is decided.
        """
        runner = self.model_runner
        drafter = getattr(runner, "drafter", None)
        found: dict[int, torch.dtype] = {}
        for root in (getattr(runner, "model", None), getattr(drafter, "model", None)):
            if root is None:
                continue
            for module in root.modules():
                dtype = getattr(module, "window_kv_dtype", None)
                if dtype is not None:
                    found[module.layer_id] = dtype
        widths = set(found.values())
        if len(widths) > 1:
            raise NotImplementedError(
                f"window KV dtypes {sorted(map(str, widths))} in one pool: the "
                "state field carrying them is one field, so one dtype"
            )
        return tuple(sorted(found)), widths.pop() if widths else None

    def _geometry_ratios(self) -> list[int]:
        """Per-layer ratios as the row space sees them.

        A layer carrying its window in a state field keeps nothing addressed
        by block or by window row, so the row space is told it is absent — it
        must not also reserve entry rows it would never read.
        """
        return [
            ABSENT_RATIO if layer_id in self._field_window_layers else ratio
            for layer_id, ratio in enumerate(self.compress_ratios)
        ]

    def _window_field_row_bytes(self) -> int:
        """Bytes one window position takes: `head_dim` of the layer's dtype."""
        return self.head_dim * self._field_window_dtype.itemsize

    def _window_field_plane_index(self) -> int:
        """Which plane `plan_field_planes` put the state-carried window in."""
        return next(
            p
            for p, group in enumerate(self._arena_planes)
            if any(f.name == STATE_WINDOW_FIELD for f in group)
        )

    def _window_field_plane_rows(self) -> int:
        """Plane rows one window position spans, in the plane it landed in.

        One when the two widths agree, which is the case a shared dense class
        would have handled — nothing downstream depends on which of the two.
        """
        plane = self._plane_row_widths()[self._window_field_plane_index()]
        row_bytes = self._window_field_row_bytes()
        if row_bytes % plane:
            raise ValueError(
                f"a {row_bytes}B window row does not divide into {plane}B plane "
                "rows; the field cannot be addressed as rows of that plane"
            )
        return row_bytes // plane

    def _slot_align_rows(self) -> int:
        """Row multiple a slot is rounded to, so every view of it divides.

        Two for the compressor state alone (`UnifiedPoolGeometry.slot_rows`).
        A state-carried window is read through a plane retyped to its own
        width, so the slot has to be a whole number of those rows too.
        """
        if self._field_window_dtype is None:
            return 2
        return math.lcm(2, self._window_field_plane_rows())

    def _window_field_plane(self) -> torch.Tensor:
        """The plane holding the state-carried window, retyped to its width.

        `[rows, head_dim]` of the layer's own dtype, where a row is one window
        position — which is what `swa_write` and the DSpark gather take, so
        neither of them learns that this window is not a plane of its own.
        """
        plane = self._kv_planes()[self._window_field_plane_index()]
        return plane.view(self._field_window_dtype).view(-1, self.head_dim)

    def _window_field_params(self, layer_id: int) -> WindowParams:
        """Where one state-carried window sits, in retyped-plane rows."""
        plane_bytes = self._plane_row_widths()[self._window_field_plane_index()]
        row_bytes = self._window_field_row_bytes()
        offset = self.model_runner.v4_state_arena.field_offset(STATE_WINDOW_FIELD) + (
            self._field_window_layers.index(layer_id) * self.win_with_spec * row_bytes
        )
        return self.pool_geometry.field_window_params(
            offset // plane_bytes, row_bytes // plane_bytes
        )

    def _state_fields(self) -> list[StateField]:
        """The per-request state one request carries: compressor, and windows
        the planes cannot hold at their own row width.

        A `[kv_state, score_state]` fp32 pair per Compressor instance — CSA
        Main, CSA Indexer, HCA Main (paper §3.6.1). Most windows are NOT here:
        they are rings in the entry region of a slot, addressed by the same
        state slot but sized per token rather than per compressor entry. A
        layer whose window KV has a dtype of its own joins as a field instead,
        because a plane is one width and a field is the one thing in this
        layout that is priced in bytes.

        Keeping it in the slot makes checkpoint scatter carry the window too.

        Field order is the wire order of a whole entry, so it is also the
        order a PD transfer or a checkpoint sees the bytes in.
        """
        neg_inf = float("-inf")
        dt = self._state_dtype
        n_csa, n_hca = len(self.csa_layers), len(self.hca_layers)
        fields = [
            StateField("csa_main_kv", n_csa, self.csa_main_state_shape, dt),
            StateField("csa_main_score", n_csa, self.csa_main_state_shape, dt, neg_inf),
            StateField("csa_idx_kv", n_csa, self.csa_idx_state_shape, dt),
            StateField("csa_idx_score", n_csa, self.csa_idx_state_shape, dt, neg_inf),
            # HCA owes a checkpoint nothing. It pools `ratio` tokens with no
            # overlap, so the first compression at or after a boundary P
            # covers `[P, P + 128)` — every row of it written by the very
            # forward that reads it — and a checkpoint sits on a multiple of
            # `hash_block_size`, which `_assert_ratios_divide_the_alignment` keeps a
            # multiple of 128. The rows past `K_pool` are speculative
            # rollback slack and are never read at all.
            StateField(
                "hca_main_kv",
                n_hca,
                self.hca_main_state_shape,
                dt,
                in_checkpoint=False,
            ),
            StateField(
                "hca_main_score",
                n_hca,
                self.hca_main_state_shape,
                dt,
                neg_inf,
                in_checkpoint=False,
            ),
        ]
        if self._field_window_dtype is not None:
            fields.append(
                StateField(
                    STATE_WINDOW_FIELD,
                    len(self._field_window_layers),
                    (self.win_with_spec, self.head_dim),
                    self._field_window_dtype,
                    # Its rows are also reached by index, so the field has to
                    # start on one of its own rows, not merely on the retype
                    # boundary every field gets.
                    align=self._window_field_row_bytes(),
                )
            )
        return fields

    def state_transfer(self) -> StateTransfer:
        """Declare PAGE-copy checkpoints with the versioned DSV4 layout."""
        ratios = ",".join(str(r) for r in self._geometry_ratios())
        nocopy = ",".join(f.name for f in self._state_fields() if not f.in_checkpoint)
        layout_id = (
            # v2: an image holds part of a slot, not all of it. Two workers
            # disagreeing about which part would read one image at two
            # layouts, so `nocopy` names the rule and the version fences it.
            # v3: it also drops the entry's interleave padding, so the image
            # is no longer a subsequence of the slot's rows and a v2 reader
            # would gather every window row shifted.
            "dsv4-paged-state-v3"
            f":block={self.block_size}:ring={self.win_with_spec}"
            f":dims={self.head_dim},{self.rope_head_dim},{self.index_head_dim}"
            f":state={self.csa_main_state_shape},{self.csa_idx_state_shape},"
            f"{self.hca_main_state_shape}"
            f":main={'fp8-2buff' if self._kv_fp8 else 'bf16'}"
            f":index={'fp4' if self._indexer_fp4 else 'fp8'}"
            f":ratios={ratios}"
            f":nocopy={nocopy}"
            ":entry=packed"
        )
        return StateTransfer.copy(layout_id)

    def relocate_state_slots(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Relocate a request's whole Active Slot."""
        views = self._slot_views()
        dsts, srcs = [], []
        for src, dst in pairs:
            dsts += views[dst]
            srcs += views[src]
        if dsts:
            torch._foreach_copy_(dsts, srcs)

    def execute_paged_state_copies(
        self,
        store_ops: Sequence[CheckpointStoreOp],
        restore_ops: Sequence[CheckpointRestoreOp],
    ) -> None:
        """Copy raw checkpoint bytes between slots and non-contiguous PAGEs.

        Every op of either direction goes into one descriptor and one launch.
        A store and a restore are the same intersection read opposite ways, so
        they share the plan too — and each direction is described in a single
        vectorised pass, which is why they are batched apart rather than
        interleaved.
        """
        if not store_ops and not restore_ops:
            return
        for op in (*store_ops, *restore_ops):
            self._validate_paged_state_op(op)

        plan = self._checkpoint_copy_plan()
        slot_bases = self._checkpoint_slot_bases()
        per_op = plan.num_spans
        total = (len(store_ops) + len(restore_ops)) * per_op
        staging = self._checkpoint_descriptor_buffer()
        if total > staging.np.shape[0]:
            raise RuntimeError(
                f"a step asked to copy {total // per_op} checkpoints, more "
                f"than the {staging.np.shape[0] // per_op} its descriptor was "
                "sized for"
            )
        descriptor = staging.np[:total]
        at = 0
        for ops, storing in ((store_ops, True), (restore_ops, False)):
            if not ops:
                continue
            end = at + len(ops) * per_op
            groups = [op.src_slot if storing else op.dst_slot for op in ops]
            plan.write_descriptor(
                descriptor[at:end],
                slot_bases[groups],
                self._page_unit_bases([op.unit_ids for op in ops]),
                forward=storing,
            )
            at = end
        launch_copy_descriptor(staging.copy_to_gpu(total), plan)

    def _validate_paged_state_op(
        self, op: CheckpointStoreOp | CheckpointRestoreOp
    ) -> None:
        spec = self.model_runner.state_runtime.checkpoint_spec
        if spec is None:
            raise RuntimeError("DSV4 PAGE/state checkpoint spec is missing")
        if op.layout_id != spec.layout_id:
            raise RuntimeError(
                f"state checkpoint layout mismatch: {op.layout_id!r} != "
                f"{spec.layout_id!r}"
            )
        if op.total_bytes != spec.image_bytes:
            raise RuntimeError(
                f"state checkpoint size mismatch: op={op.total_bytes}, "
                f"image={spec.image_bytes}"
            )
        if len(op.unit_ids) != spec.units_per_checkpoint:
            raise RuntimeError(
                "state checkpoint PAGE-unit geometry does not match this worker"
            )
        num_blocks = self.model_runner.num_physical_kvcache_blocks
        if any(unit_id < 0 or unit_id >= num_blocks for unit_id in op.unit_ids):
            raise RuntimeError("state checkpoint PAGE unit is out of range")

    def _checkpoint_slot_ranges(self) -> list[list[tuple[int, int]]]:
        """Per plane, the `(offset, num_bytes)` of a slot a checkpoint holds.

        A slot is a request's compressor state and then its sliding windows
        (`v4_pool_geometry`). The two halves answer differently: a window is a
        sliding window, so a resumer needs every row of it, while most of the
        state is dead at a boundary and says so through
        `StateField.in_checkpoint`.

        Three kinds of byte belong to neither and are left out: the padding
        the state's byte count is rounded up by, the slot's own tail
        alignment, and the entry's interleave padding — rows no `ring_row`
        reaches, so no window is missing anything without them
        (`UnifiedPoolGeometry.entry_row_runs`).

        A property of the layout, not of any one slot, so it is computed once.
        """
        if self._checkpoint_range_cache is None:
            self._assert_ratios_divide_the_alignment()
            geo = self.pool_geometry
            # Rows, so the same for every plane; only the width they are
            # priced at differs.
            window_runs = geo.entry_row_runs()
            self._checkpoint_range_cache = [
                # The last state field can end exactly where the entry begins.
                merge_abutting(
                    [
                        *checkpoint_ranges_for(fields),
                        *(
                            ((geo.arena_rows + start) * width, count * width)
                            for start, count in window_runs
                        ),
                    ]
                )
                for fields, width in zip(
                    self._arena_planes, self._plane_row_widths(), strict=True
                )
            ]
        return self._checkpoint_range_cache

    def _checkpoint_segment_sizes(self) -> list[int]:
        """The image as the copy planner reads it: one size per source segment.

        The one place the per-plane ranges are flattened. Sizing wants their
        total and the planner wants the list, and the two answering from
        different comprehensions is how an image gets priced at one shape and
        cut at another.
        """
        return [
            nbytes for ranges in self._checkpoint_slot_ranges() for _, nbytes in ranges
        ]

    def checkpoint_image_bytes(self) -> int:
        """Bytes one checkpoint image holds. Priced before the pool exists."""
        return sum(self._checkpoint_segment_sizes())

    def _assert_ratios_divide_the_alignment(self) -> None:
        """A checkpoint boundary has to be a compression boundary too.

        `StateField.in_checkpoint` says a compressor without overlap owes a
        checkpoint nothing, because the first pool at or after the boundary
        starts exactly on it. That holds only while every ratio divides the
        quantity a checkpoint is aligned to, and HCA's ratio is 128 — let the
        alignment drop under that and HCA silently starts needing rows it is
        no longer given. No crash, just a resumer reading stale KV for its
        first pool, which reads as a small accuracy loss and nothing else.

        The quantity is `BlockManager`'s `hash_block_size`, not this class's
        own `block_size`: the ladder rounds a checkpoint to the prefix-cache
        hash granularity, which is `kv_cache_block_size * dcp_world_size`.

        The ratios come from the model rather than from the two constants this
        file happens to name, and that is what makes the guard reachable at
        all. `config.py` forces `kv_cache_block_size` to 256 for every
        `DeepseekV4*` architecture, so the alignment is always a multiple of
        4 and of 128 and a check written against `CSA_RATIO`/`HCA_RATIO` could
        never fire — it would only restate what the config already pins. What
        is genuinely free to change is `hf_config.compress_ratios`: a variant
        that pools on some other stride is the edit that breaks the premise,
        and this is what catches it.
        """
        config = self.model_runner.config
        # Raw, like `BlockManager.__init__` reads it: a zero would give an
        # alignment of zero, which every ratio divides, so a `or 1` here would
        # answer a question about a pool geometry that cannot exist.
        alignment = int(config.kv_cache_block_size) * int(
            config.decode_context_parallel_size
        )
        # Its own check, not folded into the one below: every ratio divides
        # zero, so a non-positive alignment reaches that test with nothing to
        # report and would raise saying it is not a multiple of `[]`.
        if alignment <= 0:
            raise ValueError(
                f"a checkpoint aligns to {alignment} tokens "
                "(kv_cache_block_size x decode_context_parallel_size), which "
                "is not a pool geometry that can exist"
            )
        bad = sorted({r for r in self.compress_ratios if r > 0 and alignment % r})
        if bad:
            raise ValueError(
                f"a checkpoint aligns to {alignment} tokens "
                f"(kv_cache_block_size x decode_context_parallel_size), which "
                f"is not a multiple of compress ratios {bad}, so a checkpoint "
                "boundary is not a compression boundary, so what "
                "`_state_fields` leaves out of the image is no longer dead"
            )

    def _invalidate_pool_caches(self) -> None:
        """Forget everything derived from the pools' layout or addresses.

        A slot's address is a function of the *split*, not just of its group
        (`UnifiedPoolGeometry.physical_slot` counts back from the topmost
        position), so a re-carve moves every one of them. Whoever wires an
        elastic pool has to call this; it is here so that is one line rather
        than a list of fields to remember.

        `_page_unit_region_cache` is deliberately not in it: half of what it
        holds comes from pools this method's caller does not own, so being on
        the list would make it look covered when it is not. It keys on its own
        addresses instead.
        """
        self._slot_view_cache = None
        self._checkpoint_range_cache = None
        self._checkpoint_plan_cache = None
        self._checkpoint_slot_base_cache = None
        self._checkpoint_descriptor = None

    def warmup_per_req_cache(self) -> None:
        """Run one checkpoint copy now, so the first real one is only a copy.

        `execute_paged_state_copies` is reachable only from `build()`, so
        everything it builds lazily -- the copy plan, the slot views, the slot
        base table, the tiling's upload, the pinned descriptor, and the Triton
        JIT of `_copy_tiles_kernel` -- otherwise lands inside the batch of
        whichever request first crosses a rung. Hundreds of milliseconds, once,
        on one unlucky request.

        Slot 0 into the pool's first units. Both are real addresses, which is
        the point: a warmup on scratch would compile a kernel and fill nothing.
        The bytes it writes are read by nobody -- a KV block is written before
        it is read, and this runs before any block has been handed out.
        """
        if self.model_runner.state_runtime.checkpoint_spec is None:
            return
        plan = self._checkpoint_copy_plan()
        if not plan.num_spans:
            return
        units = self.model_runner.state_runtime.checkpoint_spec.units_per_checkpoint
        staging = self._checkpoint_descriptor_buffer()
        plan.write_descriptor(
            staging.np[: plan.num_spans],
            self._checkpoint_slot_bases()[:1],
            self._page_unit_bases([list(range(units))]),
        )
        launch_copy_descriptor(staging.copy_to_gpu(plan.num_spans), plan)

    def _checkpoint_descriptor_buffer(self) -> CpuGpuBuffer:
        """Pinned staging for a step's whole descriptor, sized for the worst step.

        Pinned because the alternative synchronizes: a pageable H2D from
        `build()` makes the host wait out the forward already enqueued, which
        measured 2.9 ms behind 4 ms of work against 0.1 ms staged. Reused
        because allocating pinned memory is itself a synchronizing call.

        A step can carry at most one store and one restore per sequence, so
        two per Active Slot bounds it -- 1.6 MB at the shipped geometry. The
        caller checks that bound rather than growing on demand: a descriptor
        that did not fit would otherwise be silently truncated into a copy of
        the wrong shape.

        The store half is held by `PagedStateCheckpointCoordinator._supersede`,
        which keeps one pending boundary per sequence -- see the longer note on
        `GDNStateMixin._checkpoint_descriptor_buffer`.
        """
        if self._checkpoint_descriptor is None:
            plan = self._checkpoint_copy_plan()
            max_ops = 2 * int(self.model_runner.config.max_num_seqs)
            self._checkpoint_descriptor = CpuGpuBuffer(
                max_ops * plan.num_spans,
                3,
                dtype=torch.int64,
                device=self._kv_planes()[0].device,
            )
        return self._checkpoint_descriptor

    def _checkpoint_copy_plan(self) -> SegmentedCopyPlan:
        """Where a slot's checkpoint ranges meet a whole image's PAGE regions.

        Both streams are geometry. The ranges come from the layout, and every
        image is `units_per_checkpoint` units of identical region sizes —
        `_validate_paged_state_op` refuses anything else. So the cut points are
        the same for every store and every restore this worker will ever do,
        and the walk that finds them runs once instead of once an op.
        """
        if self._checkpoint_plan_cache is None:
            spec = self.model_runner.state_runtime.checkpoint_spec
            self._checkpoint_plan_cache = plan_segmented_copy(
                self._checkpoint_segment_sizes(),
                # Sizes from the same array `_page_unit_bases` takes addresses
                # from, tiled the way it ravels. Spelling the destination
                # stream out a second time here would let the two orders
                # diverge, and a plan cut against one order and addressed
                # through the other lands whole regions in the wrong unit.
                self._page_unit_stream_sizes(spec.units_per_checkpoint),
                spec.image_bytes,
            )
        return self._checkpoint_plan_cache

    def _checkpoint_slot_bases(self) -> np.ndarray:
        """`[group, segment]` start address of every source segment of a copy.

        One row per pool group, segments in the order
        `_checkpoint_slot_ranges` walks the planes — which is the order
        `_checkpoint_copy_plan` built the source stream in, so a plan's
        segment indices address a row of this directly.

        Materialized rather than recomputed because a group's slot sits at a
        fixed address for the pool's whole life, which leaves the entire
        per-op source side as one row lookup.
        """
        if self._checkpoint_slot_base_cache is None:
            self._checkpoint_slot_base_cache = np.array(
                [
                    [
                        view.data_ptr() + offset
                        for view, ranges in zip(
                            views, self._checkpoint_slot_ranges(), strict=True
                        )
                        for offset, _ in ranges
                    ]
                    for views in self._slot_views()
                ],
                dtype=np.int64,
            )
        return self._checkpoint_slot_base_cache

    def _page_unit_regions(self) -> tuple[np.ndarray, np.ndarray]:
        """Base address and per-block stride of every region a PAGE id owns.

        Blocks sit back to back in every pool, so a block's stride there is
        its size and its address is `base + block_id * num_bytes` — affine,
        and a property of the pools rather than of any block, so it is worked
        out once. Slicing the tensors instead, which is what this replaced,
        built 22 views per unit per op and threw them away to learn addresses
        that are one multiplication each.

        Two arrays rather than a list of pairs because both callers want
        columns: one multiplies the strides by an id, the other tiles them.

        The contiguity a slice would have been checked for is asked here
        instead: once of the layout, rather than every time of a slice.

        Keyed on the addresses it was built from rather than cleared by
        `_invalidate_pool_caches`. Half of these come from the KV planes,
        which that hook covers, and half from the indexer pools, which
        `allocate_kv_cache_tensors` owns and it does not -- so the hook would
        be an invariant this cache cannot check and the next reader cannot
        see. The key is the two planes and the one or two indexer pools, so
        finding out costs four `data_ptr()`s against a copy path measured in
        milliseconds, and the
        failure it removes is a scatter into whatever the allocator handed
        that address range to next.
        """
        planes = self._kv_planes()
        pools = [pool for pool, _ in self._indexer_page_pools()]
        owners = tuple(t.data_ptr() for t in (*planes, *pools))
        # The whole test: there is always at least one plane, so the owners of
        # a built cache are never the empty tuple this starts as, and an
        # `is None` beside this would be a second condition that cannot differ
        # from it.
        if self._page_unit_region_owners != owners:
            geo = self.pool_geometry
            bases, strides = [], []
            for plane, width in zip(planes, self._plane_row_widths(), strict=True):
                if not plane.is_contiguous():
                    raise RuntimeError("a KV plane must be contiguous to be copied")
                bases.append(plane.data_ptr())
                strides.append(geo.envelope_rows * width)
            for pool in pools:
                if not pool.is_contiguous():
                    raise RuntimeError(
                        "an indexer pool must be contiguous to be copied"
                    )
                per_layer = pool.stride(0) * pool.element_size()
                per_block = pool.stride(1) * pool.element_size()
                for layer in range(len(self.csa_layers)):
                    bases.append(pool.data_ptr() + layer * per_layer)
                    strides.append(per_block)
            self._page_unit_region_cache = (
                np.array(bases, dtype=np.int64),
                np.array(strides, dtype=np.int64),
            )
            self._page_unit_region_owners = owners
        return self._page_unit_region_cache

    def _page_unit_bases(self, unit_ids: Sequence[Sequence[int]]) -> np.ndarray:
        """Start address of every destination segment, one row per image.

        `unit_ids` is `(images, units_per_checkpoint)`. A unit's regions are
        each at `base + id * stride`, so one image's worth is an outer product
        and a batch's is the same product with an image axis in front. Unit
        major, region minor — the order `_checkpoint_copy_plan` built the
        destination stream in.
        """
        base, stride = self._page_unit_regions()
        ids = np.asarray(unit_ids, dtype=np.int64)
        return (base + ids[..., None] * stride).reshape(len(ids), -1)

    def _page_unit_stream_sizes(self, units: int) -> np.ndarray:
        """Bytes in each destination segment of an image of `units` units."""
        return np.tile(self._page_unit_regions()[1], units)

    def _slot_views(self) -> list[list[torch.Tensor]]:
        """Per-group views of that request's whole slot in each plane.

        Built once: a slot's rows are at a fixed address for the pool's whole
        life, and `_foreach_copy_` wants lists, so re-slicing per call would put
        host time on the batch-construction path for nothing.
        """
        if self._slot_view_cache is None:
            geo = self.pool_geometry
            self._slot_view_cache = [
                [
                    plane[slice(*geo.slot_span(geo.physical_slot(g)))]
                    for plane in self._kv_planes()
                ]
                for g in range(self.num_state_slots)
            ]
        return self._slot_view_cache

    def nope_row_bytes(self) -> int:
        """Bytes one row costs in the NoPE plane (fp8-packed or bf16)."""
        return self.head_dim * self._classical_dtype.itemsize

    def rope_row_bytes(self) -> int:
        """Bytes one row costs in the RoPE plane; 0 on a bf16 build, which
        keeps RoPE inline in the NoPE row and has no second plane."""
        return self.rope_head_dim * self._rope_dtype.itemsize if self._kv_fp8 else 0

    def plane_row_bytes(self) -> int:
        """What one row of the shared row space costs across both planes."""
        return self.nope_row_bytes() + self.rope_row_bytes()

    @property
    def num_state_slots(self) -> int:
        """Entries sizing gave the compressor-state class."""
        return self.model_runner.pool_plan.entries.get(STATE_SLOT_CLASS, 0)

    def rows_per_block(self, compress_ratio: int) -> int:
        """Rows one V4 block compresses to in a layer of this ratio.

        0 for a dense layer (ratio 0), which keeps no compressed KV at all —
        callers use that as the "skip this layer" signal.
        """
        return self._rows_per_block.get(compress_ratio, 0)

    def _indexer_block_bytes(self) -> int:
        """Bytes one V4 block costs in the CSA Indexer pool, all CSA layers.

        The indexer is the one classical pool outside the shared row space: it
        addresses `(block, row)` in its own dtype with a runtime block stride,
        so it never has to agree with anyone else's row width.
        """
        if self._indexer_fp4:
            # Packed E2M1 data (16 B per group of 32) plus one e8m0 scale byte,
            # in the two uint8 pools `pa_mqa_logits_fp4` reads.
            groups = self._idx_k_tiles * 4 * self.csa_rows_per_block
            per_layer = groups * 16 + groups
        else:
            # Data and scale are two REGIONS inside the block, not interleaved
            # per row: `[rows*index_head_dim FP8]` then `[rows*4 fp32 scale]`.
            # Written by `indexer_k_quant_and_cache`, read by
            # `cp_gather_indexer_k_quant_cache` and
            # `deepgemm_fp8_paged_mqa_logits`.
            per_layer = self.csa_rows_per_block * self._index_row_bytes
        return len(self.csa_layers) * per_layer

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """Two entry classes: a paged block, and a request's slot.

        Both are counted in rows of the shared row space and priced by
        `pool_geometry` — an envelope for a block, a slot for a request — and
        both counts are independent of how the pool is split, which is what
        lets sizing quote them before it has chosen a split.

        The slot is one class rather than two even though it holds two
        unrelated things, the compressor state and the sliding windows. They
        are allocated and given up together and no request can have one without
        the other, so pricing them apart would only invite a split that cannot
        happen. One slot per request regardless of MTP: the draft lookahead is
        absorbed into `win_with_spec` rather than multiplying anything.

        No flat margin on the slot: a ring cannot transiently exceed itself the
        way a block-addressed window could while sliding across a boundary, and
        there is no admission-vs-materialization gap to cushion — a slot exists
        for its request's whole life.
        """
        geo = self.pool_geometry
        row_bytes = self.plane_row_bytes()
        return [
            page_pool(geo.block_bytes(row_bytes) + self._indexer_block_bytes()),
            state_pool(STATE_SLOT_CLASS, geo.slot_bytes(row_bytes), entries_per_req=1),
        ]

    def _plane_row_widths(self) -> list[int]:
        """Bytes one row costs in each plane this build has.

        Two on fp8 (packed NoPE and bf16 RoPE), one on bf16, which keeps RoPE
        inline in the NoPE row. The single answer to how many planes there are:
        sizing, the carve, the compressor state's split and the checkpoint copy
        all count them from here, so none of them can disagree.
        """
        return [w for w in (self.nope_row_bytes(), self.rope_row_bytes()) if w]

    def _kv_planes(self) -> list[torch.Tensor]:
        """The allocated plane tensors, in the order `_plane_row_widths` lists."""
        runner = self.model_runner
        planes = [
            p for p in (runner.v4_kv_plane, runner.v4_kv_plane_rope) if p is not None
        ]
        assert len(planes) == len(self._plane_row_widths())
        return planes

    def _indexer_page_pools(self) -> list[tuple[torch.Tensor, str]]:
        """Indexer pools and semantic-role prefixes in PAGE stream order.

        FP8 keeps data and its row scales in one tensor. FP4 stores packed
        data and e8m0 scales in separate block-indexed tensors, and both are
        required to restore indexer logits bit-for-bit. This is the single
        enumeration used by the internal PAGE checkpoint copier and external
        transfer/offload registration so neither path can omit a pool.
        """
        runner = self.model_runner
        if self._indexer_fp4:
            return [
                (runner.v4_csa_idx_kv, "dsv4.csa_indexer.fp4_data"),
                (
                    runner.v4_csa_idx_kv_scale,
                    "dsv4.csa_indexer.fp4_scale",
                ),
            ]
        return [(runner.v4_csa_idx_kv, "dsv4.csa_indexer")]

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict[str, torch.Tensor]:
        """Allocate KV pools that depend only on `num_blocks`.

        After Phase A (CG-friendly indexer), the SWA window AND the per-layer
        compressed pool are physically merged into a single `unified_kv`
        tensor per layer (allocated in `allocate_per_req_cache`, which is
        called later when both `num_blocks` and `num_slots` are known).

        Only the CSA Indexer FP8 cache stays as a standalone batched tensor
        — it lives in its own dtype (FP8 + fp32 scale) and is consumed by
        `cp_gather_indexer_k_quant_cache`, not the sparse-attn kernel.
        Layer-major axis order `[n_csa, NB, csa_rows_per_block,
        index_row_bytes]` so each per-CSA slice `pool[pos]` is contiguous in
        storage; the kernel infers `block_size` from `kv_cache.shape[1]`.
        """
        runner = self.model_runner
        device = runner.device
        num_blocks = runner.num_physical_kvcache_blocks
        n_csa = len(self.csa_layers)
        if self._indexer_fp4:
            # FP4 indexer cache: packed E2M1 data + e8m0 scale in the
            # `pa_mqa_logits_fp4` preshuffle layout. Two uint8 pools, both
            # layer-major so each per-CSA slice `pool[pos]` is contiguous.
            kt = self._idx_k_tiles
            return {
                "v4_csa_idx_kv": torch.zeros(
                    (n_csa, num_blocks, kt, 4, self.csa_rows_per_block, 16),
                    dtype=torch.uint8,
                    device=device,
                ),
                "v4_csa_idx_kv_scale": torch.zeros(
                    (n_csa, num_blocks, kt, 4, self.csa_rows_per_block),
                    dtype=torch.uint8,
                    device=device,
                ),
            }
        return {
            "v4_csa_idx_kv": torch.zeros(
                (n_csa, num_blocks, self.csa_rows_per_block, self._index_row_bytes),
                dtype=dtypes.fp8,
                device=device,
            ),
        }

    def allocate_per_req_cache(self, entries: dict[str, int]) -> dict[str, object]:
        """Carve the two KV planes + the compressor arena out of one allocation.

        One `torch.zeros` holds every per-request pool, in this order:

            [NoPE plane]  plane_rows x head_dim, fp8-packed or bf16
            [RoPE plane]  plane_rows x rope_head_dim, bf16, fp8 KV only
            [arena]       compressor state, one entry per request

        `plan_regions` places them, so each starts on the boundary the arena's
        own fields assume.

        Both planes materialize the SAME row space (`v4_pool_geometry`): row `I`
        is one token at one layer in both, which is what a single `kv_indices`
        buffer feeding two pools means. Compressed blocks grow from row 0 and a
        request's windows are numbered back from the last row, so neither side's
        address mentions where the other stops — the split is a pair of host-side
        counters and moving it re-carves nothing.

        A layer's view is its base row to the end of the plane. Views therefore
        OVERLAP, and every one of them reaches both regions through one base
        pointer. That is the point: `layer_base_row` cancels the layer term out
        of both address formulas, so one index buffer serves a whole compress
        class.

        Compressor state comes from a `StateArena` instead of six standalone
        tensors: the per-layer views are unchanged in shape and dtype, but a
        request's whole state is one contiguous byte range, which is what
        checkpointing, entry relocation and RDMA all need. It stays out of the
        row space because it is not organized by rows.

        Everything is setattr'd onto ModelRunner; `v4_unified_kv` is a list
        of per-layer views (length `num_layers`).
        """
        num_slots = entries.get(STATE_SLOT_CLASS, 0)
        assert self._swa_dtype == self._classical_dtype, (
            "unified_kv requires SWA dtype == classical KV dtype "
            f"(got SWA={self._swa_dtype}, classical={self._classical_dtype}). "
            "fp8 path must set both to dtypes.fp8 (rope lives in a separate "
            "bf16 pool); a genuine mismatch corrupts the unified layout."
        )
        device = self.model_runner.device
        num_blocks = self.model_runner.num_physical_kvcache_blocks
        head_dim = self.head_dim
        dtype = self._swa_dtype
        rope_dtype = self._rope_dtype

        # Anything already worked out from the old layout or the old pools is
        # now wrong, and wrong quietly: four of these hold raw addresses, so a
        # stale one is a copy to the wrong slot rather than a crash. Cleared
        # here, before `pool_geometry` is replaced, so that nothing between
        # this line and that one can answer from the old split.
        #
        # Below that line they refill freely, and the cross-check further down
        # depends on it: `checkpoint_image_bytes()` is what re-derives the
        # ranges, and it has to derive them from the geometry just installed.
        # What is not allowed is a *second* clear after that point — it would
        # throw away the answer the check just validated.
        self._invalidate_pool_caches()
        # The layout at the split sizing chose. Everything below — and every
        # index formula any kernel evaluates — reads its offsets from here.
        geo = self.pool_geometry.with_capacity(num_blocks, num_slots)
        self.pool_geometry = geo

        actual_page_bytes = (
            sum(geo.block_bytes(width) for width in self._plane_row_widths())
            + self._indexer_block_bytes()
        )
        actual_slot_bytes = sum(
            geo.slot_bytes(width) for width in self._plane_row_widths()
        )
        state_runtime = self.model_runner.state_runtime
        checkpoint_spec = state_runtime.checkpoint_spec
        if checkpoint_spec is None:
            raise RuntimeError("DSV4 PAGE/state checkpoint sizing spec is missing")
        layout_id = state_runtime.transfer.paged_layout_id
        actual_image_bytes = self.checkpoint_image_bytes()
        if (
            actual_page_bytes != checkpoint_spec.page_unit_bytes
            or actual_slot_bytes != checkpoint_spec.slot_bytes
            or actual_image_bytes != checkpoint_spec.image_bytes
            or layout_id != checkpoint_spec.layout_id
        ):
            raise RuntimeError(
                "DSV4 PAGE/state checkpoint geometry differs from sizing: "
                f"page={actual_page_bytes}/{checkpoint_spec.page_unit_bytes}, "
                f"slot={actual_slot_bytes}/{checkpoint_spec.slot_bytes}, "
                f"image={actual_image_bytes}/{checkpoint_spec.image_bytes}, "
                f"layout={layout_id!r}/{checkpoint_spec.layout_id!r}"
            )

        row_widths = self._plane_row_widths()
        offsets, total_bytes = plan_regions([geo.plane_bytes(w) for w in row_widths])

        # Zeroed once, which is also what `StateArena`'s `buf` contract asks
        # for. Nothing holds the pool but the carved views — they keep the
        # allocation alive, so it must not be dropped from any of them.
        per_req_pool = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

        def _plane(start: int, width: int, elem: torch.dtype) -> torch.Tensor:
            end = start + geo.plane_rows * width * elem.itemsize
            return per_req_pool[start:end].view(elem).view(geo.plane_rows, width)

        kv_plane = _plane(offsets[0], head_dim, dtype)
        # 2buff fp8: a second plane of the same rows at the RoPE width, bf16
        # (RoPE is never quantized). bf16 builds keep RoPE inline in the NoPE
        # row and have no second plane.
        kv_plane_rope = (
            _plane(offsets[1], self.rope_head_dim, rope_dtype) if self._kv_fp8 else None
        )

        # A plain slice, not `as_strided`: the base row is a row count into the
        # plane, and letting torch derive the storage offset is what keeps it
        # from being confused with an absolute one. A layer carrying its window
        # as a state field anchors nowhere in the row space and gets None —
        # `build_kv_cache_tensor` binds it the retyped field view instead.
        def _layer_views(plane: torch.Tensor | None) -> list[torch.Tensor | None]:
            if plane is None:
                return [None] * self.num_layers
            return [
                (
                    None
                    if i in self._field_window_layers
                    else plane[geo.layer_base_row(i) :]
                )
                for i in range(self.num_layers)
            ]

        unified_kv = _layer_views(kv_plane)
        unified_kv_rope = _layer_views(kv_plane_rope)

        # ---- Compressor state: the front rows of every slot ----------------
        # Same per-layer views the kernels bound before. What changed is where
        # they live: a slot's state now sits in the planes with that request's
        # windows, so the bytes are on the elastic line rather than in a region
        # sized once at startup. Each plane holds the fields `plan_field_planes`
        # gave it, strided by the whole slot; only the top `num_slots`
        # positions are the pool's, and the rest of the plane is blocks.
        arena = SplitStateArena(
            [
                StateArena(
                    fields,
                    geo.slot_positions,
                    device,
                    buf=per_req_pool[start : start + geo.plane_bytes(row_bytes)],
                    slot_stride=geo.slot_rows * row_bytes,
                    live_entries=num_slots,
                )
                for fields, start, row_bytes in zip(
                    self._arena_planes, offsets, row_widths, strict=True
                )
            ]
        )

        # ---- RDMA staging pool, only allocated in PD disaggregation mode --
        is_pd = _uses_pd_staging(
            getattr(self.model_runner.config, "kv_transfer_config", None)
        )
        state_slot_stride = arena.entry_bytes // self._state_dtype.itemsize
        if is_pd:
            pool_size = int(os.environ.get("ATOM_PD_STAGING_POOL", "32"))
            state_pool = torch.zeros(
                pool_size * state_slot_stride,
                dtype=self._state_dtype,
                device=device,
            )
        else:
            pool_size = 0
            state_pool = torch.empty(0, dtype=self._state_dtype, device=device)

        return {
            "v4_state_arena": arena,
            "v4_kv_plane": kv_plane,
            "v4_kv_plane_rope": kv_plane_rope,
            "v4_unified_kv": unified_kv,
            "v4_unified_kv_rope": unified_kv_rope,
            "v4_csa_main_kv_state": arena.view("csa_main_kv"),
            "v4_csa_main_score_state": arena.view("csa_main_score"),
            "v4_csa_idx_kv_state": arena.view("csa_idx_kv"),
            "v4_csa_idx_score_state": arena.view("csa_idx_score"),
            "v4_hca_main_kv_state": arena.view("hca_main_kv"),
            "v4_hca_main_score_state": arena.view("hca_main_score"),
            "v4_state_pool": state_pool,
            "v4_state_pool_size": pool_size,
            "v4_state_slot_stride": state_slot_stride,
        }

    def _compress_block_view(
        self, plane: torch.Tensor, layer_id: int, width: int
    ) -> torch.Tensor:
        """A layer's compressed rows as `[num_blocks, block_rows, width]`.

        Reached by reshaping the plane's envelope region rather than by
        `as_strided`, so the storage offset stays torch's to derive — the one
        arithmetic this repo has got wrong three times (`storage_offset` is
        absolute, and a plane view is never at offset zero).
        """
        geo = self.pool_geometry
        base = geo.layer_base_row(layer_id)
        rows = geo.layer_class(layer_id).block_rows
        num_blocks = self.model_runner.num_physical_kvcache_blocks
        envelopes = plane[: num_blocks * geo.envelope_rows]
        return envelopes.view(num_blocks, geo.envelope_rows, width)[
            :, base : base + rows
        ]

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind V4 modules' state-cache + classical-cache views.

        Called by ModelRunner.allocate_kv_cache() for every nn.Module:
          - V4 Attention: bind swa_kv (per_req_cache pool).
          - V4 Compressor: bind kv_state, score_state (per_req_cache pool)
            AND kv_cache (classical pool slice — per CSA/HCA layer).
          - V4 Indexer:    bind kv_cache (csa_idx_kv slice — per CSA layer).

        Returns None always — V4 forward consumes module attributes directly,
        not the global `forward_context.kv_cache_data` registry that ATOM's
        standard MHA path uses.
        """
        # Local imports to avoid circular dependency at module load time.
        from atom.models.deepseek_v4 import Compressor as _V4Compressor
        from atom.models.deepseek_v4 import DeepseekV4Attention as _V4Attention
        from atom.models.deepseek_v4 import Indexer as _V4Indexer

        runner = self.model_runner
        geo = self.pool_geometry

        if isinstance(module, _V4Attention):
            # A layer that declared its own window KV dtype is carried as a
            # state field instead of as rows of the entry — see
            # `_state_fields`. It still reads through a plane and a
            # `WindowParams`, so nothing below it in the stack can tell: what
            # changes is that the plane is retyped to the layer's own width,
            # where its ring is contiguous.
            if module.layer_id in self._field_window_layers:
                module.swa_plane = self._window_field_plane()
                module.swa_window = self._window_field_params(module.layer_id)
                # No second plane and no packing: this window is whatever
                # dtype the layer asked for, RoPE inline, one plane.
                module.kv_fp8 = False
                module.unified_kv = None
                module.unified_kv_rope = None
                module.swa_plane_rope = None
                return None
            # `unified_kv` is this layer's view of the plane — its base row to
            # the plane's end, so it reaches both the compressed blocks and the
            # windows through one base pointer. `swa_plane` is the same view:
            # the window rows are interleaved through the entry region, not a
            # prefix, so what a window write needs is the view plus the class's
            # `WindowParams`, not a slice.
            unified = runner.v4_unified_kv[module.layer_id]
            module.unified_kv = unified
            module.swa_plane = unified
            module.swa_window = geo.window_params(self.compress_ratios[module.layer_id])
            module.kv_fp8 = self._kv_fp8
            if self._kv_fp8:
                # 2buff: the second plane, same rows at the RoPE width. A row
                # index means the same token in both (asm decode op5 reads them
                # with one index buffer).
                rope = runner.v4_unified_kv_rope[module.layer_id]
                module.unified_kv_rope = rope
                module.swa_plane_rope = rope
            else:
                module.unified_kv_rope = None
                module.swa_plane_rope = None
            return None

        if isinstance(module, _V4Indexer):
            # Indexer.kv_cache — CSA Indexer compressed pool, per CSA layer.
            # prefix: "layers.<L>.attn.indexer"
            #
            # Shape MUST stay [NB, csa_rows_per_block, index_row_bytes] (3D,
            # row-count dim explicit) because `cp_gather_indexer_k_quant_cache`
            # infers block_size from `kv_cache.shape[1]` to compute
            # `physical_block * block_size + slot_in_block`. Flattening to
            # [NB*rows, 1, index_row_bytes] makes the kernel see block_size=1
            # and OOB-index block_table. Matches V3.2's [num_blocks,
            # block_size, head_dim] layout (deepseek_v2.py:1049).
            layer_id_from_prefix = int(module.prefix.split(".")[1])
            pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
            module.kv_cache = runner.v4_csa_idx_kv[pos]
            # Re-assert the authoritative FP4 flag (config + gfx950 fallback) onto
            # the Indexer. Its forward_pre/indexer_score_topk branch on this
            # STABLE bool rather than kv_cache.dtype, so a traced piece and the
            # eager op can never disagree (see Indexer._indexer_fp4).
            module._indexer_fp4 = bool(self._indexer_fp4)
            if self._indexer_fp4:
                # FP4: separate e8m0 scale pool consumed by the
                # `pa_mqa_logits_fp4` kernels alongside `kv_cache`.
                module.kv_scale = runner.v4_csa_idx_kv_scale[pos]
            return None

        if isinstance(module, _V4Compressor):
            # Compressor.prefix is set by the parent constructor:
            #   "layers.<L>.attn.compressor"          -> CSA Main / HCA Main
            #   "layers.<L>.attn.indexer.compressor"  -> CSA Indexer's inner
            parts = module.prefix.split(".")
            layer_id_from_prefix = int(parts[1])
            is_indexer_inner = "indexer" in parts
            ratio = module.compress_ratio

            if is_indexer_inner:
                assert ratio == 4, "Indexer-inner Compressor only on CSA layers"
                pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_csa_idx_kv_state[pos]
                module.score_state = runner.v4_csa_idx_score_state[pos]
                # Inner compressor writes target the SAME storage as the
                # outer Indexer.kv_cache (csa_idx_kv). Same 3-D FP8 shape
                # — `Compressor.forward` resolves
                # slot via block_table+ci internally (no flat slot_mapping
                # needed; matches CSA Main's path).
                idx_kv = runner.v4_csa_idx_kv[pos]
                module.kv_cache = idx_kv
                if self._indexer_fp4:
                    # FP4 path: bind the matching uint8 e8m0 scale pool.
                    # `fused_compress_attn(quant_mode="fp4")` writes both in
                    # the `pa_mqa_logits_fp4` preshuffle layout.
                    module.cache_scale = runner.v4_csa_idx_kv_scale[pos]
                else:
                    # FP8 quant path: bind a strided fp32 view of the per-block
                    # scale region. Layout per block: [rows*head_dim FP8] then
                    # [rows fp32 scale], exactly filling the block
                    # (cache_kernels.cu:1209-1239). Strides in fp32 elements.
                    nb, rows, row_bytes = idx_kv.shape
                    head_dim = self.index_head_dim
                    block_bytes = rows * row_bytes
                    assert (
                        block_bytes % 4 == 0
                    ), f"per-block bytes ({block_bytes}) must be 4-aligned"
                    block_fp32_stride = block_bytes // 4
                    scale_fp32_offset = (rows * head_dim) // 4
                    # `as_strided(storage_offset=...)` is ABSOLUTE in the underlying
                    # storage, NOT relative to `idx_kv`. Since idx_kv =
                    # v4_csa_idx_kv[pos] carries its own storage_offset (pos *
                    # block_span), it MUST be added here — otherwise every CSA
                    # layer's `cache_scale` aliases pos 0's scale region, so only
                    # the first CSA layer's indexer reads valid scale and all other
                    # layers read zeros (FP8 indexer logits collapse at long
                    # context). The FP4 path is unaffected: it binds a real per-pos
                    # tensor (v4_csa_idx_kv_scale[pos]).
                    idx_kv_f32 = idx_kv.view(torch.float32)
                    module.cache_scale = idx_kv_f32.view(-1).as_strided(
                        size=(nb, rows),
                        stride=(block_fp32_stride, 1),
                        storage_offset=idx_kv_f32.storage_offset() + scale_fp32_offset,
                    )
                # Indexer-inner cache is always fp8 (independent of
                # kv_cache_dtype); it has no separate rope pool.
                module.quant_mode = "fp4" if self._indexer_fp4 else "per_row_fp8"
                module.kv_cache_rope = None
            elif ratio in (CSA_RATIO, HCA_RATIO):
                table = (
                    self.layer_id_to_csa_pos
                    if ratio == CSA_RATIO
                    else self.layer_id_to_hca_pos
                )
                pos = table[layer_id_from_prefix]
                if ratio == CSA_RATIO:
                    module.kv_state = runner.v4_csa_main_kv_state[pos]
                    module.score_state = runner.v4_csa_main_score_state[pos]
                else:
                    module.kv_state = runner.v4_hca_main_kv_state[pos]
                    module.score_state = runner.v4_hca_main_score_state[pos]
                # `Compressor.forward` writes `kv_cache[block, row, :] = entry`,
                # so it wants a [num_blocks, rows, width] view. This layer's
                # rows are a run inside every envelope, so the view is strided
                # by the envelope and the writes are scattered — which is what
                # `fused_compress` already takes `kv_cache.stride(0)/(1)` for.
                module.kv_cache = self._compress_block_view(
                    runner.v4_kv_plane, layer_id_from_prefix, self.head_dim
                )
                if self._kv_fp8:
                    module.kv_cache_rope = self._compress_block_view(
                        runner.v4_kv_plane_rope,
                        layer_id_from_prefix,
                        self.rope_head_dim,
                    )
                    module.quant_mode = "group_fp8"
                else:
                    module.kv_cache_rope = None
                    module.quant_mode = "none"
            else:
                raise ValueError(
                    f"Unknown V4 compress_ratio={ratio} on Compressor at "
                    f"prefix={module.prefix!r}"
                )
            return None

        return super().build_kv_cache_tensor(layer_id, module)

    def get_kv_transfer_tensors(self):
        """Describe V4's compressed PAGE and full per-request SLOT storage.

        ``block_regions`` are forward-indexed compressed PAGE units: one unit
        from each shared KV plane plus each CSA indexer region.
        ``swa_block_regions`` is a legacy field name; each reverse-indexed unit
        is one complete request SLOT, including compressor state and SWA rows.
        ``staging_region`` and ``gather_slot`` describe only compressor-state
        PD staging and must never be used as the source of a SLOT sidecar.
        """
        from atom.kv_transfer.disaggregation.types import (
            KVTransferRegion,
            KVTransferTensors,
        )

        runner = self.model_runner
        if not hasattr(runner, "v4_unified_kv"):
            return None

        # `get_kv_transfer_tensors` is called unconditionally on every
        # `allocate_kv_cache`; returning None means "no transfer region."
        # Standalone LMCache offload can carry both FP4 indexer pools, but PD
        # connectors have a separate producer/consumer region contract which
        # has not been extended to the FP4 scale pool yet.
        transfer_config = getattr(runner.config, "kv_transfer_config", None)
        transfer_active = bool(transfer_config)
        if self._indexer_fp4 and transfer_active and _uses_pd_staging(transfer_config):
            raise NotImplementedError(
                "DeepSeek-V4 PD transfer with --index_cache_dtype fp4 is "
                "unsupported; standalone LMCache offload supports FP4, but "
                "Mooncake/Moriio producer-consumer staging does not yet map "
                "the separate FP4 indexer scale pool."
            )
        if transfer_active and getattr(runner.config, "pipeline_parallel_size", 1) > 1:
            raise NotImplementedError(
                "DeepSeek-V4 KV transfer/PD and sidecar offload with pipeline "
                "parallelism (pipeline_parallel_size > 1) is unsupported "
                "because each PAGE/SLOT plane region covers every layer; use "
                "PP=1 or disable DeepSeek-V4 transfer/offload."
            )
        if self._indexer_fp4 and not transfer_active:
            # No connector will consume the regions in a single-node run.
            return None

        num_slots = self.num_state_slots
        geo = self.pool_geometry
        elem_fp32 = 4

        block_regions: list[KVTransferRegion] = []
        swa_block_regions: list[KVTransferRegion] = []
        slot_regions: list[KVTransferRegion] = []

        # Compressed PAGE: one region per plane, not per layer. A block's rows
        # are one envelope — every layer of it, contiguous — so the transfer
        # unit the connector already zips over (`base + id * unit_bytes`,
        # `unit_bytes` long) describes it exactly. Under the layer-major
        # predecessor a layer's rows for a block were the contiguous thing and
        # the envelope was the strided one.
        #
        # A stage that holds a subset of the layers still holds whole envelopes
        # of that subset, so its plane is its own; `_consumer_region_map`'s
        # per-layer alignment has nothing left to align, which is why PP is
        # rejected below.
        plane_roles = [
            role
            for role, row_bytes in (
                ("dsv4.main_kv.nope", self.nope_row_bytes()),
                ("dsv4.main_kv.rope", self.rope_row_bytes()),
            )
            if row_bytes
        ]
        planes = list(
            zip(
                self._kv_planes(),
                self._plane_row_widths(),
                plane_roles,
                strict=True,
            )
        )
        for plane, row_bytes, role in planes:
            block_regions.append(
                KVTransferRegion(
                    plane.data_ptr(),
                    runner.num_physical_kvcache_blocks * geo.block_bytes(row_bytes),
                    geo.block_bytes(row_bytes),
                    semantic_role=role,
                )
            )

        # Compressed PAGE regions: one per CSA layer in each indexer pool.
        # The pool order is data then scale for FP4, matching the internal
        # checkpoint stream. Derive the block width from the actual view so
        # this remains correct for both the FP8 row layout and FP4 tiles.
        for pool, role_prefix in self._indexer_page_pools():
            for pos, layer_id in enumerate(self.csa_layers):
                view = pool[pos]
                if not view.is_contiguous():
                    raise RuntimeError(
                        "a CSA indexer layer must be contiguous to be transferred"
                    )
                block_regions.append(
                    KVTransferRegion(
                        view.data_ptr(),
                        view.numel() * view.element_size(),
                        view.stride(0) * view.element_size(),
                        semantic_role=f"{role_prefix}.layer_{layer_id}",
                    )
                )

        checkpoint_spec = runner.state_runtime.checkpoint_spec
        if checkpoint_spec is None:
            raise RuntimeError("DSV4 PAGE/state checkpoint sizing spec is missing")
        transfer_page_bytes = sum(region.unit_bytes for region in block_regions)
        if transfer_page_bytes != checkpoint_spec.page_unit_bytes:
            raise RuntimeError(
                "DSV4 PAGE transfer regions do not cover the sized PAGE unit: "
                f"regions={transfer_page_bytes}, "
                f"checkpoint={checkpoint_spec.page_unit_bytes}"
            )

        # Full per-request SLOT (legacy field name: `swa_block_regions`) is one
        # slot per plane — compressor state and SWA, every layer at once — and
        # the id the connector zips over is its pool group.
        #
        # A slot has no window-freeing and no sentinel rows: every row of a live
        # one travels. `reverse_indexed` is what the geometry's numbering costs
        # here: the pool hands out positions from the top down so that growing
        # it never relocates one, which puts group `g` at
        # `base + (num_slots - 1 - g) * unit` rather than `base + g * unit`.
        # `slot_span` takes a plane position, so the group has to be crossed
        # over first -- capacity can exceed the live slot count, and then the
        # two differ and the base lands inside the block region.
        slot_start, _ = (
            geo.slot_span(geo.physical_slot(num_slots - 1)) if num_slots else (0, 0)
        )
        for plane, row_bytes, role in planes:
            unit = geo.slot_bytes(row_bytes)
            swa_block_regions.append(
                KVTransferRegion(
                    plane.data_ptr() + slot_start * row_bytes,
                    num_slots * unit,
                    unit,
                    reverse_indexed=True,
                    semantic_role=role,
                )
            )

        # Compressor-only PD staging. It omits SWA rows, is managed separately
        # with pool acquire/release, and is invalid as a sidecar SLOT source.
        staging_region = None
        gather_slot = None
        scatter_slot = None
        if hasattr(runner, "v4_state_pool") and runner.v4_state_pool_size > 0:
            pool = runner.v4_state_pool
            stride = runner.v4_state_slot_stride
            pool_size = runner.v4_state_pool_size
            staging_region = KVTransferRegion(
                pool.data_ptr(),
                pool.numel() * elem_fp32,
                stride * elem_fp32,
                semantic_role="dsv4.pd_staging.compressor_state",
            )
            gather_slot = self._make_gather_slot(
                pool, stride, runner.v4_state_arena, self.pool_geometry
            )
            scatter_slot = self._make_scatter_slot(
                pool, stride, runner.v4_state_arena, self.pool_geometry
            )

        return KVTransferTensors(
            block_regions=block_regions,
            swa_block_regions=swa_block_regions,
            slot_regions=slot_regions,
            num_blocks=runner.num_physical_kvcache_blocks,
            num_slots=num_slots,
            expected_full_slot_region_count=len(planes),
            staging_region=staging_region,
            staging_pool_size=pool_size if staging_region else 0,
            gather_slot=gather_slot,
            scatter_slot=scatter_slot,
        )

    # ------------------------------------------------------------------ #
    # CommonAttentionBuilder abstract methods (V4 forward consumes only  #
    # `positions`; other metadata is populated for forward parity with   #
    # the rest of ATOM and to support PR3-main multi-sequence wiring).   #
    # ------------------------------------------------------------------ #

    def _attach_v4_indexer_meta(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        scheduled_bs: int,
        total_tokens: int,
        positions_gpu=None,
        buf_prefix_ubatch: str = "",
    ) -> None:
        """Build and attach the CSA Indexer per-fwd GPU metadata.

        Hoists per-CSA-layer H2D calls (batch_id_per_token / cu_committed /
        n_committed / seq_base_per_token / cu_ends) into a single per-fwd
        build. None for warmup or empty fwd; `_build_v4_indexer_meta`
        handles both.

        ``buf_prefix_ubatch`` selects the ub{idx}_ prefixed cu_committed staging
        buffer so TBO ubatches don't collide on the shared global one.
        """
        attn_metadata.indexer_meta = self._build_v4_indexer_meta(
            attn_metadata=attn_metadata,
            positions_gpu=positions_gpu,
            scheduled_bs=scheduled_bs,
            total_tokens=total_tokens,
            device=self.device,
            buf_prefix_ubatch=buf_prefix_ubatch,
        )

    def _refresh_fp4_ragged_windows(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        ragged_lens,
        positions_gpu,
        meta: dict[str, Any],
    ) -> None:
        """Rebuild the FP4 ragged-decode varlen windows + persistent-grid schedule.

        Writes into the FIXED-ADDRESS buffers the captured scorer reads (`top_k`
        reads the windows, the mqa kernel reads `cta_info`), so a CUDAGraph replay
        sees this fwd's contents through stable pointers. Windows span ALL padded
        rows (`positions_gpu.shape[0]`); the aiter kernel leaves the tail past the
        real Σ empty, so pad rows cost 0 CTAs and yield top-k -1.

        `ragged_lens` is `[scheduled_bs]` per-seq query lengths: the real lengths
        on a ragged step, a uniform `next_n` fill on a rectangular one. Sync-free
        — real Σ is baked into `cu_seq_q` on device.
        """
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
            compute_prefill_schedule,
            compute_varqlen_windows,
        )

        _rbs = int(ragged_lens.shape[0])
        cu_seq_q = torch.zeros(_rbs + 1, dtype=torch.int32, device=ragged_lens.device)
        torch.cumsum(ragged_lens.to(torch.int32), dim=0, out=cu_seq_q[1:])
        # n_committed is running_bs-long (index_topk pad sentinels); slice to `_rbs`
        # so its length matches cu_seq_q (else the binary search over cu_seq_q
        # reads OOB / maps real rows onto sentinel pad seqs).
        _ncmt = attn_metadata.n_committed_csa_per_seq[:_rbs]
        _padded = int(positions_gpu.shape[0])
        _rtb = self._v4_fp4_ragged_row_to_batch[:_padded]
        _ls = self._v4_fp4_ragged_local_starts[:_padded]
        _le = self._v4_fp4_ragged_local_ends[:_padded]
        compute_varqlen_windows(cu_seq_q, _ncmt, _padded, out=(_rtb, _ls, _le))
        if attn_metadata.csa_n_committed_per_token is not None:
            _le.copy_(attn_metadata.csa_n_committed_per_token[:_padded])
        # Fixed logits width (max_model_len_idx) → the scorer's [padded, W] buffer
        # is a static shape (CG-capturable), same as the rectangular decode path.
        compute_prefill_schedule(
            _rtb,
            _ls,
            _le,
            self._fp4_block_k,
            self._fp4_parallel_unit_num,
            self.max_model_len_idx,
            cta_info_out=self._v4_fp4_ragged_cta_info,
        )
        meta["fp4_ragged_row_to_batch"] = _rtb
        meta["fp4_ragged_local_starts"] = _ls
        meta["fp4_ragged_local_ends"] = _le
        meta["fp4_ragged_cta_info"] = self._v4_fp4_ragged_cta_info
        meta["fp4_ragged_n_ctas"] = self._fp4_parallel_unit_num

    def _build_v4_indexer_meta(
        self,
        *,
        attn_metadata: AttentionMetaData_DSV4,
        positions_gpu,
        scheduled_bs: int,
        total_tokens: int,
        device,
        buf_prefix_ubatch: str = "",
    ):
        """Build per-fwd GPU index tensors consumed by `Indexer.forward_batched`.

        Returns None for warmup batches (the indexer falls back to its
        inline H2D path) or when CSA / Indexer is not on the model. CSA
        ratio is fixed at 4; we always build under that assumption.

        Reads pre-computed `attn_metadata.n_committed_csa_per_seq_cpu`
        (set by `_attach_v4_per_fwd_meta`, which MUST run first) for the
        per-seq committed count and cumsums it on CPU.

        Reuses `attn_metadata.batch_id_per_token` ([padded_T] int32), also set
        by `_attach_v4_per_fwd_meta`.

        DECODE fast path: returns an empty dict, plus the FP4 schedule below
        when that indexer is on. Everything else here is prefill-only —
        `deepgemm_fp8_paged_mqa_logits` and `top_k_per_row_decode` read paged KV
        through `attn_metadata.csa_n_committed_per_token`, never the
        packed-cumsum / per-token `cu_starts/cu_ends` layout this builds.

        The FP8 indexer K-cache write happens inside `fused_compress_attn`
        (the unified Indexer-inner Compressor path) via the same block_tables
        that CSA Main uses; no separate slot_mapping is built here.
        """

        # Caller contract: scheduled_bs >= 1, total_tokens >= 1 (same
        # invariants as `_attach_v4_per_fwd_meta` — guaranteed by every
        # prepare_*/CG-capture path).
        bs = scheduled_bs

        # DECODE short-circuit: the decode scorers take no field from this dict
        # except the FP4 schedule added below. The prefill-only derivations
        # further down (CPU cumsum + H2D for `cu_committed_gpu`; GPU launches
        # for `seq_base`/`visible_end`/`cu_ends`) feed `_score_topk_prefill`
        # only (cp_gather + fp8_mqa_logits + per-row prefill top-k), so they are
        # dead work on the decode hot path. ~50μs / fwd saved at bs=1024.
        if attn_metadata.state is AttnState.DECODE:
            meta = {}
            # DSpark RAGGED (varlen) decode uses the varlen path
            # (`_score_topk_decode_ragged_fp4` → `flydsl_pa_mqa_logits_fp4_prefill`).
            # Its per-row windows are built HERE (layer-invariant, once per fwd)
            # instead of per-CSA-layer in the scorer; the rectangular
            # `compute_varctx_schedule` is skipped when ragged (it's unused, and
            # eager `next_n = new_len.max()` need not divide `parallel_unit_num`,
            # tripping aiter's `P % next_n == 0` assert).
            _ragged = (
                getattr(attn_metadata, "dspark_ragged_lens_gpu", None) is not None
                and getattr(attn_metadata, "dspark_full_q", 0) > 0
            )
            # NOTE: with `dspark.ragged` + `confidence_schedule` on, `prepare_decode`
            # sets `dspark_ragged_lens_gpu` on EVERY decode (uniform lengths when
            # nothing shrank), so `_ragged` is always True there and the rectangular
            # branch below only runs with those options off. The windows are
            # therefore rebuilt once per fwd; steps whose lengths did not actually
            # shrink pay for a rebuild they do not need.
            if self._indexer_fp4 and _ragged:
                self._refresh_fp4_ragged_windows(
                    attn_metadata,
                    attn_metadata.dspark_ragged_lens_gpu,
                    positions_gpu,
                    meta,
                )
            elif self._indexer_fp4 and not _ragged:
                # CUDAGraph-safe FP4 rectangular decode: precompute the
                # persistent-grid schedule eagerly here (runs pre-replay during
                # build()) into a fixed-address buffer. compute_varctx_schedule is
                # pure on-device torch (no host sync) and emits a CONSTANT [P, 4]
                # cta_info with total_ctas == P fixed — so the captured kernel
                # reads fresh per-fwd contents from a stable pointer.
                from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
                    compute_varctx_schedule,
                )

                # Treat every query row as an independent batch item. This lets
                # the existing ratio-1 kernel consume exact ratio-4 CSA bounds.
                next_n = 1
                # Single-kernel schedule written straight into the fixed-address
                # buffer (no intermediate alloc + copy). ~50 tiny torch launches
                # -> 1 Triton launch (~300us -> ~40us per decode step).
                _ncmt_sched = attn_metadata.csa_n_committed_per_token[
                    : positions_gpu.shape[0]
                ]
                compute_varctx_schedule(
                    _ncmt_sched,
                    self._fp4_block_k,
                    self._fp4_parallel_unit_num,
                    self.max_model_len_idx,
                    next_n=next_n,
                    cta_info_out=self._v4_fp4_cta_info,
                )
                meta["fp4_cta_info"] = self._v4_fp4_cta_info
                meta["fp4_total_ctas"] = self._fp4_parallel_unit_num
            return meta

        n_committed_per_seq = attn_metadata.n_committed_csa_per_seq_cpu[:bs]
        cu_committed_cpu = np.concatenate(
            [
                np.zeros(1, dtype=np.int32),
                np.cumsum(n_committed_per_seq, dtype=np.int32),
            ]
        )
        # Empty-batch guard: when no seq has committed K yet
        # (`cu_committed_cpu[-1] == 0`, e.g. fresh prefill with prompt
        # shorter than the CSA `ratio`), `cp_gather_indexer_k_quant_cache`
        # would launch with grid.x = 0 and fail with HIP "invalid
        # configuration argument". Bump the last cumsum by one so the
        # kernel sees a single dummy row to gather (charged to the last
        # seq's first cache block). Downstream readers
        # (`fp8_mqa_logits` + `top_k_per_row_prefill`) honor per-token
        # `cu_starts`/`cu_ends` derived from `cu_committed_gpu[:-1]` and
        # `n_committed_per_seq`, both of which remain 0 — so the dummy
        # row is never read and the output is `-1` sentinels everywhere,
        # matching the all-empty semantics. Pure host-side scalar
        # arithmetic on a value already host-synced two lines up; no new
        # CG/torch.compile graph branch is introduced.
        cu_committed_cpu[-1] = max(int(cu_committed_cpu[-1]), 1)
        total_committed = int(cu_committed_cpu[-1])

        # batch_id_per_token: reuse the shared GPU tensor set in
        # `_attach_v4_per_fwd_meta` (which MUST run before this helper — see
        # prepare_decode/prefill ordering). int32, which PyTorch accepts for
        # advanced-indexing and both downstream kernels want anyway.
        batch_id_per_token_gpu = attn_metadata.batch_id_per_token[:total_tokens]
        # cu_committed_gpu is consumed both as `cu_starts/cu_ends` for the
        # fp8_mqa_logits per-token range AND as `cu_seq_lens` for the
        # cp_gather_indexer_k_quant_cache call (per-seq cumulative committed K).
        cu_committed_gpu = self._stage(
            f"{buf_prefix_ubatch}v4_indexer_cu_committed", cu_committed_cpu
        )

        # Layer-invariant GPU derivations (each was previously rebuilt ~30x
        # per fwd inside the per-CSA-layer body).
        seq_base_per_token_gpu = cu_committed_gpu[batch_id_per_token_gpu].to(
            torch.int32
        )  # [total_tokens] int32 — per-token offset into concat'd seqs'
        # compressed K. Used as `cu_starts` for fp8_mqa_logits AND as the
        # subtraction base for prefill `top_k_per_row_prefill`'s GLOBAL output
        # → seq-local conversion (the indexer kernel writes
        # `seq_base + col_in_seq`; we recover col_in_seq by subtracting).
        visible_end_gpu = visible_csa(positions_gpu[:total_tokens]).to(
            torch.int32
        )  # [total_tokens] int32 — causal upper bound, see `v4_pool_geometry`
        cu_ends_gpu = (
            seq_base_per_token_gpu + visible_end_gpu
        )  # [total_tokens] int32 — fp8_mqa_logits per-token end offset

        meta = {
            "total_committed": total_committed,
            "cu_committed_gpu": cu_committed_gpu,
            "batch_id_per_token_gpu": batch_id_per_token_gpu,  # int32, [total_tokens]
            # Prefill-only fields below — decode never consults them. NOT
            # in pre-allocated buffers (per-fwd derived); CG capture path
            # would see stale pointers, but the decode path doesn't touch
            # them, so it's fine.
            "seq_base_per_token_gpu": seq_base_per_token_gpu,
            "cu_starts_gpu": seq_base_per_token_gpu,  # alias for fp8_mqa_logits
            "cu_ends_gpu": cu_ends_gpu,
            # Seq-local per-token causal upper bound — consumed by the FP4
            # prefill path as `local_ends` (the FP4 kernel + top-k are
            # seq-local, vs the FP8 path's GLOBAL packed cu_ends).
            "visible_end_gpu": visible_end_gpu,
        }

        if self._indexer_fp4:
            # Precompute the FP4 prefill persistent-grid schedule here (instead
            # of inside flydsl_pa_mqa_logits_fp4_prefill) so the kernel call is
            # a pure launch. Prefill is eager (dynamic total_tokens), so this is
            # a per-fwd tensor, not a fixed buffer. Inputs match the score
            # call: row_to_batch = batch_id_per_token, local_starts = 0,
            # local_ends = visible_end. block_k / parallel_unit_num MUST match
            # the values passed to the kernel in `_score_topk_prefill_fp4`.
            from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
                compute_prefill_schedule,
            )

            # Size the seq-local logits width to this batch's ACTUAL max
            # committed index length (max over seqs of n_committed_csa), NOT the
            # model max (`max_model_len_idx`). Every query row's visible_end is
            # bounded by its seq's committed count, so this covers all writes.
            # Pure CPU (n_committed_csa_per_seq_cpu already host-side) — no new
            # sync. This right-sizes the `[total_tokens, W]` fp32 buffer (e.g.
            # 16k ctx -> W~4096 -> ~268MB, vs ~17GB at the fixed 262144 width),
            # so `_score_topk_prefill_fp4`'s budget-chunking rarely triggers and
            # the schedule is used as a single precomputed launch in the common
            # case. Decode keeps the fixed `max_model_len_idx` (CG needs a
            # static shape); prefill is eager so a per-fwd width is fine.
            fp4_prefill_max_seq_len = max(int(n_committed_per_seq.max()), 1)

            local_starts = torch.zeros_like(visible_end_gpu)
            # parallel_unit_num is the persistent-grid CTA-count CAP; the
            # schedule uses as many CTAs as it can up to this P (smaller P ->
            # larger `safe` chunk-fold -> fewer, more-serial CTAs). It bounds
            # TWO independent axes:
            #   - rows: every (row, chunk-split) needs a slot. A prefill fwd has
            #     one row PER QUERY TOKEN, so P must be >= prefill row count or
            #     surplus rows are silently dropped (logits stay at the -inf/NaN
            #     pre-fill -> wrong top-k). This is what `prefill_rows` covers.
            #   - chunks (context length): the 512 floor keeps enough CTAs to
            #     split a long context across the GPU even when rows are few
            #     (matters for decode; harmless here where rows dominate).
            # max() of both axes -> correct rows AND adequate chunk parallelism.
            prefill_rows = int(visible_end_gpu.shape[0])
            prefill_parallel_unit_num = max(self._fp4_parallel_unit_num, prefill_rows)
            _, prefill_cta_info, prefill_n_ctas = compute_prefill_schedule(
                batch_id_per_token_gpu.to(torch.int32),
                local_starts,
                visible_end_gpu,
                self._fp4_block_k,
                prefill_parallel_unit_num,
                fp4_prefill_max_seq_len,
            )
            meta["fp4_prefill_cta_info"] = prefill_cta_info
            meta["fp4_prefill_n_ctas"] = prefill_n_ctas
            meta["fp4_prefill_local_starts"] = local_starts
            meta["fp4_prefill_max_seq_len"] = fp4_prefill_max_seq_len

        return meta

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: torch.Tensor,  # [bs] int — eagle's current draft-step positions
        only_update: bool = False,
        num_reject_tokens: torch.Tensor | None = None,
    ):
        """Per-draft-step V4 region metadata rebuild for 1-token-per-seq shape.

        Called by EagleProposer.propose at mid-step iters. Eagle has already
        updated GPU state before this call:
          - ``attn_metadata.context_lens`` (GPU view of
            ``var["context_lens"].gpu``): rolled-back by ``prepare_decode``
            and bumped by eagle (`eagle.py:443`). Already the correct
            per-seq KV length for this draft step — DO NOT subtract
            num_reject_tokens (would double-rollback).
          - ``var["cu_seqlens_q"].gpu[:bs+1]``: set to ``arange(bs+1)``
            (`eagle.py:430`) for the 1-tok-per-seq shape.
        Eagle does NOT update the CPU mirrors (``var["..."].np``), so the
        CPU-numpy path of ``_attach_v4_per_fwd_meta`` /
        ``_attach_v4_paged_decode_meta`` would see stale values from verify.
        This routine bypasses both helpers and rebuilds the only buffers
        an SWA-only MTP layer actually consumes by calling
        ``write_v4_paged_decode_indices`` directly with GPU-computed
        indptrs. No D2H, no CPU mirror touch.

        Restricted to SWA-only MTP layers (compress_ratio == 0); asserted at
        builder init via ``self._mtp_layers_are_swa_only``. ``only_update``
        / ``num_reject_tokens`` are MLA-specific knobs and are ignored — V4
        handles rollback once in ``prepare_decode``.
        """
        # Sub-pool sizing runs in `model_runner.get_num_blocks`, AFTER
        # `warmup_model`, so the state class has no entries yet during warmup.
        # No-op there — warmup discards draft output anyway, and the
        # verify-shape attn_metadata stays valid.
        if not self.num_state_slots:
            return {}
        assert self._mtp_layers_are_swa_only, (
            "prepare_mtp_decode fast path only supports MTP layers the pool "
            "serves from the dense class; got compress_ratios[mtp]="
            f"{self.compress_ratios[self._n_main_layers :]} and field-window "
            f"layers {self._field_window_layers}"
        )

        var = self.model_runner.forward_vars
        running_bs = int(positions.shape[-1])  # rows; see the base contract
        attn_metadata = cast(
            AttentionMetaData_DSV4, get_forward_context().attn_metadata
        )
        # Pre-populated by the verify-forward `prepare_decode` and kept alive
        # across eagle.propose; assert for the static checker.
        assert attn_metadata.context_lens is not None
        assert attn_metadata.state_slot_out is not None
        win = self.window_size  # SWA prefix max per token

        # ----- GPU-side SWA indptr math (no CPU numpy, no D2H) -----
        # ctx_gpu is already correct (rolled-back by prepare_decode + bumped
        # by eagle). int32 in the source buffer; keep dtype throughout.
        # Only SWA is computed; CSA/HCA indices are unused by SWA-only MTP
        # (asserted above) and will be fully rebuilt by the next verify-fwd's
        # `prepare_decode`.
        actual_swa = torch.clamp(positions + 1, max=win)

        swa_indptr = var["v4_kv_indptr_swa"][: running_bs + 1]
        # positions/actual_swa are int64 (eagle's positions buffer); cast to
        # int32 inside cumsum to match swa_indptr's int32 storage.
        torch.cumsum(actual_swa, dim=0, dtype=torch.int32, out=swa_indptr[1:])

        # batch_id_per_token: 1-tok-per-seq → arange(bs), with `-1` over the
        # drafter's pad tail. That sentinel is the SAME one the verify fwd uses
        # (`_attach_v4_per_fwd_meta` fills the padded tail with -1): the index
        # writer and `csa_translate_pack` both skip those rows, so a padded
        # draft costs nothing for the rows it invented instead of repeating a
        # real one's gather. At bs=65 padded to 128 that is half the batch.
        #
        # By its fixed name because a captured draft graph bakes this address:
        # the tensor a mid-step installs has to be the same object every step,
        # and under TBO `attn_metadata` alternates between ubatch-prefixed
        # buffers. The draft is not ubatched, so the unprefixed one and a global
        # arange are its map. `_attach_v4_per_fwd_meta` reassigns this field on
        # every verify fwd, so writing it clobbers nothing.
        #
        # Not a slice of `cu_seqlens_q` either: that one is also the q indptr,
        # and a sentinel in it would corrupt the other reader. `row_ids` is
        # the resident arange the real prefix is restored from.
        batch_id_per_token = var["v4_batch_id_per_token"].gpu[:running_bs]
        batch_id_per_token[:bs].copy_(self.row_ids[:bs])
        if running_bs > bs:
            batch_id_per_token[bs:] = -1

        # ----- Kernel: write SWA prefix paged offsets -----
        # MTP layers are dense, so only the dense class's buffer is asked for;
        # the other two are switched off. They used to be aliased onto this same
        # buffer because all three classes named the same row, which stopped
        # being true when the window started interleaving by class.
        swa_indices_buf = var["v4_kv_indices_swa"]
        dest_rows = self._dest_row_buffers()
        write_v4_paged_decode_indices(
            state_slot_per_seq=attn_metadata.state_slot_out[:running_bs],
            batch_id_per_token=batch_id_per_token,
            positions=positions,
            swa_indptr=swa_indptr,
            csa_indptr=None,
            hca_indptr=None,
            swa_indices=swa_indices_buf,
            csa_indices=None,
            hca_indices=None,
            dest_rows=dest_rows,
            T=running_bs,
            win=win,
            geometry=self.pool_geometry,
        )
        attn_metadata.swa_dest_rows = dest_rows

        # ----- Publish on attn_metadata for V4Attention.forward -----
        # MTP layer is ratio=0 → reads kv_indices_swa + kv_indptr_swa only.
        # kv_indices_{csa,hca} / kv_indptr_{csa,hca} are left at whatever
        # prepare_decode populated for the verify shape; downstream V4
        # decode kernel only touches them when ratio != 0.
        attn_metadata.state = AttnState.DECODE
        attn_metadata.max_seqlen_q = 1
        attn_metadata.kv_indices_swa = swa_indices_buf
        attn_metadata.kv_indptr_swa = swa_indptr
        attn_metadata.batch_id_per_token = batch_id_per_token

        # fp8 asm decode per-token index tensors. MTP draft step is 1-token-per-
        # seq → the asm kernel sees N = bs. Stage the constant per-token tensors
        # to that length via the same builder-staged path as the verify fwd.
        if self._kv_fp8:
            attn_metadata.qo_indptr = self._stage(
                "v4_qo_indptr", self._v4_qo_indptr_np[: running_bs + 1]
            )

        # NOT rebuilt (unused by SWA-only MTP layer; would block a future
        # CSA/HCA MTP layer — assert at top guards):
        #   - n_committed_{csa,hca}_per_seq{,_cpu} (compressor/HCA tail math)
        #   - skip_prefix_len_csa (csa_translate_pack per-layer write)
        #   - compress_plans (Compressor — only present when ratio != 0)
        #   - HCA compress tail in kv_indices_hca
        #   - v4 indexer meta (Indexer — only present when ratio == 4)
        return {}

    def prepare_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ):
        """V4-style decode prep: populates positions, cu_seqlens_q,
        block_tables, and state_slot_out.

        Uses stream overlap (like AiterMLAMetadataBuilder) to hide H2D
        latency behind CPU numpy work: basic H2D copies fire on
        ``prep_stream`` while ``_build_compress_plans`` runs on the CPU.
        """
        var = self.model_runner.forward_vars
        scheduled_bs = batch.total_seqs_num_decode
        context_lens_np = np.asarray(batch.context_lens, dtype=np.int32)
        # Per-seq decode forward length, settled by the step rather than carried
        # on the batch (= num_spec_step+1 for plain MTP, or the DSpark q-bucket
        # when shrunk). positions/attn use this so the (running_bs, q) graph is
        # selected. See `ForwardMode.max_seqlen_q`.
        # MTP: roll back ctx by `num_rejected` so this fwd's positions overwrite
        # last fwd's rejected-draft slots (matches aiter_mla.py:701 /
        # aiter_attention.py:542). `batch.context_lens` = `seq.num_tokens`
        # which the scheduler advances by `mtp_k - num_rejected` placeholders
        # per fwd (scheduler.py:789); without this rollback, MTP-k positions
        # would skip ahead by `num_rejected` and the rejected slots would
        # never be overwritten with the corrected K/V. `num_rejected` is None
        # on dummy runs and on the first fwd before any sampler output.
        # The rolled-back ctx is also what anchors `positions` (at `ctx -
        # full_q`, below), and every compress count is now `visible_*(pos)`, so
        # a rejected slot's KV falls out of range on its own — `block_tables`
        # needs no truncation here.
        if not batch.is_dummy_run and max_seqlen_q > 1:
            num_rejected = self.model_runner.tokenID_processor.num_rejected
            if num_rejected is not None:
                context_lens_np = context_lens_np - num_rejected.astype(np.int32)
        # DSpark q-shrink: anchor the forwarded q tokens to the draft span HEAD
        # (ctx-full_q), not the tail, so they stay in [ctx-full_q .. ctx-1] (never
        # OOB); dropped tail slots are re-drafted next step (lossless). No-op when
        # q == full_q.
        full_q = batch.num_spec_step + 1
        ragged_lens = getattr(batch, "dynamic_spec_query_tokens_per_req", None)
        if ragged_lens is not None:
            # RAGGED (§5.2): each seq forwards len_i tokens (no batch pad); build
            # positions via per-seq cumsum + in-seg arange, span-head anchored:
            # token j of seq i -> (ctx_i - full_q) + j.
            lens = np.asarray(ragged_lens, dtype=np.int32)[:scheduled_bs]
            # The last token of seq i lands on `ctx_i - full_q + len_i - 1`, so
            # the whole invariant reduces to one comparison here.
            require_step_within_full_q(
                int(lens.max()) if lens.size else 0, full_q, "a DSpark ragged step"
            )
            cu = np.zeros(scheduled_bs + 1, dtype=np.int64)
            np.cumsum(lens, out=cu[1:])
            batch_ids = np.repeat(np.arange(scheduled_bs, dtype=np.int32), lens)
            j_in_seq = np.arange(int(cu[-1]), dtype=np.int32) - cu[batch_ids].astype(
                np.int32
            )
            positions_np = (context_lens_np - full_q)[batch_ids] + j_in_seq
        else:
            # One scalar on the rectangle: every sequence forwards the same
            # `max_seqlen_q` tokens.
            require_step_within_full_q(
                max_seqlen_q, full_q, "a rectangular decode step"
            )
            positions_np = np.tile(
                np.arange(max_seqlen_q, dtype=np.int32), scheduled_bs
            ) + np.repeat(context_lens_np - full_q, max_seqlen_q)
        sum_scheduled_tokens = batch.total_tokens_num_decode

        # Pad positions to the rows this forward runs. That is
        # `running_tokens` by definition -- a ragged step's flat bucket and a
        # rectangular step's `running_bs * q` are the same field.
        graph_cap_tokens = int(running_tokens)
        if graph_cap_tokens > sum_scheduled_tokens:
            _pad_positions = np.zeros(graph_cap_tokens, dtype=positions_np.dtype)
            _pad_positions[:sum_scheduled_tokens] = positions_np
            positions_np = _pad_positions
            sum_scheduled_tokens_padded = graph_cap_tokens
        else:
            sum_scheduled_tokens_padded = sum_scheduled_tokens

        var["positions"].np[:sum_scheduled_tokens_padded] = positions_np

        var["context_lens"].np[:scheduled_bs] = context_lens_np

        # Inline block_tables CPU fill (H2D deferred to prep_stream).
        self.prepare_block_tables(batch)

        pool_np = np.asarray(batch.state_slots_committed[:scheduled_bs], dtype=np.int32)
        if len(pool_np) < scheduled_bs:
            pool_np = np.zeros(scheduled_bs, dtype=np.int32)
        state_slot_np = self._physical_slots(pool_np)
        ss_buf = var["v4_meta_state_slot_out"]
        ss_buf.np[:scheduled_bs] = state_slot_np
        si_buf = var["v4_meta_state_slot_in"]
        si_buf.np[:scheduled_bs] = self._state_slot_in_np(
            batch, scheduled_bs, state_slot_np
        )
        # Published at the PADDED `running_bs`, like the ubatch path below, so
        # a consumer that runs the padded batch -- a speculative drafter -- can
        # slice to it; `cu_seqlens_q` is padded to the same batch, so a consumer
        # inferring bs from these slots stays consistent. Every reader either
        # masks the pad tail out (`batch_id_per_token = -1`) or discards those
        # rows, so 0 is a legal filler.
        ss_buf.np[scheduled_bs:running_bs] = 0
        si_buf.np[scheduled_bs:running_bs] = 0

        # ---- fire H2D on prep_stream ----
        # NB: this runs inside attn_metadata_builder.build(), BEFORE
        # set_forward_context() — can't read main_stream from the context yet.
        prep_stream = self.prep_stream
        current_stream = torch.cuda.current_stream()
        prep_stream.wait_stream(current_stream)
        with torch.cuda.stream(prep_stream):
            positions = var["positions"].copy_to_gpu(sum_scheduled_tokens_padded)
            cu_seqlens_q_gpu = var["cu_seqlens_q"].copy_to_gpu(running_bs + 1)
            context_lens_gpu = var["context_lens"].copy_to_gpu(scheduled_bs)
            block_tables_gpu = var["block_tables"].copy_to_gpu(scheduled_bs)
            state_slot_gpu = ss_buf.copy_to_gpu(running_bs)
            state_slot_in_gpu = si_buf.copy_to_gpu(running_bs)

        # ---- CPU numpy work, overlapped with prep_stream H2D ----
        # RAGGED: per-seq extend lengths (else uniform max_seqlen_q). compress
        # plans + per-fwd meta are all marker-driven (repeat/cumsum over this),
        # so a ragged array flows through unchanged.
        if ragged_lens is not None:
            extend_lens_np = np.asarray(ragged_lens, dtype=np.int32)[:scheduled_bs]
        else:
            extend_lens_np = np.full(scheduled_bs, max_seqlen_q, dtype=np.int32)
        compress_plans = self._build_compress_plans(
            extend_lens_np,
            context_lens_np,
            running_bs=running_bs,
            max_q_len=max_seqlen_q,
        )

        # ---- sync, build attn_metadata, per-fwd meta ----
        current_stream.wait_stream(prep_stream)

        attn_metadata = AttentionMetaData_DSV4(
            cu_seqlens_q=cu_seqlens_q_gpu,
            cu_seqlens_k=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=int(context_lens_np.max()) if len(context_lens_np) else 1,
            min_seqlen_q=0,
            dropout_p=0.0,
            has_cached=False,
            total_kv=int(context_lens_np.sum()),
            num_cached_tokens=None,
            block_tables=block_tables_gpu,
            context_lens=context_lens_gpu,
            state=AttnState.DECODE,
        )
        attn_metadata.state_slot_out = state_slot_gpu
        attn_metadata.state_slot_in = state_slot_in_gpu
        attn_metadata.state_slot_out_cpu = state_slot_np
        attn_metadata.compress_plans = compress_plans
        # DSpark RAGGED: pass per-seq verify lengths + full_q to the (rectangular-
        # only) decode indexer so it can pad Q back to [running_bs, full_q].
        _drafter = getattr(self.model_runner, "drafter", None)
        _dspark_ragged_graph = (
            self.model_runner.config.dspark.ragged
            and _drafter is not None
            and _drafter.uses_confidence_schedule
        )
        if ragged_lens is not None or _dspark_ragged_graph:
            # Pinned staging, not `torch.as_tensor(np, device=cuda)`: that is a
            # pageable H2D and syncs here, which was the ragged decode bubble.
            attn_metadata.dspark_ragged_lens_gpu = self._stage_dspark_ragged_lens(
                extend_lens_np, self._dspark_ragged_lens_pad_to(running_bs)
            )
            attn_metadata.dspark_full_q = int(full_q)

        running_bs = int(running_bs)
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            extend_lens_np,  # = np.full(scheduled_bs, max_seqlen_q) for decode
            state_slot_np,
            scheduled_bs,
            sum_scheduled_tokens,
            running_bs=running_bs,
            max_q_len=max_seqlen_q,
            running_tokens=sum_scheduled_tokens_padded,
        )
        self._attach_v4_indexer_meta(
            attn_metadata,
            scheduled_bs,
            sum_scheduled_tokens,
            positions_gpu=positions,
        )

        self._ubatch_decode_meta = None
        if (
            self.model_runner.config.enable_tbo_decode
            and scheduled_bs > 2
            and not batch.is_dummy_run
        ):
            self._prepare_ubatch_decode(
                scheduled_bs=scheduled_bs,
                running_bs=running_bs,
                max_seqlen_q=max_seqlen_q,
                context_lens_np=context_lens_np,
                state_slot_np=state_slot_np,
                state_slot_in_np=si_buf.np[:scheduled_bs],
                positions_np=positions_np,
                extend_lens_np=extend_lens_np,
                dspark_ragged=ragged_lens is not None or _dspark_ragged_graph,
                full_q=int(full_q),
            )

        return attn_metadata, positions

    def _prepare_ubatch_decode(
        self,
        *,
        scheduled_bs: int,
        running_bs: int,
        max_seqlen_q: int,
        context_lens_np: np.ndarray,
        state_slot_np: np.ndarray,
        state_slot_in_np: np.ndarray,
        positions_np: np.ndarray,
        extend_lens_np: np.ndarray,
        dspark_ragged: bool,
        full_q: int,
    ) -> None:
        """Split a decode batch into two micro-batches (by request) and build
        each one's V4 decode metadata into ``ub{0,1}_`` prefixed buffers.

        Mirrors :meth:`prepare_decode` but operates on a per-ubatch request
        slice. The two resulting :class:`AttentionMetaData_DSV4` objects are
        cached on ``self._ubatch_decode_meta`` and returned by
        :meth:`build_ubatch_metadata`.

        Token layout is request-major. ``extend_lens_np`` supplies exact token
        boundaries for DSpark ragged verify; rectangular decode contains the
        uniform ``max_seqlen_q`` value.
        """
        var = self.model_runner.forward_vars
        N = self._NUM_TBO_UBATCHES
        enforce_eager = self.model_runner.enforce_eager
        if enforce_eager:
            split_total = scheduled_bs
            half = scheduled_bs // N
            padded_list = [half, scheduled_bs - half]
            ub_ranges = [(0, half), (half, split_total)]
        else:
            from atom.utils.tbo.ubatch_wrapper import UBatchWrapper

            ctx = get_forward_context()
            padded_list = [
                UBatchWrapper._decode_ub_running_bs(ctx, i, N, running_bs)
                for i in range(N)
            ]
            # Real-request ranges partition scheduled_bs; each ubatch owns up to
            # its padded capacity, the tail ubatch takes the remainder. Pad rows
            # beyond the real reqs carry sentinels (filled below).
            ub_ranges = []
            req_start = 0
            for i in range(N):
                if i == N - 1:
                    req_end = scheduled_bs
                else:
                    req_end = min(scheduled_bs, req_start + padded_list[i])
                ub_ranges.append((req_start, req_end))
                req_start = req_end
            split_total = scheduled_bs

        metas: list = []
        token_offsets = np.zeros(scheduled_bs + 1, dtype=np.int32)
        np.cumsum(extend_lens_np[:scheduled_bs], out=token_offsets[1:])
        for ub_idx, (req_start, req_end) in enumerate(ub_ranges):
            p = f"ub{ub_idx}_"
            # THIS ubatch's width, not the step's -- the two are different
            # numbers and both are called a running_bs, so they get distinct
            # names (cf. `UBatchWrapper._decode_ub_running_bs`, which supplies it).
            ub_running_bs = padded_list[ub_idx]
            # Real requests that fall into this ubatch's [req_start, req_end),
            # clamped to scheduled_bs (cudagraph pad rows beyond scheduled_bs
            # carry sentinels, exercised only during capture's synthetic batch).
            ub_real_reqs = max(0, min(scheduled_bs, req_end) - req_start)
            tok_start = int(token_offsets[req_start])
            tok_end = int(token_offsets[req_start + ub_real_reqs])
            ub_real_tokens = tok_end - tok_start
            ub_extend_lens_np = extend_lens_np[req_start : req_start + ub_real_reqs]

            # ---- per-seq slices into ub buffers ----
            ub_ctx_np = context_lens_np[req_start : req_start + ub_real_reqs]
            var[f"{p}context_lens"].np[:ub_real_reqs] = ub_ctx_np
            var[f"{p}context_lens"].np[ub_real_reqs:ub_running_bs] = 0

            ub_state_np = state_slot_np[req_start : req_start + ub_real_reqs]
            if len(ub_state_np) < ub_real_reqs:
                ub_state_np = np.zeros(ub_real_reqs, dtype=np.int32)
            var[f"{p}v4_meta_state_slot_out"].np[:ub_real_reqs] = ub_state_np
            var[f"{p}v4_meta_state_slot_out"].np[ub_real_reqs:ub_running_bs] = 0
            state_slot_np_ub = (
                var[f"{p}v4_meta_state_slot_out"].np[:ub_running_bs].copy()
            )
            ub_state_in_np = state_slot_in_np[req_start : req_start + ub_real_reqs]
            if len(ub_state_in_np) < ub_real_reqs:
                ub_state_in_np = np.zeros(ub_real_reqs, dtype=np.int32)
            var[f"{p}v4_meta_state_slot_in"].np[:ub_real_reqs] = ub_state_in_np
            var[f"{p}v4_meta_state_slot_in"].np[ub_real_reqs:ub_running_bs] = 0

            var[f"{p}block_tables"].np[:ub_real_reqs] = var["block_tables"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}block_tables"].np[ub_real_reqs:ub_running_bs] = 0

            # positions: copy the ubatch's token slice (values match the global
            # positions slice the UBatchWrapper Context will expose).
            ub_running_tokens = ub_running_bs * max_seqlen_q
            ub_positions_np = positions_np[tok_start : tok_start + ub_real_tokens]
            var[f"{p}positions"].np[:ub_real_tokens] = ub_positions_np
            var[f"{p}positions"].np[ub_real_tokens:ub_running_tokens] = 0

            # cu_seqlens_q: exact per-request lengths, padded tail flat.
            cu = np.zeros(ub_real_reqs + 1, dtype=np.int32)
            np.cumsum(ub_extend_lens_np, out=cu[1:])
            var[f"{p}cu_seqlens_q"].np[: ub_real_reqs + 1] = cu
            var[f"{p}cu_seqlens_q"].np[
                ub_real_reqs + 1 : ub_running_bs + 1
            ] = ub_real_tokens

            # ---- H2D ----
            ub_sum_tokens = max(ub_real_tokens, 1)
            positions_gpu = var[f"{p}positions"].copy_to_gpu(ub_running_tokens)
            cu_seqlens_q_gpu = var[f"{p}cu_seqlens_q"].copy_to_gpu(ub_running_bs + 1)
            context_lens_gpu = var[f"{p}context_lens"].copy_to_gpu(ub_running_bs)
            block_tables_gpu = var[f"{p}block_tables"].copy_to_gpu(ub_running_bs)
            state_slot_gpu = var[f"{p}v4_meta_state_slot_out"].copy_to_gpu(
                ub_running_bs
            )

            # ---- compress plans (per ubatch buffer set) ----
            ctx_for_plan = context_lens_np[req_start : req_start + ub_real_reqs]
            compress_plans = self._build_compress_plans(
                ub_extend_lens_np,
                ctx_for_plan,
                running_bs=ub_running_bs,
                max_q_len=max_seqlen_q,
                buf_prefix_ubatch=p,
            )

            attn_metadata = AttentionMetaData_DSV4(
                cu_seqlens_q=cu_seqlens_q_gpu,
                cu_seqlens_k=None,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=int(ub_ctx_np.max()) if ub_real_reqs > 0 else 1,
                min_seqlen_q=0,
                dropout_p=0.0,
                has_cached=False,
                total_kv=int(ub_ctx_np.sum()) if ub_real_reqs > 0 else 0,
                num_cached_tokens=None,
                block_tables=block_tables_gpu,
                context_lens=context_lens_gpu,
                state=AttnState.DECODE,
            )
            attn_metadata.state_slot_out = state_slot_gpu
            attn_metadata.state_slot_in = var[f"{p}v4_meta_state_slot_in"].copy_to_gpu(
                ub_running_bs
            )
            attn_metadata.state_slot_out_cpu = state_slot_np_ub
            attn_metadata.compress_plans = compress_plans
            if dspark_ragged:
                ragged_lens_buf = np.zeros(ub_running_bs, dtype=np.int32)
                ragged_lens_buf[:ub_real_reqs] = ub_extend_lens_np
                attn_metadata.dspark_ragged_lens_gpu = torch.as_tensor(
                    ragged_lens_buf, device=positions_gpu.device
                )
                attn_metadata.dspark_full_q = full_q

            self._attach_v4_per_fwd_meta(
                attn_metadata,
                ub_extend_lens_np,
                state_slot_np_ub,
                ub_real_reqs,
                ub_real_tokens,
                running_bs=ub_running_bs,
                max_q_len=max_seqlen_q,
                buf_prefix_ubatch=p,
            )
            self._attach_v4_indexer_meta(
                attn_metadata,
                max(ub_real_reqs, 1),
                ub_sum_tokens,
                positions_gpu=positions_gpu,
            )
            metas.append(attn_metadata)

        self._ubatch_decode_meta = metas

    def build_ubatch_metadata(
        self, ubatch_idx: int, running_bs: int
    ) -> AttentionMetaData_DSV4:
        assert self._ubatch_decode_meta is not None, (
            "build_ubatch_metadata called but no ubatch decode metadata was "
            "prepared — ensure enable_tbo_decode is set and prepare_decode ran."
        )
        return self._ubatch_decode_meta[ubatch_idx]

    def prepare_prefill(self, batch: ScheduledBatch, running_bs: int):
        """V4 prefill prep: extends parent to always populate block_tables
        and state_slot_out.

        The parent only emits block_tables when has_cached (prefix cache hit);
        V4 always needs block_tables because Compressor scatters compressed
        entries into the classical KV pool from token 0 onwards.

        Also publishes CPU mirrors (`v4_*_cpu`) consumed by the V4 forward
        path to avoid `.item()` / `.tolist()` syncs (PR-A Phase 2).
        """
        base_md, positions = super().prepare_prefill(batch, running_bs)
        # Promote to V4 typed metadata so V4-specific attribute assignments
        # below are well-typed. Safe because AttentionMetaData_DSV4 only adds
        # fields with defaults; the parent dataclass is non-slotted.
        base_md.__class__ = AttentionMetaData_DSV4
        attn_metadata = cast(AttentionMetaData_DSV4, base_md)
        # state defaults to PREFILL_NATIVE (set by `backends.build()` after
        # this returns); `_build_paged_prefill_meta` upgrades to
        # PREFILL_PREFIX if any seq has chunk_start > 0 (chunked prefill).
        scheduled_bs = batch.total_seqs_num_prefill
        if attn_metadata.block_tables is None:
            attn_metadata.block_tables = self._populate_block_tables(
                batch, scheduled_bs
            )
        state_slot_gpu, state_slot_np = self._populate_state_slot_mappings(
            batch, scheduled_bs, running_bs, return_cpu=True
        )
        attn_metadata.state_slot_out = state_slot_gpu
        attn_metadata.state_slot_in = self._populate_state_slot_in(
            batch, scheduled_bs, running_bs, state_slot_np
        )
        # PR-A Phase 2 CPU mirrors (generic, not V4-specific). The parent
        # populated forward_vars CPU buffers; read them back as numpy slices.
        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        positions_np = np.asarray(var["positions"].np[:sum_scheduled_tokens])
        cu_seqlens_q_np = np.asarray(var["cu_seqlens_q"].np[: scheduled_bs + 1])
        attn_metadata.state_slot_out_cpu = state_slot_np
        # `start_pos_per_seq` = position of FIRST token of each seq in this fwd.
        # Only consumed by `_build_paged_prefill_meta` below; not stashed on
        # attn_metadata (no other reader, no inter-fwd reuse).
        start_pos_per_seq_np = positions_np[cu_seqlens_q_np[:scheduled_bs]]
        # Compress plans (per ratio) for batched fused_compress + update_states.
        # Prefill batch: extend_lens read from cu_seqlens_q_np.
        # Must run BEFORE `_attach_v4_indexer_meta` (the indexer consumes
        # plan.compress_plan_cpu to derive its FP8 write-side slot_mapping).
        extend_lens_np = (
            cu_seqlens_q_np[1 : scheduled_bs + 1] - cu_seqlens_q_np[:scheduled_bs]
        ).astype(np.int32)
        # context_lens already populated on host by `super().prepare_prefill`
        # (backends.py: `var["context_lens"].np[:bs] = batch.context_lens`).
        # Mathematically equals `start_pos + extend_lens` but reading the
        # canonical buffer avoids drift if scheduler/batch semantics ever
        # change.
        context_lens_np = np.asarray(
            var["context_lens"].np[:scheduled_bs], dtype=np.int32
        )
        attn_metadata.compress_plans = self._build_compress_plans(
            extend_lens_np, context_lens_np
        )
        # Prefill goes through eager (no CG): defaults make padded_total_tokens
        # collapse to total_tokens — no padding logic kicks in. Must still run
        # BEFORE `_attach_v4_indexer_meta` so the indexer-side meta builder can
        # reuse the shared GPU tensors (batch_id_per_token, n_committed_csa).
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            extend_lens_np,  # = cu_seqlens_q[1:] - cu_seqlens_q[:bs]
            attn_metadata.state_slot_out_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
        )
        self._attach_v4_indexer_meta(
            attn_metadata,
            scheduled_bs,
            sum_scheduled_tokens,
            positions_gpu=positions,
        )
        # Two-source paged_prefill index buffers (extend + per-ratio prefix).
        # Eager-only — direct H2D, no forward_vars staging required. Sets
        # attn_metadata.{kv_indices,kv_indptr}_{extend,prefix_swa,prefix_csa,prefix_hca}
        # plus skip_prefix_len_csa and envelope_rows.
        self._build_paged_prefill_meta(
            attn_metadata,
            positions_np,
            cu_seqlens_q_np,
            extend_lens_np,
            start_pos_per_seq_np,
            attn_metadata.state_slot_out_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
        )

        # ----- PCP: reindex per-query metadata to this rank's 1/W shard -----
        # Mirrors SGLang's apply_cp_reindex (deepseek_v4_backend_hip_radix.py):
        # all metadata above was built for the FULL sequence; under PCP the
        # model.forward entry round-robin-splits hidden/positions to 1/W, so the
        # per-query (per-token) metadata must be reduced to the SAME owned-query
        # set. Per-seq / KV-write fields stay full (every rank keeps full KV).
        # PCP+TBO request-boundary split: DEFER reindex to per-group in
        # build_ubatch_prefill_metadata (each request group reindexed
        # independently on its own pcp pad). Keep the FULL un-reindexed metadata
        # here so build_ubatch can slice it per group.
        _bal = getattr(self.model_runner, "_pcp_tbo_balanced_active", False)
        if pcp_is_enabled() and not batch.is_dummy_run and not _bal:
            # Gate on `not is_dummy_run`: ForCausalLM.forward's round-robin-split is
            # skipped on dummy/warmup runs (_pcp_active() returns False there),
            # so reindexing metadata to 1/W here would pair full-size
            # input_ids/positions with 1/W metadata (length mismatch). Keeping
            # both full on dummy runs stays self-consistent.
            # Reindex metadata to 1/W in-place. We intentionally DISCARD the
            # returned 1/W positions: `positions` must stay FULL here so it
            # lands on context.positions full, and ForCausalLM.forward does the
            # one and only round-robin-split of positions (symmetric with input_ids,
            # which never passes through the builder). Splitting here too would
            # double-split positions (full -> 1/W -> 1/2W) while input_ids/kv
            # are only split once, desyncing swa_write (kv full vs positions
            # under-length). The builder still uses its internal 1/W positions
            # for indexer_meta (rebuilt inside _apply_pcp_reindex).
            self._apply_pcp_reindex(
                attn_metadata, positions, scheduled_bs, sum_scheduled_tokens
            )
        self._attach_tbo_prefill_cpu_lens(attn_metadata, scheduled_bs)
        return attn_metadata, positions

    def _apply_pcp_reindex(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        positions: torch.Tensor,
        scheduled_bs: int,
        total_tokens: int,
    ) -> torch.Tensor:
        """Reduce per-query prefill metadata to this PCP rank's round-robin shard.

        Splits the per-token / per-query fields by `token_idx % pcp == rank`
        (matching model.forward's round-robin split of hidden/positions) while
        leaving per-seq and KV-write fields full. The indexer metadata is
        REBUILT from the sliced batch_id_per_token + positions (its per-token
        fields all derive from those two), mirroring SGLang's
        init_forward_metadata_indexer(core_meta) after apply_cp_reindex.

        Returns the sliced `positions` (the model.forward entry slices its own
        copy identically; this keeps attn_metadata-internal users consistent).

        Token count is padded to a multiple of pcp_size (dummy queries with
        zero-length KV) so every rank gets an equal shard — matching
        model.forward's pad-then-split of hidden/positions.
        """
        pcp_size = get_pcp_world_size()
        device = attn_metadata.batch_id_per_token.device
        # Pad to a multiple of pcp_size; dummy (pad) queries get zero-length KV.
        # This runs on the non-TBO PCP path (full-batch reindex) and, under
        # PCP+TBO request-boundary split, per request GROUP (each group reindexed independently
        # on its own pcp pad). Either way the divisor is pcp_size.
        padded_total = pcp_pad_len(total_tokens, pcp_size)
        n_pad = padded_total - total_tokens
        owned_q = pcp_round_robin_query_indices(padded_total, pcp_size).to(device)

        # --- ragged per-query buffers: pad indptr to padded_total, then 1/W ---
        for ind_attr, idx_attr in (
            ("kv_indptr_prefix_swa", "kv_indices_prefix_swa"),
            ("kv_indptr_prefix_csa", "kv_indices_prefix_csa"),
            ("kv_indptr_prefix_hca", "kv_indices_prefix_hca"),
            ("kv_indptr_extend", "kv_indices_extend"),
        ):
            indptr = getattr(attn_metadata, ind_attr, None)
            indices = getattr(attn_metadata, idx_attr, None)
            if indptr is None or indices is None:
                continue
            indptr = pcp_pad_indptr(indptr, n_pad)  # dummy queries: 0-length KV
            new_indptr, new_indices = pcp_reindex_ragged(indptr, indices, owned_q)
            setattr(attn_metadata, ind_attr, new_indptr)
            setattr(attn_metadata, idx_attr, new_indices)

        # --- dense per-token fields: pad then round-robin-slice to 1/W ---
        if attn_metadata.skip_prefix_len_csa is not None:
            skip = pcp_pad_dense(attn_metadata.skip_prefix_len_csa, n_pad)
            attn_metadata.skip_prefix_len_csa = skip[owned_q].contiguous()
        # batch_id_per_token drives the indexer rebuild below. Pad with -1
        # (dummy-token sentinel; downstream kernels skip on bid < 0), then slice.
        bid = attn_metadata.batch_id_per_token[:total_tokens]
        if n_pad > 0:
            bid = torch.cat([bid, bid.new_full((n_pad,), -1)], dim=0)
        attn_metadata.batch_id_per_token = bid[owned_q].contiguous()
        pos_padded = positions[:total_tokens]
        if n_pad > 0:
            pos_padded = torch.cat([pos_padded, pos_padded.new_zeros(n_pad)], dim=0)
        positions_local = pos_padded[owned_q].contiguous()

        # --- rebuild indexer metadata from the sliced batch_id + positions ---
        # Its per-token fields (seq_base/cu_starts/cu_ends/visible_end) all
        # derive from batch_id_per_token + positions, so rebuilding with the
        # sliced inputs yields the 1/W layout; the per-seq field (cu_committed)
        # stays full. Skip if the model has no CSA/indexer.
        if attn_metadata.indexer_meta is not None:
            local_tokens = owned_q.shape[0]
            attn_metadata.indexer_meta = self._build_v4_indexer_meta(
                attn_metadata=attn_metadata,
                positions_gpu=positions_local,
                scheduled_bs=scheduled_bs,
                total_tokens=local_tokens,
                device=device,
            )
        return positions_local

    def _get_ubatch_compress_plan_buffers(
        self, ubatch_idx: int
    ) -> dict[int, dict[str, "CpuGpuBuffer"]]:

        if not hasattr(self, "_ubatch_compress_plan_buffers"):
            self._ubatch_compress_plan_buffers: dict[
                int, dict[int, dict[str, CpuGpuBuffer]]
            ] = {}
        cached = self._ubatch_compress_plan_buffers.get(ubatch_idx)
        if cached is not None:
            return cached

        var = self.model_runner.forward_vars
        pool: dict[int, dict[str, CpuGpuBuffer]] = {}
        for ratio, _ in self._unique_compress_ratios_overlap:
            tmpl_c = var[f"v4_compress_plan_{ratio}"]
            tmpl_w = var[f"v4_write_plan_{ratio}"]
            buf_c = CpuGpuBuffer(
                *tmpl_c.cpu.shape, dtype=tmpl_c.cpu.dtype, device=tmpl_c.gpu.device
            )
            buf_w = CpuGpuBuffer(
                *tmpl_w.cpu.shape, dtype=tmpl_w.cpu.dtype, device=tmpl_w.gpu.device
            )
            # Sentinel-fill so any unused tail rows behave like the main pool.
            buf_c.cpu.fill_(-1)
            buf_c.copy_to_gpu()
            buf_w.cpu.fill_(-1)
            buf_w.copy_to_gpu()
            pool[ratio] = {"compress": buf_c, "write": buf_w}
        self._ubatch_compress_plan_buffers[ubatch_idx] = pool
        return pool

    def build_ubatch_prefill_metadata(
        self,
        attn_metadata: AttentionMetaData,
        ub_slice,
        running_bs: int,
        ubatch_idx: int = 0,
    ) -> AttentionMetaData_DSV4:
        """Split prefill AttentionMetaData for V4 TBO micro-batches.

        Two paths:
        - PCP+TBO request-boundary split: dispatches to
          `_build_ubatch_prefill_metadata_balanced(attn_metadata, ubatch_idx)`,
          which derives the group from `model_runner._pcp_bal_groups[ubatch_idx]`
          and **ignores `ub_slice` / `running_bs`** (the group's request/token
          ranges come from the PcpBalGroup, not the ub_slice).
        - Token-split TBO (default, §11): uses `ub_slice` / `running_bs`.
        """
        from atom.utils.tbo.ubatch_splitting import split_attn_metadata

        # PCP+TBO request-boundary split: each ubatch = one request group processed as an
        # independent non-TBO PCP mini-batch. Slice the FULL (un-reindexed)
        # metadata to the group + call _apply_pcp_reindex on it (reuse the proven
        # reindex). Bypasses the token-split rebuild path entirely.
        if (
            getattr(self.model_runner, "_pcp_tbo_balanced_active", False)
            and getattr(self.model_runner, "_pcp_bal_groups", None) is not None
        ):
            return self._build_ubatch_prefill_metadata_balanced(
                attn_metadata, ubatch_idx
            )

        ub_attn = split_attn_metadata(attn_metadata, ub_slice, running_bs)
        ub_attn.__class__ = AttentionMetaData_DSV4

        src = cast(AttentionMetaData_DSV4, attn_metadata)
        rs = ub_slice.request_slice
        ts = ub_slice.token_slice
        ub_num_reqs = rs.stop - rs.start
        ub_num_tokens = ts.stop - ts.start

        if src.state_slot_out is not None:
            ub_attn.state_slot_out = src.state_slot_out[rs]
        if src.state_slot_in is not None:
            ub_attn.state_slot_in = src.state_slot_in[rs]
        if src.state_slot_out_cpu is not None:
            ub_attn.state_slot_out_cpu = src.state_slot_out_cpu[rs]

        var = self.model_runner.forward_vars
        positions_np = np.asarray(var["positions"].np[ts.start : ts.stop])
        full_cu = var["cu_seqlens_q"].np
        req_global_starts = full_cu[rs.start : rs.stop].astype(np.int64)
        req_global_ends = full_cu[rs.start + 1 : rs.stop + 1].astype(np.int64)
        clamped_starts = np.maximum(req_global_starts, ts.start)
        clamped_ends = np.minimum(req_global_ends, ts.stop)
        extend_lens_np = (clamped_ends - clamped_starts).astype(np.int32)
        ub_cu = np.zeros(ub_num_reqs + 1, dtype=np.int32)
        np.cumsum(extend_lens_np, dtype=np.int32, out=ub_cu[1:])
        ub_start_pos_for_ctx = positions_np[ub_cu[:ub_num_reqs]].astype(np.int32)
        context_lens_np = (ub_start_pos_for_ctx + extend_lens_np).astype(np.int32)
        from atom.model_ops.v4_kernels import make_compress_plans

        if self._unique_compress_ratios_overlap:
            # Per-ubatch plan buffers — sharing the main pool would let
            # ubatch 1's CPU build overwrite ubatch 0's before ubatch 0
            # launches its compressor kernel. TBO prefill is eager-only,
            # so leave running_bs/max_q_len unset (tight n_compress/n_write).
            ub_plan_buffers = self._get_ubatch_compress_plan_buffers(ubatch_idx)
            ub_attn.compress_plans = make_compress_plans(
                np.ascontiguousarray(extend_lens_np, dtype=np.int32),
                np.ascontiguousarray(context_lens_np, dtype=np.int32),
                self._unique_compress_ratios_overlap,
                plan_buffers=ub_plan_buffers,
            )
        else:
            ub_attn.compress_plans = {}

        # TBO path (_prepare_ubatch_decode). `_attach_v4_per_fwd_meta` reads
        # var[f"{p}context_lens"].np[:ub_num_reqs] for this ubatch's ctx lens;
        # its paged-decode branch is a no-op for prefill state, so only
        # context_lens needs staging into the prefixed set here.
        p = f"ub{ubatch_idx}_"
        var[f"{p}context_lens"].np[:ub_num_reqs] = context_lens_np

        self._attach_v4_per_fwd_meta(
            ub_attn,
            extend_lens_np,  # ubatch's per-seq token counts
            ub_attn.state_slot_out_cpu,
            ub_num_reqs,
            ub_num_tokens,
            buf_prefix_ubatch=p,
        )

        positions_gpu = var["positions"].gpu[ts.start : ts.stop]
        self._attach_v4_indexer_meta(
            ub_attn,
            ub_num_reqs,
            ub_num_tokens,
            positions_gpu=positions_gpu,
            buf_prefix_ubatch=p,
        )

        # start_pos = position of first token of each seq in this ubatch.
        ub_start_pos_per_seq_np = positions_np[ub_cu[:ub_num_reqs]]
        ub_positions_gpu = var["positions"].gpu[ts.start : ts.stop]
        ub_block_tables_gpu = var["block_tables"].gpu[rs.start : rs.stop]
        ub_cu_q_per_seq_gpu = torch.from_numpy(
            np.ascontiguousarray(ub_cu[:ub_num_reqs], dtype=np.int32)
        ).to(self.device, non_blocking=True)
        self._build_paged_prefill_meta(
            ub_attn,
            positions_np,
            ub_cu,
            extend_lens_np,
            ub_start_pos_per_seq_np,
            ub_attn.state_slot_out_cpu,
            ub_num_reqs,
            ub_num_tokens,
            positions_gpu=ub_positions_gpu,
            cu_q_per_seq_gpu=ub_cu_q_per_seq_gpu,
            block_tables_gpu=ub_block_tables_gpu,
        )

        # `split_attn_metadata` computed ub_attn.cu_seqlens_q/k from RAW request
        # boundaries (orig_cu[rs] - base), which is WRONG for a straddling
        # request under token-midpoint splits: it counts the request's FULL
        # length instead of only the portion owned by this ubatch, so
        # cu_seqlens_q[-1] > ub_num_tokens and any kernel indexing by it goes
        # out of bounds (SIGABRT / GPU memory fault). Overwrite with the
        # token-window-clamped `ub_cu` already computed above. For non-
        # straddling splits these are identical, so this is a no-op there.
        ub_cu_gpu = torch.from_numpy(
            np.ascontiguousarray(ub_cu[: ub_num_reqs + 1], dtype=np.int32)
        ).to(self.device, non_blocking=True)
        ub_attn.cu_seqlens_q = ub_cu_gpu
        if extend_lens_np.size > 0:
            ub_attn.max_seqlen_q = int(extend_lens_np.max())
        # cu_seqlens_k consistent with the clamped q lens (V4 prefill prefix KV
        # is read via per-ratio kv_indices_prefix_* buffers, not cu_seqlens_k).
        if ub_attn.cu_seqlens_k is not None:
            ub_attn.cu_seqlens_k = ub_cu_gpu

        # Clone all GPU tensors that are views into shared CpuGpuBuffers.
        # Without this, building the next ubatch overwrites this ubatch's
        # data via the same underlying buffer.
        if ub_attn.batch_id_per_token is not None:
            ub_attn.batch_id_per_token = ub_attn.batch_id_per_token.clone()
        if ub_attn.n_committed_csa_per_seq is not None:
            ub_attn.n_committed_csa_per_seq = ub_attn.n_committed_csa_per_seq.clone()
        if ub_attn.indexer_meta is not None:
            im = ub_attn.indexer_meta
            if im.get("cu_committed_gpu") is not None:
                im["cu_committed_gpu"] = im["cu_committed_gpu"].clone()
            if im.get("batch_id_per_token_gpu") is not None:
                im["batch_id_per_token_gpu"] = im["batch_id_per_token_gpu"].clone()

        return ub_attn

    def _build_ubatch_prefill_metadata_balanced(
        self,
        attn_metadata: AttentionMetaData,
        ubatch_idx: int,
    ) -> AttentionMetaData_DSV4:
        """PCP+TBO request-boundary split: build one request group's metadata as an
        independent non-TBO PCP mini-batch.

        `attn_metadata` is the FULL, UN-reindexed metadata (global). We slice it
        to this group's requests + global token range, then run the proven
        `_apply_pcp_reindex` on the group (pads the group to a pcp multiple and
        round-robin strides to 1/pcp — matching run_model's per-group stripe).
        Per-seq / KV-write fields (cu_seqlens_q, compress_plans, state_slot) stay
        GLOBAL for the group (the compressor/swa_write see the group's full
        all-gathered tokens), exactly as non-TBO PCP does for the whole batch.
        """
        from atom.model_ops.v4_kernels import make_compress_plans
        from atom.utils.tbo.ubatch_splitting import UBatchSlice, split_attn_metadata

        mr = self.model_runner
        grp = mr._pcp_bal_groups[ubatch_idx]  # PcpBalGroup
        rs0, rs1 = grp.req_start, grp.req_stop
        gts, gte = grp.tok_start, grp.tok_end
        group_bs = rs1 - rs0
        group_total = gte - gts  # group's global token count (real, pre-pad)
        device = self.device
        var = mr.forward_vars
        src = cast(AttentionMetaData_DSV4, attn_metadata)

        # ---- base fields via split on the GROUP's GLOBAL token range ----
        # full metadata is global, so a global token_slice slices cu_seqlens_q /
        # slot_mapping / context_lens correctly (per-request, rebased).
        g_slice = UBatchSlice(
            request_slice=slice(rs0, rs1),
            token_slice=slice(gts, gte),
        )
        ub = split_attn_metadata(attn_metadata, g_slice, group_bs)
        ub.__class__ = AttentionMetaData_DSV4
        # split_attn_metadata doesn't carry these: state drives prefill/decode
        # dispatch; indexer_meta must be non-None so _apply_pcp_reindex rebuilds
        # it for the group (it rebuilds from batch_id+positions, ignoring content).
        ub.state = src.state
        ub.indexer_meta = src.indexer_meta

        # ---- per-seq DSV4 fields sliced by request ----
        if src.state_slot_out is not None:
            ub.state_slot_out = src.state_slot_out[rs0:rs1].contiguous()
        if src.state_slot_in is not None:
            ub.state_slot_in = src.state_slot_in[rs0:rs1].contiguous()
        if src.state_slot_out_cpu is not None:
            ub.state_slot_out_cpu = src.state_slot_out_cpu[rs0:rs1]
        if src.n_committed_csa_per_seq is not None:
            ub.n_committed_csa_per_seq = src.n_committed_csa_per_seq[
                rs0:rs1
            ].contiguous()
        if src.n_committed_csa_per_seq_cpu is not None:
            ub.n_committed_csa_per_seq_cpu = src.n_committed_csa_per_seq_cpu[rs0:rs1]

        # ---- per-token DSV4 fields sliced by the GLOBAL token range [gts,gte) ----
        owned = torch.arange(gts, gte, device=device)
        for ind_attr, idx_attr in (
            ("kv_indptr_prefix_swa", "kv_indices_prefix_swa"),
            ("kv_indptr_prefix_csa", "kv_indices_prefix_csa"),
            ("kv_indptr_prefix_hca", "kv_indices_prefix_hca"),
            ("kv_indptr_extend", "kv_indices_extend"),
        ):
            indptr = getattr(src, ind_attr, None)
            indices = getattr(src, idx_attr, None)
            if indptr is None or indices is None:
                continue
            ni, nx = pcp_reindex_ragged(indptr, indices, owned)
            # kv_indices_extend are ROW offsets into the per-fwd kv_full tensor.
            # In the full metadata they index the WHOLE sequence's kv_full [0,T);
            # for this group kv_full only holds the group's tokens (global order
            # [gts,gte) → rows [0, gte-gts)), so rebase by gts. (prefix indices
            # point into unified_kv by absolute cache slot — no rebase.) Balanced
            # splits on request boundaries so each query's SWA window stays within
            # its sequence (within the group) → row >= gts, rebased value >= 0.
            if idx_attr == "kv_indices_extend" and nx.numel() > 0:
                nx = nx - gts
            setattr(ub, ind_attr, ni)
            setattr(ub, idx_attr, nx)
        # batch_id_per_token: slice + rebase global req id → group-local (keep -1).
        if src.batch_id_per_token is not None:
            bid = src.batch_id_per_token[gts:gte].clone()
            ub.batch_id_per_token = torch.where(bid >= 0, bid - rs0, bid)
        if src.skip_prefix_len_csa is not None:
            ub.skip_prefix_len_csa = src.skip_prefix_len_csa[gts:gte].contiguous()
        ub.envelope_rows = src.envelope_rows

        # ---- compress_plans: group's GLOBAL per-request (compressor all-gathers
        # the group to full order). Built from global cu / context_lens slices. ----
        if self._unique_compress_ratios_overlap:
            gcu = var[
                "cu_seqlens_q"
            ].np  # GLOBAL (not overwritten for request-boundary split)
            ext = (gcu[rs0 + 1 : rs1 + 1] - gcu[rs0:rs1]).astype(np.int32)
            ctx = np.asarray(var["context_lens"].np[rs0:rs1], dtype=np.int32)
            plan_bufs = self._get_ubatch_compress_plan_buffers(ubatch_idx)
            ub.compress_plans = make_compress_plans(
                np.ascontiguousarray(ext, dtype=np.int32),
                np.ascontiguousarray(ctx, dtype=np.int32),
                self._unique_compress_ratios_overlap,
                plan_buffers=plan_bufs,
                decode_capacity_per_ratio=None,
            )
        else:
            ub.compress_plans = {}

        # ---- reindex the group to 1/pcp (proven path) ----
        # positions: group's GLOBAL positions (forward_vars stay global for the
        # request-boundary split). _apply_pcp_reindex pads group_total to pcp + strides —
        # matching run_model's per-group pcp_round_robin_split.
        group_positions = var["positions"].gpu[gts:gte]
        self._apply_pcp_reindex(ub, group_positions, group_bs, group_total)

        # max_seqlen_q from the group's per-request extend lengths.
        if ub.cu_seqlens_q is not None and group_bs > 0:
            per_req_q = ub.cu_seqlens_q[1 : group_bs + 1] - ub.cu_seqlens_q[:group_bs]
            if per_req_q.numel() > 0:
                ub.max_seqlen_q = int(per_req_q.max().item())

        # Clone GPU tensors that are slices/views into shared CpuGpuBuffers, so a
        # later ubatch (or fwd) reusing the same buffer can't overwrite this
        # ubatch's data (mirrors the token-split path's clones).
        # n_committed_csa_per_seq is a view of src's shared buffer (the [rs0:rs1]
        # .contiguous() slice above stays a view when already contiguous).
        if ub.n_committed_csa_per_seq is not None:
            ub.n_committed_csa_per_seq = ub.n_committed_csa_per_seq.clone()
        if ub.indexer_meta is not None:
            im = ub.indexer_meta
            for k in (
                "cu_committed_gpu",
                "batch_id_per_token_gpu",
            ):
                if im.get(k) is not None:
                    im[k] = im[k].clone()
        return ub

    def _attach_v4_per_fwd_meta(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        token_num_per_seq,
        state_slot_out_cpu,
        scheduled_bs: int,
        scheduled_tokens: int,
        *,
        running_bs: int | None = None,
        max_q_len: int | None = None,
        running_tokens: int | None = None,
        buf_prefix_ubatch: str = "",
    ) -> None:
        """Hoist per-fwd, layer-invariant metadata used by every V4 layer.

        These tensors only depend on `positions`, `cu_seqlens_q`, `state_slot_out`
        and `window_size` — none of which change across layers — so building
        them once per fwd saves ~64 redundant constructions for V4-Pro.

        Sets:
          - `attn_metadata.batch_id_per_token`: [padded_T] int32 batch id
            per token (single per-token mapping; consumed by the Phase B/C/E
            paged-decode kernels and the indexer). `swa_write` no longer
            depends on this — it derives `src_id` from `cu_seqlens_q` inline.
          - `attn_metadata.n_committed_csa_per_seq`: [bs] int32 per-seq
            `ctx_len // 4`, for the two consumers that are genuinely per-seq —
            the indexer's `cu_committed` cumsum and the FP4 ragged windows.
          - `attn_metadata.state_slot_out`: [bs] int32 GPU view of
            per-seq state cache slot (already set by prepare_*; passed
            through unchanged here).

        Caller contract: `scheduled_bs >= 1` and `scheduled_tokens >= 1`.
        warmup_model + dummy_run paths both enforce these via min-1 fallbacks
        (model_runner.warmup_model:1003-1011, _populate_state_slot_mappings
        zeros-fill); CG capture uses running_bs >= 1 too.
        """
        # state is set by the caller at AttentionMetaData_DSV4 construction
        # time (single source of truth — prepare_decode / prepare_prefill /
        # prepare_mtp_decode / build_for_cudagraph_capture each set it).
        # Consumed here for padded_total_tokens sizing.
        is_pure_decode = attn_metadata.state is AttnState.DECODE

        # padded_total_tokens: CG-captured decode/MTP pads to the fixed
        # bucket `running_bs * (1+max_spec_steps)` so the per-token
        # `batch_id_per_token` buffer has a stable shape across captures.
        # Prefill states (PREFILL_NATIVE / PREFILL_PREFIX) are eager and
        # use `scheduled_tokens` exactly — no wasted padding (a long prefill
        # chunk doesn't need to be padded up to a bucket that doesn't
        # exist for it).
        if is_pure_decode:
            assert running_bs is not None and max_q_len is not None, (
                "DECODE state requires running_bs + max_q_len from caller "
                "(CG bucket size — fixed at capture)"
            )
            padded_total_tokens = (
                int(running_tokens)
                if running_tokens is not None
                else int(running_bs) * int(max_q_len)
            )
        else:
            padded_total_tokens = scheduled_tokens

        var = self.model_runner.forward_vars

        # ---- CPU numpy work (all on main thread) ----
        # The mapping exists on the host only to be staged: head = real, tail =
        # -1 sentinel, and every consumer downstream is a kernel reading the
        # device copy.
        batch_id_per_token_np = np.full(padded_total_tokens, -1, dtype=np.int32)
        batch_id_per_token_np[:scheduled_tokens] = np.repeat(
            np.arange(scheduled_bs, dtype=np.int32), token_num_per_seq
        )

        # context_lens is int32 on the buffer; keep dtype through divide so
        # n_committed_csa stays int32 (max value ~max_model_len // 4 ≪ 2^31).
        ctx_per_seq_np = var[f"{buf_prefix_ubatch}context_lens"].np[:scheduled_bs]
        # Single source of truth for n_committed_csa_per_seq on CPU. Stashed on
        # attn_metadata so `_attach_v4_indexer_meta` reads it instead of
        # re-running `ctx // 4`. HCA has no twin here: every consumer wants the
        # count a TOKEN may see, which is `(pos+1)//128`.
        n_committed_csa_per_seq_np = ctx_per_seq_np // 4
        attn_metadata.n_committed_csa_per_seq_cpu = n_committed_csa_per_seq_np

        # ---- Stage all buffers to GPU ----
        # window_topk used to be CPU-built here ([T, win] of ring indices with
        # -1 sentinels) and staged via v4_meta_window_topk. Now the ring index
        # is computed inline inside `write_v4_paged_decode_indices` kernel
        # from `var["positions"].gpu` — saves O(T·win) numpy work + 4 MB
        # staging buffer. The `positions` H2D is already done by the caller.
        attn_metadata.batch_id_per_token = self._stage(
            f"{buf_prefix_ubatch}v4_batch_id_per_token", batch_id_per_token_np
        )
        # Stage n_committed to GPU. For CG-replay safety: aiter
        # `top_k_per_row_decode` iterates the CAPTURED grid (= running_bs *
        # next_n rows) and reads `rowEnds[batch_id]` for every row. Its
        # per-row length formula is
        #   `row_len = rowEnds[bid] - next_n + (r % next_n) + 1`
        # — for pad rows `bid ∈ [scheduled_bs, running_bs)` the buffer slot
        # carries a stale value from a prior fwd; if that stale value is
        # `< next_n - 1` (easy with MTP3 next_n=4 if a prior fwd had a seq
        # in early prefill with ctx ≤ 11), row_len becomes negative and the
        # kernel's radix loop runs unbounded → GPU hang. The downstream
        # `batch_id_per_token = -1` sentinel masks pad rows out of
        # `csa_translate_pack`, so the value just needs to be "big enough"
        # to keep row_len non-negative. Use `index_topk` (≥ 1024 ≫ next_n).
        n_csa_buf = var[f"{buf_prefix_ubatch}v4_n_committed_csa_per_seq"]
        n_csa_buf.np[:scheduled_bs] = n_committed_csa_per_seq_np
        if is_pure_decode and running_bs is not None and running_bs > scheduled_bs:
            n_csa_buf.np[scheduled_bs:running_bs] = self.index_topk
            attn_metadata.n_committed_csa_per_seq = n_csa_buf.copy_to_gpu(running_bs)
        else:
            attn_metadata.n_committed_csa_per_seq = n_csa_buf.copy_to_gpu(scheduled_bs)
        self._attach_v4_paged_decode_meta(
            attn_metadata=attn_metadata,
            state_slot_out_cpu=state_slot_out_cpu,
            scheduled_bs=scheduled_bs,
            total_tokens=scheduled_tokens,
            padded_total_tokens=padded_total_tokens,
            buf_prefix_ubatch=buf_prefix_ubatch,
        )

    def _attach_v4_paged_decode_meta(
        self,
        attn_metadata,
        state_slot_out_cpu,
        scheduled_bs: int,
        total_tokens: int,
        padded_total_tokens: int | None = None,
        buf_prefix_ubatch: str = "",
    ) -> None:
        """Phase B: build per-fwd paged-decode index buffers (layer-invariant).

        All three per-token regions are RAGGED-PACKED — same layout family as
        the prefill path (`_build_paged_prefill_meta`). Per-token slot count,
        with `n = min(positions[t]+1, win)`:
          SWA: n
          CSA: n + min((pos+1)//4, index_topk)
          HCA: n + (pos+1)//128
        The cumsums and the CSA visibility are built by
        `build_v4_paged_decode_indptr` — one launch, no host arithmetic, no H2D.

        Writes into stable forward_vars buffers (attn_metadata fields are
        the V4-namespaced counterparts on `AttentionMetaData_DSV4`):
          - kv_indices_swa : per-token SWA paged offsets, ragged-packed
          - kv_indices_csa : SWA prefix at slice TAIL; CSA compress section
                             (slice head) left UNINITIALIZED — V4Attention.
                             forward fills it per-layer via csa_translate_pack
                             (Phase C)
          - kv_indices_hca : HCA compress section (head) + SWA prefix (tail),
                             both fully written (HCA is layer-invariant)
          - kv_indptr_{swa,csa,hca} : 3 ragged cumsums. Padded tail repeats
                             last value → kv_len=0 sentinel for CG-padded slots.
          - skip_prefix_len_csa : per-token SWA prefix length (the tail
                             segment); csa_translate_pack uses it to recover
                             the CSA topk length valid_k = slice_len - skip.
                             Decode derives it inline from positions.

        Reuses (built earlier in `_attach_v4_per_fwd_meta`):
          - batch_id_per_token : single per-token mapping (with -1 sentinel)
          - var["positions"] : global token positions (already H2D-copied by
                               the caller; consumed by both kernels here)
        `n_committed_csa_per_seq` is NOT among them: every count wanted here is
        one a TOKEN may see, and that follows from its position.

        Skipped when state is not DECODE. The Phase-B fields
        (kv_indices_*, kv_indptr_*, envelope_rows) stay at their dataclass
        defaults for prefill batches; downstream V4Attention.forward branches
        on state and reads prefill-mode buffers (kv_indices_prefix_*) instead.
        """
        if scheduled_bs == 0 or total_tokens == 0:
            return  # fields stay at dataclass defaults

        if attn_metadata.state is not AttnState.DECODE:
            return  # prefill: only kv_indices_prefix_* are built downstream

        if len(state_slot_out_cpu) < scheduled_bs:
            # Defensive carve-out: caller asserted DECODE but
            # state_slot_out is incomplete. Flip state to PREFILL_NATIVE.
            attn_metadata.state = AttnState.PREFILL_NATIVE
            return

        var = self.model_runner.forward_vars
        win = self.window_size  # per-token max SWA prefix slots
        envelope_rows = self.pool_geometry.envelope_rows

        T = total_tokens

        # The single per-token mapping, staged once in `_attach_v4_per_fwd_meta`.
        # Only the device copy is wanted here now: every consumer below is a
        # kernel, so the host mirror this used to fancy-index has no reader.
        batch_id_per_token_gpu = attn_metadata.batch_id_per_token

        index_topk = self.index_topk

        # CG-padding-aware T_for_indptr: indptr buffer must size to the
        # captured kernel grid (= padded_total_tokens) so padded slots see
        # `kv_len = indptr[t+1] - indptr[t] = 0` and the inner loop bails.
        T_pad = (
            total_tokens if padded_total_tokens is None else int(padded_total_tokens)
        )
        T_pad = max(T_pad, T)

        # DSpark's fp8 indexer wants the per-token visibility laid out as one
        # row per SLOT of a `full_q`-wide rectangle, right-aligned per sequence,
        # rather than one row per token. It is the same number either way — only
        # where it lands differs, which is why the rectangle is a parameter of
        # the builder below and not a second builder.
        full_q = int(getattr(attn_metadata, "dspark_full_q", 0))
        rect_full_q = full_q if (full_q > 0 and not self._indexer_fp4) else 0
        if rect_full_q:
            ragged_lens_gpu = getattr(attn_metadata, "dspark_ragged_lens_gpu", None)
            rect_bs = max(
                scheduled_bs,
                int(ragged_lens_gpu.shape[0]) if ragged_lens_gpu is not None else 0,
            )
            # Rows are `full_q` slots per sequence, so row r serves seq
            # r // full_q, and the rows past the batch are the tail.
            mqa_rows = rect_bs * full_q
            mqa_valid_rows = scheduled_bs * full_q
            mqa_row_to_batch_gpu = torch.arange(
                mqa_valid_rows, dtype=torch.int32, device=self.device
            ).div_(full_q, rounding_mode="floor")
        else:
            # One row per query token, so the row -> seq map is the one
            # already staged for this forward.
            mqa_rows = T_pad
            mqa_valid_rows = T
            mqa_row_to_batch_gpu = batch_id_per_token_gpu[:T]

        # A `[:n]` slice past the end truncates silently, so the bounds are
        # checked rather than relied on — `_stage` used to carry them. Two host
        # scalars cover every buffer below, because `_alloc_v4_metadata_buffers`
        # sizes all of them off `max_decode_tokens` times a per-token worst case
        # no token can exceed. Checking a pool's actual sum would need a D2H —
        # the one sync this build exists to avoid.
        #
        # `ValueError`, not `assert`, for the reason `require_step_within_full_q`
        # gives: a truncated slice is a device-side out-of-bounds write, and
        # `python -O` strips asserts. The rect `mqa_rows` has no second line of
        # defence — `build_v4_paged_decode_indptr` recomputes its band count from
        # whatever length it is handed, so a short buffer looks well-formed.
        if T_pad > self.max_decode_tokens:
            raise ValueError(
                f"V4 decode built {T_pad} tokens but the index buffers are sized "
                f"for {self.max_decode_tokens} (max_bs * (1 + max_spec_steps))."
            )
        if mqa_rows > self.max_decode_tokens:
            raise ValueError(
                f"V4 decode wants {mqa_rows} indexer rows but the per-token "
                f"buffers hold {self.max_decode_tokens}. Increase "
                f"max_decode_tokens."
            )

        # The three ragged cumsums and the CSA per-token visibility, in one
        # launch off buffers already resident, replacing three numpy cumsums and
        # four H2D copies of arrays whose inputs never left the device.
        swa_indptr_gpu = var[f"{buf_prefix_ubatch}v4_kv_indptr_swa"][: T_pad + 1]
        csa_indptr_gpu = var[f"{buf_prefix_ubatch}v4_kv_indptr_csa"][: T_pad + 1]
        hca_indptr_gpu = var[f"{buf_prefix_ubatch}v4_kv_indptr_hca"][: T_pad + 1]
        csa_ncmt_gpu = var[f"{buf_prefix_ubatch}v4_csa_n_committed_per_token"][
            :mqa_rows
        ]
        build_v4_paged_decode_indptr(
            batch_id_per_token=batch_id_per_token_gpu,
            positions=var[f"{buf_prefix_ubatch}positions"].gpu,
            swa_indptr=swa_indptr_gpu,
            csa_indptr=csa_indptr_gpu,
            hca_indptr=hca_indptr_gpu,
            csa_n_committed_per_token=csa_ncmt_gpu,
            T_pad=T_pad,
            win=win,
            index_topk=index_topk,
            rect_full_q=rect_full_q,
            ragged_lens=attn_metadata.dspark_ragged_lens_gpu if rect_full_q else None,
            cu_q_per_seq=(
                var[f"{buf_prefix_ubatch}cu_seqlens_q"].gpu if rect_full_q else None
            ),
        )

        # Expand block tables per query row so the unchanged aiter paged-MQA
        # kernels can run once with shape `[decode_rows, 1, ...]`. Source and
        # index are both on the device, so the gather runs there instead of the
        # host shipping the expansion; the rows it skips are the tail.
        mqa_bt = var[f"{buf_prefix_ubatch}v4_block_tables_per_token"]
        block_tables_per_token_gpu = mqa_bt[:mqa_rows]
        torch.index_select(
            var[f"{buf_prefix_ubatch}block_tables"].gpu,
            0,
            mqa_row_to_batch_gpu,
            out=block_tables_per_token_gpu[:mqa_valid_rows],
        )
        if mqa_valid_rows < mqa_rows:  # an empty `zero_` still costs a dispatch
            block_tables_per_token_gpu[mqa_valid_rows:].zero_()

        # HCA compress section: the kernel below fills it from block tables
        # already on the device, tiling each slice exactly
        # with the SWA prefix, so nothing pre-fills the buffer.
        hca_indices_buf = var[f"{buf_prefix_ubatch}v4_kv_indices_hca"]

        # ----- Write SWA / CSA / HCA window-prefix paged offsets (1 kernel) -----
        # Kernel computes `n = min(positions[t]+1, win)` and ring-index
        # `(positions[t] - n + 1 + i) % cs` inline — no window_topk staging.
        # See `write_v4_paged_decode_indices` docstring and plan
        # `sequential-noodling-turing.md` for the motivation. Reads only
        # persistent forward_vars buffers — no allocator churn (the prior
        # `index_copy_` chain raced under MTP-3 long-prefill; this kernel
        # also fixes that, see skill `debug-agent-locate-kernel`).
        swa_indices_gpu = var[f"{buf_prefix_ubatch}v4_kv_indices_swa"]
        csa_indices_gpu = var[f"{buf_prefix_ubatch}v4_kv_indices_csa"]
        dest_rows = self._dest_row_buffers(buf_prefix_ubatch)
        write_v4_paged_decode_indices(
            # The slot array must come from the SAME buffer set as
            # batch_id_per_token. In a TBO ubatch the latter holds LOCAL req
            # indices [0, ub_real_reqs), so this must be the ubatch-sliced
            # buffer whose row i == local req i, not the global one (row i ==
            # global req i). Using the global array here makes ubatch1
            # (req_start>0) read other requests' SWA rings → cross-request KV
            # contamination, wrong output without a crash. block_tables_np_full
            # above (HCA) already uses the prefixed buffer; this must match.
            # Off the ubatch path the prefix is "" and this is the global one.
            state_slot_per_seq=var[f"{buf_prefix_ubatch}v4_meta_state_slot_out"].gpu[
                :scheduled_bs
            ],
            batch_id_per_token=batch_id_per_token_gpu,
            positions=var[f"{buf_prefix_ubatch}positions"].gpu,
            swa_indptr=swa_indptr_gpu,
            csa_indptr=csa_indptr_gpu,
            hca_indptr=hca_indptr_gpu,
            swa_indices=swa_indices_gpu,
            csa_indices=csa_indices_gpu,
            hca_indices=hca_indices_buf,
            dest_rows=dest_rows,
            T=T,
            win=win,
            geometry=self.pool_geometry,
            # Same buffer set as batch_id_per_token, for the reason above.
            hca_block_tables=var[f"{buf_prefix_ubatch}block_tables"].gpu,
            hca_rows_per_block=self.hca_rows_per_block,
        )
        attn_metadata.swa_dest_rows = dest_rows

        # `skip_prefix_len_csa` is not materialized on the decode path —
        # `csa_translate_pack` is invoked with `window_size = self.window_size`
        # so the kernel derives `skip = min(positions[t]+1, win)` inline, which
        # is the same `n` the indptr build above sizes each SWA prefix by.
        # Saves a CPU write + H2D per fwd. Prefill cannot derive it from
        # positions (skip depends on `chunk_start`) and uploads its own tensor
        # in `_build_paged_prefill_meta`.

        # ----- Stash on attn_metadata for V4Attention.forward consumption -----
        # batch_id_per_token + n_committed_csa_per_seq already set in
        # `_attach_v4_per_fwd_meta` (single source of truth, also consumed by
        # swa_write / indexer outside the is_pure_decode branch).
        # is_pure_decode was set by the caller at AttentionMetaData_DSV4
        # construction time; we only flip it (True→False) above when the
        # warmup carve-out fires (incomplete state_slot_out_cpu).
        # Published WHOLE, not sliced to `indptr_np[T]`. That length is the
        # cumsum of per-token KV spans, so it varies step to step at a FIXED
        # num_tokens and no graph key pins it -- and a cudagraph bakes it. It
        # only ever stayed correct because the buffer is sized for the worst
        # case, so overshooting stayed inside the allocation.
        #
        # Nothing reads the length: `sparse_attn_v4_paged_decode` walks
        # `kv_indices[kv_indptr[t] : kv_indptr[t+1]]` (see `paged_decode.py:104`)
        # and `csa_translate_pack`'s grid comes from `topk_local.shape`. Both now
        # sit in a dense piece keyed on num_tokens alone, so this matters under
        # plain PIECEWISE too. `prepare_mtp_decode` (~:2381) and the sglang
        # bridge already publish these whole. HCA is published the same way: its
        # capacity is checked on the premises (`T_pad` against the per-token
        # worst case the pool is sized for) rather than on the sum.
        attn_metadata.kv_indices_swa = swa_indices_gpu
        attn_metadata.kv_indices_csa = csa_indices_gpu
        attn_metadata.csa_n_committed_per_token = csa_ncmt_gpu
        attn_metadata.block_tables_per_token = block_tables_per_token_gpu
        attn_metadata.kv_indices_hca = hca_indices_buf
        attn_metadata.kv_indptr_swa = swa_indptr_gpu
        attn_metadata.kv_indptr_csa = csa_indptr_gpu
        attn_metadata.kv_indptr_hca = hca_indptr_gpu
        attn_metadata.envelope_rows = envelope_rows

        # Per-token paged-decode index tensors for the fp8 asm decode kernel. The
        # kernel sees N = q_packed.shape[0] = T_pad (padded decode grid). Both
        # are re-staged every fwd (like kv_indptr_*) so the captured graph sees a
        # freshly-copied backing store at replay.
        # qo_indptr: per-token q indptr (page_size=1, max_seqlen_q=1). The REAL
        # region [0..T] is arange(T+1) — one 1-length query per real decode token.
        # The CG-padded tail [T+1..T_pad] must NOT keep counting up: repeating
        # the last real value makes each padded slot a 0-length query
        # (qo_indptr[t+1]-qo_indptr[t]==0) that the asm kernel bails on, exactly
        # like the kv_indptr pad tail. Per-token, so correct for MTP too.
        if self._kv_fp8:
            qo_indptr_np = np.empty(T_pad + 1, dtype=np.int32)
            qo_indptr_np[: T + 1] = np.arange(T + 1, dtype=np.int32)
            if T_pad > T:
                qo_indptr_np[T + 1 :] = T
            attn_metadata.qo_indptr = self._stage("v4_qo_indptr", qo_indptr_np)

    def _build_paged_prefill_meta(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        positions_np: np.ndarray,
        cu_seqlens_q_np: np.ndarray,
        token_num_per_seq: np.ndarray,
        start_pos_per_seq_np: np.ndarray,
        state_slot_out_cpu: np.ndarray,
        scheduled_bs: int,
        total_tokens: int,
        *,
        positions_gpu: torch.Tensor | None = None,
        cu_q_per_seq_gpu: torch.Tensor | None = None,
        block_tables_gpu: torch.Tensor | None = None,
    ) -> None:
        """Build per-fwd index buffers consumed by sparse_attn_v4_paged_prefill.

        Two-source layout:
          - prefix region (per-ratio): SWA history from prior chunks + CSA topk
            OR the HCA groups closed at or before the token's own position,
            from `unified_kv`. Three buffers (Dense / CSA / HCA) per fwd.
          - extend region (shared): in-chunk SWA tail from per-fwd `kv`
            tensor. One buffer.

        Per-token length formulas:
          extend_count[t]      = min(token_pos_in_chunk[t] + 1, win)
          prefix_swa_count[t]  = max(0, chunk_start[bid] - max(0, p_global - win + 1))
          prefix_swa_count[t] + extend_count[t] = min(p_global + 1, win)

        Per-ratio prefix kv_len:
          Dense:  prefix_swa_count[t]
          CSA:    prefix_swa_count[t] + min((p_global+1)//4, index_topk)
          HCA:    prefix_swa_count[t] + (p_global+1)//128

        Eager-only (chunked prefill is dynamic-shaped; no CG capture). Per-fwd
        `torch.from_numpy(...).to(device, non_blocking=True)` avoids stream drain.

        Builder fills: extend buffer, prefix_swa buffer (Dense), HCA section
        of prefix_hca buffer, SWA prefix sections of all 3 prefix buffers.
        Per-layer csa_translate_pack later fills the CSA section of
        prefix_csa buffer.

        Sets attn_metadata fields (per `AttentionMetaData_DSV4` docstrings):
          - kv_indices_extend / kv_indptr_extend (shared)
          - kv_indices_prefix_swa / kv_indptr_prefix_swa  (Dense)
          - kv_indices_prefix_csa / kv_indptr_prefix_csa  (CSA, CSA section UNINIT)
          - kv_indices_prefix_hca / kv_indptr_prefix_hca  (HCA, fully filled)
          - skip_prefix_len_csa = prefix_swa_count_per_token (per-token)
          - envelope_rows
        """
        assert scheduled_bs >= 1 and total_tokens >= 1, (
            "scheduled_bs and total_tokens must be positive for prefill meta "
            "build (got scheduled_bs={scheduled_bs}, total_tokens={total_tokens})"
        )

        device = self.device
        win = self.window_size  # per-token topk count
        index_topk = self.index_topk
        T = total_tokens
        # warmup_model runs BEFORE allocate_kv_cache binds the paged pool
        # (sub-pool sizing has not run, so the state class is still empty and
        # unified_kv is a 1-page placeholder). V4Attention.forward detects
        # `is_dummy_run` and
        # short-circuits the sparse_attn dispatch entirely, so we don't need
        # valid prefix/extend indices during warmup.
        num_slots = self.num_state_slots
        if num_slots == 0:
            return
        envelope_rows = self.pool_geometry.envelope_rows
        var = self.model_runner.forward_vars

        # ----- CPU numpy: per-token counts + indptrs -----
        # Same formulas as the old _segment_indices/scatter chain, just without
        # the segment-expansion + scatter steps — those are now done by the
        # GPU kernel below. numpy.cumsum gives us indptr totals for free
        # (no D2H sync needed to size output buffers).
        chunk_start_per_seq_np = np.asarray(
            start_pos_per_seq_np[:scheduled_bs], dtype=np.int32
        )
        token_num_per_seq = np.asarray(token_num_per_seq, dtype=np.int32)
        batch_id_per_token_np = np.repeat(
            np.arange(scheduled_bs, dtype=np.int32), token_num_per_seq
        )  # [T] int32
        positions_arr = np.asarray(positions_np[:T], dtype=np.int32)
        chunk_start_pt = chunk_start_per_seq_np[batch_id_per_token_np]
        token_pos_in_chunk = positions_arr - chunk_start_pt
        swa_low = np.maximum(positions_arr - win + 1, 0)

        extend_count_np = np.minimum(token_pos_in_chunk + 1, win).astype(np.int32)
        prefix_swa_count_np = np.maximum(chunk_start_pt - swa_low, 0).astype(np.int32)
        # These SIZE each slice while `_v4_paged_prefill_indices_kernel` FILLS
        # it, so both must stay the geometry's spelling or the tail is
        # uninitialized. Buffer size ↔ kernel-writes then match exactly and no
        # `-1` sentinel pre-fill is needed.
        csa_valid_k_per_token_np = np.minimum(
            visible_csa(positions_arr), index_topk
        ).astype(np.int32)
        n_hca_per_token_np = visible_hca(positions_arr).astype(np.int32)

        # 4 indptrs on CPU; last element = total (no D2H to size buffers).
        ext_indptr_np = np.zeros(T + 1, dtype=np.int32)
        ext_indptr_np[1:] = np.cumsum(extend_count_np, dtype=np.int32)
        swa_indptr_np = np.zeros(T + 1, dtype=np.int32)
        swa_indptr_np[1:] = np.cumsum(prefix_swa_count_np, dtype=np.int32)
        csa_indptr_np = np.zeros(T + 1, dtype=np.int32)
        csa_indptr_np[1:] = np.cumsum(
            prefix_swa_count_np + csa_valid_k_per_token_np, dtype=np.int32
        )
        hca_indptr_np = np.zeros(T + 1, dtype=np.int32)
        hca_indptr_np[1:] = np.cumsum(
            prefix_swa_count_np + n_hca_per_token_np, dtype=np.int32
        )
        ext_total = int(ext_indptr_np[T])
        swa_total = int(swa_indptr_np[T])
        csa_total = int(csa_indptr_np[T])
        hca_total = int(hca_indptr_np[T])

        # ----- H2D: 4 indptrs + 2 per-seq scalars -----
        # Sources are per-call temp np arrays, so not a cross-ubatch race source
        # (the shared-pinned-buffer race is handled by the stream sync before
        # build_ubatch_prefill_metadata's finally). Via `upload_numpy` because
        # the indptrs are `T + 1` long: 64 KB at mnbt 16384, over the pageable
        # cliff at mnbt 131072.
        chunk_start_per_seq_gpu = upload_numpy(chunk_start_per_seq_np, device)
        ext_indptr = upload_numpy(ext_indptr_np, device)
        swa_indptr = upload_numpy(swa_indptr_np, device)
        csa_indptr = upload_numpy(csa_indptr_np, device)
        hca_indptr = upload_numpy(hca_indptr_np, device)

        # Reuse already-on-GPU tensors (populated upstream).
        # Cast positions to int32: production var["positions"] is int64 but
        # the kernel was designed/tested against int32 (downstream paged
        # offsets stored in int32 buffers; int32 throughout avoids mixed-
        # dtype Triton arithmetic that can silently truncate).
        if positions_gpu is None:
            positions_gpu = var["positions"].gpu[:T]
        if cu_q_per_seq_gpu is None:
            cu_q_per_seq_gpu = var["cu_seqlens_q"].gpu[:scheduled_bs]
        if block_tables_gpu is None:
            block_tables_gpu = var["block_tables"].gpu[:scheduled_bs]
        # SWA-prefix offsets are ring-addressed by the request's state slot;
        # HCA still uses the compressed block_tables.
        state_slot_per_seq_gpu = attn_metadata.state_slot_out[:scheduled_bs]
        # batch_id_per_token is int32 in storage (accepted by PyTorch
        # advanced-indexing and the fused flydsl SWA scatter); the kernel uses
        # tl.load which is dtype-agnostic.
        bid_per_token_gpu = attn_metadata.batch_id_per_token[:T]

        # ----- Allocate output buffers (exact sizes known from CPU totals) -----
        ext_indices = torch.empty(max(ext_total, 1), dtype=torch.int32, device=device)
        swa_indices = torch.empty(max(swa_total, 1), dtype=torch.int32, device=device)
        csa_indices = torch.empty(max(csa_total, 1), dtype=torch.int32, device=device)
        hca_indices = torch.empty(max(hca_total, 1), dtype=torch.int32, device=device)
        # NB: no `csa_indices.fill_(-1)` — per-token CSA reservation now
        # matches Indexer visibility exactly (csa_valid_k_per_token), so
        # csa_translate_pack writes every reserved cell.
        # PCP exception: under prefill context parallel, _apply_pcp_reindex
        # rebuilds this buffer via pcp_reindex_ragged into a FRESH torch.empty
        # tensor and re-slices the indptr, so the "every cell written" invariant
        # no longer holds (the CSA-topk section is filled per-layer in forward
        # AFTER reindex). Restore the -1 sentinel the consumer relies on: fill
        # BEFORE the builder kernel writes the SWA section, so SWA stays real and
        # unwritten cells stay -1 through reindex until csa_translate_pack
        # overwrites the CSA head. pcp=1 keeps the original zero-fill fast path.
        if pcp_is_enabled():
            csa_indices.fill_(-1)

        # ----- Single Triton kernel: scatter SWA-prefix / extend / HCA-compress -----
        write_v4_paged_prefill_indices(
            positions=positions_gpu,
            bid_per_token=bid_per_token_gpu,
            chunk_start_per_seq=chunk_start_per_seq_gpu,
            cu_seqlens_q_per_seq=cu_q_per_seq_gpu,
            state_slot_per_seq=state_slot_per_seq_gpu,
            block_tables=block_tables_gpu,
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
            geometry=self.pool_geometry,
            hca_rows_per_block=self.hca_rows_per_block,
        )

        # ----- skip_prefix_len_csa: per-token SWA prefix length -----
        # csa_translate_pack consumes this to derive the CSA topk length
        # `valid_k = (indptr[t+1]-indptr[t]) - skip` it writes at the HEAD of
        # `kv_indices_prefix_csa[indptr[t]:indptr[t+1]]`; the SWA prefix
        # (length `skip`) occupies the slice TAIL, written by the builder.
        # Matches the per-token prefix_swa_count vector we just computed on CPU.
        skip_csa_gpu = upload_numpy(prefix_swa_count_np, device)

        # ----- Publish on attn_metadata -----
        attn_metadata.kv_indices_extend = ext_indices[:ext_total]
        attn_metadata.kv_indptr_extend = ext_indptr
        attn_metadata.kv_indices_prefix_swa = swa_indices[:swa_total]
        attn_metadata.kv_indptr_prefix_swa = swa_indptr
        attn_metadata.kv_indices_prefix_csa = csa_indices[:csa_total]
        attn_metadata.kv_indptr_prefix_csa = csa_indptr
        attn_metadata.kv_indices_prefix_hca = hca_indices[:hca_total]
        attn_metadata.kv_indptr_prefix_hca = hca_indptr
        attn_metadata.skip_prefix_len_csa = skip_csa_gpu
        attn_metadata.envelope_rows = envelope_rows

    def _build_compress_plans(
        self,
        extend_lens_np,
        context_lens_np,
        *,
        running_bs: int | None = None,
        max_q_len: int | None = None,
        buf_prefix_ubatch: str = "",
    ):
        """Build per-ratio CompressPlan dict consumed by batched compressor.

        Reuse this from prepare_decode / prepare_prefill / prepare_capture —
        caller supplies extend_lens / context_lens (np int32). context_lens
        is the absolute per-seq length AFTER the new extend tokens (i.e.
        prefix + extend); `make_compress_plans` reads it as `context_lens_cpu`
        and reconstructs prefix internally.
        Plan tensors are written into the pre-allocated
        `v4_compress_plan_{ratio}` / `v4_write_plan_{ratio}` CpuGpuBuffers
        (fixed pointers for CUDAGraph capture); the kernels skip
        sentinel-marked tail rows.

        `running_bs` / `max_q_len`: set BOTH for decode runtime AND decode CG
        capture — the returned compress/write plan_gpu are sliced to fixed
        `running_bs * per_seq_bound` capacities (per ratio) so capture/replay
        shapes match, with `[bs, running_bs)` padding rows sentinel-filled.
        Leave both None for eager prefill — the plan_gpu are sliced to the
        actual `n_compress` / `n_write` (smallest grid, no padding).
        """
        from atom.model_ops.v4_kernels import make_compress_plans

        if not self._unique_compress_ratios_overlap:
            return {}
        # Inputs MUST be numpy int32 — torch tensors would force a D2H sync.
        # Callers are responsible for staging from forward_vars np mirrors.
        assert isinstance(extend_lens_np, np.ndarray), (
            f"extend_lens_np must be np.ndarray, got {type(extend_lens_np).__name__} "
            "— passing torch.Tensor here would trigger a hidden D2H sync"
        )
        assert isinstance(context_lens_np, np.ndarray), (
            f"context_lens_np must be np.ndarray, got {type(context_lens_np).__name__} "
            "— passing torch.Tensor here would trigger a hidden D2H sync"
        )
        var = self.model_runner.forward_vars
        plan_buffers = {
            ratio: {
                "compress": var[f"{buf_prefix_ubatch}v4_compress_plan_{ratio}"],
                "write": var[f"{buf_prefix_ubatch}v4_write_plan_{ratio}"],
            }
            for ratio, _ in self._unique_compress_ratios_overlap
        }
        return make_compress_plans(
            extend_lens_np,
            context_lens_np,
            self._unique_compress_ratios_overlap,
            plan_buffers=plan_buffers,
            running_bs=running_bs,
            max_q_len=max_q_len,
        )

    def _populate_block_tables(
        self, batch: ScheduledBatch, scheduled_bs: int
    ) -> torch.Tensor:
        """Populate `forward_vars["block_tables"]` from the batch and return
        the GPU view sliced to `scheduled_bs` rows.

        Defers the marshal to `CommonAttentionBuilder.prepare_block_tables`,
        but is invoked unconditionally (parent only calls it when has_cached).
        """
        self.prepare_block_tables(batch, limit=scheduled_bs)
        return self.model_runner.forward_vars["block_tables"].copy_to_gpu(scheduled_bs)

    def _populate_state_slot_mappings(
        self,
        batch: ScheduledBatch,
        scheduled_bs: int,
        running_bs: int,
        return_cpu: bool = False,
    ):
        """Build the `[running_bs]` int32 tensor of per-request state slots.

        The state class declares `entries_per_req=1`, so a seq holds exactly one
        pool slot and `state_slots_committed` is the whole story. This
        is what V4 forward uses to index `swa_kv` and `Compressor.kv_state`
        (the per-request state pool, distinct from the per-token paged-KV
        `slot_mapping`).

        When `return_cpu=True`, returns `(gpu_tensor, cpu_numpy)`; the CPU copy
        stays at `scheduled_bs`, being the real rows the forward path reads to
        avoid `.tolist()` syncs.

        Published at `running_bs`, the width a padded draft pass reads and the
        one `prepare_decode` uses -- the block slices this to that batch, and a
        Python slice past the end truncates rather than raising, so a narrower
        view would hand the kernel fewer slot rows than query rows. The
        fabricated tail is slot 0, the inert filler `prepare_decode` uses, and
        `mask_pad_tail` keeps those rows from scattering into it.

        The width moves together with `cu_seqlens_q`: `swa_write` takes
        `bs = state_slot_per_seq.shape[0]` and asserts
        `cu_seqlens_q.shape[0] >= bs + 1`, so `prepare_prefill` pads that tail
        to the same `running_bs`.
        """
        pool_np = np.asarray(batch.state_slots_committed[:scheduled_bs], dtype=np.int32)
        # Warmup / dummy_run batches don't allocate per_req_cache slots
        # (state_slots_committed is empty). Fall back to slot 0 for all seqs
        # so V4 forward can take the normal path uniformly — slot 0's state
        # cache is reset on the first real prefill (start_pos==0 path masks
        # state reads, fresh writes overwrite warmup pollution).
        if len(pool_np) < scheduled_bs:
            pool_np = np.zeros(scheduled_bs, dtype=np.int32)
        slots_np = self._physical_slots(pool_np)
        gpu = self._stage("v4_meta_state_slot_out", slots_np, pad_to=running_bs)
        if return_cpu:
            return gpu, slots_np
        return gpu

    def _state_slot_in_np(
        self, batch: ScheduledBatch, scheduled_bs: int, out_np: np.ndarray
    ) -> np.ndarray:
        """Per-seq read slot: the fork source where set, else the write slot.

        `out_np` is already in plane positions; the fork sources arrive as pool
        slots and are converted here, so one index space comes out.
        """
        srcs = getattr(batch, "state_fork_srcs", None)
        if not srcs or len(srcs) < scheduled_bs:
            return out_np
        src_np = np.asarray(srcs[:scheduled_bs], dtype=np.int32)
        return np.where(src_np >= 0, self._physical_slots(src_np), out_np)

    def _physical_slots(self, pool_slots: np.ndarray) -> np.ndarray:
        """Pool slots as the plane positions every kernel addresses by.

        The two run in opposite directions — see
        `UnifiedPoolGeometry.physical_slot` — and this is the only crossing.
        Everything downstream speaks positions: both planes' windows, the
        compressor state's strided view, and the DSpark draft's plane.
        """
        return np.int32(self.pool_geometry.slot_positions - 1) - pool_slots

    def _populate_state_slot_in(
        self,
        batch: ScheduledBatch,
        scheduled_bs: int,
        running_bs: int,
        out_np: np.ndarray,
    ) -> torch.Tensor:
        """Read-side slot per seq: the fork source where set, else the write slot.

        A fork means the seq reads the state slot it published (or resumed
        from) and writes a fresh one, for this forward only; `BlockManager`
        decides, the scheduler ships the pairing as `state_fork_srcs` and clears
        it after one batch.

        Always staged into its own buffer, even when nothing forked. The values
        are then identical to `state_slot_out`, but the buffer is not — decode
        replays a CUDAGraph that captured this pointer, so it has to be the same
        address on every step.
        """
        return self._stage(
            "v4_meta_state_slot_in",
            self._state_slot_in_np(batch, scheduled_bs, out_np),
            pad_to=running_bs,
        )

    def build_for_cudagraph_capture(
        self, bs: int, max_q_len: int | None = None, num_tokens_pad: int | None = None
    ) -> tuple[AttentionMetaData_DSV4, Context]:
        """Build attn_metadata for CUDAGraph capture using a synthetic decode batch.

        Synthesizes bs sequences each at start_pos=window_size (so SWA window
        is full + 1 CSA committed entry — exercises the production decode
        codepath: state-cache reads, sparse_attn gather, indexer fp8 logits).

        AF_PIECEWISE: if num_tokens_pad < bs*max_q_len, build a RAGGED batch of bs
        seqs summing to num_tokens_pad, so the graph's flat token dim == what the
        dense pieces write (zero-copy row counts match). max_seqlen_q stays max_q_len
        (bakes swa write_per_batch). Default (None / == rectangle) = uniform path.

        Per-fwd metadata is populated through the SAME helpers prepare_decode
        uses (`_attach_v4_indexer_meta`, `_attach_v4_per_fwd_meta`,
        `_build_compress_plans`), so all GPU views point to the pre-allocated
        buffers in `forward_vars`. Replay-time prepare_decode writes into the
        SAME buffers — captured graph reads stable addresses.

        NOTE on the state-write kernels (`update_compressor_states` /
        `swa_write`): both are now FIXED-grid + sentinel-masked, so they are
        CUDAGraph-capturable (level-3 default). `swa_write` launches
        grid=(bs, write_per_batch) with bs baked at capture and write_per_batch a
        `constexpr`; rows past each seq's actual token count sentinel-skip.
        `update_compressor_states` launches grid=(write_plan.shape[0],) — the
        decode-tight slice `running_bs * min(qlen, K_pool)` baked at capture, NOT
        the per-fwd num_write — and inactive rows carry `position=-1` and bail
        (see state_writes.py). So model.forward inside torch.cuda.graph does NOT
        hit a variable-grid launch here. (`fused_compress_attn` is likewise
        CG-safe: launches at the decode-tight compress slice `running_bs *
        ceil(qlen/ratio)` baked at capture and sentinel-skips inactive rows for
        both BF16 Main and FP8 Indexer paths.)
        """
        var = self.model_runner.forward_vars
        # Honor MTP at capture time: V4-Pro `mtp_k=1` → 2 tokens/req. The
        # outer `model_runner.capture_cudagraph` populates cu_seqlens_q with
        # the same layout, so capture and replay see identical shapes.
        # DSpark Phase 2 (graph multi-bucket): max_q_len is parametrized so the
        # capture loop can build one graph per query-length bucket
        # (decode_query_len in 1..mtp_k+1). Default = full mtp_k+1 (unchanged).
        if max_q_len is None:
            max_q_len = 1 + self.max_spec_steps
        rectangle_tokens = bs * max_q_len
        # Per-seq query lengths of the synthetic batch. One uniform rectangle by
        # default; AF_PIECEWISE instead asks for a ragged batch summing to
        # `num_tokens_pad`, so the graph's flat row count equals what the dense
        # pieces write -- a zero-copy input is read at exactly the captured
        # length, so a padded tail would be rows nobody wrote that step.
        if num_tokens_pad is None or int(num_tokens_pad) >= rectangle_tokens:
            total_tokens = rectangle_tokens
            extend_lens_np = np.full(bs, max_q_len, dtype=np.int32)
        else:
            total_tokens = int(num_tokens_pad)
            assert total_tokens >= bs, (
                f"ragged capture needs num_tokens_pad >= bs so every seq gets at "
                f"least one token; got bs={bs} num_tokens_pad={total_tokens}"
            )
            # As even as possible, remainder on the head: every length is then in
            # [1, max_q_len].
            extend_lens_np = np.full(bs, total_tokens // bs, dtype=np.int32)
            extend_lens_np[: total_tokens % bs] += 1

        # Synthetic state: each seq has already produced `start_pos` tokens, and
        # this fwd is its own len_i decode/draft steps from there. start_pos > 0
        # hits is_pure_decode, exercising Phase B/C/E paths during capture.
        start_pos = self.window_size
        cu_seqlens_q_np = np.zeros(bs + 1, dtype=np.int32)
        np.cumsum(extend_lens_np, out=cu_seqlens_q_np[1:])
        # A token sits at start_pos + its offset within its OWN seq, i.e. flat
        # index minus where that seq starts. Uniform lengths reduce that to
        # flat_idx % max_q_len, the classic rectangle.
        batch_id_per_token = np.repeat(np.arange(bs, dtype=np.int32), extend_lens_np)
        flat_idx = np.arange(total_tokens, dtype=np.int64)
        seq_start = cu_seqlens_q_np[batch_id_per_token].astype(np.int64)
        positions_np = (flat_idx - seq_start) + start_pos
        context_lens_np = (start_pos + extend_lens_np).astype(np.int32)
        # Slot mapping: pool groups [0..bs-1] crossed to the plane positions
        # every kernel addresses by, the same crossing `prepare_decode` makes.
        # Raw group ids here are not a harmless placeholder: capture runs a
        # real eager warmup forward first, so group `g` would resolve to plane
        # rows inside the compressed-block region and the warmup would write
        # compressor state and window KV over live blocks.
        state_slot_np = self._physical_slots(np.arange(bs, dtype=np.int32))
        # Block tables: block 0 for every seq (placeholder; capture warmup
        # fills it via real reads but the data is throwaway).
        block_tables_np = np.zeros(
            (bs, var["block_tables"].np.shape[1]), dtype=np.int32
        )

        # Stage CPU mirrors → forward_vars + capture-time GPU views.
        var["positions"].np[:total_tokens] = positions_np
        positions = var["positions"].copy_to_gpu(total_tokens)
        var["cu_seqlens_q"].np[: bs + 1] = cu_seqlens_q_np
        cu_seqlens_q_gpu = var["cu_seqlens_q"].copy_to_gpu(bs + 1)
        var["context_lens"].np[:bs] = context_lens_np
        context_lens_gpu = var["context_lens"].copy_to_gpu(bs)
        var["block_tables"].np[:bs] = block_tables_np
        block_tables_gpu = var["block_tables"].copy_to_gpu(bs)
        state_slot_gpu = self._stage("v4_meta_state_slot_out", state_slot_np)
        # Read side captured from its own persistent buffer: replay-time
        # prepare_decode refills it, so a fork can change the values without
        # invalidating the graph.
        state_slot_in_gpu = self._stage("v4_meta_state_slot_in", state_slot_np)

        # Synthetic decode batch: start_pos > 0 and per-seq len_i tokens (ragged
        # under zero-copy-q, else uniform max_q_len), so is_pure_decode is True
        # by construction (capture replays the decode codepath).
        attn_metadata = AttentionMetaData_DSV4(
            cu_seqlens_q=cu_seqlens_q_gpu,
            cu_seqlens_k=None,
            max_seqlen_q=max_q_len,
            max_seqlen_k=int(context_lens_np.max()) if bs else 1,
            min_seqlen_q=0,
            dropout_p=0.0,
            has_cached=False,
            total_kv=int(context_lens_np.sum()),
            num_cached_tokens=None,
            block_tables=block_tables_gpu,
            context_lens=context_lens_gpu,
            state=AttnState.DECODE,
        )
        attn_metadata.state_slot_out = state_slot_gpu
        attn_metadata.state_slot_in = state_slot_in_gpu
        attn_metadata.state_slot_out_cpu = state_slot_np

        # DSpark TRUE-FLAT graph: capture must take the same ragged indexer branch
        # and rect shape [bs, full_q] as replay, else the graph mismatches. The
        # synthetic batch's per-seq lengths (`extend_lens_np`, ragged under
        # zero-copy-q else uniform max_q_len) drive the indexer ragged topk.
        drafter = getattr(self.model_runner, "drafter", None)
        if (
            self.model_runner.config.dspark.ragged
            and drafter is not None
            and drafter.uses_confidence_schedule
        ):
            full_q_real = drafter.mtp_k + 1
            # `max_q_len` is the bucket loop's parameter and these lengths are
            # derived from it, so nothing here ties them to the drafter's
            # `full_q` -- the same check `prepare_decode` makes on the batch it
            # was handed.
            require_step_within_full_q(
                int(extend_lens_np.max()) if extend_lens_np.size else 0,
                full_q_real,
                "a captured DSpark graph",
            )
            pad_to = self._dspark_ragged_lens_pad_to(bs)
            if pad_to is not None:
                # AF + DP: the core captured here bakes this tensor's address, so
                # capture has to write the SAME buffer replay does — otherwise
                # every replay reads this capture's throwaway allocation.
                attn_metadata.dspark_ragged_lens_gpu = self._stage_dspark_ragged_lens(
                    extend_lens_np, pad_to
                )
            else:
                # Nothing captures the core here and this is a capture-time
                # allocation, so a pageable H2D costs nothing: main's
                # `torch.full((bs,), max_q_len)`, widened to the ragged lengths
                # (identical when the synthetic batch is the uniform rectangle).
                attn_metadata.dspark_ragged_lens_gpu = torch.as_tensor(
                    extend_lens_np.astype(np.int32, copy=False),
                    device=positions.device,
                )
            attn_metadata.dspark_full_q = int(full_q_real)

        # Build compress_plans + per-fwd meta + indexer meta via the same
        # helpers used at runtime — guarantees addresses match. `extend_lens_np`
        # is the synthetic batch's per-seq lengths (computed above).
        attn_metadata.compress_plans = self._build_compress_plans(
            extend_lens_np,
            context_lens_np,
            running_bs=bs,
            max_q_len=max_q_len,
        )
        # Capture: running_bs == scheduled_bs == bs (synthetic batch is full).
        # Must run BEFORE `_attach_v4_indexer_meta` so the indexer-side meta
        # builder can reuse the shared per-fwd GPU tensors.
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            extend_lens_np,  # synthetic per-seq lengths (ragged or uniform max_q_len)
            attn_metadata.state_slot_out_cpu,
            bs,
            total_tokens,
            running_bs=bs,
            max_q_len=max_q_len,
        )
        self._attach_v4_indexer_meta(
            attn_metadata,
            bs,
            total_tokens,
            positions_gpu=positions,
        )

        if self.model_runner.config.enable_tbo_decode and bs > 2:
            self._prepare_ubatch_decode(
                scheduled_bs=bs,
                running_bs=bs,
                max_seqlen_q=max_q_len,
                context_lens_np=context_lens_np,
                state_slot_np=state_slot_np,
                # Capture has no forks, so the read side is the write side.
                state_slot_in_np=state_slot_np,
                positions_np=positions_np.astype(np.int32),
            )

        context = Context(
            positions=positions,
            is_prefill=False,
            scheduled_bs=bs,
            running_bs=bs,
            # `total_tokens` is this capture's row count already -- the ragged
            # path's flat bucket, not bs*max_q_len.
            # A capture runs a full synthetic batch: nothing is padded.
            scheduled_tokens=total_tokens,
            running_tokens=total_tokens,
        )
        return attn_metadata, context

    # ------------------------------------------------------------------ #
    # Helpers.                                                           #
    # ------------------------------------------------------------------ #

    def _dspark_ragged_lens_pad_to(self, bs: int) -> int | None:
        """Row count the dspark ragged verify-lengths must be padded to, or None
        to keep the real per-seq count.

        AF + DP replays a captured attention core that baked BOTH this tensor's
        address and its row count, so every step has to present `bs` rows with
        the tail zeroed or the core reads whatever the shorter copy left behind.
        Nothing else captures the core, and padding costs acceptance there, so
        every other configuration keeps the real count.
        """
        cfg = self.model_runner.config
        mode = getattr(cfg.compilation_config, "cudagraph_mode", None)
        if mode is None or not mode.is_attn_ffn_piecewise():
            return None
        if getattr(cfg.parallel_config, "data_parallel_size", 1) <= 1:
            return None
        return int(bs)

    def _stage_dspark_ragged_lens(self, extend_lens_np, pad_to: int | None):
        """Pinned async H2D of the per-seq verify lengths, returning the GPU view.

        Pinned staging, never `torch.as_tensor(np, device=cuda)`: that is a
        pageable H2D and synchronizes, which was the ragged decode bubble #1861
        removed. `forward_vars["ragged_extend"]` is one buffer at one address for
        both capture and replay, which is also what the AF captured core needs --
        so the fixed address costs nothing extra and only `pad_to` varies.
        """
        buf = self.model_runner.forward_vars["ragged_extend"]
        n = int(extend_lens_np.size)
        buf.np[:n] = extend_lens_np
        if pad_to is not None and pad_to > n:
            buf.np[n:pad_to] = 0
            n = int(pad_to)
        return buf.copy_to_gpu(n)

    def _alloc_v4_metadata_buffers(self) -> None:
        """Pre-allocate every buffer the V4 metadata builder writes into.

        A `CpuGpuBuffer` where the host writes and the device reads, a bare
        device tensor where both ends are kernels — the pinned host half is
        the expensive part and only the staged buffers earn it.

        Bounds:
          - per-seq:        max_bs
          - per-token:      max_num_batched_tokens
          - csa compress:   max_num_batched_tokens * index_topk
          - hca compress:   max_num_batched_tokens * max_num_blocks_per_seq
          - csa gather:     max_bs * max_num_blocks_per_seq * (block_size // 4)
          - decode swa dst: max_bs * window_size

        Memory footprint at typical config (max_bs=16, mnbt=8192, win=128,
        index_topk=1024, max_num_blocks_per_seq=64): ~80 MB total. Allocated
        once at builder init; pointers stay fixed for CUDAGraph capture.
        """
        i32 = {"dtype": torch.int32, "device": self.device}
        i64 = {"dtype": torch.int64, "device": self.device}
        mnbt = self.max_num_batched_tokens
        bs = self.max_bs
        win = self.window_size

        bufs: dict = {}

        # `kv_indptr` is touched unconditionally by the global capture loop
        # (model_runner.capture_cudagraph: `forward_vars["kv_indptr"].zero_()`).
        # MLA backends own this buffer; V4 doesn't use it for its own kernels
        # but allocates a min-size stub so the capture loop runs. Sized for
        # potential future reuse if a V4-side MLA kernel needs paged KV indices.
        bufs["kv_indptr"] = CpuGpuBuffer(bs + 1, **i32)

        # _attach_v4_per_fwd_meta + _populate_state_slot_mappings.
        # state_slot is staged ONCE into v4_meta_state_slot_out (set by
        # `_populate_state_slot_mappings`); attn_metadata.state_slot_out
        # exposes that GPU view to all downstream consumers (no second
        # H2D-staged copy).
        bufs["v4_meta_state_slot_out"] = CpuGpuBuffer(bs, **i32)
        # Read side of the compressor ring (`_populate_state_slot_in`). Its own
        # buffer on every path, forked or not, so the captured decode graph sees
        # a stable address.
        bufs["v4_meta_state_slot_in"] = CpuGpuBuffer(bs, **i32)

        # Phase B: paged-decode index buffers (consumed by Phase C/E).
        # Sized to worst-case decode shape `T = max_bs * (1 + max_spec_steps)`
        # — these buffers are decode-only; prefill goes through
        # `_build_paged_prefill_meta` (per-fwd alloc) and never touches them.
        # Per-buffer footprint at V4-Pro (T=32, win=128, index_topk=1024,
        # max_committed_hca=8192): swa 16KB / csa 144KB / hca 1.04MB; the rest
        # negligible.
        # Per-seq state (valid_count_csa) + the single per-token batch_id
        # mapping (`v4_batch_id_per_token`) replace per-token aliases of seq-
        # level data — downstream kernels do
        # `data[batch_id_per_token[t]]` instead of carrying a [T]-sized copy.
        T_dec = self.max_decode_tokens
        # Device-only: written and read entirely by kernels, so a host mirror
        # would be tens of MB of pinned memory nothing writes.
        bufs["v4_kv_indices_swa"] = torch.zeros(T_dec * win, **i32)
        bufs["v4_kv_indices_csa"] = torch.zeros(T_dec * (win + self.index_topk), **i32)
        bufs["v4_kv_indices_hca"] = torch.zeros(
            T_dec * (win + self.max_committed_hca), **i32
        )
        # Device-only for the same reason: `build_v4_paged_decode_indptr` writes
        # all four from tensors already resident, so a host mirror would only
        # ever hold whatever the last host-built forward left in it.
        bufs["v4_kv_indptr_swa"] = torch.zeros(T_dec + 1, **i32)
        bufs["v4_kv_indptr_csa"] = torch.zeros(T_dec + 1, **i32)
        bufs["v4_kv_indptr_hca"] = torch.zeros(T_dec + 1, **i32)
        bufs["v4_csa_n_committed_per_token"] = torch.zeros(T_dec, **i32)
        # Device-only, and as wide as the `block_tables` it gathers from: a host
        # mirror would be `T_dec * cols * 4` of pinned memory nothing writes.
        bufs["v4_block_tables_per_token"] = torch.zeros(
            T_dec, self.block_table_cols, **i32
        )
        # Where each decode token's own KV row goes, one buffer per compress
        # class. The fused SWA write reads these rather than deriving the row,
        # so the window layout stays inside this repo (see
        # `write_v4_paged_decode_indices`).
        for name in _DEST_ROW_BUFFERS.values():
            bufs[name] = CpuGpuBuffer(T_dec, **i32)

        # Per-token paged-decode index tensors for the fp8 asm decode kernel
        # (`mla_decode_fwd_v4_nm`, page_size=1). Values are CONSTANT — they
        # depend only on the (padded) decode token count N, not the batch:
        #   qo_indptr        = arange(N+1)   (per-token q indptr, max_seqlen_q=1)
        # A CpuGpuBuffer re-staged via `self._stage(...)` EVERY fwd, which is
        # what makes it CUDAGraph-safe (re-copied into the captured buffer
        # before graph.replay). The constant numpy source is precomputed once so
        # the per-fwd cost is a slice + H2D.
        bufs["v4_qo_indptr"] = CpuGpuBuffer(T_dec + 1, **i32)
        self._v4_qo_indptr_np = np.arange(T_dec + 1, dtype=np.int32)
        # Per-seq `ctx_len // 4` (raw, no clamp). Consumed by the indexer's
        # `cu_committed` cumsum and the FP4 ragged windows — both per-SEQUENCE.
        # Built unconditionally in `_attach_v4_per_fwd_meta`.
        bufs["v4_n_committed_csa_per_seq"] = CpuGpuBuffer(bs, **i32)
        # Single per-token mapping shared across ALL V4 consumers:
        #   - swa_write / csa_translate_pack (triton kernels)
        #   - _build_v4_indexer_meta (PyTorch fancy index — int32 indices are
        #     accepted by torch advanced-indexing)
        #   - the fused SWA scatter in qk_norm_rope_maybe_quant (flydsl kernel
        #     loads it as int32; the MTP-draft path also supplies int32 via the
        #     cu_seqlens_q slice, so int32 keeps both decode paths uniform).
        # Sized to `mnbt` (worst-case prefill total tokens) since swa_write
        # fires on prefill paths too. Phase B decode only uses [:T_dec].
        bufs["v4_batch_id_per_token"] = CpuGpuBuffer(mnbt, **i32)

        # _build_v4_indexer_meta (CSA only — but allocate unconditionally;
        # never accessed when CSA layers are absent).
        # int32 — `cp_gather_indexer_k_quant_cache` kernel signature is `int32_t*`
        # for cu_seq_lens. Also reused as cu_starts/cu_ends for fp8_mqa_logits
        # (which accepts both int32 and int64).
        bufs["v4_indexer_cu_committed"] = CpuGpuBuffer(bs + 1, **i32)
        # FP4 indexer decode: fixed-address [P, 4] cta_info schedule buffer for
        # the `pa_mqa_logits_fp4` kernel. The schedule (compute_varctx_schedule)
        # is pure on-device torch (no host sync) and emits a CONSTANT-shape
        # [P, 4] tensor with total_ctas == P fixed — so building it eagerly in
        # `_build_v4_indexer_meta` (pre-replay) into this fixed address makes
        # the captured kernel CUDAGraph-safe (grid = P is baked; only the buffer
        # CONTENTS change per fwd, refreshed before each replay). Plain GPU
        # tensor (not CpuGpuBuffer): no CPU mirror, written by a device kernel.
        # P / block_k MUST match the values passed to flydsl_pa_mqa_logits_fp4.
        if self._indexer_fp4:
            # P = persistent-grid CTA-count CAP, bounding two axes (see the
            # prefill build for the full note):
            #   - rows: a decode fwd has one row per decode token (= bs*next_n),
            #     so P must be >= max_decode_tokens (T_dec) or surplus rows are
            #     silently dropped (logits stay at the -inf pre-fill -> wrong
            #     top-k).
            #   - chunks: the 512 floor keeps enough CTAs to split a long
            #     context across the GPU when the batch is small (e.g. bs=8,
            #     ctx=128k -> only 8 rows; without the floor the schedule would
            #     fold the whole context onto 8 serial CTAs and starve the GPU).
            # The fixed CG cta_info buffer below is sized to this same P.
            self._fp4_parallel_unit_num = max(FP4_MQA_PARALLEL_UNIT_NUM, T_dec)
            self._fp4_block_k = FP4_MQA_BLOCK_K
            self._v4_fp4_cta_info = torch.zeros((self._fp4_parallel_unit_num, 4), **i32)
            # DSpark RAGGED decode CG buffers (fixed addresses). Windows are read
            # by top_k during the captured scorer; cta_info by the mqa kernel. The
            # ragged (prefill) schedule uses CTA_INFO_WIDTH=6 (vs 4 for the
            # rectangular varctx schedule). Windows sized to T_dec (max decode
            # tokens = max padded row count); refreshed each fwd in build().
            self._v4_fp4_ragged_row_to_batch = torch.zeros(T_dec, **i32)
            self._v4_fp4_ragged_local_starts = torch.zeros(T_dec, **i32)
            self._v4_fp4_ragged_local_ends = torch.zeros(T_dec, **i32)
            self._v4_fp4_ragged_cta_info = torch.zeros(
                (self._fp4_parallel_unit_num, 6), **i32
            )
        # NOTE: decode-path `logits` ([T, max_model_len_idx] fp32) and
        # `topk_indices` ([T, index_topk] int32) are NOT pre-allocated —
        # they are write-once GPU scratch with no CPU mirror, allocated
        # per-fwd inside `Indexer._score_topk_decode` via `torch.empty`.
        # Under CUDAGraph capture they land in the graph's private pool
        # and replay reuses the same address; eager keeps the standard
        # caching-allocator fast path.

        # Compress plan buffers (per-ratio) — pre-allocated for CUDAGraph
        # plan-tensor address stability. `make_compress_plans(..., plan_buffers=)`
        # writes into these and sentinel-fills the trailing rows. Worst-case
        # sizes: num_compress ≤ ⌈mnbt/ratio⌉ + bs (one boundary per seq plus
        # alignment slack); num_write ≤ bs * STATE_SIZE (per-seq ring window
        # carries STATE_SIZE rows per fwd at most).
        #
        # The decode CG path uses a much tighter capacity than the prefill
        # worst case — the kernel grid is dictated by the slice of this
        # buffer that we hand to the kernel, and decode only ever needs
        # `running_bs * ceil((1 + max_spec_steps) / ratio)` compress rows (vs
        # `mnbt // ratio + bs` for prefill, ~13× larger at typical config). We
        # still allocate the full prefill capacity (eager prefill needs it),
        # but decode capture/replay slice down via `make_compress_plans(
        # running_bs=, max_q_len=)`, which computes the per-graph-tight caps
        # `running_bs * per_seq_bound` internally (see compress_plan.py).
        for ratio, is_overlap in self._unique_compress_ratios_overlap:
            # NOTE: K_pool is the pool-window size (algorithm constant), NOT the
            # state ring buffer size. The ring buffer is K_pool + max_spec_steps
            # (see csa_main_state_shape comment for the slot-aliasing argument),
            # but write_plan still emits ≤ K_pool rows per seq per fwd because
            # `write_starts = max(0, context_lens - K_pool)` in make_compress_plans.
            K_pool = (2 if is_overlap else 1) * ratio
            max_compress = mnbt // ratio + bs
            max_write = min(mnbt, bs * K_pool)
            bufs[f"v4_compress_plan_{ratio}"] = CpuGpuBuffer(max_compress, 4, **i32)
            bufs[f"v4_write_plan_{ratio}"] = CpuGpuBuffer(max_write, 4, **i32)
            # Pre-fill with sentinel so capture-time buffer state is valid
            # even before the first non-empty fwd.
            bufs[f"v4_compress_plan_{ratio}"].cpu.fill_(-1)
            bufs[f"v4_compress_plan_{ratio}"].copy_to_gpu()
            bufs[f"v4_write_plan_{ratio}"].cpu.fill_(-1)
            bufs[f"v4_write_plan_{ratio}"].copy_to_gpu()

        # ub{0,1}_ prefixed buffer sets are used by BOTH TBO decode and TBO
        # prefill ubatch metadata builds (each ubatch reads/writes its own set
        # instead of racing on the shared global forward_vars buffers). Allocate
        # whenever TBO is on, not just for decode.
        if getattr(self.model_runner.config, "enable_tbo", False) or getattr(
            self.model_runner.config, "enable_tbo_decode", False
        ):
            self._alloc_v4_ubatch_decode_buffers(bufs, i32, i64)

        self.model_runner.forward_vars.update(bufs)

    def _alloc_v4_ubatch_decode_buffers(self, bufs: dict, i32: dict, i64: dict) -> None:
        """Clone decode-path metadata buffers into ``ub{0,1}_`` prefixed sets.

        Mirrors the sizes chosen in :meth:`_alloc_v4_metadata_buffers` for the
        decode-relevant buffers plus the global per-fwd inputs the decode
        helpers read (``positions`` / ``context_lens`` / ``block_tables`` /
        ``cu_seqlens_q``). Only invoked when ``enable_tbo_decode`` is set.
        """
        mnbt = self.max_num_batched_tokens
        bs = self.max_bs
        win = self.window_size
        T_dec = self.max_decode_tokens

        for ub_idx in range(self._NUM_TBO_UBATCHES):
            p = f"ub{ub_idx}_"
            # Global per-fwd decode inputs (live in model_runner.forward_vars
            # for the non-TBO path; cloned here so each ubatch slices its own).
            bufs[f"{p}positions"] = CpuGpuBuffer(T_dec, **i64)
            bufs[f"{p}context_lens"] = CpuGpuBuffer(bs, **i32)
            bufs[f"{p}block_tables"] = CpuGpuBuffer(bs, self.block_table_cols, **i32)
            bufs[f"{p}cu_seqlens_q"] = CpuGpuBuffer(bs + 1, **i32)

            # V4 decode metadata buffers.
            bufs[f"{p}v4_meta_state_slot_out"] = CpuGpuBuffer(bs, **i32)
            bufs[f"{p}v4_meta_state_slot_in"] = CpuGpuBuffer(bs, **i32)
            bufs[f"{p}v4_kv_indices_swa"] = torch.zeros(T_dec * win, **i32)
            bufs[f"{p}v4_kv_indices_csa"] = torch.zeros(
                T_dec * (win + self.index_topk), **i32
            )
            bufs[f"{p}v4_kv_indices_hca"] = torch.zeros(
                T_dec * (win + self.max_committed_hca), **i32
            )
            bufs[f"{p}v4_kv_indptr_swa"] = torch.zeros(T_dec + 1, **i32)
            bufs[f"{p}v4_kv_indptr_csa"] = torch.zeros(T_dec + 1, **i32)
            bufs[f"{p}v4_kv_indptr_hca"] = torch.zeros(T_dec + 1, **i32)
            bufs[f"{p}v4_csa_n_committed_per_token"] = torch.zeros(T_dec, **i32)
            bufs[f"{p}v4_block_tables_per_token"] = torch.zeros(
                T_dec, self.block_table_cols, **i32
            )
            for name in _DEST_ROW_BUFFERS.values():
                bufs[f"{p}{name}"] = CpuGpuBuffer(T_dec, **i32)
            bufs[f"{p}v4_n_committed_csa_per_seq"] = CpuGpuBuffer(bs, **i32)
            bufs[f"{p}v4_batch_id_per_token"] = CpuGpuBuffer(mnbt, **i32)
            bufs[f"{p}v4_indexer_cu_committed"] = CpuGpuBuffer(bs + 1, **i32)

            for ratio, is_overlap in self._unique_compress_ratios_overlap:
                K_pool = (2 if is_overlap else 1) * ratio
                max_compress = mnbt // ratio + bs
                max_write = min(mnbt, bs * K_pool)
                cbuf = CpuGpuBuffer(max_compress, 4, **i32)
                wbuf = CpuGpuBuffer(max_write, 4, **i32)
                cbuf.cpu.fill_(-1)
                cbuf.copy_to_gpu()
                wbuf.cpu.fill_(-1)
                wbuf.copy_to_gpu()
                bufs[f"{p}v4_compress_plan_{ratio}"] = cbuf
                bufs[f"{p}v4_write_plan_{ratio}"] = wbuf

    def _dest_row_buffers(self, buf_prefix_ubatch: str = "") -> dict[int, torch.Tensor]:
        """The per-class destination-row buffers for this (ubatch's) forward."""
        var = self.model_runner.forward_vars
        return {
            ratio: var[f"{buf_prefix_ubatch}{name}"].gpu
            for ratio, name in _DEST_ROW_BUFFERS.items()
        }

    def _stage(self, name: str, arr, pad_to: int | None = None) -> torch.Tensor:
        """Write numpy `arr` into `forward_vars[name]` (CpuGpuBuffer) and
        return its GPU view sliced to len(arr). Asserts the buffer is large
        enough and that `arr.dtype` matches the buffer dtype (callers must
        cast to the buffer dtype before staging).

        `pad_to` zero-fills the tail out to a wider view -- the padded batch a
        drafter runs. Zero is a real slot, so the caller owes those rows a
        reason they are never read.
        """
        buf = self.model_runner.forward_vars[name]
        n = arr.shape[0] if arr.ndim > 0 else 1
        assert (
            n > 0
        ), f"Cannot stage empty array for {name!r} — ensure the input array has at least one element."
        cap = buf.np.shape[0]
        assert n <= cap, (
            f"V4 buffer {name!r} too small: need {n}, have {cap}. "
            f"Increase the corresponding bound in _alloc_v4_metadata_buffers."
        )
        assert arr.dtype == buf.np.dtype, (
            f"V4 buffer {name!r} dtype mismatch: buffer is {buf.np.dtype}, "
            f"but got arr with dtype {arr.dtype}. Cast arr to the correct "
            f"dtype before calling _stage."
        )
        width = n if pad_to is None else pad_to
        assert width <= cap, (
            f"V4 buffer {name!r} too small: need {width} padded, have {cap}. "
            f"Increase the corresponding bound in _alloc_v4_metadata_buffers."
        )
        buf.np[n:width] = 0
        buf.np[:n] = arr
        return buf.copy_to_gpu(width)

    @staticmethod
    def _make_gather_slot(
        buf: torch.Tensor,
        stride: int,
        arena: SplitStateArena,
        geo: UnifiedPoolGeometry,
    ):
        """Callable copying one request's state → the staging buffer.

        One copy per plane, concatenated in plane order, which is the order the
        staging slot is defined to hold them in — `_make_scatter_slot` is the
        only reader and undoes exactly this. The staging hop itself only
        survives because the connector registers `buf` rather than the planes;
        registering them directly retires both this and the pool.
        """
        assert stride == arena.entry_bytes // buf.dtype.itemsize

        def gather_slot(compute_slot: int, pool_idx: int) -> None:
            slot = geo.physical_slot(compute_slot)
            dst = pool_idx * stride
            for part in arena.arenas:
                width = part.entry_bytes // buf.dtype.itemsize
                buf[dst : dst + width] = part.entry(slot).view(buf.dtype)
                dst += width

        return gather_slot

    @staticmethod
    def _make_scatter_slot(
        buf: torch.Tensor,
        stride: int,
        arena: SplitStateArena,
        geo: UnifiedPoolGeometry,
    ):
        """Callable copying the staging buffer → one request's state."""

        def scatter_slot(compute_slot: int, pool_idx: int) -> None:
            src = pool_idx * stride
            slot = geo.physical_slot(compute_slot)
            for part in arena.arenas:
                width = part.entry_bytes // buf.dtype.itemsize
                part.entry(slot).view(buf.dtype).copy_(buf[src : src + width])
                src += width

        return scatter_slot
