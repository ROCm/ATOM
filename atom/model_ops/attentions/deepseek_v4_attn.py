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
                   lcm(m, m') = 128 original tokens. Compressor + Indexer
                   .kv_cache attributes bound to per-layer pool slices.
PR3-main:   multi-sequence dispatch (slot=0 -> per-seq slot).

Per-slot cost (V4-Pro, BF16 SWA + fp32 tail buffers, 30 CSA + 31 HCA + 1 dense):
  SWA:         62 layers * 128 * 512 * 2B  =  8.0 MB
  CSA Main:    30 * 2 * (8 * 1024)  * 4B   =  1.875 MB
  CSA Indexer: 30 * 2 * (8 * 256)   * 4B   =  0.469 MB
  HCA Main:    31 * 2 * (128 * 512) * 4B   = 16.0 MB
  Total                                      = ~26.5 MB / slot
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, cast

import numpy as np
import torch
from aiter import dtypes
from atom.distributed.pcp_utils import (
    get_pcp_world_size,
    pcp_is_enabled,
    pcp_pad_dense,
    pcp_pad_indptr,
    pcp_pad_len,
    pcp_reindex_ragged,
    pcp_round_robin_query_indices,
)
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attentions.backends import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionBuilder,
)
from atom.model_ops.v4_kernels import (
    write_v4_paged_decode_indices,
    write_v4_paged_prefill_indices,
)
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import (
    AttentionMetaData,
    AttnState,
    Context,
    get_forward_context,
)

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
      bs         = scheduled_bs            actual decode/prefill seqs
      padded_bs  = capture-time graph_bs   (= bs in eager / prefill)
      T          = total_tokens this fwd   (= sum of token_num_per_seq)
      padded_T   = padded_bs * max_q_len   (>= T; captured kernels iterate this)
      win        = self.window_size        (128 for V4-Pro)
      index_topk = self.index_topk         (1024 for V4-Pro)
    """

    # ----- CPU mirrors (avoid GPU→CPU `.item()` / `.tolist()` syncs) -----
    state_slot_mapping_cpu: Optional[Any] = None
    """[bs] np.int32 — per-seq state cache slot id (host copy)."""
    n_committed_csa_per_seq_cpu: Optional[Any] = None
    """[bs] np.int32 — `ctx_len // 4` (CSA committed K per seq). Built once
    in `_attach_v4_per_fwd_meta` from `var["context_lens"].np`; consumed by
    `_attach_v4_paged_decode_meta`, `_build_paged_prefill_meta`, and
    `_build_v4_indexer_meta` (indptr cumsums). Single source of truth so
    those callers don't each re-read context_lens + divide."""
    n_committed_hca_per_seq_cpu: Optional[Any] = None
    """[bs] np.int32 — `ctx_len // 128` (HCA committed compress entries per
    seq). Same lifecycle as `n_committed_csa_per_seq_cpu`."""

    # ----- Per-seq GPU scalars (single-source-of-truth, shared by kernels) -----
    state_slot_mapping: Optional[torch.Tensor] = None
    """[bs] int32 GPU — per-seq state cache slot. Shared by swa_write +
    Compressor + paged-decode kernels (looked up via batch_id_per_token)."""
    n_committed_csa_per_seq: Optional[torch.Tensor] = None
    """[bs] int32 GPU — RAW `ctx_len // 4` per-seq committed count. Consumed
    by the indexer (cast to long inline) AND by csa_translate_pack
    (kernel derives per-token valid_k inline from this + positions +
    index_topk; no separate per-token tensor needed)."""

    # ----- Per-fwd hoisted (built in `_attach_v4_per_fwd_meta`) -----
    batch_id_per_token: Optional[torch.Tensor] = None
    """[padded_T] int32 GPU — the SINGLE per-token mapping
    (token_idx → seq_idx). int32 indices are accepted by PyTorch
    advanced-indexing (used in the indexer); triton kernels (swa_write,
    csa_translate_pack) and the fused flydsl SWA scatter read int32. Padded
    tail [T:padded_T] = -1 sentinel; consumer kernels skip on `bid < 0`. All
    other per-token quantities resolved as `per_seq_data[batch_id_per_token[t]]`
    — no [T] aliases of seq data."""
    batch_id_per_token_cpu: Optional[Any] = None
    """[T] int32 — CPU mirror of the unpadded batch_id slice. Built once in
    `_attach_v4_per_fwd_meta` (host-side `np.repeat`); reused by
    `_attach_v4_paged_decode_meta` for indptr fancy-index math. Avoids a
    duplicate `np.repeat` per fwd. None for prefill paths that don't go
    through paged_decode_meta (it's only consumed there)."""
    compress_plans: Optional[Dict[int, Any]] = None
    """dict[ratio:int -> CompressPlan] — packed plan tensors per
    compress_ratio (4=CSA, 128=HCA)."""

    # ----- Phase B paged-decode metadata (set when state is DECODE) -----
    # `state` lives on the base AttentionMetaData; every V4 `prepare_*` path
    # overrides it. Below buffers are populated only when state is DECODE
    # (built by `_attach_v4_paged_decode_meta`).
    kv_indices_swa: Optional[torch.Tensor] = None
    """[swa_indptr[T]] int32 GPU — ragged-packed paged offsets into `unified_kv`
    for the SWA path (per-token length `min(positions[t]+1, win)`)."""
    kv_indices_csa: Optional[torch.Tensor] = None
    """[csa_indptr[T]] int32 GPU — packed paged offsets for CSA layers
    (CSA topk compress at slice head + SWA window prefix at tail; topk section
    filled per-layer by csa_translate_pack)."""
    kv_indices_hca: Optional[torch.Tensor] = None
    """[hca_indptr[T]] int32 GPU — packed paged offsets for HCA layers
    (HCA compress at slice head + SWA window prefix at tail; layer-invariant)."""
    kv_indptr_swa: Optional[torch.Tensor] = None
    """[padded_T+1] int32 GPU — ragged cumsum of per-token SWA length
    `min(positions[t]+1, win)`. Padded tail repeats last value → kv_len=0
    sentinel for CG-padded slots."""
    kv_indptr_csa: Optional[torch.Tensor] = None
    """[padded_T+1] int32 GPU — packed cumsum of per-token CSA kv_len
    (= `min(positions[t]+1, win) + min(n_committed_csa[bid], index_topk)`).
    Padded tail = last value."""
    kv_indptr_hca: Optional[torch.Tensor] = None
    """[padded_T+1] int32 GPU — packed cumsum of per-token HCA kv_len
    (= `min(positions[t]+1, win) + n_committed_hca[bid]`). Padded tail = last value."""
    swa_pages: int = 0
    """Boundary in `unified_kv`: index < swa_pages → SWA region; index >=
    swa_pages → compress region. Equal to
    `max_per_req_cache_slots * win_with_spec` (per-slot SWA region holds
    `win + mtp_k` ring entries; reduces to `win` when MTP is off)."""

    # ----- Indexer / sparse-layout side metadata -----
    indexer_meta: Optional[Dict[str, Any]] = None
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
    skip_prefix_len_csa: Optional[torch.Tensor] = None
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
    kv_indices_prefix_swa: Optional[torch.Tensor] = None
    """[sum(prefix_swa_count)] int32 GPU — flat paged offsets into
    `unified_kv` for Dense (ratio==0) layers' prefix region (SWA history
    only)."""
    kv_indptr_prefix_swa: Optional[torch.Tensor] = None
    """[total_tokens + 1] int32 GPU — packed cumsum of `prefix_swa_count`."""
    kv_indices_prefix_csa: Optional[torch.Tensor] = None
    """[sum(prefix_swa_count + min(n_csa, index_topk))] int32 GPU — CSA topk
    (head) + SWA history (tail) per token. CSA section is filled per-layer by
    `csa_translate_pack`; SWA prefix section is filled by builder at the slice
    tail (head-CSA / tail-SWA convention, matching decode, #1116)."""
    kv_indptr_prefix_csa: Optional[torch.Tensor] = None
    """[total_tokens + 1] int32 GPU — packed cumsum of
    `prefix_swa_count + min(n_committed_csa, index_topk)`."""
    kv_indices_prefix_hca: Optional[torch.Tensor] = None
    """[sum(prefix_swa_count + n_committed_hca)] int32 GPU — SWA history
    (head) + HCA all-committed compress entries (tail). Layer-invariant,
    fully filled by builder."""
    kv_indptr_prefix_hca: Optional[torch.Tensor] = None
    """[total_tokens + 1] int32 GPU — packed cumsum of
    `prefix_swa_count + n_committed_hca`."""
    kv_indices_extend: Optional[torch.Tensor] = None
    """[sum(extend_count)] int32 GPU — flat row offsets into the per-fwd
    `kv` tensor (in-chunk SWA tail) for the extend region. Layer-invariant
    (same `kv` shared by all 3 ratios; one builder pass)."""
    kv_indptr_extend: Optional[torch.Tensor] = None
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
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        return DeepseekV4AttentionMetadataBuilder


class DeepseekV4AttentionMetadataBuilder(CommonAttentionBuilder):
    """Per-request cache owner for V4's state-cache buffers.

    Inherits CommonAttentionBuilder for the standard prefill/decode prep
    (slot_mapping, block_tables, cu_seqlens). PR3-pre2c-B sets `block_size`
    to lcm(m, m') = 128 (V4-Pro: m=4 CSA, m'=128 HCA), matching paper §3.6.1's
    requirement that each classical KV cache block hold an integral number of
    compressed entries per layer (k1=lcm/m=32 CSA, k2=lcm/m'=1 HCA).
    """

    block_size = 128

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

        # Classical KV pool geometry. block_size=128 original tokens means
        # each V4 block holds k1=128/4=32 CSA entries and k2=128/128=1 HCA
        # entry per layer (paper §3.6.1).
        self.k1_csa = self.block_size // 4  # = 32
        self.k2_hca = self.block_size // 128  # = 1

        self._state_dtype = torch.float32  # fp32 required for softmax-pool
        self._swa_dtype = torch.bfloat16  # SWA window matches KV dtype
        self._classical_dtype = torch.bfloat16  # CSA Main / HCA Main KV is BF16
        # CSA Indexer cache is FP8 + 4-byte fp32 scale per row, aligned to 16
        # bytes (matches V3.2 sparse MLA pattern; avoids torch inductor
        # unaligned-access slowdowns). Written by `indexer_k_quant_and_cache`,
        # read by `cp_gather_indexer_k_quant_cache`.
        self._aligned_index_dim = ((self.index_head_dim + 4 + 15) // 16) * 16

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
        # Worst-case HCA per-token committed compress count
        # (= max_model_len // 128 for V4-Pro = 8192 at 1M context).
        self.max_committed_hca = model_runner.config.max_model_len // 128

        # Sparse-attn + per-fwd metadata buffers (CG-A: pre-allocate for fixed
        # GPU pointers, prerequisite for CUDAGraph capture). All H2D copies in
        # the V4 metadata builder go through these buffers via the
        # `np[:n] = arr; copy_to_gpu(n)` pattern instead of per-call
        # `torch.as_tensor(arr)` allocations.
        self._alloc_v4_metadata_buffers()

        self._ubatch_decode_meta: Optional[list] = None

    @property
    def prep_stream(self):
        return self.model_runner.async_execute_stream

    # ------------------------------------------------------------------ #
    # AttentionMetadataBuilder hooks (per-request cache abstraction).    #
    # ------------------------------------------------------------------ #

    def compute_per_req_cache_bytes(self) -> int:
        """Bytes for ONE request's state cache across all layers.

        State cache contents (paper §3.6.1):
          - SWA segment: [n_win, head_dim] BF16, every layer.
          - Compressor tail buffers: [kv_state, score_state] fp32 pairs
            for every Compressor instance (CSA Main / CSA Indexer / HCA Main).
        """
        elem_state = self._state_dtype.itemsize  # fp32 = 4
        elem_swa = self._swa_dtype.itemsize  # bf16 = 2
        # Tail buffers (kv_state + score_state pair per Compressor instance).
        csa_main = self._numel(self.csa_main_state_shape) * 2 * elem_state
        csa_idx = self._numel(self.csa_idx_state_shape) * 2 * elem_state
        hca_main = self._numel(self.hca_main_state_shape) * 2 * elem_state
        # SWA window per layer. Cache holds `win_with_spec = win + mtp_k`
        # slots so MTP draft tokens don't alias verified-token slots.
        swa_per_layer = self.win_with_spec * self.head_dim * elem_swa
        return (
            len(self.csa_layers) * (csa_main + csa_idx)
            + len(self.hca_layers) * hca_main
            + self.num_layers * swa_per_layer
        )

    def slots_per_req(self) -> int:
        # State cache is one slot per req regardless of MTP. The MTP draft
        # lookahead bytes are absorbed into per-slot SWA size via
        # `win_with_spec` (above), not into a slots_per_req multiplier.
        return 1

    def compute_block_bytes(self) -> int:
        """Per-V4-block bytes for the three classical KV pools.

        Each V4 block (block_size=128 original tokens) stores per layer:
          - CSA Main:   k1=32 entries × head_dim BF16
          - CSA Indexer: k1=32 entries × aligned_index_dim bytes FP8
                        (= ((index_head_dim + 4 + 15) // 16) * 16 — 16-byte
                        alignment matches V3.2 sparse MLA index cache and
                        avoids unaligned-access slowdowns in torch inductor.
                        FP8 quantized data + 4-byte fp32 scale interleaved
                        per row; written by `indexer_k_quant_and_cache`,
                        read by `cp_gather_indexer_k_quant_cache`).
          - HCA Main:   k2=1 entry × head_dim BF16
        """
        elem_bf16 = self._classical_dtype.itemsize
        csa_main_per_block = self.k1_csa * self.head_dim * elem_bf16
        csa_idx_per_block = self.k1_csa * self._aligned_index_dim  # fp8 = 1B
        hca_main_per_block = self.k2_hca * self.head_dim * elem_bf16
        return (
            len(self.csa_layers) * (csa_main_per_block + csa_idx_per_block)
            + len(self.hca_layers) * hca_main_per_block
        )

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
        Layer-major axis order `[n_csa, NB, k1, aligned_dim]` so each
        per-CSA slice `pool[pos]` is contiguous in storage; the kernel
        infers `block_size` from `kv_cache.shape[1]`.
        """
        runner = self.model_runner
        device = runner.device
        num_blocks = runner.num_physical_kvcache_blocks
        n_csa = len(self.csa_layers)
        return {
            "v4_csa_idx_kv": torch.zeros(
                (n_csa, num_blocks, self.k1_csa, self._aligned_index_dim),
                dtype=dtypes.fp8,
                device=device,
            ),
        }

    def allocate_per_req_cache(self, num_slots: int) -> dict[str, object]:
        """Allocate per-layer `unified_kv` + Compressor state caches.

        Per-layer `unified_kv` layout (decode-time paged_decode kernel reads
        a single base ptr; offsets `[0, swa_pages)` are SWA, `[swa_pages, ..)`
        are compress). Per-slot SWA region is `win_with_spec = win + mtp_k`
        (extra slack so MTP draft tokens don't overwrite the verified token's
        ring slot mid-fwd):
            Dense layer: [num_slots*win_with_spec,            head_dim] BF16
            CSA   layer: [num_slots*win_with_spec + NB*k1,    head_dim] BF16
            HCA   layer: [num_slots*win_with_spec + NB*k2,    head_dim] BF16

        `build_kv_cache_tensor` slices per-layer views to bind into
        `attn.swa_kv` (SWA portion, reshape to [num_slots, win, head_dim])
        and `compressor.kv_cache` (compress portion, reshape to
        [num_blocks, k_per_block, head_dim]). The full unified pool is also
        bound as `attn.unified_kv` for the paged_decode dispatch.

        Tensors are setattr'd onto ModelRunner; `v4_unified_kv` is a list of
        per-layer tensors (length `num_layers`). State caches stay
        layer-major batched (compressor scatter binds per-layer slices).

        Total bytes match the pre-Phase-A layout (SWA + CSA Main + HCA Main
        bytes redistributed across per-layer tensors; no extra overhead).
        """
        assert self._swa_dtype == self._classical_dtype, (
            "unified_kv requires SWA dtype == classical KV dtype "
            f"(got SWA={self._swa_dtype}, classical={self._classical_dtype})"
        )
        device = self.model_runner.device
        num_blocks = self.model_runner.num_physical_kvcache_blocks
        n_csa = len(self.csa_layers)
        n_hca = len(self.hca_layers)
        swa_pages = num_slots * self.win_with_spec
        head_dim = self.head_dim
        dtype = self._swa_dtype

        # Per-layer unified_kv: SWA prefix + (CSA/HCA) compress tail.
        unified_kv: list[torch.Tensor] = []
        ratios = self.compress_ratios
        for layer_id in range(self.num_layers):
            ratio = ratios[layer_id]
            if ratio == 4:
                compress_pages = num_blocks * self.k1_csa
            elif ratio == 128:
                compress_pages = num_blocks * self.k2_hca
            else:
                compress_pages = 0  # Dense
            unified_kv.append(
                torch.zeros(
                    (swa_pages + compress_pages, head_dim),
                    dtype=dtype,
                    device=device,
                )
            )

        # ---- Compressor state tensors (compute-contiguous) ------------------
        csa_main_kv = self._zero_state(
            (n_csa, num_slots, *self.csa_main_state_shape), device
        )
        csa_main_score = self._neg_inf_state(
            (n_csa, num_slots, *self.csa_main_state_shape), device
        )
        csa_idx_kv = self._zero_state(
            (n_csa, num_slots, *self.csa_idx_state_shape), device
        )
        csa_idx_score = self._neg_inf_state(
            (n_csa, num_slots, *self.csa_idx_state_shape), device
        )
        hca_main_kv = self._zero_state(
            (n_hca, num_slots, *self.hca_main_state_shape), device
        )
        hca_main_score = self._neg_inf_state(
            (n_hca, num_slots, *self.hca_main_state_shape), device
        )

        # ---- RDMA staging pool, only allocated in PD disaggregation mode --
        is_pd = bool(getattr(self.model_runner.config, "kv_transfer_config", None))
        state_tensors = [
            csa_main_kv,
            csa_main_score,
            csa_idx_kv,
            csa_idx_score,
            hca_main_kv,
            hca_main_score,
        ]
        state_slot_stride = sum(t[0, 0].numel() * t.shape[0] for t in state_tensors)
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
            "v4_unified_kv": unified_kv,
            "v4_csa_main_kv_state": csa_main_kv,
            "v4_csa_main_score_state": csa_main_score,
            "v4_csa_idx_kv_state": csa_idx_kv,
            "v4_csa_idx_score_state": csa_idx_score,
            "v4_hca_main_kv_state": hca_main_kv,
            "v4_hca_main_score_state": hca_main_score,
            "v4_state_pool": state_pool,
            "v4_state_pool_size": pool_size,
            "v4_state_slot_stride": state_slot_stride,
        }

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
        num_slots = self.model_runner.max_per_req_cache_slots
        swa_pages = num_slots * self.win_with_spec

        if isinstance(module, _V4Attention):
            # Bind both:
            #   - `attn.unified_kv`: the full per-layer pool (paged_decode reads).
            #   - `attn.swa_kv`: a [num_slots, win_with_spec, head_dim] view
            #     onto the SWA prefix. Per-slot dim is `win + mtp_k` so MTP
            #     draft tokens have their own ring slots; `swa_write` modulo
            #     and `paged_decode` per-row case_c modulo both use this
            #     dim (= `swa_kv.shape[1]`).
            unified = runner.v4_unified_kv[module.layer_id]
            module.unified_kv = unified
            module.swa_kv = unified[:swa_pages].view(
                num_slots, self.win_with_spec, self.head_dim
            )
            return None

        if isinstance(module, _V4Indexer):
            # Indexer.kv_cache — CSA Indexer compressed pool, per CSA layer.
            # prefix: "layers.<L>.attn.indexer"
            #
            # Shape MUST stay [NB, k1_csa, aligned_dim] (3D, block_size dim
            # explicit) because `cp_gather_indexer_k_quant_cache` infers
            # block_size from `kv_cache.shape[1]` to compute
            # `physical_block * block_size + slot_in_block`. Flattening to
            # [NB*k1, 1, aligned_dim] makes the kernel see block_size=1 and
            # OOB-index block_table. Matches V3.2's [num_blocks, block_size,
            # head_dim] layout (deepseek_v2.py:1049).
            layer_id_from_prefix = int(module.prefix.split(".")[1])
            pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
            module.kv_cache = runner.v4_csa_idx_kv[pos]
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
                # outer Indexer.kv_cache (csa_idx_kv). Same [NB, k1_csa,
                # aligned_dim] FP8 shape — `Compressor.forward` resolves
                # slot via block_table+ci internally (no flat slot_mapping
                # needed; matches CSA Main's path).
                idx_kv = runner.v4_csa_idx_kv[pos]
                module.kv_cache = idx_kv
                # FP8 quant path: bind a strided fp32 view of the per-block
                # scale region. Layout per block: [k1*head_dim FP8 region]
                # then [k1 fp32 scale region] then padding (cache_kernels.cu
                # :1209-1239). Strides expressed in fp32 elements.
                nb, k1, aligned_dim = idx_kv.shape
                head_dim = self.index_head_dim
                assert (
                    k1 * aligned_dim
                ) % 4 == 0, f"per-block bytes ({k1 * aligned_dim}) must be 4-aligned"
                block_fp32_stride = (k1 * aligned_dim) // 4
                scale_fp32_offset = (k1 * head_dim) // 4
                module.cache_scale = (
                    idx_kv.view(torch.float32)
                    .view(-1)
                    .as_strided(
                        size=(nb, k1),
                        stride=(block_fp32_stride, 1),
                        storage_offset=scale_fp32_offset,
                    )
                )
            elif ratio == 4:
                pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_csa_main_kv_state[pos]
                module.score_state = runner.v4_csa_main_score_state[pos]
                # CSA Main compressed pool now lives in the tail of the
                # owning layer's `unified_kv`. Compressor.forward writes via
                # `kv_cache[block_id, slot_in_block, :] = entry`, so we hand
                # it the standard [num_blocks, k1, head_dim] view.
                num_blocks = runner.num_physical_kvcache_blocks
                unified = runner.v4_unified_kv[layer_id_from_prefix]
                module.kv_cache = unified[swa_pages:].view(
                    num_blocks, self.k1_csa, self.head_dim
                )
            elif ratio == 128:
                pos = self.layer_id_to_hca_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_hca_main_kv_state[pos]
                module.score_state = runner.v4_hca_main_score_state[pos]
                num_blocks = runner.num_physical_kvcache_blocks
                unified = runner.v4_unified_kv[layer_id_from_prefix]
                module.kv_cache = unified[swa_pages:].view(
                    num_blocks, self.k2_hca, self.head_dim
                )
            else:
                raise ValueError(
                    f"Unknown V4 compress_ratio={ratio} on Compressor at "
                    f"prefix={module.prefix!r}"
                )
            return None

        return super().build_kv_cache_tensor(layer_id, module)

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.types import (
            KVTransferRegion,
            KVTransferTensors,
        )

        runner = self.model_runner
        if not hasattr(runner, "v4_unified_kv"):
            return None

        num_slots = runner.max_per_req_cache_slots
        swa_pages = num_slots * self.win_with_spec
        elem_bf16 = 2
        elem_fp32 = 4

        block_regions: list[KVTransferRegion] = []
        slot_regions: list[KVTransferRegion] = []

        # Block regions: compress tail per layer
        for layer_id in range(self.num_layers):
            uv = runner.v4_unified_kv[layer_id]
            compress_base = uv.data_ptr() + swa_pages * self.head_dim * elem_bf16
            compress_total = (
                uv.numel() * elem_bf16 - swa_pages * self.head_dim * elem_bf16
            )
            if compress_total <= 0:
                continue
            ratio = self.compress_ratios[layer_id]
            if ratio == 4:
                bpb = self.k1_csa * self.head_dim * elem_bf16
            elif ratio == 128:
                bpb = self.k2_hca * self.head_dim * elem_bf16
            else:
                continue
            block_regions.append(KVTransferRegion(compress_base, compress_total, bpb))

        # Block regions: CSA Indexer KV (FP8)
        for pos in range(len(self.csa_layers)):
            t = runner.v4_csa_idx_kv[pos]
            bpb = self.k1_csa * self._aligned_index_dim
            block_regions.append(
                KVTransferRegion(t.data_ptr(), t.numel() * t.element_size(), bpb)
            )

        # Slot regions: SWA per layer
        swa_slot_bytes = self.win_with_spec * self.head_dim * elem_bf16
        for layer_id in range(self.num_layers):
            uv = runner.v4_unified_kv[layer_id]
            slot_regions.append(
                KVTransferRegion(
                    uv.data_ptr(),
                    swa_pages * self.head_dim * elem_bf16,
                    swa_slot_bytes,
                )
            )

        # Staging pool for compressor states (not in slot_regions — managed
        # separately by the connector with pool acquire/release).
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
            )
            state_tensors = [
                runner.v4_csa_main_kv_state,
                runner.v4_csa_main_score_state,
                runner.v4_csa_idx_kv_state,
                runner.v4_csa_idx_score_state,
                runner.v4_hca_main_kv_state,
                runner.v4_hca_main_score_state,
            ]
            gather_slot = self._make_gather_slot(pool, stride, state_tensors)
            scatter_slot = self._make_scatter_slot(pool, stride, state_tensors)

        return KVTransferTensors(
            block_regions=block_regions,
            slot_regions=slot_regions,
            num_blocks=runner.num_physical_kvcache_blocks,
            num_slots=num_slots,
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
    ) -> None:
        """Build and attach the CSA Indexer per-fwd GPU metadata.

        Hoists per-CSA-layer H2D calls (batch_id_per_token / cu_committed /
        n_committed / seq_base_per_token / cu_ends) into a single per-fwd
        build. None for warmup or empty fwd; `_build_v4_indexer_meta`
        handles both.
        """
        attn_metadata.indexer_meta = self._build_v4_indexer_meta(
            attn_metadata=attn_metadata,
            positions_gpu=positions_gpu,
            scheduled_bs=scheduled_bs,
            total_tokens=total_tokens,
            device=self.device,
        )

    def _build_v4_indexer_meta(
        self,
        *,
        attn_metadata: AttentionMetaData_DSV4,
        positions_gpu,
        scheduled_bs: int,
        total_tokens: int,
        device,
    ):
        """Build per-fwd GPU index tensors consumed by `Indexer.forward_batched`.

        Returns None for warmup batches (the indexer falls back to its
        inline H2D path) or when CSA / Indexer is not on the model. CSA
        ratio is fixed at 4; we always build under that assumption.

        Reads pre-computed `attn_metadata.n_committed_csa_per_seq_cpu`
        (set by `_attach_v4_per_fwd_meta`, which MUST run first) for the
        per-seq committed count and cumsums it on CPU.

        Reuses two shared GPU tensors also set by `_attach_v4_per_fwd_meta`:
          - `attn_metadata.batch_id_per_token`        [padded_T] int32
          - `attn_metadata.n_committed_csa_per_seq`   [bs] int32

        DECODE fast path: returns a minimal dict with only
        `n_committed_per_seq_gpu` (the single field `_score_topk_decode`
        reads). The cumsum + H2D + per-token GPU derivations below are all
        prefill-only — `deepgemm_fp8_paged_mqa_logits` + `top_k_per_row_decode`
        operate directly on paged KV via `n_committed_per_seq_gpu`, never on
        the packed-cumsum / per-token `cu_starts/cu_ends` layout.

        The FP8 indexer K-cache write happens inside `fused_compress_attn`
        (the unified Indexer-inner Compressor path) via the same block_tables
        that CSA Main uses; no separate slot_mapping is built here.
        """

        # Caller contract: scheduled_bs >= 1, total_tokens >= 1 (same
        # invariants as `_attach_v4_per_fwd_meta` — guaranteed by every
        # prepare_*/CG-capture path).
        bs = scheduled_bs

        # DECODE short-circuit: the only field `_score_topk_decode` consumes is
        # `n_committed_per_seq_gpu`, which is the same tensor as
        # `attn_metadata.n_committed_csa_per_seq` (already staged by
        # `_attach_v4_per_fwd_meta`). The prefill-only derivations below
        # (CPU cumsum + H2D for `cu_committed_gpu`; 7 GPU launches for
        # `seq_base`/`visible_end`/`cu_ends`) feed `_score_topk_prefill` only
        # (cp_gather + fp8_mqa_logits + per-row prefill top-k), so they are
        # dead work on the decode hot path. ~50μs / fwd saved at bs=1024.
        if attn_metadata.state is AttnState.DECODE:
            return {
                "n_committed_per_seq_gpu": attn_metadata.n_committed_csa_per_seq,
            }

        ratio = 4  # CSA — also referenced by `visible_end_gpu` below
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

        # batch_id_per_token + n_committed_csa: reuse the shared GPU
        # tensors set in `_attach_v4_per_fwd_meta` (which MUST run before
        # this helper — see prepare_decode/prefill ordering). int64
        # batch_id is mandated by PyTorch fancy indexing; int32 n_committed
        # is the gather SOURCE so any dtype works (and both downstream
        # kernels — `deepgemm_fp8_paged_mqa_logits`, `top_k_per_row_decode`
        # — want int32 anyway).
        batch_id_per_token_gpu = attn_metadata.batch_id_per_token[:total_tokens]
        n_committed_per_seq_gpu = attn_metadata.n_committed_csa_per_seq
        # cu_committed_gpu is consumed both as `cu_starts/cu_ends` for the
        # fp8_mqa_logits per-token range AND as `cu_seq_lens` for the
        # cp_gather_indexer_k_quant_cache call (per-seq cumulative committed K).
        cu_committed_gpu = self._stage("v4_indexer_cu_committed", cu_committed_cpu)

        # Layer-invariant GPU derivations (each was previously rebuilt ~30x
        # per fwd inside the per-CSA-layer body).
        seq_base_per_token_gpu = cu_committed_gpu[batch_id_per_token_gpu].to(
            torch.int32
        )  # [total_tokens] int32 — per-token offset into concat'd seqs'
        # compressed K. Used as `cu_starts` for fp8_mqa_logits AND as the
        # subtraction base for prefill `top_k_per_row_prefill`'s GLOBAL output
        # → seq-local conversion (the indexer kernel writes
        # `seq_base + col_in_seq`; we recover col_in_seq by subtracting).
        visible_end_gpu = torch.minimum(
            (positions_gpu[:total_tokens] + 1) // ratio,
            n_committed_per_seq_gpu[batch_id_per_token_gpu],
        ).to(
            torch.int32
        )  # [total_tokens] int32 — per-token causal upper bound
        cu_ends_gpu = (
            seq_base_per_token_gpu + visible_end_gpu
        )  # [total_tokens] int32 — fp8_mqa_logits per-token end offset

        return {
            "total_committed": total_committed,
            "cu_committed_gpu": cu_committed_gpu,
            "n_committed_per_seq_gpu": n_committed_per_seq_gpu,  # int32, [bs]
            "batch_id_per_token_gpu": batch_id_per_token_gpu,  # int32, [total_tokens]
            # Prefill-only fields below — decode never consults them. NOT
            # in pre-allocated buffers (per-fwd derived); CG capture path
            # would see stale pointers, but the decode path doesn't touch
            # them, so it's fine.
            "seq_base_per_token_gpu": seq_base_per_token_gpu,
            "cu_starts_gpu": seq_base_per_token_gpu,  # alias for fp8_mqa_logits
            "cu_ends_gpu": cu_ends_gpu,
        }

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: torch.Tensor,  # [bs] int — eagle's current draft-step positions
        only_update: bool = False,
        num_reject_tokens: Optional[torch.Tensor] = None,
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
        # `max_per_req_cache_slots` is set inside `model_runner.get_num_blocks`
        # AFTER `warmup_model`. During warmup we no-op — warmup discards
        # draft output anyway, and the verify-shape attn_metadata stays valid.
        if not getattr(self.model_runner, "max_per_req_cache_slots", 0):
            return {}
        assert self._mtp_layers_are_swa_only, (
            "prepare_mtp_decode fast path only supports SWA-only MTP layers "
            f"(compress_ratio==0); got compress_ratios[mtp]="
            f"{self.compress_ratios[self._n_main_layers:]}"
        )

        var = self.model_runner.forward_vars
        attn_metadata = cast(
            AttentionMetaData_DSV4, get_forward_context().attn_metadata
        )
        # Pre-populated by the verify-forward `prepare_decode` and kept alive
        # across eagle.propose; assert for the static checker.
        assert attn_metadata.context_lens is not None
        assert attn_metadata.state_slot_mapping is not None
        win = self.window_size  # SWA prefix max per token
        cs = self.win_with_spec  # SWA ring stride = win + max_spec_steps

        # ----- GPU-side SWA indptr math (no CPU numpy, no D2H) -----
        # ctx_gpu is already correct (rolled-back by prepare_decode + bumped
        # by eagle). int32 in the source buffer; keep dtype throughout.
        # Only SWA is computed; CSA/HCA indices are unused by SWA-only MTP
        # (asserted above) and will be fully rebuilt by the next verify-fwd's
        # `prepare_decode`.
        actual_swa = torch.clamp(positions + 1, max=win)

        swa_indptr = var["v4_kv_indptr_swa"].gpu[: bs + 1]
        # positions/actual_swa are int64 (eagle's positions buffer); cast to
        # int32 inside cumsum to match swa_indptr's int32 storage.
        torch.cumsum(actual_swa, dim=0, dtype=torch.int32, out=swa_indptr[1:])

        # batch_id_per_token: 1-tok-per-seq → arange(bs). Eagle already
        # populated cu_seqlens_q as arange(bs+1) (eagle.py:430), so its
        # [:bs] slice IS [0,1,...,bs-1] — exactly the per-token batch id.
        # No extra alloc / arange kernel.
        assert attn_metadata.cu_seqlens_q is not None
        batch_id_per_token = attn_metadata.cu_seqlens_q[:bs]

        # ----- Kernel: write SWA prefix paged offsets -----
        # `write_v4_paged_decode_indices` writes to swa/csa/hca indices in
        # one pass. For SWA-only MTP we alias csa/hca slots to swa so the
        # kernel writes the same value three times to swa_indices_buf —
        # redundant ~bs*win stores (~8 KB at V4-Pro bs=64, win=128), saves
        # building two extra unused indptr buffers.
        swa_indices_buf = var["v4_kv_indices_swa"].gpu
        write_v4_paged_decode_indices(
            state_slot_per_seq=attn_metadata.state_slot_mapping,
            batch_id_per_token=batch_id_per_token,
            positions=positions,
            swa_indptr=swa_indptr,
            csa_indptr=swa_indptr,
            hca_indptr=swa_indptr,
            swa_indices=swa_indices_buf,
            csa_indices=swa_indices_buf,
            hca_indices=swa_indices_buf,
            T=bs,
            win=win,
            cs=cs,
        )

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

        # NOT rebuilt (unused by SWA-only MTP layer; would block a future
        # CSA/HCA MTP layer — assert at top guards):
        #   - n_committed_{csa,hca}_per_seq{,_cpu} (compressor/HCA tail math)
        #   - skip_prefix_len_csa (csa_translate_pack per-layer write)
        #   - compress_plans (Compressor — only present when ratio != 0)
        #   - HCA compress tail in kv_indices_hca
        #   - v4 indexer meta (Indexer — only present when ratio == 4)
        return {}

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        """V4-style decode prep: populates positions, cu_seqlens_q,
        block_tables, and state_slot_mapping.

        Uses stream overlap (like AiterMLAMetadataBuilder) to hide H2D
        latency behind CPU numpy work: basic H2D copies fire on
        ``prep_stream`` while ``_build_compress_plans`` runs on the CPU.
        """
        var = self.model_runner.forward_vars
        scheduled_bs = batch.total_seqs_num_decode
        context_lens_np = np.asarray(batch.context_lens, dtype=np.int32)
        max_seqlen_q = batch.num_spec_step + 1
        # MTP: roll back ctx by `num_rejected` so this fwd's positions overwrite
        # last fwd's rejected-draft slots (matches aiter_mla.py:701 /
        # aiter_attention.py:542). `batch.context_lens` = `seq.num_tokens`
        # which the scheduler advances by `mtp_k - num_rejected` placeholders
        # per fwd (scheduler.py:789); without this rollback, MTP-k positions
        # would skip ahead by `num_rejected` and the rejected slots would
        # never be overwritten with the corrected K/V. `num_rejected` is None
        # on dummy runs and on the first fwd before any sampler output.
        # Bound n_committed_csa/hca via the rolled-back ctx (n_committed_* =
        # ctx // 4 / 128 in `_attach_v4_paged_decode_meta`), so block_tables
        # truncation isn't needed here — the per-token kv_len already shrinks.
        if not batch.is_dummy_run and max_seqlen_q > 1:
            num_rejected = self.model_runner.tokenID_processor.num_rejected
            if num_rejected is not None:
                context_lens_np = context_lens_np - num_rejected.astype(np.int32)
        positions_np = np.tile(
            np.arange(max_seqlen_q, dtype=np.int32), scheduled_bs
        ) + np.repeat(context_lens_np - max_seqlen_q, max_seqlen_q)
        sum_scheduled_tokens = batch.total_tokens_num_decode

        var["positions"].np[:sum_scheduled_tokens] = positions_np

        var["context_lens"].np[:scheduled_bs] = context_lens_np

        # Inline block_tables CPU fill (H2D deferred to prep_stream).
        self.prepare_block_tables(batch)

        state_slot_np = np.asarray(
            batch.per_req_cache_groups[:scheduled_bs], dtype=np.int32
        )
        if len(state_slot_np) < scheduled_bs:
            state_slot_np = np.zeros(scheduled_bs, dtype=np.int32)
        ss_buf = var["v4_meta_state_slot_groups"]
        ss_buf.np[:scheduled_bs] = state_slot_np

        # ---- fire H2D on prep_stream ----
        # NB: this runs inside attn_metadata_builder.build(), BEFORE
        # set_forward_context() — can't read main_stream from the context yet.
        prep_stream = self.prep_stream
        current_stream = torch.cuda.current_stream()
        prep_stream.wait_stream(current_stream)
        with torch.cuda.stream(prep_stream):
            positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
            cu_seqlens_q_gpu = var["cu_seqlens_q"].copy_to_gpu(bs + 1)
            context_lens_gpu = var["context_lens"].copy_to_gpu(scheduled_bs)
            block_tables_gpu = var["block_tables"].copy_to_gpu(scheduled_bs)
            state_slot_gpu = ss_buf.copy_to_gpu(scheduled_bs)

        # ---- CPU numpy work, overlapped with prep_stream H2D ----
        extend_lens_np = np.full(scheduled_bs, max_seqlen_q, dtype=np.int32)
        compress_plans = self._build_compress_plans(
            extend_lens_np,
            context_lens_np,
            for_decode_cg=True,
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
        attn_metadata.state_slot_mapping = state_slot_gpu
        attn_metadata.state_slot_mapping_cpu = state_slot_np
        attn_metadata.compress_plans = compress_plans

        padded_bs = int(bs)
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            extend_lens_np,  # = np.full(scheduled_bs, max_seqlen_q) for decode
            state_slot_np,
            scheduled_bs,
            sum_scheduled_tokens,
            padded_bs=padded_bs,
            max_q_len=max_seqlen_q,
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
                bs=bs,
                max_seqlen_q=max_seqlen_q,
                context_lens_np=context_lens_np,
                state_slot_np=state_slot_np,
                positions_np=positions_np,
            )

        return attn_metadata, positions

    def _prepare_ubatch_decode(
        self,
        *,
        scheduled_bs: int,
        bs: int,
        max_seqlen_q: int,
        context_lens_np: np.ndarray,
        state_slot_np: np.ndarray,
        positions_np: np.ndarray,
    ) -> None:
        """Split a decode batch into two micro-batches (by request) and build
        each one's V4 decode metadata into ``ub{0,1}_`` prefixed buffers.

        Mirrors :meth:`prepare_decode` but operates on a per-ubatch request
        slice. The two resulting :class:`AttentionMetaData_DSV4` objects are
        cached on ``self._ubatch_decode_meta`` and returned by
        :meth:`build_ubatch_metadata`.

        Token layout in a decode fwd is request-major with ``max_seqlen_q``
        tokens per request, so ubatch token ranges fall on request boundaries.
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
                UBatchWrapper._decode_ub_padded_bs(ctx, i, N, bs) for i in range(N)
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
        for ub_idx, (req_start, req_end) in enumerate(ub_ranges):
            p = f"ub{ub_idx}_"
            padded_bs = padded_list[ub_idx]
            # Real requests that fall into this ubatch's [req_start, req_end),
            # clamped to scheduled_bs (cudagraph pad rows beyond scheduled_bs
            # carry sentinels, exercised only during capture's synthetic batch).
            ub_real_reqs = max(0, min(scheduled_bs, req_end) - req_start)
            tok_start = req_start * max_seqlen_q
            ub_real_tokens = ub_real_reqs * max_seqlen_q

            # ---- per-seq slices into ub buffers ----
            ub_ctx_np = context_lens_np[req_start : req_start + ub_real_reqs]
            var[f"{p}context_lens"].np[:ub_real_reqs] = ub_ctx_np
            var[f"{p}context_lens"].np[ub_real_reqs:padded_bs] = 0

            ub_state_np = state_slot_np[req_start : req_start + ub_real_reqs]
            if len(ub_state_np) < ub_real_reqs:
                ub_state_np = np.zeros(ub_real_reqs, dtype=np.int32)
            var[f"{p}v4_meta_state_slot_groups"].np[:ub_real_reqs] = ub_state_np
            var[f"{p}v4_meta_state_slot_groups"].np[ub_real_reqs:padded_bs] = 0
            state_slot_np_ub = (
                var[f"{p}v4_meta_state_slot_groups"].np[:padded_bs].copy()
            )

            var[f"{p}block_tables"].np[:ub_real_reqs] = var["block_tables"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}block_tables"].np[ub_real_reqs:padded_bs] = 0

            # positions: copy the ubatch's token slice (values match the global
            # positions slice the UBatchWrapper Context will expose).
            ub_positions_np = positions_np[tok_start : tok_start + ub_real_tokens]
            var[f"{p}positions"].np[:ub_real_tokens] = ub_positions_np
            var[f"{p}positions"].np[ub_real_tokens : padded_bs * max_seqlen_q] = 0

            # cu_seqlens_q: uniform max_seqlen_q per real req, padded tail flat.
            cu = np.arange(
                0, (ub_real_reqs + 1) * max_seqlen_q, max_seqlen_q, dtype=np.int32
            )
            var[f"{p}cu_seqlens_q"].np[: ub_real_reqs + 1] = cu
            var[f"{p}cu_seqlens_q"].np[ub_real_reqs + 1 : padded_bs + 1] = (
                ub_real_reqs * max_seqlen_q
            )

            # ---- H2D ----
            ub_sum_tokens = max(ub_real_tokens, 1)
            positions_gpu = var[f"{p}positions"].copy_to_gpu(padded_bs * max_seqlen_q)
            cu_seqlens_q_gpu = var[f"{p}cu_seqlens_q"].copy_to_gpu(padded_bs + 1)
            context_lens_gpu = var[f"{p}context_lens"].copy_to_gpu(padded_bs)
            block_tables_gpu = var[f"{p}block_tables"].copy_to_gpu(padded_bs)
            state_slot_gpu = var[f"{p}v4_meta_state_slot_groups"].copy_to_gpu(padded_bs)

            # ---- compress plans (per ubatch buffer set) ----
            extend_lens_np = np.full(ub_real_reqs, max_seqlen_q, dtype=np.int32)
            ctx_for_plan = context_lens_np[req_start : req_start + ub_real_reqs]
            compress_plans = self._build_compress_plans(
                extend_lens_np,
                ctx_for_plan,
                for_decode_cg=True,
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
            attn_metadata.state_slot_mapping = state_slot_gpu
            attn_metadata.state_slot_mapping_cpu = state_slot_np_ub
            attn_metadata.compress_plans = compress_plans

            # token_num_per_seq over PADDED bs (pad reqs contribute max_seqlen_q
            # each so batch_id_per_token covers padded_total_tokens).
            token_num_per_seq = np.full(ub_real_reqs, max_seqlen_q, dtype=np.int32)
            self._attach_v4_per_fwd_meta(
                attn_metadata,
                token_num_per_seq,
                state_slot_np_ub,
                ub_real_reqs,
                ub_real_tokens,
                padded_bs=padded_bs,
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
        self, ubatch_idx: int, padded_bs: int
    ) -> AttentionMetaData_DSV4:
        assert self._ubatch_decode_meta is not None, (
            "build_ubatch_metadata called but no ubatch decode metadata was "
            "prepared — ensure enable_tbo_decode is set and prepare_decode ran."
        )
        return self._ubatch_decode_meta[ubatch_idx]

    def prepare_prefill(self, batch: ScheduledBatch):
        """V4 prefill prep: extends parent to always populate block_tables
        and state_slot_mapping.

        The parent only emits block_tables when has_cached (prefix cache hit);
        V4 always needs block_tables because Compressor scatters compressed
        entries into the classical KV pool from token 0 onwards.

        Also publishes CPU mirrors (`v4_*_cpu`) consumed by the V4 forward
        path to avoid `.item()` / `.tolist()` syncs (PR-A Phase 2).
        """
        base_md, positions = super().prepare_prefill(batch)
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
        state_slot_gpu, state_slot_np = self._populate_state_slot_mapping(
            batch, scheduled_bs, return_cpu=True
        )
        attn_metadata.state_slot_mapping = state_slot_gpu
        # PR-A Phase 2 CPU mirrors (generic, not V4-specific). The parent
        # populated forward_vars CPU buffers; read them back as numpy slices.
        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        positions_np = np.asarray(var["positions"].np[:sum_scheduled_tokens])
        cu_seqlens_q_np = np.asarray(var["cu_seqlens_q"].np[: scheduled_bs + 1])
        attn_metadata.state_slot_mapping_cpu = state_slot_np
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
            extend_lens_np, context_lens_np, for_decode_cg=False
        )
        # Prefill goes through eager (no CG): defaults make padded_total_tokens
        # collapse to total_tokens — no padding logic kicks in. Must still run
        # BEFORE `_attach_v4_indexer_meta` so the indexer-side meta builder can
        # reuse the shared GPU tensors (batch_id_per_token, n_committed_csa).
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            extend_lens_np,  # = cu_seqlens_q[1:] - cu_seqlens_q[:bs]
            attn_metadata.state_slot_mapping_cpu,
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
        # plus skip_prefix_len_csa and swa_pages.
        self._build_paged_prefill_meta(
            attn_metadata,
            positions_np,
            cu_seqlens_q_np,
            extend_lens_np,
            start_pos_per_seq_np,
            attn_metadata.state_slot_mapping_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
        )

        # ----- PCP: reindex per-query metadata to this rank's 1/W shard -----
        # Mirrors SGLang's apply_cp_reindex (deepseek_v4_backend_hip_radix.py):
        # all metadata above was built for the FULL sequence; under PCP the
        # model.forward entry round-robin-splits hidden/positions to 1/W, so the
        # per-query (per-token) metadata must be reduced to the SAME owned-query
        # set. Per-seq / KV-write fields stay full (every rank keeps full KV).
        if pcp_is_enabled() and not batch.is_dummy_run:
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
        # sliced inputs yields the 1/W layout; per-seq fields (cu_committed,
        # n_committed_per_seq) stay full. Skip if the model has no CSA/indexer.
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
        padded_bs: int,
        ubatch_idx: int = 0,
    ) -> AttentionMetaData_DSV4:
        """Split prefill AttentionMetaData for V4 TBO micro-batches."""
        from atom.utils.tbo.ubatch_splitting import split_attn_metadata

        ub_attn = split_attn_metadata(attn_metadata, ub_slice, padded_bs)
        ub_attn.__class__ = AttentionMetaData_DSV4

        src = cast(AttentionMetaData_DSV4, attn_metadata)
        rs = ub_slice.request_slice
        ts = ub_slice.token_slice
        ub_num_reqs = rs.stop - rs.start
        ub_num_tokens = ts.stop - ts.start

        if src.state_slot_mapping is not None:
            ub_attn.state_slot_mapping = src.state_slot_mapping[rs]
        if src.state_slot_mapping_cpu is not None:
            ub_attn.state_slot_mapping_cpu = src.state_slot_mapping_cpu[rs]

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
            # so `decode_capacity_per_ratio=None` (tight n_compress slice).
            ub_plan_buffers = self._get_ubatch_compress_plan_buffers(ubatch_idx)
            ub_attn.compress_plans = make_compress_plans(
                np.ascontiguousarray(extend_lens_np, dtype=np.int32),
                np.ascontiguousarray(context_lens_np, dtype=np.int32),
                self._unique_compress_ratios_overlap,
                plan_buffers=ub_plan_buffers,
                decode_capacity_per_ratio=None,
            )
        else:
            ub_attn.compress_plans = {}

        # Multiple helpers read context_lens and block_tables from
        # forward_vars by position [0:scheduled_bs]. For ubatch 1 the
        # relevant rows live at [rs.start:rs.stop], not [0:ub_num_reqs].
        # Temporarily place the ubatch's slices at the front so helpers
        # see the right values.
        bt = var["block_tables"].np
        saved_ctx = var["context_lens"].np[:ub_num_reqs].copy()
        saved_bt = bt[:ub_num_reqs].copy()
        try:
            var["context_lens"].np[:ub_num_reqs] = context_lens_np
            bt[:ub_num_reqs] = bt[rs.start : rs.stop].copy()

            self._attach_v4_per_fwd_meta(
                ub_attn,
                extend_lens_np,  # ubatch's per-seq token counts
                ub_attn.state_slot_mapping_cpu,
                ub_num_reqs,
                ub_num_tokens,
            )

            positions_gpu = var["positions"].gpu[ts.start : ts.stop]
            self._attach_v4_indexer_meta(
                ub_attn,
                ub_num_reqs,
                ub_num_tokens,
                positions_gpu=positions_gpu,
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
                ub_attn.state_slot_mapping_cpu,
                ub_num_reqs,
                ub_num_tokens,
                positions_gpu=ub_positions_gpu,
                cu_q_per_seq_gpu=ub_cu_q_per_seq_gpu,
                block_tables_gpu=ub_block_tables_gpu,
            )
        finally:
            bt[:ub_num_reqs] = saved_bt
            var["context_lens"].np[:ub_num_reqs] = saved_ctx

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
            if im.get("n_committed_per_seq_gpu") is not None:
                im["n_committed_per_seq_gpu"] = im["n_committed_per_seq_gpu"].clone()

        return ub_attn

    def _attach_v4_per_fwd_meta(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        token_num_per_seq,
        state_slot_mapping_cpu,
        scheduled_bs: int,
        total_tokens: int,
        *,
        padded_bs: Optional[int] = None,
        max_q_len: Optional[int] = None,
        buf_prefix_ubatch: str = "",
    ) -> None:
        """Hoist per-fwd, layer-invariant metadata used by every V4 layer.

        These tensors only depend on `positions`, `cu_seqlens_q`, `state_slot_mapping`
        and `window_size` — none of which change across layers — so building
        them once per fwd saves ~64 redundant constructions for V4-Pro.

        Sets:
          - `attn_metadata.batch_id_per_token`: [padded_T] int32 batch id
            per token (single per-token mapping; consumed by the Phase B/C/E
            paged-decode kernels and the indexer). `swa_write` no longer
            depends on this — it derives `src_id` from `cu_seqlens_q` inline.
          - `attn_metadata.n_committed_csa_per_seq`: [bs] int32 per-seq
            `ctx_len // 4` (shared by csa_translate_pack + indexer; kernels
            do their own `min(., index_topk)` clamp via mask).
          - `attn_metadata.state_slot_mapping`: [bs] int32 GPU view of
            per-seq state cache slot (already set by prepare_*; passed
            through unchanged here).

        Caller contract: `scheduled_bs >= 1` and `total_tokens >= 1`.
        warmup_model + dummy_run paths both enforce these via min-1 fallbacks
        (model_runner.warmup_model:1003-1011, _populate_state_slot_mapping
        zeros-fill); CG capture uses graph_bs >= 1 too.
        """
        # state is set by the caller at AttentionMetaData_DSV4 construction
        # time (single source of truth — prepare_decode / prepare_prefill /
        # prepare_mtp_decode / build_for_cudagraph_capture each set it).
        # Consumed here for padded_total_tokens sizing.
        is_pure_decode = attn_metadata.state is AttnState.DECODE

        # padded_total_tokens: CG-captured decode/MTP pads to the fixed
        # bucket `padded_bs * (1+max_spec_steps)` so the per-token
        # `batch_id_per_token` buffer has a stable shape across captures.
        # Prefill states (PREFILL_NATIVE / PREFILL_PREFIX) are eager and
        # use `total_tokens` exactly — no wasted padding (a long prefill
        # chunk doesn't need to be padded up to a bucket that doesn't
        # exist for it).
        if is_pure_decode:
            assert padded_bs is not None and max_q_len is not None, (
                "DECODE state requires padded_bs + max_q_len from caller "
                "(CG bucket size — fixed at capture)"
            )
            padded_total_tokens = int(padded_bs) * int(max_q_len)
        else:
            padded_total_tokens = total_tokens

        var = self.model_runner.forward_vars

        # ---- CPU numpy work (all on main thread) ----
        # Build the unpadded mapping once; the padded GPU staging buffer wraps
        # it (head = real, tail = -1 sentinel). Stash the unpadded slice on
        # attn_metadata so `_attach_v4_paged_decode_meta` reuses it instead of
        # re-running `np.repeat(arange, token_num_per_seq)` (saves ~10μs/fwd
        # at bs=1024 + one allocation).
        batch_id_unpadded_np = np.repeat(
            np.arange(scheduled_bs, dtype=np.int32), token_num_per_seq
        )
        batch_id_per_token_np = np.full(padded_total_tokens, -1, dtype=np.int32)
        batch_id_per_token_np[:total_tokens] = batch_id_unpadded_np
        attn_metadata.batch_id_per_token_cpu = batch_id_unpadded_np

        # context_lens is int32 on the buffer; keep dtype through divide so
        # n_committed_{csa,hca} stay int32 (max value ~max_model_len // 4 ≪ 2^31).
        ctx_per_seq_np = var[f"{buf_prefix_ubatch}context_lens"].np[:scheduled_bs]
        # Single source of truth for n_committed_{csa,hca}_per_seq on CPU.
        # Stashed on attn_metadata so paged_decode_meta / paged_prefill_meta /
        # v4_indexer_meta can read instead of each re-running `ctx // k`.
        n_committed_csa_per_seq_np = ctx_per_seq_np // 4
        n_committed_hca_per_seq_np = ctx_per_seq_np // 128
        attn_metadata.n_committed_csa_per_seq_cpu = n_committed_csa_per_seq_np
        attn_metadata.n_committed_hca_per_seq_cpu = n_committed_hca_per_seq_np

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
        # `top_k_per_row_decode` iterates the CAPTURED grid (= padded_bs *
        # next_n rows) and reads `rowEnds[batch_id]` for every row. Its
        # per-row length formula is
        #   `row_len = rowEnds[bid] - next_n + (r % next_n) + 1`
        # — for pad rows `bid ∈ [scheduled_bs, padded_bs)` the buffer slot
        # carries a stale value from a prior fwd; if that stale value is
        # `< next_n - 1` (easy with MTP3 next_n=4 if a prior fwd had a seq
        # in early prefill with ctx ≤ 11), row_len becomes negative and the
        # kernel's radix loop runs unbounded → GPU hang. The downstream
        # `batch_id_per_token = -1` sentinel masks pad rows out of
        # `csa_translate_pack`, so the value just needs to be "big enough"
        # to keep row_len non-negative. Use `index_topk` (≥ 1024 ≫ next_n).
        n_csa_buf = var[f"{buf_prefix_ubatch}v4_n_committed_csa_per_seq"]
        n_csa_buf.np[:scheduled_bs] = n_committed_csa_per_seq_np
        if is_pure_decode and padded_bs is not None and padded_bs > scheduled_bs:
            n_csa_buf.np[scheduled_bs:padded_bs] = self.index_topk
            attn_metadata.n_committed_csa_per_seq = n_csa_buf.copy_to_gpu(padded_bs)
        else:
            attn_metadata.n_committed_csa_per_seq = n_csa_buf.copy_to_gpu(scheduled_bs)

        self._attach_v4_paged_decode_meta(
            attn_metadata=attn_metadata,
            token_num_per_seq=token_num_per_seq,
            state_slot_mapping_cpu=state_slot_mapping_cpu,
            scheduled_bs=scheduled_bs,
            total_tokens=total_tokens,
            padded_total_tokens=padded_total_tokens,
            buf_prefix_ubatch=buf_prefix_ubatch,
        )

    def _attach_v4_paged_decode_meta(
        self,
        attn_metadata,
        token_num_per_seq,
        state_slot_mapping_cpu,
        scheduled_bs: int,
        total_tokens: int,
        padded_total_tokens: Optional[int] = None,
        buf_prefix_ubatch: str = "",
    ) -> None:
        """Phase B: build per-fwd paged-decode index buffers (layer-invariant).

        All three per-token regions are RAGGED-PACKED — same layout family as
        the prefill path (`_build_paged_prefill_meta`). Per-token slot count:
          SWA: actual_swa_count = min(positions[t]+1, win)
          CSA: actual_swa_count + min(n_committed_csa, index_topk)
          HCA: actual_swa_count + n_committed_hca

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
          - n_committed_csa_per_seq : per-seq `ctx_len // 4`
          - var["positions"] : global token positions (already H2D-copied by
                               the caller; consumed by the index-write kernel
                               + CPU-side actual_swa_count cumsum here)

        Skipped when state is not DECODE. The Phase-B fields
        (kv_indices_*, kv_indptr_*, swa_pages) stay at their dataclass
        defaults for prefill batches; downstream V4Attention.forward branches
        on state and reads prefill-mode buffers (kv_indices_prefix_*) instead.
        """
        if scheduled_bs == 0 or total_tokens == 0:
            return  # fields stay at dataclass defaults

        if attn_metadata.state is not AttnState.DECODE:
            return  # prefill: only kv_indices_prefix_* are built downstream

        if len(state_slot_mapping_cpu) < scheduled_bs:
            # Defensive carve-out: caller asserted DECODE but
            # state_slot_mapping is incomplete. Flip state to PREFILL_NATIVE.
            attn_metadata.state = AttnState.PREFILL_NATIVE
            return

        var = self.model_runner.forward_vars
        win = self.window_size  # per-token max SWA prefix slots
        cs = self.win_with_spec  # SWA region per-slot stride (W + mtp_k)
        # swa_pages = num_slots * cs, layer-invariant; matches the boundary
        # between SWA and compress regions in unified_kv (Phase A).
        swa_pages = self.model_runner.max_per_req_cache_slots * cs

        T = total_tokens

        # ----- Per-seq scalars (CPU numpy) -----
        # The single per-token mapping. Built once in `_attach_v4_per_fwd_meta`
        # — both the GPU staging tensor and the unpadded CPU mirror — so we
        # just borrow both here. int32 (numpy fancy-index source dtype is
        # irrelevant; consumers below produce int32 outputs).
        batch_id_per_token_np = attn_metadata.batch_id_per_token_cpu  # [T] int32
        batch_id_per_token_gpu = attn_metadata.batch_id_per_token

        # Read pre-computed `ctx // {4,128}` from attn_metadata — populated by
        # `_attach_v4_per_fwd_meta` (always runs first). int32.
        n_committed_csa_per_seq = attn_metadata.n_committed_csa_per_seq_cpu
        n_committed_hca_per_seq = attn_metadata.n_committed_hca_per_seq_cpu

        # ----- 3 indptr cumsums (CPU numpy, ragged) -----
        # Per-token kv_len = actual_swa_count + n_compress. CSA length now
        # matches Indexer's per-row visibility exactly (= csa_translate_pack
        # kernel's per-token valid_k formula), so buffer reserves only the
        # cells the kernel actually writes — no `-1` sentinel pre-fill, no
        # over-allocation for tokens with per-row visibility < seq-level
        # n_csa (which happens for early tokens in chunked-prefill verify
        # batches and MTP draft mid-iters).
        index_topk = self.index_topk
        positions_np_view = var[f"{buf_prefix_ubatch}positions"].np[:T]
        n_committed_hca_per_token = n_committed_hca_per_seq[batch_id_per_token_np]

        # actual_swa_count[t] = min(positions[t]+1, win). Matches the kernel's
        # inline `n = tl.minimum(pos+1, win)` so SWA-prefix segment sizes line
        # up perfectly. `var["positions"]` is the int64 CpuGpuBuffer populated
        # + H2D-copied by the caller (prepare_decode / build_for_cudagraph_capture).
        actual_swa_count_np = np.minimum(positions_np_view + 1, win).astype(np.int32)
        # csa_valid_k_per_token = min((pos+1)//4, n_committed_csa[bid], index_topk)
        # — mirrors `_attach_v4_indexer_meta`'s `visible_end_gpu` and the
        # `csa_translate_pack` kernel's inline computation, so buffer size ↔
        # kernel-writes match exactly.
        csa_valid_k_per_token = np.minimum(
            np.minimum(
                (positions_np_view + 1) // 4,
                n_committed_csa_per_seq[batch_id_per_token_np],
            ),
            index_topk,
        ).astype(np.int32)

        # CG-padding-aware T_for_indptr: indptr buffer must size to the
        # captured kernel grid (= padded_total_tokens) so padded slots see
        # `kv_len = indptr[t+1] - indptr[t] = 0` and the inner loop bails.
        T_pad = (
            total_tokens if padded_total_tokens is None else int(padded_total_tokens)
        )
        if T_pad < T:
            T_pad = T

        # All three indptr cumsums output int32 directly. Values are bounded
        # (T ≤ mnbt=8192, per-tok ≤ win + index_topk ≈ 2200 → max cumsum ~18M,
        # well within int32).
        # SWA: ragged, per-token len = actual_swa_count[t].
        swa_indptr_np = np.zeros(T_pad + 1, dtype=np.int32)
        swa_indptr_np[1 : T + 1] = np.cumsum(actual_swa_count_np, dtype=np.int32)
        if T_pad > T:
            swa_indptr_np[T + 1 :].fill(int(swa_indptr_np[T]))
        # CSA: ragged, per-token len = actual_swa_count + csa_valid_k_per_token
        csa_per_tok = actual_swa_count_np + csa_valid_k_per_token
        csa_indptr_np = np.zeros(T_pad + 1, dtype=np.int32)
        csa_indptr_np[1 : T + 1] = np.cumsum(csa_per_tok, dtype=np.int32)
        if T_pad > T:
            csa_indptr_np[T + 1 :].fill(int(csa_indptr_np[T]))
        # HCA: ragged, per-token len = actual_swa_count + n_committed_hca
        hca_per_tok = actual_swa_count_np + n_committed_hca_per_token
        hca_indptr_np = np.zeros(T_pad + 1, dtype=np.int32)
        hca_indptr_np[1 : T + 1] = np.cumsum(hca_per_tok, dtype=np.int32)
        if T_pad > T:
            hca_indptr_np[T + 1 :].fill(int(hca_indptr_np[T]))

        swa_indptr_gpu = self._stage(
            f"{buf_prefix_ubatch}v4_kv_indptr_swa", swa_indptr_np
        )
        csa_indptr_gpu = self._stage(
            f"{buf_prefix_ubatch}v4_kv_indptr_csa", csa_indptr_np
        )
        hca_indptr_gpu = self._stage(
            f"{buf_prefix_ubatch}v4_kv_indptr_hca", hca_indptr_np
        )
        # batch_id_per_token + n_committed_csa_per_seq already staged in
        # `_attach_v4_per_fwd_meta`.

        # ----- HCA compress paged offsets (CPU numpy, vectorized) -----
        block_tables_np_full = var[f"{buf_prefix_ubatch}block_tables"].np[:scheduled_bs]
        hca_total_indices = int(hca_indptr_np[T])
        hca_indices_np = np.full(hca_total_indices, -1, dtype=np.int32)
        # n_committed_hca_per_seq is int32; gather stays int32.
        n_h_per_token = n_committed_hca_per_seq[batch_id_per_token_np[:T]]
        total_hca_entries = int(n_h_per_token.sum())
        if total_hca_entries > 0:
            token_indices = np.repeat(np.arange(T, dtype=np.int32), n_h_per_token)
            cu_n_h = np.zeros(T + 1, dtype=np.int32)
            np.cumsum(n_h_per_token, out=cu_n_h[1:], dtype=np.int32)
            entry_offsets = np.arange(total_hca_entries, dtype=np.int32) - np.repeat(
                cu_n_h[:T], n_h_per_token
            )
            # HCA compress section occupies the slice HEAD (offset 0); the SWA
            # prefix segment sits at the tail, written below by
            # write_v4_paged_decode_indices.
            write_pos = hca_indptr_np[token_indices] + entry_offsets
            bid_expanded = batch_id_per_token_np[token_indices]
            hca_indices_np[write_pos] = (
                swa_pages + block_tables_np_full[bid_expanded, entry_offsets]
            ).astype(np.int32)
        # Stage to GPU (HCA compress section at head; SWA prefix scattered below).
        hca_indices_gpu = self._stage(
            f"{buf_prefix_ubatch}v4_kv_indices_hca", hca_indices_np
        )

        # ----- Write SWA / CSA / HCA window-prefix paged offsets (1 kernel) -----
        # Kernel computes `n = min(positions[t]+1, win)` and ring-index
        # `(positions[t] - n + 1 + i) % cs` inline — no window_topk staging.
        # See `write_v4_paged_decode_indices` docstring and plan
        # `sequential-noodling-turing.md` for the motivation. Reads only
        # persistent forward_vars buffers — no allocator churn (the prior
        # `index_copy_` chain raced under MTP-3 long-prefill; this kernel
        # also fixes that, see skill `debug-agent-locate-kernel`).
        swa_indices_gpu = var[f"{buf_prefix_ubatch}v4_kv_indices_swa"].gpu
        csa_indices_gpu = var[f"{buf_prefix_ubatch}v4_kv_indices_csa"].gpu
        write_v4_paged_decode_indices(
            state_slot_per_seq=attn_metadata.state_slot_mapping,
            batch_id_per_token=batch_id_per_token_gpu,
            positions=var[f"{buf_prefix_ubatch}positions"].gpu,
            swa_indptr=swa_indptr_gpu,
            csa_indptr=csa_indptr_gpu,
            hca_indptr=hca_indptr_gpu,
            swa_indices=swa_indices_gpu,
            csa_indices=csa_indices_gpu,
            hca_indices=hca_indices_gpu,
            T=T,
            win=win,
            cs=cs,
        )

        # `skip_prefix_len_csa` is no longer materialized on the decode path —
        # `csa_translate_pack` is invoked with `window_size = self.window_size`
        # so the kernel derives `skip = min(positions[t]+1, win)` inline,
        # which is identical to the value we used to write here
        # (`actual_swa_count_np`). Saves a CPU write + H2D per fwd. The
        # `v4_skip_prefix_len_csa` forward_var is retained for the (unrelated)
        # prefill path where skip depends on `chunk_start` and cannot be
        # derived from positions alone.

        # ----- Stash on attn_metadata for V4Attention.forward consumption -----
        # batch_id_per_token + n_committed_csa_per_seq already set in
        # `_attach_v4_per_fwd_meta` (single source of truth, also consumed by
        # swa_write / indexer outside the is_pure_decode branch).
        # is_pure_decode was set by the caller at AttentionMetaData_DSV4
        # construction time; we only flip it (True→False) above when the
        # warmup carve-out fires (incomplete state_slot_mapping_cpu).
        attn_metadata.kv_indices_swa = swa_indices_gpu[: int(swa_indptr_np[T])]
        attn_metadata.kv_indices_csa = csa_indices_gpu[: int(csa_indptr_np[T])]
        attn_metadata.kv_indices_hca = hca_indices_gpu  # already exact len
        attn_metadata.kv_indptr_swa = swa_indptr_gpu
        attn_metadata.kv_indptr_csa = csa_indptr_gpu
        attn_metadata.kv_indptr_hca = hca_indptr_gpu
        attn_metadata.swa_pages = swa_pages

    def _build_paged_prefill_meta(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        positions_np: np.ndarray,
        cu_seqlens_q_np: np.ndarray,
        token_num_per_seq: np.ndarray,
        start_pos_per_seq_np: np.ndarray,
        state_slot_mapping_cpu: np.ndarray,
        scheduled_bs: int,
        total_tokens: int,
        *,
        positions_gpu: Optional[torch.Tensor] = None,
        cu_q_per_seq_gpu: Optional[torch.Tensor] = None,
        block_tables_gpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Build per-fwd index buffers consumed by sparse_attn_v4_paged_prefill.

        Two-source layout:
          - prefix region (per-ratio): SWA history from prior chunks +
            CSA topk OR HCA all-committed from `unified_kv`. Three buffers
            (Dense / CSA / HCA) per fwd.
          - extend region (shared): in-chunk SWA tail from per-fwd `kv`
            tensor. One buffer.

        Per-token length formulas:
          extend_count[t]      = min(token_pos_in_chunk[t] + 1, win)
          prefix_swa_count[t]  = max(0, chunk_start[bid] - max(0, p_global - win + 1))
          prefix_swa_count[t] + extend_count[t] = min(p_global + 1, win)

        Per-ratio prefix kv_len:
          Dense:  prefix_swa_count[t]
          CSA:    prefix_swa_count[t] + min(n_committed_csa[bid], index_topk)
          HCA:    prefix_swa_count[t] + n_committed_hca[bid]

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
          - swa_pages
        """
        assert scheduled_bs >= 1 and total_tokens >= 1, (
            "scheduled_bs and total_tokens must be positive for prefill meta "
            "build (got scheduled_bs={scheduled_bs}, total_tokens={total_tokens})"
        )

        device = self.device
        win = self.window_size  # per-token topk count
        cs = self.win_with_spec  # SWA region per-slot stride (W + mtp_k)
        index_topk = self.index_topk
        T = total_tokens
        # warmup_model runs BEFORE allocate_kv_cache binds the paged pool
        # (max_per_req_cache_slots not set yet, unified_kv is a 1-page
        # placeholder). V4Attention.forward detects `is_dummy_run` and
        # short-circuits the sparse_attn dispatch entirely, so we don't need
        # valid prefix/extend indices during warmup.
        num_slots = getattr(self.model_runner, "max_per_req_cache_slots", 0)
        if num_slots == 0:
            return
        swa_pages = num_slots * cs
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
        n_committed_csa_per_seq_np = attn_metadata.n_committed_csa_per_seq_cpu
        n_committed_hca_per_seq_np = attn_metadata.n_committed_hca_per_seq_cpu
        # Per-token CSA valid_k = Indexer's per-row visibility, matching
        # `_attach_v4_indexer_meta`'s `visible_end_gpu` formula and the
        # `csa_translate_pack` kernel's inline computation. Buffer size ↔
        # kernel-writes match exactly, so no `-1` sentinel pre-fill is needed.
        csa_valid_k_per_token_np = np.minimum(
            np.minimum(
                (positions_arr + 1) // 4,
                n_committed_csa_per_seq_np[batch_id_per_token_np],
            ),
            index_topk,
        ).astype(np.int32)
        # Per-token CAUSAL HCA visibility (mirrors CSA above and the reference
        # `get_compress_topk_idxs` prefill mask): token at `pos` sees only the
        # `(pos+1)//128` HCA groups committed up to its own position, capped by
        # the per-seq committed count. Without `(pos+1)//128`, every token used
        # the per-seq `ctx_end//128`, over-reading FUTURE groups and making a
        # token's output depend on the forward's total length (chunked breaks).
        # MUST stay in sync with the kernel's inline cap in
        # `_v4_paged_prefill_indices_kernel` (HCA_RATIO).
        n_hca_per_token_np = np.minimum(
            (positions_arr + 1) // 128,
            n_committed_hca_per_seq_np[batch_id_per_token_np],
        ).astype(np.int32)

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
        # All non-blocking; bounded by `total ≤ T*max_per_tok`.
        chunk_start_per_seq_gpu = torch.from_numpy(chunk_start_per_seq_np).to(
            device, non_blocking=True
        )
        n_committed_hca_per_seq_gpu = torch.from_numpy(
            np.asarray(n_committed_hca_per_seq_np[:scheduled_bs], dtype=np.int32)
        ).to(device, non_blocking=True)
        ext_indptr = torch.from_numpy(ext_indptr_np).to(device, non_blocking=True)
        swa_indptr = torch.from_numpy(swa_indptr_np).to(device, non_blocking=True)
        csa_indptr = torch.from_numpy(csa_indptr_np).to(device, non_blocking=True)
        hca_indptr = torch.from_numpy(hca_indptr_np).to(device, non_blocking=True)

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
        state_slot_per_seq_gpu = attn_metadata.state_slot_mapping[:scheduled_bs]
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
            n_committed_hca_per_seq=n_committed_hca_per_seq_gpu,
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
            cs=cs,
            swa_pages=swa_pages,
        )

        # ----- skip_prefix_len_csa: per-token SWA prefix length -----
        # csa_translate_pack consumes this to derive the CSA topk length
        # `valid_k = (indptr[t+1]-indptr[t]) - skip` it writes at the HEAD of
        # `kv_indices_prefix_csa[indptr[t]:indptr[t+1]]`; the SWA prefix
        # (length `skip`) occupies the slice TAIL, written by the builder.
        # Matches the per-token prefix_swa_count vector we just computed on CPU.
        skip_csa_gpu = torch.from_numpy(prefix_swa_count_np).to(
            device, non_blocking=True
        )

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
        attn_metadata.swa_pages = swa_pages

    def _build_compress_plans(
        self,
        extend_lens_np,
        context_lens_np,
        *,
        for_decode_cg: bool,
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

        `for_decode_cg`: True for decode runtime AND decode CG capture —
        the returned plan_gpu is sliced to a fixed `_decode_compress_cap`
        per ratio so capture/replay shapes match. False for eager prefill —
        the plan_gpu is sliced to the actual `n_compress` (smallest grid).
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
            decode_capacity_per_ratio=(
                self._decode_compress_cap if for_decode_cg else None
            ),
        )

    def _populate_block_tables(
        self, batch: ScheduledBatch, scheduled_bs: int
    ) -> torch.Tensor:
        """Populate `forward_vars["block_tables"]` from the batch and return
        the GPU view sliced to `scheduled_bs` rows.

        Mirrors `CommonAttentionBuilder.prepare_block_tables` but is invoked
        unconditionally (parent only calls it when has_cached).
        """
        var = self.model_runner.forward_vars
        block_tables_np = var["block_tables"].np
        for i, block_table in enumerate(batch.block_tables[:scheduled_bs]):
            block_tables_np[i] = 0
            block_tables_np[i, : len(block_table)] = block_table
        return var["block_tables"].copy_to_gpu(scheduled_bs)

    def _populate_state_slot_mapping(
        self, batch: ScheduledBatch, scheduled_bs: int, return_cpu: bool = False
    ):
        """Build `[scheduled_bs]` int32 tensor of per-request state-cache slots.

        With slots_per_req() == 1, slot index == per_req_cache_group. This
        is what V4 forward uses to index `swa_kv` and `Compressor.kv_state`
        (the per-request state pool, distinct from the per-token paged-KV
        `slot_mapping`).

        When `return_cpu=True`, returns `(gpu_tensor, cpu_numpy)`. The CPU
        copy is consumed by the V4 forward path to avoid `.tolist()` syncs
        (PR-A Phase 2).
        """
        groups_np = np.asarray(
            batch.per_req_cache_groups[:scheduled_bs], dtype=np.int32
        )
        # Warmup / dummy_run batches don't allocate per_req_cache slots
        # (per_req_cache_groups is empty). Fall back to slot 0 for all seqs
        # so V4 forward can take the normal path uniformly — slot 0's state
        # cache is reset on the first real prefill (start_pos==0 path masks
        # state reads, fresh writes overwrite warmup pollution).
        if len(groups_np) < scheduled_bs:
            groups_np = np.zeros(scheduled_bs, dtype=np.int32)
        gpu = self._stage("v4_meta_state_slot_groups", groups_np)
        if return_cpu:
            return gpu, groups_np
        return gpu

    def build_for_cudagraph_capture(
        self, bs: int
    ) -> tuple[AttentionMetaData_DSV4, Context]:
        """Build attn_metadata for CUDAGraph capture using a synthetic decode batch.

        Synthesizes bs sequences each at start_pos=window_size (so SWA window
        is full + 1 CSA committed entry — exercises the production decode
        codepath: state-cache reads, sparse_attn gather, indexer fp8 logits).

        Per-fwd metadata is populated through the SAME helpers prepare_decode
        uses (`_attach_v4_indexer_meta`, `_attach_v4_per_fwd_meta`,
        `_build_compress_plans`), so all GPU views point to the pre-allocated
        buffers in `forward_vars`. Replay-time prepare_decode writes into the
        SAME buffers — captured graph reads stable addresses.

        NOTE on dynamic-shape kernels (`update_compressor_states` / `swa_write`):
        these currently use variable kernel grids (`grid=(num_compress,)`),
        which CUDAGraph capture rejects. A follow-up PR converts them to fixed
        grid + sentinel masking. Until then, capture itself can succeed (the
        helpers run on CPU + small H2D), but model.forward inside torch.cuda.graph
        will likely fail at the first such kernel launch — the user can detect
        this via capture log output. (`fused_compress_attn` is already
        CG-safe: launches at the decode-tight slice
        (`_decode_compress_cap[ratio]`, baked at capture) and
        sentinel-skips inactive rows internally for both BF16 Main and FP8
        Indexer paths.)
        """
        var = self.model_runner.forward_vars
        # Honor MTP at capture time: V4-Pro `mtp_k=1` → 2 tokens/req. The
        # outer `model_runner.capture_cudagraph` populates cu_seqlens_q with
        # the same layout, so capture and replay see identical shapes.
        max_q_len = 1 + self.max_spec_steps
        total_tokens = bs * max_q_len
        win = self.window_size

        # Synthetic state: each seq has already produced `win` tokens; this
        # fwd is `max_q_len` decode/draft steps at positions
        # [win, win+max_q_len). Hits is_pure_decode (start_pos > 0, uniform
        # tok-per-seq), exercising Phase B/C/E paths during capture.
        start_pos = win
        positions_np = (np.arange(total_tokens, dtype=np.int64) % max_q_len) + start_pos
        cu_seqlens_q_np = np.arange(0, bs + 1, dtype=np.int32) * max_q_len
        context_lens_np = np.full(bs, start_pos + max_q_len, dtype=np.int32)
        # Slot mapping: use real per-req cache slots [0..bs-1].
        state_slot_np = np.arange(bs, dtype=np.int32)
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
        state_slot_gpu = self._stage("v4_meta_state_slot_groups", state_slot_np)

        # Synthetic decode batch: start_pos = win > 0 and uniform
        # max_q_len tokens per seq, so is_pure_decode is True by
        # construction (capture replays the decode codepath).
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
        attn_metadata.state_slot_mapping = state_slot_gpu
        attn_metadata.state_slot_mapping_cpu = state_slot_np

        # Build compress_plans + per-fwd meta + indexer meta via the same
        # helpers used at runtime — guarantees addresses match.
        extend_lens_np = np.full(bs, max_q_len, dtype=np.int32)
        attn_metadata.compress_plans = self._build_compress_plans(
            extend_lens_np, context_lens_np, for_decode_cg=True
        )
        # Capture: padded_bs == scheduled_bs == bs (synthetic batch is full).
        # Must run BEFORE `_attach_v4_indexer_meta` so the indexer-side meta
        # builder can reuse the shared per-fwd GPU tensors.
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            extend_lens_np,  # = np.full(bs, max_q_len) — synthetic uniform decode batch
            attn_metadata.state_slot_mapping_cpu,
            bs,
            total_tokens,
            padded_bs=bs,
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
                bs=bs,
                max_seqlen_q=max_q_len,
                context_lens_np=context_lens_np,
                state_slot_np=state_slot_np,
                positions_np=positions_np.astype(np.int32),
            )

        context = Context(
            positions=positions,
            is_prefill=False,
            batch_size=bs,
            graph_bs=bs,
        )
        return attn_metadata, context

    # ------------------------------------------------------------------ #
    # Helpers.                                                           #
    # ------------------------------------------------------------------ #

    def _alloc_v4_metadata_buffers(self) -> None:
        """Pre-allocate every CpuGpuBuffer the V4 metadata builder writes into.

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

        # _attach_v4_per_fwd_meta + _populate_state_slot_mapping.
        # state_slot is staged ONCE into v4_meta_state_slot_groups (set by
        # `_populate_state_slot_mapping`); attn_metadata.state_slot_mapping
        # exposes that GPU view to all downstream consumers (no second
        # H2D-staged copy).
        bufs["v4_meta_state_slot_groups"] = CpuGpuBuffer(bs, **i32)

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
        bufs["v4_kv_indices_swa"] = CpuGpuBuffer(T_dec * win, **i32)
        bufs["v4_kv_indices_csa"] = CpuGpuBuffer(T_dec * (win + self.index_topk), **i32)
        bufs["v4_kv_indices_hca"] = CpuGpuBuffer(
            T_dec * (win + self.max_committed_hca), **i32
        )
        bufs["v4_kv_indptr_swa"] = CpuGpuBuffer(T_dec + 1, **i32)
        bufs["v4_kv_indptr_csa"] = CpuGpuBuffer(T_dec + 1, **i32)
        bufs["v4_kv_indptr_hca"] = CpuGpuBuffer(T_dec + 1, **i32)
        # Per-seq `ctx_len // 4` (raw, no clamp). Consumed by csa_translate_pack
        # (kernel masks `(k < n_committed) & (k < index_topk)`) AND by the
        # indexer (cast to int64 inline). Built unconditionally in
        # `_attach_v4_per_fwd_meta`.
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
        # NOTE: decode-path `logits` ([T, max_model_len_idx] fp32) and
        # `topk_indices` ([T, index_topk] int32) are NOT pre-allocated —
        # they are write-once GPU scratch with no CPU mirror, allocated
        # per-fwd inside `Indexer._score_topk_decode` via `torch.empty`.
        # Under CUDAGraph capture they land in the graph's private pool
        # and replay reuses the same address; eager keeps the standard
        # caching-allocator fast path.
        # Per-token write offset consumed by `csa_translate_pack` (decode path
        # fills with `window_size`; prefill path will fill with per-token
        # prior_swa_count once the new dual-source kernel lands). Allocated
        # `mnbt` (worst-case prefill) so prefill writes don't overflow.
        bufs["v4_skip_prefix_len_csa"] = CpuGpuBuffer(mnbt, **i32)

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
        # `bs * ceil((1 + max_spec_steps) / ratio)` rows (vs `mnbt // ratio
        # + bs` for prefill, which is ~13× larger at typical config). We
        # still allocate the full prefill capacity (eager prefill needs it),
        # but both decode capture and replay slice down to
        # `_decode_compress_cap` so the captured grid is the decode-tight
        # bound. capture and replay MUST use the same value (CG kernel call
        # args are baked).
        self._decode_compress_cap: dict[int, int] = {}
        for ratio, is_overlap in self._unique_compress_ratios_overlap:
            # NOTE: this is the pool-window size (algorithm constant), NOT the
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
            # Decode-tight bound: each seq's qlen-token window contains at
            # most ceil(qlen / ratio) ratio-aligned boundaries (qlen =
            # 1 + max_spec_steps), so total ≤ bs * ceil(qlen / ratio).
            # The integer expression `(max_spec_steps + ratio) // ratio`
            # is ceil((1 + max_spec_steps) / ratio).
            #
            # Tighter than the old `max_decode_tokens // ratio + max_bs`:
            #   CSA r=4 MTP3 bs=256:  256 vs 512 (2× smaller)
            #   HCA r=128 MTP3 bs=256: 256 vs 264 (8 smaller)
            #
            # Smaller plan slice → fewer sentinel rows the kernel skips →
            # smaller grid → less launch + scheduling overhead.
            per_seq_max = (self.max_spec_steps + ratio) // ratio
            self._decode_compress_cap[ratio] = bs * per_seq_max

        if getattr(self.model_runner.config, "enable_tbo_decode", False):
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
        max_blocks = self.max_num_blocks_per_seq // self.block_ratio

        for ub_idx in range(self._NUM_TBO_UBATCHES):
            p = f"ub{ub_idx}_"
            # Global per-fwd decode inputs (live in model_runner.forward_vars
            # for the non-TBO path; cloned here so each ubatch slices its own).
            bufs[f"{p}positions"] = CpuGpuBuffer(T_dec, **i64)
            bufs[f"{p}context_lens"] = CpuGpuBuffer(bs, **i32)
            bufs[f"{p}block_tables"] = CpuGpuBuffer(bs, max_blocks, **i32)
            bufs[f"{p}cu_seqlens_q"] = CpuGpuBuffer(bs + 1, **i32)

            # V4 decode metadata buffers.
            bufs[f"{p}v4_meta_state_slot_groups"] = CpuGpuBuffer(bs, **i32)
            bufs[f"{p}v4_kv_indices_swa"] = CpuGpuBuffer(T_dec * win, **i32)
            bufs[f"{p}v4_kv_indices_csa"] = CpuGpuBuffer(
                T_dec * (win + self.index_topk), **i32
            )
            bufs[f"{p}v4_kv_indices_hca"] = CpuGpuBuffer(
                T_dec * (win + self.max_committed_hca), **i32
            )
            bufs[f"{p}v4_kv_indptr_swa"] = CpuGpuBuffer(T_dec + 1, **i32)
            bufs[f"{p}v4_kv_indptr_csa"] = CpuGpuBuffer(T_dec + 1, **i32)
            bufs[f"{p}v4_kv_indptr_hca"] = CpuGpuBuffer(T_dec + 1, **i32)
            bufs[f"{p}v4_n_committed_csa_per_seq"] = CpuGpuBuffer(bs, **i32)
            bufs[f"{p}v4_batch_id_per_token"] = CpuGpuBuffer(mnbt, **i64)
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

    def _stage(self, name: str, arr) -> torch.Tensor:
        """Write numpy `arr` into `forward_vars[name]` (CpuGpuBuffer) and
        return its GPU view sliced to len(arr). Auto-casts dtype to match
        the buffer (e.g. int64 → int32). Asserts the buffer is large enough.
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
        buf.np[:n] = arr
        return buf.copy_to_gpu(n)

    @staticmethod
    def _numel(shape: tuple) -> int:
        n = 1
        for s in shape:
            n *= s
        return n

    @staticmethod
    def _make_gather_slot(
        buf: torch.Tensor,
        stride: int,
        state_tensors: list[torch.Tensor],
    ):
        """Return a callable that copies compute tensors → staging buffer for one slot."""
        offsets_and_sizes = []
        off = 0
        for t in state_tensors:
            n_layers = t.shape[0]
            per_layer = t[0, 0].numel()
            total = n_layers * per_layer
            offsets_and_sizes.append((off, n_layers, per_layer))
            off += total
        assert off == stride

        def gather_slot(compute_slot: int, pool_idx: int) -> None:
            dst_start = pool_idx * stride
            for t, (off, n_layers, per_layer) in zip(state_tensors, offsets_and_sizes):
                buf[dst_start + off : dst_start + off + n_layers * per_layer] = t[
                    :, compute_slot
                ].reshape(-1)

        return gather_slot

    @staticmethod
    def _make_scatter_slot(
        buf: torch.Tensor,
        stride: int,
        state_tensors: list[torch.Tensor],
    ):
        """Return a callable that copies staging buffer → compute tensors for one slot."""
        offsets_and_sizes = []
        off = 0
        for t in state_tensors:
            n_layers = t.shape[0]
            per_layer = t[0, 0].numel()
            total = n_layers * per_layer
            offsets_and_sizes.append((off, n_layers, per_layer))
            off += total
        assert off == stride

        def scatter_slot(compute_slot: int, pool_idx: int) -> None:
            src_start = pool_idx * stride
            for t, (off, n_layers, per_layer) in zip(state_tensors, offsets_and_sizes):
                chunk = buf[src_start + off : src_start + off + n_layers * per_layer]
                t[:, compute_slot] = chunk.view(t[:, compute_slot].shape)

        return scatter_slot

    def _zero_state(self, shape: tuple, device) -> torch.Tensor:
        return torch.zeros(shape, dtype=self._state_dtype, device=device)

    def _neg_inf_state(self, shape: tuple, device) -> torch.Tensor:
        return torch.full(shape, float("-inf"), dtype=self._state_dtype, device=device)
