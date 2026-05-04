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

from typing import Type

import numpy as np
import torch
from aiter import dtypes
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attentions.backends import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionBuilder,
)
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import AttentionMetaData, Context

# ---------------------------------------------------------------------------
# Builder-local helpers (private). Used by `_build_v4_pack_meta_for_ratio`
# and `_attach_v4_per_fwd_meta` for ragged-segment index math + per-token
# sliding-window topk index generation. Live here (not in the model file)
# because their only callers are inside this builder.
# ---------------------------------------------------------------------------


def _segment_indices(
    seq_ids: np.ndarray, lens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For ragged segments (one per `seq_ids[i]` of length `lens[i]`), return
    flat (per-row seq id, per-row local position) arrays of total length
    `sum(lens)`.
    """
    total = int(lens.sum())
    if total == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    token_seq_ids = np.repeat(seq_ids.astype(np.int64), lens)
    cum = np.concatenate(([0], np.cumsum(lens.astype(np.int64))[:-1]))
    local_pos = np.arange(total, dtype=np.int64) - np.repeat(cum, lens)
    return token_seq_ids, local_pos


def _build_window_topk_batched(
    positions: torch.Tensor,  # [total_tokens] int (abs token positions)
    start_pos_per_token: torch.Tensor,  # [total_tokens] int (each token's seq start_pos)
    window_size: int,
) -> torch.Tensor:  # [total_tokens, window_size] int32
    """Per-token sliding-window topk indices for the whole batch.

    Three-branch semantics:
      - sp == 0 (fresh prefill): matrix entries = abs positions in the window
        [pos-win+1, pos] clamped to [0, pos]; mask future via -1.
      - 0 < sp < win-1 (prefix mode): all tokens in the seq share a single
        matrix [0..sp, -1, ..., -1] (matches original semantics, including
        MTP-N where the same start_pos is reused).
      - sp >= win-1 (cyclic mode): cyclic ring offsets starting at sp+1 mod win.
    """
    device = positions.device
    total = positions.size(0)
    arange_w = torch.arange(window_size, device=device, dtype=positions.dtype).view(
        1, window_size
    )
    pos_col = positions.view(total, 1)
    sp_col = start_pos_per_token.view(total, 1)
    neg1 = torch.tensor(-1, device=device, dtype=positions.dtype)

    # Case A: sp == 0 (fresh prefill) — abs positions [pos-win+1, pos] clamped.
    case_a = (pos_col - window_size + 1).clamp(min=0) + arange_w
    case_a = torch.where(case_a > pos_col, neg1, case_a)

    # Case B: 0 < sp < win-1 (prefix mode) — shared per-seq matrix.
    case_b = arange_w.expand(total, window_size).clone()
    case_b = torch.where(arange_w > sp_col, neg1, case_b)

    # Case C: sp >= win-1 (cyclic mode) — ring offsets.
    sp_mod = sp_col % window_size
    case_c = (sp_mod + 1 + arange_w) % window_size

    sp_eq_0 = sp_col == 0
    sp_in_prefix = (sp_col > 0) & (sp_col < window_size - 1)

    out = case_c
    out = torch.where(sp_in_prefix.expand_as(out), case_b, out)
    out = torch.where(sp_eq_0.expand_as(out), case_a, out)
    return out.to(torch.int32)


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
        # `deepgemm_fp8_paged_mqa_logits` decode-path output column count
        # = max compressed K positions per seq. CSA ratio=4 is the
        # max-density ratio (1 indexer slot per 4 source tokens).
        self.max_model_len_idx = model_runner.config.max_model_len // 4

        # Compressor state shape: [coff * ratio, coff * head_dim], fp32.
        # CSA: ratio=4, overlap=True -> coff=2 -> [8, 2*head_dim]
        # HCA: ratio=128, overlap=False -> coff=1 -> [128, head_dim]
        self.csa_main_state_shape = (2 * 4, 2 * self.head_dim)
        self.csa_idx_state_shape = (2 * 4, 2 * self.index_head_dim)
        self.hca_main_state_shape = (128, self.head_dim)

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

        # Sparse-attn + per-fwd metadata buffers (CG-A: pre-allocate for fixed
        # GPU pointers, prerequisite for CUDAGraph capture). All H2D copies in
        # the V4 metadata builder go through these buffers via the
        # `np[:n] = arr; copy_to_gpu(n)` pattern instead of per-call
        # `torch.as_tensor(arr)` allocations.
        self._alloc_v4_metadata_buffers()

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
        # SWA window per layer.
        swa_per_layer = self.window_size * self.head_dim * elem_swa
        return (
            len(self.csa_layers) * (csa_main + csa_idx)
            + len(self.hca_layers) * hca_main
            + self.num_layers * swa_per_layer
        )

    def slots_per_req(self) -> int:
        # No spec decoding lookahead in V4 (pre-PR3-main).
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
        """Allocate the three classical KV pools.

        Pool layout: `[num_blocks, n_layers_of_type, k_per_block, head_dim]`.
        Per-layer view (`pool[:, layer_pos]` shape `[num_blocks, k, head_dim]`)
        is what `Compressor.kv_cache` / `Indexer.kv_cache` get bound to in
        `build_kv_cache_tensor`. Compressor.forward then writes individual
        compressed entries via `pool[block_id, slot_in_block, :] = entry`.

        Returns a dict; ModelRunner setattr's each as `runner.<name>`.
        """
        runner = self.model_runner
        device = runner.device
        num_blocks = runner.num_physical_kvcache_blocks
        n_csa = len(self.csa_layers)
        n_hca = len(self.hca_layers)
        return {
            "v4_csa_main_kv": torch.zeros(
                (num_blocks, n_csa, self.k1_csa, self.head_dim),
                dtype=self._classical_dtype,
                device=device,
            ),
            # CSA Indexer cache: FP8 + 4-byte fp32 scale per row, packed as
            # FP8 (1 byte per element) and aligned to 16 bytes per row to
            # match V3.2 sparse MLA layout (avoid unaligned access).
            # Written by `indexer_k_quant_and_cache` (V4 inner Compressor),
            # read by `cp_gather_indexer_k_quant_cache` (Indexer.forward_batched).
            #
            # Layer-major axis order `[n_csa, NB, k1, aligned_dim]` so each
            # per-CSA slice `pool[pos]` is contiguous in storage and can be
            # `.view`-flattened to `[NB*k1, 1, aligned_dim]` (the layout aiter
            # kernels expect: a flat slot index in [0, NB*k1) selects the row).
            "v4_csa_idx_kv": torch.zeros(
                (n_csa, num_blocks, self.k1_csa, self._aligned_index_dim),
                dtype=dtypes.fp8,
                device=device,
            ),
            "v4_hca_main_kv": torch.zeros(
                (num_blocks, n_hca, self.k2_hca, self.head_dim),
                dtype=self._classical_dtype,
                device=device,
            ),
        }

    def allocate_per_req_cache(self, num_slots: int) -> dict[str, torch.Tensor]:
        """Allocate the state-cache pool.

        Tensors are setattr'd onto ModelRunner so model layers can access them
        as `model_runner.<name>`. `build_kv_cache_tensor` then slices per layer
        and binds the corresponding module attribute (Compressor.kv_state,
        Compressor.score_state, Attention.swa_kv) in place.
        """
        device = self.model_runner.device
        n_csa = len(self.csa_layers)
        n_hca = len(self.hca_layers)
        return {
            # SWA window — every layer (Dense / CSA / HCA all have SWA branch).
            "v4_swa_kv": torch.zeros(
                (self.num_layers, num_slots, self.window_size, self.head_dim),
                dtype=self._swa_dtype,
                device=device,
            ),
            # CSA Main Compressor.
            "v4_csa_main_kv_state": self._zero_state(
                (n_csa, num_slots, *self.csa_main_state_shape), device
            ),
            "v4_csa_main_score_state": self._neg_inf_state(
                (n_csa, num_slots, *self.csa_main_state_shape), device
            ),
            # CSA Indexer's inner Compressor.
            "v4_csa_idx_kv_state": self._zero_state(
                (n_csa, num_slots, *self.csa_idx_state_shape), device
            ),
            "v4_csa_idx_score_state": self._neg_inf_state(
                (n_csa, num_slots, *self.csa_idx_state_shape), device
            ),
            # HCA Main Compressor.
            "v4_hca_main_kv_state": self._zero_state(
                (n_hca, num_slots, *self.hca_main_state_shape), device
            ),
            "v4_hca_main_score_state": self._neg_inf_state(
                (n_hca, num_slots, *self.hca_main_state_shape), device
            ),
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
        from atom.models.deepseek_v4 import (
            Compressor as _V4Compressor,
            DeepseekV4Attention as _V4Attention,
            Indexer as _V4Indexer,
        )

        runner = self.model_runner

        if isinstance(module, _V4Attention):
            # Attention.swa_kv — every layer.
            module.swa_kv = runner.v4_swa_kv[module.layer_id]
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

            # Per-kind shared compress output buffer (CUDAGraph: stable
            # data pointer + fixed shape across captures of the same kind).
            # Read from forward_vars (allocated in _alloc_v4_metadata_buffers).
            compress_out_buffers = self.model_runner.forward_vars
            if is_indexer_inner:
                assert ratio == 4, "Indexer-inner Compressor only on CSA layers"
                pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_csa_idx_kv_state[pos]
                module.score_state = runner.v4_csa_idx_score_state[pos]
                # Inner compressor writes target the SAME storage as the
                # outer Indexer.kv_cache (csa_idx_kv). Same 3D shape — write
                # via slot_mapping is shape-agnostic (flat indexing), but we
                # keep [NB, k1_csa, aligned_dim] for symmetry with the read
                # binding above.
                module.kv_cache = runner.v4_csa_idx_kv[pos]
                module.compress_out = compress_out_buffers["v4_csa_idx_compress_out"]
            elif ratio == 4:
                pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_csa_main_kv_state[pos]
                module.score_state = runner.v4_csa_main_score_state[pos]
                module.kv_cache = runner.v4_csa_main_kv[:, pos]
                module.compress_out = compress_out_buffers["v4_csa_main_compress_out"]
            elif ratio == 128:
                pos = self.layer_id_to_hca_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_hca_main_kv_state[pos]
                module.score_state = runner.v4_hca_main_score_state[pos]
                module.kv_cache = runner.v4_hca_main_kv[:, pos]
                module.compress_out = compress_out_buffers["v4_hca_main_compress_out"]
            else:
                raise ValueError(
                    f"Unknown V4 compress_ratio={ratio} on Compressor at "
                    f"prefix={module.prefix!r}"
                )
            return None

        return super().build_kv_cache_tensor(layer_id, module)

    # ------------------------------------------------------------------ #
    # CommonAttentionBuilder abstract methods (V4 forward consumes only  #
    # `positions`; other metadata is populated for forward parity with   #
    # the rest of ATOM and to support PR3-main multi-sequence wiring).   #
    # ------------------------------------------------------------------ #

    def _attach_sparse_layout_metadata(
        self,
        attn_metadata: AttentionMetaData,
        cu_seqlens_q_np,
        start_pos_per_seq,
        scheduled_bs: int,
        total_tokens: int,
        positions_np=None,
        positions_gpu=None,
    ) -> None:
        """Precompute per-token ragged sparse-attn layout for each ratio type.

        The actual topk index values are layer-specific (CSA Indexer depends on
        weights), but the per-token topk span and global-KV offset layout only
        depends on the request geometry and compress_ratio.

        For prefill (start_pos == 0), `kv_len` consumes the actual per-seq
        compressed-boundary count from `attn_metadata.compress_plans[ratio]`
        (computed by the SGLang-style plan generator). The plan returns the
        EXACT count `cu_compress_cpu[i+1] - cu_compress_cpu[i]` (= floor
        formula), not the loose `(token_num + ratio - 1) // ratio` ceil
        upper bound — keeping the sparse-attn `kv_offsets` aligned with the
        batched compressor's tightly-packed output.
        """
        import numpy as np

        var = self.model_runner.forward_vars
        layouts = {}
        ratio_specs = {
            0: ("dense", 0),
            4: ("csa", 4),
            128: ("hca", 128),
        }
        compress_plans = getattr(attn_metadata, "compress_plans", None) or {}
        for ratio_key, (kind, ratio) in ratio_specs.items():
            starts = var[f"v4_{kind}_sparse_topk_starts"].np
            lens = var[f"v4_{kind}_sparse_topk_lens"].np
            offsets = var[f"v4_{kind}_sparse_kv_offsets"].np
            topk_base = 0
            kv_base = 0
            token_base = 0
            max_topk = 0
            cu_compress_cpu = None
            if ratio in compress_plans:
                cu_compress_cpu = compress_plans[ratio].cu_compress_cpu
            for seq_idx in range(scheduled_bs):
                seq_start = int(cu_seqlens_q_np[seq_idx])
                seq_end = int(cu_seqlens_q_np[seq_idx + 1])
                token_num = seq_end - seq_start
                if token_num == 0:
                    continue
                start_pos = int(start_pos_per_seq[seq_idx])
                end_pos = start_pos + token_num
                window_topk = (
                    self.window_size
                    if start_pos > 0
                    else min(token_num, self.window_size)
                )
                compress_topk = 0
                kv_len = token_num if start_pos == 0 else self.window_size
                # n_compress: this-fwd's compressed boundary count for this
                # seq. Floor formula matches batched compressor's tightly-
                # packed output length. Falls back to numpy formula when
                # compress_plans is absent (warmup / non-V4 dispatch).
                if cu_compress_cpu is not None:
                    n_compress = int(
                        cu_compress_cpu[seq_idx + 1] - cu_compress_cpu[seq_idx]
                    )
                else:
                    n_compress = end_pos // ratio - start_pos // ratio if ratio else 0
                if ratio == 4:
                    compress_topk = min(self.index_topk, end_pos // ratio)
                    if start_pos == 0:
                        kv_len = token_num + n_compress
                    else:
                        kv_len = self.window_size + end_pos // ratio
                elif ratio == 128:
                    if start_pos > 0:
                        compress_topk = (start_pos + 1) // ratio
                    else:
                        compress_topk = token_num // ratio
                    if start_pos == 0:
                        kv_len = token_num + n_compress
                    else:
                        kv_len = self.window_size + end_pos // ratio
                topk_len = window_topk + compress_topk
                starts[token_base : token_base + token_num] = np.arange(
                    topk_base,
                    topk_base + token_num * topk_len,
                    topk_len,
                    dtype=np.int64,
                )
                lens[token_base : token_base + token_num] = topk_len
                offsets[token_base : token_base + token_num] = kv_base
                token_base += token_num
                topk_base += token_num * topk_len
                kv_base += kv_len
                max_topk = max(max_topk, topk_len)

            topk_starts = var[f"v4_{kind}_sparse_topk_starts"].copy_to_gpu(total_tokens)
            topk_lens = var[f"v4_{kind}_sparse_topk_lens"].copy_to_gpu(total_tokens)
            kv_offsets = var[f"v4_{kind}_sparse_kv_offsets"].copy_to_gpu(total_tokens)
            layouts[ratio_key] = {
                "topk_starts": topk_starts,
                "topk_lens": topk_lens,
                "kv_offsets": kv_offsets,
                "max_topk": max_topk,
            }
            # Build the GPU pack_meta consumed by `_v4_build_sparse_inputs_batched`.
            # Replaces ~14 per-layer torch.as_tensor() H2D copies with a single
            # H2D per index tensor at prepare_*. Returns None during warmup
            # (no positions/slots) — V4Attention.forward then falls back to
            # the inline path.
            pack_meta = self._build_v4_pack_meta_for_ratio(
                kind=kind,
                ratio=ratio,
                has_indexer=(ratio == 4),
                cu_seqlens_q_np=cu_seqlens_q_np,
                start_pos_per_seq_np=np.asarray(start_pos_per_seq, dtype=np.int64),
                slots_np=getattr(attn_metadata, "state_slot_mapping_cpu", None),
                positions_np=positions_np,
                cu_compress_cpu=cu_compress_cpu,
                scheduled_bs=scheduled_bs,
                total_tokens=total_tokens,
                device=self.device,
            )
            if pack_meta is not None:
                layouts[ratio_key]["pack_meta"] = pack_meta
        attn_metadata.v4_sparse_layouts = layouts
        # Indexer (CSA only) per-fwd GPU metadata. Hoists ~6 H2D calls per
        # CSA layer (batch_id_per_token / cu_committed / n_committed /
        # k_per_token / offset_per_token / is_prefill_per_token).
        # compress_plans[4] (CSA, ratio=4) carries the compress rows that the
        # indexer-FP8 path needs to derive its write-side slot_mapping. None
        # for warmup or empty fwd; _build_v4_indexer_meta handles both.
        csa_compress_plan_cpu = None
        plans = getattr(attn_metadata, "compress_plans", None) or {}
        if 4 in plans:
            csa_compress_plan_cpu = plans[4].compress_plan_cpu
        attn_metadata.v4_indexer_meta = self._build_v4_indexer_meta(
            cu_seqlens_q_np=cu_seqlens_q_np,
            start_pos_per_seq_np=np.asarray(start_pos_per_seq, dtype=np.int64),
            positions_gpu=positions_gpu,
            scheduled_bs=scheduled_bs,
            total_tokens=total_tokens,
            device=self.device,
            csa_compress_plan_cpu=csa_compress_plan_cpu,
        )

    def _build_v4_indexer_meta(
        self,
        *,
        cu_seqlens_q_np,
        start_pos_per_seq_np,
        positions_gpu,
        scheduled_bs: int,
        total_tokens: int,
        device,
        csa_compress_plan_cpu,
    ):
        """Build per-fwd GPU index tensors consumed by `Indexer.forward_batched`.

        Returns None for warmup batches (the indexer falls back to its
        inline H2D path) or when CSA / Indexer is not on the model. CSA
        ratio is fixed at 4; we always build under that assumption.

        The FP8-cache write side (`indexer_k_quant_and_cache`) needs a flat
        `compress_slot_mapping` int64 tensor — one entry per row in
        `csa_compress_plan_cpu` mapping the compress entry to a global slot
        in the `[n_csa, NB, k1, aligned_dim=144]` cache pool (layer-major,
        FP8 + 4-byte fp32 scale per row, 16B-aligned). Computed here host-side
        because the plan rows + block_tables_cpu are both already on host.
        """
        from atom.models.deepseek_v4 import _V4_BLOCK_SIZE

        if scheduled_bs == 0 or total_tokens == 0:
            return None

        bs = scheduled_bs
        ratio = 4  # CSA
        k_per_block = _V4_BLOCK_SIZE // ratio  # 32
        cu_seqlens_q_arr = cu_seqlens_q_np[: bs + 1].astype(np.int64)
        token_num_per_seq = (cu_seqlens_q_arr[1:] - cu_seqlens_q_arr[:bs]).astype(
            np.int64
        )
        start_pos_per_seq = start_pos_per_seq_np[:bs].astype(np.int64)
        end_pos_per_seq = start_pos_per_seq + token_num_per_seq
        n_committed_per_seq = end_pos_per_seq // ratio
        k_per_seq_cpu = np.minimum(self.index_topk, n_committed_per_seq).astype(
            np.int64
        )
        cu_committed_cpu = np.concatenate(
            [
                np.zeros(1, dtype=np.int64),
                np.cumsum(n_committed_per_seq, dtype=np.int64),
            ]
        )
        total_committed = int(cu_committed_cpu[-1])

        # FP8 write-side slot_mapping (independent of `total_committed` —
        # written even when there's nothing to read because num_compress can
        # be > 0 on the same fwd that crosses a compress boundary for the
        # FIRST committed entry of a fresh seq).
        compress_slot_mapping_gpu = self._build_indexer_compress_slot_mapping(
            csa_compress_plan_cpu, scheduled_bs, k_per_block, ratio
        )

        # All-empty batch: forward_batched short-circuits on
        # `total_committed == 0` and returns -1; the FP8 read side is unused.
        if total_committed == 0:
            return {
                "total_committed": 0,
                "cu_committed_gpu": None,
                "compress_slot_mapping_gpu": compress_slot_mapping_gpu,
            }

        batch_id_per_token_np = np.repeat(
            np.arange(bs, dtype=np.int64), token_num_per_seq
        )
        is_prefill = start_pos_per_seq == 0
        offset_per_seq_np = np.where(
            is_prefill, token_num_per_seq, self.window_size
        ).astype(np.int64)
        k_per_token_np = k_per_seq_cpu[batch_id_per_token_np]

        # H2D once for each tensor consumed by `Indexer.forward_batched`.
        batch_id_per_token_gpu = self._stage(
            "v4_indexer_batch_id_per_token", batch_id_per_token_np
        )
        # cu_committed_gpu is consumed both as `cu_starts/cu_ends` for the
        # fp8_mqa_logits per-token range AND as `cu_seq_lens` for the
        # cp_gather_indexer_k_quant_cache call (per-seq cumulative committed K).
        cu_committed_gpu = self._stage(
            "v4_indexer_cu_committed", cu_committed_cpu.astype(np.int32)
        )
        n_committed_per_seq_gpu = self._stage(
            "v4_indexer_n_committed_per_seq", n_committed_per_seq
        )
        k_per_token_gpu = self._stage(
            "v4_indexer_k_per_token", k_per_token_np.astype(np.int32)
        )

        # Layer-invariant GPU derivations (each was previously rebuilt ~30x
        # per fwd inside the per-CSA-layer body).
        seq_base_per_token_gpu = cu_committed_gpu[batch_id_per_token_gpu].to(
            torch.int32
        )  # [total_tokens] int32 — per-token offset into concat'd seqs' compressed K
        visible_end_gpu = torch.minimum(
            (positions_gpu[:total_tokens] + 1) // ratio,
            n_committed_per_seq_gpu[batch_id_per_token_gpu],
        ).to(
            torch.int32
        )  # [total_tokens] int32 — per-token causal upper bound
        cu_ends_gpu = (
            seq_base_per_token_gpu + visible_end_gpu
        )  # [total_tokens] int32 — fp8_mqa_logits per-token end offset
        future_threshold_gpu = (
            ((positions_gpu[:total_tokens] + 1) // ratio).to(torch.int32).unsqueeze(1)
        )  # [total_tokens, 1] int32 — broadcast threshold for prefill causal mask
        # Width-overflow mask shaped [total_tokens, index_topk] — matches the
        # uniform indexer output layout (both prefill torch.topk + pad and
        # decode `top_k_per_row_decode` write `index_topk` cols per row, with
        # cols past per-row valid range carrying -1 sentinels).
        col_arange = torch.arange(
            self.index_topk, device=device, dtype=torch.int32
        )  # [index_topk] int32
        width_mask_gpu = col_arange.unsqueeze(0) >= k_per_token_gpu.unsqueeze(
            1
        )  # [total_tokens, index_topk] bool

        offset_per_token_gpu = self._stage(
            "v4_indexer_offset_per_token",
            offset_per_seq_np[batch_id_per_token_np].astype(np.int32),
        )
        is_prefill_per_token_gpu = self._stage(
            "v4_indexer_is_prefill_per_token",
            is_prefill[batch_id_per_token_np],
        ).unsqueeze(1)

        # Pre-allocated decode-path buffers (full [max_bs, ...] views). Decode
        # helper slices each to [:total_tokens]. Returned full because
        # `_build_v4_indexer_meta` is called for both prefill and decode
        # batches; prefill's `total_tokens` may exceed `max_bs` so builder
        # can't pre-slice. Prefill path doesn't read these.
        decode_logits_gpu = self.model_runner.forward_vars[
            "v4_indexer_decode_logits"
        ].gpu  # [max_bs, max_model_len_idx] fp32
        decode_topk_indices_gpu = self.model_runner.forward_vars[
            "v4_indexer_decode_topk_indices"
        ].gpu  # [max_bs, index_topk] int32

        return {
            "total_committed": total_committed,
            "cu_committed_gpu": cu_committed_gpu,
            "n_committed_per_seq_gpu": n_committed_per_seq_gpu,  # int32, [bs]
            "compress_slot_mapping_gpu": compress_slot_mapping_gpu,
            "seq_base_per_token_gpu": seq_base_per_token_gpu,
            "cu_starts_gpu": seq_base_per_token_gpu,  # alias for fp8_mqa_logits
            "cu_ends_gpu": cu_ends_gpu,
            "future_threshold_gpu": future_threshold_gpu,
            "width_mask_gpu": width_mask_gpu,
            "offset_per_token_gpu": offset_per_token_gpu,
            "is_prefill_per_token_gpu": is_prefill_per_token_gpu,
            "decode_logits_gpu": decode_logits_gpu,
            "decode_topk_indices_gpu": decode_topk_indices_gpu,
        }

    def _build_indexer_compress_slot_mapping(
        self,
        csa_compress_plan_cpu,
        scheduled_bs: int,
        k_per_block: int,
        ratio: int,
    ):
        """Compute the per-compress-row flat slot in `v4_csa_idx_kv` pool.

        For each row in `csa_compress_plan_cpu` (= `(ragged_id, batch_id,
        position, window_len)`):
          ci = position // ratio                        # compress entry idx in seq
          block_in_seq  = ci // k_per_block
          slot_in_block = ci %  k_per_block
          physical_block = block_tables_cpu[batch_id, block_in_seq]
          slot = physical_block * k_per_block + slot_in_block

        Returns None when the plan is empty (no boundary crossed this fwd) —
        the caller skips the `indexer_k_quant_and_cache` write entirely.
        """
        if csa_compress_plan_cpu is None or csa_compress_plan_cpu.shape[0] == 0:
            return None
        var = self.model_runner.forward_vars
        block_tables_np = var["block_tables"].np[:scheduled_bs]
        bid = csa_compress_plan_cpu[:, 1]
        pos = csa_compress_plan_cpu[:, 2]
        ci = pos // ratio
        block_in_seq = ci // k_per_block
        slot_in_block = ci % k_per_block
        physical_block = block_tables_np[bid, block_in_seq]
        # int64 — `indexer_k_quant_and_cache` kernel signature is `int64_t*`
        # (matches V3.2's `attn_metadata.slot_mapping` dtype). int32 caused
        # GPU memory access faults from 2x stride mis-stepping.
        slot_mapping_np = physical_block.astype(np.int64) * k_per_block + slot_in_block
        return self._stage("v4_indexer_compress_slot_mapping", slot_mapping_np)

    def _build_v4_gather_indices(
        self,
        *,
        tag: str,
        n_per_seq: np.ndarray,
        k_per_block: int,
        cu_committed_cpu: np.ndarray,
        device,
    ):
        """Pre-build the per-seq gather indices consumed by
        `_v4_gather_compressed_batched`. Returns a dict with all-None GPU
        tensors when total == 0 (caller's gather returns empty).

        `tag` selects which pre-allocated buffer set this call writes into
        (one of "csa_dc", "hca_dc"). Each tag has its own
        `v4_{tag}_gather_{batch_ids,block_in_seq,slot_in_block}` buffer.
        """
        bs = len(n_per_seq)
        total = int(n_per_seq.sum())
        if total == 0:
            return {
                "batch_ids_gpu": None,
                "block_in_seq_gpu": None,
                "slot_in_block_gpu": None,
                "cu_committed_cpu": cu_committed_cpu,
            }
        batch_ids_np = np.repeat(np.arange(bs, dtype=np.int64), n_per_seq)
        committed_idx_np = (
            np.arange(total, dtype=np.int64) - cu_committed_cpu[batch_ids_np]
        )
        block_in_seq_np = committed_idx_np // k_per_block
        slot_in_block_np = committed_idx_np % k_per_block
        return {
            "batch_ids_gpu": self._stage(f"v4_{tag}_gather_batch_ids", batch_ids_np),
            "block_in_seq_gpu": self._stage(
                f"v4_{tag}_gather_block_in_seq", block_in_seq_np
            ),
            "slot_in_block_gpu": self._stage(
                f"v4_{tag}_gather_slot_in_block", slot_in_block_np
            ),
            "cu_committed_cpu": cu_committed_cpu,
        }

    def _build_v4_pack_meta_for_ratio(
        self,
        *,
        kind: str,
        ratio: int,
        has_indexer: bool,
        cu_seqlens_q_np,
        start_pos_per_seq_np,
        slots_np,
        positions_np,
        cu_compress_cpu,
        scheduled_bs: int,
        total_tokens: int,
        device,
    ):
        """Return GPU index tensors consumed by `_v4_build_sparse_inputs_batched`.

        All CPU index math runs once here per (fwd, ratio) — replacing the
        per-layer H2D copies (one `torch.as_tensor` per index buffer) with a
        single H2D each. Returns None for warmup batches that lack
        `state_slot_mapping_cpu`; the V4 forward then falls back to its
        inline path.
        """
        from atom.models.deepseek_v4 import _V4_BLOCK_SIZE

        if (
            scheduled_bs == 0
            or total_tokens == 0
            or slots_np is None
            or len(slots_np) < scheduled_bs
        ):
            return None

        bs = scheduled_bs
        win = self.window_size
        cu_seqlens_q_arr = cu_seqlens_q_np[: bs + 1].astype(np.int64)
        token_num_per_seq = (cu_seqlens_q_arr[1:] - cu_seqlens_q_arr[:bs]).astype(
            np.int64
        )
        start_pos_per_seq = start_pos_per_seq_np[:bs].astype(np.int64)
        end_pos_per_seq = start_pos_per_seq + token_num_per_seq
        is_prefill = start_pos_per_seq == 0
        slots_arr = np.asarray(slots_np[:bs], dtype=np.int64)

        # ---- Per-seq kv layout ----
        part_a_len = np.where(is_prefill, token_num_per_seq, win).astype(np.int64)
        if ratio == 0:
            n_compress_prefill = np.zeros(bs, dtype=np.int64)
            n_committed_decode = np.zeros(bs, dtype=np.int64)
        else:
            n_compress_prefill = (
                (cu_compress_cpu[1:] - cu_compress_cpu[:bs]).astype(np.int64)
                if cu_compress_cpu is not None
                else np.zeros(bs, dtype=np.int64)
            )
            n_committed_decode = (end_pos_per_seq // ratio).astype(np.int64)
        part_b_len = np.where(
            is_prefill, n_compress_prefill, n_committed_decode
        ).astype(np.int64)
        kv_lens = part_a_len + part_b_len
        cu_kv_off = np.concatenate(([0], np.cumsum(kv_lens, dtype=np.int64)))
        total_kv = int(cu_kv_off[bs])

        # ---- Per-seq topk widths ----
        window_topk_width_per_seq = np.where(
            is_prefill, np.minimum(token_num_per_seq, win), win
        ).astype(np.int64)
        if ratio == 0:
            compress_topk_width_per_seq = np.zeros(bs, dtype=np.int64)
        elif has_indexer:
            # CSA: width is min(index_topk, n_committed). Aligned with
            # Indexer.forward_batched's k_per_seq.
            compress_topk_width_per_seq = np.minimum(
                self.index_topk, end_pos_per_seq // ratio
            ).astype(np.int64)
        elif ratio == 128:
            compress_topk_width_per_seq = np.where(
                is_prefill, n_compress_prefill, n_committed_decode
            ).astype(np.int64)
        else:
            raise AssertionError(f"unexpected ratio={ratio}")
        topk_len_per_seq = window_topk_width_per_seq + compress_topk_width_per_seq
        topk_base_per_seq = np.concatenate(
            ([0], np.cumsum(token_num_per_seq * topk_len_per_seq, dtype=np.int64)[:-1])
        )
        total_topk = int((token_num_per_seq * topk_len_per_seq).sum())

        seq_id_per_token = np.repeat(np.arange(bs, dtype=np.int64), token_num_per_seq)
        window_topk_width_per_token = window_topk_width_per_seq[seq_id_per_token]
        compress_topk_width_per_token = compress_topk_width_per_seq[seq_id_per_token]
        topk_starts_per_token = (
            topk_base_per_seq[seq_id_per_token]
            + (
                np.arange(total_tokens, dtype=np.int64)
                - cu_seqlens_q_arr[seq_id_per_token]
            )
            * topk_len_per_seq[seq_id_per_token]
        )

        # ---- Part A: prefill kv slice ----
        prefill_seqs = np.flatnonzero(is_prefill)
        prefill_kv_dst_gpu = prefill_kv_src_gpu = None
        if prefill_seqs.size:
            tok_seq, local = _segment_indices(
                prefill_seqs, token_num_per_seq[prefill_seqs]
            )
            prefill_kv_src_gpu = self._stage(
                f"v4_{kind}_pack_prefill_kv_src", cu_seqlens_q_arr[tok_seq] + local
            )
            prefill_kv_dst_gpu = self._stage(
                f"v4_{kind}_pack_prefill_kv_dst", cu_kv_off[tok_seq] + local
            )

        # ---- Part A: decode swa gather ----
        decode_seqs = np.flatnonzero(~is_prefill)
        decode_state_slots_gpu = decode_swa_dst_gpu = None
        if decode_seqs.size:
            decode_state_slots_gpu = self._stage(
                f"v4_{kind}_pack_decode_state_slots", slots_arr[decode_seqs]
            )
            decode_swa_dst_gpu = self._stage(
                f"v4_{kind}_pack_decode_swa_dst",
                (cu_kv_off[decode_seqs][:, None] + np.arange(win)).reshape(-1),
            )

        # ---- Part B: prefill compress ----
        prefill_compress_dst_gpu = prefill_compress_src_gpu = None
        if ratio > 0 and prefill_seqs.size and cu_compress_cpu is not None:
            pc_seqs = prefill_seqs[n_compress_prefill[prefill_seqs] > 0]
            if pc_seqs.size:
                seg_lens = n_compress_prefill[pc_seqs]
                tok_seq, local = _segment_indices(pc_seqs, seg_lens)
                prefill_compress_src_gpu = self._stage(
                    f"v4_{kind}_pack_prefill_compress_src",
                    cu_compress_cpu[tok_seq].astype(np.int64) + local,
                )
                prefill_compress_dst_gpu = self._stage(
                    f"v4_{kind}_pack_prefill_compress_dst",
                    cu_kv_off[tok_seq] + token_num_per_seq[tok_seq] + local,
                )

        # ---- Part B: decode compress ----
        decode_compress_seqs_gpu = decode_compress_dst_gpu = None
        decode_compress_gather_indices = None
        if ratio > 0 and decode_seqs.size:
            dc_seqs = decode_seqs[n_committed_decode[decode_seqs] > 0]
            if dc_seqs.size:
                decode_compress_seqs_gpu = self._stage(
                    f"v4_{kind}_pack_decode_compress_seqs",
                    dc_seqs.astype(np.int64),
                )
                seg_lens = n_committed_decode[dc_seqs]
                tok_seq, local = _segment_indices(dc_seqs, seg_lens)
                decode_compress_dst_gpu = self._stage(
                    f"v4_{kind}_pack_decode_compress_dst",
                    cu_kv_off[tok_seq] + win + local,
                )
                # Gather indices for `_v4_gather_compressed_batched` on the
                # filtered subset (dc_seqs). cu_committed here is the
                # subset-local prefix sum. Per-kind tag selects buffer set.
                dc_cu_committed = np.concatenate(
                    [np.zeros(1, dtype=np.int64), np.cumsum(seg_lens, dtype=np.int64)]
                )
                decode_compress_gather_indices = self._build_v4_gather_indices(
                    tag=f"{kind}_dc",
                    n_per_seq=seg_lens,
                    k_per_block=_V4_BLOCK_SIZE // ratio,
                    cu_committed_cpu=dc_cu_committed,
                    device=device,
                )

        # ---- Topk: window part ----
        window_topk_dst_gpu = window_topk_src_gpu = None
        # window_topk_batched has fixed width = win across the whole batch
        # (see _build_window_topk_batched), so src offsets use win as stride.
        if total_tokens > 0 and window_topk_width_per_token.sum() > 0:
            tok_idx_w, k_idx_w = _segment_indices(
                np.arange(total_tokens, dtype=np.int64),
                window_topk_width_per_token,
            )
            window_topk_src_gpu = self._stage(
                f"v4_{kind}_pack_window_topk_src", tok_idx_w * win + k_idx_w
            )
            window_topk_dst_gpu = self._stage(
                f"v4_{kind}_pack_window_topk_dst",
                topk_starts_per_token[tok_idx_w] + k_idx_w,
            )

        # ---- Topk: compress part ----
        compress_topk_dst_gpu = None
        compress_topk_src_gpu = None
        compress_topk_values_gpu = None
        if compress_topk_width_per_token.sum() > 0:
            tok_idx_c, k_idx_c = _segment_indices(
                np.arange(total_tokens, dtype=np.int64),
                compress_topk_width_per_token,
            )
            compress_topk_dst_gpu = self._stage(
                f"v4_{kind}_pack_compress_topk_dst",
                topk_starts_per_token[tok_idx_c]
                + window_topk_width_per_token[tok_idx_c]
                + k_idx_c,
            )
            if has_indexer:
                # CSA: src is per-token gather offset into indexer_topk_batched
                # (flattened with `.reshape(-1)` then `index_select`-ed). The
                # gather index is `tok_idx * stride + k_idx`, so the stride
                # MUST match the batched indexer's actual col count.
                #
                # Indexer emits a uniform [total_tokens, index_topk] int32
                # layout for BOTH prefill and decode paths (cols past per-row
                # valid range hold -1 sentinels; consumer width-masks them).
                indexer_max = int(self.index_topk)
                compress_topk_src_gpu = self._stage(
                    f"v4_{kind}_pack_compress_topk_src",
                    tok_idx_c * indexer_max + k_idx_c,
                )
            elif positions_np is not None:
                # HCA: precompute the int32 values (k_idx + offset, future-mask
                # for prefill). Depends on `positions` which is per-fwd / layer-
                # invariant, so this is safe to hoist. Skip when positions_np
                # isn't threaded (caller relies on the V4 forward inline path).
                offset_per_seq = np.where(is_prefill, token_num_per_seq, win).astype(
                    np.int64
                )
                offset_per_token = offset_per_seq[seq_id_per_token]
                positions_arr = positions_np[:total_tokens].astype(np.int64)
                # Prefill future-mask: k >= (pos+1)//ratio → -1
                future_threshold_per_token = (positions_arr + 1) // ratio
                is_prefill_per_token = is_prefill[seq_id_per_token]
                future_mask = is_prefill_per_token[tok_idx_c] & (
                    k_idx_c >= future_threshold_per_token[tok_idx_c]
                )
                values = np.where(
                    future_mask, -1, k_idx_c + offset_per_token[tok_idx_c]
                ).astype(np.int32)
                compress_topk_values_gpu = self._stage(
                    f"v4_{kind}_pack_compress_topk_values", values
                )

        return {
            "total_kv": total_kv,
            "total_topk": total_topk,
            "prefill_kv_dst_gpu": prefill_kv_dst_gpu,
            "prefill_kv_src_gpu": prefill_kv_src_gpu,
            "decode_state_slots_gpu": decode_state_slots_gpu,
            "decode_swa_dst_gpu": decode_swa_dst_gpu,
            "prefill_compress_dst_gpu": prefill_compress_dst_gpu,
            "prefill_compress_src_gpu": prefill_compress_src_gpu,
            "decode_compress_seqs_gpu": decode_compress_seqs_gpu,
            "decode_compress_dst_gpu": decode_compress_dst_gpu,
            "decode_compress_gather_indices": decode_compress_gather_indices,
            "window_topk_dst_gpu": window_topk_dst_gpu,
            "window_topk_src_gpu": window_topk_src_gpu,
            "compress_topk_dst_gpu": compress_topk_dst_gpu,
            "compress_topk_src_gpu": compress_topk_src_gpu,
            "compress_topk_values_gpu": compress_topk_values_gpu,
        }

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        """V4-style decode prep: populates positions, cu_seqlens_q,
        block_tables, and state_slot_mapping.

        For PR3-main multi-seq:
          - cu_seqlens_q gives V4 forward per-seq token-range slicing.
          - block_tables[i] is the per-seq block list for compressor scatter
            and sparse_attn gather.
          - state_slot_mapping[i] is the per-seq state-cache slot (swa_kv +
            Compressor.kv_state row index). Distinct from `slot_mapping`
            which is per-token paged-KV-pool index.

        Also publishes CPU mirrors (`v4_*_cpu` attrs) so the V4 forward path
        can read per-seq metadata without GPU→CPU `.tolist()`/.item() syncs
        (PR-A Phase 2: required to unlock CUDAGraph).
        """
        import numpy as np

        var = self.model_runner.forward_vars
        scheduled_bs = batch.total_seqs_num_decode
        # Naming: `_np` suffix marks numpy arrays, `_gpu` marks GPU tensors.
        # Both representations of context_lens coexist (CPU mirror feeds the
        # plan builder; GPU copy goes into attn_metadata for kernels).
        context_lens_np = np.asarray(batch.context_lens, dtype=np.int32)
        max_seqlen_q = batch.num_spec_step + 1
        positions_np = np.tile(
            np.arange(max_seqlen_q, dtype=np.int32), scheduled_bs
        ) + np.repeat(context_lens_np - max_seqlen_q, max_seqlen_q)
        sum_scheduled_tokens = batch.total_tokens_num_decode
        var["positions"].np[:sum_scheduled_tokens] = positions_np
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)

        # cu_seqlens_q for decode: each seq has max_seqlen_q tokens.
        cu_seqlens_q_np = np.arange(
            0, (scheduled_bs + 1) * max_seqlen_q, max_seqlen_q, dtype=np.int32
        )
        var["cu_seqlens_q"].np[: scheduled_bs + 1] = cu_seqlens_q_np
        cu_seqlens_q_gpu = var["cu_seqlens_q"].copy_to_gpu(scheduled_bs + 1)

        # PR-A: V4 Compressor state-cache update reads context_lens to
        # discriminate fresh prefill vs decode/prefix-cache. Parent's
        # prepare_prefill pushes this; decode path must do the same.
        var["context_lens"].np[:scheduled_bs] = context_lens_np
        context_lens_gpu = var["context_lens"].copy_to_gpu(scheduled_bs)

        block_tables_gpu = self._populate_block_tables(batch, scheduled_bs)
        state_slot_gpu, state_slot_np = self._populate_state_slot_mapping(
            batch, scheduled_bs, return_cpu=True
        )
        attn_metadata = AttentionMetaData(
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
        )
        # Attach V4-specific per-seq metadata via dynamic attribute.
        # AttentionMetaData is a regular class (not slotted), so this works
        # without modifying its schema.
        attn_metadata.state_slot_mapping = state_slot_gpu
        # PR-A Phase 2: CPU mirrors so the V4 forward path can read per-seq
        # metadata without `.tolist()` / `.item()` GPU→CPU syncs. Generic
        # naming (no `v4_` prefix) — these are not V4-specific concepts and
        # other backends with stateful per-request buffers can adopt them.
        attn_metadata.cu_seqlens_q_cpu = cu_seqlens_q_np
        attn_metadata.state_slot_mapping_cpu = state_slot_np
        attn_metadata.start_pos_per_seq_cpu = positions_np[
            cu_seqlens_q_np[:scheduled_bs]
        ]
        # Compress plans (per ratio) for batched fused_compress + update_states.
        # Decode batch: extend_lens = max_seqlen_q for all seqs (uniform).
        # `context_lens_np` is post-extend (from batch.context_lens, set by
        # scheduler after appending this fwd's tokens) — this is what plan
        # generation needs as `seq_lens`. Must run BEFORE
        # `_attach_sparse_layout_metadata` since it consumes plan.cu_compress_cpu
        # to compute the per-token kv_len for sparse-attn ragged layout.
        extend_lens_np = np.full(scheduled_bs, max_seqlen_q, dtype=np.int32)
        attn_metadata.compress_plans = self._build_compress_plans(
            extend_lens_np, context_lens_np, positions.device
        )
        self._attach_sparse_layout_metadata(
            attn_metadata,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
            positions_np=positions_np,
            positions_gpu=positions,
        )
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            positions,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            attn_metadata.state_slot_mapping_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
        )
        return attn_metadata, positions

    def prepare_prefill(self, batch: ScheduledBatch):
        """V4 prefill prep: extends parent to always populate block_tables
        and state_slot_mapping.

        The parent only emits block_tables when has_cached (prefix cache hit);
        V4 always needs block_tables because Compressor scatters compressed
        entries into the classical KV pool from token 0 onwards.

        Also publishes CPU mirrors (`v4_*_cpu`) consumed by the V4 forward
        path to avoid `.item()` / `.tolist()` syncs (PR-A Phase 2).
        """
        import numpy as np

        attn_metadata, positions = super().prepare_prefill(batch)
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
        attn_metadata.cu_seqlens_q_cpu = cu_seqlens_q_np
        attn_metadata.state_slot_mapping_cpu = state_slot_np
        attn_metadata.start_pos_per_seq_cpu = positions_np[
            cu_seqlens_q_np[:scheduled_bs]
        ]
        # Compress plans (per ratio) for batched fused_compress + update_states.
        # Prefill batch: extend_lens read from cu_seqlens_q_np.
        # Must run BEFORE `_attach_sparse_layout_metadata` (sparse layout
        # consumes plan.cu_compress_cpu for the per-token kv_len computation).
        extend_lens_np = (
            cu_seqlens_q_np[1 : scheduled_bs + 1] - cu_seqlens_q_np[:scheduled_bs]
        ).astype(np.int32)
        # context_lens for prefill = positions[seq_start_in] + extend_lens
        # (= absolute seq_len incl. this fwd's tokens).
        context_lens_np = (attn_metadata.start_pos_per_seq_cpu + extend_lens_np).astype(
            np.int32
        )
        attn_metadata.compress_plans = self._build_compress_plans(
            extend_lens_np, context_lens_np, positions.device
        )
        self._attach_sparse_layout_metadata(
            attn_metadata,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
            positions_np=positions_np,
            positions_gpu=positions,
        )
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            positions,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            attn_metadata.state_slot_mapping_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
        )
        return attn_metadata, positions

    def _attach_v4_per_fwd_meta(
        self,
        attn_metadata: AttentionMetaData,
        positions: torch.Tensor,
        cu_seqlens_q_np,
        start_pos_per_seq_cpu,
        state_slot_mapping_cpu,
        scheduled_bs: int,
        total_tokens: int,
    ) -> None:
        """Hoist per-fwd, layer-invariant metadata used by every V4 layer.

        These tensors only depend on `positions`, `cu_seqlens_q`, `state_slot_mapping`
        and `window_size` — none of which change across layers — so building
        them once per fwd saves ~64 redundant constructions for V4-Pro.

        Sets:
          - `attn_metadata.window_topk_batched`: [total_tokens, win] int32
            sliding-window absolute-position matrix (consumed by the sparse-input
            packer for window-side topk indices).
          - `attn_metadata.swa_write_indices`: [num_write] int64 row indices
            (last `win` tokens per seq) into the per-token KV.
          - `attn_metadata.swa_positions_filtered`: positions[swa_write_indices]
            contiguous int64.
          - `attn_metadata.swa_slot_per_token_filtered`: per-token state-slot
            ids restricted to `swa_write_indices`, contiguous int32.

        When the per-fwd write set is empty (rare: bs==0 or every seq has 0
        tokens — only possible during cudagraph dry runs that go through this
        path), the swa_* fields are set to None.
        """
        if scheduled_bs == 0 or total_tokens == 0:
            attn_metadata.window_topk_batched = None
            attn_metadata.swa_write_indices = None
            attn_metadata.swa_positions_filtered = None
            attn_metadata.swa_slot_per_token_filtered = None
            return

        win = self.window_size

        cu_seqlens_q_arr = np.asarray(
            cu_seqlens_q_np[: scheduled_bs + 1], dtype=np.int64
        )
        token_num_per_seq = cu_seqlens_q_arr[1:] - cu_seqlens_q_arr[:scheduled_bs]
        repeat_output_size = int(token_num_per_seq.sum())
        if repeat_output_size != total_tokens:
            raise ValueError(
                "DeepSeek-V4 metadata token count mismatch: "
                f"sum(cu_seqlens_q diff)={repeat_output_size}, "
                f"total_tokens={total_tokens}"
            )
        start_pos_per_seq_np = np.asarray(
            start_pos_per_seq_cpu[:scheduled_bs], dtype=np.int64
        )

        # ----- window_topk_batched (per-token) -----
        start_pos_per_seq_gpu = self._stage(
            "v4_meta_start_pos_per_seq", start_pos_per_seq_np
        )
        token_num_per_seq_gpu = self._stage(
            "v4_meta_token_num_per_seq", token_num_per_seq
        )
        # `repeats` is a CUDA tensor. Supplying the CPU-known output size avoids
        # PyTorch synchronizing to compute sum(repeats) on every high-conc prefill.
        start_pos_per_token = torch.repeat_interleave(
            start_pos_per_seq_gpu,
            token_num_per_seq_gpu,
            output_size=repeat_output_size,
        )
        attn_metadata.window_topk_batched = _build_window_topk_batched(
            positions[:total_tokens].to(torch.long), start_pos_per_token, win
        )

        # ----- SWA write indices (last `win` tokens per seq) -----
        # Warmup dummy batches may pass scheduled_bs > 0 but with an empty
        # `state_slot_mapping_cpu` (no per-req cache groups assigned). In that
        # case skip the swa hoist and let V4Attention.forward's inline path
        # handle it (or its `_v4_is_dummy_run` short-circuit).
        if (
            len(state_slot_mapping_cpu) < scheduled_bs
            or int(token_num_per_seq.sum()) == 0
        ):
            attn_metadata.swa_write_indices = None
            attn_metadata.swa_positions_filtered = None
            attn_metadata.swa_slot_per_token_filtered = None
            return
        write_starts = cu_seqlens_q_arr[:scheduled_bs] + np.maximum(
            0, token_num_per_seq - win
        )
        write_ends = cu_seqlens_q_arr[1:]
        if int((write_ends - write_starts).sum()) == 0:
            attn_metadata.swa_write_indices = None
            attn_metadata.swa_positions_filtered = None
            attn_metadata.swa_slot_per_token_filtered = None
            return
        write_indices_np = np.concatenate(
            [np.arange(s, e, dtype=np.int64) for s, e in zip(write_starts, write_ends)]
        )
        write_indices_gpu = self._stage("v4_meta_swa_write_indices", write_indices_np)
        state_slot_mapping_gpu_i32 = self._stage(
            "v4_meta_state_slot_i32",
            np.asarray(state_slot_mapping_cpu[:scheduled_bs], dtype=np.int32),
        )
        slot_per_token_full = torch.repeat_interleave(
            state_slot_mapping_gpu_i32,
            token_num_per_seq_gpu,
            output_size=repeat_output_size,
        )
        attn_metadata.swa_write_indices = write_indices_gpu
        attn_metadata.swa_positions_filtered = positions[write_indices_gpu].contiguous()
        attn_metadata.swa_slot_per_token_filtered = slot_per_token_full[
            write_indices_gpu
        ].contiguous()

    def _build_compress_plans(self, extend_lens_np, seq_lens_np, device):
        """Build per-ratio CompressPlan dict consumed by batched compressor.

        Reuse this from both prepare_decode and prepare_prefill — caller
        supplies extend_lens / seq_lens (np int32) and target device. Plan
        tensors are written into the pre-allocated `v4_compress_plan_{ratio}`
        / `v4_write_plan_{ratio}` CpuGpuBuffers (fixed pointers for
        CUDAGraph capture); the kernels skip sentinel-marked tail rows.
        """
        from atom.model_ops.v4_kernels import make_compress_plans

        if not self._unique_compress_ratios_overlap:
            return {}
        # Ensure inputs are np int32 (callers may pass torch tensors / lists).
        if isinstance(extend_lens_np, torch.Tensor):
            extend_lens_np = extend_lens_np.cpu().numpy().astype(np.int32)
        if isinstance(seq_lens_np, torch.Tensor):
            seq_lens_np = seq_lens_np.cpu().numpy().astype(np.int32)
        var = self.model_runner.forward_vars
        plan_buffers = {
            ratio: {
                "compress": var[f"v4_compress_plan_{ratio}"],
                "write": var[f"v4_write_plan_{ratio}"],
            }
            for ratio, _ in self._unique_compress_ratios_overlap
        }
        return make_compress_plans(
            np.ascontiguousarray(extend_lens_np, dtype=np.int32),
            np.ascontiguousarray(seq_lens_np, dtype=np.int32),
            self._unique_compress_ratios_overlap,
            device,
            plan_buffers=plan_buffers,
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
        import numpy as np

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

    def build_for_cudagraph_capture(self, bs: int) -> tuple[AttentionMetaData, Context]:
        """Build attn_metadata for CUDAGraph capture using a synthetic decode batch.

        Synthesizes bs sequences each at start_pos=window_size (so SWA window
        is full + 1 CSA committed entry — exercises the production decode
        codepath: state-cache reads, sparse_attn gather, indexer fp8 logits).

        Per-fwd metadata is populated through the SAME helpers prepare_decode
        uses (`_attach_sparse_layout_metadata`, `_attach_v4_per_fwd_meta`,
        `_build_compress_plans`), so all GPU views point to the pre-allocated
        buffers in `forward_vars`. Replay-time prepare_decode writes into the
        SAME buffers — captured graph reads stable addresses.

        NOTE on dynamic-shape kernels (fused_compress_attn / update_compressor_states /
        swa_write): these currently use variable kernel grids (`grid=(num_compress,)`),
        which CUDAGraph capture rejects. A follow-up PR converts them to fixed
        grid + sentinel masking. Until then, capture itself can succeed (the
        helpers run on CPU + small H2D), but model.forward inside torch.cuda.graph
        will likely fail at the first such kernel launch — the user can detect
        this via capture log output.
        """
        device = self.model_runner.device
        var = self.model_runner.forward_vars
        max_q_len = 1
        total_tokens = bs * max_q_len
        win = self.window_size

        # Synthetic state: each seq has already produced `win` tokens; this fwd
        # is one decode step at position `win`.
        start_pos = win
        positions_np = np.full(total_tokens, start_pos, dtype=np.int64)
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

        attn_metadata = AttentionMetaData(
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
        )
        attn_metadata.state_slot_mapping = state_slot_gpu
        attn_metadata.cu_seqlens_q_cpu = cu_seqlens_q_np
        attn_metadata.state_slot_mapping_cpu = state_slot_np
        attn_metadata.start_pos_per_seq_cpu = positions_np[cu_seqlens_q_np[:bs]]

        # Build compress_plans + sparse layouts + per-fwd meta via the same
        # helpers used at runtime — guarantees addresses match.
        extend_lens_np = np.full(bs, max_q_len, dtype=np.int32)
        attn_metadata.compress_plans = self._build_compress_plans(
            extend_lens_np, context_lens_np, device
        )
        self._attach_sparse_layout_metadata(
            attn_metadata,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            bs,
            total_tokens,
            positions_np=positions_np,
            positions_gpu=positions,
        )
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            positions,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            attn_metadata.state_slot_mapping_cpu,
            bs,
            total_tokens,
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
          - window_topk:    max_num_batched_tokens * window_size
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
        bool_kw = {"dtype": torch.bool, "device": self.device}
        mnbt = self.max_num_batched_tokens
        bs = self.max_bs
        win = self.window_size
        nbps = self.max_num_blocks_per_seq
        k1_csa = self.k1_csa  # block_size // 4 = 32

        bufs: dict = {}

        # Sparse-attn layout (per-token, per-kind). Existing v4_{kind}_sparse_*.
        for kind in ("dense", "csa", "hca"):
            bufs[f"v4_{kind}_sparse_topk_starts"] = CpuGpuBuffer(mnbt, **i64)
            bufs[f"v4_{kind}_sparse_topk_lens"] = CpuGpuBuffer(mnbt, **i32)
            bufs[f"v4_{kind}_sparse_kv_offsets"] = CpuGpuBuffer(mnbt, **i32)

        # `kv_indptr` is touched unconditionally by the global capture loop
        # (model_runner.capture_cudagraph: `forward_vars["kv_indptr"].zero_()`).
        # MLA backends own this buffer; V4 doesn't use it for its own kernels
        # but allocates a min-size stub so the capture loop runs. Sized for
        # potential future reuse if a V4-side MLA kernel needs paged KV indices.
        bufs["kv_indptr"] = CpuGpuBuffer(bs + 1, **i32)

        # _attach_v4_per_fwd_meta + _populate_state_slot_mapping.
        bufs["v4_meta_start_pos_per_seq"] = CpuGpuBuffer(bs, **i64)
        bufs["v4_meta_token_num_per_seq"] = CpuGpuBuffer(bs, **i64)
        bufs["v4_meta_state_slot_i32"] = CpuGpuBuffer(bs, **i32)
        bufs["v4_meta_state_slot_groups"] = CpuGpuBuffer(bs, **i32)
        bufs["v4_meta_swa_write_indices"] = CpuGpuBuffer(mnbt, **i64)

        # _build_v4_indexer_meta (CSA only — but allocate unconditionally;
        # never accessed when CSA layers are absent).
        bufs["v4_indexer_batch_id_per_token"] = CpuGpuBuffer(mnbt, **i64)
        # int32 — `cp_gather_indexer_k_quant_cache` kernel signature is `int32_t*`
        # for cu_seq_lens. Also reused as cu_starts/cu_ends for fp8_mqa_logits
        # (which accepts both int32 and int64).
        bufs["v4_indexer_cu_committed"] = CpuGpuBuffer(bs + 1, **i32)
        # int32 — consumed both by `deepgemm_fp8_paged_mqa_logits` and
        # `top_k_per_row_decode`, both of which require int32.
        bufs["v4_indexer_n_committed_per_seq"] = CpuGpuBuffer(bs, **i32)
        # Decode-path logits buffer for `deepgemm_fp8_paged_mqa_logits`.
        # Sized [max_bs, max_model_len_idx] fp32 — assumes V4-Pro next_n=1.
        # deepgemm writes valid cols [0, n_committed_per_seq[batch]) per row;
        # padding cols carry stale data but `top_k_per_row_decode` honors
        # `n_committed_per_seq` per-row so unwritten cols are never selected.
        bufs["v4_indexer_decode_logits"] = CpuGpuBuffer(
            bs, self.max_model_len_idx, dtype=torch.float32, device=self.device
        )
        # Decode-path top-k indices buffer (consumed by `top_k_per_row_decode`).
        # Sized [max_bs, index_topk] int32 — V4-Pro next_n=1; multi-token decode
        # would scale rows by next_n. Replaces the per-fwd torch.topk allocation.
        bufs["v4_indexer_decode_topk_indices"] = CpuGpuBuffer(
            bs, self.index_topk, dtype=torch.int32, device=self.device
        )
        bufs["v4_indexer_k_per_token"] = CpuGpuBuffer(mnbt, **i32)
        bufs["v4_indexer_offset_per_token"] = CpuGpuBuffer(mnbt, **i32)
        bufs["v4_indexer_is_prefill_per_token"] = CpuGpuBuffer(mnbt, **bool_kw)
        # FP8 cache write-side slot mapping (one entry per compress row).
        # Bound = ⌈mnbt/4⌉ + bs (worst-case num_compress for ratio=4 CSA;
        # matches v4_compress_plan_4 row count).
        # int64 — `indexer_k_quant_and_cache` requires int64 slot_mapping.
        idx_compress_bound = mnbt // 4 + bs
        bufs["v4_indexer_compress_slot_mapping"] = CpuGpuBuffer(
            idx_compress_bound, **i64
        )

        # Gather indices — 2 sets: csa decode_compress, hca decode_compress.
        # Each set has 3 i64 tensors. Sized to the CSA bound
        # (max_bs * max_num_blocks_per_seq * k1_csa) since CSA is the ratio-4
        # (largest) case; HCA reuses the same bound for simplicity.
        # The indexer side no longer needs a gather buffer set —
        # `cp_gather_indexer_k_quant_cache` consumes block_tables + cu_seq_lens
        # directly.
        gather_max = bs * nbps * k1_csa
        for tag in ("csa_dc", "hca_dc"):
            bufs[f"v4_{tag}_gather_batch_ids"] = CpuGpuBuffer(gather_max, **i64)
            bufs[f"v4_{tag}_gather_block_in_seq"] = CpuGpuBuffer(gather_max, **i64)
            bufs[f"v4_{tag}_gather_slot_in_block"] = CpuGpuBuffer(gather_max, **i64)

        # _build_v4_pack_meta_for_ratio (per-(kind) variant).
        # window_topk fixed-width = win.
        # compress_topk width per kind: csa = index_topk, hca = nbps.
        compress_topk_max_per_kind = {
            "dense": 0,
            "csa": self.index_topk,
            "hca": nbps,
        }
        for kind in ("dense", "csa", "hca"):
            ckmax = compress_topk_max_per_kind[kind] * mnbt
            bufs[f"v4_{kind}_pack_prefill_kv_src"] = CpuGpuBuffer(mnbt, **i64)
            bufs[f"v4_{kind}_pack_prefill_kv_dst"] = CpuGpuBuffer(mnbt, **i64)
            bufs[f"v4_{kind}_pack_decode_state_slots"] = CpuGpuBuffer(bs, **i64)
            bufs[f"v4_{kind}_pack_decode_swa_dst"] = CpuGpuBuffer(bs * win, **i64)
            bufs[f"v4_{kind}_pack_prefill_compress_src"] = CpuGpuBuffer(mnbt, **i64)
            bufs[f"v4_{kind}_pack_prefill_compress_dst"] = CpuGpuBuffer(mnbt, **i64)
            bufs[f"v4_{kind}_pack_decode_compress_seqs"] = CpuGpuBuffer(bs, **i64)
            bufs[f"v4_{kind}_pack_decode_compress_dst"] = CpuGpuBuffer(
                gather_max, **i64
            )
            bufs[f"v4_{kind}_pack_window_topk_src"] = CpuGpuBuffer(mnbt * win, **i64)
            bufs[f"v4_{kind}_pack_window_topk_dst"] = CpuGpuBuffer(mnbt * win, **i64)
            if ckmax > 0:
                bufs[f"v4_{kind}_pack_compress_topk_src"] = CpuGpuBuffer(ckmax, **i64)
                bufs[f"v4_{kind}_pack_compress_topk_dst"] = CpuGpuBuffer(ckmax, **i64)
                bufs[f"v4_{kind}_pack_compress_topk_values"] = CpuGpuBuffer(
                    ckmax, **i32
                )

        # Compress plan buffers (per-ratio) — pre-allocated for CUDAGraph
        # plan-tensor address stability. `make_compress_plans(..., plan_buffers=)`
        # writes into these and sentinel-fills the trailing rows. Worst-case
        # sizes: num_compress ≤ ⌈mnbt/ratio⌉ + bs (one boundary per seq plus
        # alignment slack); num_write ≤ bs * STATE_SIZE (per-seq ring window
        # carries STATE_SIZE rows per fwd at most).
        max_compress_per_ratio = {}
        for ratio, is_overlap in self._unique_compress_ratios_overlap:
            state_size = (2 if is_overlap else 1) * ratio
            max_compress = mnbt // ratio + bs
            max_write = min(mnbt, bs * state_size)
            max_compress_per_ratio[ratio] = max_compress
            bufs[f"v4_compress_plan_{ratio}"] = CpuGpuBuffer(max_compress, 4, **i32)
            bufs[f"v4_write_plan_{ratio}"] = CpuGpuBuffer(max_write, 4, **i32)
            # Pre-fill with sentinel so capture-time buffer state is valid
            # even before the first non-empty fwd.
            bufs[f"v4_compress_plan_{ratio}"].cpu.fill_(-1)
            bufs[f"v4_compress_plan_{ratio}"].copy_to_gpu()
            bufs[f"v4_write_plan_{ratio}"].cpu.fill_(-1)
            bufs[f"v4_write_plan_{ratio}"].copy_to_gpu()

        # Compressor output buffers (one per kind, shared across same-kind
        # layers within a single fwd — Compressor outputs are consumed
        # immediately by the layer's sparse_attn before the next layer runs).
        # Sized to (max_compress_per_ratio, head_dim) so fused_compress_attn
        # can launch with full-capacity grid + sentinel-skip; output rows
        # past the actual num_compress carry stale data but are never read
        # (consumer slices via `cu_compress_cpu`).
        bf16 = {"dtype": torch.bfloat16, "device": self.device}
        if 4 in max_compress_per_ratio:
            mc = max_compress_per_ratio[4]
            bufs["v4_csa_main_compress_out"] = torch.empty((mc, self.head_dim), **bf16)
            bufs["v4_csa_idx_compress_out"] = torch.empty(
                (mc, self.index_head_dim), **bf16
            )
        if 128 in max_compress_per_ratio:
            mc = max_compress_per_ratio[128]
            bufs["v4_hca_main_compress_out"] = torch.empty((mc, self.head_dim), **bf16)

        self.model_runner.forward_vars.update(bufs)

    def _stage(self, name: str, arr) -> torch.Tensor:
        """Write numpy `arr` into `forward_vars[name]` (CpuGpuBuffer) and
        return its GPU view sliced to len(arr). Auto-casts dtype to match
        the buffer (e.g. int64 → int32). Asserts the buffer is large enough.
        """
        buf = self.model_runner.forward_vars[name]
        n = arr.shape[0] if arr.ndim > 0 else 1
        if n == 0:
            return buf.gpu[:0]
        cap = buf.np.shape[0]
        assert n <= cap, (
            f"V4 buffer {name!r} too small: need {n}, have {cap}. "
            f"Increase the corresponding bound in _alloc_v4_metadata_buffers."
        )
        if arr.dtype != buf.np.dtype:
            arr = arr.astype(buf.np.dtype, copy=False)
        buf.np[:n] = arr
        return buf.copy_to_gpu(n)

    @staticmethod
    def _numel(shape: tuple) -> int:
        n = 1
        for s in shape:
            n *= s
        return n

    def _zero_state(self, shape: tuple, device) -> torch.Tensor:
        return torch.zeros(shape, dtype=self._state_dtype, device=device)

    def _neg_inf_state(self, shape: tuple, device) -> torch.Tensor:
        return torch.full(shape, float("-inf"), dtype=self._state_dtype, device=device)
