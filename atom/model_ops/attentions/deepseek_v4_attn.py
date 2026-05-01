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

import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attentions.backends import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionBuilder,
)
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import AttentionMetaData


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
        self.layer_id_to_csa_pos = {l: p for p, l in enumerate(self.csa_layers)}
        self.layer_id_to_hca_pos = {l: p for p, l in enumerate(self.hca_layers)}

        # Geometry from HF config.
        self.head_dim = getattr(hf, "kv_head_dim", 512)
        self.index_head_dim = getattr(hf, "index_head_dim", 128)
        self.window_size = getattr(hf, "sliding_window", 128)
        self.index_topk = getattr(hf, "index_topk", 1024)

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
        self._classical_dtype = torch.bfloat16  # compressed KV is BF16

        # Sparse-attn layout metadata, aligned with aiter_mla's CpuGpuBuffer
        # convention. Values are per-token and reusable across layers with the
        # same compress_ratio; the actual topk indices remain layer-specific.
        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        v4_sparse_metadata = {}
        for kind in ("dense", "csa", "hca"):
            v4_sparse_metadata[f"v4_{kind}_sparse_topk_starts"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i64_kwargs
            )
            v4_sparse_metadata[f"v4_{kind}_sparse_topk_lens"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i32_kwargs
            )
            v4_sparse_metadata[f"v4_{kind}_sparse_kv_offsets"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i32_kwargs
            )
        self.model_runner.forward_vars.update(v4_sparse_metadata)

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
        """Per-V4-block bytes for the three classical KV pools (BF16).

        Each V4 block (block_size=128 original tokens) stores per layer:
          - CSA layer: k1=32 entries × head_dim (Main) + k1×idx_head_dim (Indexer)
          - HCA layer: k2=1 entry × head_dim
        """
        elem = self._classical_dtype.itemsize
        csa_main_per_block = self.k1_csa * self.head_dim * elem
        csa_idx_per_block = self.k1_csa * self.index_head_dim * elem
        hca_main_per_block = self.k2_hca * self.head_dim * elem
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
            "v4_csa_idx_kv": torch.zeros(
                (num_blocks, n_csa, self.k1_csa, self.index_head_dim),
                dtype=self._classical_dtype,
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
            layer_id_from_prefix = int(module.prefix.split(".")[1])
            pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
            module.kv_cache = runner.v4_csa_idx_kv[:, pos]
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
                # outer Indexer.kv_cache (csa_idx_kv).
                module.kv_cache = runner.v4_csa_idx_kv[:, pos]
            elif ratio == 4:
                pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_csa_main_kv_state[pos]
                module.score_state = runner.v4_csa_main_score_state[pos]
                module.kv_cache = runner.v4_csa_main_kv[:, pos]
            elif ratio == 128:
                pos = self.layer_id_to_hca_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_hca_main_kv_state[pos]
                module.score_state = runner.v4_hca_main_score_state[pos]
                module.kv_cache = runner.v4_hca_main_kv[:, pos]
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
    ) -> None:
        """Precompute per-token ragged sparse-attn layout for each ratio type.

        The actual topk index values are layer-specific (CSA Indexer depends on
        weights), but the per-token topk span and global-KV offset layout only
        depends on the request geometry and compress_ratio.
        """
        import numpy as np

        var = self.model_runner.forward_vars
        layouts = {}
        ratio_specs = {
            0: ("dense", 0),
            4: ("csa", 4),
            128: ("hca", 128),
        }
        for ratio_key, (kind, ratio) in ratio_specs.items():
            starts = var[f"v4_{kind}_sparse_topk_starts"].np
            lens = var[f"v4_{kind}_sparse_topk_lens"].np
            offsets = var[f"v4_{kind}_sparse_kv_offsets"].np
            topk_base = 0
            kv_base = 0
            token_base = 0
            max_topk = 0
            for seq_idx in range(scheduled_bs):
                seq_start = int(cu_seqlens_q_np[seq_idx])
                seq_end = int(cu_seqlens_q_np[seq_idx + 1])
                token_num = seq_end - seq_start
                if token_num == 0:
                    continue
                start_pos = int(start_pos_per_seq[seq_idx])
                end_pos = start_pos + token_num
                window_topk = self.window_size if start_pos > 0 else min(
                    token_num, self.window_size
                )
                compress_topk = 0
                kv_len = token_num if start_pos == 0 else self.window_size
                if ratio == 4:
                    compress_topk = min(self.index_topk, end_pos // ratio)
                    if start_pos == 0:
                        kv_len = token_num + (token_num + ratio - 1) // ratio
                    else:
                        kv_len = self.window_size + end_pos // ratio
                elif ratio == 128:
                    if start_pos > 0:
                        compress_topk = (start_pos + 1) // ratio
                    else:
                        compress_topk = token_num // ratio
                    if start_pos == 0:
                        kv_len = token_num + (token_num + ratio - 1) // ratio
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

            topk_starts = var[f"v4_{kind}_sparse_topk_starts"].copy_to_gpu(
                total_tokens
            )
            topk_lens = var[f"v4_{kind}_sparse_topk_lens"].copy_to_gpu(total_tokens)
            kv_offsets = var[f"v4_{kind}_sparse_kv_offsets"].copy_to_gpu(total_tokens)
            layouts[ratio_key] = {
                "topk_starts": topk_starts,
                "topk_lens": topk_lens,
                "kv_offsets": kv_offsets,
                "max_topk": max_topk,
            }
        attn_metadata.v4_sparse_layouts = layouts

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
        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        max_seqlen_q = batch.num_spec_step + 1
        positions_np = np.tile(
            np.arange(max_seqlen_q, dtype=np.int32), scheduled_bs
        ) + np.repeat(context_lens - max_seqlen_q, max_seqlen_q)
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
        var["context_lens"].np[:scheduled_bs] = context_lens
        context_lens_gpu = var["context_lens"].copy_to_gpu(scheduled_bs)

        block_tables_gpu = self._populate_block_tables(batch, scheduled_bs)
        state_slot_gpu, state_slot_np = self._populate_state_slot_mapping(
            batch, scheduled_bs, return_cpu=True
        )
        attn_metadata = AttentionMetaData(
            cu_seqlens_q=cu_seqlens_q_gpu,
            cu_seqlens_k=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=int(context_lens.max()) if len(context_lens) else 1,
            min_seqlen_q=0,
            dropout_p=0.0,
            has_cached=False,
            total_kv=int(context_lens.sum()),
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
        self._attach_sparse_layout_metadata(
            attn_metadata,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
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
        self._attach_sparse_layout_metadata(
            attn_metadata,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
        )
        return attn_metadata, positions

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
        gpu = torch.from_numpy(groups_np).to(self.model_runner.device)
        if return_cpu:
            return gpu, groups_np
        return gpu

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        # CUDA Graph capture is disabled for V4 (PR4 scope). Return a stub.
        return AttentionMetaData(
            cu_seqlens_k=None,
            max_seqlen_q=1,
            max_seqlen_k=1,
            min_seqlen_q=0,
            dropout_p=0.0,
            has_cached=False,
            total_kv=bs,
            num_cached_tokens=None,
        )

    # ------------------------------------------------------------------ #
    # Helpers.                                                           #
    # ------------------------------------------------------------------ #

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
