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
PR3-pre2c-A (this revision): adds SWA buffer migration. Classical KV cache
                   for compressed entries (Compressor.kv_cache /
                   Indexer.kv_cache) is still register_buffer-backed on the
                   model and slot-major (1 slot at index 0). PR3-pre2c-B will
                   move it to the block_table.
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
    (slot_mapping, block_tables, cu_seqlens). PR3-pre2c-A keeps `block_size = 1`
    (token-level placeholder) because the classical KV pool (Compressor/Indexer
    .kv_cache) is still register_buffer on the model. Block_size will be raised
    to 128 (lcm(m, m')) in PR3-pre2c-B alongside the block_table migration.
    """

    block_size = 1

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

        # Compressor state shape: [coff * ratio, coff * head_dim], fp32.
        # CSA: ratio=4, overlap=True -> coff=2 -> [8, 2*head_dim]
        # HCA: ratio=128, overlap=False -> coff=1 -> [128, head_dim]
        self.csa_main_state_shape = (2 * 4, 2 * self.head_dim)
        self.csa_idx_state_shape = (2 * 4, 2 * self.index_head_dim)
        self.hca_main_state_shape = (128, self.head_dim)

        self._state_dtype = torch.float32  # fp32 required for softmax-pool
        self._swa_dtype = torch.bfloat16  # SWA window matches KV dtype

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
        """Phantom per-token bytes for V4's planned classical KV cache.

        PR3-pre2c-A still register_buffer's the compressed KV pools on the
        model (Compressor/Indexer.kv_cache). PR3-pre2c-B will migrate them
        into block-table-backed pools, at which point this returns the
        true per-V4-block cost. For now we report the planned per-token
        cost so ModelRunner's pool-sizing math (per_req_cache_equiv_blocks
        and num_kvcache_blocks) stays consistent.
        """
        bf16 = 2
        # k1=lcm/m=32 CSA entries per layer per V4 block of 128 original tokens.
        # k2=lcm/m'=1 HCA entry per layer per V4 block.
        csa_main_per_block = 32 * self.head_dim * bf16
        csa_idx_per_block = 32 * self.index_head_dim * bf16
        hca_main_per_block = 1 * self.head_dim * bf16
        bytes_per_v4_block = (
            len(self.csa_layers) * (csa_main_per_block + csa_idx_per_block)
            + len(self.hca_layers) * hca_main_per_block
        )
        return bytes_per_v4_block // 128  # per-token (block_size=1 in pre2c-A)

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
        """Bind V4 modules' state-cache views (no KVCacheTensor returned).

        Called by ModelRunner.allocate_kv_cache() for every nn.Module:
          - V4 Compressor: bind kv_state + score_state (per_req_cache pool).
          - V4 Attention: bind swa_kv (per_req_cache pool).

        Returns None always — V4's classical KV cache (compressed entries) is
        still register_buffer on the model in PR3-pre2c-A, so there are no
        KVCacheTensor entries to register from this builder.
        """
        # Local imports to avoid circular dependency at module load time.
        from atom.models.deepseek_v4 import (
            Compressor as _V4Compressor,
            DeepseekV4Attention as _V4Attention,
        )

        runner = self.model_runner

        if isinstance(module, _V4Attention):
            # Attention.swa_kv — every layer (one slice of v4_swa_kv per layer).
            module.swa_kv = runner.v4_swa_kv[module.layer_id]
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
            elif ratio == 4:
                pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_csa_main_kv_state[pos]
                module.score_state = runner.v4_csa_main_score_state[pos]
            elif ratio == 128:
                pos = self.layer_id_to_hca_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_hca_main_kv_state[pos]
                module.score_state = runner.v4_hca_main_score_state[pos]
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

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        """V4-style decode prep: populates positions like AiterBackend.

        V4 forward currently consumes only `positions[0]` to drive its
        KV ring-buffer index; once PR3-main lands, block_tables / slot_mapping
        will also be consumed.
        """
        import numpy as np

        var = self.model_runner.forward_vars
        scheduled_bs = batch.total_seqs_num_decode
        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        max_seqlen_q = batch.num_spec_step + 1
        positions = np.tile(
            np.arange(max_seqlen_q, dtype=np.int32), scheduled_bs
        ) + np.repeat(context_lens - max_seqlen_q, max_seqlen_q)
        sum_scheduled_tokens = batch.total_tokens_num_decode
        var["positions"].np[:sum_scheduled_tokens] = positions
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        attn_metadata = AttentionMetaData(
            cu_seqlens_k=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=int(context_lens.max()) if len(context_lens) else 1,
            min_seqlen_q=0,
            dropout_p=0.0,
            has_cached=False,
            total_kv=int(context_lens.sum()),
            num_cached_tokens=None,
        )
        return attn_metadata, positions

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
