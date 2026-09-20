# SPDX-License-Identifier: MIT
"""CSA2 model projections; cache storage and sparse kernels have separate owners."""

import torch
from aiter import QuantType
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.model_ops.attentions.deepseek_v41.packed_attention import (
    packed_decode,
    packed_prefill,
)
from atom.model_ops.blockscale import (
    dequantize_fp8_weight,
    quantize_fp8,
)
from atom.model_ops.deepseek_v41.compressor import Compressor
from atom.model_ops.deepseek_v41.paged_scoring import score_topk_paged
from atom.model_ops.deepseek_v41.projections import grouped_output_projection
from atom.model_ops.layernorm import DualRMSNormMXFP8, RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.utils import atom_parameter
from atom.model_ops.v4_kernels import (
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_prefill,
)

from .config import AttentionMode
from .layers import native_quant_config


class Indexer(nn.Module):
    def __init__(self, config, spec):
        super().__init__()
        self.spec = spec
        self.heads, self.head_dim = config.index_n_heads, config.index_head_dim
        self.topk = config.index_topk
        self.block_size, self.topk_blocks = (
            config.candidate_block_size,
            config.candidate_topk_blocks,
        )
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.heads * self.head_dim,
            quant_config=native_quant_config(),
        )
        self.weights_proj = ReplicatedLinear(config.hidden_size, self.heads, bias=False)
        if spec.mode == AttentionMode.FULL:
            self.wk = ReplicatedLinear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def project_keys(self, latent, rope, positions):
        """The BF16 index key. Whatever the plane stores it as is the cache's."""
        return rope(self.k_norm(self.wk(latent)), positions)

    @property
    def weights_scale(self):
        return self.head_dim**-0.5 * self.heads**-0.5

    def project_query(self, qr, qr_scale, rope, positions):
        """The index query, left on the grid its reader rounds it to.

        The published model FP4-rounds both sides whatever it holds; the one
        scorer here quantizes the query itself, so rounding first would round
        twice. `qr` arrives quantized, so this GEMM reuses that pair rather
        than quantizing the tensor a second time.
        """
        return rope(
            self.wq_b(qr, x_scale=qr_scale).unflatten(-1, (self.heads, self.head_dim)),
            positions,
        )


class Attention(nn.Module):
    def __init__(self, config, spec):
        super().__init__()
        self.spec = spec
        self.head_dim, self.o_rank = config.head_dim, config.o_lora_rank
        tp_size = get_tp_group().world_size
        self.heads, self.groups = (
            config.num_attention_heads // tp_size,
            config.o_groups // tp_size,
        )
        self.softmax_scale = self.head_dim**-0.5
        self.attn_sink = atom_parameter(torch.empty(self.heads, dtype=torch.float32))
        # Fused [wq_a; wkv], as V4 declares it: one GEMM, and one
        # quantization of the residual instead of two of each.
        self.wqkv_a = MergedReplicatedLinear(
            config.hidden_size,
            [config.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=native_quant_config(),
        )
        # Fused: the norm emits `(qr, qr_scale)` in the one launch, and both
        # readers of `qr` -- this layer's `wq_b` and the indexer's -- take that
        # pair instead of quantizing the same tensor once each.
        self.q_norm = RMSNorm(
            config.q_lora_rank,
            config.rms_norm_eps,
            fused_quant=True,
            quant_config=native_quant_config(),
        )
        self.wq_b = ColumnParallelLinear(
            config.q_lora_rank,
            config.num_attention_heads * self.head_dim,
            quant_config=native_quant_config(),
        )
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        # Both latents are slices of the one `wqkv_a` output and neither norm
        # reads the other's result, so they are one launch rather than two.
        self.qk_norm = DualRMSNormMXFP8(
            self.q_norm, self.kv_norm, f"layers.{spec.layer_id}.qk_norm"
        )
        # wo_a: grouped LoRA. FP8 + e8m0 block scale on disk, BF16 in the
        # grouped einsum. Allocated as a quantized ColumnParallelLinear so both
        # tensors load through the standard FP8 path, then dequantized in
        # `process_weights_after_loading` -- V4's arrangement, unchanged.
        self.wo_a = ColumnParallelLinear(
            config.num_attention_heads * self.head_dim // config.o_groups,
            config.o_groups * self.o_rank,
            bias=False,
            quant_config=native_quant_config(),
        )
        self.wo_b = RowParallelLinear(
            config.o_groups * self.o_rank,
            config.hidden_size,
            quant_config=native_quant_config(),
            reduce_results=True,
        )
        self.compressor = (
            Compressor(
                config.hidden_size, self.head_dim, spec.ratio, config.rms_norm_eps
            )
            if spec.mode == AttentionMode.FULL
            else None
        )
        self.indexer = (
            Indexer(config, spec)
            if spec.mode in (AttentionMode.FULL, AttentionMode.REINDEX)
            else None
        )

    def _compress_batch(self, hidden, cache, step, rope):
        """Every compression boundary in the batch, in one call.

        The cache owns where the main latent lands; what comes back is the
        unrotated latent, which is all the index key needs.
        """
        if not self.spec.ratio or self.compressor is None:
            return
        owner = self.spec.kv_owner
        values, scores = self.compressor.project(hidden)
        latent = cache.compress(owner, self.compressor, values, scores, step, rope)
        if latent is not None:
            index = self.indexer.project_keys(
                latent, rope, step.plans[self.spec.ratio].key_rope_positions_gpu
            )
            cache.write_index(owner, step, index, self.spec.ratio)

    def _select_indices(self, hidden, qr, qr_scale, cache, step, rope):
        """This layer's top-k index rows, out of the paged plane.

        Every query row carries its own bound and its own tile list, so a
        prefill token, a decode token and a drafted token are one shape to the
        scorer and a ragged batch is not a case to it.
        """
        indexer, spec = self.indexer, self.spec
        positions = cache.rope_positions(step)
        source = spec.candidate_source
        selected, chosen = score_topk_paged(
            indexer.project_query(qr, qr_scale, rope, positions)[0],
            indexer.weights_proj(hidden)[0],
            cache.index_units[spec.kv_owner],
            cache.unit_tiles(step, spec.ratio),
            step.visible[spec.ratio],
            topk=indexer.topk,
            weights_scale=indexer.weights_scale,
            candidates=None if source is None else step.candidates[source],
            block_size=indexer.block_size,
            candidate_count=indexer.topk_blocks if spec.produces_candidates else 0,
        )
        step.selected[spec.layer_id] = selected.unsqueeze(0)
        if chosen is not None:
            step.candidates[spec.layer_id] = chosen

    def process_weights_after_loading(self) -> None:
        """Dequantize wo_a to BF16 for the grouped LoRA einsum.

        Copied from V4, minus its gfx950/gfx1250 mxscale branches: this einsum
        path wants BF16. Idempotent -- a checkpoint that already ships wo_a as
        BF16 lands here with nothing to do. Suppressing `quant_type` afterwards
        is what stops `LinearBase.process_weights_after_loading` from applying
        the FP8 CK 16x16 shuffle to a matrix `torch.einsum` then reads, which
        would permute rows inside each block. Load order is parent first, so
        this runs before that hook.
        """
        weight = self.wo_a.weight
        if weight.dtype == torch.bfloat16:
            return
        scale = getattr(self.wo_a, "weight_scale", None)
        if (
            weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            or scale is None
        ):
            return
        # The scale stays in its native e8m0: this helper exists for exactly
        # this weight and reads the grid in that encoding.
        self.wo_a.weight = atom_parameter(
            dequantize_fp8_weight(weight.data, scale.data)
        )
        try:
            delattr(self.wo_a, "weight_scale")
        except AttributeError:
            pass
        # The weight is BF16 now and its scale is gone, so both remaining FP8
        # post-load steps have to be cancelled, not just the shuffle: the other
        # one re-encodes an FP8 weight and its scale from e4m3fn to e4m3fnuz for
        # the AMD parts that use that encoding. It is already off on e4m3fn
        # hardware; clearing it is what makes this correct on the parts where
        # `LinearBase` turned it on.
        self.wo_a.quant_type = QuantType.No
        self.wo_a.need_normalize_e4m3fn_to_e4m3fnuz = False

    def project_qkv(self, hidden):
        """The one GEMM the query and the KV latent both come out of."""
        return torch.split(self.wqkv_a(hidden), self.wqkv_a.output_sizes, dim=-1)

    def forward(self, hidden, cache, step, rope):
        positions = cache.rope_positions(step)
        q_lora, kv_pre = self.project_qkv(hidden)
        qr, qr_scale, kv_normed = self.qk_norm(q_lora, kv_pre)
        query, raw_kv = rope.pair(
            self.wq_b(qr, x_scale=qr_scale).unflatten(-1, (self.heads, self.head_dim)),
            kv_normed,
            positions,
        )
        # Decode consumes only the stored window; do not materialize an unused
        # BF16 copy when the persistent cache is packed.
        kv = (
            None
            if cache.packed and step.decode
            else quantize_fp8(raw_kv, dequantize=True)
        )
        window_kv = quantize_fp8(raw_kv) if cache.packed else kv
        self._compress_batch(hidden, cache, step, rope)
        if self.indexer is not None:
            self._select_indices(hidden, qr, qr_scale, cache, step, rope)
        prefix, prefix_indptr, extend, extend_indptr = cache.attention_indices(
            self.spec, step
        )
        flat_query = query.flatten(0, 1)
        if step.decode:
            cache.write_window(self.spec.layer_id, window_kv, step)
            decode = packed_decode if cache.packed else sparse_attn_v4_paged_decode
            output = decode(
                flat_query,
                cache.pool,
                prefix,
                prefix_indptr,
                self.attn_sink,
                self.softmax_scale,
            )
        else:
            prefill = packed_prefill if cache.packed else sparse_attn_v4_paged_prefill
            output = prefill(
                flat_query,
                cache.pool,
                prefix,
                prefix_indptr,
                kv.flatten(0, 1),
                extend,
                extend_indptr,
                self.attn_sink,
                self.softmax_scale,
                out=flat_query,
            )
            # Preserve the prior ring until every query has consumed its prefix.
            cache.write_window(self.spec.layer_id, window_kv, step)
        output = output.view_as(query)
        output = (
            rope(output, positions, inverse=True)
            .unflatten(-2, (self.groups, -1))
            .flatten(-2)
        )
        grouped_weight = self.wo_a.weight.view(self.groups, self.o_rank, -1)
        output = grouped_output_projection(output, grouped_weight)
        return self.wo_b(output.flatten(-2))
