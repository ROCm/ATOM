# SPDX-License-Identifier: MIT
"""CSA2 model projections; cache storage and sparse kernels have separate owners."""

import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.model_ops.blockscale import quantize_fp4, quantize_fp8
from atom.model_ops.deepseek_v41.compressor import Compressor
from atom.model_ops.deepseek_v41.indexer import select_indices
from atom.model_ops.deepseek_v41.normalization import FusedRMSNorm, RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.v4_kernels import (
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_prefill,
)

from .config import AttentionMode, IndexTieBreak
from .layers import native_quant_config, reduce_output


class Indexer(nn.Module):
    def __init__(self, config, spec):
        super().__init__()
        self.spec = spec
        self.heads, self.head_dim = config.index_n_heads, config.index_head_dim
        self.topk = config.index_topk
        self.tie_break = IndexTieBreak(config.index_topk_tie_break)
        self.block_size, self.topk_blocks = (
            config.candidate_block_size,
            config.candidate_topk_blocks,
        )
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.heads * self.head_dim,
            quant_config=native_quant_config(),
        )
        self.weights_proj = nn.Linear(
            config.hidden_size, self.heads, bias=False, dtype=torch.bfloat16
        )
        if spec.mode == AttentionMode.FULL:
            self.wk = nn.Linear(
                config.head_dim, self.head_dim, bias=False, dtype=torch.bfloat16
            )
            self.k_norm = FusedRMSNorm(self.head_dim, config.rms_norm_eps)

    def project_keys(self, latent, rope, positions):
        key = self.k_norm(self.wk(latent))
        return quantize_fp4(rope(key, positions), dequantize=True)

    def forward(self, hidden, qr, keys, rope, step, candidates=None):
        positions = torch.arange(
            step.position, step.position + step.length, device=hidden.device
        )
        query = self.wq_b(qr).unflatten(-1, (self.heads, self.head_dim))
        query = quantize_fp4(rope(query, positions), dequantize=True)
        weights = self.weights_proj(hidden) * (self.head_dim**-0.5 * self.heads**-0.5)
        return select_indices(
            query,
            weights,
            keys,
            (positions + 1) // self.spec.ratio,
            topk=self.topk,
            candidate_blocks=candidates,
            make_candidates=self.spec.layer_id == self.spec.candidate_owner,
            block_size=self.block_size,
            topk_blocks=self.topk_blocks,
            tie_break=self.tie_break,
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
        self.attn_sink = nn.Parameter(
            torch.empty(self.heads, dtype=torch.float32), requires_grad=False
        )
        self.wq_a = ReplicatedLinear(
            config.hidden_size, config.q_lora_rank, quant_config=native_quant_config()
        )
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            config.q_lora_rank,
            config.num_attention_heads * self.head_dim,
            quant_config=native_quant_config(),
        )
        self.wkv = ReplicatedLinear(
            config.hidden_size, self.head_dim, quant_config=native_quant_config()
        )
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        # wo_a is used as grouped BF16 weights, not a dense linear forward.
        self.wo_a = nn.Linear(
            config.num_attention_heads * self.head_dim // config.o_groups,
            self.groups * self.o_rank,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.wo_b = RowParallelLinear(
            config.o_groups * self.o_rank,
            config.hidden_size,
            quant_config=native_quant_config(),
            reduce_results=False,
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

    def _update_global(self, hidden, qr, cache, step, rope):
        if self.spec.ratio:
            owner = self.spec.kv_owner
            count = (step.position + step.length) // self.spec.ratio
            if self.compressor is not None:
                latent, tail = self.compressor(
                    hidden, step.position, cache.read_tail(owner, step.position)
                )
                cache.write_tail(owner, tail)
                if latent is not None:
                    begin = step.position // self.spec.ratio
                    end = begin + latent.shape[1]
                    latent_positions = (
                        torch.arange(begin, end, device=hidden.device) * self.spec.ratio
                    )
                    # Derive index keys before the main latent is rotated in place.
                    index = self.indexer.project_keys(latent, rope, latent_positions)
                    main = quantize_fp4(
                        rope(latent, latent_positions),
                        group_size=16,
                        scale_dtype=torch.float8_e4m3fn,
                        dequantize=True,
                    )
                    cache.write_global(owner, begin, main, index)
            if self.indexer is not None:
                candidate_owner = self.spec.candidate_owner
                candidates = (
                    step.candidates[candidate_owner]
                    if candidate_owner is not None
                    and candidate_owner != self.spec.layer_id
                    else None
                )
                selected, candidates_out = self.indexer(
                    hidden, qr, cache.index_keys(owner, count), rope, step, candidates
                )
                step.indices[self.spec.layer_id] = selected
                if candidates_out is not None:
                    step.candidates[self.spec.layer_id] = candidates_out

    def forward(self, hidden, cache, step, rope):
        positions = cache.rope_positions(step)
        qr = self.q_norm(self.wq_a(hidden))
        query = rope(
            self.wq_b(qr).unflatten(-1, (self.heads, self.head_dim)), positions
        )
        kv = quantize_fp8(
            rope(self.kv_norm(self.wkv(hidden)), positions), dequantize=True
        )
        for request_cache, request_step, rows in cache.requests(step):
            self._update_global(
                hidden[:, rows], qr[:, rows], request_cache, request_step, rope
            )
        prefix, prefix_indptr, extend, extend_indptr = cache.attention_indices(
            self.spec, step
        )
        flat_query = query.flatten(0, 1)
        if step.decode:
            cache.write_window(self.spec.layer_id, kv, step)
            output = sparse_attn_v4_paged_decode(
                flat_query,
                cache.pool,
                prefix,
                prefix_indptr,
                self.attn_sink,
                self.head_dim**-0.5,
            )
        else:
            output = sparse_attn_v4_paged_prefill(
                flat_query,
                cache.pool,
                prefix,
                prefix_indptr,
                kv.flatten(0, 1),
                extend,
                extend_indptr,
                self.attn_sink,
                self.head_dim**-0.5,
                out=flat_query,
            )
            # Preserve the prior ring until every query has consumed its prefix.
            cache.write_window(self.spec.layer_id, kv, step)
        output = output.view_as(query)
        output = (
            rope(output, positions, inverse=True)
            .unflatten(-2, (self.groups, -1))
            .flatten(-2)
        )
        grouped_weight = self.wo_a.weight.view(self.groups, self.o_rank, -1)
        output = torch.einsum("bsgd,grd->bsgr", output, grouped_weight)
        return reduce_output(self.wo_b(output.flatten(-2)))
