# SPDX-License-Identifier: MIT
"""CSA2 non-overlapping latent pooling; caller owns the incomplete group."""

from dataclasses import dataclass

import torch
from torch import nn

from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import MergedReplicatedLinear
from atom.model_ops.v4_kernels import fused_compress_attn
from atom.model_ops.v4_kernels.state_writes import update_compressor_states


@dataclass(frozen=True)
class CompressorTail:
    values: torch.Tensor
    scores: torch.Tensor


def compress_batch(cache, owner, compressor, values, scores, step, rope, *, scatter):
    """Every compression boundary in the batch, through V4's fused kernel.

    One driver for both caches: the indexer's top-k is discrete, so two
    implementations agreeing to a BF16 ulp still select different rows. Each
    cache supplies only its addressing, as `scatter`.

    `scatter` is the `(pages, block_tables)` the kernel writes the rotated
    latent into, or `None` for a pool whose layout it cannot write -- the
    packed one interleaves FP4 with its scales, so the caller gets the rotated
    value echoed back and packs it itself. Rotating a second time in torch
    would round differently, and the FP4 grid turns that into level flips.

    Returns `(latent, rotated)`: the post-norm PRE-RoPE latent the index key
    projects from, and the echo when asked. Both `None` when no request
    crossed a boundary.
    """
    ratio = compressor.ratio
    plan = step.plans[ratio]
    kv_state, score_state = cache.compress_state(owner)
    head_dim = values.shape[-1]
    pages, block_tables = scatter or (None, None)
    # Ratio 1 has no gate, and a one-element softmax weighs 1 whatever the
    # score is, so the gate input only has to be finite.
    gate = values if scores is None else scores
    # One row per plan row, not per boundary the batch happens to cross: that
    # is the kernel's grid, and a decode plan is cut to a capacity a capture
    # can record. The sentinel tail is projected like any other row and then
    # skipped by the writers, which is cheaper than a host-side count.
    capacity = plan.compress_plan_gpu.shape[0]
    latent = rotated = None
    if capacity:
        # BF16: what the index key's BF16 `wk` consumes.
        latent = torch.empty(
            capacity, head_dim, dtype=torch.bfloat16, device=values.device
        )
        rotated = torch.empty_like(latent) if scatter is None else None
        fused_compress_attn(
            kv_in=values[0],
            score_in=gate[0],
            kv_state=kv_state,
            score_state=score_state,
            plan=plan,
            state_slot_mapping=step.slots,
            ape=compressor.ape,
            rms_weight=compressor.norm.weight,
            rms_eps=compressor.norm.eps,
            cos_cache=rope.cos_cache,
            sin_cache=rope.sin_cache,
            kv_cache=pages,
            block_tables=block_tables,
            k_per_block=0 if pages is None else pages.shape[1],
            overlap=False,
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=rope.cos_cache.shape[-1] * 2,
            quant_mode="none",
            latent_out=latent,
            rotated_out=rotated,
            prefix=f"csa2.compress_{owner}",
        )
    # After the boundary kernel, never before: that reads the ring as of the
    # previous forward, this overwrites it for the next. Unconditional -- a
    # forward crossing no boundary still owes the ring its projections, or the
    # round that does cross one reads a position nobody wrote.
    update_compressor_states(
        values[0],
        gate[0],
        compressor.ape,
        kv_state,
        score_state,
        write_plan=plan.write_plan_gpu,
        state_slot_mapping=step.slots,
        ratio=ratio,
        overlap=False,
        prefix=f"csa2.compress_state_{owner}",
    )
    if latent is None:
        return None, None
    return latent.unsqueeze(0), None if rotated is None else rotated.unsqueeze(0)


class Compressor(nn.Module):
    def __init__(self, hidden_size, head_dim, ratio, eps):
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError("CSA2 compressor ratio must be 1 or 2")
        self.ratio = ratio
        # Fused [wkv; wgate], as V4 declares it. A ratio-1 layer ships no
        # `wgate`, so there the matrix is the pooling half alone.
        self.wkv_gate = MergedReplicatedLinear(
            hidden_size,
            [head_dim] * (1 if ratio == 1 else 2),
            bias=False,
            quant_config=None,
        )
        self.norm = RMSNorm(head_dim, eps)
        # V4 adds a learned position encoding to the gate before the softmax;
        # CSA2 does not, and the batched kernel takes it as a tensor rather
        # than a flag. Zeros say the same thing without a second code path.
        self.register_buffer(
            "ape", torch.zeros(ratio, head_dim, dtype=torch.float32), persistent=False
        )

    def project(self, x):
        """Per-token projections; ratio 1 pools nothing, so it has no gate.

        `otype` is load-bearing above ratio 1: the weights are BF16, and
        without it the accumulator is rounded to BF16 before the pool that
        the published model specifies in FP32 ever sees it.
        """
        if self.ratio == 1:
            return self.wkv_gate(x), None
        # Zero-copy halves; both readers take a row stride and need only unit
        # stride along the head dimension.
        return self.wkv_gate(x, otype=torch.float32).chunk(2, dim=-1)

    def pool(self, values, scores, start_position, tail=None, *, dtype):
        if self.ratio == 1:
            return self.norm(values), None
        remainder = start_position % self.ratio
        if remainder:
            expected = (values.shape[0], remainder, values.shape[-1])
            if (
                tail is None
                or tail.values.shape != expected
                or tail.scores.shape != expected
            ):
                raise ValueError(
                    "Incomplete compressor group is missing or has the wrong shape"
                )
            values = torch.cat((tail.values, values), dim=1)
            scores = torch.cat((tail.scores, scores), dim=1)
        elif tail is not None:
            raise ValueError("Unexpected compressor tail at a group boundary")
        cutoff = values.shape[1] // self.ratio * self.ratio
        next_tail = None
        if cutoff < values.shape[1]:
            next_tail = CompressorTail(
                values[:, cutoff:].clone(), scores[:, cutoff:].clone()
            )
        if cutoff == 0:
            return None, next_tail
        grouped_values = values[:, :cutoff].unflatten(1, (-1, self.ratio))
        grouped_scores = scores[:, :cutoff].unflatten(1, (-1, self.ratio))
        latent = (grouped_values * grouped_scores.softmax(dim=2)).sum(dim=2)
        return self.norm(latent.to(dtype)), next_tail

    def forward(self, x, start_position, tail=None):
        return self.pool(*self.project(x), start_position, tail, dtype=x.dtype)
