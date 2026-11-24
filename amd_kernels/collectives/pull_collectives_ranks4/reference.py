# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of the 4-rank pull collectives (root README, rule 3).

``xs[p]`` is rank p's input (p = 0..3), ``r`` this rank, a row block is a
quarter of the rows:

- all-gather (``AllGatherComm.all_gather(outs, ins)``): each output is
  ``cat(in_0, in_1, in_2, in_3)`` along the rows, the bytes as they are.
- all-to-all: row block p of the output is row block r of ``x_p``.
- unpacking all-to-all (``FencedAllGatherComm.all_to_all_unpack_shuffled``, d 2048):
  the all-to-all of packed rows ``[4 rows, d + d/32]`` (``d`` code bytes, then
  ``d / 32`` scale bytes per row), unpacked on the way: ``codes[i, p d : (p + 1) d]``
  / ``scales[i, p d / 32 : (p + 1) d / 32]`` are the code / scale bytes of row i of
  the block that came from rank p (the layout of the sequence-parallel o_proj
  input: the whole o_proj input of this rank's rows); the kernel writes the scales
  in the GEMM's shuffled layout.
- reduce-scatter (``SlotsReduceScatterComm.reduce_scatter(x)``):
  ``out = bf16((((+0 + b_0) + b_1) + b_2) + b_3)`` with ``b_p`` row block r of
  ``x_p`` in fp32: rank order from +0 (a sum of -0.0 is +0.0), denormals kept, one
  RNE rounding. RCCL's ring adds in another order (measured: up to 0.25 apart), so
  the two are not bitwise.

The MoE combine variant is ``slots4_reduce_scatter_reference`` below. Its
norm/quant variant applies ``norm/gemma_add_rmsnorm_h6144/reference.py`` steps
2-4 to this rank's shard, with row-major codes and shuffled scales.
The shard residual and rank sum are bitwise; the norm follows that reference's
FP64 tolerances and matches the existing FlyDSL norm's output bits.

Tolerances: none for the communication, every collective is bit for bit.
"""

import torch

WORLD = 4


def all_gather_reference(ins: list[torch.Tensor]) -> torch.Tensor:
    """The 4 ranks' ``[rows, ...]`` inputs -> ``[4 rows, ...]``."""
    assert len(ins) == WORLD
    return torch.cat(ins)


def all_to_all_reference(xs: list[torch.Tensor], rank: int) -> torch.Tensor:
    """Rank ``rank``'s output of the all-to-all of the 4 ranks' ``xs``."""
    return torch.cat([x.reshape(WORLD, -1)[rank] for x in xs]).view(xs[0].dtype).view(
        xs[0].shape
    )


def unpack_reference(
    recv: torch.Tensor, rows: int, d: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[4 rows, d + d / 32]`` received packed rows (block p from rank p) ->
    (codes ``[rows, 4 d]``, scales ``[rows, 4 d / 32]``), uint8."""
    blocks = recv.view(torch.uint8).view(WORLD, rows, d + d // 32).transpose(0, 1)
    return (
        blocks[:, :, :d].reshape(rows, WORLD * d),
        blocks[:, :, d:].reshape(rows, WORLD * (d // 32)),
    )


def reduce_scatter_reference(xs: list[torch.Tensor], rank: int) -> torch.Tensor:
    """Rank ``rank``'s ``[rows, H]`` bf16 sums of the 4 ranks' ``[4 rows, H]`` bf16
    partials (module docstring)."""
    rows = xs[0].shape[0] // WORLD
    blk = slice(rank * rows, (rank + 1) * rows)
    acc = torch.zeros((rows, *xs[0].shape[1:]), dtype=torch.float32, device=xs[0].device)
    for x in xs:
        acc = acc + x[blk].float()
    return acc.bfloat16()


def slots4_reduce_scatter_reference(
    slots: list[torch.Tensor], shared: list[torch.Tensor], rank: int
) -> torch.Tensor:
    """Weighted BF16 [M,4,H] slots -> BF16 [M/4,H].

    Local order: (((s0+s1)+s2)+s3)+shared in FP32, one BF16 RNE rounding.
    Rank order: +0, rank0, rank1, rank2, rank3 in FP32, one BF16 RNE.
    Denormals are retained, as in the existing pull reduce-scatter.
    Exact bytes are the contract; a reduce-then-add-shared path rounds twice
    locally and therefore has a different numerical contract.
    """
    xs = []
    for s, h in zip(slots, shared):
        acc = s[:, 0].float()
        for k in range(1, 4):
            acc = acc + s[:, k].float()
        xs.append((acc + h.float()).bfloat16())
    return reduce_scatter_reference(xs, rank)


def norm_all_gather_reference(xs, residuals, weights, eps=1e-6):
    """Dense SP norm + all-gather contract, four BF16 [rows,6144] shards.

    Each shard follows norm/reference.py's FP64 add/RMSNorm contract. Quantize
    the FP32 norm result (not its BF16 rounding) at every M. The exact shader
    oracle is gemma_add_rmsnorm_mxfp8, untiled, with its eight-wave SSQ tree;
    old SP's sixteen-wave SSQ tree may differ in the final norm bits (gate 2).
    Codes concatenate row-major. Scale bytes concatenate four independent
    AITER-shuffled blocks; shards must have a multiple of 32 rows.
    Returns FP64 norm shards and BF16 residual shards for compare_mxfp8;
    tests additionally require byte equality to the standalone shader oracle.
    """
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import add_rmsnorm_reference
    assert len(xs) == len(residuals) == len(weights) == WORLD
    return [add_rmsnorm_reference(x, r, w, eps) for x,r,w in zip(xs,residuals,weights)]
