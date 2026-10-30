"""The sparse draft q a probability-ratio verify reads.

A drafter that samples its tokens keeps, for each draft row, the top of the
distribution q it drew from, and the verify (`rejection_sampler`) tests the
row's token against that q. The store (`SparseQStore`), per draft row
R = q_row_base[b] + position:
  ids   [R, :n]  distinct vocab ids;
  probs [R, :n]  their q, > 0 and summing to 1 in one fixed-order reduction;
  meta  [R]      (n, xslot): how many are stored, and the slot of the token
                 the draft emitted. Slots >= n are never read.
q is 0 for every id not stored. The sampled DSpark draft
(`dspark_markov_sample`) writes it.
"""

from typing import NamedTuple

import torch


class SparseQStore(NamedTuple):
    """The persistent draft q, ``rows`` draft rows of up to ``width`` entries;
    see the module notes for the contract."""

    ids: torch.Tensor  # [rows, width] int32
    probs: torch.Tensor  # [rows, width] fp32
    meta: torch.Tensor  # [rows, 2] int32: (n, xslot)

    @classmethod
    def allocate(cls, rows: int, width: int, device) -> "SparseQStore":
        return cls(
            torch.empty((rows, width), dtype=torch.int32, device=device),
            torch.empty((rows, width), dtype=torch.float32, device=device),
            torch.empty((rows, 2), dtype=torch.int32, device=device),
        )

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self)


class RatioDraftQ(NamedTuple):
    """The q a ratio verify reads: the draft's sparse store, plus a dense
    ``[>= batch_size, V]`` fp32 scratch a rejecting request's q row is spread
    into for the residual. Its contents are overwritten; nothing else may need
    them across the verify."""

    store: SparseQStore
    dense: torch.Tensor
