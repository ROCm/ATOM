"""The sparse draft q a probability-ratio verify reads.

A drafter that samples its tokens keeps, for each draft row, the top of the
distribution q it drew from, and the verify (`ratio_verify`) tests the
row's token against that q. The store (`SparseQStore`), per draft row
R = q_row_base[b] + position:
  ids   [R, :n]  vocab ids, strictly ascending (so distinct): the verify
                 binary-searches them, and a row out of order would lose
                 entries there and with them the target law;
  probs [R, :n]  their q, > 0 and summing to 1 in one fixed-order reduction;
  meta  [R]      (n, xslot): how many are stored, and the slot of the token
                 the draft emitted. Slots >= n are never read.
q is 0 for every id not stored. The sampled DSpark draft
(`dspark_markov_sample`) writes it.
"""

from typing import NamedTuple

import torch

# How many entries of each draft row's q the store keeps (K). The draft samples
# from exactly the stored entries, renormalized, so any K is exact; a smaller
# one only gives up q's tail mass, i.e. some acceptance. A drafter clamps it to
# the vocab size, which stores everything drawable: the full-q baseline for
# acceptance-length studies, not for throughput, since one program per row
# normalizes and draws from the stored entries.
Q_WIDTH = 1024


def store_width(vocab_size: int) -> int:
    """Entries per stored row for a draft over ``vocab_size`` ids: K, never
    wider than the vocab. The store and the verify's warmup both size by it."""
    return min(Q_WIDTH, vocab_size)


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
