# SPDX-License-Identifier: MIT
"""How `DPMetadata.make` arrives at the per-rank token counts.

Three ways in, and the choice is a correctness question, not a performance one:
the table sizes every DP collective in the forward, so a table that does not
describe what the ranks will actually run shows up as a fixed-size collective
posted at mismatched heights -- a hang on eight ranks, nothing here.

No aiter and no DP group: the all_reduce lives behind
`DPMetadata.num_tokens_across_dp`, which these tests replace. That is also what
makes "did it ask the group" observable at all.
"""

import types

import pytest
import torch

from atom.utils.forward_context import DPMetadata


def _cfg(dp_size=4, dp_rank=1):
    return types.SimpleNamespace(data_parallel_size=dp_size, data_parallel_rank=dp_rank)


@pytest.fixture
def no_group(monkeypatch):
    """Make the ask itself fail, rather than count calls.

    A version that asks would pass against a stub that answers.
    """

    def _refuse(*a, **k):
        raise AssertionError("asked the group for a height it was told")

    monkeypatch.setattr(DPMetadata, "num_tokens_across_dp", staticmethod(_refuse))


def test_a_unified_height_is_written_out_rather_than_reduced(no_group):
    """Every rank runs `num_tokens`, so every rank can write the whole table."""
    md = DPMetadata.make(_cfg(dp_size=4), 336, unified=True)

    assert md.get_sizes_across_dp() == [336, 336, 336, 336]
    assert md.max_tokens_across_dp == 336
    assert md.cu_tokens_across_dp_cpu.tolist() == [336, 672, 1008, 1344]


def test_a_height_not_declared_unified_is_still_discovered(monkeypatch):
    """The default is unchanged: no table and no claim means one all_reduce.

    Its answer is honoured whole -- the reduction, not this rank's own count,
    is what the derived sizes come from.
    """
    asked = {}

    def _reduce(num_tokens, dp_size, dp_rank):
        asked["args"] = (num_tokens, dp_size, dp_rank)
        return torch.tensor([10, 336, 7, 1], dtype=torch.int32)

    monkeypatch.setattr(DPMetadata, "num_tokens_across_dp", staticmethod(_reduce))

    md = DPMetadata.make(_cfg(dp_size=4, dp_rank=1), 336)

    assert asked["args"] == (336, 4, 1)
    assert md.get_sizes_across_dp() == [10, 336, 7, 1]
    assert md.max_tokens_across_dp == 336


def test_a_supplied_table_and_a_unified_claim_cannot_both_be_given(no_group):
    """Handing over the table already says what every rank runs.

    Saying it again can only agree redundantly or disagree silently, and the
    silent case is the one that costs eight ranks.
    """
    table = torch.tensor([336, 336, 336, 336], dtype=torch.int32)

    with pytest.raises(AssertionError):
        DPMetadata.make(_cfg(dp_size=4), 336, table, unified=True)

    # ...and the table alone is still taken as given.
    md = DPMetadata.make(_cfg(dp_size=4), 336, table)
    assert md.get_sizes_across_dp() == [336, 336, 336, 336]


def test_a_unified_claim_is_not_checked_against_the_group(no_group):
    """It cannot be, and the pass that makes it has to know that.

    `make` sees one rank, and the entry it asserts is the one a repeated fill
    satisfies by construction -- so a caller declaring uniformity off a
    rank-local number gets a well-formed table describing a group that does not
    exist. Pinned because the guarantee lives entirely in the callers.
    """
    md = DPMetadata.make(_cfg(dp_size=4, dp_rank=3), 7, unified=True)

    assert md.get_sizes_across_dp() == [7, 7, 7, 7]
