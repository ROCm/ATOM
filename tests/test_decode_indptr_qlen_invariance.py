# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A decode token's SWA / CSA / HCA metadata must not depend on its batch.

`test_decode_indptr_build.py` pins the builder to its reference and to a stated
answer. Neither settles the property this file is about, because both compare a
batch against a number rather than a batch against another batch.

The property: the token at absolute position `p` owns one SWA prefix length,
one CSA visibility and one HCA group count, and all three follow from `p`. They
may not move when the same token arrives on a step with a different **query
length**. Speculative decode makes qlen vary step to step by acceptance, so a
count that tracks qlen tracks something the model never authorised.

Which token is probed is the whole design. The rule this replaced took the
sequence's `ctx // ratio`, and `ctx` is the position of the group's LAST token
plus one -- so on that rule the last token is the one that stays right and every
EARLIER token inherits a count belonging to positions it may not see. A fixture
that pins the group's end and varies its start therefore probes the one token
both rules agree on, and passes either way. So `_group` holds the probed token
fixed and varies how many drafts trail BEHIND it, which is what a change in
acceptance actually does to a sequence.
"""

import itertools

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "drives a Triton builder on real tensors; needs a GPU",
        allow_module_level=True,
    )

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import CSA_RATIO, HCA_RATIO
from atom.model_ops.v4_kernels.paged_decode_indices import (
    build_v4_paged_decode_indptr,
)

DEV = "cuda"
WIN = 128
INDEX_TOPK = 1024  # above every visibility here, so it never masks a difference


def _build(groups, t_pad=None):
    """Run the builder over `groups`, a list of per-sequence position lists.

    Returns `{position: (swa, csa_head, hca_head, visibility)}` for every token,
    so two batches can be compared token by token however they were packed.
    """
    positions = [p for g in groups for p in g]
    batch_id = [i for i, g in enumerate(groups) for _ in g]
    t = len(positions)
    t_pad = t if t_pad is None else t_pad
    positions = positions + [0] * (t_pad - t)
    batch_id = batch_id + [-1] * (t_pad - t)

    out = {
        k: torch.full((t_pad + 1,), -7, dtype=torch.int32, device=DEV)
        for k in ("swa_indptr", "csa_indptr", "hca_indptr")
    }
    out["csa_n_committed_per_token"] = torch.zeros(t_pad, dtype=torch.int32, device=DEV)
    build_v4_paged_decode_indptr(
        batch_id_per_token=torch.tensor(batch_id, dtype=torch.int32, device=DEV),
        positions=torch.tensor(positions, dtype=torch.int64, device=DEV),
        T_pad=t_pad,
        win=WIN,
        index_topk=INDEX_TOPK,
        **out,
    )
    span = {}
    for name in ("swa", "csa", "hca"):
        d = out[f"{name}_indptr"]
        span[name] = (d[1:] - d[:-1]).tolist()
    vis = out["csa_n_committed_per_token"].tolist()
    return {
        positions[i]: (
            span["swa"][i],
            span["csa"][i] - span["swa"][i],
            span["hca"][i] - span["swa"][i],
            vis[i],
        )
        for i in range(t)
    }


def _group(probe, trailing):
    """One sequence whose FIRST token is `probe`, with `trailing` drafts after."""
    return [list(range(probe, probe + trailing + 1))]


# Probes chosen so the drafts behind them cross a boundary of one ratio or the
# other: 128 is the HCA group size, 4 the CSA one. 126 is the sharp one -- its
# own count is 0 HCA groups, and any group of 2+ pushes the sequence past 128.
PROBES = [1, 2, 3, 126, 127, 128, 129, 253, 254, 255, 1000]
TRAILING = [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("probe", PROBES)
def test_a_token_reads_the_same_counts_however_many_drafts_trail_it(probe):
    seen = {n: _build(_group(probe, n))[probe] for n in TRAILING}
    assert len(set(seen.values())) == 1, (
        f"position {probe} changed its (swa, csa, hca, visibility) with the "
        f"number of trailing drafts: {seen}"
    )


@pytest.mark.parametrize(
    "probe,trailing", itertools.product([126, 128, 254], [0, 3, 5])
)
def test_a_token_reads_the_same_counts_next_to_any_batch_mate(probe, trailing):
    """Nor does it care what the other sequences on the step look like."""
    mine = _group(probe, trailing)
    alone = _build(mine)[probe]
    for mate, mate_trailing in ((1000, 0), (1000, 5), (5, 2)):
        mates = _group(mate, mate_trailing)
        assert _build(mine + mates)[probe] == alone, "a batch-mate moved it"
        assert _build(mates + mine)[probe] == alone, "so did the packing order"


def test_padding_the_step_to_a_captured_size_moves_nothing():
    """A CUDAGraph step is padded to a capture size; the real tokens' counts
    must equal those of the unpadded step they stand for."""
    groups = _group(126, 4) + _group(1000, 1)
    tight = _build(groups)
    for pad in (8, 16, 64):
        assert _build(groups, t_pad=pad) == tight, f"t_pad={pad} changed a count"


@pytest.mark.parametrize("probe", PROBES)
def test_the_slice_a_token_owns_is_exactly_what_it_uses(probe):
    """Head and tail must tile each slice: compress section plus SWA prefix, no
    gap. A gap is uninitialised memory the attention kernel goes on to read."""
    swa, csa_head, hca_head, vis = _build(_group(probe, 5))[probe]
    assert swa == min(probe + 1, WIN)
    assert vis == (probe + 1) // CSA_RATIO
    assert csa_head == min(vis, INDEX_TOPK)
    assert hca_head == (probe + 1) // HCA_RATIO


def test_the_fixtures_can_tell_the_two_rules_apart():
    """Teeth check for everything above.

    The rule this file guards against is `ctx // ratio`, the group's last
    position plus one. If no fixture makes that disagree with the per-token
    count, the invariance tests would hold on either rule and prove nothing.
    Assert some fixture does disagree, and that the builder picks the token's.
    """
    disagree = [
        (probe, n)
        for probe in PROBES
        for n in TRAILING
        if (probe + 1) // HCA_RATIO != (probe + n + 1) // HCA_RATIO
    ]
    assert disagree, "no fixture separates the per-token rule from the per-sequence one"
    for probe, n in disagree:
        assert (
            _build(_group(probe, n))[probe][2] == (probe + 1) // HCA_RATIO
        ), f"probe {probe} with {n} trailing drafts took the sequence's count"
