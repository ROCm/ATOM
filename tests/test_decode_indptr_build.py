# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gate for the V4 decode indptr builder: kernel vs reference, and both vs the
one rule neither of them may be free to restate.

The rule is that a token's compress counts follow from ITS OWN position. A
sequence hands `1 + k` tokens to one MTP or DSpark forward, so a per-sequence
count is the last token's; giving it to the earlier ones lets them read
compressed groups holding their own future drafts.
`_v4_paged_prefill_indices_kernel` derives per-token for exactly this reason,
and the decode side did not until this builder replaced its host arithmetic.

Kernel-vs-reference cannot settle that on its own -- both compute the count from
the same expression, so a regression moves the pair together. So the counts are
also checked against a batch whose answer is written out here: a group
straddling an HCA group boundary, where per-token and per-sequence differ by
exactly one for the tokens before it.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "compares a Triton kernel against its reference; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import CSA_RATIO, HCA_RATIO
from atom.model_ops.v4_kernels.paged_decode_indices import (
    build_v4_paged_decode_indptr,
    build_v4_paged_decode_indptr_reference,
)

DEV = "cuda"
WIN = 128
INDEX_TOPK = 16  # far below the visible counts below, so the CSA cap is live


# Every destination arrives poisoned, never zeroed. The builder owns all four
# whole, and 0 is exactly what an unmapped row is supposed to end up holding --
# so on a zero-filled buffer the tests below would pass whether the builder
# wrote those rows or nobody did.
POISON = -7


def _outputs(n_rows, t_pad):
    def poison(n):
        return torch.full((n,), POISON, dtype=torch.int32, device=DEV)

    return {
        "swa_indptr": poison(t_pad + 1),
        "csa_indptr": poison(t_pad + 1),
        "hca_indptr": poison(t_pad + 1),
        "csa_n_committed_per_token": poison(n_rows),
    }


def _run(builder, *, positions, batch_id, t_pad):
    """Drive one builder over a batch and hand back its four outputs."""
    out = _outputs(t_pad, t_pad)
    builder(
        batch_id_per_token=torch.tensor(batch_id, dtype=torch.int32, device=DEV),
        # int64, as the production `positions` buffer is.
        positions=torch.tensor(positions, dtype=torch.int64, device=DEV),
        T_pad=t_pad,
        win=WIN,
        index_topk=INDEX_TOPK,
        **out,
    )
    return out


def _per_token(out, cls):
    """Recover each token's compress-section length from the two indptrs."""
    swa, other = out["swa_indptr"], out[f"{cls}_indptr"]
    return ((other[1:] - other[:-1]) - (swa[1:] - swa[:-1])).tolist()


# One MTP-4 group per sequence. Seq 0 straddles the boundary at 128: its first
# token may see zero HCA groups and the rest exactly one. Seq 1 sits well past a
# boundary so it exercises the un-straddled case in the same batch. The two
# trailing `-1` rows are the CG pad. Sequence context ends one past each group's
# last position (130 and 304) -- the `pos + 1 <= ctx` the builders rely on.
POSITIONS = [126, 127, 128, 129, 300, 301, 302, 303, 0, 0]
BATCH_ID = [0, 0, 0, 0, 1, 1, 1, 1, -1, -1]
T_PAD = len(BATCH_ID)
BATCH = {"positions": POSITIONS, "batch_id": BATCH_ID, "t_pad": T_PAD}


def test_kernel_matches_reference():
    kern = _run(build_v4_paged_decode_indptr, **BATCH)
    ref = _run(build_v4_paged_decode_indptr_reference, **BATCH)
    for name in kern:
        assert torch.equal(kern[name], ref[name]), name


def test_hca_count_comes_from_the_token_s_own_position():
    """The straddling group is the whole point: seq 0's tokens differ."""
    per_token_hca = _per_token(_run(build_v4_paged_decode_indptr, **BATCH), "hca")
    # pos 126 -> (126+1)//128 == 0 groups; the next three -> 1. Reading the
    # sequence's `ctx//128 == 1` instead would give all four a 1, and the token
    # at 126 would see the group covering positions 128..255 -- its own drafts.
    assert per_token_hca[:4] == [0, 1, 1, 1]
    # Seq 1 never straddles, so every one of its tokens lands on 304//128 == 2.
    assert per_token_hca[4:8] == [2, 2, 2, 2]


def test_csa_count_comes_from_the_position_and_the_slice_is_capped_at_topk():
    out = _run(build_v4_paged_decode_indptr, **BATCH)
    # Visible ends are 31/32/32/32 for seq 0, all far above INDEX_TOPK, so the
    # reserved slice length is the cap for every token.
    assert _per_token(out, "csa")[:8] == [INDEX_TOPK] * 8
    # The visibility itself is NOT capped by index_topk -- that bounds what the
    # slice reserves, not what the indexer may scan.
    assert out["csa_n_committed_per_token"][:4].tolist() == [31, 32, 32, 32]


def test_padded_tail_is_flat_and_zero():
    out = _run(build_v4_paged_decode_indptr, **BATCH)
    for name in ("swa_indptr", "csa_indptr", "hca_indptr"):
        ind = out[name]
        assert ind[0].item() == 0
        # `-1` rows contribute nothing, which is the `kv_len == 0` every reader
        # bails on. Anything else and a padded slot walks a live token's slice.
        assert ind[-1].item() == ind[-3].item()
    assert out["csa_n_committed_per_token"][8:].tolist() == [0, 0]


def test_a_position_past_its_sequence_is_not_clamped():
    """Positive control for a bound that was deleted, not a behaviour anyone
    wants. `min(per-token, ctx//ratio)` used to sit in both builders; it never
    fired, because every caller derives `positions` FROM `context_lens` --
    `prepare_decode` and `build_for_cudagraph_capture` -- so `pos + 1 <= ctx`.
    (Named, not cited by
    line: a line number in a docstring is stale the next time anyone edits
    above it, and these three were already wrong.)

    Feeding a position past its sequence's context is therefore off-contract,
    and this pins what the builders now do with it: answer from the position.
    A reader who re-adds the clamp on the theory that it guarded something will
    find this red, with the count it would have produced spelled out.
    """
    out = _run(
        build_v4_paged_decode_indptr,
        positions=[1000],  # a sequence whose ctx was, say, 130
        batch_id=[0],
        t_pad=1,
    )
    assert _per_token(out, "hca") == [1000 // HCA_RATIO]  # 7, not min(7, 130//128)=1
    assert out["csa_n_committed_per_token"].tolist() == [1001 // CSA_RATIO]


# --- The cases above are all narrower than one iteration of the kernel's loop.
#
# `tl.arange(0, BLOCK)` is 1024 lanes wide whatever `t_pad` is, so a batch of ten
# tokens leaves every lane past the tenth masked off: one loop trip, live lanes in
# the first wave only. That reaches neither the running offsets carried BETWEEN
# trips nor any interaction between waves. The two cases below are sized past both
# -- several trips, live lanes throughout -- and are checked against the reference
# rather than a written-out answer, because at this width nobody can read one.


def test_a_token_indexed_destination_must_be_exactly_the_token_axis():
    """The destination IS the token axis, so a longer buffer
    has a tail holding tokens this forward never declared -- and on a persistent
    forward_vars buffer those are the last forward's live ids, not zeros. Caught
    on the host rather than answered from them.

    `ValueError`, so the check survives `python -O`."""
    out = _outputs(8, 4)
    with pytest.raises(ValueError, match="exactly T_pad"):
        build_v4_paged_decode_indptr(
            batch_id_per_token=torch.zeros(8, dtype=torch.int32, device=DEV),
            positions=torch.arange(8, device=DEV),
            T_pad=4,
            win=WIN,
            index_topk=INDEX_TOPK,
            **out,
        )


def _mtp_batch(num_seqs, group, first_pos):
    """`num_seqs` sequences of `group` consecutive positions, one per MTP step."""
    positions, batch_id = [], []
    for s in range(num_seqs):
        start = first_pos + s * 37  # a stride that is coprime with both ratios
        positions += list(range(start, start + group))
        batch_id += [s] * group
    return positions, batch_id


def test_wide_token_indexed_batch_matches_reference():
    positions, batch_id = _mtp_batch(num_seqs=900, group=4, first_pos=125)
    t_pad = 4096  # > 3 loop trips, and the tail is CG pad
    batch = {
        "positions": positions + [0] * (t_pad - len(positions)),
        "batch_id": batch_id + [-1] * (t_pad - len(batch_id)),
        "t_pad": t_pad,
    }
    kern = _run(build_v4_paged_decode_indptr, **batch)
    ref = _run(build_v4_paged_decode_indptr_reference, **batch)
    for name in kern:
        assert torch.equal(kern[name], ref[name]), name
    # An offset dropped between trips would still leave each trip internally
    # consistent, so assert the total the readers size their pools by.
    assert kern["swa_indptr"][-1].item() == sum(min(p + 1, WIN) for p in positions)


def test_wide_batch_of_unequal_spans_matches_reference():
    """Sequences forwarding different numbers of tokens, wide enough to cross
    several loop trips. The case above holds every span at 4, so on its own it
    cannot tell a builder that walks tokens from one that assumes a fixed stride
    per sequence -- here every span width from 1 to 8 appears."""
    max_span = 8
    bs = 96
    spans = [1 + (s * 5) % max_span for s in range(bs)]  # every width appears
    positions, batch_id = [], []
    for s, n in enumerate(spans):
        start = 200 + s * 41  # stride coprime with both ratios
        positions += list(range(start, start + n))
        batch_id += [s] * n
    t_pad = len(positions)
    batch = {"positions": positions, "batch_id": batch_id, "t_pad": t_pad}
    kern = _run(build_v4_paged_decode_indptr, **batch)
    ref = _run(build_v4_paged_decode_indptr_reference, **batch)
    for name in kern:
        assert torch.equal(kern[name], ref[name]), name
    assert kern["swa_indptr"][-1].item() == sum(min(p + 1, WIN) for p in positions)
