# SPDX-License-Identifier: MIT
# The GDN state-index buffers must not carry a previous batch's slots into rows
# that hold no state.
#
# `prepare_state_indices` publishes one row per scheduled seq, but only seqs with
# a state slot are in `batch.state_slots`. The DP-sync dummy decode
# (`ModelRunner.dummy_execution`) is one seq with no slot. Its row used to keep
# the last real batch's first slot, and the dummy's decode then updated that
# request's conv and recurrent state in place: the GDN/KDA kernels skip a row
# only when its index is PAD_SLOT_ID.

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions.gdn_attn import (
    PAD_SLOT_ID,
    GDNAttentionMetadataBuilder,
)

MAX_BS = 8
NUM_SPEC = 2


def make_builder(replayssm: bool = False):
    """What `prepare_state_indices` touches: the three CPU index buffers."""

    def buf(*shape):
        return SimpleNamespace(np=np.full(shape, PAD_SLOT_ID, dtype=np.int32))

    return SimpleNamespace(
        non_spec_state_indices_tensor=buf(MAX_BS),
        non_spec_state_indices_in_tensor=buf(MAX_BS),
        spec_state_indices_tensor=buf(MAX_BS, NUM_SPEC + 1),
        replayssm=replayssm,
        _check_state_indices_writable=lambda: None,
    )


def batch(state_slots, fork_srcs=()):
    return SimpleNamespace(state_slots=state_slots, state_fork_srcs=list(fork_srcs))


# What `ModelRunner.dummy_execution` schedules: one seq, no state slot.
DUMMY = batch([])


def prepare(builder, scheduled, with_spec=False):
    GDNAttentionMetadataBuilder.prepare_state_indices(
        builder, scheduled, with_spec=with_spec
    )


def indices(builder):
    return (
        builder.non_spec_state_indices_tensor.np,
        builder.non_spec_state_indices_in_tensor.np,
        builder.spec_state_indices_tensor.np,
    )


def test_dummy_decode_after_a_real_batch_reads_pad():
    b = make_builder()
    prepare(b, batch([[5], [9], [3]]))
    out, out_in, _ = indices(b)
    assert out[:3].tolist() == [5, 9, 3]
    assert out_in[:3].tolist() == [5, 9, 3]

    prepare(b, DUMMY)
    out, out_in, spec = indices(b)
    # Row 0 is the dummy's: slot 5 belongs to a request that is not in this forward.
    assert (out == PAD_SLOT_ID).all()
    assert (out_in == PAD_SLOT_ID).all()
    assert (spec == PAD_SLOT_ID).all()


def test_a_narrower_batch_clears_the_wider_batch_tail():
    b = make_builder()
    prepare(b, batch([[1], [2], [3], [4]]))
    prepare(b, batch([[7], [8]]))
    out, out_in, _ = indices(b)
    assert out[:2].tolist() == [7, 8]
    assert out_in[:2].tolist() == [7, 8]
    assert (out[2:] == PAD_SLOT_ID).all()
    assert (out_in[2:] == PAD_SLOT_ID).all()


def test_fork_source_still_feeds_the_read_side():
    b = make_builder()
    prepare(b, batch([[4], [6]], fork_srcs=[11, -1]))
    out, out_in, _ = indices(b)
    assert out[:2].tolist() == [4, 6]
    assert out_in[:2].tolist() == [11, 6]
    assert (out[2:] == PAD_SLOT_ID).all()
    assert (out_in[2:] == PAD_SLOT_ID).all()


@pytest.mark.parametrize("replayssm", [False, True])
def test_spec_decode_dummy_after_a_real_batch_reads_pad(replayssm):
    b = make_builder(replayssm=replayssm)
    prepare(b, batch([[5, 6, 7], [9, 10, 11]]), with_spec=True)
    out, _, spec = indices(b)
    assert spec[:2].tolist() == [[5, 6, 7], [9, 10, 11]]
    if replayssm:
        # ReplaySSM mirrors the committed slot into the 1-D tensor for `slot_idx`.
        assert out[:2].tolist() == [5, 9]

    prepare(b, DUMMY, with_spec=True)
    out, out_in, spec = indices(b)
    assert (spec == PAD_SLOT_ID).all()
    assert (out == PAD_SLOT_ID).all()
    assert (out_in == PAD_SLOT_ID).all()
