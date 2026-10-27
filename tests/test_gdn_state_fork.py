# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A state fork reaching a decode forward, on both the plain and spec paths.

Under pp=1 the checkpoint is taken between prompt chunks, so the forward that
consumes the fork is the next chunk. Under pp>1 it lands after the prompt, so
the fork falls on the first decode -- and with DSpark on, that decode is a
spec-verify step. `prepare_state_indices` is the host half of both: it has to
publish the slot the state is READ from, whichever path is running.

`prepare_state_indices` reads a handful of attributes off the builder, so it
is driven here unbound over a stand-in rather than through a real one.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip(
    "atom.model_ops.attentions.gdn_attn",
    reason="requires AITER to import the attention backends",
    exc_type=ImportError,
)

from atom.model_ops.attentions.gdn_attn import PAD_SLOT_ID, GDNStateMixin

WIDTH = 4  # mtp_k + 1 reserved slots per request


def _builder(honours_fork, rows=8):
    def buf(shape):
        return SimpleNamespace(np=np.full(shape, -7, dtype=np.int32))

    return SimpleNamespace(
        _check_state_indices_writable=lambda: None,
        non_spec_state_indices_tensor=buf(rows),
        non_spec_state_indices_in_tensor=buf(rows),
        spec_state_indices_tensor=buf((rows, WIDTH)),
        replayssm=False,
        spec_honours_state_fork=honours_fork,
    )


def _batch(fork_srcs, slots):
    return SimpleNamespace(state_fork_srcs=list(fork_srcs), state_slots=list(slots))


def _slots(n):
    return [list(range(i * WIDTH, (i + 1) * WIDTH)) for i in range(n)]


def test_decode_fork_publishes_the_source_as_the_read_slot():
    b = _builder(honours_fork=False)
    GDNStateMixin.prepare_state_indices(b, _batch([-1, 30], _slots(2)), with_spec=False)
    assert b._has_state_fork
    np.testing.assert_array_equal(b.non_spec_state_indices_tensor.np[:2], [0, 4])
    # Row 0 has no fork and reads what it writes; row 1 reads the checkpoint.
    np.testing.assert_array_equal(b.non_spec_state_indices_in_tensor.np[:2], [0, 30])


def test_spec_fork_publishes_the_source_too():
    b = _builder(honours_fork=True)
    GDNStateMixin.prepare_state_indices(b, _batch([-1, 30], _slots(2)), with_spec=True)
    assert b._has_state_fork
    # The spec kernels resume from column 0, which is the write slot...
    np.testing.assert_array_equal(b.spec_state_indices_tensor.np[:2, 0], [0, 4])
    # ...and the model moves the state there from the slot published here.
    np.testing.assert_array_equal(b.non_spec_state_indices_in_tensor.np[:2], [0, 30])


def test_no_fork_reads_the_slot_it_writes_on_both_paths():
    for with_spec in (False, True):
        b = _builder(honours_fork=True)
        GDNStateMixin.prepare_state_indices(
            b, _batch([-1, -1], _slots(2)), with_spec=with_spec
        )
        assert not b._has_state_fork
        np.testing.assert_array_equal(b.non_spec_state_indices_in_tensor.np[:2], [0, 4])


def test_spec_fork_is_refused_by_a_backend_that_cannot_move_the_state():
    b = _builder(honours_fork=False)
    with pytest.raises(AssertionError, match="spec_honours_state_fork"):
        GDNStateMixin.prepare_state_indices(
            b, _batch([-1, 30], _slots(2)), with_spec=True
        )


def test_rows_past_the_batch_are_padded_on_every_table():
    b = _builder(honours_fork=True, rows=8)
    GDNStateMixin.prepare_state_indices(b, _batch([-1], _slots(1)), with_spec=True)
    assert (b.non_spec_state_indices_tensor.np[1:] == PAD_SLOT_ID).all()
    assert (b.non_spec_state_indices_in_tensor.np[1:] == PAD_SLOT_ID).all()
    assert (b.spec_state_indices_tensor.np[1:] == PAD_SLOT_ID).all()
