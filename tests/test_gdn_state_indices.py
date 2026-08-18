# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The index tensors the GDN kernels gather their recurrent state through.

This is the one place a slot number crosses from the scheduler into a kernel,
and nothing downstream can catch a mistake here: every value is a valid index
into the pool, so a wrong one reads another request's state and produces
plausible tokens. There is no shape check, no assert, no NaN.

What is pinned:

  * the slot list is written through verbatim. Slots are handed out one at a
    time and a request's set is NOT adjacent, so anything computing
    `base + i` would be inventing a contiguity the pool does not provide. The
    kernels never wanted one either: the ssm kernel loads each index out of
    this tensor separately, and the conv path is handed column 0 alone.
  * column 0 is the committed state on both paths. The spec rows above it are
    rollback scratch.
  * a forked seq READS its source and WRITES its fresh slot. Collapsing the
    two would make the resumed forward read the slot it is about to fill.

`np` on the CpuGpuBuffer is the host staging array, so this runs on CPU: the
copy to device is a separate step and not what could be wrong here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions.gdn_attn import GDNStateMixin

MAX_BATCH = 8
WIDTH = 3  # 1 + num_spec, at --num-speculative-tokens 2


def builder(replayssm: bool = False):
    """A GDNStateMixin with only the three staging arrays the method touches.

    Prefilled with a sentinel no real slot can be, so "the method wrote this"
    and "the method left this alone" stay distinguishable — zero would not,
    since slot 0 is a legitimate index.

    `replayssm` is a real `__init__` attribute, not a stub convenience: the
    spec path forks on it, so leaving it off the stub would make these tests
    raise rather than measure.
    """
    stub = object.__new__(GDNStateMixin)
    stub.replayssm = replayssm
    stub.non_spec_state_indices_tensor = SimpleNamespace(
        np=np.full(MAX_BATCH, -99, dtype=np.int32)
    )
    stub.non_spec_state_indices_in_tensor = SimpleNamespace(
        np=np.full(MAX_BATCH, -99, dtype=np.int32)
    )
    stub.spec_state_indices_tensor = SimpleNamespace(
        np=np.full((MAX_BATCH, WIDTH), -99, dtype=np.int32)
    )
    return stub


def batch(state_slots, fork_srcs=None):
    return SimpleNamespace(
        state_slots=state_slots,
        state_fork_srcs=fork_srcs if fork_srcs is not None else [-1] * len(state_slots),
    )


class TestNonSpecPath:

    def test_writes_the_committed_slot(self):
        b = builder()
        b.prepare_state_indices(batch([[5], [2], [9]]))
        assert list(b.non_spec_state_indices_tensor.np[:3]) == [5, 2, 9]

    def test_reads_where_it_writes_when_there_is_no_fork(self):
        """No fork means the forward continues its own state in place."""
        b = builder()
        b.prepare_state_indices(batch([[5], [2]]))
        out = b.non_spec_state_indices_tensor.np
        assert list(b.non_spec_state_indices_in_tensor.np[:2]) == list(out[:2])

    def test_a_fork_reads_the_source_and_writes_the_fresh_slot(self):
        """The whole point of the fork: the checkpoint is read-only for this
        forward, and the state it produces lands somewhere else."""
        b = builder()
        b.prepare_state_indices(batch([[7], [2]], fork_srcs=[4, -1]))
        assert list(b.non_spec_state_indices_tensor.np[:2]) == [7, 2]
        assert list(b.non_spec_state_indices_in_tensor.np[:2]) == [4, 2]

    def test_the_source_is_one_slot_and_needs_no_translation(self):
        """A checkpoint is a single slot whatever the request width is, so the
        fork source goes in as-is — no multiply, no base."""
        b = builder()
        b.prepare_state_indices(batch([[7, 8, 9]], fork_srcs=[1]))
        assert b.non_spec_state_indices_in_tensor.np[0] == 1

    def test_only_the_scheduled_rows_are_written(self):
        b = builder()
        b.prepare_state_indices(batch([[5], [2]]))
        assert (b.non_spec_state_indices_tensor.np[2:] == -99).all()

    def test_a_short_fork_src_list_is_not_an_index_error(self):
        """`state_fork_srcs` is built from the same seqs in the same order, but
        the method reads it defensively; a missing tail means no fork."""
        b = builder()
        b.prepare_state_indices(batch([[5], [2], [9]], fork_srcs=[3]))
        assert list(b.non_spec_state_indices_in_tensor.np[:3]) == [3, 2, 9]


class TestSpecPath:

    def test_writes_the_slot_list_verbatim(self):
        """Scattered on purpose. `[4, 1, 6]` is what the pool actually hands
        out; a `base + i` implementation would write `[4, 5, 6]` here and pass
        every other assertion in this file.
        """
        b = builder()
        b.prepare_state_indices(batch([[4, 1, 6]]), with_spec=True)
        assert list(b.spec_state_indices_tensor.np[0]) == [4, 1, 6]

    def test_column_zero_is_the_committed_slot(self):
        b = builder()
        b.prepare_state_indices(batch([[4, 1, 6], [0, 9, 2]]), with_spec=True)
        assert list(b.spec_state_indices_tensor.np[:2, 0]) == [4, 0]

    def test_several_requests_keep_their_own_rows(self):
        b = builder()
        b.prepare_state_indices(batch([[4, 1, 6], [0, 9, 2]]), with_spec=True)
        assert list(b.spec_state_indices_tensor.np[1]) == [0, 9, 2]

    def test_a_narrow_seq_zero_fills_the_rest_of_its_row(self):
        """Zero rather than the sentinel: the row is cleared before the write,
        so a request holding fewer slots than the tensor is wide cannot leave
        a previous batch's indices for the kernel to gather.
        """
        b = builder()
        b.prepare_state_indices(batch([[4]]), with_spec=True)
        assert list(b.spec_state_indices_tensor.np[0]) == [4, 0, 0]

    def test_replayssm_addresses_one_slot_and_fans_out_to_neither_column(self):
        """Under ReplaySSM a request holds ONE slot: rollback is a cursor move
        over the cached (k, u, g) records, not a resume from a spare state. So
        the committed slot is the whole answer and columns 1..n stay cleared —
        writing scratch indices there would hand the kernel slots this request
        was never given.
        """
        b = builder(replayssm=True)
        b.prepare_state_indices(batch([[4]]), with_spec=True)
        assert list(b.spec_state_indices_tensor.np[0]) == [4, 0, 0]

    def test_replayssm_also_fills_the_1d_tensor_on_the_spec_path(self):
        """`_attach_replayssm` reads `slot_idx` out of the 1-D tensor even
        here, where the baseline spec path leaves it untouched. Pinned
        separately from the row above because the two tensors are written by
        different lines and only this one is load-bearing off the spec path.
        """
        b = builder(replayssm=True)
        b.prepare_state_indices(batch([[4], [7]]), with_spec=True)
        assert list(b.non_spec_state_indices_tensor.np[:2]) == [4, 7]

    def test_baseline_still_fans_out_across_the_row(self):
        """The contrast case: with ReplaySSM off the rollback slots are real
        and must reach the kernel, so this must NOT collapse to column 0.
        """
        b = builder(replayssm=False)
        b.prepare_state_indices(batch([[4, 1, 6]]), with_spec=True)
        assert list(b.spec_state_indices_tensor.np[0]) == [4, 1, 6]

    def test_a_fork_on_the_spec_path_is_refused(self):
        """There is no read-side spec tensor, so a fork here would silently
        read the slot it is about to write. BlockManager only forks onto
        prefill; this asserts rather than trusting that from a distance.
        """
        b = builder()
        with pytest.raises(AssertionError, match="state fork on the spec-decode path"):
            b.prepare_state_indices(batch([[4, 1, 6]], fork_srcs=[2]), with_spec=True)


def test_an_empty_batch_writes_nothing():
    b = builder()
    b.prepare_state_indices(batch([]))
    assert (b.non_spec_state_indices_tensor.np == -99).all()
    assert (b.spec_state_indices_tensor.np == -99).all()
