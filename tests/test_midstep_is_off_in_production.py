# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The midstep write path is present, tested, and deliberately not enabled.

`TestMidstepCheckpoints` builds its own `StateTransfer(readable_midstep=True)`,
so the whole midstep suite passes whatever production declares. That is the
right call for those tests -- they are about the mechanism -- but it means
nothing anywhere would notice `GDNStateMixin.state_transfer` flipping the flag
by accident. This is that notice.

Why it is off, in one line each, so re-enabling is a decision rather than a
patch that looks harmless:

  * the runner declines to write on six conditions `commit_midstep` cannot
    see, and publishes the hash regardless -- a findable image over bytes
    nobody wrote;
  * `_checkpoint_targets` indexes three differently scoped sequence lists with
    one `i`, so one request's state can land in another's checkpoint;
  * the SSM read floors to a 64-token grid that `midstep_positions` does not
    enforce (`hash_block_size` defaults to 16);
  * the conv window is `conv_kernel-1+num_spec` wide in the kernel and
    `conv_kernel-1` in the producer's guard, an out-of-bounds read under
    speculation.

None of it has run under a server: Kimi-K3 takes the PAGE path and cannot
reach this one. The numerical core *is* verified --
`test_gdn_midstep_state_gpu.py` shows a slice of `h` is bit-exact against a
forward stopped there -- which is why the machinery stays rather than being
deleted.

Deliberately not behind `importorskip`: a module that skips itself when aiter
is absent is a module the non-GPU CI runner never runs, which is how this
would rot back to True unnoticed.
"""


def test_gdn_does_not_declare_itself_midstep_readable():
    from atom.model_ops.attentions.gdn_attn import GDNStateMixin

    transfer = GDNStateMixin.state_transfer(object.__new__(GDNStateMixin))
    assert transfer.readable_midstep is False, (
        "GDN midstep writes silently store wrong state; see this module's "
        "docstring before turning it back on"
    )
    assert transfer.forks, "the fork itself is unaffected"
    assert transfer.fork_tokens == 1


def test_the_paged_coordinator_agrees():
    """Whatever GDN says, a PAGE class is never midstep-readable.

    Its image is copied out of a slot after the forward, so there are no
    interior positions to slice -- a different reason from GDN's, reaching the
    same answer, and the two must not drift into disagreeing.
    """
    from atom.model_engine.page_unit_checkpoint import (
        PagedStateCheckpointCoordinator,
    )

    assert PagedStateCheckpointCoordinator.readable_midstep is False


def test_the_machinery_is_still_here():
    """Off, not deleted -- the copy-out kernel and its entry point remain.

    If this fails, someone removed the implementation instead of the flag, and
    re-enabling is no longer a one-line decision.
    """
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints

    assert callable(write_state_checkpoints)
    from atom.model_ops.fla_ops.chunk import pop_last_intermediate_states

    assert callable(pop_last_intermediate_states)
