# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The midstep write path is present and deliberately not enabled.

`TestMidstepCheckpoints` builds its own `StateTransfer(readable_midstep=True)`,
so the whole midstep suite passes whatever production declares. That is the
right call for those tests -- they are about the mechanism -- but it leaves
nothing watching the flag itself. This is that watch, for the half of it that
a CPU-only runner can see.

Why it is off, so re-enabling is a decision rather than a patch that looks
harmless:

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
reach this one.

`GDNStateMixin.state_transfer` is the other half of the flag and is NOT
checked here. Reaching it means importing `gdn_attn`, which imports aiter at
module level, and this suite runs on a plain CPU runner with neither aiter nor
triton installed -- an import that would abort collection rather than skip.
The GDN side belongs with the kernel tests, outside this repo's unit suite.
"""


def test_the_paged_coordinator_is_not_midstep_readable():
    """A PAGE class is never midstep-readable, whatever GDN declares.

    Its image is copied out of a slot after the forward, so there are no
    interior positions to slice. A different reason from GDN's, reaching the
    same answer today -- and if GDN's answer ever changes, this one must not
    follow it by accident.
    """
    from atom.model_engine.page_unit_checkpoint import (
        PagedStateCheckpointCoordinator,
    )

    assert PagedStateCheckpointCoordinator.readable_midstep is False


def test_the_midstep_protocol_is_still_declared():
    """Off, not deleted: `StateCache` still names the three-call protocol.

    If this fails, someone removed the interface instead of the flag, and
    turning midstep back on is no longer a one-line decision.
    """
    from atom.model_engine.state_cache import StateCache

    for name in ("reserve_midstep", "publish_midstep", "cancel_midstep"):
        assert hasattr(StateCache, name), f"{name} left the protocol"


def test_a_readable_transfer_is_still_expressible():
    """The flag is a real bivalued field, not a constant folded to False.

    `StateTransfer` is pure data with no GPU dependency, so this holds on a
    CPU runner: whoever re-enables the path needs the field to still carry the
    claim, and whoever deletes the field should fail here first.
    """
    from atom.model_engine.state_runtime import StateTransfer

    assert StateTransfer.fork(1, readable_midstep=True).readable_midstep is True
    assert StateTransfer.fork(1).readable_midstep is False
