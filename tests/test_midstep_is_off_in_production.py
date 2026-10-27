# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The midstep write path is opt-in, and off unless asked for.

`TestMidstepCheckpoints` builds its own `StateTransfer(readable_midstep=True)`,
so the whole midstep suite passes whatever production declares. That is the
right call for those tests -- they are about the mechanism -- but it leaves
nothing watching the default itself. This is that watch, for the half of it
that a CPU-only runner can see.

Qwen3-Next / Qwen3.5 turn it on with `ATOM_GDN_STATE_MIDSTEP=1`, on the PAGE
checkpoint path only. The four defects that kept it off are fixed, and each
has its own test:

  * the runner declined to write on conditions `commit_midstep` could not
    see, and the hash was published regardless. Both ends now share
    `midstep_target_ok`, the scheduler filters with it and the runner raises
    on a reservation that fails it (`TestMidstepTargets`);
  * `_checkpoint_targets` indexed three differently scoped sequence lists with
    one `i`. Rows now go through `ScheduledBatch.state_rows`
    (`TestMidstepTargets`);
  * the SSM read floors to a 64-token grid counted from the forward's start.
    `midstep_positions` now snaps onto that grid (`TestMidstepGrid`);
  * the conv window was `conv_kernel-1+num_spec` wide and written at the wrong
    positions. It is now `conv_kernel-1`, passed explicitly (GPU test).

`GDNStateMixin.state_transfer` (the fork path) still declares False, and is
NOT checked here: reaching it means importing `gdn_attn`, which imports aiter
at module level, and this suite runs on a plain CPU runner without it.
"""


def test_the_paged_coordinator_is_not_midstep_readable_by_default():
    """A PAGE class reads midstep only when its backend declares it.

    The image is copied out of a slot, and only a backend whose kernel exposes
    interior states (GDN's `h`) can fill that slot from inside a forward.
    """
    from test_state_checkpoint import (
        PAGED_COPY_RUNTIME,
        PAGED_COPY_SPEC,
        make_block_manager,
        paged_copy_config,
    )

    from atom.model_engine.state_runtime import StateRuntime, StateTransfer

    bm = make_block_manager(paged_copy_config(), state_runtime=PAGED_COPY_RUNTIME)
    assert not bm.paged_state_checkpoints.readable_midstep

    readable = StateRuntime(
        transfer=StateTransfer.copy(
            PAGED_COPY_SPEC.layout_id, readable_midstep=True, midstep_grid=4
        ),
        checkpoint_spec=PAGED_COPY_SPEC,
    )
    bm = make_block_manager(paged_copy_config(), state_runtime=readable)
    assert bm.paged_state_checkpoints.readable_midstep


def test_the_gdn_midstep_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_GDN_STATE_MIDSTEP", raising=False)
    assert envs.ATOM_GDN_STATE_MIDSTEP is False
    monkeypatch.setenv("ATOM_GDN_STATE_MIDSTEP", "1")
    assert envs.ATOM_GDN_STATE_MIDSTEP is True


def test_the_midstep_protocol_is_still_declared():
    """`StateCache` still names the three-call protocol."""
    from atom.model_engine.state_cache import StateCache

    for name in ("reserve_midstep", "publish_midstep", "cancel_midstep"):
        assert hasattr(StateCache, name), f"{name} left the protocol"


def test_a_readable_transfer_must_name_its_grid():
    """A readable claim without a grid cannot be placed against.

    `midstep_positions` snaps every position onto the grid; a readable
    transfer with grid 0 would leave it nothing to snap to.
    """
    import pytest

    from atom.model_engine.state_runtime import StateTransfer

    assert (
        StateTransfer.fork(1, readable_midstep=True, midstep_grid=64).readable_midstep
        is True
    )
    assert StateTransfer.fork(1).readable_midstep is False
    with pytest.raises(ValueError):
        StateTransfer.fork(1, readable_midstep=True)
    with pytest.raises(ValueError):
        StateTransfer.fork(1, midstep_grid=64)
