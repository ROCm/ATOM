# SPDX-License-Identifier: MIT
# A spill must be issued before this batch's checkpoint copies, because a
# group can be both. Getting it backwards stores one request's state under
# another request's hash -- valid-looking bytes no load path could reject --
# so it gets its own test file.

import sys

from atom.model_engine.state_offload import StateOffloadIndex
from atom.model_engine.state_pool import StateGroupPool, StateTransfer


def real_attention_builder():
    """Import `CommonAttentionBuilder` against the real package tree.

    `tests/plugin/test_rtpllm_forward_context_semantics.py` installs bare
    `types.ModuleType` stubs for `atom.utils.forward_context`,
    `atom.model_ops.attentions.gdn_attn` and others at *import* time and never
    removes them. Collection imports it before this file, so under the full
    suite `atom.model_ops.attentions.backends` fails on
    `ImportError: cannot import name 'ForwardContext'` -- which is already why
    `test_state_entry_views.py`, `test_gdn_state_copy.py` and
    `test_cudagraph_capture_bounds.py` error out. Dropping every `atom.*` entry
    forces a clean import from disk; restoring the snapshot afterwards puts the
    stubs back, so any plugin test that lazily imports still sees what it set up.
    """
    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] == "atom"}
    for name in saved:
        del sys.modules[name]
    try:
        from atom.model_ops.attentions.backends import CommonAttentionBuilder

        return CommonAttentionBuilder
    finally:
        sys.modules.update(saved)


class Seq:
    """The two fields `_commit_pending` reads, and nothing else."""

    has_per_req_cache = True
    num_tokens = 999

    def __init__(self, group, pending):
        self.per_req_cache_group = group
        self.pending_checkpoint = pending


def collided_pool():
    """A pool in the state where one group is both spill source and copy dest.

    The precondition is the ordinary warm one: every group has been claimed at
    least once, so `_vacant` holds nothing usable and the only free group is
    carrying a checkpoint. `pop()` then takes the `_checkpointed` branch,
    spills that group under its old hash, and hands the very same group back
    as the new checkpoint's destination.
    """
    pool = StateGroupPool(
        num_groups=3, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = StateOffloadIndex(staging_depth=2, kv_offload_enabled=True)
    for group in (0, 1, 2):
        pool.claim(group)
    pool._index(777, 2)  # group 2 now backs the old hash
    pool.release(2)  # ...and is free, but on the checkpointed LRU
    pool._checkpoint_pending = [Seq(group=0, pending=888)]
    return pool


def test_a_spill_source_really_can_be_a_checkpoint_destination():
    """The premise. If this ever stops holding, the ordering test below is
    vacuous rather than passing, so it is asserted separately."""
    pool = collided_pool()
    copies = pool.take_copies()
    spills = pool.take_spill_copies()
    ((_src, dst),) = copies
    ((spill_source, _slot),) = spills
    assert dst == spill_source == 2


def test_the_spill_is_filed_under_the_hash_of_the_bytes_already_there():
    """`pop()` spills before `invalidate()` on purpose: the spill wants the
    group's *previous* occupant, so it is keyed by the previous hash."""
    pool = collided_pool()
    pool.take_copies()
    spills = pool.take_spill_copies()
    ((_group, slot),) = spills
    assert {h: s for h, s in pool.offload.take_pending()} == {777: slot}
    # The same group is simultaneously indexed under the new hash.
    assert pool.hash_to_group == {888: 2}


def test_build_issues_the_spill_copy_before_the_checkpoint_copy():
    """The fix, pinned where it is decided.

    Both lists are consumed by `CommonAttentionBuilder.build()`, which is the
    only place holding the stream they are ordered on. With the checkpoint copy
    first, the staging entry receives group 0's state while the tier stores it
    under hash 777 -- group 2's hash. Present, plausible, and someone else's.
    """
    CommonAttentionBuilder = real_attention_builder()

    pool = collided_pool()
    copy_pairs = pool.take_copies()
    spill_pairs = [
        (group, pool.num_groups + slot, slot, h)
        for (group, slot), (h, _s) in zip(
            pool.take_spill_copies(), pool.offload.take_pending(), strict=True
        )
    ]

    issued = []

    class Batch:
        state_copy_pairs = copy_pairs
        state_spill_pairs = spill_pairs
        total_tokens_num_prefill = 0

    class Builder(CommonAttentionBuilder):
        # No __init__: the real one wants a whole ModelRunner, and `build`'s
        # ordering decision reads nothing it sets.
        def __init__(self):
            pass

        def copy_state_entries(self, pairs):
            issued.extend(pairs)

        def _submit_state_spills(self, pairs):
            pass

        def prepare_decode(self, batch, bs):
            return None

        def build_for_cudagraph_capture(self, bs):
            return None

    Builder().build(Batch(), bs=1)

    spill_copy = (2, pool.num_groups + 0)  # group 2 -> its staging entry
    checkpoint_copy = (0, 2)  # group 0 -> group 2, overwriting it
    assert issued.index(spill_copy) < issued.index(checkpoint_copy), (
        "the spill must read group 2 before the checkpoint copy overwrites it; "
        f"got {issued}"
    )
