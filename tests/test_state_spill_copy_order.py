# SPDX-License-Identifier: MIT
# A spill reads a group's *previous* occupant, so it has to be keyed by the
# previous hash and issued before anything else that batch copies into that
# group. Getting either wrong stores one request's state under another
# request's hash -- valid-looking bytes no load path could reject -- so the two
# get their own test file.

import sys

from atom.model_engine.state_offload import StateOffloadIndex
from atom.model_engine.state_pool import StateGroupPool
from atom.model_engine.state_runtime import StateMaintenanceOps, StateTransfer


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


def warm_pool():
    """A pool whose only free group carries a checkpoint.

    The ordinary warm state: every group has been claimed at least once, so
    `_vacant` holds nothing usable and `pop()` has to spend a checkpoint.
    """
    pool = StateGroupPool(
        num_groups=3, transfer=StateTransfer.fork(1), hash_block_size=4
    )
    pool.offload = StateOffloadIndex(staging_depth=2, kv_offload_enabled=True)
    for group in (0, 1, 2):
        pool.claim(group)
    pool._index(777, 2)  # group 2 now backs hash 777
    pool.release(2)  # ...and is free, but on the checkpointed LRU
    return pool


def test_the_spill_is_filed_under_the_hash_of_the_bytes_already_there():
    """`pop()` spills before `invalidate()` on purpose: the spill wants the
    group's *previous* occupant, so it is keyed by the previous hash."""
    pool = warm_pool()

    group = pool.pop()

    assert group == 2, "the LRU checkpoint is what a warm pool has to spend"
    ((spill_source, slot),) = pool.take_spill_copies()
    assert spill_source == 2
    assert {h: s for h, s in pool.offload.take_pending()} == {777: slot}
    # The group itself no longer claims to hold that checkpoint.
    assert pool.hash_to_group == {}


def test_build_issues_the_spill_copy_before_the_other_relocations():
    """The ordering, pinned where it is decided.

    Both lists reach the device through `CommonAttentionBuilder.build()`, which
    is the only place holding the stream they are ordered on. A spill's source
    is a group the pool is handing out, so anything else that batch copies into
    it must land after the spill has read it -- otherwise the staging entry
    receives the new occupant's state while the tier stores it under the
    evicted checkpoint's hash: present, plausible, and someone else's.
    """
    CommonAttentionBuilder = real_attention_builder()

    pool = warm_pool()
    group = pool.pop()
    ((spill_source, slot),) = pool.take_spill_copies()
    ((h, _slot),) = pool.offload.take_pending()
    spill = (spill_source, pool.num_groups + slot, slot, h)

    issued = []

    class Batch:
        # A relocation into the group the spill has to read first.
        state_maintenance_ops = StateMaintenanceOps(
            relocations=((0, group),), spills=(spill,)
        )
        total_tokens_num_prefill = 0

    class Builder(CommonAttentionBuilder):
        # No __init__: the real one wants a whole ModelRunner, and `build`'s
        # ordering decision reads nothing it sets.
        def __init__(self):
            pass

        def relocate_state_slots(self, pairs):
            issued.extend(pairs)

        def _submit_state_spills(self, pairs):
            pass

        def prepare_decode(self, batch, bs):
            return None

        def build_for_cudagraph_capture(self, bs):
            return None

    Builder().build(Batch(), bs=1)

    spill_copy = (spill_source, pool.num_groups + slot)
    relocation = (0, group)
    assert issued.index(spill_copy) < issued.index(relocation), (
        f"the spill must read group {group} before anything overwrites it; "
        f"got {issued}"
    )
