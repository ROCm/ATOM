# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""A P/D receive of a hybrid model keeps the prefix the consumer already holds.

The producer pushes the per-request state whole, so the consumer's prefix needs
no state checkpoint behind it and must not restore one.
"""

from test_state_checkpoint import (
    BLOCK,
    PAGED_COPY_RUNTIME,
    ckpt_config,
    make_block_manager,
)

from atom.model_engine.sequence import Sequence

PROMPT = list(range(100, 100 + 6 * BLOCK))


def receive(bm, token_ids, *, remote=True):
    seq = Sequence(list(token_ids), BLOCK, has_per_req_cache=True)
    if remote:
        seq.kv_transfer_params = {"do_remote_prefill": True}
    return seq


def consumer(reuses_local_prefix=True):
    """A decode consumer that received PROMPT once, with no checkpoint."""
    bm = make_block_manager(
        ckpt_config(state_checkpoint_interval_tokens=0),
        state_runtime=PAGED_COPY_RUNTIME,
    )
    bm.remote_state_reuses_local_prefix = reuses_local_prefix
    first = receive(bm, PROMPT)
    bm.allocate(first, bm.can_allocate(first))
    bm.register_received_prefix(first)
    bm.deallocate(first)
    return bm


def test_a_receive_keeps_the_prefix_without_a_checkpoint():
    bm = consumer()
    seq = receive(bm, PROMPT + list(range(900, 900 + 2 * BLOCK)))
    hit = bm.can_allocate(seq)
    # Every full prompt block of the earlier receive.
    assert hit == len(PROMPT) // BLOCK
    bm.allocate(seq, hit)
    assert seq.num_cached_tokens == len(PROMPT)
    # A fresh state slot that nothing restores: the producer fills it.
    assert seq.state_slot >= 0
    assert not bm.paged_state_checkpoints.restore_queued_for(seq.state_slot)
    assert not seq.replays_hit


def test_without_the_connector_capability_the_gate_still_applies():
    bm = consumer(reuses_local_prefix=False)
    seq = receive(bm, PROMPT + list(range(900, 900 + 2 * BLOCK)))
    assert bm.can_allocate(seq) == 0


def test_a_local_prefill_still_needs_a_checkpoint():
    bm = consumer()
    seq = receive(bm, PROMPT + list(range(900, 900 + 2 * BLOCK)), remote=False)
    assert bm.can_allocate(seq) == 0
