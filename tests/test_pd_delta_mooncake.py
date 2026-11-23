# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""A Mooncake receive of a hybrid model keeps the prefix the consumer holds.

One TP producer writes the paired blocks and then the per-request state by
slot, so the state arrives whole whatever block the pairing starts at. The
Mooncake consumer says so before allocation (`receives_suffix_and_state`), and
then pairs only the blocks past the prefix the block manager kept.
"""

import pytest
from test_pd_delta_hybrid import GEOMETRY, LONGER, PROMPT, admit, consumer, receive
from test_pd_pp import _mooncake_consumer_scheduler
from test_state_checkpoint import BLOCK

mc = pytest.importorskip("atom.kv_transfer.disaggregation.mooncake.mooncake_connector")

KEPT = len(PROMPT) // BLOCK


def mooncake(dcp_size=1):
    return _mooncake_consumer_scheduler(mc, block_size=BLOCK, dcp_size=dcp_size)


def test_mooncake_brings_the_state_from_one_tp_producer_paging_alike():
    sched = mooncake()
    assert sched.receives_suffix_and_state(receive(LONGER, **GEOMETRY))
    assert sched.receives_suffix_and_state(
        receive(LONGER, remote_pp_size=1, **GEOMETRY)
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"remote_pp_size": 8, **GEOMETRY}, id="pp-producer"),
        pytest.param({"chunked_transfer": True, **GEOMETRY}, id="chunked"),
        pytest.param({"block_size": 2 * BLOCK, "dcp_size": 1}, id="other-block-size"),
        pytest.param({"block_size": BLOCK, "dcp_size": 2}, id="other-dcp"),
        pytest.param({}, id="no-geometry"),
        pytest.param({"do_remote_prefill": False, **GEOMETRY}, id="local-prefill"),
    ],
)
def test_mooncake_keeps_the_full_transfer_otherwise(params):
    assert not mooncake().receives_suffix_and_state(receive(LONGER, **params))


def test_an_unsharded_producer_into_a_dcp_consumer_keeps_the_full_transfer():
    # A stateless receive pairs dcp_size source blocks per destination block
    # here; with per-request state only a producer sharded alike says yes.
    sched = mooncake(dcp_size=4)
    assert not sched.receives_suffix_and_state(receive(LONGER, **GEOMETRY))
    assert sched.receives_suffix_and_state(
        receive(LONGER, block_size=BLOCK, dcp_size=4)
    )


def test_a_producer_never_says_yes():
    sched = mooncake()
    sched.is_producer = True
    assert not sched.receives_suffix_and_state(receive(LONGER, **GEOMETRY))


def test_a_receive_keeps_the_prefix_and_pairs_only_the_suffix():
    bm = consumer()
    sched = mooncake()
    seq = receive(LONGER, **GEOMETRY)
    assert admit(sched, seq)
    hit = bm.can_allocate(seq)
    assert hit == KEPT
    bm.allocate(seq, hit)
    # A fresh state slot that nothing restores: the producer fills it.
    assert not bm.paged_state_checkpoints.restore_queued_for(seq.state_slot)

    sched.update_state_after_alloc(seq)

    # The transfer starts where the block manager's claim ends.
    assert seq.kv_transfer_params["num_computed_blocks"] == KEPT
    assert seq.kv_transfer_params["src_block_skip_factor"] == 1


def test_a_receive_from_a_pp_producer_claims_no_prefix_and_pairs_every_block():
    bm = consumer()
    sched = mooncake()
    seq = receive(LONGER, remote_pp_size=8, **GEOMETRY)
    assert not admit(sched, seq)
    hit = bm.can_allocate(seq)
    assert hit == 0
    bm.allocate(seq, hit)

    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == 0


def test_a_local_hit_without_the_peers_state_is_still_received_whole():
    # A checkpoint-backed local hit of a request whose producer does not send
    # the state whole: the prefix is claimed for the local state, and the
    # transfer still pairs every block.
    sched = mooncake()
    seq = receive(LONGER, remote_pp_size=8, **GEOMETRY)
    seq.block_table = list(range(len(LONGER) // BLOCK))
    seq.num_cached_tokens = KEPT * BLOCK

    sched.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == 0
