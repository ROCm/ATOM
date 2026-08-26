# SPDX-License-Identifier: MIT
"""PD incremental transfer offset on a DCP decode node.

``update_state_after_alloc`` tells the producer how many leading blocks the
decode node already holds in its prefix cache. Under DCP one block-table entry
is a *virtual* block covering ``block_size * dcp_size`` global tokens (the
prefix cache hashes at that granularity too, see ``BlockManager``), so counting
the offset in plain blocks overstates it by ``dcp_size`` and the producer skips
tokens the consumer never received. Nothing faults -- decode just attends to
uninitialised KV -- so the divisor is pinned here.
"""

from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
        MooncakeConnectorScheduler,
    )

BLOCK_SIZE = 16


def _scheduler(dcp_size):
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "kv_consumer", "handshake_port": 6301},
        tensor_parallel_size=4,
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        pipeline_parallel_size=1,
        kv_cache_block_size=BLOCK_SIZE,
        decode_context_parallel_size=dcp_size,
    )
    return MooncakeConnectorScheduler(config)


def _seq(num_cached_tokens, num_blocks):
    return SimpleNamespace(
        id="req-0",
        kv_transfer_params={"do_remote_prefill": True, "transfer_id": "t0"},
        block_table=list(range(num_blocks)),
        num_cached_tokens=num_cached_tokens,
        has_per_req_cache=False,
        per_req_cache_group=-1,
    )


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("cached_virtual_blocks", [0, 1, 5])
def test_offset_counts_virtual_blocks(dcp_size, cached_virtual_blocks):
    virtual_block = BLOCK_SIZE * dcp_size
    seq = _seq(cached_virtual_blocks * virtual_block, num_blocks=64)

    _scheduler(dcp_size).update_state_after_alloc(seq)

    assert seq.kv_transfer_params["num_computed_blocks"] == cached_virtual_blocks


def test_partial_virtual_block_is_not_counted():
    """A prefix hit shorter than one virtual block gives no skippable block: the
    consumer holds rank-local fragments of it, not a whole transferable unit."""
    seq = _seq(BLOCK_SIZE * 3, num_blocks=64)  # 3 blocks, but W = 4
    _scheduler(4).update_state_after_alloc(seq)
    assert seq.kv_transfer_params["num_computed_blocks"] == 0


def test_per_request_cache_forces_a_full_transfer():
    seq = _seq(BLOCK_SIZE * 4 * 3, num_blocks=64)
    seq.has_per_req_cache = True
    _scheduler(4).update_state_after_alloc(seq)
    assert seq.kv_transfer_params["num_computed_blocks"] == 0
