# SPDX-License-Identifier: MIT
"""Producer-side RDMA descriptors for a DCP decode node.

``_execute_block_transfer`` normally emits one whole-block descriptor per
region per block. A DCP consumer rank owns only part of each block, so the
producer has to relayout: descriptors become token runs, scaled by the region's
per-token bytes. Which relayout depends on the region -- the MLA latent is
interleave-sharded (``plan_sharded``), while a replicated indexer cache is held
whole by every rank (``plan_replicated``). Getting the scale, the rank, or the
per-region choice wrong writes real KV to the wrong slots, which decode reads
back as plausible-looking garbage rather than a fault, so the addresses are
pinned here.
"""

import threading

import msgpack
import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
        MooncakeConnector,
        plan_replicated,
        plan_sharded,
    )
    from atom.kv_transfer.disaggregation.types import (
        INDEX_CACHE_ROLE,
        MLA_KV_ROLE,
        ConnectorMetadata,
    )

BLOCK_SIZE = 16
# Two regions with different per-token widths: MLA's 576-wide latent KV and the
# narrower indexer cache, which shares the KV slot mapping.
TOKEN_BYTES = [576 * 2, 144]
REGION_BASES = [0x1000_0000, 0x9000_0000]
CONSUMER_BASES = [0x2000_0000, 0xA000_0000]


def _connector():
    """A connector with only the state ``_execute_block_transfer`` reads."""
    conn = object.__new__(MooncakeConnector)
    conn.block_size = BLOCK_SIZE
    conn.pp_size = 1
    conn.pp_rank = 0
    conn._start_layer = 0
    conn._num_local_layers = len(REGION_BASES)
    conn._block_region_consumer_indices = None
    conn.kv_caches_base_addr = list(REGION_BASES)
    conn._per_block_bytes_list = [t * BLOCK_SIZE for t in TOKEN_BYTES]
    conn._block_region_roles = [MLA_KV_ROLE, INDEX_CACHE_ROLE]
    conn.written = []
    conn._rdma_write_with_retry = (
        lambda target, src, dst, sizes, req_id, kind: conn.written.append(
            list(zip(src, dst, sizes))
        )
        or True
    )
    return conn


def _request(dcp_size, dcp_rank, interleave, replicate_index=False):
    return {
        "consumer_replicates_index_cache": replicate_index,
        "consumer_base_addrs": list(CONSUMER_BASES),
        "consumer_num_layers": len(CONSUMER_BASES),
        "consumer_region_roles": [MLA_KV_ROLE, INDEX_CACHE_ROLE],
        # Deliberately different from dcp_size: reading the wrong field here is
        # the easiest way to get a plausible but wrong plan.
        "consumer_tp_size": 8,
        "consumer_dcp_size": dcp_size,
        "consumer_dcp_rank": dcp_rank,
        "consumer_dcp_interleave": interleave,
    }


def _descriptors(conn, request_data, src_block_ids, dst_block_ids):
    assert conn._execute_block_transfer(
        request_data, "host:1", src_block_ids, dst_block_ids, "req-0"
    )
    (batch,) = conn.written
    per_region = len(batch) // len(REGION_BASES)
    return [
        batch[i * per_region : (i + 1) * per_region] for i in range(len(REGION_BASES))
    ]


def test_a_small_layout_by_hand():
    """block_size 4, dcp_size 2, block interleave, one region-token of 1 byte.

    Rank 1 of a 2-way split with S == block_size owns whole source blocks
    ``2b + 1``, so virtual block 0 (dst id 7) pulls src id 30 and virtual block
    1 (dst id 5) pulls src id 33 -- two whole-block descriptors, no merging
    because 30 and 33 are not adjacent.
    """
    conn = object.__new__(MooncakeConnector)
    conn.block_size = 4
    conn.pp_size = 1
    conn._start_layer = 0
    conn._num_local_layers = 1
    conn._block_region_consumer_indices = None
    conn.kv_caches_base_addr = [1000]
    conn._per_block_bytes_list = [4]
    conn._block_region_roles = [MLA_KV_ROLE]
    conn.written = []
    conn._rdma_write_with_retry = (
        lambda target, src, dst, sizes, req_id, kind: conn.written.append(
            list(zip(src, dst, sizes))
        )
        or True
    )
    request_data = {
        "consumer_base_addrs": [2000],
        "consumer_num_layers": 1,
        "consumer_dcp_size": 2,
        "consumer_dcp_rank": 1,
        "consumer_dcp_interleave": 4,
    }

    assert conn._execute_block_transfer(
        request_data, "host:1", [10, 30, 20, 33], [7, 5], "req-0"
    )
    assert conn.written == [
        [(1000 + 30 * 4, 2000 + 7 * 4, 4), (1000 + 33 * 4, 2000 + 5 * 4, 4)]
    ]


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("interleave", [1, 4, BLOCK_SIZE])
def test_every_region_scales_the_same_plan_by_its_token_width(dcp_size, interleave):
    src_block_ids = [40, 3, 17, 62, 9, 51, 28, 6, 33]
    n_dst = -(-len(src_block_ids) // dcp_size)
    dst_block_ids = [11, 4, 27][:n_dst]

    for dcp_rank in range(dcp_size):
        conn = _connector()
        regions = _descriptors(
            conn,
            _request(dcp_size, dcp_rank, interleave),
            src_block_ids,
            dst_block_ids,
        )
        src_off, dst_off, run_len = plan_sharded(
            src_block_ids,
            dst_block_ids,
            BLOCK_SIZE,
            dcp_size,
            dcp_rank,
            interleave,
        )
        for region_idx, descriptors in enumerate(regions):
            tok = TOKEN_BYTES[region_idx]
            assert descriptors == [
                (
                    REGION_BASES[region_idx] + int(s) * tok,
                    CONSUMER_BASES[region_idx] + int(d) * tok,
                    int(n) * tok,
                )
                for s, d, n in zip(src_off, dst_off, run_len)
            ]


def test_the_ranks_together_move_every_source_token_once():
    src_block_ids = [40, 3, 17, 62, 9, 51, 28, 6, 33]
    dst_block_ids = [11, 4, 27]
    dcp_size = 4
    covered = []

    for dcp_rank in range(dcp_size):
        conn = _connector()
        first_region = _descriptors(
            conn,
            _request(dcp_size, dcp_rank, 4),
            src_block_ids,
            dst_block_ids,
        )[0]
        for src_addr, _, size in first_region:
            start = (src_addr - REGION_BASES[0]) // TOKEN_BYTES[0]
            covered.extend(range(start, start + size // TOKEN_BYTES[0]))

    expected = {
        block_id * BLOCK_SIZE + j
        for block_id in src_block_ids
        for j in range(BLOCK_SIZE)
    }
    assert sorted(covered) == sorted(expected)


def test_without_dcp_the_descriptors_stay_whole_blocks():
    """The non-DCP path is the entire existing PD fleet: byte-identical output,
    one descriptor per block per region, no token arithmetic and no merging of
    the consecutive ids."""
    src_block_ids = [40, 41, 42]
    dst_block_ids = [11, 12, 13]

    for request_data in (
        {"consumer_base_addrs": list(CONSUMER_BASES), "consumer_num_layers": 2},
        _request(1, 0, BLOCK_SIZE),
    ):
        conn = _connector()
        assert conn._execute_block_transfer(
            request_data, "host:1", src_block_ids, dst_block_ids, "req-0"
        )
        (batch,) = conn.written
        assert batch == [
            (
                REGION_BASES[region_idx] + sb * TOKEN_BYTES[region_idx] * BLOCK_SIZE,
                CONSUMER_BASES[region_idx] + db * TOKEN_BYTES[region_idx] * BLOCK_SIZE,
                TOKEN_BYTES[region_idx] * BLOCK_SIZE,
            )
            for region_idx in range(len(REGION_BASES))
            for sb, db in zip(src_block_ids, dst_block_ids)
        ]


def _consumer(dcp_size, dcp_rank, interleave, replicate_index=False):
    """A consumer with only the state the write_request path reads."""
    conn = object.__new__(MooncakeConnector)
    conn.is_producer = False
    conn.tp_size = 4
    conn.tp_rank = dcp_rank
    conn.dcp_size = dcp_size
    conn.dcp_rank = dcp_rank
    conn.dcp_interleave_size = interleave
    conn.replicate_index_cache = replicate_index
    conn.local_ip = "10.0.0.1"
    conn.rpc_port = 7000
    conn._notification_port = 7001
    conn._num_local_layers = len(CONSUMER_BASES)
    conn._block_region_roles = [MLA_KV_ROLE, INDEX_CACHE_ROLE]
    conn._has_slot_regions = False
    conn.kv_caches_base_addr = list(CONSUMER_BASES)
    conn._completion_lock = threading.Lock()
    conn._pending_recv_expected = {}
    conn._pending_recv_nonce = {}
    conn._release_targets = {}
    conn._pending_recv = set()
    conn._pending_recv_blocks = {}
    conn._pending_recv_slots = {}
    conn.sent = []
    conn._send_on_socket = lambda addr, frames: conn.sent.append(frames)
    return conn


def _write_request(dcp_size, dcp_rank, interleave, src_block_ids, dst_block_ids, off):
    conn = _consumer(dcp_size, dcp_rank, interleave)
    metadata = ConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="req-0",
        local_block_ids=dst_block_ids,
        kv_transfer_params={
            "remote_block_ids": src_block_ids,
            "remote_host": "10.0.0.2",
            "remote_handshake_port": 6301,
            "tp_size": 1,
            "remote_pp_size": 1,
            "transfer_id": 1,
            "num_computed_blocks": off,
        },
    )
    conn.start_load_kv(metadata)
    (frames,) = conn.sent
    return msgpack.loads(frames[1], raw=False)


@pytest.mark.parametrize("dcp_size", [1, 2, 4])
def test_the_consumer_skips_dcp_size_source_blocks_per_cached_block(dcp_size):
    """One cached destination block is a virtual block: the producer must skip
    ``dcp_size`` of its own blocks for it, or it re-sends a prefix the consumer
    already has and shifts every later block by that much."""
    src_block_ids = list(range(100, 116))
    dst_block_ids = list(range(50, 50 + 16 // dcp_size))
    off = 2

    body = _write_request(dcp_size, 0, 4, src_block_ids, dst_block_ids, off)

    assert body["num_computed_blocks"] == off
    assert body["dst_block_ids"] == dst_block_ids[off:]
    assert body["src_block_ids"] == src_block_ids[off * dcp_size :]
    assert len(body["dst_block_ids"]) == -(-len(body["src_block_ids"]) // dcp_size)


def test_a_short_remote_list_falls_back_to_a_full_transfer():
    """The remote block list crosses a node boundary, so it can disagree with
    what the local offset assumes. Sending the whole thing costs bandwidth; the
    alternative is a src/dst pair the producer can only reject."""
    src_block_ids = list(range(100, 106))  # fewer than off * dcp_size = 8
    dst_block_ids = list(range(50, 54))

    body = _write_request(4, 0, 4, src_block_ids, dst_block_ids, off=2)

    assert body["num_computed_blocks"] == 0
    assert body["src_block_ids"] == src_block_ids
    assert body["dst_block_ids"] == dst_block_ids


@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("dcp_rank", [0, 3])
def test_the_producer_serves_the_request_the_consumer_actually_sent(dcp_size, dcp_rank):
    """Close the loop: the descriptors come out of the consumer's own payload,
    so a renamed or dropped protocol field fails here rather than at runtime."""
    src_block_ids = [40, 3, 17, 62, 9, 51, 28, 6, 33, 12, 71, 5]
    dst_block_ids = [11, 4, 27, 8, 19, 30, 2, 44, 21, 13, 60, 7][: 12 // dcp_size]
    body = _write_request(
        dcp_size, dcp_rank % dcp_size, 4, src_block_ids, dst_block_ids, off=1
    )
    body["consumer_base_addrs"] = list(CONSUMER_BASES)

    conn = _connector()
    assert conn._execute_block_transfer(
        body, "host:1", body["src_block_ids"], body["dst_block_ids"], "req-0"
    )
    (batch,) = conn.written

    src_off, dst_off, run_len = plan_sharded(
        body["src_block_ids"],
        body["dst_block_ids"],
        BLOCK_SIZE,
        dcp_size,
        dcp_rank % dcp_size,
        4,
    )
    if dcp_size == 1:
        assert len(batch) == len(REGION_BASES) * len(body["dst_block_ids"])
    else:
        assert len(batch) == len(REGION_BASES) * len(run_len)
        assert batch[: len(run_len)] == [
            (
                REGION_BASES[0] + int(s) * TOKEN_BYTES[0],
                CONSUMER_BASES[0] + int(d) * TOKEN_BYTES[0],
                int(n) * TOKEN_BYTES[0],
            )
            for s, d, n in zip(src_off, dst_off, run_len)
        ]


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("interleave", [1, 4, BLOCK_SIZE])
def test_a_replicated_index_region_goes_whole_to_every_rank(dcp_size, interleave):
    """With a replicated index cache the two regions take different plans.

    The MLA latent stays interleave-sharded, so a rank still receives only its
    own 1/W of it. The indexer page is held whole by every rank, so it is the
    concatenation of all W source blocks, laid out in the destination's wider
    token space -- identical on every rank, and independent of the interleave.
    """
    src_block_ids = [40, 3, 17, 62, 9, 51, 28, 6, 33]
    n_dst = -(-len(src_block_ids) // dcp_size)
    dst_block_ids = [11, 4, 27][:n_dst]

    index_per_rank = []
    for dcp_rank in range(dcp_size):
        conn = _connector()
        assert conn._execute_block_transfer(
            _request(dcp_size, dcp_rank, interleave, replicate_index=True),
            "host:1",
            src_block_ids,
            dst_block_ids,
            "req-0",
        )
        (batch,) = conn.written
        # The two regions no longer contribute the same number of descriptors,
        # so split on the sharded plan's length rather than in half.
        tok = TOKEN_BYTES[0]
        expected_mla = [
            (
                REGION_BASES[0] + int(so) * tok,
                CONSUMER_BASES[0] + int(do) * tok,
                int(n) * tok,
            )
            for so, do, n in zip(
                *plan_sharded(
                    src_block_ids,
                    dst_block_ids,
                    BLOCK_SIZE,
                    dcp_size,
                    dcp_rank,
                    interleave,
                )
            )
        ]
        mla, index = batch[: len(expected_mla)], batch[len(expected_mla) :]
        assert mla == expected_mla

        tok = TOKEN_BYTES[1]
        assert index == [
            (
                REGION_BASES[1] + int(so) * tok,
                CONSUMER_BASES[1] + int(do) * tok,
                int(n) * tok,
            )
            for so, do, n in zip(
                *plan_replicated(src_block_ids, dst_block_ids, BLOCK_SIZE, dcp_size)
            )
        ]
        index_per_rank.append(index)

    assert all(r == index_per_rank[0] for r in index_per_rank)
    # Every source block the destination table can hold reaches the rank
    # exactly once -- W times the bytes a sharded index region would carry.
    # A dst list too short to cover every source block clips the tail, which
    # is what plan_replicated's `keep` mask is for.
    covered = min(len(src_block_ids), len(dst_block_ids) * dcp_size)
    assert sum(n for _, _, n in index_per_rank[0]) == (
        covered * BLOCK_SIZE * TOKEN_BYTES[1]
    )


def test_a_sharded_index_region_is_unchanged_when_the_consumer_does_not_replicate():
    """The flag is per-consumer: without it the indexer takes the sharded plan."""
    src_block_ids = [40, 3, 17, 62]
    dst_block_ids = [11]

    conn = _connector()
    plain = _descriptors(conn, _request(4, 2, 1), src_block_ids, dst_block_ids)
    conn = _connector()
    explicit_off = _descriptors(
        conn,
        _request(4, 2, 1, replicate_index=False),
        src_block_ids,
        dst_block_ids,
    )
    assert plain == explicit_off


def test_a_consumer_region_list_in_another_order_is_rejected():
    """Region counts match whichever order the two sides register in, so a
    count check passes a mapping that pairs the latent with the indexer. The
    plan is chosen from the producer's role, and the wrong one writes real KV
    into the index page without faulting."""
    request_data = _request(4, 2, 1, replicate_index=True)
    request_data["consumer_region_roles"] = [INDEX_CACHE_ROLE, MLA_KV_ROLE]

    with pytest.raises(RuntimeError, match="Region role mismatch"):
        _connector()._execute_block_transfer(
            request_data, "host:1", [40, 3, 17, 62], [11], "req-0"
        )


def test_a_region_whose_block_does_not_split_into_tokens_is_rejected():
    """The relayout addresses single tokens, which needs a token's bytes
    contiguous. On a layout that interleaves them the per-token size truncates
    and every descriptor is short by the remainder."""
    conn = _connector()
    conn._per_block_bytes_list = [TOKEN_BYTES[0] * BLOCK_SIZE, 145]

    with pytest.raises(RuntimeError, match="bytes per block"):
        conn._execute_block_transfer(
            _request(2, 0, 1), "host:1", [40, 3, 17, 62], [11, 4], "req-0"
        )
