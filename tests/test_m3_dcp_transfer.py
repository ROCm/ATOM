# SPDX-License-Identifier: MIT
"""Copy real PP4 cache bytes into token shards; read independent physical views."""

import json

import pytest
import test_m3_kv_transfer_layout as cache
import torch


def logical_bytes(role, view, *, nvfp4, dtype):
    """Decode native KV formats without using the transfer's token metadata."""
    if ".index" in role:
        return view.reshape(cache.BLOCKS, cache.BLOCK, -1)
    draft = role.startswith("draft.")
    field = role.split(".")[1 if draft else 2]
    sparse = not draft and int(role.split(".")[1][6:]) in cache.SPARSE
    packed = sparse and nvfp4
    page = 128 if packed or not (draft or sparse) else 16
    subpages, heads = 128 // page, 4
    if packed or field.endswith("scale"):
        width = (8 if field.endswith("scale") else 64) if packed else 4
        return (
            view.reshape(cache.BLOCKS, subpages, heads, page, width)
            .permute(0, 1, 3, 2, 4)
            .reshape(cache.BLOCKS, 128, -1)
        )
    size = 1 if draft else dtype.itemsize
    x = 16 // size
    if field == "k":
        return (
            view.reshape(cache.BLOCKS, subpages, heads, 128 // x, page, x * size)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(cache.BLOCKS, 128, -1)
        )
    return (
        view.reshape(cache.BLOCKS, subpages, heads, page // x, 128, x, size)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .reshape(cache.BLOCKS, 128, -1)
    )


def endpoints(held, nvfp4, dtype, rank, world, interleave):
    tt, source = cache._layout(held, 4, nvfp4=nvfp4, dtype=dtype, draft=7 in held)
    cache._fill(source)
    consumer, destination = cache._layout(
        range(8), 4, nvfp4=nvfp4, dtype=dtype, draft=True
    )
    cache._fill(destination, zeros=True)
    c, req = cache._producer(tt, 4)
    c._block_token_layouts = [p.token_layout for p in tt.pages]
    cache._request(req, consumer, rank)
    req.update(
        consumer_dcp_size=world,
        consumer_dcp_rank=rank,
        consumer_dcp_interleave=interleave,
        consumer_token_layouts=json.loads(
            json.dumps([p.token_layout for p in consumer.pages])
        ),
    )
    return c, req, source, destination


@pytest.mark.parametrize("held", cache.STAGES)
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 16, 128])
@pytest.mark.parametrize(
    "nvfp4,dtype",
    [
        (False, torch.float8_e4m3fn),
        (False, torch.bfloat16),
        (True, torch.float8_e4m3fn),
    ],
)
def test_pp4_dcp_real_cache_bytes(held, world, interleave, nvfp4, dtype):
    src_ids, dst_ids = [3, 1, 2], [2, 0][: (3 + world - 1) // world]
    for rank in range(world):
        c, req, source, destination = endpoints(
            held, nvfp4, dtype, rank, world, interleave
        )
        assert c._execute_block_transfer(req, "cpu", src_ids, dst_ids, "dcp-test")
        for role, dst in destination.items():
            got = logical_bytes(role, dst, nvfp4=nvfp4, dtype=dtype)
            expected = torch.zeros_like(got)
            if role in source:
                src = logical_bytes(role, source[role], nvfp4=nvfp4, dtype=dtype)
                local = 0
                for pos in range(len(src_ids) * cache.BLOCK):
                    if (pos // interleave) % world == rank:
                        expected[dst_ids[local // cache.BLOCK], local % cache.BLOCK] = (
                            src[src_ids[pos // cache.BLOCK], pos % cache.BLOCK]
                        )
                        local += 1
            assert torch.equal(got, expected), (role, world, rank, interleave)


@pytest.mark.parametrize("bad", ["missing", "axes", "shape", "last_role", "last_width"])
def test_malformed_layout_fails_before_any_write(bad):
    c, req, _source, destination = endpoints(
        [6, 7], True, torch.float8_e4m3fn, 1, 4, 128
    )
    if bad == "missing":
        del req["consumer_token_layouts"]
    elif bad == "axes":
        req["consumer_token_layouts"][-1][1] = [99]
    elif bad == "shape":
        req["consumer_token_layouts"][-1][0][0] += 1
    elif bad == "last_role":
        req["consumer_region_roles"][-1] = "wrong"
    else:
        req["consumer_block_bpb"][-1] += 1
    with pytest.raises((ValueError, RuntimeError)):
        c._execute_block_transfer(req, "cpu", [3, 1, 2], [2], "bad-dcp")
    assert c.transfer_engine.writes.empty()
    assert all(not v.any() for v in destination.values())


@pytest.mark.parametrize("cached", [0, 1])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("interleave", [16, 128])
def test_chunked_dcp_prefix_counts_source_blocks_and_waits_for_handoff(
    cached, rank, interleave
):
    from concurrent.futures import ThreadPoolExecutor

    import test_pd_chunked_transfer as protocol

    from atom.kv_transfer.disaggregation.chunked_prefill import ChunkedPrefill

    c, req, source, destination = endpoints(
        [6, 7], True, torch.float8_e4m3fn, rank, 2, interleave
    )
    c._MAX_RDMA_ENTRIES_PER_BATCH = 100000
    src_ids, dst_ids = [3, 1, 2], [2, 0]
    req.update(
        num_computed_blocks=cached,
        src_block_skip_factor=2,
        dst_block_ids=dst_ids[cached:],
    )
    state = ChunkedPrefill(7, src_ids, 258, cache.BLOCK)
    c._chunked_prefills["xfer-a"] = state
    protocol.publish(state, 256)
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(protocol.dispatch, c, req)
        try:
            if not cached:
                c.transfer_engine.writes.get(timeout=5)
                assert not future.done()
            protocol.publish(state, 258)
            handoff = {"first_token_id": 42, "draft_token_ids": []}
            state.finish(handoff)
            future.result(timeout=5)
        finally:
            state.cancel()
    c._notify_transfer_result.assert_called_once_with(req, success=True)
    assert req["prefill_handoff"] == handoff
    for role, dst in destination.items():
        actual = logical_bytes(role, dst, nvfp4=True, dtype=torch.float8_e4m3fn)
        expected = torch.zeros_like(actual)
        if role in source:
            src = logical_bytes(
                role, source[role], nvfp4=True, dtype=torch.float8_e4m3fn
            )
            local = 0
            for pos in range(len(src_ids) * 128):
                if (pos // interleave) % 2 == rank:
                    if pos >= cached * 2 * 128:
                        expected[dst_ids[local // 128], local % 128] = src[
                            src_ids[pos // 128], pos % 128
                        ]
                    local += 1
        assert torch.equal(actual, expected), (role, cached, rank, interleave)


@pytest.mark.parametrize("source_pages", [0, 3, 16389])
@pytest.mark.parametrize("rank", [0, 3])
def test_whole_page_batches_preserve_noncontiguous_ids_and_padding(source_pages, rank):
    import numpy as np

    from atom.kv_transfer.disaggregation.mha_token_transfer import dcp_page_descriptors
    from atom.kv_transfer.disaggregation.sharded_transfer import build_dcp_shard_plan

    src = np.arange(source_pages, dtype=np.int64)[::-1] * 3 + 7
    pages = (source_pages + 3) // 4
    dst = np.arange(pages, dtype=np.int64)[::-1] * 5 + 11
    plan = build_dcp_shard_plan(
        src,
        block_size=128,
        dcp_size=4,
        dcp_rank=rank,
        interleave_size=128,
        dst_pages=pages,
    )
    unit = 4 * 128 * 64
    batches = list(
        dcp_page_descriptors(plan, dst, 100000, 200000, unit, ((4, 128, 64), (1,)))
    )
    expected_source = src[rank::4]
    if not expected_source.size:
        assert not batches
        return
    addresses = [np.concatenate([batch[i] for batch in batches]) for i in range(3)]
    np.testing.assert_array_equal(addresses[0], 100000 + expected_source * unit)
    np.testing.assert_array_equal(
        addresses[1], 200000 + dst[: len(expected_source)] * unit
    )
    np.testing.assert_array_equal(addresses[2], np.full(len(expected_source), unit))
    assert all(len(batch[0]) <= 4096 for batch in batches)
