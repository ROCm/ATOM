# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The NVFP4 BF16 tail follows the request, not its batch row."""

import pytest
import torch

from atom.model_ops.attentions.nvfp4_mla import Nvfp4MlaCache

DIM = 576


def _cache(max_reqs=4, pages=8):
    cache = Nvfp4MlaCache(1, pages, max_reqs, DIM, device="cpu")
    writes = []

    def record(layer, tiles, pages, tile_ids, valid):
        writes.append(
            {
                "tiles": tiles.clone(),
                "pages": pages.tolist(),
                "tile_ids": tile_ids.tolist(),
                "valid": valid.tolist(),
            }
        )

    cache._write = record
    return cache, writes


def _tok(value):
    return torch.full((DIM,), value, dtype=torch.bfloat16)


def _decode(cache, req_ids, values, positions, pages):
    rows = torch.as_tensor(cache.bind_rows(req_ids))
    slots = torch.tensor([page * 64 + pos % 64 for page, pos in zip(pages, positions)])
    cache.publish_decode(
        0,
        rows,
        torch.stack([_tok(v) for v in values]),
        torch.tensor(positions),
        slots,
    )


def test_reordered_rows_keep_their_own_tail():
    cache, writes = _cache()
    _decode(cache, [11, 22], [1.0, 2.0], [0, 0], [0, 1])
    # Request 22 is now batch row 0 and request 11 is row 1.
    _decode(cache, [22, 11], [3.0, 4.0], [1, 1], [1, 0])

    last = writes[-1]
    assert last["valid"] == [2, 2]
    assert last["pages"] == [1, 0]
    b, a = last["tiles"]
    assert b[0].float().unique().tolist() == [2.0]
    assert b[1].float().unique().tolist() == [3.0]
    assert a[0].float().unique().tolist() == [1.0]
    assert a[1].float().unique().tolist() == [4.0]
    assert b[2:].abs().sum() == 0 and a[2:].abs().sum() == 0


def test_released_row_does_not_leak_into_new_request():
    cache, writes = _cache()
    _decode(cache, [11, 22], [1.0, 2.0], [0, 0], [0, 1])
    _decode(cache, [11, 22], [5.0, 6.0], [1, 1], [0, 1])
    cache.release([11])
    # Request 33 takes 11's ring row and starts its own tile at position 0.
    _decode(cache, [22, 33], [7.0, 9.0], [2, 0], [1, 2])

    last = writes[-1]
    assert last["valid"] == [3, 1]
    b, c = last["tiles"]
    assert [b[i].float().unique().item() for i in range(3)] == [2.0, 6.0, 7.0]
    assert c[0].float().unique().tolist() == [9.0]
    assert c[1:].abs().sum() == 0


def test_chunked_prefill_continues_the_same_tile():
    cache, writes = _cache()
    row = int(cache.bind_rows([5])[0])
    first = torch.stack([_tok(float(i)) for i in range(10)])
    cache.publish_tokens(0, row, first, torch.arange(10), torch.arange(10))
    second = torch.stack([_tok(float(i)) for i in range(10, 21)])
    cache.publish_tokens(0, row, second, torch.arange(10, 21), torch.arange(10, 21))

    tail = writes[-1]
    assert tail["valid"] == [16, 5]
    assert tail["tile_ids"] == [0, 1]
    full, short = tail["tiles"]
    assert [full[i].float().unique().item() for i in range(16)] == [
        float(i) for i in range(16)
    ]
    assert [short[i].float().unique().item() for i in range(5)] == [
        float(i) for i in range(16, 21)
    ]
    assert short[5:].abs().sum() == 0


def _tiles_by_key(writes):
    out = {}
    for w in writes:
        for i, (page, tile) in enumerate(zip(w["pages"], w["tile_ids"])):
            v = w["valid"][i]
            out[(page, tile)] = (v, w["tiles"][i][:v].clone())
    return out


def test_prefill_plan_writes_what_the_per_request_path_writes():
    lengths = [21, 16, 5]
    tables = [[3, 4], [1], [6]]
    kv = torch.stack([_tok(float(i)) for i in range(sum(lengths))])

    loop, loop_writes = _cache(max_reqs=4)
    rows = loop.bind_rows([1, 2, 3])
    t0 = 0
    for row, n, table in zip(rows, lengths, tables):
        slots = torch.tensor([table[p // 64] * 64 + p % 64 for p in range(n)])
        loop.publish_tokens(0, int(row), kv[t0 : t0 + n], torch.arange(n), slots)
        t0 += n

    plan_cache, plan_writes = _cache(max_reqs=4)
    rows = plan_cache.bind_rows([1, 2, 3])
    plan = plan_cache.upload_plan(
        plan_cache.plan_prefill(rows, [0, 0, 0], lengths, tables)
    )
    plan_cache.publish_prefill(0, kv, plan)
    assert len(plan_writes) == 1  # one producer launch for the step

    expect, got = _tiles_by_key(loop_writes), _tiles_by_key(plan_writes)
    assert expect.keys() == got.keys() == {(3, 0), (3, 1), (1, 0), (6, 0)}
    for key in expect:
        assert expect[key][0] == got[key][0]
        assert torch.equal(expect[key][1], got[key][1])
    # The short last tiles' BF16 is what the next decode token requantizes.
    for row, n in zip(rows, lengths):
        short = n % 16
        assert torch.equal(
            loop.ring.buf[row, 0, :short], plan_cache.ring.buf[row, 0, :short]
        )


def test_prefill_plan_refuses_a_mid_tile_start():
    cache, _ = _cache()
    with pytest.raises(ValueError, match="tile start"):
        cache.plan_prefill([0], [8], [4], [[0]])


def test_padding_rows_bind_to_the_scratch_row():
    cache, _ = _cache(max_reqs=4)
    assert cache.pad_row == 4
    assert cache.bind_rows([11], pad_to=3).tolist() == [0, 4, 4]
    assert cache.bind_rows([], pad_to=2).tolist() == [4, 4]
    with pytest.raises(ValueError, match="pad_to"):
        cache.bind_rows([11, 22], pad_to=1)


def test_padding_rows_skip_pages_and_leave_live_tails_alone():
    cache, writes = _cache(max_reqs=4)
    rows = torch.as_tensor(cache.bind_rows([11], pad_to=3))
    cache.publish_decode(
        0,
        rows,
        torch.stack([_tok(1.0), _tok(5.0), _tok(6.0)]),
        torch.tensor([0, 0, 0]),
        torch.tensor([0, -1, -1]),
    )
    # slot -1 -> page -1, which the AITER producer drops on device.
    assert writes[-1]["pages"] == [0, -1, -1]
    ring = cache.ring.buf
    assert ring[0, 0, 0].float().unique().tolist() == [1.0]
    assert ring[1:4].abs().sum() == 0


def _fake_attn(cache):
    from types import SimpleNamespace

    return SimpleNamespace(
        nvfp4=cache,
        nvfp4_layer=0,
        num_heads=128,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        scale=1.0,
        _q_scale=None,
        rotary_emb=lambda positions, q, k: None,
        _v_up_proj_and_o_proj=lambda o: o,
    )


def test_decode_runs_every_padded_row_with_the_step_csr():
    from types import SimpleNamespace

    from atom.model_ops.attentions.nvfp4_mla import try_nvfp4_decode

    cache, writes = _cache(max_reqs=4)
    seen = {}

    def fake_decode(layer, q, csr, sm_scale):
        seen["q"] = tuple(q.shape)
        seen["csr"] = tuple(t.shape[0] for t in csr)
        return torch.zeros(q.shape[0], 128, 512, dtype=torch.bfloat16)

    cache.decode = fake_decode
    rows = 3
    meta = SimpleNamespace(
        max_seqlen_q=1,
        nvfp4_tail_slots=torch.as_tensor(cache.bind_rows([11], pad_to=rows)),
        # Builder buffers are max_bs wide; the decode slices them to the batch.
        nvfp4_csr=(
            torch.zeros(9, dtype=torch.int32),
            torch.zeros(16, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int32),
        ),
        slot_mapping=torch.tensor([0, -1, -1]),
    )
    out = try_nvfp4_decode(
        _fake_attn(cache),
        torch.zeros(rows, 128, 512, dtype=torch.bfloat16),
        torch.zeros(rows, 128, 64, dtype=torch.bfloat16),
        torch.zeros(rows, 512, dtype=torch.bfloat16),
        torch.zeros(rows, 1, 64, dtype=torch.bfloat16),
        torch.zeros(rows, dtype=torch.long),
        meta,
    )
    assert tuple(out.shape) == (rows, 128, 512)
    assert seen == {"q": (rows, 128, 576), "csr": (rows + 1, 16, rows)}
    assert writes[-1]["pages"] == [0, -1, -1]


def test_decode_without_csr_is_refused():
    from types import SimpleNamespace

    from atom.model_ops.attentions.nvfp4_mla import try_nvfp4_decode

    cache, _ = _cache(max_reqs=4)
    meta = SimpleNamespace(
        max_seqlen_q=1,
        nvfp4_tail_slots=torch.as_tensor(cache.bind_rows([11])),
        slot_mapping=torch.tensor([0]),
    )
    with pytest.raises(RuntimeError, match="csr missing"):
        try_nvfp4_decode(
            _fake_attn(cache),
            torch.zeros(1, 128, 512, dtype=torch.bfloat16),
            torch.zeros(1, 128, 64, dtype=torch.bfloat16),
            torch.zeros(1, 512, dtype=torch.bfloat16),
            torch.zeros(1, 1, 64, dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.long),
            meta,
        )


def test_ring_full_raises():
    cache, _ = _cache(max_reqs=1)
    cache.bind_rows([1])
    with pytest.raises(RuntimeError, match="ring is full"):
        cache.bind_rows([1, 2])
    cache.release([1])
    assert cache.bind_rows([2]).tolist() == [0]
