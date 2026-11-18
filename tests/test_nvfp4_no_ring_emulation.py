# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_MLA_NVFP4_EMULATE_NO_RING: the tail holds what the FP4 cache holds."""

import numpy as np
import torch

from atom.model_ops.attentions.nvfp4_mla import Nvfp4MlaCache, nvfp4_roundtrip

DIM = 576
GAIN = np.float32(89.6)

# The producer's NumPy oracle, from AITER PR 7987 op_tests/test_mla_nvfp4.py.
F4 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)


def _e4(c):
    c = np.asarray(c, np.uint8)
    e, m = (c >> 3).astype(np.int32), (c & 7).astype(np.float32)
    return np.where(
        c == 127, np.nan, np.where(e == 0, m / 512, (1 + m / 8) * np.exp2(e - 7))
    ).astype(np.float32)


S8 = _e4(np.arange(127))


def _nearest(x, table):
    hi = np.searchsorted(table, x).clip(0, len(table) - 1)
    lo = (hi - 1).clip(0)
    a, b = x - table[lo], table[hi] - x
    return np.where((a < b) | ((a == b) & ((lo & 1) == 0)), lo, hi).astype(np.uint8)


def _quant_blocks(x, ideal):
    am = np.max(np.abs(x), axis=-1)
    div = np.where(am == 0, np.float32(1), am / np.float32(6))
    sf = _nearest(np.clip(GAIN * div, 1 / 512, 448), S8)
    if not ideal:
        div = S8[sf] / GAIN
    scaled = x / div[..., None]
    code = _nearest(np.abs(scaled), F4[:8]) | (np.signbit(scaled).astype(np.uint8) * 8)
    return F4[code] * S8[sf][..., None] / GAIN


def _oracle(tile):
    x = tile.astype(np.float32)
    nope = x[:, :512].reshape(16, 32, 16).transpose(1, 0, 2).reshape(32, 256)
    nope = _quant_blocks(nope, True).reshape(32, 16, 16).transpose(1, 0, 2)
    rope = x[:, 512:].reshape(16, 4, 16)
    base = _quant_blocks(rope, False)
    rope = base + _quant_blocks(rope - base, False)
    return np.concatenate((nope.reshape(16, 512), rope.reshape(16, 64)), axis=1)


def _tiles(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 16, DIM, generator=gen) * 0.1
    # Heavy-tailed channels and exact zeros, like a real latent.
    x *= torch.exp(torch.randn(1, 1, DIM, generator=gen))
    x[:, :, ::37] = 0
    valid = torch.randint(1, 17, (n,), generator=gen)
    x[torch.arange(16)[None, :] >= valid[:, None]] = 0
    return x.to(torch.bfloat16)


def test_roundtrip_matches_the_producer_oracle_bit_for_bit():
    tiles = _tiles(64)
    got = nvfp4_roundtrip(tiles)
    for i in range(tiles.shape[0]):
        ref = torch.from_numpy(_oracle(tiles[i].float().numpy())).to(torch.bfloat16)
        assert torch.equal(got[i], ref), i


def _cache(emulate):
    cache = Nvfp4MlaCache(1, 8, 4, DIM, device="cpu", emulate_no_ring=emulate)
    writes = []
    cache._write = lambda layer, tiles, *_: writes.append(tiles.clone())
    return cache, writes


def test_decode_requantizes_the_fp4_of_earlier_rows():
    tokens = _tiles(1, seed=1)[0]
    tokens[:] = (torch.randn(16, DIM) * torch.linspace(0.05, 2.0, 16)[:, None]).to(
        torch.bfloat16
    )
    cache, writes = _cache(emulate=True)
    row = torch.as_tensor(cache.bind_rows([7]))
    for pos in range(16):
        cache.publish_decode(
            0, row, tokens[pos : pos + 1], torch.tensor([pos]), torch.tensor([pos])
        )
    for pos in range(1, 16):
        # Earlier rows are the round trip of the previous write, not originals.
        assert torch.equal(
            writes[pos][0, :pos], nvfp4_roundtrip(writes[pos - 1])[0, :pos]
        )
        assert torch.equal(writes[pos][0, pos], tokens[pos])
    assert not torch.equal(writes[15][0, :15], tokens[:15])


def test_decode_without_emulation_keeps_originals():
    tokens = _tiles(1, seed=2)[0]
    tokens[:] = (torch.randn(16, DIM) * 0.3).to(torch.bfloat16)
    cache, writes = _cache(emulate=False)
    row = torch.as_tensor(cache.bind_rows([7]))
    for pos in range(16):
        cache.publish_decode(
            0, row, tokens[pos : pos + 1], torch.tensor([pos]), torch.tensor([pos])
        )
    assert torch.equal(writes[15][0], tokens)


def test_prefill_short_tile_tail_is_its_round_trip():
    lengths, tables = [21, 16, 5], [[3, 4], [1], [6]]
    kv = (torch.randn(sum(lengths), DIM) * 0.3).to(torch.bfloat16)
    cache, _ = _cache(emulate=True)
    rows = cache.bind_rows([1, 2, 3])
    plan = cache.upload_plan(cache.plan_prefill(rows, [0, 0, 0], lengths, tables))
    cache.publish_prefill(0, kv, plan)
    t0 = 0
    for row, n in zip(rows, lengths):
        short = n % 16
        if short:
            tile = torch.zeros(1, 16, DIM, dtype=torch.bfloat16)
            tile[0, :short] = kv[t0 + n - short : t0 + n]
            assert torch.equal(
                cache.ring.buf[row, 0, :short], nvfp4_roundtrip(tile)[0, :short]
            )
        t0 += n
