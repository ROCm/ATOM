# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NVFP4 decode split-K: page ranges per request, merged back by LSE."""

from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.attentions.nvfp4_mla import (
    Nvfp4MlaCache,
    merge_splits,
    nvfp4_split_count,
    split_page_csr,
    try_nvfp4_decode,
)


def _base_csr(lens):
    pages = [(n + 63) // 64 for n in lens]
    indptr = torch.tensor(
        [0] + list(torch.tensor(pages).cumsum(0).tolist()), dtype=torch.int32
    )
    last = torch.tensor(
        [0 if n == 0 else n - (p - 1) * 64 for n, p in zip(lens, pages)],
        dtype=torch.int32,
    )
    return indptr, last, pages


def test_split_count_fills_the_row_budget():
    assert nvfp4_split_count(1536, 2048, 16) == 1
    assert nvfp4_split_count(600, 2048, 16) == 3
    assert nvfp4_split_count(140, 2048, 16) == 14
    assert nvfp4_split_count(20, 2048, 16) == 16
    assert nvfp4_split_count(0, 2048, 16) == 1


@pytest.mark.parametrize("splits", [1, 2, 3, 8])
def test_split_ranges_tile_each_request_in_order(splits):
    lens = [1, 64, 65, 200, 0, 1000]
    indptr, last, pages = _base_csr(lens)
    rows = len(lens) * splits
    out_indptr = torch.full((rows + 1,), -7, dtype=torch.int32)
    out_last = torch.full((rows,), -7, dtype=torch.int32)
    split_page_csr(indptr, last, splits, out_indptr, out_last)

    assert out_indptr[0] == 0 and out_indptr[-1] == indptr[-1]
    assert bool((out_indptr[1:] >= out_indptr[:-1]).all())
    for b, n in enumerate(lens):
        first = out_indptr[b * splits].item()
        stop = out_indptr[(b + 1) * splits].item()
        # The request's ranges cover exactly its pages, in order.
        assert (first, stop) == (indptr[b].item(), indptr[b + 1].item())
        counts = [
            (out_indptr[b * splits + s + 1] - out_indptr[b * splits + s]).item()
            for s in range(splits)
        ]
        assert sum(counts) == pages[b]
        assert max(counts) - min(counts) <= 1
        for s, count in enumerate(counts):
            got = out_last[b * splits + s].item()
            ends_request = sum(counts[: s + 1]) == pages[b]
            if count == 0:
                assert got == 0
            elif ends_request:
                assert got == last[b].item()
            else:
                assert got == 64


def _attend(logits, values):
    """logits [B, H, T], values [B, T, D] -> (out [B, H, D], lse [B, H])."""
    if logits.shape[-1] == 0:
        b, h = logits.shape[:2]
        return torch.zeros(b, h, values.shape[-1]), torch.full((b, h), float("-inf"))
    return torch.softmax(logits, -1) @ values, torch.logsumexp(logits, -1)


@pytest.mark.parametrize("bounds", [[0, 7, 19, 40], [0, 0, 25, 40], [0, 40, 40, 40]])
def test_merge_restores_the_full_softmax(bounds):
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 4, 40, generator=gen) * 3
    values = torch.randn(2, 40, 16, generator=gen)
    full, _ = _attend(logits, values)
    parts = [_attend(logits[..., a:b], values[:, a:b]) for a, b in pairwise(bounds)]
    out = torch.stack([p[0] for p in parts], dim=1)
    lse = torch.stack([p[1] for p in parts], dim=1)
    assert torch.allclose(merge_splits(out, lse), full, atol=1e-5)


def test_merge_of_a_request_with_no_keys_is_zero():
    out = torch.zeros(1, 3, 2, 4)
    lse = torch.full((1, 3, 2), float("-inf"))
    assert merge_splits(out, lse).abs().sum() == 0


def test_decode_passes_the_split_rows_to_the_shader():
    cache = Nvfp4MlaCache(1, 8, 4, 576, device="cpu")
    cache._write = lambda *args: None
    seen = {}

    def fake_decode(layer, q, csr, sm_scale, splits=1):
        seen.update(rows=q.shape[0], splits=splits, csr=tuple(t.shape[0] for t in csr))
        return torch.zeros(q.shape[0], 128, 512, dtype=torch.bfloat16)

    cache.decode = fake_decode
    attn = SimpleNamespace(
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
    rows, splits = 2, 3
    meta = SimpleNamespace(
        max_seqlen_q=1,
        nvfp4_tail_slots=torch.as_tensor(cache.bind_rows([5, 6])),
        nvfp4_splits=splits,
        nvfp4_csr=(
            torch.zeros(32, dtype=torch.int32),
            torch.zeros(16, dtype=torch.int32),
            torch.zeros(31, dtype=torch.int32),
        ),
        slot_mapping=torch.tensor([0, 64]),
    )
    bf16 = {"dtype": torch.bfloat16}
    out = try_nvfp4_decode(
        attn,
        torch.zeros(rows, 128, 512, **bf16),
        torch.zeros(rows, 128, 64, **bf16),
        torch.zeros(rows, 512, **bf16),
        torch.zeros(rows, 1, 64, **bf16),
        torch.zeros(rows, dtype=torch.long),
        meta,
    )
    assert tuple(out.shape) == (rows, 128, 512)
    assert seen == {
        "rows": rows,
        "splits": splits,
        "csr": (rows * splits + 1, 16, rows * splits),
    }
