# SPDX-License-Identifier: MIT
"""Dispatch coverage for V4 FP8 ASM decode paths and split tuning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="paged_decode defines Triton kernels")
pytest.importorskip("aiter", reason="paged_decode imports the AITER runtime")

from aiter.ops import mla_sparse_prefill

from atom.model_ops.v4_kernels import paged_decode


def _dispatch(
    monkeypatch,
    *,
    enabled: bool,
    heads: int = 128,
    gfx: str = "gfx1250",
    with_empty_indptr: bool = True,
):
    monkeypatch.setattr(
        paged_decode.envs, "ATOM_USE_V4_PREFILL_ASM_FOR_DECODE", enabled
    )
    monkeypatch.setattr(paged_decode, "get_gfx", lambda: gfx)
    monkeypatch.setattr(
        paged_decode,
        "_sparse_attn_v4_paged_decode_prefill_asm",
        lambda *args, **kwargs: "prefill",
    )
    monkeypatch.setattr(
        paged_decode,
        "_sparse_attn_v4_paged_decode_asm",
        lambda *args, **kwargs: "decode",
    )

    n = 2
    return paged_decode.sparse_attn_v4_paged_decode(
        q=torch.empty((n, heads, 512)),
        unified_kv=torch.empty((4, 512)),
        kv_indices=torch.empty(0, dtype=torch.int32),
        kv_indptr=torch.zeros(n + 1, dtype=torch.int32),
        attn_sink=torch.empty(heads),
        softmax_scale=512**-0.5,
        unified_kv_rope=torch.empty((4, 64)),
        q_packed_in=torch.empty((n, heads, 512)),
        q_rope_in=torch.empty((n, heads, 64)),
        qo_indptr=torch.arange(n + 1, dtype=torch.int32),
        empty_kv_indptr=(
            torch.zeros(n + 1, dtype=torch.int32) if with_empty_indptr else None
        ),
    )


def test_enabled_ep4_head128_uses_prefill_asm(monkeypatch):
    assert _dispatch(monkeypatch, enabled=True) == "prefill"


def test_decode_csr_becomes_prefix_and_extend_is_empty(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_prefill_asm(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        mla_sparse_prefill, "mla_sparse_prefill_fp8_asm", fake_prefill_asm
    )
    n, h = 2, 128
    q_nope = torch.empty((n, h, 512))
    q_rope = torch.empty((n, h, 64))
    unified_kv = torch.empty((4, 512))
    unified_kv_rope = torch.empty((4, 64))
    kv_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
    empty_kv_indptr = torch.zeros(n + 1, dtype=torch.int32)

    result = paged_decode._sparse_attn_v4_paged_decode_prefill_asm(
        unified_kv,
        kv_indices,
        kv_indptr,
        empty_kv_indptr,
        torch.empty(h),
        512**-0.5,
        unified_kv_rope,
        q_nope,
        q_rope,
    )

    assert result is sentinel
    assert torch.equal(captured["kv_indptr_prefix"], kv_indptr)
    assert torch.equal(captured["kv_indices_prefix"], kv_indices)
    assert captured["kv_nope"] is unified_kv
    assert captured["kv_rope"] is unified_kv_rope
    assert captured["kv_indices_extend"].numel() == 0
    assert torch.count_nonzero(captured["kv_indptr_extend"]) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": False},
        {"enabled": True, "heads": 64},
        {"enabled": True, "gfx": "gfx950"},
        {"enabled": True, "with_empty_indptr": False},
    ],
)
def test_ineligible_decode_keeps_dedicated_asm(monkeypatch, kwargs):
    assert _dispatch(monkeypatch, **kwargs) == "decode"


@pytest.mark.parametrize(
    "requests,expected",
    [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (32, 1), (64, 1), (256, 1)],
)
def test_aiter_csa_uses_graphsafe_batch_split_table(monkeypatch, requests, expected):
    monkeypatch.setattr(paged_decode, "_device_arch", lambda _index: "gfx950")
    q_packed = SimpleNamespace(
        shape=(requests * 7, 128, 512), device=SimpleNamespace(index=0)
    )

    assert (
        paged_decode._v4_aiter_fp8_decode_splits(
            q_packed,
            query_group=7,
            kv_kind="csa",
        )
        == expected
    )


@pytest.mark.parametrize(
    "tokens,heads,query_group,kv_kind,arch",
    [
        (6 * 7, 128, 7, "hca", "gfx950"),
        (6 * 7, 64, 7, "csa", "gfx950"),
        (6 * 4, 128, 4, "csa", "gfx950"),
        (6 * 7, 128, 7, "csa", "gfx1250"),
    ],
)
def test_aiter_split_table_rejects_unqualified_shapes(
    monkeypatch, tokens, heads, query_group, kv_kind, arch
):
    monkeypatch.setattr(paged_decode, "_device_arch", lambda _index: arch)
    q_packed = SimpleNamespace(
        shape=(tokens, heads, 512), device=SimpleNamespace(index=0)
    )

    assert (
        paged_decode._v4_aiter_fp8_decode_splits(
            q_packed,
            query_group=query_group,
            kv_kind=kv_kind,
        )
        is None
    )


def test_dedicated_asm_receives_selected_split_count(monkeypatch):
    captured = {}

    monkeypatch.setattr(paged_decode.envs, "ATOM_USE_V4_PREFILL_ASM_FOR_DECODE", False)
    monkeypatch.setattr(
        paged_decode,
        "_v4_aiter_fp8_decode_splits",
        lambda q_packed, *, query_group, kv_kind: 3,
    )

    def fake_decode(*args, **kwargs):
        captured.update(kwargs)
        return "decode"

    monkeypatch.setattr(paged_decode, "_sparse_attn_v4_paged_decode_asm", fake_decode)
    n, heads = 21, 128
    result = paged_decode.sparse_attn_v4_paged_decode(
        q=None,
        unified_kv=torch.empty((4, 512)),
        kv_indices=torch.empty(0, dtype=torch.int32),
        kv_indptr=torch.zeros(n + 1, dtype=torch.int32),
        attn_sink=torch.empty(heads),
        softmax_scale=512**-0.5,
        unified_kv_rope=torch.empty((4, 64)),
        q_packed_in=torch.empty((n, heads, 512)),
        q_rope_in=torch.empty((n, heads, 64)),
        qo_indptr=torch.arange(n + 1, dtype=torch.int32),
        query_group=7,
        kv_kind="csa",
    )

    assert result == "decode"
    assert captured["num_kv_splits"] == 3
