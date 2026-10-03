# SPDX-License-Identifier: MIT
"""Dispatch coverage for V4 FP8 ASM decode paths and their split plans."""

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


def _decode_rows(n: int, heads: int = 128) -> dict:
    return dict(
        unified_kv=torch.empty((4, 512)),
        kv_indices=torch.empty(0, dtype=torch.int32),
        kv_indptr=torch.zeros(n + 1, dtype=torch.int32),
        attn_sink=torch.empty(heads),
        softmax_scale=512**-0.5,
        unified_kv_rope=torch.empty((4, 64)),
        q_packed_in=torch.empty((n, heads, 512)),
        q_rope_in=torch.empty((n, heads, 64)),
    )


@pytest.mark.parametrize(
    "split_plan,expected",
    [
        ((4, torch.arange(0, 4 * 22, 4, dtype=torch.int32)), 4),
        (None, None),
    ],
)
def test_split_plan_reaches_decode_asm(monkeypatch, split_plan, expected):
    captured = {}
    monkeypatch.setattr(paged_decode.envs, "ATOM_USE_V4_PREFILL_ASM_FOR_DECODE", False)
    monkeypatch.setattr(
        paged_decode,
        "_sparse_attn_v4_paged_decode_asm",
        lambda *args, **kwargs: captured.update(kwargs) or "decode",
    )
    n = 21
    result = paged_decode.sparse_attn_v4_paged_decode(
        q=None,
        qo_indptr=torch.arange(n + 1, dtype=torch.int32),
        split_plan=split_plan,
        **_decode_rows(n),
    )

    assert result == "decode"
    assert captured["num_kv_splits"] == expected
    assert captured["split_indptr"] is (split_plan[1] if split_plan else None)


def test_decode_asm_trims_split_indptr_to_real_rows(monkeypatch):
    # An eager forward can run fewer rows than the padded grid its plan was
    # built for; the uniform plan's prefix must go along with qo_indptr's.
    import aiter.mla

    captured = {}
    monkeypatch.setattr(
        aiter.mla,
        "mla_decode_fwd_v4_nm",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    n, padded = 3, 8
    rows = _decode_rows(n)
    paged_decode._sparse_attn_v4_paged_decode_asm(
        rows["unified_kv"],
        rows["kv_indices"],
        torch.zeros(padded + 1, dtype=torch.int32),
        rows["attn_sink"],
        rows["softmax_scale"],
        rows["unified_kv_rope"],
        rows["q_packed_in"],
        rows["q_rope_in"],
        qo_indptr=torch.arange(padded + 1, dtype=torch.int32),
        num_kv_splits=2,
        split_indptr=torch.arange(0, 2 * (padded + 1), 2, dtype=torch.int32),
    )

    assert captured["num_kv_splits"] == 2
    assert captured["split_indptr"].tolist() == [0, 2, 4, 6]


@pytest.mark.parametrize("has_planner", [True, False])
def test_v4_decode_split_plan_follows_aiter(monkeypatch, has_planner):
    import aiter.mla

    calls = []

    def planner(rows, heads, kv_len, *, split_indptr):
        calls.append((rows, heads, kv_len))
        return 2, split_indptr[: rows + 1]

    if has_planner:
        monkeypatch.setattr(
            aiter.mla, "get_mla_v4_nm_split_plan", planner, raising=False
        )
    else:
        monkeypatch.delattr(aiter.mla, "get_mla_v4_nm_split_plan", raising=False)
    buf = torch.zeros(65, dtype=torch.int32)
    plan = paged_decode.v4_decode_split_plan(28, 128, 1152, buf)

    if has_planner:
        assert calls == [(28, 128, 1152)]
        assert plan[0] == 2 and plan[1].shape[0] == 29
    else:
        assert plan is None


@pytest.mark.parametrize("kv_fp8", [True, False])
def test_builder_decode_split_plan(monkeypatch, kv_fp8):
    from atom.model_ops.attentions import deepseek_v4_attn

    calls = []
    monkeypatch.setattr(
        deepseek_v4_attn,
        "v4_decode_split_plan",
        lambda rows, heads, kv_len, buf: calls.append((rows, heads, kv_len)) or "plan",
    )
    builder = SimpleNamespace(_kv_fp8=kv_fp8, _local_heads=128)
    plan = deepseek_v4_attn.DeepseekV4AttentionMetadataBuilder._decode_split_plan(
        builder, 28, 1152, torch.zeros(65, dtype=torch.int32)
    )

    if kv_fp8:
        assert plan == "plan" and calls == [(28, 128, 1152)]
    else:
        assert plan is None and calls == []
