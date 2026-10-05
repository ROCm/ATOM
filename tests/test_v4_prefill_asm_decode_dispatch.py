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
    return {
        "unified_kv": torch.empty((4, 512)),
        "kv_indices": torch.empty(0, dtype=torch.int32),
        "kv_indptr": torch.zeros(n + 1, dtype=torch.int32),
        "attn_sink": torch.empty(heads),
        "softmax_scale": 512**-0.5,
        "unified_kv_rope": torch.empty((4, 64)),
        "q_packed_in": torch.empty((n, heads, 512)),
        "q_rope_in": torch.empty((n, heads, 64)),
    }


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


def test_uniform_split_table_rows():
    table = paged_decode.v4_uniform_split_table(4, "cpu")
    assert table.shape == (paged_decode.V4_DECODE_MAX_SPLITS, 5)
    assert table.dtype == torch.int32
    assert table[0].tolist() == [0, 1, 2, 3, 4]
    assert table[2].tolist() == [0, 3, 6, 9, 12]


@pytest.mark.parametrize("has_count", [True, False])
def test_v4_decode_split_plan_follows_aiter(monkeypatch, has_count):
    import aiter.mla

    calls = []

    def count(rows, heads, kv_len):
        calls.append((rows, heads, kv_len))
        return 3

    if has_count:
        monkeypatch.setattr(
            aiter.mla, "get_mla_v4_nm_num_kv_splits", count, raising=False
        )
    else:
        monkeypatch.delattr(aiter.mla, "get_mla_v4_nm_num_kv_splits", raising=False)
    table = paged_decode.v4_uniform_split_table(64, "cpu")
    plan = paged_decode.v4_decode_split_plan(28, 128, 1152, table)

    if has_count:
        # A row of the constant table, not a copy: nothing is written per call.
        assert calls == [(28, 128, 1152)]
        assert plan[0] == 3 and plan[1].data_ptr() == table[2].data_ptr()
        assert plan[1][:29].tolist() == list(range(0, 3 * 29, 3))
    else:
        assert plan is None


@pytest.mark.parametrize("splits,rows", [(17, 28), (3, 64)])
def test_v4_decode_split_plan_outside_table_leaves_pick_to_aiter(
    monkeypatch, splits, rows
):
    import aiter.mla

    monkeypatch.setattr(
        aiter.mla,
        "get_mla_v4_nm_num_kv_splits",
        lambda *a: splits,
        raising=False,
    )
    table = paged_decode.v4_uniform_split_table(63, "cpu")
    assert paged_decode.v4_decode_split_plan(rows, 128, 1152, table) is None


@pytest.mark.parametrize("kv_fp8", [True, False])
def test_builder_decode_split_plan(monkeypatch, kv_fp8):
    from atom.model_ops.attentions import deepseek_v4_attn

    calls = []
    monkeypatch.setattr(
        deepseek_v4_attn,
        "v4_decode_split_plan",
        lambda rows, heads, kv_len, table: calls.append((rows, heads, kv_len, table))
        or "plan",
    )
    table = paged_decode.v4_uniform_split_table(64, "cpu")
    builder = SimpleNamespace(_kv_fp8=kv_fp8, _local_heads=128, _split_table=table)
    plan = deepseek_v4_attn.DeepseekV4AttentionMetadataBuilder._decode_split_plan(
        builder, 28, 1152
    )

    if kv_fp8:
        assert plan == "plan" and calls == [(28, 128, 1152, table)]
    else:
        assert plan is None and calls == []
