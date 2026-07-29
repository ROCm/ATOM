"""Unit tests for MTP draft index_share_for_mtp_iteration helpers.

Avoid importing ``atom.models.deepseek_mtp`` here: that module pulls in
``atom.config`` (torch / aiter / transformers) and is brittle in lightweight
test collection.  The helpers below mirror
``DeepSeekMultiTokenPredictor.set_skip_topk`` /
``compact_topk_indices`` — keep them in sync when editing production code.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch


class _FakeMlaAttn:
    def __init__(self, buf: torch.Tensor):
        self.sparse_kv_indices_buffer = buf


class _FakeSelfAttn:
    def __init__(self, *, has_indexer: bool, buf: torch.Tensor):
        self.skip_topk = False
        self.indexer = object() if has_indexer else None
        self.mla_attn = _FakeMlaAttn(buf)


class _FakeMtpBlock:
    def __init__(self, self_attn: _FakeSelfAttn):
        self.self_attn = self_attn


class _FakeLayer:
    def __init__(self, self_attn: _FakeSelfAttn):
        self.mtp_block = _FakeMtpBlock(self_attn)


def _set_skip_topk(layers: dict[str, Any], skip: bool) -> None:
    """Mirror of ``DeepSeekMultiTokenPredictor.set_skip_topk``."""
    for layer in layers.values():
        mtp_block = getattr(layer, "mtp_block", None)
        if mtp_block is None:
            continue
        self_attn = getattr(mtp_block, "self_attn", None)
        if self_attn is None or not hasattr(self_attn, "skip_topk"):
            continue
        if getattr(self_attn, "indexer", None) is not None:
            self_attn.skip_topk = skip


def _compact_topk_indices(layers: dict[str, Any], slot_ids: torch.Tensor) -> None:
    """Mirror of ``DeepSeekMultiTokenPredictor.compact_topk_indices``."""
    num_slots = slot_ids.numel()
    for layer in layers.values():
        mtp_block = getattr(layer, "mtp_block", None)
        if mtp_block is None:
            continue
        self_attn = getattr(mtp_block, "self_attn", None)
        if self_attn is None:
            continue
        mla_attn = getattr(self_attn, "mla_attn", None)
        if mla_attn is None:
            continue
        sparse_buf = getattr(mla_attn, "sparse_kv_indices_buffer", None)
        if sparse_buf is not None and sparse_buf.numel() > 0:
            sparse_buf[:num_slots] = sparse_buf[slot_ids]


def test_set_skip_topk_only_layers_with_indexer():
    buf0 = torch.zeros(4, 8, dtype=torch.int32)
    buf1 = torch.zeros(4, 8, dtype=torch.int32)
    layers = {
        "80": _FakeLayer(_FakeSelfAttn(has_indexer=True, buf=buf0)),
        "81": _FakeLayer(_FakeSelfAttn(has_indexer=False, buf=buf1)),
    }

    _set_skip_topk(layers, True)

    assert layers["80"].mtp_block.self_attn.skip_topk is True
    assert layers["81"].mtp_block.self_attn.skip_topk is False


def test_compact_topk_indices_gathers_rows_to_front():
    buf = torch.arange(20, dtype=torch.int32).reshape(10, 2)
    layers = {"80": _FakeLayer(_FakeSelfAttn(has_indexer=True, buf=buf))}

    slot_ids = torch.tensor([3, 7], dtype=torch.int64)
    _compact_topk_indices(layers, slot_ids)

    expected_row0 = torch.arange(6, 8, dtype=torch.int32)
    expected_row1 = torch.arange(14, 16, dtype=torch.int32)
    assert torch.equal(buf[0], expected_row0)
    assert torch.equal(buf[1], expected_row1)


def test_compact_topk_indices_skips_empty_buffer():
    empty = torch.empty(0, dtype=torch.int32)
    layers = {"80": _FakeLayer(_FakeSelfAttn(has_indexer=True, buf=empty))}
    _compact_topk_indices(layers, torch.tensor([0], dtype=torch.int64))


@pytest.mark.parametrize(
    "method,index_share,index_topk,has_api,expected",
    [
        ("mtp", True, 2048, True, True),
        ("mtp", False, 2048, True, False),
        ("eagle3", True, 2048, True, False),
        ("mtp", True, None, True, False),
        ("mtp", True, 2048, False, False),
    ],
)
def test_share_mtp_indices_gate(method, index_share, index_topk, has_api, expected):
    draft_hf = SimpleNamespace(
        index_share_for_mtp_iteration=index_share,
    )
    if index_topk is not None:
        draft_hf.index_topk = index_topk
    mtp_inner = (
        SimpleNamespace(set_skip_topk=lambda _: None) if has_api else SimpleNamespace()
    )

    share = (
        method == "mtp"
        and getattr(draft_hf, "index_share_for_mtp_iteration", False)
        and hasattr(draft_hf, "index_topk")
        and hasattr(mtp_inner, "set_skip_topk")
    )
    assert share is expected
