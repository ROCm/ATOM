"""Unit tests for MTP draft ``index_share_for_mtp_iteration`` helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from atom.spec_decode.mtp_index_share import (
    can_reuse_mtp_indices,
    compact_mtp_sparse_indices,
    forward_with_fresh_mtp_indices,
    forward_with_reused_mtp_indices,
    set_mtp_index_reuse,
    supports_mtp_index_share,
)


class _FakeMlaAttn:
    def __init__(self, buf: torch.Tensor, topk: int):
        self.sparse_kv_indices_buffer = buf
        self.topk_tokens = topk


class _FakeMlaFrontend:
    """Match native ATOM, where the sparse buffer lives on ``mla_attn.impl``."""

    def __init__(self, impl: _FakeMlaAttn):
        self.impl = impl


class _FakeSelfAttn:
    def __init__(self, *, has_indexer: bool, buf: torch.Tensor, topk: int = 2):
        self.skip_topk = False
        self.indexer = SimpleNamespace(topk_tokens=topk) if has_indexer else None
        self.mla_attn = _FakeMlaFrontend(_FakeMlaAttn(buf, topk))


class _FakeMtpBlock:
    def __init__(self, self_attn: _FakeSelfAttn):
        self.self_attn = self_attn


class _FakeLayer:
    def __init__(self, self_attn: _FakeSelfAttn):
        self.mtp_block = _FakeMtpBlock(self_attn)


class _FakePredictor:
    def __init__(self, layers):
        self.layers = layers


class _FakeCompiledModel:
    """``__call__`` represents the compiled dispatcher; ``forward`` is eager."""

    def __init__(self, predictor: _FakePredictor, *, fail: bool = False):
        self.model = predictor
        self.fail = fail
        self.compiled_calls = 0
        self.compiled_skip_values = []
        self.eager_skip_values = []

    def __call__(self, **kwargs):
        del kwargs
        self.compiled_calls += 1
        self.compiled_skip_values.append(
            [
                layer.mtp_block.self_attn.skip_topk
                for layer in self.model.layers.values()
            ]
        )
        if self.fail:
            raise RuntimeError("forward failed")
        return "compiled-output"

    def forward(self, **kwargs):
        del kwargs
        self.eager_skip_values.append(
            [
                layer.mtp_block.self_attn.skip_topk
                for layer in self.model.layers.values()
            ]
        )
        if self.fail:
            raise RuntimeError("forward failed")
        return "eager-output"


def test_set_skip_topk_only_layers_with_indexer():
    buf0 = torch.zeros(4, 8, dtype=torch.int32)
    buf1 = torch.zeros(4, 8, dtype=torch.int32)
    indexed = _FakeSelfAttn(has_indexer=True, buf=buf0, topk=8)
    unindexed = _FakeSelfAttn(has_indexer=False, buf=buf1, topk=8)
    predictor = _FakePredictor(
        {"80": _FakeLayer(indexed), "81": _FakeLayer(unindexed)}
    )

    set_mtp_index_reuse(predictor, True)

    assert indexed.skip_topk is True
    assert unindexed.skip_topk is False


def test_compact_flat_topk_buffer_gathers_whole_rows_to_front():
    buf = torch.arange(20, dtype=torch.int32)
    sparse_indptr = torch.tensor(
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 20], dtype=torch.int32
    )
    predictor = _FakePredictor(
        {"80": _FakeLayer(_FakeSelfAttn(has_indexer=True, buf=buf, topk=2))}
    )
    model = _FakeCompiledModel(predictor)

    slot_ids = torch.tensor([3, 7], dtype=torch.int64)
    compact_mtp_sparse_indices(model, slot_ids, sparse_indptr, running_rows=2)

    assert torch.equal(buf[:4], torch.tensor([5, 6, 13, 14], dtype=torch.int32))


def test_compact_shared_buffer_only_once():
    buf = torch.arange(12, dtype=torch.int32)
    predictor = _FakePredictor(
        {
            "80": _FakeLayer(_FakeSelfAttn(has_indexer=True, buf=buf, topk=2)),
            "81": _FakeLayer(_FakeSelfAttn(has_indexer=True, buf=buf, topk=2)),
        }
    )

    compact_mtp_sparse_indices(
        _FakeCompiledModel(predictor),
        torch.tensor([2, 0], dtype=torch.int64),
        torch.arange(0, 14, 2, dtype=torch.int32),
        running_rows=2,
    )

    assert torch.equal(buf[:4], torch.tensor([4, 5, 0, 1], dtype=torch.int32))


def test_compact_repeats_last_real_row_for_graph_padding():
    buf = torch.arange(12, dtype=torch.int32)
    predictor = _FakePredictor(
        {"80": _FakeLayer(_FakeSelfAttn(has_indexer=True, buf=buf, topk=2))}
    )

    compact_mtp_sparse_indices(
        _FakeCompiledModel(predictor),
        torch.tensor([2, 0], dtype=torch.int64),
        torch.arange(0, 14, 2, dtype=torch.int32),
        running_rows=4,
    )

    assert torch.equal(
        buf[:8],
        torch.tensor([4, 5, 0, 1, 0, 1, 0, 1], dtype=torch.int32),
    )


def test_fresh_index_forward_is_compiled_and_restores_reuse():
    attn = _FakeSelfAttn(
        has_indexer=True, buf=torch.arange(8, dtype=torch.int32), topk=2
    )
    predictor = _FakePredictor({"80": _FakeLayer(attn)})
    model = _FakeCompiledModel(predictor)
    set_mtp_index_reuse(predictor, True)

    result = forward_with_fresh_mtp_indices(model, input_ids=torch.tensor([1]))

    assert result == "compiled-output"
    assert model.compiled_calls == 1
    assert model.compiled_skip_values == [[False]]
    assert model.eager_skip_values == []
    assert attn.skip_topk is True


def test_fresh_index_forward_restores_reuse_after_failure():
    attn = _FakeSelfAttn(
        has_indexer=True, buf=torch.arange(8, dtype=torch.int32), topk=2
    )
    predictor = _FakePredictor({"80": _FakeLayer(attn)})
    model = _FakeCompiledModel(predictor, fail=True)
    set_mtp_index_reuse(predictor, True)

    with pytest.raises(RuntimeError, match="forward failed"):
        forward_with_fresh_mtp_indices(model)

    assert attn.skip_topk is True


def test_reused_index_forward_is_eager_with_skip_enabled():
    attn = _FakeSelfAttn(
        has_indexer=True, buf=torch.arange(8, dtype=torch.int32), topk=2
    )
    predictor = _FakePredictor({"80": _FakeLayer(attn)})
    model = _FakeCompiledModel(predictor)

    result = forward_with_reused_mtp_indices(model, input_ids=torch.tensor([1]))

    assert result == "eager-output"
    assert model.compiled_calls == 0
    assert model.eager_skip_values == [[True]]


@pytest.mark.parametrize(
    "context_lens,expected",
    [
        ([2048, 4096], True),
        ([2047, 4096], False),
        ([4096], False),
    ],
)
def test_reuse_requires_every_scheduled_row_to_reach_topk(context_lens, expected):
    assert can_reuse_mtp_indices(context_lens, num_rows=2, topk=2048) is expected


def test_support_gate_requires_index_owning_sparse_attention():
    supported = _FakePredictor(
        {
            "80": _FakeLayer(
                _FakeSelfAttn(
                    has_indexer=True,
                    buf=torch.empty(0, dtype=torch.int32),
                    topk=2,
                )
            )
        }
    )
    unsupported = _FakePredictor(
        {
            "80": _FakeLayer(
                _FakeSelfAttn(
                    has_indexer=False,
                    buf=torch.empty(0, dtype=torch.int32),
                    topk=2,
                )
            )
        }
    )

    assert supports_mtp_index_share(supported)
    assert not supports_mtp_index_share(unsupported)
