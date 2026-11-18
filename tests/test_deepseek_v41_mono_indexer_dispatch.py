# SPDX-License-Identifier: MIT
"""Native hook activation, layer ownership and prefill/width fallback."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")
from atom.models.deepseek_v41.mono import indexer as module
from atom.models.deepseek_v41.mono.config import indexer_enabled


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [SimpleNamespace(attn=SimpleNamespace(indexer=SimpleNamespace()))]
        self.calls = []
        self.layer = False
        self.fail = False

    def uses_layer_mono(self, *args):
        return self.layer

    def forward(self, ids, positions, **kwargs):
        self.calls.append(self.layers[0].attn.indexer._mono.runner.active)
        if self.fail:
            raise RuntimeError("forward failed")
        return ids + 1


def setup(monkeypatch):
    model = Model()
    runner = SimpleNamespace(active=False, begin_step=Mock())
    lazy = SimpleNamespace(runner=runner, ready=Mock(return_value=True))
    monkeypatch.setattr(module, "LazyRunner", lambda *args: lazy)
    md = SimpleNamespace(cache=object(), step=object())
    monkeypatch.setattr(
        module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=md)
    )
    monkeypatch.setattr(module, "step_supported", lambda *args: True)
    config = SimpleNamespace(
        max_model_len=8192, index_cache_dtype="fp4", index_cache_layout="mfma"
    )
    return model, module.MonoIndexerModel(model, model, config), lazy, md


def test_hook_is_active_only_for_the_supported_forward(monkeypatch):
    raw, wrapped, lazy, md = setup(monkeypatch)
    ids = torch.arange(8)
    assert torch.equal(wrapped(ids, ids), ids + 1)
    assert raw.calls == [True]
    lazy.runner.begin_step.assert_called_once_with(8, md.cache, md.step)
    assert not lazy.runner.active
    raw.fail = True
    with pytest.raises(RuntimeError, match="forward failed"):
        wrapped(ids, ids)
    assert not lazy.runner.active


def test_layer_runner_owns_its_indexer_without_preparing_native_buffers(monkeypatch):
    raw, wrapped, lazy, _ = setup(monkeypatch)
    raw.layer = True
    ids = torch.arange(16)
    wrapped(ids, ids)
    lazy.ready.assert_not_called()
    lazy.runner.begin_step.assert_not_called()
    assert raw.calls == [False]


def test_unsupported_step_keeps_native_route(monkeypatch):
    raw, wrapped, lazy, _ = setup(monkeypatch)
    monkeypatch.setattr(module, "step_supported", lambda *args: False)
    ids = torch.arange(65)
    wrapped(ids, ids)
    lazy.ready.assert_not_called()
    assert raw.calls == [False]


def test_fp8_and_opt_out_keep_the_embedded_indexer(monkeypatch):
    monkeypatch.delenv("ATOM_DSV41_MONO_INDEXER", raising=False)
    assert indexer_enabled(True)
    assert not indexer_enabled(False)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "0")
    assert not indexer_enabled(True)


@pytest.mark.parametrize("tp,rows", [(2, 40), (2, 48)])
def test_tp2_large_batch_keeps_native_layers_and_mono_indexer(monkeypatch, tp, rows):
    from atom.models.deepseek_v41.mono import dispatch

    raw, _, lazy, md = setup(monkeypatch)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "1")
    monkeypatch.setattr(dispatch, "step_supported", lambda *args: True)
    layer = dispatch.MonoDecodeModel(raw, None, 8192, True, 0, tp)
    layer._mono = SimpleNamespace(ready=Mock(), runner=SimpleNamespace(forward=Mock()))
    cfg = SimpleNamespace(
        max_model_len=8192, index_cache_dtype="fp4", index_cache_layout="mfma"
    )
    wrapped = module.MonoIndexerModel(layer, raw, cfg)
    ids = torch.arange(rows)
    assert torch.equal(wrapped(ids, ids), ids + 1)
    assert raw.calls == [True]
    layer._mono.ready.assert_not_called()
    layer._mono.runner.forward.assert_not_called()
    lazy.runner.begin_step.assert_called_once_with(rows, md.cache, md.step)
    assert not lazy.runner.active


@pytest.mark.parametrize(
    "tp,rows,fp4,enabled",
    [
        (2, 8, True, "1"),
        (4, 40, True, "1"),
        (8, 40, True, "1"),
        (2, 40, False, "1"),
        (2, 40, True, "0"),
    ],
)
def test_other_layer_routes_do_not_prepare_native_indexer(
    monkeypatch, tp, rows, fp4, enabled
):
    from atom.models.deepseek_v41.mono import dispatch

    raw, _, lazy, _ = setup(monkeypatch)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", enabled)
    monkeypatch.setattr(dispatch, "step_supported", lambda *args: True)
    layer = dispatch.MonoDecodeModel(raw, None, 8192, fp4, 0, tp)
    layer_run = Mock(side_effect=lambda ids, positions: ids + 2)
    layer._mono = SimpleNamespace(
        ready=Mock(return_value=True), runner=SimpleNamespace(forward=layer_run)
    )
    cfg = SimpleNamespace(
        max_model_len=8192, index_cache_dtype="fp4", index_cache_layout="mfma"
    )
    wrapped = module.MonoIndexerModel(layer, raw, cfg)
    ids = torch.arange(rows)
    assert torch.equal(wrapped(ids, ids), ids + 2)
    assert raw.calls == []
    layer_run.assert_called_once()
    lazy.ready.assert_not_called()
    lazy.runner.begin_step.assert_not_called()
