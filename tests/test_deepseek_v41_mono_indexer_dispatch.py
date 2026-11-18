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
