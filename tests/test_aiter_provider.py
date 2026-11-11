# SPDX-License-Identifier: MIT
"""The provider rail with more than one provider enabled.

``warmup``/``prepare`` hand one op's call to the providers that serve that op,
and ``configure`` resolves and checks the routed ops of the providers it
enables. CPU only: aiter and the providers are stand-ins.
"""

import logging
import sys
import types
from types import SimpleNamespace

import atom.aiter_provider as rail

SP_QKV = "sp-qkv-provider"
SPARSE_MLA = "flydsl-sparse-mla-provider"


class _SpQkv:
    op_versions = {"sp_qkv_fused": {1}}

    def __init__(self):
        self.warmed = []
        self.prepared = []

    def warmup(self, payload, *, group=None, device=None):
        self.warmed.append(payload)
        return True

    def prepare(self, payload, *, group=None, device=None):
        self.prepared.append(payload)
        return True


class _SparseMla:
    """The real provider's shape: warmup() takes no ``group``, no prepare()."""

    op_versions = {"mla_sparse_prefill_v41": frozenset({1})}

    def __init__(self):
        self.warmed = []

    def warmup(self, payload=None, *, device=None):
        self.warmed.append(payload)
        return True


def _providers(monkeypatch):
    loaded = {SP_QKV: _SpQkv(), SPARSE_MLA: _SparseMla()}
    monkeypatch.setattr(rail, "_load_provider", loaded.get)
    return loaded


def _config(*names):
    return SimpleNamespace(aiter_provider_enable=names)


def test_warmup_reaches_only_the_providers_of_its_op(monkeypatch, caplog):
    loaded = _providers(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=rail.logger.name):
        assert rail.warmup_aiter_providers(
            _config(SP_QKV, SPARSE_MLA), {"mode": "dense"}, group="g"
        )
    assert loaded[SP_QKV].warmed == [{"mode": "dense"}]
    assert loaded[SPARSE_MLA].warmed == []
    assert caplog.records == []


def test_prepare_reaches_only_the_providers_of_its_op(monkeypatch, caplog):
    loaded = _providers(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=rail.logger.name):
        assert rail.prepare_aiter_providers(
            _config(SPARSE_MLA, SP_QKV), {"m": 2}, group="g"
        )
    assert loaded[SP_QKV].prepared == [{"m": 2}]
    assert caplog.records == []


def test_an_op_no_enabled_provider_serves_warms_nothing(monkeypatch):
    loaded = _providers(monkeypatch)
    assert not rail.warmup_aiter_providers(_config(SPARSE_MLA), {"mode": "dense"})
    assert loaded[SPARSE_MLA].warmed == []


def _aiter(monkeypatch, routed):
    resolved = []
    module = types.ModuleType("aiter")

    def lazy(name):
        if not name.startswith("__"):
            resolved.append(name)
        if name in routed:
            return object()
        raise AttributeError(name)

    module.__getattr__ = lazy
    module.plugin = SimpleNamespace(
        API_VERSION=1,
        configure=lambda **kwargs: None,
        status=lambda: {"routed_ops": {op: {} for op in routed}},
    )
    monkeypatch.setitem(sys.modules, "aiter", module)
    return resolved


def test_configure_checks_only_the_ops_of_the_enabled_providers(monkeypatch, caplog):
    resolved = _aiter(monkeypatch, routed=("sp_qkv_fused",))
    with caplog.at_level(logging.WARNING, logger=rail.logger.name):
        assert rail.configure_aiter_providers(_config(SPARSE_MLA))
    assert resolved == ["mla_sparse_prefill_v41"]
    assert "mla_sparse_prefill_v41" in caplog.text
    assert "sp_qkv_fused" not in caplog.text


def test_configure_resolves_every_enabled_providers_op(monkeypatch, caplog):
    resolved = _aiter(monkeypatch, routed=("sp_qkv_fused", "mla_sparse_prefill_v41"))
    with caplog.at_level(logging.WARNING, logger=rail.logger.name):
        assert rail.configure_aiter_providers(_config(SP_QKV, SPARSE_MLA))
    assert resolved == ["sp_qkv_fused", "mla_sparse_prefill_v41"]
    assert "missing routed ops" not in caplog.text
