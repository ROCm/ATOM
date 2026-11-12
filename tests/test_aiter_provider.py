# SPDX-License-Identifier: MIT
"""The provider rail with more than one provider enabled.

``configure`` stops startup when an enabled provider cannot serve the ops it
declares, and the gemm-fusion hooks reach only the providers of
``sp_qkv_fused``. CPU only: aiter and the providers are stand-ins.
"""

import logging
import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import pytest

import atom.aiter_provider as rail
from atom.model_ops.gemm_fusion import initialize_gemm_fusion

SP_QKV = "sp-qkv-provider"
SPARSE_MLA = "flydsl-sparse-mla-provider"


class _SpQkv:
    op_versions: ClassVar[dict[str, set[int]]] = {"sp_qkv_fused": {1}}

    def __init__(self):
        self.warmed = []
        self.prepared = []
        self.shutdowns = 0

    def warmup(self, payload, *, group=None, device=None):
        self.warmed.append(payload)
        return True

    def prepare(self, payload, *, group=None, device=None):
        self.prepared.append(payload)
        return True

    def shutdown(self, *, group=None, device=None):
        self.shutdowns += 1


class _SparseMla:
    """The real provider's shape: warmup() takes no ``group``, no prepare()."""

    op_versions: ClassVar[dict[str, frozenset[int]]] = {
        "mla_sparse_prefill_v41": frozenset({1})
    }

    def __init__(self):
        self.warmed = []

    def warmup(self, payload=None, *, device=None):
        self.warmed.append(payload)
        return True


def _providers(monkeypatch, **loaded):
    loaded = {SP_QKV: _SpQkv(), SPARSE_MLA: _SparseMla(), **loaded}
    monkeypatch.setattr(rail, "_load_provider", loaded.get)
    return loaded


def _config(*names):
    return SimpleNamespace(aiter_provider_enable=names)


def test_warmup_reaches_only_the_sp_qkv_providers(monkeypatch, caplog):
    loaded = _providers(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=rail.logger.name):
        assert rail.warmup_aiter_providers(
            _config(SP_QKV, SPARSE_MLA), {"mode": "dense"}, group="g"
        )
    assert loaded[SP_QKV].warmed == [{"mode": "dense"}]
    assert loaded[SPARSE_MLA].warmed == []
    assert caplog.records == []


def test_prepare_and_shutdown_reach_only_the_sp_qkv_providers(monkeypatch, caplog):
    loaded = _providers(monkeypatch)
    config = _config(SPARSE_MLA, SP_QKV)
    with caplog.at_level(logging.WARNING, logger=rail.logger.name):
        assert rail.prepare_aiter_providers(config, {"m": 2}, group="g")
        rail.shutdown_aiter_providers(config, group="g")
    assert loaded[SP_QKV].prepared == [{"m": 2}]
    assert loaded[SP_QKV].shutdowns == 1
    assert caplog.records == []


def test_sparse_mla_alone_installs_no_gemm_fusion(monkeypatch, caplog):
    loaded = _providers(monkeypatch)
    config = _config(SPARSE_MLA)
    with caplog.at_level(logging.WARNING):
        assert initialize_gemm_fusion(None, config) is None
        assert not rail.warmup_aiter_providers(config, {"mode": "dense"})
    assert loaded[SPARSE_MLA].warmed == []
    assert caplog.records == []


def _aiter(monkeypatch, routed, *, api=1, skipped=()):
    """An aiter whose ops resolve lazily; ``routed`` maps op -> eligible names."""
    resolved = []
    module = types.ModuleType("aiter")

    def lazy(name):
        if name.startswith("__"):
            raise AttributeError(name)
        resolved.append(name)
        if name in routed:
            return object()
        raise AttributeError(name)

    def status():
        return {
            "enabled": enabled,
            "routed_ops": {
                op: {
                    "eligible": eligible,
                    "ineligible": {
                        name: "serves it at version 1, not 2"
                        for name in enabled
                        if name not in eligible
                    },
                }
                for op, eligible in routed.items()
            },
            "skipped_providers": skipped,
        }

    enabled = ()

    def configure(*, source, enable):
        nonlocal enabled
        enabled = tuple(enable)

    module.__getattr__ = lazy
    module.plugin = SimpleNamespace(API_VERSION=api, configure=configure, status=status)
    monkeypatch.setitem(sys.modules, "aiter", module)
    return resolved


def test_configure_resolves_and_checks_every_enabled_providers_op(monkeypatch):
    _providers(monkeypatch)
    resolved = _aiter(
        monkeypatch,
        {"sp_qkv_fused": (SP_QKV,), "mla_sparse_prefill_v41": (SPARSE_MLA,)},
    )
    assert rail.configure_aiter_providers(_config(SP_QKV, SPARSE_MLA))
    assert resolved == ["sp_qkv_fused", "mla_sparse_prefill_v41"]


def test_configure_fails_when_aiter_does_not_route_a_declared_op(monkeypatch):
    _providers(monkeypatch)
    _aiter(monkeypatch, {"sp_qkv_fused": (SP_QKV,)})
    with pytest.raises(RuntimeError, match="mla_sparse_prefill_v41, which this aiter"):
        rail.configure_aiter_providers(_config(SPARSE_MLA))


def test_configure_fails_when_aiter_would_not_ask_the_provider(monkeypatch):
    _providers(monkeypatch)
    _aiter(
        monkeypatch,
        {"mla_sparse_prefill_v41": ()},
        skipped=({"entry_point": "old-wheel", "reason": "declares API versions [0]"},),
    )
    with pytest.raises(RuntimeError) as failure:
        rail.configure_aiter_providers(_config(SPARSE_MLA))
    assert "cannot serve mla_sparse_prefill_v41: serves it at version 1, not 2" in str(
        failure.value
    )
    assert "aiter skipped old-wheel: declares API versions [0]" in str(failure.value)


def test_configure_fails_for_a_provider_that_cannot_be_loaded(monkeypatch):
    _providers(monkeypatch, **{SPARSE_MLA: None})
    _aiter(monkeypatch, {"mla_sparse_prefill_v41": (SPARSE_MLA,)})
    with pytest.raises(RuntimeError, match="cannot be loaded"):
        rail.configure_aiter_providers(_config(SPARSE_MLA))


def test_configure_reads_op_versions_rather_than_probing_them(monkeypatch):
    _providers(monkeypatch, **{SPARSE_MLA: object()})
    _aiter(monkeypatch, {"mla_sparse_prefill_v41": (SPARSE_MLA,)})
    with pytest.raises(AttributeError, match="op_versions"):
        rail.configure_aiter_providers(_config(SPARSE_MLA))


def test_configure_fails_without_the_plugin_interface(monkeypatch):
    _providers(monkeypatch)
    monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
    with pytest.raises(RuntimeError, match="no aiter.plugin"):
        rail.configure_aiter_providers(_config(SPARSE_MLA))


def test_configure_fails_on_another_plugin_api(monkeypatch):
    _providers(monkeypatch)
    _aiter(monkeypatch, {"mla_sparse_prefill_v41": (SPARSE_MLA,)}, api=2)
    with pytest.raises(RuntimeError, match="plugin API 2"):
        rail.configure_aiter_providers(_config(SPARSE_MLA))
