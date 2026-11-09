# SPDX-License-Identifier: MIT
"""CPU tests for the opt-in aiter Provider enablement rail.

No aiter and no GPU: ``aiter.plugin`` is stubbed through ``sys.modules``. The
tests pin the contract ATOM relies on -- default off never calls ``configure()``,
one call with a provenance ``source`` when enabled, fail-open when the interface
is absent/incompatible/broken, routed-op presence warnings, and the Config
normalization/hash factors.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from atom.aiter_provider import (
    _SUPPORTED_PLUGIN_API,
    atom_version,
    configure_aiter_providers,
    normalize_provider_names,
)
from atom.config import Config


def _install_aiter(monkeypatch, *, plugin=None, **attrs):
    module = types.ModuleType("aiter")
    for name, value in attrs.items():
        setattr(module, name, value)
    if plugin is not None:
        module.plugin = plugin
    monkeypatch.setitem(sys.modules, "aiter", module)
    return module


class _Plugin:
    def __init__(self, *, api=_SUPPORTED_PLUGIN_API, status=None, fail=None):
        self.API_VERSION = api
        self.calls = []
        self._status = status
        self._fail = fail

    def configure(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail is not None:
            raise self._fail

    def status(self):
        return self._status or {
            "enabled": (),
            "api_versions_accepted": (self.API_VERSION,),
            "routed_ops": {},
            "skipped_providers": (),
        }


def test_normalize_provider_names():
    assert normalize_provider_names(None) == ()
    assert normalize_provider_names("") == ()
    assert normalize_provider_names("  ") == ()
    assert normalize_provider_names("b, a ,,b") == ("b", "a")
    assert normalize_provider_names(["b", "a", "b"]) == ("b", "a")
    with pytest.raises(TypeError, match="must be strings"):
        normalize_provider_names([1])
    assert normalize_provider_names([None]) == ()


def test_atom_version_is_a_string():
    assert isinstance(atom_version(), str) and atom_version()


def test_empty_enable_list_never_calls_configure(monkeypatch):
    plugin = _Plugin()
    _install_aiter(monkeypatch, plugin=plugin)
    assert configure_aiter_providers(SimpleNamespace(aiter_provider_enable=())) is False
    assert configure_aiter_providers(SimpleNamespace(aiter_provider_enable="")) is False
    assert plugin.calls == []


def test_enable_calls_configure_with_source_and_status(monkeypatch, caplog):
    plugin = _Plugin(
        status={
            "enabled": ("aklr",),
            "api_versions_accepted": (_SUPPORTED_PLUGIN_API,),
            "routed_ops": {"sp_qkv_fused": {"version": 1}},
            "skipped_providers": (),
        }
    )
    _install_aiter(monkeypatch, plugin=plugin)
    config = SimpleNamespace(aiter_provider_enable=("aklr", "other"))

    with caplog.at_level("INFO", logger="atom.aiter_provider"):
        assert configure_aiter_providers(config) is True
    assert plugin.calls == [
        {"source": f"atom {atom_version()}", "enable": ("aklr", "other")}
    ]
    assert "aiter providers: enabled=('aklr',)" in caplog.text


def test_missing_plugin_fails_open(monkeypatch, caplog):
    _install_aiter(monkeypatch)  # aiter without a plugin attribute
    with caplog.at_level("WARNING", logger="atom.aiter_provider"):
        assert (
            configure_aiter_providers(SimpleNamespace(aiter_provider_enable=("aklr",)))
            is False
        )
    assert "has no aiter.plugin" in caplog.text


def test_api_version_mismatch_fails_open(monkeypatch, caplog):
    plugin = _Plugin(api=_SUPPORTED_PLUGIN_API + 1)
    _install_aiter(monkeypatch, plugin=plugin)
    with caplog.at_level("WARNING", logger="atom.aiter_provider"):
        assert (
            configure_aiter_providers(SimpleNamespace(aiter_provider_enable=("aklr",)))
            is False
        )
    assert plugin.calls == []
    assert "plugin API" in caplog.text


def test_broken_configure_fails_open_and_never_raises(monkeypatch, caplog):
    plugin = _Plugin(fail=RuntimeError("provider exploded"))
    _install_aiter(monkeypatch, plugin=plugin)
    with caplog.at_level("WARNING", logger="atom.aiter_provider"):
        assert (
            configure_aiter_providers(SimpleNamespace(aiter_provider_enable=("aklr",)))
            is False
        )
    assert "native path" in caplog.text


def test_missing_routed_op_warns_but_enables(monkeypatch, caplog):
    plugin = _Plugin()
    _install_aiter(monkeypatch, plugin=plugin)
    with caplog.at_level("WARNING", logger="atom.aiter_provider"):
        assert (
            configure_aiter_providers(SimpleNamespace(aiter_provider_enable=("aklr",)))
            is True
        )
    assert "missing routed ops" in caplog.text
    assert plugin.calls


def _hash(**overrides):
    fields = {
        "quant_config": None,
        "compilation_config": None,
        "parallel_config": None,
        "tensor_parallel_size": 4,
        "prefill_context_parallel_size": 1,
        "dcp_config": None,
        "enable_dp_attention": False,
        "index_cache_dtype": None,
        "hf_config": SimpleNamespace(),
        "max_model_len": 4096,
        "max_num_seqs": 256,
        "speculative_config": None,
    }
    fields.update(overrides)
    return Config.compute_hash(SimpleNamespace(**fields))


def test_compute_hash_keys_the_provider_knobs():
    baseline = _hash()
    assert _hash(aiter_provider_enable=()) == baseline
    assert _hash(enable_sp_qkv_fused=False) == baseline
    assert _hash(aiter_provider_enable=("aklr",)) != baseline
    assert _hash(aiter_provider_enable=("aklr", "other")) != _hash(
        aiter_provider_enable=("aklr",)
    )
    assert _hash(enable_sp_qkv_fused=True) != baseline


def test_config_fields_exist_with_off_defaults():
    from dataclasses import fields

    defaults = {f.name: f.default for f in fields(Config)}
    assert defaults["aiter_provider_enable"] == ()
    assert defaults["enable_sp_qkv_fused"] is False
