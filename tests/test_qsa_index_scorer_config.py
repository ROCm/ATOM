# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests: aiter provider enablement + QSA scorer observer config, CLI and env
(default OFF, hash-neutral)."""

from __future__ import annotations

import argparse
import dataclasses
from types import SimpleNamespace

import pytest

from atom.aiter_provider import normalize_provider_names
from atom.config import Config, QSAIndexScorerConfig

PROVIDER = "qsa-index-scorer-provider"


def test_default_off():
    cfg = QSAIndexScorerConfig()
    assert cfg.capture == "" and not cfg.active
    assert QSAIndexScorerConfig(capture=" /tmp/c.json ").capture == "/tmp/c.json"
    with pytest.raises(ValueError, match="Unknown qsa_index_scorer"):
        QSAIndexScorerConfig.from_dict({"profile": "x"})  # the old knob is gone
    assert QSAIndexScorerConfig.from_dict(None) == QSAIndexScorerConfig()


def test_config_fields_default_off():
    f = {x.name: x for x in dataclasses.fields(Config)}
    assert f["qsa_index_scorer"].default_factory() == QSAIndexScorerConfig()
    assert f["aiter_provider_enable"].default == ()


def test_normalize_provider_names():
    assert normalize_provider_names(None) == ()
    assert normalize_provider_names("") == ()
    assert normalize_provider_names(" a, b ,a,,") == ("a", "b")
    assert normalize_provider_names(["b", "a", "b"]) == ("b", "a")
    with pytest.raises(TypeError):
        normalize_provider_names([1])


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
        "index_cache_layout": None,
        "sequence_parallel_size": 1,
        "hf_config": SimpleNamespace(),
        "max_model_len": 4096,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 16384,
        "kv_cache_dtype": "bf16",
        "speculative_config": None,
    }
    fields.update(overrides)
    return Config.compute_hash(SimpleNamespace(**fields))


def test_hash_unchanged_when_off_keyed_when_on():
    base = _hash()  # a config without the fields, i.e. the pre-feature hash
    assert _hash(aiter_provider_enable=()) == base
    assert _hash(qsa_index_scorer=QSAIndexScorerConfig(capture="/tmp/c.json")) == base
    on = _hash(aiter_provider_enable=(PROVIDER,))
    assert on != base
    other = _hash(aiter_provider_enable=("other", PROVIDER))
    assert other not in (on, base)


def _engine_args():
    try:
        from atom.model_engine.arg_utils import EngineArgs
    except Exception as exc:  # noqa: BLE001 -- engine deps (zmq, aiter) absent
        pytest.skip(f"arg_utils not importable: {exc!r}")
    return EngineArgs


def test_cli_flags_round_trip(monkeypatch):
    EngineArgs = _engine_args()
    monkeypatch.delenv("ATOM_AITER_PROVIDER_ENABLE", raising=False)
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    off = EngineArgs.from_cli_args(parser.parse_args([]))
    assert off.aiter_provider_enable == "" and off.qsa_index_scorer_capture == ""
    kw = off._get_engine_kwargs()
    assert kw["qsa_index_scorer"] == QSAIndexScorerConfig()
    assert kw["aiter_provider_enable"] == ()
    assert "qsa_index_scorer_capture" not in kw
    on = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--aiter-provider-enable",
                f"{PROVIDER},other",
                "--qsa-index-scorer-capture",
                "/tmp/c_{rank}.json",
            ]
        )
    )
    kw = on._get_engine_kwargs()
    assert kw["aiter_provider_enable"] == (PROVIDER, "other")
    assert kw["qsa_index_scorer"] == QSAIndexScorerConfig(capture="/tmp/c_{rank}.json")


def test_env_override_only_when_flag_absent(monkeypatch):
    EngineArgs = _engine_args()
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    monkeypatch.delenv("ATOM_AITER_PROVIDER_ENABLE", raising=False)
    kw = EngineArgs.from_cli_args(parser.parse_args([]))._get_engine_kwargs()
    assert kw["aiter_provider_enable"] == ()
    monkeypatch.setenv("ATOM_AITER_PROVIDER_ENABLE", PROVIDER)
    kw = EngineArgs.from_cli_args(parser.parse_args([]))._get_engine_kwargs()
    assert kw["aiter_provider_enable"] == (PROVIDER,)
    flag = parser.parse_args(["--aiter-provider-enable", "other"])
    kw = EngineArgs.from_cli_args(flag)._get_engine_kwargs()
    assert kw["aiter_provider_enable"] == ("other",)  # the flag wins
