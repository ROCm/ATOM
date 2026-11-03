# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests: QSA index scorer config + CLI switch (default OFF, hash-neutral)."""

from __future__ import annotations

import argparse
import ast
import dataclasses
from pathlib import Path

import pytest

from atom.config import Config, QSAIndexScorerConfig

ROOT = Path(__file__).resolve().parents[1]


def test_default_off():
    cfg = QSAIndexScorerConfig()
    assert cfg.profile == "" and cfg.capture == ""
    assert not cfg.enabled and not cfg.active
    assert cfg.hash_factors() == []


def test_config_field_defaults_off():
    f = {x.name: x for x in dataclasses.fields(Config)}["qsa_index_scorer"]
    assert f.default_factory() == QSAIndexScorerConfig()


def test_enabled_profile_keys_hash_capture_does_not():
    on = QSAIndexScorerConfig(profile="qsa_page4_gfx1250_v1")
    assert on.enabled and on.active
    assert on.hash_factors() == [("qsa_index_scorer", "qsa_page4_gfx1250_v1")]
    cap = QSAIndexScorerConfig(capture="/tmp/cap.json")
    assert cap.active and not cap.enabled and cap.hash_factors() == []


def test_profile_validation():
    assert QSAIndexScorerConfig(profile=" /x/y/p.json ").profile == "/x/y/p.json"
    with pytest.raises(ValueError):
        QSAIndexScorerConfig(profile="bad name;rm")
    with pytest.raises(ValueError, match="Unknown qsa_index_scorer"):
        QSAIndexScorerConfig.from_dict({"profle": "x"})
    assert QSAIndexScorerConfig.from_dict(None) == QSAIndexScorerConfig()


def test_compute_hash_extends_only_hash_factors():
    """compute_hash appends the scorer factors via hash_factors() (empty when OFF)."""
    src = (ROOT / "atom" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_hash"
    )
    body = ast.unparse(fn)
    assert "factors.extend(self.qsa_index_scorer.hash_factors())" in body
    assert body.count("qsa_index_scorer") == 1


def test_cli_flags_round_trip():
    try:
        from atom.model_engine.arg_utils import EngineArgs
    except Exception as exc:  # noqa: BLE001 -- engine deps (zmq, aiter) absent
        pytest.skip(f"arg_utils not importable: {exc!r}")
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    off = EngineArgs.from_cli_args(parser.parse_args([]))
    assert off.qsa_index_scorer_profile == "" and off.qsa_index_scorer_capture == ""
    kw = off._get_engine_kwargs()
    assert kw["qsa_index_scorer"] == QSAIndexScorerConfig()
    assert "qsa_index_scorer_profile" not in kw and "qsa_index_scorer_capture" not in kw
    on = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--qsa-index-scorer-profile",
                "qsa_page4_gfx1250_v1",
                "--qsa-index-scorer-capture",
                "/tmp/c_{rank}.json",
            ]
        )
    )
    kw = on._get_engine_kwargs()
    assert kw["qsa_index_scorer"] == QSAIndexScorerConfig(
        profile="qsa_page4_gfx1250_v1", capture="/tmp/c_{rank}.json"
    )
