# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import logging

import pytest
from atomrl.core.registry import Registry


def test_register_and_get() -> None:
    r = Registry("component")
    r.register("foo", 123)
    assert r.get("foo") == 123
    assert "foo" in r
    assert r.keys() == ["foo"]


def test_unknown_key_raises() -> None:
    r = Registry("worker")
    with pytest.raises(KeyError, match="Unknown worker"):
        r.get("missing")


def test_decorator_register() -> None:
    r = Registry("algorithm")

    @r.decorator("mine")
    class Alg:
        pass

    assert r.get("mine") is Alg


def test_overwrite_warning(caplog: pytest.LogCaptureFixture) -> None:
    r = Registry("test_registry")
    r.register("k", 1)
    with caplog.at_level(logging.WARNING):
        r.register("k", 2)
    assert r.get("k") == 2
    assert any("Overwriting" in rec.message for rec in caplog.records)
