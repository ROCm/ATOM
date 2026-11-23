# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import resource

import pytest

from atom.utils import raise_memlock_limit


def _limits(monkeypatch, soft, hard):
    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (soft, hard))
    monkeypatch.setattr(
        resource, "setrlimit", lambda kind, lim: calls.append((kind, lim))
    )
    return calls


@pytest.mark.parametrize("hard", [resource.RLIM_INFINITY, 64 << 30])
def test_the_soft_limit_rises_to_the_hard_limit(monkeypatch, hard):
    calls = _limits(monkeypatch, 8 << 20, hard)

    raise_memlock_limit()

    assert calls == [(resource.RLIMIT_MEMLOCK, (hard, hard))]


def test_a_soft_limit_already_at_the_hard_limit_is_left_alone(monkeypatch):
    calls = _limits(monkeypatch, resource.RLIM_INFINITY, resource.RLIM_INFINITY)

    raise_memlock_limit()

    assert calls == []


def test_a_refused_raise_only_warns(monkeypatch, caplog):
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (8 << 20, 64 << 30))

    def refuse(kind, lim):
        raise OSError("refused")

    monkeypatch.setattr(resource, "setrlimit", refuse)

    raise_memlock_limit()

    assert "Failed to raise RLIMIT_MEMLOCK" in caplog.text
