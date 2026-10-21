# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mixed batching caps the caching allocator at the memory budget; nothing else does.

Mixed steps run eagerly with shapes that change every step, and the uncapped
allocator's cache grew past the budget into the headroom HSA queues and RCCL
allocate from (see `ModelRunner._cap_allocator_at_budget`).
"""

from types import SimpleNamespace

import pytest
import torch

# `model_runner` imports aiter at module scope; same convention as
# test_mixed_spec_decode.py.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_engine.model_runner import ModelRunner

_GB = 1 << 30


def _cap(monkeypatch, *, mixed):
    calls = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda f: calls.append(f)
    )
    runner = SimpleNamespace(config=SimpleNamespace(enable_mixed_prefill_decode=mixed))
    ModelRunner._cap_allocator_at_budget(runner, int(0.9 * 288 * _GB), 288 * _GB)
    return calls


def test_mixed_caps_the_allocator_at_the_budget(monkeypatch):
    calls = _cap(monkeypatch, mixed=True)
    assert calls == [pytest.approx(0.9, abs=1e-9)]


def test_without_mixed_the_allocator_is_left_alone(monkeypatch):
    assert _cap(monkeypatch, mixed=False) == []


def test_the_cap_is_set_where_the_budget_is_computed():
    """Before the KV cache and the graphs are allocated, which it must bound."""
    import inspect

    src = inspect.getsource(ModelRunner.get_num_blocks)
    budget = src.index("budget = int(total * config.gpu_memory_utilization)")
    cap = src.index("self._cap_allocator_at_budget(budget, total)")
    assert budget < cap
