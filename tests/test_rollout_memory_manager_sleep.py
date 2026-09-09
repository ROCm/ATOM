# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sleep and wake: what a release frees, and what it invalidates.

A decode graph replays the addresses it captured -- every weight, and the base
of the KV pool. Whichever of the two a release frees, the graphs that captured
it have to be dropped and recaptured on wake, including on the KV-only release
that `AsyncLLMEngine.sleep(level=1)`, the default level, performs.

That is about addresses, which a test on `updated`/`released` counters cannot
see, so these drive the real `release_memory` / `resume_memory` flow.
"""

from types import SimpleNamespace

import pytest
import torch
from conftest import atom_config_double
from torch import nn

from atom.rollout import memory_manager
from atom.rollout.memory_manager import MemoryManagerMixin


class _Runner(MemoryManagerMixin):
    """The surface `MemoryManagerMixin` documents, and nothing else."""

    def __init__(self, *, enforce_eager, with_graphs=True):
        self.device = torch.device("cpu")
        self.label = "test"
        self.enforce_eager = enforce_eager
        self.config = atom_config_double(
            num_kvcache_blocks=7,
            enforce_eager=enforce_eager,
        )
        self.model = nn.Linear(4, 4, bias=False)
        self.kv_cache = torch.zeros(8)
        self.graphs = {1: object(), 2: object()} if with_graphs else {}
        self.graph_pool = object()
        self.tokenID_processor = SimpleNamespace(clean=lambda: None)
        self.allocated_blocks = []
        self.captures = 0

    def _get_models_with_kv(self):
        return [self.model]

    def get_num_blocks(self):
        return {"num_kvcache_blocks": 7}

    def allocate_kv_cache(self, num_blocks):
        self.allocated_blocks.append(num_blocks)
        self.kv_cache = torch.zeros(8)

    def capture_cudagraph(self):
        self.captures += 1


@pytest.fixture(autouse=True)
def _no_gpu_calls(monkeypatch):
    """The mixin is written against a live device; the policy it implements is not."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(memory_manager, "set_kv_cache_data", lambda _value: None)


def _report_weights_on_device(runner):
    """`_recapture_cudagraphs_if_needed` gates on `param.is_cuda`.

    A CPU runner defers instead of recapturing, which is the right answer for
    a half-woken engine but not the case under test here.
    """
    runner.model = SimpleNamespace(parameters=lambda: [SimpleNamespace(is_cuda=True)])


def test_releasing_the_weights_invalidates_the_graphs():
    runner = _Runner(enforce_eager=False)

    runner.release_memory()

    assert runner.model.weight.numel() == 0
    assert runner.kv_cache is None
    assert runner._kv_cache_num_blocks == 7
    assert runner.graphs == {}
    assert runner._graphs_backup_keys == [1, 2]


def test_releasing_only_the_kv_pool_still_invalidates_the_graphs():
    """`AsyncLLMEngine.sleep(level=1)`, the default, frees the pool and nothing else.

    The graphs captured the base of that pool, so they cannot be replayed
    against the one `_resume_kv_cache` allocates in its place. Nothing else
    would drop them either: the weights never moved, so the release path that
    used to own the graphs is not the one that runs.
    """
    runner = _Runner(enforce_eager=False)

    runner.release_memory(tags=["kv_cache"])

    assert runner.graphs == {}
    assert runner._graphs_backup_keys == [1, 2]

    # The weights never left the device on a level-1 sleep.
    _report_weights_on_device(runner)
    runner.resume_memory(tags=["kv_cache"])

    assert runner.allocated_blocks == [7]
    assert runner.captures == 1
    assert not hasattr(runner, "_graphs_backup_keys")


def test_wake_recaptures_the_graphs_it_released():
    runner = _Runner(enforce_eager=False)

    runner.release_memory()
    runner.resume_memory()

    assert runner.allocated_blocks == [7]
    # Deferred: the weights are still CPU tensors on this runner.
    assert runner.captures == 0
    assert runner._graphs_backup_keys == [1, 2]

    _report_weights_on_device(runner)
    runner._recapture_cudagraphs_if_needed()

    assert runner.captures == 1
    assert not hasattr(runner, "_graphs_backup_keys")


def test_eager_mode_has_no_graphs_to_invalidate():
    runner = _Runner(enforce_eager=True, with_graphs=False)

    runner.release_memory()

    assert runner.model.weight.numel() == 0
    assert runner.kv_cache is None
    assert not hasattr(runner, "_graphs_backup_keys")


def test_a_host_without_enforce_eager_releases():
    """`enforce_eager` defaults to True, i.e. to releasing.

    `tests/test_rollout_memory_manager.py` calls `_release_kv_cache` on a
    `SimpleNamespace` that has no such attribute, and a bare `self.enforce_eager`
    fails all three of its cases with `AttributeError` -- with no textual
    conflict for a rebase to report.
    """
    runner = SimpleNamespace(
        kv_cache=torch.zeros(8),
        config=SimpleNamespace(num_kvcache_blocks=7),
        model=nn.Linear(4, 4, bias=False),
        label="test",
    )
    runner._get_models_with_kv = lambda: [runner.model]

    MemoryManagerMixin._release_kv_cache(runner)

    assert runner.kv_cache is None
    assert runner._kv_cache_num_blocks == 7
