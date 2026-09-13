# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sleep and wake: what a release frees, and what it invalidates.

Sleep frees the weights and the KV pool and wake recaptures the decode graphs.
That recapture faults under `expandable_segments`, so there is an option --
`Config.sleep_keeps_memory_resident` -- to keep both allocated and skip it.
The whole point of the option is that nothing moves, which a test on
`updated`/`released` counters cannot see, so these assert on `data_ptr()` and
on object identity. It is off by default because keeping them costs exactly
the memory a colocated trainer sleeps the rollout engine to reclaim.

Either release invalidates the graphs, including the KV-only one that
`sleep(level=1)` performs.
"""

from types import SimpleNamespace

import pytest
import torch
from conftest import atom_config_double
from torch import nn

from atom.rollout import memory_manager
from atom.rollout.memory_manager import (
    MemoryManagerMixin,
    sleep_keeps_memory_resident,
)


class _Runner(MemoryManagerMixin):
    """The surface `MemoryManagerMixin` documents, and nothing else."""

    def __init__(self, *, enforce_eager, keep_resident, with_graphs=True):
        self.device = torch.device("cpu")
        self.label = "test"
        self.enforce_eager = enforce_eager
        self.config = atom_config_double(
            num_kvcache_blocks=7,
            enforce_eager=enforce_eager,
            sleep_keeps_memory_resident=keep_resident,
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


def _weight_addresses(runner):
    return [p.data_ptr() for p in runner.model.parameters()]


def test_default_still_releases_everything_outside_eager_mode():
    """The behaviour every non-eager deployment had before the option existed."""
    runner = _Runner(enforce_eager=False, keep_resident=False)

    runner.release_memory()

    assert runner.model.weight.numel() == 0
    assert runner.kv_cache is None
    assert runner._kv_cache_num_blocks == 7
    assert runner.graphs == {}
    assert runner._graphs_backup_keys == [1, 2]


def test_resident_sleep_moves_no_weight_and_frees_no_pool():
    runner = _Runner(enforce_eager=False, keep_resident=True)
    pool = runner.kv_cache
    addresses = _weight_addresses(runner)

    runner.release_memory()

    assert _weight_addresses(runner) == addresses
    assert runner.kv_cache is pool
    # Not recorded, because nothing has to be re-allocated on wake.
    assert not hasattr(runner, "_kv_cache_num_blocks")
    assert runner.graphs.keys() == {1, 2}
    assert not hasattr(runner, "_graphs_backup_keys")


def test_resident_sleep_and_wake_leaves_every_address_alone():
    """Two cycles: an address that survives one sleep has to survive the next."""
    runner = _Runner(enforce_eager=False, keep_resident=True)
    pool = runner.kv_cache
    addresses = _weight_addresses(runner)

    for _ in range(2):
        runner.release_memory()
        runner.resume_memory()

    assert _weight_addresses(runner) == addresses
    assert runner.kv_cache is pool
    assert runner.kv_cache.data_ptr() == pool.data_ptr()
    # Nothing was released, so nothing was re-allocated or recaptured.
    assert runner.allocated_blocks == []
    assert runner.captures == 0


def test_clear_kv_cache_still_zeroes_the_resident_pool():
    """The pool stays where it is; its contents do not survive the sleep."""
    runner = _Runner(enforce_eager=False, keep_resident=True)
    runner.kv_cache.fill_(3.0)
    pool = runner.kv_cache

    runner.release_memory()
    runner.clear_kv_cache()

    assert runner.kv_cache is pool
    assert torch.count_nonzero(pool) == 0


def _report_weights_on_device(runner):
    """`_recapture_cudagraphs_if_needed` gates on `param.is_cuda`.

    A CPU runner defers instead of recapturing, which is the right answer for
    a half-woken engine but not the case under test here.
    """
    runner.model = SimpleNamespace(parameters=lambda: [SimpleNamespace(is_cuda=True)])


def test_default_wake_recaptures_the_graphs_it_released():
    runner = _Runner(enforce_eager=False, keep_resident=False)

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


def test_releasing_only_the_kv_pool_still_invalidates_the_graphs():
    """`AsyncLLMEngine.sleep(level=1)`, the default, frees the pool and nothing else.

    The graphs captured the base of that pool, so they cannot be replayed
    against the one `_resume_kv_cache` allocates in its place. Nothing else
    would drop them either: the weights never moved, so the release path that
    used to own the graphs is not the one that runs.
    """
    runner = _Runner(enforce_eager=False, keep_resident=False)

    runner.release_memory(tags=["kv_cache"])

    assert runner.graphs == {}
    assert runner._graphs_backup_keys == [1, 2]

    # The weights never left the device on a level-1 sleep.
    _report_weights_on_device(runner)
    runner.resume_memory(tags=["kv_cache"])

    assert runner.allocated_blocks == [7]
    assert runner.captures == 1
    assert not hasattr(runner, "_graphs_backup_keys")


def test_resident_sleep_keeps_the_graphs_on_a_kv_only_release():
    runner = _Runner(enforce_eager=False, keep_resident=True)
    pool = runner.kv_cache

    runner.release_memory(tags=["kv_cache"])
    runner.resume_memory(tags=["kv_cache"])

    assert runner.kv_cache is pool
    assert runner.graphs.keys() == {1, 2}
    assert runner.captures == 0


def test_resident_wake_has_nothing_to_recapture():
    """Even with the weights on device: the graphs were never released."""
    runner = _Runner(enforce_eager=False, keep_resident=True)

    runner.release_memory()
    runner.resume_memory()
    _report_weights_on_device(runner)
    runner._recapture_cudagraphs_if_needed()

    assert runner.captures == 0
    assert runner.graphs.keys() == {1, 2}


def test_option_is_inert_under_enforce_eager():
    """No graphs to keep valid, so keeping the memory buys nothing."""
    runner = _Runner(enforce_eager=True, keep_resident=True, with_graphs=False)

    runner.release_memory()

    assert runner.model.weight.numel() == 0
    assert runner.kv_cache is None


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


def test_a_host_without_the_config_field_releases():
    """An older Config reaching a newer mixin must not silently keep memory."""
    runner = _Runner(enforce_eager=False, keep_resident=False)
    runner.config = SimpleNamespace(num_kvcache_blocks=7)

    assert sleep_keeps_memory_resident(runner) is False
