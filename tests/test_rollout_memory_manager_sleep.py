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

"the graphs" is four stores, not one. `runner.graphs` is the manual
whole-forward capture; PIECEWISE -- `--level 3`, the default -- fills only the
per-piece wrappers, TBO keeps a parallel entry per shape, and the drafter holds
a recording per warmed batch. Each is asked separately below, because a gate
they shared is what left three of them behind.
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
from atom.spec_decode.draft_graph import DraftGraph, StagedInput
from atom.utils.graph_holders import register_graph_holder


class _Runner(MemoryManagerMixin):
    """The surface `MemoryManagerMixin` documents, and nothing else."""

    def __init__(
        self, *, enforce_eager, keep_resident, with_graphs=True, available_blocks=7
    ):
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
        # What a fresh sizing would come back with, and how often one is asked
        # for. Default equals the startup count so a re-derivation is invisible;
        # pass less to stand in for a peer holding memory at wake time.
        self._available_blocks = available_blocks
        self.sizings = 0

    def _get_models_with_kv(self):
        return [self.model]

    def get_num_blocks(self):
        self.sizings += 1
        return {"num_kvcache_blocks": self._available_blocks}

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
    # `_resume_kv_cache` logs the headroom it is about to allocate into.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (1 << 30, 2 << 30))
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


def _as_if_released(runner, keys=(1,)):
    """The state a release leaves, for a test that cannot get there by calling
    one -- `sleep_keeps_memory_resident` declines to release at all.

    Both halves, because they mean different things: the keys are the manual
    store's record, and the flag is "anything at all was dropped", which is what
    the wake keys on.
    """
    runner._graphs_backup_keys = list(keys)
    runner._graphs_released_for_sleep = True


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


def test_a_wake_allocates_the_size_it_slept_at_however_little_is_free():
    """The regression: a peer holding memory at wake time must not resize us.

    A wake used to re-derive the count and `min()` it with the saved one, which
    silently handed back a smaller pool. Two things outside this process are
    built against the startup count and neither hears about the new one: the
    decode graphs captured against the original pool, and -- the one that
    corrupts -- `BlockManager`'s `BlockPool`, sized in the *engine* process from
    the count `EngineCore.__init__` copied out of the startup reply. The
    scheduler went on issuing block ids past the end of the reallocated pool.

    Measured on Qwen3-30B-A3B: 33562 blocks at startup, 8316 on the first wake
    with 176GB still free, then `Memory access fault by GPU node-N` on all eight
    replicas.
    """
    runner = _Runner(enforce_eager=False, keep_resident=False, available_blocks=3)

    runner.release_memory(tags=["kv_cache"])
    _report_weights_on_device(runner)
    runner.resume_memory(tags=["kv_cache"])

    assert runner.allocated_blocks == [7]


def test_a_wake_that_cannot_allocate_keeps_the_size_for_the_next_one():
    """Dropping the clamp made `allocate_kv_cache` the thing that raises.

    `_kv_cache_num_blocks` is the only record of how big the pool was, so
    clearing it before the allocation loses that record on the way out. The next
    wake then takes the "No KV cache num_blocks to resume from" guard, returns,
    and `resume_memory` reports success for an engine that has no pool at all --
    which faults on the first forward instead of at the point of failure.
    """
    runner = _Runner(enforce_eager=False, keep_resident=False)
    runner.release_memory(tags=["kv_cache"])

    def _oom(num_blocks):
        raise torch.cuda.OutOfMemoryError("out of memory")

    runner.allocate_kv_cache = _oom
    with pytest.raises(torch.cuda.OutOfMemoryError):
        runner._resume_kv_cache()

    assert runner._kv_cache_num_blocks == 7
    assert runner.kv_cache is None


def test_a_wake_does_not_size_the_pool_at_all():
    """Not just the same answer -- the question is not asked.

    `resume_memory` broadcasts the command rather than a count, so every TP rank
    ran its own sizing against its own `mem_get_info()`. Ranks could therefore
    disagree with each other, and a block id valid on rank 0 be out of bounds on
    rank 3. One `get_num_blocks` at startup is what makes them agree.
    """
    runner = _Runner(enforce_eager=False, keep_resident=False)

    runner.release_memory(tags=["kv_cache"])
    _report_weights_on_device(runner)
    runner.resume_memory(tags=["kv_cache"])

    assert runner.sizings == 0


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


# ── the TBO graph store ───────────────────────────────────────────────────


def _with_tbo(runner, graphs=(("bs", "q"),)):
    """`UBatchWrapper` keeps a second store beside `runner.graphs`, holding each
    graph, its per-ubatch contexts and the output tensor it captured.

    Hung on the real model rather than a stand-in, because the release path
    walks it for parameters and KV views on the way past.
    """
    runner.model.tbo_graphs = {key: object() for key in graphs}
    return runner.model.tbo_graphs


@pytest.mark.parametrize("tags", [None, ["kv_cache"]])
def test_a_release_clears_the_tbo_graphs_too(tags):
    """`ModelRunner.exit()` clears both stores; a release for sleep has to
    clear the same two. Left behind, the entry pins the graph's private memory
    pool -- the footprint the caller went to sleep to reclaim."""
    runner = _Runner(enforce_eager=False, keep_resident=False)
    tbo_graphs = _with_tbo(runner)

    runner.release_memory(**({} if tags is None else {"tags": tags}))

    assert tbo_graphs == {}
    assert runner.graphs == {}


def test_the_tbo_store_is_cleared_even_with_no_plain_graphs():
    """The early return used to be `not runner.graphs`, which skipped the TBO
    store along with it."""
    runner = _Runner(enforce_eager=False, keep_resident=False, with_graphs=False)
    tbo_graphs = _with_tbo(runner)

    memory_manager.release_cudagraphs(runner)

    assert tbo_graphs == {}


def test_a_resident_sleep_keeps_the_tbo_graphs():
    runner = _Runner(enforce_eager=False, keep_resident=True)
    tbo_graphs = _with_tbo(runner)

    runner.release_memory()

    assert tbo_graphs.keys() == {("bs", "q")}


def test_a_release_clears_the_captured_logits_too():
    """`graph_logits` holds the logits tensor each capture produced, allocated
    from the graph's own pool. Clearing `graphs` stops the replay but leaves
    that reference, so the pool `empty_cache()` is about to reclaim stays
    pinned -- the same leak shape as the TBO store."""
    runner = _Runner(enforce_eager=False, keep_resident=False)
    runner.graph_logits = {(1, 1): torch.zeros(4)}

    runner.release_memory(tags=["kv_cache"])

    assert runner.graph_logits == {}


def test_a_resident_sleep_keeps_the_captured_logits():
    runner = _Runner(enforce_eager=False, keep_resident=True)
    runner.graph_logits = {(1, 1): torch.zeros(4)}

    runner.release_memory()

    assert runner.graph_logits.keys() == {(1, 1)}


def test_a_host_without_graph_logits_still_releases():
    """`graph_logits` only exists once `capture_cudagraph` has run, and the
    mixin's methods are called on stand-ins that provide what they touch."""
    runner = _Runner(enforce_eager=False, keep_resident=False)
    assert not hasattr(runner, "graph_logits")

    runner.release_memory(tags=["kv_cache"])

    assert runner.graphs == {}


# ── the piecewise store, which is what the default configuration fills ────


class _PieceHolder:
    """Stands in for a `CUDAGraphWrapper`: the graphs of one compiled piece.

    Not the real one: `atom/utils/cuda_graph.py` reaches aiter, which CI has no
    build of. What the release path depends on is the registration protocol, and
    the real wrapper's conformance to it is asserted in
    `tests/test_graph_holders.py`.
    """

    def __init__(self, graphs=2):
        self.graphs = graphs
        register_graph_holder(self)

    def release_graphs(self) -> int:
        dropped, self.graphs = self.graphs, 0
        return dropped


def _with_piecewise(runner, tokens=(128, 256)):
    """The state a PIECEWISE capture pass leaves behind.

    `runner.graphs` stays EMPTY -- the capture loop moves on before the
    assignment, every shape's graph living in the per-piece wrappers instead --
    and the runner's record of which shapes got one is
    `_piecewise_captured_tokens`. PIECEWISE is `--level 3`, the default, so this
    is the ordinary case rather than a corner of one.
    """
    runner._piecewise_captured_tokens = set(tokens)
    runner._piecewise_sorted_tokens = sorted(tokens)
    return _PieceHolder()


@pytest.mark.parametrize("tags", [None, ["kv_cache"]])
def test_a_release_drops_the_graphs_a_piecewise_capture_made(tags):
    """A gate on `runner.graphs` made the whole release a no-op here, on the
    configuration whose fault it exists to prevent."""
    runner = _Runner(enforce_eager=False, keep_resident=False, with_graphs=False)
    holder = _with_piecewise(runner)

    runner.release_memory(**({} if tags is None else {"tags": tags}))

    assert holder.graphs == 0
    assert runner._piecewise_captured_tokens == set()
    # Not bookkeeping: an uncleared record is what would let the next step
    # dispatch PIECEWISE and replay into the pool this release just freed.
    assert runner._piecewise_sorted_tokens == []


def test_a_piecewise_only_release_still_arranges_the_recapture():
    """`_graphs_backup_keys` records the manual store and is never set here, so
    keying the wake on it skipped the same configuration the release did."""
    runner = _Runner(enforce_eager=False, keep_resident=False, with_graphs=False)
    _with_piecewise(runner)

    runner.release_memory(tags=["kv_cache"])

    assert not hasattr(runner, "_graphs_backup_keys")
    assert runner._graphs_released_for_sleep is True

    _report_weights_on_device(runner)
    runner.resume_memory(tags=["kv_cache"])

    assert runner.captures == 1
    assert runner._graphs_released_for_sleep is False


def test_a_resident_sleep_keeps_the_piecewise_graphs():
    runner = _Runner(enforce_eager=False, keep_resident=True, with_graphs=False)
    holder = _with_piecewise(runner)

    runner.release_memory()

    assert holder.graphs == 2
    assert runner._piecewise_captured_tokens == {128, 256}


def test_the_captured_logits_are_dropped_with_no_manual_graphs_to_gate_on():
    """The clear sat behind the gate on `runner.graphs`, so under PIECEWISE the
    leak it was added for survived the release."""
    runner = _Runner(enforce_eager=False, keep_resident=False, with_graphs=False)
    runner.graph_logits = {(1, 1): torch.zeros(4)}

    runner.release_memory(tags=["kv_cache"])

    assert runner.graph_logits == {}
    assert runner._graphs_released_for_sleep is True


def test_the_warning_reaches_the_configuration_that_walks_into_the_fault(
    monkeypatch, caplog
):
    """Behind the same gate, and so silent for the same runners."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    runner = _Runner(enforce_eager=False, keep_resident=False, with_graphs=False)
    _with_piecewise(runner)

    with caplog.at_level("WARNING", logger="atom"):
        runner.release_memory(tags=["kv_cache"])

    assert "sleep_keeps_memory_resident" in caplog.text


def test_a_runner_that_captured_nothing_is_told_nothing():
    """ "Was anything captured" still has to be a gate: a non-eager runner that
    never got as far as a capture has no recapture to arrange and no fault to
    warn about."""
    runner = _Runner(enforce_eager=False, keep_resident=False, with_graphs=False)

    runner.release_memory(tags=["kv_cache"])

    assert not getattr(runner, "_graphs_released_for_sleep", False)
    assert runner.captures == 0


# ── the drafter's recordings ──────────────────────────────────────────────


def _with_draft_graphs(runner, batches=(48,)):
    """A draft pass holding a recording per batch it was warmed at.

    The real `DraftGraph`: it is deliberately importable without a GPU aiter
    build, so nothing here has to stand in for it.
    """
    pass_ = DraftGraph(forward=lambda bs, **staged: None, inputs={"ids": StagedInput()})
    for bs in batches:
        pass_._cuda_graphs[bs] = ("graph", "out")
    runner.drafter = SimpleNamespace(draft_graphs=(pass_,))
    return pass_


def test_a_release_drops_the_draft_recordings_too():
    """A draft pass WRITES the KV it attends, so its recording holds the base of
    the pool exactly the way a decode graph does -- and `ATOM_DRAFT_CUDAGRAPH`
    is on by default."""
    runner = _Runner(enforce_eager=False, keep_resident=False, with_graphs=False)
    pass_ = _with_draft_graphs(runner)

    runner.release_memory(tags=["kv_cache"])

    assert not pass_.is_captured(48)
    assert runner._graphs_released_for_sleep is True


def test_a_resident_sleep_keeps_the_draft_recordings():
    runner = _Runner(enforce_eager=False, keep_resident=True, with_graphs=False)
    pass_ = _with_draft_graphs(runner)

    runner.release_memory()

    assert pass_.is_captured(48)


def test_a_drafter_with_no_recordings_releases_nothing():
    """`draft_graphs` is empty for a flavor that declares no warmable pass, and
    absent entirely on a runner without a drafter."""
    runner = _Runner(enforce_eager=False, keep_resident=False, with_graphs=False)
    runner.drafter = SimpleNamespace(draft_graphs=())

    runner.release_memory(tags=["kv_cache"])

    assert not getattr(runner, "_graphs_released_for_sleep", False)


def test_eager_has_no_tbo_graphs_to_clear():
    runner = _Runner(enforce_eager=True, keep_resident=False, with_graphs=False)
    tbo_graphs = _with_tbo(runner)

    memory_manager.release_cudagraphs(runner)

    assert tbo_graphs.keys() == {("bs", "q")}


# ── saying so before the fault ────────────────────────────────────────────


def test_releasing_under_expandable_segments_names_the_way_out(monkeypatch, caplog):
    """Recapture on wake is what faults under expandable segments, and the
    handler that knows it only speaks after the fact -- by which point it has
    pinned the runner to eager. The release site knows both facts."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    runner = _Runner(enforce_eager=False, keep_resident=False)

    with caplog.at_level("WARNING", logger="atom"):
        runner.release_memory(tags=["kv_cache"])

    assert "sleep_keeps_memory_resident" in caplog.text
    assert "expandable_segments" in caplog.text


def test_no_warning_when_the_allocator_is_not_configured_that_way(monkeypatch, caplog):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    runner = _Runner(enforce_eager=False, keep_resident=False)

    with caplog.at_level("WARNING", logger="atom"):
        runner.release_memory(tags=["kv_cache"])

    assert "expandable_segments" not in caplog.text


def test_no_warning_when_the_operator_already_took_the_way_out(monkeypatch, caplog):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    runner = _Runner(enforce_eager=False, keep_resident=True)

    with caplog.at_level("WARNING", logger="atom"):
        runner.release_memory(tags=["kv_cache"])

    assert "expandable_segments" not in caplog.text


def test_a_failed_recapture_says_the_option_is_now_inert(caplog):
    """`sleep_keeps_memory_resident()` reads `enforce_eager` first and answers
    False for an eager runner, which is right -- nothing left to keep valid --
    but it means the option the operator set stops taking effect, and from here
    its log lines never appear again."""
    runner = _Runner(enforce_eager=False, keep_resident=True)
    runner.graphs = {1: object()}
    _as_if_released(runner)
    _report_weights_on_device(runner)

    def _boom():
        raise RuntimeError("recapture faulted")

    runner.capture_cudagraph = _boom

    with caplog.at_level("WARNING", logger="atom"):
        runner._recapture_cudagraphs_if_needed()

    assert runner.enforce_eager is True
    assert sleep_keeps_memory_resident(runner) is False
    assert "now inert" in caplog.text


def test_a_failed_recapture_is_quiet_about_an_option_nobody_set(caplog):
    runner = _Runner(enforce_eager=False, keep_resident=False)
    _as_if_released(runner)
    _report_weights_on_device(runner)
    runner.capture_cudagraph = lambda: (_ for _ in ()).throw(RuntimeError("boom"))

    with caplog.at_level("WARNING", logger="atom"):
        runner._recapture_cudagraphs_if_needed()

    assert runner.enforce_eager is True
    # Cleared even though it failed: there is nothing left to recapture, and a
    # wake that kept asking would try again on every cycle.
    assert runner._graphs_released_for_sleep is False
    assert "now inert" not in caplog.text
