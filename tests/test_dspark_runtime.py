from types import SimpleNamespace

import pytest
import torch

from atom.config import CUDAGraphMode, DSparkConfig
from atom.spec_decode.dspark_runtime import (
    MAX_PENDING,
    DSparkRuntime,
    validate_online_runtime,
)


class Event:
    def __init__(self):
        self.ready = False
        self.records = 0

    def record(self):
        self.records += 1

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        assert other.ready
        return 3.0

    def synchronize(self):
        pytest.fail("online timing must never synchronize a GPU event")


@pytest.fixture
def runtime(monkeypatch):
    config = SimpleNamespace(
        dspark=DSparkConfig(
            online_sps=True,
            confidence_schedule=True,
            ragged=True,
            online_sps_sample_interval=2,
            online_sps_update_interval=16,
        ),
        max_num_seqs=8,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.TR_HYBRID),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_tbo=False,
        enforce_eager=False,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # A single worker: every gather returns its own rows.
    comm = SimpleNamespace(
        dp_size=1,
        tp_leader=True,
        gather_tp=lambda value: value.unsqueeze(0),
        gather_dp=lambda value: value.unsqueeze(0),
    )
    rt = DSparkRuntime(SimpleNamespace(config=config), comm=comm, event_factory=Event)
    # What VerifyScheduler.bind_sps hands it: offline profile or startup sweep.
    rt.bind([1000 / (1 + i * 0.1) for i in range(49)])
    return rt


def test_nothing_is_timed_before_a_curve_is_bound(runtime):
    """Warmup and capture forwards run before the startup sweep binds a curve."""
    runtime.model = None
    assert runtime.begin(batch(req_ids=[1]), MODE) is None
    assert runtime.step == 0
    assert runtime.table("base", 1) == "base"


def batch(**overrides):
    b = {
        "is_dummy_run": False,
        "total_tokens_num_prefill": 0,
        "total_tokens_num_decode": 6,
        "total_seqs_num": 1,
    }
    b.update(overrides)
    return SimpleNamespace(**b)


MODE = SimpleNamespace(use_cudagraph=False, token_graph=True, running_tokens=6)


def complete(rt, timing):
    if timing:
        rt.target_done(timing)
        rt.finish(timing)
        for event in (timing.start, timing.target_end, timing.end):
            event.ready = True


def test_runtime_updates_and_tables_refresh(runtime):
    rt = runtime
    base = torch.tensor([1000 / (1 + i * 0.1) for i in range(49)])
    assert rt.table(base, 1) is base
    for _ in range(80):
        complete(rt, rt.begin(batch(), MODE))
    assert rt.model.version == 5
    assert rt.model.accepted > 20
    assert rt.model.slope_ms == 0
    assert rt.model.fixed_ms(1) == pytest.approx(3 - 1.6)
    table = rt.table(base, 1)
    assert torch.all(table <= base)
    assert torch.all(table[1:] <= table[:-1])
    assert rt.table(base, 1) is table
    assert float(table[6]) == pytest.approx(1000 / 6)
    for _ in range(16):
        complete(rt, rt.begin(batch(), MODE))
    assert rt.table(base, 1) is not table


def test_inflight_samples_are_bounded_and_never_reused(runtime):
    rt = runtime
    events = []
    for _ in range(80):
        timing = rt.begin(batch(), MODE)
        if timing:
            rt.target_done(timing)
            rt.finish(timing)
            events.append(timing)
    assert len(rt.pending) == MAX_PENDING
    assert rt.model.accepted == 0
    assert rt.skipped_busy > 0
    assert all(t.start.records == 1 for t in events)
    for t in events:
        for e in (t.start, t.target_end, t.end):
            e.ready = True
    rt.begin(batch(), MODE)
    assert not rt.pending
    assert len(rt.free) == MAX_PENDING
    # Retired epoch cannot reappear as a late sample.
    assert rt.model.accepted == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_dummy_run": True},
        {"total_tokens_num_prefill": 1},
        {"total_tokens_num_decode": 0},
    ],
)
def test_non_decode_steps_do_not_advance_collective_cadence(runtime, overrides):
    for _ in range(20):
        assert runtime.begin(batch(**overrides), MODE) is None
    assert runtime.step == 0


@pytest.mark.parametrize(
    "mode", [None, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY]
)
def test_online_rejects_graph_modes_that_do_not_apply_ragged(runtime, mode):
    runtime.runner.config.compilation_config.cudagraph_mode = mode
    with pytest.raises(ValueError, match="TR_HYBRID token graphs"):
        validate_online_runtime(runtime.runner.config)


@pytest.mark.parametrize("sizes,ok", [("1,3,6", True), ("1,2,3,6", False)])
def test_online_requires_nested_query_buckets(runtime, sizes, ok):
    """The top-m search only visits survival prefixes; with non-nested caps
    (2 and 3) a smaller divisible graph can need a non-prefix allocation."""
    runtime.runner.config.dspark.ragged_graph_sizes = sizes
    if ok:
        validate_online_runtime(runtime.runner.config)
    else:
        with pytest.raises(ValueError, match="nested query buckets"):
            validate_online_runtime(runtime.runner.config)


@pytest.mark.parametrize("observe_only", [False, True])
def test_measured_cost_steers_verification_unless_observe_only(runtime, observe_only):
    """A large measured post-target cost makes another verified token cheaper
    than another round; observe-only keeps the static decision."""
    from atom.spec_decode.dspark_verify import VerifyScheduler

    scheduler = VerifyScheduler.__new__(VerifyScheduler)
    scheduler.runner, scheduler.runtime = runtime.runner, runtime
    scheduler.sps_table = torch.tensor([1000 / (1 + i * 0.1) for i in range(49)])
    scheduler.sts_temperatures = None
    confidence = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
    before = scheduler.compute_ell(confidence)
    runtime.runner.config.dspark.online_sps_observe_only = observe_only
    for _ in range(10):
        runtime.model.observe(1, 6, 1.6, 20.0)
    runtime.model.publish()
    after = scheduler.compute_ell(confidence)
    if observe_only:
        assert torch.equal(before, after) and 1 in runtime.table_cache
    else:
        assert after.item() > before.item()


def test_request_set_generation_only_changes_on_membership_or_order(runtime):
    rt = runtime
    complete(rt, rt.begin(batch(req_ids=[101]), MODE))
    sample = rt.begin(batch(req_ids=[101]), MODE)
    assert sample.context == 1
    complete(rt, sample)
    rt.begin(batch(req_ids=[102]), MODE)
    sample = rt.begin(batch(req_ids=[102]), MODE)
    assert sample.context == 2
    complete(rt, sample)
