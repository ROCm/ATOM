import json
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
def runtime(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps({"sps_table": [1000 / (1 + i * 0.1) for i in range(49)]})
    )
    config = SimpleNamespace(
        dspark=DSparkConfig(
            online_sps=True,
            confidence_schedule=True,
            ragged=True,
            calibration_profile=str(path),
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
    return DSparkRuntime(
        SimpleNamespace(config=config), group=object(), event_factory=Event
    )


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


def test_capture_and_eager_are_not_samples(runtime, monkeypatch):
    assert runtime.begin(batch(), SimpleNamespace(token_graph=False)) is None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert runtime.begin(batch(), MODE) is None
    assert runtime.step == 0


@pytest.mark.parametrize(
    "mode", [None, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY]
)
def test_online_rejects_graph_modes_that_do_not_apply_ragged(runtime, mode):
    runtime.runner.config.compilation_config.cudagraph_mode = mode
    with pytest.raises(ValueError, match="TR_HYBRID token graphs"):
        validate_online_runtime(runtime.runner.config)


def test_measured_cost_changes_verification_for_identical_confidence(runtime):
    from atom.spec_decode.dspark_verify import VerifyScheduler

    scheduler = VerifyScheduler.__new__(VerifyScheduler)
    scheduler.runner = runtime.runner
    scheduler.runtime = runtime
    scheduler.sps_table = torch.tensor([1000 / (1 + i * 0.1) for i in range(49)])
    scheduler.sts_temperatures = None
    confidence = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
    initial = scheduler.compute_ell(confidence)
    # A large measured post-target cost makes an extra accepted token cheaper
    # than another target/draft round, without changing confidence or sampling.
    for _ in range(10):
        runtime.model.observe(1, 6, 1.6, 20.0)
    runtime.model.publish()
    updated = scheduler.compute_ell(confidence)
    assert updated.item() > initial.item()


def test_observe_only_still_builds_measured_table_but_keeps_static_decision(runtime):
    from atom.spec_decode.dspark_verify import VerifyScheduler

    scheduler = VerifyScheduler.__new__(VerifyScheduler)
    scheduler.runner, scheduler.runtime = runtime.runner, runtime
    scheduler.sps_table = torch.tensor([1000 / (1 + i * 0.1) for i in range(49)])
    scheduler.sts_temperatures = None
    confidence = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
    before = scheduler.compute_ell(confidence)
    runtime.runner.config.dspark.online_sps_observe_only = True
    for _ in range(10):
        runtime.model.observe(1, 6, 1.6, 20)
    runtime.model.publish()
    after = scheduler.compute_ell(confidence)
    assert torch.equal(before, after)
    assert 1 in runtime.table_cache
    assert not torch.equal(runtime.table_cache[1], scheduler.sps_table)


def test_publication_without_ready_timings_expires_and_invalidates_cache(runtime):
    rt = runtime
    rt.model.window = 20
    for step in range(1, 11):
        rt.model.observe(1, 6, 7.6, 2, step=step)
    rt.model.publish(step=16)
    base = torch.tensor([1000 / (1 + i * 0.1) for i in range(49)])
    rt.table(base, 1)
    rt.step = 64
    rt._publish()
    assert not rt.model.samples
    assert rt.model.slope_ms == 0
    assert not rt.table_cache


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


def test_runtime_tables_use_batch_local_affine_coefficients(runtime):
    rt = runtime
    for i in range(12):
        rows = 6 if i % 2 else 12
        rt.model.observe(
            1, rows, rt.model.baseline_ms[rows] + 7 + 0.2 * rows, 2, context=1
        )
    rt.model.publish()
    base = torch.tensor([1000 / (1 + i * 0.1) for i in range(49)])
    for b in [1, 2]:
        table = rt.table(base, b)
        for rows in [6, 12]:
            assert table[rows].item() == pytest.approx(
                1000 / rt.model.estimate_ms(rows, b)
            )
    assert not torch.equal(rt.table(base, 1), rt.table(base, 2))
