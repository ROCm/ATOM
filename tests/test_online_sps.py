from types import SimpleNamespace

import pytest

from atom.config import CUDAGraphMode, DSparkConfig
from atom.spec_decode.dspark_runtime import validate_online_runtime
from atom.spec_decode.online_sps import OnlineSPS, common_rank_samples


def estimator(**kw):
    return OnlineSPS([1000 / (1 + i * 0.1) for i in range(65)], 8, **kw)


def test_constant_runtime_overhead_is_fixed_not_marginal():
    model = estimator()
    for _ in range(12):
        model.observe(1, 6, 8.6, 2, context=1)
    model.publish()
    assert model.slope_ms_for(1) == 0
    assert model.fixed_ms(1) == pytest.approx(7)
    assert model.estimate_ms(12, 1) - model.estimate_ms(6, 1) == pytest.approx(0.6)
    assert model.estimate_ms(6, 1) == pytest.approx(10.6)


def test_genuine_marginal_cost_is_learned_only_for_observed_batch():
    model = estimator()
    for i in range(12):
        rows = 6 if i % 2 else 12
        residual = 7 + 0.2 * rows + (100 if i == 0 else 0)
        model.observe(1, rows, model.baseline_ms[rows] + residual, 2, context=1)
    model.publish()
    assert model.slope_ms_for(1) == pytest.approx(0.2)
    assert model.fixed_ms(1) == pytest.approx(7)
    assert model.slope_ms_for(2) == 0
    assert model.fixed_ms(2) == pytest.approx(7)
    assert model.support_by_batch[1] == 12


@pytest.mark.parametrize(
    "separation", ["batch", "request_set", "time", "unknown_context"]
)
def test_drift_cannot_masquerade_as_marginal_cost(separation):
    model = estimator()
    for i in range(24):
        high = i >= 12
        rows = 12 if high else 6
        batch = 2 if high and separation == "batch" else 1
        context = 2 if high and separation == "request_set" else 1
        if separation == "unknown_context":
            context = None
        step = i + 1 + (64 if high and separation == "time" else 0)
        model.observe(
            batch,
            rows,
            model.baseline_ms[rows] + (10 if high else 7),
            2,
            step=step,
            context=context,
        )
    model.publish()
    assert model.slope_ms == 0


def test_sparse_shape_support_does_not_invent_slope():
    model = estimator()
    for i in range(12):
        rows = 12 if i == 11 else 6
        model.observe(1, rows, model.baseline_ms[rows] + rows, 2, context=1)
    model.publish()
    assert model.slope_ms == 0


def test_window_expires_all_components_and_returns_to_baseline_floor():
    model = estimator(window=20)
    for _ in range(10):
        model.observe(1, 6, 7.6, 1)
    model.publish()
    assert model.fixed_ms(1) == pytest.approx(6)
    for _ in range(20):
        model.observe(2, 12, 0.2, 0)
    model.publish()
    assert len(model.samples) == 20
    assert all(s[2] == 12 for s in model.samples)
    assert model.slope_ms == 0
    assert model.estimate_ms(12, 2) == pytest.approx(2.2)
    model.publish(step=100)
    assert model.fixed_ms(1) == model.post_ms(1) == 0
    assert not model.samples


def test_min_samples_and_post_window():
    model = estimator()
    for _ in range(4):
        model.observe(1, 6, 7.6, 2)
    model.publish()
    assert model.slope_ms == model.fixed_ms(1) == model.post_ms(1) == 0
    model.observe(1, 6, 7.6, 1000)
    model.publish()
    assert model.post_ms(1) == 2
    for _ in range(10):
        model.observe(1, 6, 7.6, 3)
    model.publish()
    assert model.fixed_ms(1) == pytest.approx(6)
    assert model.post_ms(1) == 3


def test_tp_context_disagreement_rejects_sample():
    assert common_rank_samples([[(8, 1, 6, 2, 4, 1)], [(8, 1, 6, 2, 4, 2)]]) == []


@pytest.mark.parametrize(
    "args",
    [
        (0, 6, 2, 1),
        (9, 12, 2, 1),
        (1, 65, 2, 1),
        (1, 0, 2, 1),
        (1, 6, float("nan"), 1),
        (1, 6, 0, 1),
        (1, 6, 2, float("inf")),
        (1, 6, 2, -1),
    ],
)
def test_invalid_samples_do_not_update(args):
    model = estimator()
    assert not model.observe(*args)
    assert model.accepted == 0
    assert model.rejected == 1


def test_tp_matching_ignores_missing_and_shape_disagreement():
    rows = [
        [(8, 1, 6, 2, 4), (16, 1, 6, 3, 2), (24, 2, 12, 4, 1)],
        [(8, 1, 6, 5, 1), (24, 2, 6, 5, 1), (32, 1, 6, 1, 1)],
    ]
    assert common_rank_samples(rows) == [(8, 1, 6, 5, 4)]
    assert common_rank_samples([rows[0], []]) == []
    assert common_rank_samples([]) == []


def test_tp_invalid_duration_rejects_whole_sample():
    assert common_rank_samples([[(8, 1, 6, 1, 1)], [(8, 1, 6, float("nan"), 1)]]) == []


def test_costs_remain_positive_monotone_and_coverage_is_not_clamped():
    model = estimator()
    for _ in range(12):
        model.observe(1, 6, 7.6, 2)
    model.publish()
    times = [model.estimate_ms(i, 1) for i in range(65)]
    assert times == sorted(times)
    assert min(times) > 0
    with pytest.raises(ValueError):
        model.estimate_ms(65, 1)


@pytest.mark.parametrize(
    "kw",
    [
        {"online_sps": True},
        {"online_sps": True, "confidence_schedule": True, "ragged": True},
        {"online_sps_sample_interval": 0},
        {"online_sps_sample_interval": True},
        {"online_sps_sample_interval": 1.5},
        {"online_sps_update_interval": 1},
    ],
)
def test_invalid_online_config(kw):
    with pytest.raises(ValueError):
        DSparkConfig(**kw)


def test_off_by_default_and_explicit_profile_required():
    assert not DSparkConfig().online_sps
    cfg = DSparkConfig(
        online_sps=True,
        confidence_schedule=True,
        ragged=True,
        calibration_profile="measured.json",
    )
    assert cfg.online_sps


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("enable_expert_parallel", True),
        ("enable_tbo", True),
        ("enforce_eager", True),
    ],
)
def test_unvalidated_topologies_fail_closed(field, value):
    config = SimpleNamespace(
        dspark=SimpleNamespace(online_sps=True),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.TR_HYBRID),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_tbo=False,
        enforce_eager=False,
    )
    validate_online_runtime(config)
    setattr(config, field, value)
    with pytest.raises(ValueError):
        validate_online_runtime(config)


def test_data_parallel_rejected():
    config = SimpleNamespace(
        dspark=SimpleNamespace(online_sps=True),
        parallel_config=SimpleNamespace(data_parallel_size=2),
    )
    with pytest.raises(ValueError):
        validate_online_runtime(config)


def test_sparse_samples_expire_by_real_step_not_number_of_records():
    model = estimator(window=5000)
    for step in range(8, 6001, 8):
        model.observe(1, 6, 7.6, 2, step=step)
    model.publish(step=6000)
    assert len(model.samples) == 625
    assert model.samples[0][0] == 1008
    assert model.fixed_ms(1) == pytest.approx(6)


def test_skipped_samples_do_not_stretch_history_and_no_sample_epoch_expires():
    model = estimator(window=100)
    for step in range(1, 11):
        model.observe(1, 6, 7.6, 2, step=step)
    model.publish(step=64)
    assert model.fixed_ms(1) == pytest.approx(6)
    assert model.publish(step=105)
    assert len(model.samples) == 5
    assert model.slope_ms == 0  # Fewer than ten genuine observations remain.
    assert model.publish(step=128)
    assert not model.samples and model.fixed_ms(1) == 0 and model.post_ms(1) == 0
    assert not model.publish(step=192)


def test_duplicate_or_stale_sample_cannot_reenter_retired_window():
    model = estimator(window=100)
    assert model.observe(1, 6, 7.6, 2, step=8)
    assert not model.observe(1, 6, 7.6, 2, step=8)
    model.publish(step=256)
    assert not model.observe(1, 6, 7.6, 2, step=16)
    assert model.observe(1, 6, 7.6, 2, step=248)
    assert model.accepted == 2
