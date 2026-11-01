from types import SimpleNamespace

import pytest

from atom.config import CUDAGraphMode, DSparkConfig
from atom.spec_decode.dspark_runtime import validate_online_runtime
from atom.spec_decode.online_sps import OnlineSPS, common_rank_samples


def estimator(**kw):
    return OnlineSPS([1000 / (1 + i * 0.1) for i in range(65)], 8, **kw)


def fill(
    model,
    residual,
    sizes=(6, 12, 18),
    n=30,
    batch=lambda rows: 1,
    context=lambda i: None,
):
    for i in range(n):
        rows = sizes[i % len(sizes)]
        model.observe(
            batch(rows),
            rows,
            model.baseline_ms[rows] + residual(rows, i),
            2,
            context=context(i),
        )
    model.publish()


def test_marginal_cost_is_learned_beside_a_separate_intercept():
    model = estimator()
    fill(model, lambda rows, i: 7 + 0.2 * rows)
    assert model.slope_ms_for(1) == pytest.approx(0.2)
    assert model.fixed_ms(1) == pytest.approx(7)


def test_a_constant_offset_never_becomes_a_per_row_cost():
    """DeepInfer divides each residual by the graph size; the intercept keeps
    a constant offset flat."""
    model = estimator()
    fill(model, lambda rows, i: 7, sizes=(6, 12, 24))
    assert model.slope_ms == 0
    assert model.fixed_ms(1) == pytest.approx(7)
    assert model.estimate_ms(24, 1) - model.estimate_ms(6, 1) == pytest.approx(
        model.baseline_ms[24] - model.baseline_ms[6]
    )


@pytest.mark.parametrize("vary", ["batch", "context"])
def test_fixed_overheads_of_other_batches_or_contexts_are_not_a_slope(vary):
    """Fixed-only 6/7/8 ms at (batch, rows) = (1,6)/(2,12)/(3,18), or across
    three request sets of one batch: no per-row cost exists, so none is learned
    and batch 3's six-row cost is not underestimated."""
    model = estimator()
    level = {6: 6, 12: 7, 18: 8}
    if vary == "batch":
        fill(model, lambda rows, i: level[rows], n=60, batch=lambda rows: rows // 6)
    else:
        # Each request set runs one graph size: a context boundary, not a q change.
        fill(model, lambda rows, i: level[rows], n=60, context=lambda i: i % 3)
    assert model.slope_ms == 0
    assert all(model.slope_ms_for(b) == 0 for b in (1, 2, 3))


def test_contrasts_inside_each_batch_and_context_supply_and_share_the_slope():
    """Within one request set the graph size changes with the verify cap; three
    contexts with different fixed costs (6/9/12 ms) still give the true slope,
    and a batch with no contrasts of its own borrows it."""
    model = estimator()
    fill(
        model,
        lambda rows, i: 3 * ((i // 3) % 3 + 2) + 0.2 * rows,
        n=90,
        context=lambda i: (i // 3) % 3,
    )
    assert model.slope_ms_for(1) == pytest.approx(0.2)
    model.observe(2, 6, model.baseline_ms[6] + 9, 2)
    model.publish()
    assert 2 not in model.slope_by_batch
    assert model.slope_ms_for(2) == pytest.approx(0.2)


@pytest.mark.parametrize("sizes", [(6,), (6, 12)])
def test_fewer_than_three_sizes_give_no_slope(sizes):
    """Two points always admit a line; a level shift is not a slope."""
    model = estimator()
    fill(model, lambda rows, i: 7 if rows == 6 else 10, sizes=sizes)
    assert model.slope_ms == 0


def test_estimates_stay_positive_monotone_and_inside_the_profile():
    model = estimator()
    fill(model, lambda rows, i: 7)
    times = [model.estimate_ms(i, 1) for i in range(65)]
    assert times == sorted(times) and min(times) > 0
    with pytest.raises(ValueError):
        model.estimate_ms(65, 1)


def test_window_expires_by_step_and_returns_to_the_baseline():
    model = estimator(window=100)
    for step in range(1, 11):
        model.observe(1, 6, 7.6, 2, step=step)
    model.publish(step=64)
    assert model.fixed_ms(1) == pytest.approx(6) and model.post_ms(1) == 2
    assert model.publish(step=128)
    assert not model.samples and model.fixed_ms(1) == model.post_ms(1) == 0
    assert not model.observe(1, 6, 7.6, 2, step=8)  # retired steps stay out


@pytest.mark.parametrize(
    "args", [(0, 6, 2, 1), (9, 12, 2, 1), (1, 65, 2, 1), (1, 6, float("nan"), 1)]
)
def test_invalid_samples_are_rejected(args):
    model = estimator()
    assert not model.observe(*args) and model.rejected == 1


def test_tp_samples_match_on_step_shape_and_context_with_max_times():
    rows = [
        [(8, 1, 6, 2, 4, 1), (16, 1, 6, 3, 2, 1), (24, 2, 12, 4, 1, 1)],
        [(8, 1, 6, 5, 1, 1), (24, 2, 6, 5, 1, 1), (32, 1, 6, 1, 1, 1)],
    ]
    assert common_rank_samples(rows) == [(8, 1, 6, 5, 4)]
    assert common_rank_samples([[(8, 1, 6, 2, 4, 1)], [(8, 1, 6, 2, 4, 2)]]) == []
    assert common_rank_samples([[(8, 1, 6, 1, 1)], [(8, 1, 6, float("nan"), 1)]]) == []


@pytest.mark.parametrize(
    "kw",
    [
        {"online_sps": True},
        {"online_sps_sample_interval": 0},
        {"online_sps_update_interval": 1},
    ],
)
def test_invalid_online_config(kw):
    with pytest.raises(ValueError):
        DSparkConfig(**kw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_parallel_size", 2),
        ("decode_context_parallel_size", 2),
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
        enable_tbo=False,
        enforce_eager=False,
    )
    validate_online_runtime(config)
    setattr(config, field, value)
    with pytest.raises(ValueError):
        validate_online_runtime(config)
