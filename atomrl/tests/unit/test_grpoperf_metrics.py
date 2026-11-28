# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""GRPOPerf's metrics.py: NeMo-RL's Tokens/sec/GPU formula and gate."""

import importlib.util
import math
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "examples" / "GRPOPerf" / "metrics.py"
_SPEC = importlib.util.spec_from_file_location("metrics", _PATH)
metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metrics)


def _line(step, tokens, seconds, resp, prompt=100.0, mpe=1.01, grad=0.5):
    return (
        f"2026-11-24 INFO step={step} perf/total_num_tokens={tokens:g} "
        f"timing_s/step={seconds:g} response_length/mean={resp:g} "
        f"prompt_length/mean={prompt:g} rollout_corr/token_mult_prob_error={mpe:g} "
        f"timing_s/gen={seconds / 2:g} grad_norm={grad:g} loss=0.1 reward/accuracy=0.5"
    )


def test_tokens_per_sec_per_gpu_uses_nv_formula_over_steps_3_to_8():
    lines = [_line(s, 2048 * 3300.0, 100.0 + s, 3200.0) for s in range(1, 11)]
    steps = metrics.parse_steps(lines)
    out = metrics.summarize(steps, gpus=16, first=3, last=8)
    want = sum(2048 * 3300.0 / (100.0 + s) / 16 for s in range(3, 9)) / 6
    assert math.isclose(out["tokens_per_sec_per_gpu"], want)
    assert math.isclose(
        out["total_step_time_s"], sum(100.0 + s for s in range(3, 9)) / 6
    )
    assert math.isclose(out["samples_per_step"], 2048.0)
    assert out["g_average_seq_len"] == 3200.0


def test_gate_fails_on_large_mismatch_and_non_finite_grad():
    good = metrics.parse_steps([_line(s, 1e6, 10.0, 3000.0) for s in range(1, 11)])
    assert metrics.gate(good)["pass"]

    bad_mpe = metrics.parse_steps(
        [_line(s, 1e6, 10.0, 3000.0, mpe=1.2) for s in range(1, 11)]
    )
    assert not metrics.gate(bad_mpe)["checks"]["token_mult_prob_error_median_lt_1.1"]

    bad_grad = metrics.parse_steps(
        [_line(s, 1e6, 10.0, 3000.0, grad=float("nan")) for s in range(1, 11)]
    )
    assert not metrics.gate(bad_grad)["checks"]["loss_and_grad_norm_finite"]


def test_gate_keeps_each_value_on_its_own_step_when_one_is_missing():
    """Step 3 did not log the metric; steps 4 and 5 must keep their own values
    rather than shift onto 3 and 4."""
    values = {1: 1.01, 2: 1.02, 3: 1.03, 4: 1.04, 5: 1.05}
    lines = [_line(s, 1e6, 10.0, 3000.0, mpe=v) for s, v in values.items()]
    lines[2] = lines[2].replace("rollout_corr/token_mult_prob_error=", "other=")

    out = metrics.gate(metrics.parse_steps(lines))

    assert out["token_mult_prob_error"] == {1: 1.01, 2: 1.02, 4: 1.04, 5: 1.05}
    assert out["pass"]
