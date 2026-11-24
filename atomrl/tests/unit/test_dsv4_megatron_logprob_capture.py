# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import pytest

from tests.integration import run_dsv4_megatron_logprob_capture as capture


def test_build_fixed_batch_marks_only_response_predictions() -> None:
    input_ids, attention_mask, response_mask = capture.build_fixed_batch(
        sequence_length=8,
        response_length=3,
    )

    assert input_ids.shape == (1, 8)
    assert attention_mask.tolist() == [[True] * 8]
    assert response_mask.tolist() == [[False, False, False, False, True, True, True]]


def test_build_batch_from_ids_preserves_natural_tokens() -> None:
    input_ids, _, response_mask = capture.build_batch_from_ids(
        [101, 102, 103, 104],
        response_length=2,
    )

    assert input_ids.tolist() == [[101, 102, 103, 104]]
    assert response_mask.tolist() == [[False, True, True]]


def test_compare_artifacts_reports_masked_logprob_difference() -> None:
    baseline = {
        "log_probs": [-1.0, -2.0, -3.0],
        "response_mask": [False, True, True],
    }
    candidate = {
        "log_probs": [-10.0, -2.5, -1.0],
        "response_mask": [False, True, True],
    }

    result = capture.compare_artifacts(baseline, candidate)

    assert result["token_count"] == 2
    assert result["mean_delta"] == pytest.approx(-0.75)
    assert result["mae"] == pytest.approx(1.25)
    assert result["max_abs"] == pytest.approx(2.0)


def test_compare_artifacts_rejects_different_masks() -> None:
    with pytest.raises(ValueError, match="response masks differ"):
        capture.compare_artifacts(
            {"log_probs": [-1.0], "response_mask": [True]},
            {"log_probs": [-1.0], "response_mask": [False]},
        )
