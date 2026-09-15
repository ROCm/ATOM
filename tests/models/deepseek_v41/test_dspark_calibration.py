# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from .calibrate_dspark_confidence import (
    ece,
    fit_temperatures,
    observations,
    scaled_survival,
)


def test_ece_includes_probability_one_and_empty_bins():
    assert ece(np.array([0.0, 1.0]), np.array([False, True])) == 0
    assert ece(np.array([0.25, 0.75]), np.array([False, True])) == 0.25


def test_unverified_suffixes_and_other_cohort_are_excluded():
    report = {
        "workload": {"batches": [{"cohort": "fit"}, {"cohort": "holdout"}]},
        "cases": [
            {
                "case": 0,
                "confidence_observations": [
                    {
                        "confidence": [0.8, 0.7, 0.6],
                        "verified_drafts": 2,
                        "accepted_drafts": 1,
                    }
                ],
            },
            {
                "case": 1,
                "confidence_observations": [
                    {
                        "confidence": [0.1, 0.2, 0.3],
                        "verified_drafts": 3,
                        "accepted_drafts": 3,
                    }
                ],
            },
        ],
    }
    confidence, labels, mask = observations(report, "fit")
    assert confidence.tolist() == [[0.8, 0.7, 0.6]]
    assert labels.tolist() == [[True, False, False]]
    assert mask.tolist() == [[True, True, False]]
    with pytest.raises(ValueError, match="position 3"):
        fit_temperatures(confidence, labels, mask)


def test_temperature_fit_reduces_error_for_known_overconfidence():
    # Each bin has exactly 80% observed survival, while the head predicts 99%.
    confidence = np.full((100, 1), 0.99)
    labels = (np.arange(100) < 80)[:, None]
    observed = np.ones_like(labels)
    temperature = fit_temperatures(confidence, labels, observed)
    calibrated = scaled_survival(confidence, temperature)
    assert temperature[0] > 1
    assert abs(calibrated.mean() - 0.8) < 0.005
    assert ece(calibrated[:, 0], labels[:, 0]) < ece(confidence[:, 0], labels[:, 0])
