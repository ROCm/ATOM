# SPDX-License-Identifier: MIT
"""Numerical acceptance weighs every scored label and rejects non-finite logits."""

import pytest
import torch

from .validate_checkpoint import compare_logits, summarize_records


def test_numerical_report_uses_token_weighted_nll():
    # Different sequence lengths must not give a one-token decode the same
    # weight as a long prefill.
    actual = torch.tensor([[[0.0, 2.0], [1.0, 0.0], [0.0, 3.0]]])
    expected = torch.tensor([[[0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]])
    labels = torch.tensor([1, 0, 1])
    full = compare_logits(actual, expected, labels)
    pieces = [
        compare_logits(actual[:, :2], expected[:, :2], labels[:2]),
        compare_logits(actual[:, 2:], expected[:, 2:], labels[2:]),
    ]
    combined = summarize_records(pieces)
    assert combined["tokens"] == combined["positions"] == 3
    assert combined["top1_agreement"] == 1
    assert combined["mean_nll_delta"] == pytest.approx(
        (full["target_nll_sum"] - full["reference_nll_sum"]) / 3, abs=1e-7
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_numerical_report_rejects_nonfinite_logits(value):
    actual = torch.tensor([[[0.0, value]]])
    with pytest.raises(AssertionError, match="Non-finite"):
        compare_logits(actual, torch.zeros_like(actual), torch.tensor([0]))
