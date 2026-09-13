# SPDX-License-Identifier: MIT
"""A paired quality report must compare the same prompts and retain losses."""

from copy import deepcopy

import pytest


@pytest.fixture
def paired_summary():
    pytest.importorskip("scipy.stats")
    from .compare_lm_eval import paired_summary as summarize

    return summarize


def result(scores):
    return {
        "reference_manifest": {"revision": "fixed"},
        "results": {"task": {"acc,none": sum(scores) / len(scores)}},
        "samples": {
            "task": [
                {
                    "doc_id": i,
                    "filter": "none",
                    "doc_hash": str(i),
                    "prompt_hash": f"prompt-{i}",
                    "target_hash": f"label-{i}",
                    "acc": score,
                }
                for i, score in enumerate(scores)
            ]
        },
    }


def test_paired_report_counts_both_gains_and_losses(paired_summary):
    target, reference = result([1, 0, 1, 1]), result([0, 1, 1, 0])
    metrics = paired_summary(target, reference)["task"]["acc,none"]
    assert metrics["gains"] == 2
    assert metrics["losses"] == 1
    assert metrics["delta"] == 0.25
    assert metrics["documents"] == 4
    lo, hi = metrics["paired_bootstrap_95_ci"]
    assert lo < 0 < hi


@pytest.mark.parametrize("field", ["doc_hash", "prompt_hash", "target_hash"])
def test_paired_report_rejects_different_inputs(field, paired_summary):
    target = result([1, 0])
    reference = deepcopy(target)
    reference["samples"]["task"][0][field] = "different"
    with pytest.raises(ValueError, match=field):
        paired_summary(target, reference)


@pytest.mark.parametrize("side", ["target", "reference"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_paired_report_rejects_nonfinite_likelihood(side, value, paired_summary):
    target, reference = result([1, 0]), result([1, 0])
    artifact = target if side == "target" else reference
    artifact["samples"]["task"][0]["filtered_resps"] = [(value, False)]
    with pytest.raises(ValueError, match="Non-finite"):
        paired_summary(target, reference)


@pytest.mark.parametrize("side", ["target", "reference"])
def test_paired_report_rejects_duplicate_documents(side, paired_summary):
    target, reference = result([1, 0]), result([1, 0])
    artifact = target if side == "target" else reference
    artifact["samples"]["task"].append(deepcopy(artifact["samples"]["task"][0]))
    with pytest.raises(ValueError, match="repeated"):
        paired_summary(target, reference)
