# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from types import SimpleNamespace

import pytest

from tests.integration import run_dsv4_weight_sync_integrity as integrity


def test_summarize_replica_reports_locates_first_bad_worker() -> None:
    reports = [
        [
            {"local_rank": 0, "all_finite": True, "first_bad": None},
            {
                "local_rank": 1,
                "all_finite": False,
                "first_bad": {"name": "layers.2.weight_scale_inv"},
            },
        ]
    ]

    summary = integrity.summarize_replica_reports(reports)

    assert summary["all_finite"] is False
    assert summary["first_bad"]["replica_rank"] == 0
    assert summary["first_bad"]["worker_rank"] == 1
    assert (
        summary["first_bad"]["report"]["first_bad"]["name"]
        == "layers.2.weight_scale_inv"
    )


def test_configure_diagnostic_supports_resume_and_repeated_syncs(monkeypatch) -> None:
    config = SimpleNamespace(
        checkpointing=SimpleNamespace(resume=False),
        eval=SimpleNamespace(enabled=True),
        logger=SimpleNamespace(wandb_enabled=True),
        moe=SimpleNamespace(r3=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setenv("ATOMRL_WEIGHT_SYNC_RESUME", "1")
    monkeypatch.setenv("ATOMRL_WEIGHT_SYNC_REPEAT", "3")

    repeats = integrity.configure_diagnostic(config)

    assert repeats == 3
    assert config.checkpointing.resume is True
    assert config.eval.enabled is False
    assert config.logger.wandb_enabled is False
    assert config.moe.r3.enabled is False


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_configure_diagnostic_rejects_invalid_repeat_count(
    monkeypatch,
    value,
) -> None:
    config = SimpleNamespace(
        checkpointing=SimpleNamespace(resume=False),
        eval=SimpleNamespace(enabled=True),
        logger=SimpleNamespace(wandb_enabled=True),
        moe=SimpleNamespace(r3=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setenv("ATOMRL_WEIGHT_SYNC_REPEAT", value)

    with pytest.raises(ValueError, match="ATOMRL_WEIGHT_SYNC_REPEAT"):
        integrity.configure_diagnostic(config)
