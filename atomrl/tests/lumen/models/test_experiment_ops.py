# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for shared experiment ops helpers."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT.parent / "atomrl" / "lumen" / "models"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_experiment_ops():
    lumen_pkg = sys.modules.setdefault("atomrl.lumen", types.ModuleType("atomrl.lumen"))
    lumen_pkg.__path__ = [str(REPO_ROOT.parent / "atomrl" / "lumen")]

    models_pkg = sys.modules.setdefault(
        "atomrl.lumen.models", types.ModuleType("atomrl.lumen.models")
    )
    models_pkg.__path__ = [str(MODELS_DIR)]

    return _load_module(
        "atomrl.lumen.models.experiment_ops", MODELS_DIR / "experiment_ops.py"
    )


class _MemoryLogger:
    def __init__(self):
        self.calls = []

    def start(self, **kwargs):
        self.calls.append(("start", kwargs))

    def end(self, **kwargs):
        self.calls.append(("end", kwargs))

    def event(self, **kwargs):
        self.calls.append(("event", kwargs))


class TestExperimentOps:
    def test_resolve_quality_target_prefers_val_loss_target(self):
        mod = _load_experiment_ops()
        args = SimpleNamespace(val_loss_target=0.9, target_log_ppl=3.3)
        assert mod.resolve_quality_target(args) == 0.9

    def test_resolve_quality_target_falls_back_to_target_log_ppl(self):
        mod = _load_experiment_ops()
        args = SimpleNamespace(val_loss_target=None, target_log_ppl=3.3)
        assert mod.resolve_quality_target(args) == 3.3

    def test_get_effective_stop_step_adds_checkpoint_offset(self):
        mod = _load_experiment_ops()
        args = SimpleNamespace(ckpt_start_step=128, max_steps=512)
        assert mod.get_effective_stop_step(args) == 640

    def test_tracker_marks_success_when_target_reached(self):
        mod = _load_experiment_ops()
        args = SimpleNamespace(
            val_loss_target=None,
            target_log_ppl=3.3,
            step_time_atol=18000,
            tag="",
        )
        tracker = mod.ExperimentTracker(
            args=args,
            global_batch_size=8,
            seq_length=4096,
            backend="fsdp",
            task_name="test",
            rank=0,
        )
        assert tracker.should_stop_on_validation(global_step=16, val_loss=3.0) is True
        assert tracker.success is True

    def test_step_timer_reports_elapsed_ms_and_throughput(self):
        mod = _load_experiment_ops()
        timestamps = iter([10.0, 10.5, 20.0, 20.5])
        timer = mod.StepTimer(global_batch_size=8, time_fn=lambda: next(timestamps))

        timer.start()
        assert timer.stop() == 500.0
        timer.start()
        assert timer.stop() == 500.0
        assert timer.get_throughput() == 16.0

    def test_tracker_enforces_step_time_budget(self):
        mod = _load_experiment_ops()
        timestamps = iter([0.0, 20.0])
        args = SimpleNamespace(
            val_loss_target=None,
            target_log_ppl=3.3,
            step_time_atol=18000,
            tag="",
        )
        tracker = mod.ExperimentTracker(
            args=args,
            global_batch_size=8,
            seq_length=4096,
            backend="fsdp",
            task_name="test",
            rank=0,
            time_fn=lambda: next(timestamps),
        )

        tracker.record_train_step_start()
        with pytest.raises(AssertionError):
            tracker.record_train_step_end(global_step=1, train_loss=1.0, lr=1e-4)

    def test_tracker_honors_preemptive_stop_env(self, monkeypatch):
        mod = _load_experiment_ops()
        monkeypatch.setenv("LUMEN_PREEMPTIVE_STOP_STEP", "32")
        args = SimpleNamespace(
            val_loss_target=None,
            target_log_ppl=3.3,
            step_time_atol=18000,
            tag="",
        )
        tracker = mod.ExperimentTracker(
            args=args,
            global_batch_size=8,
            seq_length=4096,
            backend="fsdp",
            task_name="test",
            rank=0,
        )

        assert tracker.should_preempt(global_step=31) is False
        assert tracker.should_preempt(global_step=32) is True

    def test_tracker_marks_full_length_run_success(self):
        mod = _load_experiment_ops()
        args = SimpleNamespace(
            val_loss_target=None,
            target_log_ppl=3.3,
            step_time_atol=18000,
            tag="",
            ckpt_start_step=128,
            max_steps=512,
        )
        tracker = mod.ExperimentTracker(
            args=args,
            global_batch_size=8,
            seq_length=4096,
            backend="fsdp",
            task_name="test",
            rank=0,
            event_logger=_MemoryLogger(),
        )

        tracker.on_train_start()
        tracker.finish_run(global_step=640)
        assert tracker.success is True
        assert tracker.status == "success"

    def test_invalid_preemptive_stop_env_falls_back_to_default(self, monkeypatch):
        mod = _load_experiment_ops()
        monkeypatch.setenv("LUMEN_PREEMPTIVE_STOP_STEP", "invalid")
        args = SimpleNamespace(
            val_loss_target=None,
            target_log_ppl=3.3,
            step_time_atol=18000,
            tag="",
        )
        tracker = mod.ExperimentTracker(
            args=args,
            global_batch_size=8,
            seq_length=4096,
            backend="fsdp",
            task_name="test",
            rank=0,
        )

        assert tracker.preemptive_stop_step is None
