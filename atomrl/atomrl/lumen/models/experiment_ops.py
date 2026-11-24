# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Shared experiment management helpers for Lumen training flows."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from mlperf_logging import mllog
    from mlperf_logging.mllog import constants as _mllog_constants
except ImportError:  # pragma: no cover - optional dependency
    mllog = None
    _mllog_constants = None

try:
    from pyinstrument import Profiler as _PyInstrumentProfiler
except ImportError:  # pragma: no cover - optional dependency
    _PyInstrumentProfiler = None

try:
    from rpdTracerControl import rpdTracerControl as _RpdTracerControl
except ImportError:  # pragma: no cover - optional dependency
    _RpdTracerControl = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

logger = logging.getLogger(__name__)

__all__ = [
    "ExperimentTracker",
    "MLPerfEventLogger",
    "StepTimer",
    "get_effective_stop_step",
    "resolve_quality_target",
]


class _FallbackConstants:
    ABORTED = "aborted"
    SUCCESS = "success"
    CACHE_CLEAR = "cache_clear"
    INIT_START = "init_start"
    INIT_STOP = "init_stop"
    RUN_START = "run_start"
    RUN_STOP = "run_stop"
    BLOCK_START = "block_start"
    BLOCK_STOP = "block_stop"
    EVAL_START = "eval_start"
    EVAL_STOP = "eval_stop"
    EVAL_ACCURACY = "eval_accuracy"
    EPOCH_NUM = "epoch_num"
    SAMPLES_COUNT = "samples_count"


MLLOG_CONSTANTS = _mllog_constants or _FallbackConstants


def _get_int_env(*names: str, default: int | None = None) -> int | None:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            logger.warning("Ignoring invalid integer env %s=%r", name, raw)
    return default


def _get_str_env(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw == "":
            continue
        return raw
    return default


def _env_flag(name: str) -> bool:
    raw = os.getenv(name, "")
    return raw.lower() in {"1", "true", "yes", "on"}


def resolve_quality_target(args) -> float | None:
    """Return the effective validation target for the active run."""
    val_loss_target = getattr(args, "val_loss_target", None)
    if val_loss_target is not None:
        return float(val_loss_target)

    target_log_ppl = getattr(args, "target_log_ppl", None)
    if target_log_ppl is None:
        return None
    return float(target_log_ppl)


def get_effective_stop_step(args) -> int:
    """Return the absolute stop step, including checkpoint offset."""
    ckpt_start_step = max(int(getattr(args, "ckpt_start_step", 0) or 0), 0)
    max_steps = max(int(getattr(args, "max_steps", 0) or 0), 0)
    return ckpt_start_step + max_steps


@dataclass
class StepTimer:
    """Track train step duration and derived throughput."""

    global_batch_size: int
    time_fn: Callable[[], float] = time.time
    _start_time: float | None = None
    elapsed_time_s: float = 0.0
    samples: int = 0
    last_step_time_ms: float = 0.0

    def start(self) -> None:
        self._start_time = self.time_fn()

    def stop(self) -> float:
        if self._start_time is None:
            raise RuntimeError("StepTimer.stop() called before StepTimer.start().")

        elapsed_s = self.time_fn() - self._start_time
        self._start_time = None
        self.elapsed_time_s += elapsed_s
        self.samples += self.global_batch_size
        self.last_step_time_ms = elapsed_s * 1000.0
        return self.last_step_time_ms

    def get_throughput(self) -> float:
        if self.elapsed_time_s <= 0.0:
            return 0.0

        throughput = self.samples / self.elapsed_time_s
        self.samples = 0
        self.elapsed_time_s = 0.0
        return throughput


class MLPerfEventLogger:
    """Rank-aware wrapper over ``mlperf_logging`` with a no-op fallback."""

    def __init__(
        self,
        *,
        rank: int = 0,
        filepath: str | None = None,
        default_stack_offset: int = 2,
    ) -> None:
        self.rank = rank
        self.logger = None

        if mllog is None:
            return

        config_kwargs = {"default_stack_offset": default_stack_offset}
        if filepath:
            config_kwargs["filename"] = filepath
        mllog.config(**config_kwargs)
        self.logger = mllog.get_mllogger()

    def start(self, **kwargs) -> None:
        if self.rank == 0 and self.logger is not None:
            self.logger.start(**kwargs)

    def end(self, **kwargs) -> None:
        if self.rank == 0 and self.logger is not None:
            self.logger.end(**kwargs)

    def event(self, **kwargs) -> None:
        if self.rank == 0 and self.logger is not None:
            self.logger.event(**kwargs)


class _NullProfiler:
    kind = ""

    def on_train_step_start(self, global_step: int) -> None:
        del global_step

    def on_train_end(self, global_step: int, rank: int) -> None:
        del global_step, rank


class _TorchProfilerController:
    kind = "pytorch"

    def __init__(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for PROFILER=pytorch.")

        self.output_dir = Path(
            _get_str_env("TORCHPROF_OUTPUT_DIR", default="runs/torchprof")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        devices_raw = _get_str_env("TORCHPROF_DEVICES", default="GPU")
        activities = []
        for device in {
            item.strip().lower() for item in devices_raw.split(",") if item.strip()
        }:
            if device == "cpu":
                activities.append(torch.profiler.ProfilerActivity.CPU)
            elif device == "gpu":
                activities.append(torch.profiler.ProfilerActivity.CUDA)

        if not activities:
            activities = [torch.profiler.ProfilerActivity.CUDA]

        skip_first = _get_int_env("PROF_SKIP_FIRST", default=1)
        wait = _get_int_env("PROF_WAIT", default=0)
        warmup = _get_int_env("PROF_WARMUP_STEPS", default=3)
        active = _get_int_env("PROF_ACTIVE_STEPS", default=2)
        repeat = _get_int_env("PROF_REPEAT", "PROF_REPITIONS", default=0)
        self.stop_after_step = skip_first + wait + warmup + active
        self._stopped = False

        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                skip_first=skip_first,
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.output_dir)
            ),
            profile_memory=False,
            record_shapes=False,
            with_stack=True,
        )
        self._profiler.start()

    def on_train_step_start(self, global_step: int) -> None:
        if self._stopped:
            return
        self._profiler.step()
        if global_step > self.stop_after_step:
            self._profiler.stop()
            self._stopped = True

    def on_train_end(self, global_step: int, rank: int) -> None:
        del global_step, rank
        if not self._stopped:
            self._profiler.stop()
            self._stopped = True


class _RpdProfilerController:
    kind = "rpd"

    def __init__(self) -> None:
        if _RpdTracerControl is None:
            raise RuntimeError("rpdTracerControl is required for PROFILER=rpd.")

        output_dir = Path(_get_str_env("PROF_OUTPUT_PATH", default="runs/rpd"))
        output_dir.mkdir(parents=True, exist_ok=True)
        _RpdTracerControl.setFilename(name=str(output_dir / "trace.rpd"), append=True)
        self._profiler = _RpdTracerControl()
        self._started = False
        self._finished = False
        warmup = _get_int_env("PROF_WARMUP_STEPS", default=3)
        active = _get_int_env("PROF_ACTIVE_STEPS", default=2)
        self.start_at_step = warmup
        self.stop_after_step = warmup + active

    def on_train_step_start(self, global_step: int) -> None:
        if self._finished:
            return
        if not self._started and global_step >= self.start_at_step:
            self._profiler.start()
            self._started = True
        if self._started and global_step > self.stop_after_step:
            self._profiler.stop()
            self._finished = True

    def on_train_end(self, global_step: int, rank: int) -> None:
        del global_step, rank
        if self._started and not self._finished:
            self._profiler.stop()
            self._finished = True


class _PyInstrumentProfilerController:
    kind = "pyinstrument"

    def __init__(self) -> None:
        if _PyInstrumentProfiler is None:
            raise RuntimeError("pyinstrument is required for PROFILER=pyinstrument.")

        self.output_dir = Path(
            _get_str_env("PYINSTRUMENT_OUTPUT_DIR", default="runs/pyinstrument")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._profiler = _PyInstrumentProfiler()
        self._profiler.start()

    def on_train_step_start(self, global_step: int) -> None:
        del global_step

    def on_train_end(self, global_step: int, rank: int) -> None:
        output_path = (
            self.output_dir / f"pyinstrument_{global_step}_steps_rank_{rank}.html"
        )
        self._profiler.stop()
        self._profiler.write_html(path=str(output_path), timeline=True, show_all=True)


def _build_profiler_controller(kind: str):
    normalized = kind.strip().lower()
    if normalized in {"", "none"}:
        return _NullProfiler()
    if normalized in {"torchprof", "pytorch"}:
        return _TorchProfilerController()
    if normalized == "rpd":
        return _RpdProfilerController()
    if normalized == "pyinstrument":
        return _PyInstrumentProfilerController()
    raise ValueError(f"Unsupported profiler kind: {kind}")


@dataclass
class ExperimentTracker:
    """Shared experiment lifecycle helper for Lumen training loops."""

    args: Any
    global_batch_size: int
    seq_length: int
    backend: str
    task_name: str
    rank: int = 0
    time_fn: Callable[[], float] = time.time
    event_logger: MLPerfEventLogger | None = None
    target: float | None = field(init=False)
    effective_stop_step: int = field(init=False)
    preemptive_stop_step: int | None = field(init=False)
    step_time_atol: int = field(init=False)
    train_loss_log_freq: int = field(init=False)
    status: str = field(init=False)
    success: bool = field(init=False)
    profiler_kind: str = field(init=False)

    def __post_init__(self) -> None:
        self.target = resolve_quality_target(self.args)
        self.effective_stop_step = get_effective_stop_step(self.args)
        self.preemptive_stop_step = _get_int_env(
            "LUMEN_PREEMPTIVE_STOP_STEP",
            "STOP_ON_STEP",
            default=getattr(self.args, "preemptive_stop_step", None),
        )
        self.step_time_atol = int(getattr(self.args, "step_time_atol", 0) or 0)
        self.train_loss_log_freq = max(
            _get_int_env("TRAINING_LOSS_LOG_FREQ", "LOG_FREQ", default=1000000),
            1,
        )
        self.status = MLLOG_CONSTANTS.ABORTED
        self.success = _env_flag("FORCE_SUCCESS_STATUS")
        self.timer = StepTimer(
            global_batch_size=self.global_batch_size, time_fn=self.time_fn
        )
        self._run_started_at = None

        filepath = _get_str_env("LUMEN_MLPERF_LOG_PATH", default="")
        if self.event_logger is None:
            self.event_logger = MLPerfEventLogger(
                rank=self.rank, filepath=filepath or None
            )

        profiler_setting = _get_str_env(
            "PROFILER", default=getattr(self.args, "profiler", "")
        )
        self.profiler_kind = profiler_setting.strip().lower()
        try:
            self.profiler = _build_profiler_controller(self.profiler_kind)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Experiment profiler disabled: %s", exc)
            self.profiler = _NullProfiler()
            self.profiler_kind = ""

    def _consumed_samples(self, global_step: int) -> int:
        return int(global_step * self.global_batch_size)

    def on_train_start(self, configs: Mapping[str, Any] | None = None) -> None:
        self.event_logger.event(key=MLLOG_CONSTANTS.CACHE_CLEAR, value=True)
        self.event_logger.start(key=MLLOG_CONSTANTS.INIT_START)
        for key, value in (configs or {}).items():
            self.event_logger.event(key=key, value=value)
        self.event_logger.end(key=MLLOG_CONSTANTS.INIT_STOP)
        self.event_logger.start(key=MLLOG_CONSTANTS.RUN_START)
        self.event_logger.start(
            key=MLLOG_CONSTANTS.BLOCK_START,
            metadata={
                MLLOG_CONSTANTS.SAMPLES_COUNT: self._consumed_samples(
                    int(getattr(self.args, "ckpt_start_step", 0))
                )
            },
        )
        self._run_started_at = self.time_fn()

    def record_train_step_start(self, global_step: int | None = None) -> None:
        self.timer.start()
        self.profiler.on_train_step_start(int(global_step or 0))

    def record_train_step_end(
        self,
        *,
        global_step: int,
        train_loss: float | None = None,
        lr: float | None = None,
    ) -> float:
        step_time_ms = self.timer.stop()
        if self.step_time_atol > 0 and step_time_ms > self.step_time_atol:
            raise AssertionError(
                f"Logged train step time ({step_time_ms:.1f} ms) is slower than "
                f"tolerable ({self.step_time_atol} ms)."
            )

        if train_loss is not None and global_step % self.train_loss_log_freq == 0:
            metadata = {
                MLLOG_CONSTANTS.SAMPLES_COUNT: self._consumed_samples(global_step)
            }
            if lr is not None:
                metadata["lr"] = lr
            self.event_logger.event(
                key="train_loss", value=float(train_loss), metadata=metadata
            )

        return step_time_ms

    def log_validation_start(self, global_step: int) -> None:
        throughput = self.timer.get_throughput()
        self.event_logger.event(
            key="tracked_stats",
            metadata={"step": self._consumed_samples(global_step)},
            value={"throughput": throughput},
        )
        metadata = {MLLOG_CONSTANTS.SAMPLES_COUNT: self._consumed_samples(global_step)}
        self.event_logger.end(key=MLLOG_CONSTANTS.BLOCK_STOP, metadata=metadata)
        self.event_logger.start(key=MLLOG_CONSTANTS.EVAL_START, metadata=metadata)

    def should_stop_on_validation(self, *, global_step: int, val_loss: float) -> bool:
        consumed_samples = self._consumed_samples(global_step)
        metadata = {
            MLLOG_CONSTANTS.EPOCH_NUM: consumed_samples,
            MLLOG_CONSTANTS.SAMPLES_COUNT: consumed_samples,
        }
        self.event_logger.event(
            key=MLLOG_CONSTANTS.EVAL_ACCURACY,
            value=float(val_loss),
            metadata=metadata,
        )
        self.event_logger.end(
            key=MLLOG_CONSTANTS.EVAL_STOP,
            metadata={MLLOG_CONSTANTS.SAMPLES_COUNT: consumed_samples},
        )

        should_stop = self.success
        if self.target is not None and val_loss <= self.target:
            should_stop = True

        if should_stop:
            self.success = True
            self.status = MLLOG_CONSTANTS.SUCCESS
            return True

        self.event_logger.start(
            key=MLLOG_CONSTANTS.BLOCK_START,
            metadata={MLLOG_CONSTANTS.SAMPLES_COUNT: consumed_samples},
        )
        return False

    def should_preempt(self, *, global_step: int) -> bool:
        return (
            self.preemptive_stop_step is not None
            and global_step >= self.preemptive_stop_step
        )

    def finish_run(self, *, global_step: int) -> None:
        self.profiler.on_train_end(global_step, self.rank)
        duration_s = 0.0
        if self._run_started_at is not None:
            duration_s = self.time_fn() - self._run_started_at

        if global_step >= self.effective_stop_step:
            self.success = True

        status = MLLOG_CONSTANTS.SUCCESS if self.success else MLLOG_CONSTANTS.ABORTED
        self.status = status
        self.event_logger.end(
            key=MLLOG_CONSTANTS.RUN_STOP,
            metadata={
                MLLOG_CONSTANTS.SAMPLES_COUNT: self._consumed_samples(global_step),
                "status": status,
                "duration": f"{duration_s} sec -> {duration_s / 60.0} minutes",
            },
        )
        self.event_logger.event(
            key="train_samples",
            value=self._consumed_samples(global_step),
        )
