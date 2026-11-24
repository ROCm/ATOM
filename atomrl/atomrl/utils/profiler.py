# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lightweight distributed profiler dispatcher for AtomRL.

This module mirrors the most-used VERL profiler semantics:
- rank filtering (`all_ranks` / `ranks`)
- step-gated activation
- explicit `start(profile_step=...)` / `stop()` control

Current backend support:
- `torch`: writes chrome traces via ``torch.profiler``.
- `rocprof`: emits step ranges (via optional roctx) and generates launch
  scripts/flags for external `rocprof` wrapping.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import Any

import torch

from atomrl.core.config import (
    ProfilerConfig,
    RocprofToolConfig,
    TorchProfilerToolConfig,
)

logger = logging.getLogger(__name__)


class _NoOpProfiler:
    def start(self, **kwargs: Any) -> None:
        return

    def step(self) -> None:
        return

    def stop(self) -> None:
        return


class _TorchStepProfiler:
    """Torch profiler backend with optional ``torch.profiler.schedule`` support.

    The profiler context is created once in ``start()`` and destroyed in
    ``stop()``.  Callers drive the schedule by calling ``step()`` once per
    mini-batch.  Traces are exported automatically via ``on_trace_ready``.
    """

    def __init__(self, rank: int, config: ProfilerConfig) -> None:
        self.rank = rank
        self.config = config
        self._ctx: torch.profiler.profile | None = None
        self._step_tag: int | None = None

    def _resolve_schedule(self) -> dict[str, int] | None:
        tool_cfg = self.config.tool_config
        if not isinstance(tool_cfg, TorchProfilerToolConfig):
            return None
        sched = getattr(tool_cfg, "schedule", None)
        if sched is None or getattr(sched, "active", 0) <= 0:
            return None
        return {
            "skip_first": sched.skip_first,
            "wait": sched.wait,
            "warmup": sched.warmup,
            "active": sched.active,
            "repeat": sched.repeat,
        }

    def _make_trace_handler(self) -> Any:
        out_dir = Path(self.config.save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        state: dict[str, int] = {"count": 0}
        rank = self.rank

        def handler(prof: torch.profiler.profile) -> None:
            idx = state["count"]
            state["count"] += 1
            suffix = f"_cycle{idx}" if idx > 0 else ""
            tag = f"step{self._step_tag}" if self._step_tag is not None else "trace"
            path = out_dir / f"{tag}_rank{rank}{suffix}_torch.json"
            try:
                prof.export_chrome_trace(str(path))
                logger.info("Profiler trace exported: %s", path)
            except (OSError, RuntimeError) as exc:
                logger.warning("Failed to export profiler trace to %s: %s", path, exc)

        return handler

    def start(self, **kwargs: Any) -> None:
        if self._ctx is not None:
            return
        self._step_tag = kwargs.get("profile_step")
        tool_cfg = self.config.tool_config
        if not isinstance(tool_cfg, TorchProfilerToolConfig):
            tool_cfg = TorchProfilerToolConfig()
        contents = set(tool_cfg.contents)
        activities: list[torch.profiler.ProfilerActivity] = []
        if "cpu" in contents:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "cuda" in contents and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if not activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)

        profile_kwargs: dict[str, Any] = {
            "activities": activities,
            "record_shapes": "shapes" in contents,
            "profile_memory": "memory" in contents,
            "with_stack": "stack" in contents,
            "on_trace_ready": self._make_trace_handler(),
        }
        sched = self._resolve_schedule()
        if sched:
            profile_kwargs["schedule"] = torch.profiler.schedule(**sched)

        self._ctx = torch.profiler.profile(**profile_kwargs)
        self._ctx.start()
        logger.info("Profiler started for rank %d", self.rank)

    def step(self) -> None:
        if self._ctx is not None:
            self._ctx.step()

    def stop(self) -> None:
        if self._ctx is None:
            return
        self._ctx.stop()
        logger.info("Profiler stopped for rank %d", self.rank)
        self._ctx = None
        self._step_tag = None


class _RocprofParamProfiler:
    """Generate rocprof launch artifacts and optional roctx step markers.

    `rocprof` is typically launched outside the Python process:
      rocprof [flags] python -m atomrl.trainer.main ...

    This profiler helps by:
    1) mapping config -> rocprof CLI flags
    2) writing per-rank launch helper scripts under `save_path`
    3) optionally pushing roctx ranges at profiled steps (if `roctx` is installed)
    """

    def __init__(self, rank: int, config: ProfilerConfig) -> None:
        self.rank = rank
        self.config = config
        tool_cfg = config.tool_config
        if isinstance(tool_cfg, RocprofToolConfig):
            self.tool_config = tool_cfg
        else:
            self.tool_config = RocprofToolConfig()
        self._range_started = False
        self._step: int | None = None
        self._roctx = self._try_import_roctx()
        self._wrote_hints = False

    @staticmethod
    def _try_import_roctx() -> Any | None:
        try:
            import roctx  # type: ignore[import-not-found]

            return roctx
        except (ImportError, OSError):
            return None

    def _build_rocprof_flags(self, rank: int) -> list[str]:
        cfg = self.tool_config
        flags: list[str] = []
        if cfg.hip_trace:
            flags.append("--hip-trace")
        if cfg.hsa_trace:
            flags.append("--hsa-trace")
        if cfg.kernel_trace:
            flags.append("--kernel-trace")
        if cfg.memory_copy_trace:
            flags.append("--memory-copy-trace")
        if cfg.sys_trace:
            flags.append("--sys-trace")
        if cfg.timestamp_on:
            flags.append("--timestamp")
        if cfg.stats:
            flags.append("--stats")
        if cfg.output_format:
            flags.extend(["--format", cfg.output_format])
        if cfg.kernel_regex:
            flags.extend(["--kernel-regex", cfg.kernel_regex])
        if cfg.output_file:
            out_dir = Path(self.config.save_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{cfg.output_file}_rank{rank}"
            flags.extend(["--output", str(out_file)])
        if cfg.extra_args:
            flags.extend(cfg.extra_args)
        return flags

    def _write_launch_hints(self) -> None:
        if self._wrote_hints:
            return
        out_dir = Path(self.config.save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        flags = self._build_rocprof_flags(self.rank)
        payload = {
            "tool": "rocprof",
            "rank": self.rank,
            "flags": flags,
            "note": "Launch profiler externally: rocprof <flags> python -m atomrl.trainer.main ...",
        }
        (out_dir / f"rocprof_rank{self.rank}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        cmd = (
            "rocprof "
            + " ".join(shlex.quote(f) for f in flags)
            + " python -m atomrl.trainer.main --config <config>"
        )
        (out_dir / f"rocprof_rank{self.rank}.sh").write_text(
            cmd + "\n", encoding="utf-8"
        )
        logger.info("rocprof launch hints written under %s", out_dir)
        self._wrote_hints = True

    def step(self) -> None:
        return

    def start(self, **kwargs: Any) -> None:
        self._write_launch_hints()
        profile_step = kwargs.get("profile_step")
        self._step = int(profile_step) if profile_step is not None else None
        if self._roctx is not None:
            tag = (
                f"atomrl_step_{self._step}" if self._step is not None else "atomrl_step"
            )
            try:
                self._roctx.range_push(tag)
                self._range_started = True
            except RuntimeError:
                self._range_started = False

    def stop(self) -> None:
        if self._roctx is not None and self._range_started:
            try:
                self._roctx.range_pop()
            except (
                RuntimeError
            ):  # ROCTX may reject a range pop after profiler teardown.
                pass
        self._range_started = False
        self._step = None


class DistProfiler:
    """Dispatcher that gates profiling by rank and tool."""

    def __init__(self, rank: int, config: ProfilerConfig | None = None) -> None:
        self.rank = rank
        self.config = config or ProfilerConfig(enable=False)
        self._enable = bool(self.config.enable)
        self._this_step = False
        self._this_rank = self._resolve_rank_gate()

        tool = (self.config.tool or "").lower()
        if tool == "torch":
            self._impl = _TorchStepProfiler(rank=rank, config=self.config)
        elif tool == "rocprof":
            self._impl = _RocprofParamProfiler(rank=rank, config=self.config)
        else:
            self._impl = _NoOpProfiler()
            if self._enable:
                logger.warning(
                    "Unsupported profiler tool '%s'; using no-op profiler.", tool
                )

    def _resolve_rank_gate(self) -> bool:
        if self.config.all_ranks:
            return True
        if self.config.ranks:
            return self.rank in self.config.ranks
        return self.rank == 0

    def check_enable(self) -> bool:
        return self._enable

    def check_this_rank(self) -> bool:
        return self._this_rank

    def check_this_step(self) -> bool:
        return self._this_step

    def start(self, **kwargs: Any) -> None:
        if not (self.check_enable() and self.check_this_rank()):
            return
        self._this_step = True
        self._impl.start(**kwargs)

    def step(self) -> None:
        """Advance the profiler schedule by one step (per mini-batch)."""
        if not (self.check_enable() and self.check_this_rank()):
            return
        self._impl.step()

    def stop(self) -> None:
        if not (self.check_enable() and self.check_this_rank()):
            return
        self._this_step = False
        self._impl.stop()
