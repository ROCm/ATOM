# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Per-GPU executor for diffusion pipelines.

Mirrors the role of ``atom.model_engine.model_runner.ModelRunner``, minus
everything that exists to serve a KV cache. What it owns: device/process-group
setup, component placement, running one job through the pipeline, and peak
memory accounting.
"""

import logging
import time
from typing import TYPE_CHECKING

from atom.diffusion.config import DiffusionConfig, PerformanceMode
from atom.diffusion.pipeline import DiffusionBatch
from atom.diffusion.request import DiffusionJob
from atom.diffusion.ulysses import UlyssesGroup

if TYPE_CHECKING:  # pragma: no cover - typing only
    from atom.diffusion.pipeline import ComposedPipeline

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs one diffusion pipeline on one GPU (one rank of a replica)."""

    def __init__(
        self,
        config: DiffusionConfig,
        pipeline: "ComposedPipeline",
        ulysses: UlyssesGroup | None = None,
        device: "object | None" = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.ulysses = ulysses or pipeline.ulysses
        self.device = device
        self._steps_done = 0

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------

    def place_components(self) -> None:
        """Move every component to the device, and verify it landed.

        Asserts rather than trusts ``.to()``: this is the step whose absence
        wrecked the sglang baseline on ROCm, where platform detection failed,
        the runtime initialised with ``device=cpu`` and the model loaded to host
        RAM with no error until the first matmul.
        """
        if self.config.performance_mode is not PerformanceMode.SPEED:
            raise NotImplementedError(
                f"only PerformanceMode.SPEED is implemented, got "
                f"{self.config.performance_mode}"
            )
        if self.device is None:
            return

        import torch

        staged = set(getattr(self.pipeline, "host_staged_components", ()))
        for name, module in self.pipeline.components.items():
            if name in staged:
                logger.info("leaving component %s on the host (staged per use)", name)
                continue
            if isinstance(module, torch.nn.Module):
                module.to(self.device).eval()
                bad = [
                    pname
                    for pname, p in module.named_parameters()
                    if p.device.type != torch.device(self.device).type
                ]
                if bad:
                    raise RuntimeError(
                        f"component {name!r} left {len(bad)} parameter(s) off "
                        f"{self.device} (first: {bad[0]}); refusing to run"
                    )
                logger.info("placed component %s on %s", name, self.device)

    def warmup(self) -> bool:
        """Run the pipeline's warmup, if it has one and config allows it.

        A failure here is logged, not raised: the same work runs again on the
        first real request, which is where the error belongs -- attributed to a
        job, reported to its caller, rather than killing a replica that has
        just spent minutes loading. The peak-memory reset afterwards keeps the
        throwaway step out of the first job's accounting.
        """
        if not self.config.warmup or self.device is None:
            return False

        t0 = time.perf_counter()
        try:
            warmed = self.pipeline.warmup(self.device)
        except Exception as exc:
            logger.warning(
                "warmup failed on rank %d (%s); the first request will pay "
                "the first-forward cost instead",
                self.ulysses.rank,
                exc,
                exc_info=True,
            )
            return False

        if warmed and self.ulysses.is_main:
            logger.info("warmup took %.1fs", time.perf_counter() - t0)
        self._reset_peak_memory()
        return warmed

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def _reset_peak_memory(self) -> None:
        # Telemetry must never fail a run, but swallowing silently hides a
        # broken accounting path, so log at debug.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            logger.debug("could not reset peak memory stats: %s", exc)

    def _peak_memory_mb(self) -> float | None:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            logger.debug("could not read peak memory: %s", exc)
        return None

    def run_job(
        self,
        job: DiffusionJob,
        *,
        is_warmup: bool = False,
        on_progress: "object | None" = None,
    ) -> DiffusionBatch:
        """Run one job end to end through the pipeline.

        ``on_progress(step, total)`` is called from inside the denoise loop.
        It travels on the batch rather than as a stage argument because the
        loop is several stages deep and only one stage can report meaningfully.
        """
        self._reset_peak_memory()
        batch = DiffusionBatch(job=job, is_warmup=is_warmup)
        batch.meta["ulysses_world"] = self.ulysses.world_size
        batch.meta["ulysses_rank"] = self.ulysses.rank
        batch.meta["device"] = str(self.device) if self.device is not None else "cpu"
        if on_progress is not None:
            batch.meta["on_progress"] = on_progress

        t0 = time.perf_counter()
        batch = self.pipeline.forward(batch)
        elapsed = time.perf_counter() - t0

        job.peak_memory_mb = self._peak_memory_mb()
        batch.meta["elapsed_s"] = elapsed

        if self.ulysses.is_main and not is_warmup:
            logger.info(
                "job %s finished on rank %d in %.3fs (peak %.0f MB)",
                job.job_id,
                self.ulysses.rank,
                elapsed,
                job.peak_memory_mb or 0.0,
            )
            logger.debug("%s", self.pipeline.stage_timing_report())
        return batch
