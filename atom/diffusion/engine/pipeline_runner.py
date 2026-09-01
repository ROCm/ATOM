# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Per-GPU executor for diffusion pipelines.

The ``ModelRunner`` role minus everything that serves a KV cache: device setup,
component placement, running one job, and peak-memory accounting.
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

        Asserts rather than trusting ``.to()``: a platform-detection failure
        loads 144 GB to host RAM and stays silent until the first matmul.
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
                # eval() does not clear requires_grad, and aiter's varlen
                # attention asserts when it sees a grad-requiring input. At
                # Ulysses >= 2 the all-to-all writes into a fresh buffer and
                # launders the flag away, so this only ever bites at degree 1.
                module.requires_grad_(False)
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

        A failure is logged, not raised: the same work reruns on the first
        request, where the error is attributable to a job rather than killing a
        replica that just spent minutes loading. The peak-memory reset keeps
        the throwaway step out of the first job's accounting.
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

        ``on_progress(step, total)`` travels on the batch rather than as a
        stage argument: the denoise loop is several stages deep.
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
