# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Per-rank GPU worker for diffusion serving.

``atom.model_engine.engine_core.EngineCore``'s role minus the scheduling loop:
a replica runs one job at a time, so this is a receive-run-reply loop and
admission lives in the API process where the queue is observable.

Ulysses is collective, so every rank enters every job in the same order --
aborts are honoured only *between* jobs and the loop never skips a received ADD.
Only rank 0 has a result; the others report READY and errors and stay silent.
"""

import logging
import os
import pickle
import signal
import traceback
from typing import Any

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.engine.pipeline_runner import PipelineRunner
from atom.diffusion.engine.protocol import (
    EngineOutput,
    EngineRequest,
    OutputType,
    RequestType,
)
from atom.diffusion.registry import resolve_pipeline_class
from atom.diffusion.request import DiffusionJob, JobStatus
from atom.diffusion.ulysses import UlyssesGroup

__all__ = ["DiffusionEngineCore", "resolve_pipeline_class"]

logger = logging.getLogger(__name__)

# Emit at most one progress message per this many denoise steps. At ~10 s/step
# a message per step is fine for the socket but noisy in logs; the API's
# progress fraction does not get more useful than this.
PROGRESS_EVERY = 1


class DiffusionEngineCore:
    """One rank's worker: owns a pipeline, runs jobs, reports outcomes."""

    def __init__(
        self,
        config: DiffusionConfig,
        rank: int,
        *,
        pipeline: Any | None = None,
        runner: PipelineRunner | None = None,
        ulysses: UlyssesGroup | None = None,
    ) -> None:
        self.config = config
        self.rank = rank
        self.ulysses = ulysses or (pipeline.ulysses if pipeline is not None else None)
        self.pipeline = pipeline
        self.runner = runner
        self._aborted: set[str] = set()
        self._shutdown = False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct the pipeline, place components, and warm up."""
        import torch

        device = None
        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
            device = torch.device(f"cuda:{self.rank}")

        if self.config.num_gpus > 1:
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl" if device is not None else "gloo",
                    rank=self.rank,
                    world_size=self.config.num_gpus,
                )
        self.ulysses = self.ulysses or UlyssesGroup()

        pipeline_cls = resolve_pipeline_class(self.config.pipeline_class)
        self.pipeline = pipeline_cls(
            self.config, self.ulysses, model_root=self.config.model_path
        )
        self.pipeline.load_components()
        self.runner = PipelineRunner(
            self.config, self.pipeline, self.ulysses, device=device
        )
        self.runner.place_components()
        self.runner.warmup()

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def handle(self, request: EngineRequest, emit) -> None:
        """Process one request; ``emit`` publishes an :class:`EngineOutput`."""
        if request.type is RequestType.SHUTDOWN:
            self._shutdown = True
            return
        if request.type is RequestType.ABORT:
            # Honoured only between jobs: a denoise step cannot be interrupted
            # mid-kernel, and tearing one rank out of an all-to-all would hang
            # the rest of the replica.
            self._aborted.add(request.job_id)
            return

        job = request.job
        if job.job_id in self._aborted:
            self._aborted.discard(job.job_id)
            job.status = JobStatus.ABORTED
            if self.is_main:
                emit(EngineOutput.from_job(job, OutputType.RESULT, rank=self.rank))
            return

        self.run_job(job, emit)

    @property
    def is_main(self) -> bool:
        return self.ulysses is None or self.ulysses.is_main

    def run_job(self, job: DiffusionJob, emit) -> None:
        import time

        job.status = JobStatus.RUNNING
        job.start_time = time.monotonic()

        def on_progress(step: int, total: int) -> None:
            if self.is_main and (step % PROGRESS_EVERY == 0 or step == total):
                emit(
                    EngineOutput(
                        type=OutputType.PROGRESS,
                        rank=self.rank,
                        job_id=job.job_id,
                        status=JobStatus.RUNNING,
                        current_step=step,
                        total_steps=total,
                    )
                )

        try:
            self.runner.run_job(job, on_progress=on_progress)
        except Exception as exc:
            job.finish_time = time.monotonic()
            job.mark_failed(f"{type(exc).__name__}: {exc}")
            logger.exception("job %s failed on rank %d", job.job_id, self.rank)
            # Every rank reports a failure. A hang in an all-to-all shows up as
            # one rank erroring and the others silent, and that asymmetry is
            # the diagnostic.
            emit(
                EngineOutput.from_job(
                    job,
                    OutputType.ERROR,
                    rank=self.rank,
                    extra={"traceback": traceback.format_exc()},
                )
            )
            return

        job.finish_time = time.monotonic()
        job.status = JobStatus.COMPLETED
        if self.is_main:
            emit(EngineOutput.from_job(job, OutputType.RESULT, rank=self.rank))

    # ------------------------------------------------------------------
    # process entry point
    # ------------------------------------------------------------------

    @staticmethod
    def run_worker(
        config: DiffusionConfig,
        rank: int,
        input_address: str,
        output_address: str,
    ) -> None:
        """Worker process body: build, announce READY, then serve until told to stop."""
        import zmq

        from atom.utils import make_zmq_socket

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")

        ctx = zmq.Context(io_threads=1)
        # PULL binds on the manager side, so the worker connects; linger keeps
        # a final ERROR from being dropped when the process exits right after.
        out_socket = make_zmq_socket(ctx, output_address, zmq.PUSH, linger=4000)
        in_socket = make_zmq_socket(ctx, input_address, zmq.PULL, bind=False)

        def emit(output: EngineOutput) -> None:
            out_socket.send(pickle.dumps(output))

        core = DiffusionEngineCore(config, rank)
        try:
            core.build()
        except Exception as exc:
            emit(
                EngineOutput(
                    type=OutputType.DEAD,
                    rank=rank,
                    error=f"{type(exc).__name__}: {exc}",
                    extra={"traceback": traceback.format_exc()},
                )
            )
            raise

        emit(EngineOutput(type=OutputType.READY, rank=rank))
        logger.info("diffusion worker rank %d ready", rank)

        try:
            while not core._shutdown:
                request: EngineRequest = pickle.loads(in_socket.recv())
                core.handle(request, emit)
        except Exception as exc:
            emit(
                EngineOutput(
                    type=OutputType.DEAD,
                    rank=rank,
                    error=f"{type(exc).__name__}: {exc}",
                    extra={"traceback": traceback.format_exc()},
                )
            )
            raise
        finally:
            in_socket.close(linger=0)
            out_socket.close()
            ctx.term()
            logger.info("diffusion worker rank %d exiting", rank)
