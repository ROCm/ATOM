# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""User-facing diffusion engine: submit a job, poll it, collect the file.

``LLMEngine``'s role, with an API shape that follows the workload: a generation
takes minutes, so there is nothing to stream and the engine hands back a job id
immediately. One job in flight per replica -- the resident DiT plus activations
own the GPU -- and admission *rejects* past the queue cap rather than queueing.
"""

import logging
import queue
import threading
import time
from typing import Self

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.engine.core_manager import DiffusionCoreManager
from atom.diffusion.engine.job_scheduler import AdmissionError, JobScheduler
from atom.diffusion.engine.protocol import EngineRequest, OutputType, RequestType
from atom.diffusion.request import DiffusionJob, JobStatus

logger = logging.getLogger(__name__)

__all__ = ["AdmissionError", "DiffusionEngine"]


class DiffusionEngine:
    """Owns the scheduler, the workers, and the job table."""

    def __init__(
        self,
        config: DiffusionConfig,
        *,
        manager: DiffusionCoreManager | None = None,
    ) -> None:
        self.config = config
        self.scheduler = JobScheduler(config)
        self.manager = manager or DiffusionCoreManager(config)
        self._jobs: dict[str, DiffusionJob] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name="diffusion-dispatch", daemon=True
        )
        self._started = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self.manager.start()
        self._dispatch_thread.start()
        self._started = True
        logger.info(
            "diffusion engine ready: %s on %d GPU(s), ulysses=%d",
            self.config.pipeline_class,
            self.config.num_gpus,
            self.config.ulysses_degree,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.manager.close()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------

    def submit(self, job: DiffusionJob) -> DiffusionJob:
        """Admit a job and return it immediately with a queued status.

        Raises :class:`AdmissionError` when the queue is full -- backpressure a
        caller can act on, rather than a wait it cannot see.
        """
        with self._lock:
            self.scheduler.add_job(job)
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> DiffusionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def abort(self, job_id: str) -> bool:
        """Ask for a job to be dropped.

        A job that has already started keeps running to completion: a denoise
        step cannot be interrupted mid-kernel, and pulling one rank out of a
        collective hangs the rest. So this is honoured promptly for queued jobs
        and at the next boundary for running ones.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.is_finished:
                return False
            dropped = self.scheduler.abort(job_id)
        self.manager.send(EngineRequest(type=RequestType.ABORT, job_id=job_id))
        return dropped or job.status is JobStatus.RUNNING

    def wait(self, job_id: str, *, timeout: float | None = None) -> DiffusionJob:
        """Block until a job reaches a terminal state. Convenience for offline use."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job is None:
                raise KeyError(f"unknown job {job_id!r}")
            if job.is_finished:
                return job
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
            time.sleep(0.2)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return self.scheduler.stats()

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        """Feed the workers one job at a time and apply their reports.

        Single-threaded on purpose: the replica is a single collective group,
        so "next job" is a strictly serial decision and a second dispatcher
        could only ever race with this one.
        """
        while not self._closed:
            self._dispatch_next()
            self._drain_outputs(timeout=0.2)

    def _dispatch_next(self) -> None:
        with self._lock:
            job = self.scheduler.schedule()
        if job is None:
            return
        logger.info("dispatching job %s (task=%s)", job.job_id, job.task or "-")
        self.manager.send(EngineRequest(type=RequestType.ADD, job=job))

    def _drain_outputs(self, *, timeout: float) -> None:
        try:
            output = self.manager.outputs.get(timeout=timeout)
        except queue.Empty:
            return
        self._apply(output)
        # Take whatever else is already queued without waiting again: progress
        # arrives in bursts and the loop should not fall a step behind per tick.
        while True:
            try:
                self._apply(self.manager.outputs.get_nowait())
            except queue.Empty:
                return

    def _apply(self, output) -> None:
        if output.type is OutputType.READY:
            return
        if output.type is OutputType.DEAD:
            logger.error("rank %d died: %s", output.rank, output.error)
            self._fail_in_flight(f"rank {output.rank} died: {output.error}")
            return

        with self._lock:
            job = self._jobs.get(output.job_id)
            if job is None:
                logger.warning("output for unknown job %s", output.job_id)
                return

            if output.type is OutputType.PROGRESS:
                job.status = JobStatus.RUNNING
                job.current_step = output.current_step
                job.total_steps = output.total_steps or job.total_steps
                return

            if output.type is OutputType.ERROR:
                # Every rank reports a failure, so keep the first: it is the
                # one most likely to be the cause rather than a collateral
                # collective timeout on a peer.
                if not job.is_finished:
                    logger.error(
                        "job %s failed on rank %d: %s",
                        job.job_id,
                        output.rank,
                        output.error,
                    )
                    self.scheduler.complete(job, error=output.error or "unknown error")
                return

            job.status = output.status or JobStatus.COMPLETED
            job.current_step = output.current_step
            job.peak_memory_mb = output.peak_memory_mb
            self.scheduler.complete(job, output_path=output.output_path)
            logger.info(
                "job %s %s in %.1fs -> %s",
                job.job_id,
                job.status.name.lower(),
                output.elapsed_s or 0.0,
                job.output_path,
            )

    def _fail_in_flight(self, reason: str) -> None:
        with self._lock:
            for job in self._jobs.values():
                if not job.is_finished:
                    self.scheduler.complete(job, error=reason)
