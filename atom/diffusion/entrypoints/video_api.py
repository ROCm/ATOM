# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""OpenAI-shaped ``/v1/videos`` routes over a :class:`DiffusionEngine`.

Asynchronous by contract, because the work is: submitting returns a job id in
milliseconds while the generation runs for minutes. Three routes carry it --
create, poll, fetch content -- plus an abort.

The route handlers stay synchronous-in-spirit (the engine's methods are quick,
lock-guarded dict operations) and never touch the GPU; everything expensive
happens in the worker processes behind ZMQ.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from atom.diffusion.engine.diffusion_engine import AdmissionError, DiffusionEngine
from atom.diffusion.request import DiffusionJob, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter()


class VideoCreateRequest(BaseModel):
    """A generation request.

    ``task`` is required and has no default. Its absence from sglang's offline
    CLI is exactly what makes that CLI unable to drive MiniMax-H3, so it is a
    first-class field here rather than something inferred from the conditions.
    """

    prompt: str
    task: str
    model: str | None = None
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    target: dict[str, Any] = Field(default_factory=dict)
    num_inference_steps: int = 50
    seed: int | None = None
    seconds: float | None = None


class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    status: str
    progress: float
    task: str
    error: str | None = None
    output_path: str | None = None
    inference_time_s: float | None = None
    peak_memory_mb: float | None = None


def job_to_response(job: DiffusionJob) -> VideoResponse:
    payload = job.to_dict()
    return VideoResponse(
        id=payload["id"],
        status=payload["status"],
        progress=payload["progress"],
        task=payload["task"],
        error=payload["error"],
        output_path=payload["output_path"],
        inference_time_s=payload["inference_time_s"],
        peak_memory_mb=payload["peak_memory_mb"],
    )


def get_engine(request: Request) -> DiffusionEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="engine is not ready")
    return engine


@router.post("/v1/videos", response_model=VideoResponse, status_code=202)
def create_video(body: VideoCreateRequest, request: Request) -> VideoResponse:
    engine = get_engine(request)
    target = dict(body.target)
    if body.seconds is not None:
        target.setdefault("duration_seconds", float(body.seconds))
    job = DiffusionJob(
        prompt=body.prompt,
        task=body.task,
        conditions=body.conditions,
        target=target,
        num_inference_steps=body.num_inference_steps,
        seed=body.seed,
    )
    try:
        engine.submit(job)
    except AdmissionError as exc:
        # 429, not 503: the replica is healthy, the caller is early. Queueing
        # instead would turn a visible rejection into an invisible multi-minute
        # wait.
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job_to_response(job)


@router.get("/v1/videos/{job_id}", response_model=VideoResponse)
def get_video(job_id: str, request: Request) -> VideoResponse:
    job = get_engine(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return job_to_response(job)


@router.delete("/v1/videos/{job_id}", response_model=VideoResponse)
def abort_video(job_id: str, request: Request) -> VideoResponse:
    engine = get_engine(request)
    job = engine.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    engine.abort(job_id)
    return job_to_response(job)


@router.get("/v1/videos/{job_id}/content")
def get_video_content(job_id: str, request: Request):
    job = get_engine(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    if job.status is not JobStatus.COMPLETED:
        # 409, not 404: the job exists and the caller polled too early, which
        # is a different fix from a bad id.
        raise HTTPException(
            status_code=409,
            detail=f"job {job_id} is {job.status.name.lower()}, not completed",
        )
    if not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(
            status_code=500, detail=f"job {job_id} completed but its file is missing"
        )
    return FileResponse(
        job.output_path, media_type="video/mp4", filename=f"{job_id}.mp4"
    )


@router.get("/health")
def health(request: Request) -> JSONResponse:
    """Liveness plus queue depth.

    Reports the engine's own view rather than a bare 200: on the LLM side a
    healthy-looking ``/health`` with no model loaded has cost real debugging
    time, so this fails loudly when the engine is absent.
    """
    engine = get_engine(request)
    return JSONResponse({"status": "ok", **engine.stats()})
