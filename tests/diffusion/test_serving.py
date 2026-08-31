# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Serving-layer tests: the CLI's argument and path resolution, the
checkpoint -> pipeline registry, the HTTP surface and the engine's job
lifecycle.
"""

import pickle

import pytest

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.engine.diffusion_engine import DiffusionEngine
from atom.diffusion.engine.engine_core import (
    DiffusionEngineCore,
    resolve_pipeline_class,
)
from atom.diffusion.engine.job_scheduler import AdmissionError
from atom.diffusion.engine.protocol import (
    EngineOutput,
    EngineRequest,
    OutputType,
    RequestType,
)
from atom.diffusion.request import DiffusionJob, JobStatus

# Tests about path and degree resolution name a pipeline explicitly, so they do
# not depend on a checkpoint being present at the path they pass.
PIPELINE_ARG_NAME = "--pipeline"
DUMMY_PIPELINE = "atom.diffusion.examples.dummy_pipeline.DummyPipeline"


def make_config(**kwargs) -> DiffusionConfig:
    defaults = {
        "model_path": "<test>",
        "pipeline_class": "atom.diffusion.models.minimax_h3.pipeline.MiniMaxH3Pipeline",
        "num_gpus": 1,
        "ulysses_degree": 1,
    }
    defaults.update(kwargs)
    return DiffusionConfig(**defaults)


def make_job(**kwargs) -> DiffusionJob:
    defaults = {"prompt": "three cats marching", "task": "t2va", "seed": 1101}
    defaults.update(kwargs)
    return DiffusionJob(**defaults)


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------


def test_requests_and_outputs_survive_pickle():
    """The transport is pickle over ZMQ, so anything unpicklable on a payload
    fails at dispatch time on a live GPU rather than in CI."""
    job = make_job()
    request = EngineRequest(type=RequestType.ADD, job=job)
    restored = pickle.loads(pickle.dumps(request))
    assert restored.type is RequestType.ADD
    assert restored.job.prompt == job.prompt
    assert restored.job_id == job.job_id

    output = EngineOutput.from_job(job, OutputType.RESULT, rank=2)
    restored_output = pickle.loads(pickle.dumps(output))
    assert restored_output.rank == 2
    assert restored_output.job_id == job.job_id


def test_add_without_a_job_is_rejected():
    with pytest.raises(ValueError, match="ADD requires a job"):
        EngineRequest(type=RequestType.ADD)


def test_abort_without_an_id_is_rejected():
    with pytest.raises(ValueError, match="ABORT requires a job_id"):
        EngineRequest(type=RequestType.ABORT)


def test_add_takes_its_id_from_the_job():
    job = make_job()
    assert EngineRequest(type=RequestType.ADD, job=job).job_id == job.job_id


def test_config_is_picklable_across_the_spawn_boundary():
    """Workers are spawned, so the config crosses a pickle boundary; a torch
    dtype on it would only fail once a real replica started."""
    config = make_config()
    assert pickle.loads(pickle.dumps(config)).model_path == config.model_path


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


class FakeUlysses:
    def __init__(self, rank=0, world_size=1):
        self.rank, self.world_size = rank, world_size

    @property
    def is_main(self):
        return self.rank == 0


class FakeRunner:
    """Records jobs and optionally fails, without touching a GPU."""

    def __init__(self, *, fail: bool = False, steps: int = 3):
        self.jobs = []
        self.fail = fail
        self.steps = steps

    def run_job(self, job, *, is_warmup=False, on_progress=None):
        self.jobs.append(job)
        if self.fail:
            raise RuntimeError("kernel exploded")
        for step in range(1, self.steps + 1):
            if on_progress is not None:
                on_progress(step, self.steps)
        job.output_path = f"/tmp/{job.job_id}.mp4"


def make_core(rank=0, world_size=1, **runner_kwargs):
    core = DiffusionEngineCore(
        make_config(), rank, ulysses=FakeUlysses(rank, world_size)
    )
    core.runner = FakeRunner(**runner_kwargs)
    return core


def collect(core, request):
    outputs = []
    core.handle(request, outputs.append)
    return outputs


def test_main_rank_reports_progress_then_a_result():
    core = make_core()
    outputs = collect(core, EngineRequest(type=RequestType.ADD, job=make_job()))
    assert [o.type for o in outputs] == [OutputType.PROGRESS] * 3 + [OutputType.RESULT]
    assert outputs[-1].status is JobStatus.COMPLETED
    assert outputs[-1].output_path.endswith(".mp4")


def test_non_main_ranks_run_the_job_but_stay_silent():
    """Every rank must enter every job -- Ulysses is collective -- but only
    rank 0 has a result, so the others would be N-1 duplicate completions."""
    core = make_core(rank=1, world_size=4)
    outputs = collect(core, EngineRequest(type=RequestType.ADD, job=make_job()))
    assert outputs == []
    assert len(core.runner.jobs) == 1


def test_every_rank_reports_a_failure():
    """The asymmetry is the diagnostic: one rank erroring while its peers are
    silent is what a hung all-to-all looks like."""
    core = make_core(rank=2, world_size=4, fail=True)
    outputs = collect(core, EngineRequest(type=RequestType.ADD, job=make_job()))
    assert [o.type for o in outputs] == [OutputType.ERROR]
    assert outputs[0].rank == 2
    assert "kernel exploded" in outputs[0].error
    assert "Traceback" in outputs[0].extra["traceback"]


def test_a_failed_job_is_marked_failed_not_left_running():
    core = make_core(fail=True)
    job = make_job()
    collect(core, EngineRequest(type=RequestType.ADD, job=job))
    assert job.status is JobStatus.FAILED
    assert job.finish_time is not None


def test_abort_before_dispatch_skips_the_job_entirely():
    core = make_core()
    job = make_job()
    core.handle(
        EngineRequest(type=RequestType.ABORT, job_id=job.job_id), lambda _o: None
    )
    outputs = collect(core, EngineRequest(type=RequestType.ADD, job=job))
    assert core.runner.jobs == []
    assert outputs[-1].status is JobStatus.ABORTED


def test_abort_applies_once_and_does_not_stick():
    """A stale abort must not swallow a later job that reuses the id."""
    core = make_core()
    job = make_job()
    core.handle(
        EngineRequest(type=RequestType.ABORT, job_id=job.job_id), lambda _o: None
    )
    collect(core, EngineRequest(type=RequestType.ADD, job=job))
    collect(core, EngineRequest(type=RequestType.ADD, job=job))
    assert len(core.runner.jobs) == 1


def test_shutdown_sets_the_loop_flag():
    core = make_core()
    core.handle(EngineRequest(type=RequestType.SHUTDOWN), lambda _o: None)
    assert core._shutdown is True


def test_pipeline_class_resolves_from_a_dotted_path():
    from atom.diffusion.models.minimax_h3.pipeline import MiniMaxH3Pipeline

    resolved = resolve_pipeline_class(
        "atom.diffusion.models.minimax_h3.pipeline.MiniMaxH3Pipeline"
    )
    assert resolved is MiniMaxH3Pipeline


def test_a_bare_class_name_is_refused():
    with pytest.raises(ValueError, match="dotted path"):
        resolve_pipeline_class("MiniMaxH3Pipeline")


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class FakeManager:
    """Stands in for the spawned workers: records sends, replays outputs."""

    def __init__(self, config):
        import queue

        self.config = config
        self.outputs = queue.Queue()
        self.sent = []
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True

    def send(self, request):
        self.sent.append(request)


def make_engine(**config_kwargs):
    config = make_config(**config_kwargs)
    engine = DiffusionEngine(config, manager=FakeManager(config))
    return engine


def test_submit_returns_immediately_with_a_queued_job():
    """Minutes-long work must not block the caller."""
    engine = make_engine()
    job = engine.submit(make_job())
    assert job.status is JobStatus.QUEUED
    assert engine.get(job.job_id) is job


def test_admission_rejects_rather_than_queueing_without_bound():
    engine = make_engine(max_queued_jobs=1)
    engine.submit(make_job())
    with pytest.raises(AdmissionError):
        engine.submit(make_job())


def test_dispatch_broadcasts_one_add_per_job():
    engine = make_engine()
    job = engine.submit(make_job())
    engine._dispatch_next()
    assert [r.type for r in engine.manager.sent] == [RequestType.ADD]
    assert engine.manager.sent[0].job.job_id == job.job_id
    # One job at a time: nothing more goes out until this one finishes.
    engine.submit(make_job())
    engine._dispatch_next()
    assert len(engine.manager.sent) == 1


def test_progress_updates_the_job_without_finishing_it():
    engine = make_engine()
    job = engine.submit(make_job(num_inference_steps=50))
    engine._dispatch_next()
    engine._apply(
        EngineOutput(
            type=OutputType.PROGRESS,
            job_id=job.job_id,
            current_step=25,
            total_steps=50,
        )
    )
    assert job.status is JobStatus.RUNNING
    assert job.progress == pytest.approx(0.5)
    assert not job.is_finished


def test_a_result_completes_the_job_and_frees_the_slot():
    engine = make_engine()
    first = engine.submit(make_job())
    engine._dispatch_next()
    engine._apply(
        EngineOutput(
            type=OutputType.RESULT,
            job_id=first.job_id,
            status=JobStatus.COMPLETED,
            output_path="/tmp/a.mp4",
        )
    )
    assert first.status is JobStatus.COMPLETED
    assert first.output_path == "/tmp/a.mp4"

    second = engine.submit(make_job())
    engine._dispatch_next()
    assert engine.manager.sent[-1].job.job_id == second.job_id


def test_the_first_rank_error_wins():
    """Later ranks usually report a collateral collective failure, so keeping
    the first error keeps the cause rather than the symptom."""
    engine = make_engine()
    job = engine.submit(make_job())
    engine._dispatch_next()
    engine._apply(
        EngineOutput(
            type=OutputType.ERROR, rank=2, job_id=job.job_id, error="real cause"
        )
    )
    engine._apply(
        EngineOutput(type=OutputType.ERROR, rank=0, job_id=job.job_id, error="timeout")
    )
    assert job.status is JobStatus.FAILED
    assert job.error == "real cause"


def test_a_dead_rank_fails_everything_in_flight():
    """A dead rank cannot be recovered from -- the collective is gone -- so a
    job left QUEUED would hang the caller forever."""
    engine = make_engine()
    job = engine.submit(make_job())
    engine._apply(EngineOutput(type=OutputType.DEAD, rank=3, error="OOM"))
    assert job.status is JobStatus.FAILED
    assert "rank 3" in job.error


def test_outputs_for_unknown_jobs_are_ignored():
    engine = make_engine()
    engine._apply(EngineOutput(type=OutputType.RESULT, job_id="ghost"))


def test_ready_messages_are_not_job_updates():
    engine = make_engine()
    engine._apply(EngineOutput(type=OutputType.READY, rank=0))


def test_abort_broadcasts_and_reports_whether_it_took():
    engine = make_engine()
    job = engine.submit(make_job())
    assert engine.abort(job.job_id) is True
    assert engine.manager.sent[-1].type is RequestType.ABORT
    assert engine.abort("ghost") is False


def test_close_is_idempotent_and_tears_the_workers_down():
    engine = make_engine()
    engine.close()
    engine.close()
    assert engine.manager.closed is True


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from atom.diffusion.entrypoints.video_api import router

    engine = make_engine()
    app = fastapi.FastAPI()
    app.state.engine = engine
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False), engine


def create_body(**kwargs):
    body = {"prompt": "three cats marching", "task": "t2va", "seed": 1101}
    body.update(kwargs)
    return body


def test_post_accepts_and_returns_a_job_id(client):
    http, _engine = client
    response = http.post("/v1/videos", json=create_body())
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["task"] == "t2va"
    assert payload["id"]


def test_task_is_required(client):
    """Its absence from sglang's offline CLI is what makes that CLI unable to
    drive H3 at all, so it is not inferred here."""
    http, _engine = client
    assert http.post("/v1/videos", json={"prompt": "x"}).status_code == 422


def test_seconds_becomes_a_target_duration(client):
    http, engine = client
    job_id = http.post("/v1/videos", json=create_body(seconds=5)).json()["id"]
    assert engine.get(job_id).target["duration_seconds"] == pytest.approx(5.0)


def test_an_explicit_target_duration_wins_over_seconds(client):
    http, engine = client
    body = create_body(seconds=5, target={"duration_seconds": 7.5})
    job_id = http.post("/v1/videos", json=body).json()["id"]
    assert engine.get(job_id).target["duration_seconds"] == pytest.approx(7.5)


def test_a_full_queue_is_backpressure_not_an_error(client):
    http, engine = client
    engine.scheduler.config.max_queued_jobs = 1
    http.post("/v1/videos", json=create_body())
    response = http.post("/v1/videos", json=create_body())
    assert response.status_code == 429


def test_polling_reports_progress(client):
    http, engine = client
    job_id = http.post("/v1/videos", json=create_body()).json()["id"]
    engine._apply(
        EngineOutput(
            type=OutputType.PROGRESS, job_id=job_id, current_step=10, total_steps=50
        )
    )
    payload = http.get(f"/v1/videos/{job_id}").json()
    assert payload["status"] == "running"
    assert payload["progress"] == pytest.approx(0.2)


def test_unknown_job_is_404(client):
    http, _engine = client
    assert http.get("/v1/videos/ghost").status_code == 404


def test_content_before_completion_is_409_not_404(client):
    """The job exists and the caller polled early; that is a different fix
    from a bad id."""
    http, _engine = client
    job_id = http.post("/v1/videos", json=create_body()).json()["id"]
    response = http.get(f"/v1/videos/{job_id}/content")
    assert response.status_code == 409
    assert "queued" in response.json()["detail"]


def test_content_returns_the_mp4(client, tmp_path):
    http, engine = client
    job_id = http.post("/v1/videos", json=create_body()).json()["id"]
    path = tmp_path / "out.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    engine._apply(
        EngineOutput(
            type=OutputType.RESULT,
            job_id=job_id,
            status=JobStatus.COMPLETED,
            output_path=str(path),
        )
    )
    response = http.get(f"/v1/videos/{job_id}/content")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == path.read_bytes()


def test_a_completed_job_whose_file_vanished_is_a_server_error(client, tmp_path):
    http, engine = client
    job_id = http.post("/v1/videos", json=create_body()).json()["id"]
    engine._apply(
        EngineOutput(
            type=OutputType.RESULT,
            job_id=job_id,
            status=JobStatus.COMPLETED,
            output_path=str(tmp_path / "gone.mp4"),
        )
    )
    assert http.get(f"/v1/videos/{job_id}/content").status_code == 500


def test_delete_aborts(client):
    http, engine = client
    job_id = http.post("/v1/videos", json=create_body()).json()["id"]
    assert http.delete(f"/v1/videos/{job_id}").status_code == 200
    assert engine.manager.sent[-1].type is RequestType.ABORT


def test_health_reports_queue_depth_not_just_ok(client):
    """A bare 200 from /health with no model loaded has cost real debugging
    time on the LLM side."""
    http, _engine = client
    http.post("/v1/videos", json=create_body())
    payload = http.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["waiting"] == 1


def test_health_fails_when_the_engine_is_absent():
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from atom.diffusion.entrypoints.video_api import router

    app = fastapi.FastAPI()
    app.include_router(router)
    http = TestClient(app, raise_server_exceptions=False)
    assert http.get("/health").status_code == 503


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_ulysses_degree_defaults_to_the_gpu_count():
    from atom.diffusion.entrypoints.diffusion_server import (
        build_parser,
        config_from_args,
    )

    args = build_parser().parse_args(
        [PIPELINE_ARG_NAME, DUMMY_PIPELINE, "--model", "/m", "--num-gpus", "4"]
    )
    assert config_from_args(args).ulysses_degree == 4


def test_a_bad_ulysses_degree_is_refused_at_config_time():
    from atom.diffusion.entrypoints.diffusion_server import (
        build_parser,
        config_from_args,
    )

    args = build_parser().parse_args(
        [
            PIPELINE_ARG_NAME,
            DUMMY_PIPELINE,
            "--model",
            "/m",
            "--num-gpus",
            "4",
            "--ulysses-degree",
            "3",
        ]
    )
    with pytest.raises(ValueError):
        config_from_args(args)


def test_a_queued_job_is_not_resurrected_after_a_rank_dies():
    """A rank dying takes the replica with it, so a job that was still queued
    is FAILED before it ever starts. Dispatching it anyway would flip it back
    to RUNNING and leave the caller polling a replica that no longer exists."""
    engine = make_engine(max_queued_jobs=8)
    running = engine.submit(make_job())
    engine._dispatch_next()
    queued = engine.submit(make_job())

    engine._apply(EngineOutput(type=OutputType.DEAD, rank=3, error="OOM"))
    assert running.status is JobStatus.FAILED
    assert queued.status is JobStatus.FAILED

    engine._dispatch_next()
    assert queued.status is JobStatus.FAILED
    assert len(engine.manager.sent) == 1


def test_the_scheduler_drops_any_terminal_job_before_start():
    from atom.diffusion.engine.job_scheduler import JobScheduler

    scheduler = JobScheduler(make_config(max_queued_jobs=8))
    failed = scheduler.add_job(make_job())
    failed.mark_failed("rank died")
    runnable = scheduler.add_job(make_job())
    assert scheduler.schedule() is runnable
    assert failed.job_id in scheduler.finished


def test_model_variant_resolves_to_the_partition_path(tmp_path):
    from atom.diffusion.entrypoints.diffusion_server import (
        build_parser,
        config_from_args,
    )

    args = build_parser().parse_args(
        [
            PIPELINE_ARG_NAME,
            DUMMY_PIPELINE,
            "--model",
            str(tmp_path),
            "--model-variant",
            "FL2VA",
        ]
    )
    assert config_from_args(args).model_path == str(tmp_path / "FL2VA")


def test_a_model_path_that_already_names_the_partition_is_left_alone(tmp_path):
    """Both spellings are in circulation; joining blindly would produce
    .../FL2VA/FL2VA and fail at load with a confusing missing-file error."""
    from atom.diffusion.entrypoints.diffusion_server import (
        build_parser,
        config_from_args,
    )

    root = tmp_path / "FL2VA"
    args = build_parser().parse_args(
        [
            PIPELINE_ARG_NAME,
            DUMMY_PIPELINE,
            "--model",
            str(root),
            "--model-variant",
            "FL2VA",
        ]
    )
    assert config_from_args(args).model_path == str(root)


def test_the_pipeline_base_takes_a_model_root():
    """The worker builds whatever pipeline the config names, so it cannot pass
    a kwarg only one subclass accepts."""
    from atom.diffusion.models.minimax_h3.pipeline import MiniMaxH3Pipeline

    config = make_config(model_path="/data/root")
    assert MiniMaxH3Pipeline(config).model_root == "/data/root"
    assert MiniMaxH3Pipeline(config, model_root="/other").model_root == "/other"


# ----------------------------------------------------------------------
# checkpoint architecture -> pipeline, mirroring the LLM side's arch dicts
# ----------------------------------------------------------------------


def write_model_index(root, class_name):
    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
    return root


def test_a_known_checkpoint_selects_its_pipeline_without_a_flag(tmp_path):
    from atom.diffusion.entrypoints.diffusion_server import (
        build_parser,
        config_from_args,
    )

    root = write_model_index(tmp_path / "FL2VA", "MiniMaxH3ModularPipeline")
    args = build_parser().parse_args(["--model", str(root)])
    assert config_from_args(args).pipeline_class == (
        "atom.diffusion.models.minimax_h3.pipeline.MiniMaxH3Pipeline"
    )


def test_a_partition_manifest_resolves_like_the_root(tmp_path):
    """``--model <root>/FL2VA`` is the documented invocation, and a partition's
    model_index.json declares MiniMaxH3Pipeline where the root declares
    MiniMaxH3ModularPipeline. Both have to land on the same pipeline."""
    from atom.diffusion.registry import pipeline_class_for_checkpoint

    root = write_model_index(tmp_path / "root", "MiniMaxH3ModularPipeline")
    part = write_model_index(tmp_path / "root" / "FL2VA", "MiniMaxH3Pipeline")
    assert pipeline_class_for_checkpoint(str(part)) == pipeline_class_for_checkpoint(
        str(root)
    )


def test_the_pipeline_flag_overrides_the_checkpoint(tmp_path):
    """Out-of-tree pipelines have no registry entry, so the flag has to win."""
    from atom.diffusion.entrypoints.diffusion_server import (
        build_parser,
        config_from_args,
    )

    root = write_model_index(tmp_path / "FL2VA", "MiniMaxH3ModularPipeline")
    args = build_parser().parse_args(
        [PIPELINE_ARG_NAME, DUMMY_PIPELINE, "--model", str(root)]
    )
    assert config_from_args(args).pipeline_class == DUMMY_PIPELINE


def test_an_unknown_architecture_names_what_is_supported(tmp_path):
    from atom.diffusion.entrypoints.diffusion_server import (
        build_parser,
        config_from_args,
    )

    root = write_model_index(tmp_path / "ckpt", "SomeOtherPipeline")
    args = build_parser().parse_args(["--model", str(root)])
    with pytest.raises(ValueError, match="SomeOtherPipeline"):
        config_from_args(args)


def test_a_checkpoint_with_no_index_says_so(tmp_path):
    """The failure has to name the missing file: the common cause is pointing
    --model one level too high, at a directory that holds checkpoints rather
    than at one."""
    from atom.diffusion.entrypoints.diffusion_server import (
        build_parser,
        config_from_args,
    )

    args = build_parser().parse_args(["--model", str(tmp_path)])
    with pytest.raises(ValueError, match="model_index.json"):
        config_from_args(args)


def test_a_malformed_index_is_a_missing_one(tmp_path):
    from atom.diffusion.registry import pipeline_class_for_checkpoint

    tmp_path.joinpath("model_index.json").write_text("{not json")
    assert pipeline_class_for_checkpoint(str(tmp_path)) is None
