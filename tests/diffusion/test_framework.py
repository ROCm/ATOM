# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Framework-level tests for the diffusion subsystem: config, request,
scheduler, the Ulysses group's shape contracts, stage/pipeline execution,
component placement and the runner.

Follows the repo convention in tests/conftest.py: import the real classes, do
not fake modules. Nothing here needs a GPU or AITER.
"""

from typing import ClassVar

import pytest
import torch

from atom.diffusion.config import DiffusionConfig, PerformanceMode
from atom.diffusion.engine.job_scheduler import AdmissionError, JobScheduler
from atom.diffusion.engine.pipeline_runner import PipelineRunner
from atom.diffusion.pipeline import (
    ComponentPlacement,
    ComposedPipeline,
    DiffusionBatch,
    PipelineStage,
    StageParallelism,
)
from atom.diffusion.request import DiffusionJob, JobStatus
from atom.diffusion.ulysses import UlyssesGroup


def make_config(**overrides) -> DiffusionConfig:
    kwargs = {
        "model_path": "<test>",
        "pipeline_class": "tests.diffusion.test_framework._Pipeline",
        "num_gpus": 1,
        "ulysses_degree": 1,
        "num_inference_steps": 3,
    }
    kwargs.update(overrides)
    return DiffusionConfig(**kwargs)


# ── config ────────────────────────────────────────────────────────────────


def test_ulysses_degree_must_equal_num_gpus():
    with pytest.raises(ValueError, match="must equal num_gpus"):
        make_config(num_gpus=8, ulysses_degree=4)
    make_config(num_gpus=8, ulysses_degree=8)


def test_queue_cap_cannot_be_below_concurrency():
    with pytest.raises(ValueError, match="max_queued_jobs"):
        make_config(max_queued_jobs=1, max_concurrent_jobs=2)


# ── request ───────────────────────────────────────────────────────────────


def test_job_progress_and_terminal_states():
    job = DiffusionJob(num_inference_steps=50)
    assert job.progress == 0.0
    assert not job.is_finished
    job.current_step = 25
    assert job.progress == pytest.approx(0.5)
    # progress is clamped: a scheduler bug must not report 140% done
    job.current_step = 70
    assert job.progress == 1.0
    job.mark_failed("boom")
    assert job.is_finished and job.status is JobStatus.FAILED


def test_job_ids_are_unique():
    assert DiffusionJob().job_id != DiffusionJob().job_id


def test_elapsed_is_none_until_finished():
    job = DiffusionJob()
    assert job.elapsed is None
    job.start_time = 10.0
    assert job.elapsed is None
    job.finish_time = 12.5
    assert job.elapsed == pytest.approx(2.5)


# ── scheduler ─────────────────────────────────────────────────────────────


def test_admission_rejects_when_full():
    sched = JobScheduler(make_config(max_queued_jobs=2, max_concurrent_jobs=1))
    sched.add_job(DiffusionJob())
    sched.add_job(DiffusionJob())
    with pytest.raises(AdmissionError, match="queue full"):
        sched.add_job(DiffusionJob())


def test_one_job_runs_at_a_time():
    sched = JobScheduler(make_config(max_concurrent_jobs=1))
    sched.add_job(DiffusionJob())
    sched.add_job(DiffusionJob())
    first = sched.schedule()
    assert first is not None
    assert sched.schedule() is None, "second job started while one was running"
    sched.complete(first, output_path="/tmp/a.mp4")
    assert sched.schedule() is not None


def test_aborted_job_is_dropped_not_run():
    sched = JobScheduler(make_config())
    job = DiffusionJob()
    sched.add_job(job)
    assert sched.abort(job.job_id) is True
    assert sched.schedule() is None
    assert sched.get(job.job_id).status is JobStatus.ABORTED


def test_complete_records_output_and_failure():
    sched = JobScheduler(make_config())
    job = DiffusionJob()
    sched.add_job(job)
    sched.schedule()
    sched.complete(job, output_path="/tmp/x.mp4")
    assert job.status is JobStatus.COMPLETED
    assert job.output_path == "/tmp/x.mp4"
    assert job.current_step == job.total_steps

    other = DiffusionJob()
    sched.add_job(other)
    sched.schedule()
    sched.complete(other, error="kernel exploded")
    assert other.status is JobStatus.FAILED
    assert other.error == "kernel exploded"


def test_abort_unknown_job_returns_false():
    assert JobScheduler(make_config()).abort("no-such-job") is False


# ── ulysses (single process => identity, but shape checks still apply) ────


def test_ulysses_single_process_is_identity():
    g = UlyssesGroup()
    assert g.world_size == 1 and g.is_main and not g.enabled
    x = torch.randn(8, 4, 16)
    assert torch.equal(g.scatter_heads(x), x)
    assert torch.equal(g.gather_heads(x), x)


def test_ulysses_broadcast_is_identity_single_process():
    assert UlyssesGroup().broadcast_object({"a": 1}) == {"a": 1}


def test_ulysses_rejects_non_3d():
    g = UlyssesGroup()
    g._world_size = 2  # simulate a group without needing torch.distributed
    with pytest.raises(ValueError, match="3-D"):
        g.scatter_heads(torch.randn(8, 4))


def test_ulysses_rejects_indivisible_shapes():
    g = UlyssesGroup()
    g._world_size = 8
    # heads not divisible by world size
    with pytest.raises(ValueError, match="head count"):
        g.scatter_heads(torch.randn(10, 7, 16))
    # sequence not divisible by world size
    with pytest.raises(ValueError, match="must be divisible"):
        g.gather_heads(torch.randn(10, 2, 16))


# ── stages & pipeline ─────────────────────────────────────────────────────


class _Produce(PipelineStage):
    produces = ("a",)

    def forward(self, batch, config):
        batch.set("a", 1)
        return batch


class _Consume(PipelineStage):
    requires = ("a",)
    produces = ("b",)

    def forward(self, batch, config):
        batch.set("b", batch.require("a") + 1)
        return batch


class _Liar(PipelineStage):
    produces = ("never",)

    def forward(self, batch, config):
        return batch


class _Null(PipelineStage):
    def forward(self, batch, config):
        return None


class _Pipeline(ComposedPipeline):
    pipeline_name = "TestPipeline"
    required_components = ("transformer",)

    def build_stages(self):
        return [_Produce(), _Consume()]


def _batch() -> DiffusionBatch:
    return DiffusionBatch(job=DiffusionJob())


def test_batch_require_names_missing_key():
    with pytest.raises(KeyError, match="latents"):
        _batch().require("latents")


def test_stage_missing_input_fails_loudly():
    with pytest.raises(KeyError, match="requires"):
        _Consume()(_batch(), make_config())


def test_stage_must_produce_what_it_declares():
    with pytest.raises(KeyError, match="did not produce"):
        _Liar()(_batch(), make_config())


def test_stage_returning_none_is_rejected():
    with pytest.raises(TypeError, match="returned None"):
        _Null()(_batch(), make_config())


def test_pipeline_runs_stages_in_order():
    pipe = _Pipeline(make_config())
    pipe.register_component("transformer", torch.nn.Identity())
    out = pipe.forward(_batch())
    assert out.get("a") == 1 and out.get("b") == 2
    assert set(pipe.last_stage_times) == {"_Produce", "_Consume"}


def test_pipeline_requires_its_components():
    pipe = _Pipeline(make_config())
    with pytest.raises(RuntimeError, match="missing required components"):
        pipe.forward(_batch())


# ── component placement ───────────────────────────────────────────────────
#
# These run at world size 1 and move ``_rank`` by hand, the same trick
# test_main_rank_only_stage_skipped_off_main uses. That exercises the
# bookkeeping, not the collectives -- multi-rank behaviour still needs a real
# multi-GPU run.


class _Placed(ComposedPipeline):
    pipeline_name = "PlacedPipeline"
    required_components = ("transformer", "shared_vae", "encoder")
    component_placement: ClassVar[dict[str, ComponentPlacement]] = {
        "shared_vae": ComponentPlacement.ALL_RANKS,
        "encoder": ComponentPlacement.MAIN_RANK,
    }

    def build_stages(self):
        return [_Produce(), _Consume()]


def _placed_off_main() -> _Placed:
    pipe = _Placed(make_config())
    pipe.ulysses._rank = 1
    return pipe


def test_an_unlisted_component_lives_on_every_rank():
    # "transformer" is absent from component_placement.
    assert _placed_off_main().holds("transformer")


def test_a_rank_refuses_a_component_it_should_not_hold():
    pipe = _placed_off_main()
    with pytest.raises(RuntimeError, match="rank 1 registered 'encoder'"):
        pipe.register_component("encoder", torch.nn.Identity())


def test_a_main_rank_component_is_not_missing_where_it_never_belonged():
    pipe = _placed_off_main()
    pipe.register_component("transformer", torch.nn.Identity())
    pipe.register_component("shared_vae", torch.nn.Identity())
    pipe.verify_components()  # "encoder" is rank 0's, so its absence is fine


def test_a_shared_component_is_still_required_off_main():
    # The regression the hand-written per-pipeline override let through: an
    # all-ranks VAE that failed to load on rank 1 used to surface at decode.
    pipe = _placed_off_main()
    pipe.register_component("transformer", torch.nn.Identity())
    with pytest.raises(RuntimeError, match=r"rank 1 .*\['shared_vae'\]"):
        pipe.verify_components()


def test_the_main_rank_holds_everything():
    pipe = _Placed(make_config())
    for name in _Placed.required_components:
        pipe.register_component(name, torch.nn.Identity())
    pipe.verify_components()


def test_pipeline_rejects_empty_stage_list():
    class _Empty(ComposedPipeline):
        def build_stages(self):
            return []

    with pytest.raises(ValueError, match="declared no stages"):
        _Empty(make_config())


def test_main_rank_only_stage_skipped_off_main():
    class _MainOnly(PipelineStage):
        parallelism = StageParallelism.MAIN_RANK_ONLY

        def forward(self, batch, config):
            batch.set("ran", True)
            return batch

    class _P(ComposedPipeline):
        required_components = ()

        def build_stages(self):
            return [_MainOnly()]

    pipe = _P(make_config())
    pipe.ulysses._rank = 1  # pretend we are not rank 0
    assert pipe.forward(_batch()).get("ran") is None


# ── runner ────────────────────────────────────────────────────────────────


def test_runner_rejects_unimplemented_performance_mode():
    pipe = _Pipeline(make_config())
    pipe.register_component("transformer", torch.nn.Identity())
    runner = PipelineRunner(make_config(), pipe, device="cpu")
    runner.config.performance_mode = "memory"  # not a PerformanceMode member
    with pytest.raises(NotImplementedError):
        runner.place_components()


def test_runner_runs_a_job_on_cpu():
    cfg = make_config()
    pipe = _Pipeline(cfg)
    pipe.register_component("transformer", torch.nn.Identity())
    runner = PipelineRunner(cfg, pipe, device=None)
    job = DiffusionJob(num_inference_steps=3)
    batch = runner.run_job(job)
    assert batch.get("b") == 2
    assert batch.meta["ulysses_world"] == 1
    assert cfg.performance_mode is PerformanceMode.SPEED


# ── warmup ────────────────────────────────────────────────────────────────


def test_base_pipeline_does_not_warm_by_default():
    """Warming is opt-in: what to warm is model-specific."""
    assert _Pipeline(make_config()).warmup("cpu") is False


def test_runner_skips_warmup_when_disabled():
    cfg = make_config(warmup=False)
    pipe = _Pipeline(cfg)
    calls = []
    pipe.warmup = lambda device: calls.append(device) or True
    runner = PipelineRunner(cfg, pipe, device="cpu")
    assert runner.warmup() is False
    assert calls == []


def test_runner_warms_and_passes_the_device():
    cfg = make_config()
    pipe = _Pipeline(cfg)
    calls = []
    pipe.warmup = lambda device: calls.append(device) or True
    runner = PipelineRunner(cfg, pipe, device="cpu")
    assert runner.warmup() is True
    assert calls == ["cpu"]


def test_runner_skips_warmup_without_a_device():
    """A CPU-only worker has no first-forward cost worth pre-paying."""
    cfg = make_config()
    pipe = _Pipeline(cfg)
    pipe.warmup = lambda device: True
    assert PipelineRunner(cfg, pipe, device=None).warmup() is False


def test_a_failing_warmup_does_not_kill_the_replica():
    """The same work reruns on the first request, where the error is reportable
    against a job rather than taking down a freshly loaded replica."""
    cfg = make_config()
    pipe = _Pipeline(cfg)

    def boom(device):
        raise RuntimeError("aiter jit exploded")

    pipe.warmup = boom
    assert PipelineRunner(cfg, pipe, device="cpu").warmup() is False


def test_placement_clears_requires_grad():
    """aiter's varlen attention asserts on a grad-requiring input.

    `.eval()` does not clear the flag. At Ulysses >= 2 the all-to-all writes
    into a fresh buffer and launders it away, so the failure only appears at
    degree 1 -- which is why every validated topology missed it.
    """
    cfg = make_config()
    pipe = _Pipeline(cfg)
    module = torch.nn.Linear(4, 4)
    assert all(p.requires_grad for p in module.parameters())
    pipe.register_component("transformer", module)
    PipelineRunner(cfg, pipe, device="cpu").place_components()
    assert not any(p.requires_grad for p in module.parameters())
