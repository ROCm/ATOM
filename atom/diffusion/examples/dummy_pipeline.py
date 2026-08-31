# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A minimal end-to-end diffusion pipeline used to validate the skeleton.

It carries no real weights, but it exercises every mechanism the real
MiniMax-H3 pipeline will depend on:

  * all three :class:`StageParallelism` modes, including a broadcast
  * a denoise loop shaped like the real one (Ulysses all-to-all -> attention ->
    all-to-all back), at H3's actual packed geometry
  * a round-trip assertion on the Ulysses transforms, so a layout bug fails
    here rather than as silent garbage in a generated video

Run on 8 GPUs:

    torchrun --nproc_per_node=8 -m atom.diffusion.examples.dummy_pipeline
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.engine.job_scheduler import JobScheduler
from atom.diffusion.engine.pipeline_runner import PipelineRunner
from atom.diffusion.pipeline import (
    ComposedPipeline,
    DiffusionBatch,
    PipelineStage,
    StageParallelism,
)
from atom.diffusion.request import DiffusionJob
from atom.diffusion.ulysses import UlyssesGroup

logger = logging.getLogger(__name__)

# MiniMax-H3 at 1344x768x124f, captured from a live forward.
S_TOTAL = 37760
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128


class TextEncodingStage(PipelineStage):
    """Serial work on rank 0, result needed everywhere -> broadcast."""

    parallelism = StageParallelism.MAIN_RANK_BROADCAST
    produces = ("prompt_embeds",)

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        # Real pipelines run a 66 GB Qwen3-VL here; a list stands in because
        # broadcast_object moves Python objects, not device tensors.
        batch.set("prompt_embeds", [float(len(batch.job.prompt))] * 8)
        return batch


class LatentPreparationStage(PipelineStage):
    """Replicated: every rank builds its own shard of the sequence."""

    parallelism = StageParallelism.REPLICATED
    requires = ("prompt_embeds",)
    produces = ("latents",)

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        world = batch.meta["ulysses_world"]
        device = batch.meta.get("device", "cpu")
        s_local = S_TOTAL // world
        gen = torch.Generator(device="cpu").manual_seed(batch.job.seed or 0)
        latents = torch.randn(
            s_local, HEADS, HEAD_DIM, generator=gen, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)
        batch.set("latents", latents)
        return batch


class DenoiseStage(PipelineStage):
    """Replicated collective work: the Ulysses round trip, once per step."""

    parallelism = StageParallelism.REPLICATED
    requires = ("latents",)
    produces = ("denoised",)

    def __init__(self, ulysses: UlyssesGroup) -> None:
        self.ulysses = ulysses

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        x = batch.require("latents")
        reference = x.clone()

        for step in range(batch.job.num_inference_steps):
            # sequence-parallel -> head-parallel
            heads_shard = self.ulysses.scatter_heads(x)
            expected = (S_TOTAL, HEADS // max(self.ulysses.world_size, 1), HEAD_DIM)
            if tuple(heads_shard.shape) != expected:
                raise AssertionError(
                    f"scatter_heads gave {tuple(heads_shard.shape)}, want {expected}"
                )
            # stand-in for attention over the full sequence
            heads_shard = heads_shard * 1.0
            # head-parallel -> sequence-parallel
            x = self.ulysses.gather_heads(heads_shard)
            batch.job.current_step = step + 1

        # A pure round trip must be the identity. If the two permutes disagree
        # the result is a plausible-looking but wrongly-shuffled tensor, which
        # is exactly the kind of bug that only shows up as a corrupted frame.
        if not torch.equal(x, reference):
            mismatched = int((x != reference).sum().item())
            raise AssertionError(
                f"ulysses round trip corrupted {mismatched} elements; "
                "scatter_heads/gather_heads are not inverses"
            )

        batch.set("denoised", x)
        return batch


class PresentationStage(PipelineStage):
    """Terminal side effect on rank 0 only; nothing to share."""

    parallelism = StageParallelism.MAIN_RANK_ONLY
    requires = ("denoised",)

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        out = batch.require("denoised")
        batch.meta["output_summary"] = {
            "shape": tuple(out.shape),
            "dtype": str(out.dtype),
        }
        return batch


class DummyPipeline(ComposedPipeline):
    pipeline_name = "DummyPipeline"
    required_components = ("transformer",)

    def build_stages(self):
        return [
            TextEncodingStage(),
            LatentPreparationStage(),
            DenoiseStage(self.ulysses),
            PresentationStage(),
        ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = "cpu"

    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif torch.cuda.is_available():
        device = "cuda:0"

    config = DiffusionConfig(
        model_path="<dummy>",
        pipeline_class="atom.diffusion.examples.dummy_pipeline.DummyPipeline",
        num_gpus=world,
        ulysses_degree=world,
        num_inference_steps=args.steps,
    )

    ulysses = UlyssesGroup()
    pipeline = DummyPipeline(config, ulysses)
    pipeline.register_component("transformer", torch.nn.Identity())

    runner = PipelineRunner(config, pipeline, ulysses, device=device)
    runner.place_components()

    scheduler = JobScheduler(config)
    scheduler.add_job(
        DiffusionJob(
            prompt="a skeleton pipeline validating its own plumbing",
            task="dummy",
            num_inference_steps=args.steps,
            seed=1101,
        )
    )

    job = scheduler.schedule()
    assert job is not None, "scheduler admitted a job but would not schedule it"

    try:
        batch = runner.run_job(job)
        scheduler.complete(job, output_path="<none>")
    except Exception as exc:  # noqa: BLE001
        scheduler.complete(job, error=str(exc))
        if ulysses.is_main:
            logger.error("FAILED: %s", exc)
        if world > 1:
            dist.destroy_process_group()
        return 1

    if ulysses.is_main:
        print()
        print(f"world_size            : {ulysses.world_size}")
        print(f"device                : {device}")
        print(f"packed tokens         : {S_TOTAL} ({S_TOTAL // world} per rank)")
        print(f"heads                 : {HEADS} ({HEADS // world} per rank)")
        print(f"denoise steps         : {args.steps}")
        print("ulysses round trip    : exact")
        print(f"output                : {batch.meta.get('output_summary')}")
        print(f"job                   : {job.to_dict()}")
        print(f"scheduler             : {scheduler.stats()}")
        print()
        print(pipeline.stage_timing_report())
        print()
        print("DUMMY PIPELINE OK")

    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
