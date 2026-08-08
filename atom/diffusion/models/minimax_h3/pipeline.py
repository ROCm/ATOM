# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 pipeline: t2va, fl2va and ref2va.

    TextEncoding (rank 0, broadcast)
      -> Plan             geometry + sigma schedules
      -> ConditionEncode  conditioning -> packed rows (rank 0, broadcast)
      -> LatentPrep       seeded noise, row form
      -> PackedSequence   layout, position grid, per-rank shard
      -> Denoise          N-1 steps of DiT + Euler-ancestral
      -> Decode           video/audio VAE (rank 0)
      -> Present          H.264 + AAC mux (rank 0)

fl2va conditions on its anchor **twice** and both are required: the Qwen3-VL
vision tower folds it into the prompt sequence (1,010 of 1,029 tokens for a
1344x768 anchor) and the video VAE encodes it into 1,008 packed rows. Wiring
only one produces a plausible video that ignores its keyframe.
"""

import contextlib
import logging
import os

import torch

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.models.minimax_h3.condition_noise import (
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    imgvid_cond_noise_aug_rows,
)
from atom.diffusion.models.minimax_h3.denoise import run_denoise_loop
from atom.diffusion.models.minimax_h3.geometry import (
    VAE_SPATIAL_COMPRESSION,
    MiniMaxH3Geometry,
    align_frame_count,
    time_shift_sigmas,
)
from atom.diffusion.models.minimax_h3.keyframe import (
    encode_keyframe_cond_rows,
    prepare_keyframe_canvas,
    stretch_keyframe_canvas,
)
from atom.diffusion.models.minimax_h3.latent_prep import build_initial_latents
from atom.diffusion.models.minimax_h3.packed_sequence import (
    build_packed_sequence,
    build_packed_sequence_ref2va,
    validate_keyframe_signature,
)
from atom.diffusion.models.minimax_h3.presentation import ref2va_presentation
from atom.diffusion.models.minimax_h3.reference_encoding import (
    AUDIO_SAMPLE_RATE,
    decode_reference_video_frames,
    encode_reference_audio_rows,
    encode_reference_video_rows,
    resize_reference_image,
    resolve_reference_image_shape,
    sample_reference_video_frames,
)
from atom.diffusion.models.minimax_h3.scheduler import (
    MiniMaxH3EulerAncestralEta0Scheduler,
)
from atom.diffusion.models.minimax_h3.vae import (
    decode_audio_rows,
    decode_video_rows,
    latent_stats,
)
from atom.diffusion.mux import write_video_with_audio
from atom.diffusion.pipeline import (
    ComposedPipeline,
    DiffusionBatch,
    PipelineStage,
    StageParallelism,
)

logger = logging.getLogger(__name__)

# Matches the reference's fallback when a request carries no seed.
DEFAULT_NOISE_SEED = 42
DEFAULT_FLOW_SHIFT = 12.0
DEFAULT_AUDIO_FLOW_SHIFT = 3.0
SUPPORTED_TASKS = ("t2va", "fl2va", "ref2va")
KEYFRAME_CONDITION_TYPES = ("image",)
REFERENCE_CONDITION_TYPES = ("image", "audio", "video", "video_audio")


def keyframe_conditions(job) -> list[dict]:
    """Resolve a job's fl2va keyframes to canvas-sized PIL images.

    The *same* prepared image feeds Qwen and the VAE, so preparation happens
    once here rather than in each consumer -- a mismatch between the two would
    show up only as a keyframe the video drifts away from.

    The first keyframe is the geometry anchor and is **stretched** onto the
    canvas; a second one is a follower and gets an aspect-preserving cover
    crop. Getting that backwards silently changes the anchor's framing.
    """
    if not job.conditions:
        return []
    from PIL import Image, ImageOps

    target = job.target or {}
    width = int(target.get("width", 1344))
    height = int(target.get("height", 768))

    prepared: list[dict] = []
    for index, condition in enumerate(job.conditions):
        kind = str(condition.get("type", "image"))
        if kind not in KEYFRAME_CONDITION_TYPES:
            raise ValueError(
                f"conditions[{index}] type {kind!r} is not a keyframe; "
                f"expected one of {list(KEYFRAME_CONDITION_TYPES)}"
            )
        uri = str(condition.get("uri") or condition.get("path") or "")
        if not uri:
            raise ValueError(f"conditions[{index}] has no uri")
        path = uri.removeprefix("file://")
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        prepared.append(
            {
                "image": (
                    stretch_keyframe_canvas(
                        image, target_width=width, target_height=height
                    )
                    if index == 0
                    else prepare_keyframe_canvas(
                        image,
                        target_width=width,
                        target_height=height,
                        allow_upscale=True,
                    )
                ),
                "frame_index": condition.get("frame_index", 0),
            }
        )
    return prepared


def reference_materials(job) -> list[dict]:
    """Resolve a ref2va job's references to encode-ready material.

    References do **not** bind the target canvas -- that is the standing
    difference from an fl2va keyframe. An image reference goes to its own
    2048px short edge; a video reference is decoded at the target canvas and
    capped to the target's 17n+5 frame count.

    Ordinals are per *type* and 1-based, because that is what the presentation
    labels ("<Picture 1>", "<Audio 1>") count.
    """
    if not job.conditions:
        return []

    target = job.target or {}
    width = int(target.get("width", 1344))
    height = int(target.get("height", 768))
    fps = int(target.get("fps", 24))
    frame_count = align_frame_count(
        round(float(target.get("duration_seconds", 5.0)) * fps)
    )

    ordinals: dict[str, int] = {}
    prepared: list[dict] = []
    for index, condition in enumerate(job.conditions):
        kind = str(condition.get("type", "image"))
        if kind not in REFERENCE_CONDITION_TYPES:
            raise ValueError(
                f"conditions[{index}] type {kind!r} is not a ref2va reference; "
                f"expected one of {list(REFERENCE_CONDITION_TYPES)}"
            )
        uri = str(condition.get("uri") or condition.get("path") or "")
        if not uri:
            raise ValueError(f"conditions[{index}] has no uri")
        path = uri.removeprefix("file://")
        label_kind = (
            "image" if kind == "image" else ("audio" if kind == "audio" else "video")
        )
        ordinals[label_kind] = ordinals.get(label_kind, 0) + 1
        item = {
            "kind": kind,
            "label_kind": label_kind,
            "ordinal": ordinals[label_kind],
            "path": path,
            "start_time_seconds": float(condition.get("start_time_seconds", 0.0)),
        }

        if kind == "image":
            from PIL import Image, ImageOps

            with Image.open(path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
            shape = resolve_reference_image_shape(
                width=image.size[0], height=image.size[1]
            )
            item["image"] = resize_reference_image(
                image, target_width=shape["width"], target_height=shape["height"]
            )
            item["width"], item["height"] = shape["width"], shape["height"]
        elif kind == "audio":
            item["source_sample_rate"] = int(
                condition.get("sample_rate", AUDIO_SAMPLE_RATE)
            )
            item["max_duration_seconds"] = condition.get("max_duration_seconds")
        else:
            # No pre-queue shape hook here, so default to the target canvas;
            # an explicit width/height on the condition overrides it.
            ref_width = int(condition.get("width", width))
            ref_height = int(condition.get("height", height))
            frames = decode_reference_video_frames(
                path,
                target_width=ref_width,
                target_height=ref_height,
                target_frame_count=frame_count,
                fps=float(fps),
                start_time_seconds=item["start_time_seconds"],
            )
            item["frames"] = frames
            item["width"], item["height"] = ref_width, ref_height
            item.update(sample_reference_video_frames(frames))
        prepared.append(item)
    return prepared


class TextEncodingStage(PipelineStage):
    """Encode the prompt on rank 0 and share the result.

    Running a 66 GB encoder on every rank would be pure waste, and every
    downstream rank needs the embeddings, so this is the broadcast case.
    """

    parallelism = StageParallelism.MAIN_RANK_BROADCAST
    produces = ("prompt_embeds", "text_token_tags")

    def __init__(self, pipeline: "ComposedPipeline") -> None:
        self.pipeline = pipeline

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        encoder = self.pipeline.component("text_encoder")
        device = getattr(self.pipeline, "encode_device", None)
        # A CPU-staged encoder is moved in for the call and back out after; one
        # already on the target device (offline runs, tests) is left alone.
        scope = (
            encoder.resident_on(device)
            if device is not None and hasattr(encoder, "resident_on")
            else contextlib.nullcontext(encoder)
        )
        with scope as active:
            if batch.job.task == "ref2va":
                rows, tags = self._encode_ref2va(active, batch)
            else:
                images = [item["image"] for item in keyframe_conditions(batch.job)]
                rows, tags = active.encode_with_tags(batch.job.prompt, images or None)
            rows = rows.to(torch.bfloat16).cpu()
        batch.set("prompt_embeds", rows)
        batch.set("text_token_tags", tags.cpu())
        return batch

    @staticmethod
    def _encode_ref2va(encoder, batch: DiffusionBatch):
        """Label every reference in request order, then the prompt.

        Audio contributes a **label only** -- its content never reaches Qwen,
        only the audio VAE. Video contributes one timestamped block per
        temporal chunk of its 2 FPS sampled view.
        """
        materials = reference_materials(batch.job)
        if not materials:
            raise ValueError("ref2va requires at least one reference condition")

        images = [m["image"] for m in materials if m["kind"] == "image"]
        videos = [m for m in materials if m["kind"] in ("video", "video_audio")]

        vision, image_counts = (
            encoder.image_token_counts(images) if images else (None, [])
        )
        video_inputs, video_counts, video_stamps = None, [], []
        if videos:
            if encoder.processor is None:
                raise RuntimeError("video references need AutoProcessor")
            video_inputs = encoder.processor.video_processor(
                videos=[m["frames"] for m in videos], return_tensors="pt"
            )
            merge = int(encoder.processor.image_processor.merge_size) ** 2
            grid = video_inputs["video_grid_thw"]
            for index, material in enumerate(videos):
                stamps = list(material["block_timestamps"])
                total = int(grid[index].prod().item()) // merge
                if not stamps:
                    raise ValueError("video reference produced no Qwen blocks")
                if total % len(stamps):
                    raise ValueError(
                        f"video reference has {total} tokens across "
                        f"{len(stamps)} blocks, which does not divide evenly"
                    )
                video_counts.append([total // len(stamps)] * len(stamps))
                video_stamps.append(stamps)

        ids, tags = ref2va_presentation(
            encoder.tokenizer,
            prompt=batch.job.prompt,
            condition_labels=[(m["label_kind"], m["ordinal"]) for m in materials],
            image_token_counts=image_counts,
            video_block_token_counts=video_counts,
            video_block_timestamps=video_stamps,
        )
        rows = encoder.encode_ids(
            ids,
            pixel_values=None if vision is None else vision["pixel_values"],
            image_grid_thw=None if vision is None else vision["image_grid_thw"],
            pixel_values_videos=(
                None if video_inputs is None else video_inputs["pixel_values_videos"]
            ),
            video_grid_thw=(
                None if video_inputs is None else video_inputs["video_grid_thw"]
            ),
        )
        if int(tags.numel()) != int(rows.shape[0]):
            raise ValueError(
                f"ref2va presentation produced {int(tags.numel())} tags for "
                f"{int(rows.shape[0])} rows"
            )
        return rows, tags


class PlanStage(PipelineStage):
    """Resolve request geometry and the per-modality sigma schedules."""

    parallelism = StageParallelism.REPLICATED
    produces = ("geometry", "video_sigmas", "audio_sigmas", "keyframe_indices")

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        job = batch.job
        if job.task and job.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"task {job.task!r} is not implemented; supported: "
                f"{list(SUPPORTED_TASKS)}"
            )
        if job.task == "ref2va":
            if not job.conditions:
                raise ValueError("ref2va requires at least one reference condition")
            keyframe_indices: list[int] = []
        else:
            keyframes = keyframe_conditions(job)
            if keyframes and job.task != "fl2va":
                raise ValueError(
                    f"task {job.task!r} cannot carry keyframe conditions; use fl2va"
                )
            if job.task == "fl2va" and not keyframes:
                raise ValueError("fl2va requires at least one keyframe condition")
            keyframe_indices = (
                list(validate_keyframe_signature([k["frame_index"] for k in keyframes]))
                if keyframes
                else []
            )
        target = job.target or {}
        height = int(target.get("height", 768))
        width = int(target.get("width", 1344))
        duration = float(target.get("duration_seconds", 5.0))
        fps = int(target.get("fps", 24))
        text_len = int(batch.require("prompt_embeds").shape[0])

        geometry = MiniMaxH3Geometry.resolve(
            height=height,
            width=width,
            frame_count=round(duration * fps),
            duration_seconds=duration,
            text_len=text_len,
        )
        geometry.validate_ulysses(batch.meta["ulysses_world"])

        steps = job.num_inference_steps
        batch.set("geometry", geometry)
        batch.set(
            "video_sigmas",
            time_shift_sigmas(
                num_steps=steps,
                shift_scale=float(target.get("flow_shift", DEFAULT_FLOW_SHIFT)),
            ),
        )
        batch.set(
            "audio_sigmas",
            time_shift_sigmas(
                num_steps=steps,
                shift_scale=float(
                    target.get("audio_flow_shift", DEFAULT_AUDIO_FLOW_SHIFT)
                ),
            ),
        )
        batch.set("keyframe_indices", keyframe_indices)
        batch.job.total_steps = steps - 1
        return batch


class ConditionEncodeStage(PipelineStage):
    """Encode conditioning material into packed rows (rank 0, broadcast).

    Mirrors text encoding: the encode is cheap next to broadcasting ~1k rows,
    and it keeps the VAEs off the other ranks. t2va passes straight through.

    fl2va contributes visual rows only. ref2va contributes both, plus the block
    descriptors the packed layout needs -- and the two conditioning kinds are
    augmented differently (visual rows are mixed with seeded noise, audio
    references are not), which is why they are produced together here.
    """

    parallelism = StageParallelism.MAIN_RANK_BROADCAST
    requires = ("geometry",)
    produces = ("cond_rows", "cond_audio_rows", "ref_blocks")

    def __init__(self, pipeline: "ComposedPipeline") -> None:
        self.pipeline = pipeline

    @property
    def video_stats(self):
        return self.pipeline.video_stats

    @property
    def audio_stats(self):
        return self.pipeline.audio_stats

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        batch.set("cond_audio_rows", None)
        batch.set("ref_blocks", None)
        if batch.job.task == "ref2va":
            return self._encode_references(batch, config)

        keyframes = keyframe_conditions(batch.job)
        if not keyframes:
            batch.set("cond_rows", None)
            return batch
        g = batch.require("geometry")
        mean, std = self.video_stats
        clean = torch.cat(
            [
                encode_keyframe_cond_rows(
                    self.pipeline.component("video_vae"),
                    item["image"],
                    latents_mean=mean,
                    latents_std=std,
                )
                for item in keyframes
            ],
            dim=0,
        ).cpu()
        batch.set(
            "cond_rows",
            self._noise_augment(
                clean,
                [(1, g.latent_h, g.latent_w) for _ in keyframes],
                g.latent_t,
                batch,
                config,
            ),
        )
        return batch

    def _encode_references(self, batch: DiffusionBatch, config: DiffusionConfig):
        g = batch.require("geometry")
        v_mean, v_std = self.video_stats
        a_stats = self.audio_stats or (None, None)
        materials = reference_materials(batch.job)

        blocks: list[dict] = []
        visual: list[torch.Tensor] = []
        audio: list[torch.Tensor] = []
        shapes: list[tuple[int, int, int]] = []
        for material in materials:
            kind = material["kind"]
            if kind == "image":
                rows = encode_keyframe_cond_rows(
                    self.pipeline.component("video_vae"),
                    material["image"],
                    latents_mean=v_mean,
                    latents_std=v_std,
                )
                latent_h = material["height"] // VAE_SPATIAL_COMPRESSION
                latent_w = material["width"] // VAE_SPATIAL_COMPRESSION
                visual.append(rows)
                shapes.append((1, latent_h, latent_w))
                blocks.append(
                    {"kind": "image", "latent_h": latent_h, "latent_w": latent_w}
                )
            elif kind == "audio":
                if a_stats[0] is None:
                    raise ValueError(
                        "an audio reference needs the audio VAE's latent stats"
                    )
                out = encode_reference_audio_rows(
                    self.pipeline.component("audio_vae"),
                    material["path"],
                    latents_mean=a_stats[0],
                    latents_std=a_stats[1],
                    max_duration_seconds=material.get("max_duration_seconds"),
                    start_time_seconds=material["start_time_seconds"],
                    source_sample_rate=material["source_sample_rate"],
                )
                audio.append(out["rows"])
                blocks.append({"kind": "audio", "ref_audio_t": out["ref_audio_t"]})
            else:
                rows, latent_t, latent_h, latent_w = encode_reference_video_rows(
                    self.pipeline.component("video_vae"),
                    material["frames"],
                    latents_mean=v_mean,
                    latents_std=v_std,
                )
                visual.append(rows)
                shapes.append((latent_t, latent_h, latent_w))
                block = {
                    "kind": kind,
                    "latent_t": latent_t,
                    "latent_h": latent_h,
                    "latent_w": latent_w,
                    "ref_audio_t": 0,
                }
                if kind == "video_audio":
                    out = encode_reference_audio_rows(
                        self.pipeline.component("audio_vae"),
                        material["path"],
                        latents_mean=a_stats[0],
                        latents_std=a_stats[1],
                        material_chain="video_audio.reference_preserve",
                        start_time_seconds=material["start_time_seconds"],
                    )
                    audio.append(out["rows"])
                    block["ref_audio_t"] = out["ref_audio_t"]
                blocks.append(block)

        batch.set("ref_blocks", blocks)
        batch.set(
            "cond_rows",
            (
                self._noise_augment(
                    torch.cat(visual, dim=0).cpu(), shapes, g.latent_t, batch, config
                )
                if visual
                else None
            ),
        )
        # Audio references are *not* noise-augmented -- their coefficient is
        # 1.0, unlike the visual 0.999. Different constants, on purpose.
        batch.set("cond_audio_rows", torch.cat(audio, dim=0).cpu() if audio else None)
        return batch

    @staticmethod
    def _noise_augment(clean, shapes, latent_t, batch, config):
        """Mix conditioning rows with seeded noise at the DiT's own timestep.

        Skipping this leaves plausible anchors the model treats as cleaner than
        it is told they are.
        """
        seed = batch.job.seed if batch.job.seed is not None else config.seed
        return imgvid_cond_noise_aug_rows(
            clean,
            condition_shapes=shapes,
            target_latent_t=latent_t,
            seed=DEFAULT_NOISE_SEED if seed is None else int(seed),
            noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        )


class LatentPreparationStage(PipelineStage):
    parallelism = StageParallelism.REPLICATED
    requires = ("geometry",)
    produces = ("video_rows", "audio_rows")

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        g = batch.require("geometry")
        video, audio = build_initial_latents(
            latent_t=g.latent_t,
            latent_h=g.latent_h,
            latent_w=g.latent_w,
            audio_t=g.audio_t,
            seed=batch.job.seed if batch.job.seed is not None else config.seed,
        )
        batch.set("video_rows", video)
        batch.set("audio_rows", audio)
        return batch


class PackedSequenceStage(PipelineStage):
    parallelism = StageParallelism.REPLICATED
    requires = ("geometry", "keyframe_indices", "ref_blocks")
    produces = ("packed",)

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        g = batch.require("geometry")
        keyframe_indices = batch.get("keyframe_indices") or None
        ref_blocks = batch.get("ref_blocks")
        # Conditioning material occupies real positions in the Qwen sequence
        # and those are tagged VIDEO, not TEXT; the encoder is the only thing
        # that knows where they landed.
        text_token_tags = batch.get("text_token_tags")
        if ref_blocks:
            packed = build_packed_sequence_ref2va(
                text_len=g.text_len,
                latent_t=g.latent_t,
                latent_h=g.latent_h,
                latent_w=g.latent_w,
                audio_t=g.audio_t,
                ref_blocks=ref_blocks,
                text_token_tags=text_token_tags,
            )
        else:
            packed = build_packed_sequence(
                text_len=g.text_len,
                latent_t=g.latent_t,
                latent_h=g.latent_h,
                latent_w=g.latent_w,
                audio_t=g.audio_t,
                keyframe_frame_indices=keyframe_indices,
                frame_count=g.frame_count if keyframe_indices else None,
                text_token_tags=text_token_tags,
            )
        # Geometry's seq_len predates the conditioning block, so for fl2va the
        # authoritative Ulysses check is here, against the real packed length
        # (39,808 with one keyframe vs 37,760 without).
        world = int(batch.meta["ulysses_world"])
        if int(packed["seq_len"]) % world:
            raise ValueError(
                f"packed sequence {packed['seq_len']} (including "
                f"{packed['cond_rows']} conditioning rows) does not divide "
                f"across ulysses world size {world}"
            )
        for name, key in (("visual", "cond_rows"), ("audio", "cond_audio_rows")):
            rows = batch.get(key)
            expected = int(packed.get(key, 0) or 0)
            got = 0 if rows is None else int(rows.shape[0])
            if got != expected:
                raise ValueError(
                    f"condition encode produced {got} {name} conditioning rows "
                    f"but the packed layout reserves {expected}"
                )
        batch.set("packed", packed)
        return batch


def _progress_reporter(batch: DiffusionBatch):
    """Record denoise progress on the job and forward it to any listener.

    The job field is what an in-process caller polls; the listener is how the
    serving worker pushes progress to the API process without the denoise loop
    knowing a socket exists.
    """
    sink = batch.meta.get("on_progress")

    def report(step: int, total: int) -> None:
        batch.job.current_step = step
        if sink is not None:
            sink(step, total)

    return report


class DenoiseStage(PipelineStage):
    """The hot loop: 49 collectively-parallel DiT steps."""

    parallelism = StageParallelism.REPLICATED
    requires = ("packed", "video_rows", "audio_rows", "video_sigmas", "prompt_embeds")

    produces = ("denoised_video", "denoised_audio")

    def __init__(self, pipeline: "ComposedPipeline", ulysses) -> None:
        self.pipeline = pipeline
        self.ulysses = ulysses

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        dit = self.pipeline.component("transformer")
        packed = batch.require("packed")
        device = batch.meta.get("device", "cpu")
        world = self.ulysses.world_size
        local = int(packed["seq_len"]) // world
        row_start = self.ulysses.rank * local

        rope_cache = dit.build_rope_cache(
            packed["img_position_ids"].unsqueeze(0).to(device),
            row_start,
            row_start + local,
        )
        # Once, not per step: the refiner is request-static. It also reconciles
        # widths -- the encoder emits text_dim (5120), the loop wants 5376.
        raw_embeds = batch.require("prompt_embeds").to(device)
        text_len = int(raw_embeds.shape[0])
        refiner_cu = torch.tensor([0, text_len], dtype=torch.int32, device=device)
        prompt_embeds = dit.refine_prompt_embeds(
            raw_embeds, refiner_cu, device=torch.device(device)
        )

        video, audio = run_denoise_loop(
            dit=dit,
            video_rows=batch.require("video_rows"),
            audio_rows=batch.require("audio_rows"),
            packed=packed,
            cond_rows=batch.get("cond_rows"),
            cond_audio_rows=batch.get("cond_audio_rows"),
            video_sigmas=batch.require("video_sigmas"),
            audio_sigmas=batch.require("audio_sigmas"),
            rank_slice=(row_start, row_start + local),
            device=device,
            prompt_embeds=prompt_embeds,
            refined_prompt_embeds_length=text_len,
            rope_cache=rope_cache,
            scheduler=MiniMaxH3EulerAncestralEta0Scheduler(),
            progress=_progress_reporter(batch),
        )
        batch.set("denoised_video", video)
        batch.set("denoised_audio", audio)
        return batch


class DecodeStage(PipelineStage):
    """VAE decode on rank 0 only; the result is written, not consumed."""

    parallelism = StageParallelism.MAIN_RANK_ONLY
    requires = ("denoised_video", "denoised_audio", "geometry")
    produces = ("frames",)

    def __init__(self, pipeline: "ComposedPipeline") -> None:
        self.pipeline = pipeline

    @property
    def video_stats(self):
        return self.pipeline.video_stats

    @property
    def audio_stats(self):
        return self.pipeline.audio_stats

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        g = batch.require("geometry")
        mean, std = self.video_stats
        frames = decode_video_rows(
            self.pipeline.component("video_vae"),
            batch.require("denoised_video"),
            latent_t=g.latent_t,
            latent_h=g.latent_h,
            latent_w=g.latent_w,
            height=g.height,
            width=g.width,
            mean=mean,
            std=std,
        )
        batch.set("frames", frames)
        audio_vae = self.pipeline.components.get("audio_vae")
        if audio_vae is not None:
            a_mean, a_std = self.audio_stats or (None, None)
            batch.set(
                "audio",
                decode_audio_rows(
                    audio_vae,
                    batch.require("denoised_audio"),
                    mean=a_mean,
                    std=a_std,
                ),
            )
        return batch


class PresentationStage(PipelineStage):
    parallelism = StageParallelism.MAIN_RANK_ONLY
    requires = ("frames",)

    def forward(self, batch: DiffusionBatch, config: DiffusionConfig):
        os.makedirs(config.output_dir, exist_ok=True)
        path = os.path.join(config.output_dir, f"{batch.job.job_id}.mp4")
        write_video_with_audio(path, batch.require("frames"), batch.get("audio"))
        batch.job.output_path = path
        return batch


class MiniMaxH3Pipeline(ComposedPipeline):
    """t2va pipeline for MiniMax-H3."""

    pipeline_name = "MiniMaxH3Pipeline"
    required_components = ("transformer", "video_vae", "text_encoder")
    # ~67 GB, used once per request. Co-resident with the DiT it overflows a
    # 192 GB card during denoise; measured, not predicted.
    host_staged_components = ("text_encoder",)

    # Identity defaults so construction never touches the filesystem; the real
    # values load in load_components(), where a wrong path fails loudly.
    video_stats: tuple[list[float], list[float]] = ([0.0] * 24, [1.0] * 24)
    audio_stats: tuple[list[float], list[float]] | None = None

    def load_components(self) -> None:
        """Load the four networks from the checkpoint root.

        None is a plain state-dict load: the DiT needs the grouped-QKV reorder,
        the VAEs come from the checkpoint's own ``auto_map`` classes, and the
        encoder is Qwen3-VL truncated one layer past the one it reads. Encoder
        and VAEs are main-rank only -- a copy per rank is ~70 GB of waste.
        """
        import torch

        from atom.diffusion.models.minimax_h3.arch import MiniMaxH3DiTArchConfig
        from atom.diffusion.models.minimax_h3.dit import MiniMaxH3DiTModel
        from atom.diffusion.models.minimax_h3.loader import load_minimax_h3_dit_weights
        from atom.diffusion.models.minimax_h3.text_encoder import MiniMaxH3TextEncoder
        from atom.diffusion.models.minimax_h3.vae import (
            VIDEO_VAE_DECODE_DTYPE,
            load_checkpoint_vae,
        )

        root = self.model_root or self.config.model_path
        if not root:
            raise ValueError("MiniMaxH3Pipeline needs a model root to load from")
        # Required: decode de-normalises latents with these, and the identity
        # default would produce structurally valid, quietly wrong pixels.
        video_stats = latent_stats(os.path.join(root, "video_vae"))
        if video_stats is None:
            raise ValueError(
                f"video VAE config at {root} declares no latents_mean/std; "
                "decode cannot de-normalise without them"
            )
        self.video_stats = video_stats
        self.audio_stats = latent_stats(os.path.join(root, "audio_vae"))
        device = (
            torch.device(f"cuda:{self.ulysses.rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        dit = (
            MiniMaxH3DiTModel(MiniMaxH3DiTArchConfig(), self.ulysses).to(device).eval()
        )
        load_minimax_h3_dit_weights(
            dit, os.path.join(root, "transformer"), device=device
        )
        self.register_component("transformer", dit)

        if not self.ulysses.is_main:
            return

        # bf16 for the video VAE: it is transformer-based, so decode is GEMM
        # bound, and fp32 costs 88 s against bf16's 24 s for a 51 dB difference.
        self.register_component(
            "video_vae",
            load_checkpoint_vae(
                os.path.join(root, "video_vae"),
                device=device,
                dtype=VIDEO_VAE_DECODE_DTYPE,
            ),
        )
        self.register_component(
            "audio_vae",
            load_checkpoint_vae(os.path.join(root, "audio_vae"), device=device),
        )
        # Staged on the host: it is used once per request and is the largest
        # single component, so co-residency with the DiT is what overflows the
        # card. TextEncodingStage swaps it in for the encode.
        self.register_component(
            "text_encoder",
            MiniMaxH3TextEncoder.from_pretrained(
                os.path.join(root, "text_encoder"),
                device="cpu",
                dtype=torch.bfloat16,
            ),
        )
        self.encode_device = device

    def verify_components(self) -> None:
        """Only the main rank holds the encoder and VAEs, so only it is checked."""
        if self.ulysses.is_main:
            super().verify_components()
        elif "transformer" not in self.components:
            raise RuntimeError(
                f"{self.pipeline_name} is missing required components: "
                "['transformer']"
            )

    def build_stages(self):
        # Stages resolve components and stats at forward time: build_stages runs
        # inside __init__, before either exists.
        return [
            TextEncodingStage(self),
            PlanStage(),
            ConditionEncodeStage(self),
            LatentPreparationStage(),
            PackedSequenceStage(),
            DenoiseStage(self, self.ulysses),
            DecodeStage(self),
            PresentationStage(),
        ]
