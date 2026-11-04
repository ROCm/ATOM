# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 image and video request inputs: the prompt with its media tokens
expanded and the processor outputs the vision tower takes. Host-side only, so
building a request does not import the vision model or AITER."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from atom.model_engine.multimodal_runtime import multimodal_cache_seed
from atom.multimodal.processing import expand_media_placeholders

if TYPE_CHECKING:
    import numpy as np

    from atom.config import Config


def as_pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected an int or a length-2 sequence, got {value!r}")


def kimi_k3_tokens_per_image(grid_thws, merge_kernel_size) -> list[int]:
    """Image-token count per grid after the ``sd2_tpool`` merge.

    The merge pools the temporal axis away and downsamples each spatial axis by
    the merge kernel, so a ``(t, h, w)`` patch grid yields ``(h // kh) * (w //
    kw)`` tokens regardless of ``t``.
    """
    kernel_h, kernel_w = as_pair(merge_kernel_size)
    grids = grid_thws.tolist() if hasattr(grid_thws, "tolist") else grid_thws
    return [(int(h) // kernel_h) * (int(w) // kernel_w) for _, h, w in grids]


def build_kimi_k3_inputs(
    atom_config: "Config",
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
    videos: list | None = None,
) -> tuple[list[int], dict]:
    """Build Kimi-K3 inputs via ``KimiK3Processor``.

    The K3 processor takes messages plus a separate ``medias`` list (its chat
    encoder is Python, not Jinja), returns ``grid_thws`` rather than
    ``image_grid_thw``, and emits a single ``<|media_pad|>`` per image that the
    reference model expands while merging embeddings. Normalize all three so the
    engine sees the same contract as every other multimodal model.

    Each expanded run becomes one embedding span, so an image prefill can be
    chunked like text and its embeddings are encoded once per request (see
    ``multimodal_runtime``); without spans a prompt longer than
    ``--max-num-batched-tokens`` could never be scheduled.
    """
    multimodal_config = getattr(atom_config, "multimodal_config", None)
    if multimodal_config is None:
        raise ValueError(
            "Kimi-K3 image requests need the full HF config; start the server "
            "with --trust-remote-code."
        )

    template_kwargs = dict(chat_template_kwargs)
    template_kwargs.pop("tokenize", None)
    if tools:
        template_kwargs["tools"] = tools

    if videos:
        return _build_kimi_k3_visual_inputs(
            multimodal_config, processor, messages, template_kwargs
        )

    medias = [{"type": "image", "image": image} for image in images]
    inputs = processor(
        messages=messages,
        medias=medias,
        return_tensors="pt",
        **template_kwargs,
    )

    grid_thws = inputs["grid_thws"]
    input_ids = inputs["input_ids"][0].tolist()
    placeholder_token_id = int(
        getattr(multimodal_config, "media_placeholder_token_id", 163605)
    )
    tokens_per_image = kimi_k3_tokens_per_image(
        grid_thws, multimodal_config.vision_config.merge_kernel_size
    )
    input_ids, spans = expand_media_placeholders(
        input_ids, tokens_per_image, placeholder_token_id
    )

    multimodal_data = {
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": grid_thws,
        "embedding_spans": spans,
    }
    multimodal_data["cache_seed"] = multimodal_cache_seed(multimodal_data)
    return input_ids, multimodal_data


build_kimi_k3_inputs.supports_video = True


def _format_k3_timestamp(seconds: float, mode: str) -> str:
    total = max(0, int(seconds))
    millis = int((max(0.0, seconds) % 1) * 1000)
    hours, minutes, secs = (total // 3600) % 24, (total // 60) % 60, total % 60
    if mode == "hh:mm:ss.fff":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    if mode == "mm:ss.fff":
        return f"{minutes:02d}:{secs:02d}.{millis:03d}"
    if mode == "mm:ss":
        return f"{minutes:02d}:{secs:02d}"
    raise ValueError(f"Invalid Kimi-K3 timestamp mode: {mode}")


def _k3_video_chunks(video, cfg: dict) -> list[tuple[str, "np.ndarray", int]]:
    """Sample a video and split it into MoonViT temporal chunks.

    Follows the video settings in the Kimi-K3 processor config (the same
    contract as the open sglang PR #35325): sample at most ``sample_fps``, cap each frame's patches so the
    whole clip stays within ``in_patch_limit_video``, group
    ``temporal_merge_kernel_size`` frames per grid (the tower mean-pools them),
    and label each grid with the timestamp of its first frame. Returns
    ``(timestamp_text, frames THWC uint8, per_frame_patch_limit)`` per chunk.
    """
    from atom.multimodal import VideoData
    from atom.multimodal.media.video import decode_video, sample_frame_indices

    sample_fps = float(cfg.get("sample_fps", 8.0))
    max_frames = cfg.get("max_num_frames_each_video")
    if isinstance(video, VideoData):
        if video.timestamps is not None:
            source_times = list(video.timestamps)
        else:
            source_times = [i / video.fps for i in range(len(video.frames))]
        if video.fps is not None:
            avg_fps = float(video.fps)
        elif len(source_times) > 1 and source_times[-1] > source_times[0]:
            avg_fps = (len(source_times) - 1) / (source_times[-1] - source_times[0])
        else:
            avg_fps = sample_fps
        indices = sample_frame_indices(
            len(video.frames), avg_fps, sample_fps, max_frames
        )
        frames = video.frames[indices]
        times = [source_times[i] for i in indices]
    else:
        decoded = decode_video(video, sample_fps=sample_fps, max_frames=max_frames)
        frames, times = decoded.frames, list(decoded.timestamps)

    count = len(frames)
    merge = int(cfg.get("temporal_merge_kernel_size", 4))
    if not 1 <= merge <= 4:
        raise ValueError("Kimi-K3 temporal_merge_kernel_size must be between 1 and 4")
    limit = int(cfg.get("in_patch_limit_each_frame") or cfg["in_patch_limit"])
    if cfg.get("in_patch_limit_video") is not None:
        limit = min(limit, max(1, round(int(cfg["in_patch_limit_video"]) / count)))
    mode = cfg.get("timestamp_mode", "hh:mm:ss.fff")
    return [
        (_format_k3_timestamp(times[start], mode), frames[start : start + merge], limit)
        for start in range(0, count, merge)
    ]


def _build_kimi_k3_visual_inputs(
    multimodal_config, processor: Any, messages: list[dict], template_kwargs: dict
) -> tuple[list[int], dict]:
    """Images and videos in one prompt, in source order.

    The model's own processor is image-only, so this keeps its image path
    (same preprocessing, same prompt) and adds video grids beside it. K3's
    chat encoder only knows image parts; each source item renders as one
    image part whose prompt is either the image prompt or the video's chunk
    prompts, and every grid ends in exactly one ``<|media_pad|>``.
    """
    import sys

    import numpy as np
    import torch
    from PIL import Image

    media_processor = processor.media_processor
    utils = sys.modules[type(media_processor).__module__]
    cfg = media_processor.media_proc_cfg
    mean = np.array(cfg["image_mean"])
    std_inv = 1.0 / np.array(cfg["image_std"])

    prompts: list[str] = []
    pixel_values: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    rendered = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rendered.append(message)
            continue
        parts = []
        for part in content:
            if part["type"] == "image":
                media = utils.ensure_media_type(
                    {"type": "image", "image": part["image"]},
                    transparent_bg_config=media_processor._transparent_bg_config,
                    transparent_bg_fill_stage=media_processor._transparent_bg_fill_stage,
                )
                width, height = media["image"].size
                prompts.append(media_processor.make_image_prompt(width, height))
                features = media_processor.preprocess([media], return_tensors="pt")
                pixel_values.append(features["pixel_values"])
                grids.append(features["grid_thws"])
                parts.append({"type": "image", "image": media["image"]})
            elif part["type"] == "video":
                chunk_prompts = []
                for stamp, frames, limit in _k3_video_chunks(part["video"], cfg):
                    height, width = frames.shape[1:3]
                    resize = utils.navit_resize_image(
                        width,
                        height,
                        cfg["patch_size"],
                        cfg["merge_kernel_size"],
                        limit,
                        cfg["patch_limit_on_one_side"],
                        cfg["fixed_output_tokens"],
                    )
                    stack = np.stack(
                        [
                            media_processor.resize_image(
                                Image.fromarray(frame),
                                resize["new_width"],
                                resize["new_height"],
                                resize["pad_width"],
                                resize["pad_height"],
                            )
                            for frame in frames
                        ]
                    )
                    patched = utils.navit_patchify(
                        utils.normalize(stack, mean, std_inv), cfg["patch_size"]
                    )
                    pixel_values.append(utils._to_tensor(patched["pixel_values"]))
                    grids.append(
                        utils._to_tensor(
                            patched["grid_thw"], dtype=torch.int64
                        ).unsqueeze(0)
                    )
                    chunk_prompts.append(
                        f"{stamp}<|media_begin|>video<|media_content|>"
                        "<|media_pad|><|media_end|>"
                    )
                prompts.append("".join(chunk_prompts))
                parts.append({"type": "image", "image": None})
            else:
                parts.append(part)
        rendered.append({**message, "content": parts})

    text_inputs = processor.tokenizer.apply_chat_template(
        rendered,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        image_prompts=prompts,
        **template_kwargs,
    )
    grid_thws = torch.cat(grids)
    input_ids = text_inputs["input_ids"][0].tolist()
    placeholder_token_id = int(
        getattr(multimodal_config, "media_placeholder_token_id", 163605)
    )
    input_ids, spans = expand_media_placeholders(
        input_ids,
        kimi_k3_tokens_per_image(
            grid_thws, multimodal_config.vision_config.merge_kernel_size
        ),
        placeholder_token_id,
    )
    multimodal_data = {
        "pixel_values": torch.cat(pixel_values),
        "image_grid_thw": grid_thws,
        "embedding_spans": spans,
    }
    multimodal_data["cache_seed"] = multimodal_cache_seed(multimodal_data)
    return input_ids, multimodal_data
