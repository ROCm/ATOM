# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 image and video request inputs: the prompt with its media tokens
expanded and the processor outputs the vision tower takes. Host-side only, so
building a request does not import the vision model or AITER."""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from atom.model_engine.multimodal_runtime import multimodal_cache_seed
from atom.multimodal.processing import expand_media_placeholders

if TYPE_CHECKING:
    import numpy as np
    import torch

    from atom.config import Config

# ``media_placeholder_token_id`` of the released checkpoints, for configs that
# omit it.
DEFAULT_MEDIA_PLACEHOLDER_TOKEN_ID = 163605


def as_pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected an int or a length-2 sequence, got {value!r}")


def media_placeholder_token_id(multimodal_config) -> int:
    return int(
        getattr(
            multimodal_config,
            "media_placeholder_token_id",
            DEFAULT_MEDIA_PLACEHOLDER_TOKEN_ID,
        )
    )


def kimi_k3_tokens_per_grid(grid_thws, merge_kernel_size) -> list[int]:
    """Media-token count per image or video grid after the ``sd2_tpool`` merge.

    The merge pools the temporal axis away and downsamples each spatial axis by
    the merge kernel, so a ``(t, h, w)`` patch grid yields ``(h // kh) * (w //
    kw)`` tokens regardless of ``t``.
    """
    kernel_h, kernel_w = as_pair(merge_kernel_size)
    grids = grid_thws.tolist() if hasattr(grid_thws, "tolist") else grid_thws
    return [(int(h) // kernel_h) * (int(w) // kernel_w) for _, h, w in grids]


def kimi_k3_vision_pass_patches(cfg: dict) -> int:
    """Patch budget of one vision-tower pass, from ``media_proc_cfg``.

    The largest grid the config targets: an image of ``in_patch_limit`` patches
    or a video grid of ``temporal_merge_kernel_size`` frames of
    ``in_patch_limit_each_frame``. Padding to the merge kernel can take a grid
    slightly past it; the runtime then encodes that grid in a pass of its own.
    """
    image = int(cfg["in_patch_limit"])
    frame = int(cfg.get("in_patch_limit_each_frame") or image)
    return max(image, int(cfg.get("temporal_merge_kernel_size") or 1) * frame)


def load_kimi_k3_vision_pass_patches(model: str | None) -> int | None:
    """``kimi_k3_vision_pass_patches`` of a checkpoint, or None without one."""
    import json

    from transformers.utils import cached_file

    if not model:
        return None
    try:
        path = cached_file(
            model,
            "preprocessor_config.json",
            _raise_exceptions_for_missing_entries=False,
        )
        if path is None:
            return None
        with open(path, encoding="utf-8") as stream:
            cfg = json.load(stream).get("media_proc_cfg") or {}
        return kimi_k3_vision_pass_patches(cfg)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_kimi_k3_inputs(
    atom_config: "Config",
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
) -> tuple[list[int], dict]:
    """Build Kimi-K3 inputs from a conversation with image and video parts.

    The K3 processor takes messages plus separate media (its chat encoder is
    Python, not Jinja), returns ``grid_thws`` rather than ``image_grid_thw``,
    and emits a single ``<|media_pad|>`` per grid that the reference model
    expands while merging embeddings. Normalize all three so the engine sees
    the same contract as every other multimodal model.

    Every request takes one path. Each image goes through the processor's own
    ``preprocess_medias`` and ``media_processor.preprocess``, in its decoded
    mode, so the processor's ``transparent_bg_config`` fill applies. Each
    video becomes timestamped grids (``_video_grids``). One
    ``apply_chat_template(image_prompts=...)`` then renders the prompt; the
    checkpoint's encoder knows only image parts, so a video renders as one
    image part whose prompt is its grid prompts.

    Image parts take their bound value, or else the next item of ``images``;
    video parts must be bound (``prepare_multimodal_inputs`` binds them).

    Each expanded run becomes one embedding span, so a media prefill can be
    chunked like text and its embeddings are encoded once per request (see
    ``multimodal_runtime``); without spans a prompt longer than
    ``--max-num-batched-tokens`` could never be scheduled.

    Images reach the processor in their decoded mode, so its own
    ``media_proc_cfg.transparent_bg_config`` fill applies to transparent ones.
    """
    import torch

    multimodal_config = getattr(atom_config, "multimodal_config", None)
    if multimodal_config is None:
        raise ValueError(
            "Kimi-K3 media requests need the full HF config; start the server "
            "with --trust-remote-code."
        )

    template_kwargs = dict(chat_template_kwargs)
    template_kwargs.pop("tokenize", None)
    if tools:
        template_kwargs["tools"] = tools

    merge_kernel_size = multimodal_config.vision_config.merge_kernel_size
    max_tokens = getattr(atom_config, "max_model_len", None)
    unbound_images = iter(images)
    image_count = 0
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
                image_count += 1
                image = part.get("image")
                if image is None:
                    image = next(unbound_images, None)
                    if image is None:
                        raise ValueError(
                            f"Kimi-K3 image part {image_count} has no image"
                        )
                medias, image_prompts = processor.preprocess_medias(
                    [{"type": "image", "image": image}]
                )
                features = processor.media_processor.preprocess(
                    medias, return_tensors="pt"
                )
                prompts += image_prompts
                pixel_values.append(features["pixel_values"])
                grids.append(features["grid_thws"])
                parts.append({**part, "image": image})
            elif part["type"] == "video":
                if part.get("video") is None:
                    raise ValueError("Kimi-K3 video part has no video")
                video_prompts = []
                for prompt, values, grid in _video_grids(
                    part["video"],
                    processor.media_processor,
                    merge_kernel_size,
                    max_tokens,
                ):
                    video_prompts.append(prompt)
                    pixel_values.append(values)
                    grids.append(grid)
                prompts.append("".join(video_prompts))
                parts.append({"type": "image", "image": None})
            else:
                parts.append(part)
        rendered.append({**message, "content": parts})
    if image_count != len(images):
        raise ValueError(
            f"Conversation has {image_count} image parts but {len(images)} images"
        )

    text_inputs = processor.tokenizer.apply_chat_template(
        rendered,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        image_prompts=prompts,
        **template_kwargs,
    )
    input_ids = text_inputs["input_ids"][0].tolist()
    if not grids:
        raise ValueError("Kimi-K3 multimodal request has no image or video parts")
    grid_thws = torch.cat(grids)
    input_ids, spans = expand_media_placeholders(
        input_ids,
        kimi_k3_tokens_per_grid(grid_thws, merge_kernel_size),
        media_placeholder_token_id(multimodal_config),
    )
    multimodal_data = {
        "pixel_values": torch.cat(pixel_values),
        # The engine-wide name for the grids, images and video alike.
        "image_grid_thw": grid_thws,
        "embedding_spans": spans,
    }
    multimodal_data["cache_seed"] = multimodal_cache_seed(multimodal_data)
    return input_ids, multimodal_data


def navit_frame_size(width: int, height: int, cfg: dict, patch_limit: int) -> dict:
    """The processor's NaViT resize arithmetic for one frame.

    Scale down (never up) to at most ``patch_limit`` patches and
    ``patch_limit_on_one_side`` patches per side, then pad each side to a
    multiple of ``merge_kernel_size * patch_size``.
    """
    patch = int(cfg["patch_size"])
    side = int(cfg["patch_limit_on_one_side"])
    scale = min(
        1.0,
        math.sqrt(patch_limit / (max(1.0, width // patch) * max(1.0, height // patch))),
        side * patch / width,
        side * patch / height,
    )
    new_width = min(max(1, int(width * scale)), side * patch)
    new_height = min(max(1, int(height * scale)), side * patch)
    factor = int(cfg["merge_kernel_size"]) * patch
    return {
        "new_width": new_width,
        "new_height": new_height,
        "pad_width": (factor - new_width % factor) % factor,
        "pad_height": (factor - new_height % factor) % factor,
    }


def _padded_patches(size: dict, patch_size: int) -> int:
    return ((size["new_width"] + size["pad_width"]) // patch_size) * (
        (size["new_height"] + size["pad_height"]) // patch_size
    )


def video_frame_size(width: int, height: int, cfg: dict, num_frames: int) -> dict:
    """Frame size for a video of ``num_frames`` sampled frames.

    Each frame gets at most ``in_patch_limit_each_frame`` patches, and the
    clip's frames share ``in_patch_limit_video``: the per-frame limit is
    ``in_patch_limit_video / num_frames``, as the Kimi processors'
    ``navit_resize_video`` divides its ``in_patch_limit_total`` by the clip's
    sampled frame count. Padding to the merge kernel can push a frame past
    that limit, so the limit is lowered until the padded clip fits.
    """
    limit = int(cfg.get("in_patch_limit_each_frame") or cfg["in_patch_limit"])
    total = cfg.get("in_patch_limit_video")
    if total is None:
        return navit_frame_size(width, height, cfg, limit)
    total = int(total)
    limit = max(1, min(limit, round(total / num_frames)))
    size = navit_frame_size(width, height, cfg, limit)
    while (
        limit > 1 and _padded_patches(size, int(cfg["patch_size"])) * num_frames > total
    ):
        limit -= max(1, limit // 64)
        size = navit_frame_size(width, height, cfg, limit)
    return size


def _format_timestamp(seconds: float, mode: str) -> str:
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


def _patchify(frames: "np.ndarray", cfg: dict) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Normalize ``(t, h, w, 3)`` uint8 frames and cut them into patches.

    Same arithmetic as the processor's image path: scale to [0, 1] in fp32,
    subtract the mean, multiply by the inverse std, then order patches by
    ``(t, h, w)`` with each patch as ``(c, p, p)``.
    """
    import numpy as np
    import torch

    patch = int(cfg["patch_size"])
    values = (frames / 255.0).astype(np.float32)
    values -= np.array(cfg["image_mean"])
    values *= 1.0 / np.array(cfg["image_std"])
    t, h, w, c = values.shape
    patches = values.reshape(t, h // patch, patch, w // patch, patch, c)
    patches = patches.transpose(0, 1, 3, 5, 2, 4).reshape(-1, c, patch, patch)
    grid = torch.tensor([[t, h // patch, w // patch]], dtype=torch.int64)
    return torch.from_numpy(np.ascontiguousarray(patches)), grid


def _video_grids(
    video, media_processor, merge_kernel_size, max_tokens: int | None
) -> list[tuple[str, "torch.Tensor", "torch.Tensor"]]:
    """Sample a video and cut it into timestamped MoonViT grids.

    Samples at most ``sample_fps`` (and ``max_num_frames_each_video``) frames,
    sizes every frame with ``video_frame_size``, groups
    ``temporal_merge_kernel_size`` of them per grid and labels each grid with
    its first frame's timestamp. The token count is known from the probe, so a video that cannot
    fit the context is refused before any frame is decoded; frames are resized
    as they are decoded. Returns ``(prompt, pixel_values, grid_thw)`` per grid.
    """
    import numpy as np
    from PIL import Image

    from atom.multimodal import VideoData
    from atom.multimodal.media.video import (
        decode_video_frames,
        probe_video,
        sample_frame_indices,
    )

    cfg = media_processor.media_proc_cfg
    sample_fps = float(cfg.get("sample_fps") or 1.0)
    max_frames = cfg.get("max_num_frames_each_video")
    merge = int(cfg.get("temporal_merge_kernel_size") or 1)
    timestamp_mode = cfg.get("timestamp_mode", "hh:mm:ss.fff")
    kernel_h, kernel_w = as_pair(merge_kernel_size)

    if isinstance(video, VideoData):
        num_frames, (height, width) = len(video.frames), video.frames.shape[1:3]
        times = video.timestamps
        if video.fps is not None:
            fps = float(video.fps)
        elif len(times) > 1 and times[-1] > times[0]:
            fps = (len(times) - 1) / (times[-1] - times[0])
        else:
            fps = sample_fps
        if times is None:
            times = [index / fps for index in range(num_frames)]
    else:
        info = probe_video(video.data)
        num_frames, fps, width, height = (
            info.num_frames,
            info.fps,
            info.width,
            info.height,
        )
        times = None

    # A decode can yield fewer frames than probing counted packets; plan again
    # from the decoded count once.
    for _ in range(2):
        indices = sample_frame_indices(num_frames, fps, sample_fps, max_frames)
        size = video_frame_size(width, height, cfg, len(indices))
        grid_tokens = (
            (size["new_height"] + size["pad_height"]) // cfg["patch_size"] // kernel_h
        ) * ((size["new_width"] + size["pad_width"]) // cfg["patch_size"] // kernel_w)
        tokens = grid_tokens * math.ceil(len(indices) / merge)
        if max_tokens is not None and tokens > max_tokens:
            raise ValueError(
                f"Video needs {tokens} tokens ({len(indices)} frames sampled), "
                f"more than the {max_tokens}-token context"
            )

        def resize(_position: int, image, size=size) -> "np.ndarray":
            return media_processor.resize_image(
                image,
                size["new_width"],
                size["new_height"],
                size["pad_width"],
                size["pad_height"],
            )

        if isinstance(video, VideoData):
            frames = [
                resize(position, Image.fromarray(video.frames[index]))
                for position, index in enumerate(indices)
            ]
            break
        frames, decoded = decode_video_frames(video.data, indices, resize)
        if len(frames) == len(indices):
            break
        num_frames = decoded
    else:
        raise ValueError("Video frames changed between decoding passes")

    grids = []
    for start in range(0, len(frames), merge):
        first = indices[start]
        stamp = times[first] if times is not None else first / fps
        values, grid = _patchify(np.stack(frames[start : start + merge]), cfg)
        prompt = (
            f"{_format_timestamp(stamp, timestamp_mode)}<|media_begin|>video"
            "<|media_content|><|media_pad|><|media_end|>"
        )
        grids.append((prompt, values, grid))
    return grids
