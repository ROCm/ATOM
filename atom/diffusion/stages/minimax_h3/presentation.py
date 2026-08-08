# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Presentation format follows the reference in sgl-project/sglang
# (.../minimax_h3/presentation.py, Apache-2.0).

"""Build MiniMax-H3's Qwen presentation token stream.

H3 does **not** use a chat template. It builds the conditioning sequence
explicitly:

    t2va   prompt                                       (verbatim, no specials)
    fl2va  "<Picture 1>: " + vision-block + [ "<Picture 2>: " + block ] + prompt
    ref2va per-condition labels ("<Picture i>: " / "<Audio j>: ") + prompt

A vision block is ``<|vision_start|>`` + N x pad token + ``<|vision_end|>``.
Everything in a vision block is tagged VIDEO(0); everything else TEXT(1), and
those tags become the DiT's AdaLN modality gather for the text region.

Using ``apply_chat_template`` instead produces a *plausible* sequence with the
right image span but the wrong wrapper -- measured 3/1010/18 tokens against the
reference's 6/1010/13 -- so ids and tags must be built here, together, to stay
aligned.
"""

from collections.abc import Sequence
from typing import Any

import torch

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

TAG_TEXT = 1
TAG_VIDEO = 0


def text_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize without special tokens -- the presentation supplies its own."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def vision_block_ids(tokenizer: Any, pad_token: str, count: int) -> list[int]:
    if int(count) <= 0:
        raise ValueError(f"vision block needs a positive token count, got {count}")
    return (
        [tokenizer.convert_tokens_to_ids(VISION_START)]
        + [tokenizer.convert_tokens_to_ids(pad_token)] * int(count)
        + [tokenizer.convert_tokens_to_ids(VISION_END)]
    )


class Presentation:
    """Accumulates ids and modality tags together so they cannot drift."""

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.tags: list[int] = []

    def text(self, token_ids: list[int]) -> "Presentation":
        self.ids += token_ids
        self.tags += [TAG_TEXT] * len(token_ids)
        return self

    def vision(self, token_ids: list[int]) -> "Presentation":
        self.ids += token_ids
        self.tags += [TAG_VIDEO] * len(token_ids)
        return self

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.ids, dtype=torch.long),
            torch.tensor(self.tags, dtype=torch.long),
        )


def text_only_presentation(
    tokenizer: Any, *, prompt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """t2va: the prompt verbatim, no special tokens, all TEXT."""
    if not prompt:
        raise ValueError("prompt must be non-empty")
    return Presentation().text(text_ids(tokenizer, prompt)).build()


def multi_image_presentation(
    tokenizer: Any, *, prompt: str, image_token_counts: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """fl2va: one ``<Picture i>: `` label + vision block per keyframe."""
    counts = [int(c) for c in image_token_counts]
    if not counts:
        raise ValueError("image_token_counts must be non-empty")
    p = Presentation()
    for index, count in enumerate(counts, start=1):
        p.text(text_ids(tokenizer, f"<Picture {index}>: "))
        p.vision(vision_block_ids(tokenizer, IMAGE_PAD, count))
    p.text(text_ids(tokenizer, prompt))
    return p.build()


def ref2va_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    condition_labels: Sequence[tuple[str, int]],
    image_token_counts: Sequence[int] | None = None,
    video_block_token_counts: Sequence[Sequence[int]] | None = None,
    video_block_timestamps: Sequence[Sequence[float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ref2va: per-condition labels in request order, then the prompt.

    ``condition_labels`` is ``[("image", 1), ("audio", 1), ("video", 1), ...]``
    with 1-based ordinals per type. Audio contributes a **label only** -- audio
    content never enters Qwen. Video contributes one timestamped block per
    temporal chunk.
    """
    images = list(image_token_counts or [])
    vid_counts = [list(c) for c in (video_block_token_counts or [])]
    vid_stamps = [list(t) for t in (video_block_timestamps or [])]

    p = Presentation()
    img_i = vid_i = 0
    for kind, ordinal in condition_labels:
        if kind == "image":
            if img_i >= len(images):
                raise ValueError("more image conditions than image_token_counts")
            p.text(text_ids(tokenizer, f"<Picture {ordinal}>: "))
            p.vision(vision_block_ids(tokenizer, IMAGE_PAD, images[img_i]))
            img_i += 1
        elif kind == "audio":
            p.text(text_ids(tokenizer, f"<Audio {ordinal}>: "))
        elif kind == "video":
            if vid_i >= len(vid_counts) or vid_i >= len(vid_stamps):
                raise ValueError("video condition without block counts/timestamps")
            counts, stamps = vid_counts[vid_i], vid_stamps[vid_i]
            if not counts or len(counts) != len(stamps):
                raise ValueError("video block counts and timestamps must align")
            for count, stamp in zip(counts, stamps):
                p.text(text_ids(tokenizer, f"<{stamp:.1f} seconds>"))
                p.vision(vision_block_ids(tokenizer, VIDEO_PAD, count))
            vid_i += 1
        else:
            raise ValueError(f"unknown condition kind {kind!r}")
    p.text(text_ids(tokenizer, prompt))
    return p.build()
