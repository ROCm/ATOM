# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 text conditioning: Qwen3-VL truncated at layer 50.

H3 consumes ``hidden_states[50]`` -- the *unnormalized* state after LM layer 49
-- as a ``[T, 5120]`` sequence, not a pooled vector.

The trap: transformers appends the *post-norm* activation as the last entry of
``hidden_states``, so truncating to exactly 50 layers makes index 50 that
normalised entry (measured cos 0.78 against the reference). Load **51** layers
to keep index 50 an intermediate -- one extra layer instead of fourteen.
"""

import contextlib
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Output after layer 49 == hidden_states[50].
MINIMAX_H3_SELECTED_LM_LAYER = 50
MINIMAX_H3_TEXT_DIM = 5120


class MiniMaxH3TextEncoder:
    """Wraps Qwen3-VL and returns H3's ``[T, 5120]`` conditioning rows."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | str,
        processor: Any | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        num_layers: int = MINIMAX_H3_SELECTED_LM_LAYER,
        attn_implementation: str | None = None,
    ) -> "MiniMaxH3TextEncoder":
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        # Keep one layer beyond the one we read: the selected state must be an
        # *intermediate* entry of hidden_states, because transformers puts the
        # post-final-norm activation in the last slot.
        keep_layers = num_layers + 1
        if hasattr(config, "text_config"):
            config.text_config.num_hidden_layers = keep_layers
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = keep_layers

        try:
            from transformers import Qwen3VLForConditionalGeneration as _Cls
        except ImportError as exc:  # pragma: no cover - depends on transformers
            raise ImportError(
                "transformers is too old for Qwen3-VL; MiniMax-H3 text "
                "conditioning needs Qwen3VLForConditionalGeneration"
            ) from exc

        kwargs: dict[str, Any] = {}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = _Cls.from_pretrained(
            path, config=config, torch_dtype=dtype, trust_remote_code=True, **kwargs
        )
        model = model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001 - text-only still works
            logger.warning("no AutoProcessor (%s); fl2va images unavailable", exc)
            processor = None
        logger.info(
            "loaded MiniMax-H3 text encoder from %s (%d layers kept, reading "
            "hidden_states[%d], %s)",
            path,
            keep_layers,
            num_layers,
            dtype,
        )
        return cls(model, tokenizer, device=device, processor=processor)

    @torch.no_grad()
    def encode(self, prompt: str, images: Any | list | None = None) -> torch.Tensor:
        """Prompt (optionally with keyframes) -> ``[T, 5120]`` rows.

        For fl2va the keyframe also goes through Qwen3-VL's vision tower, so it
        occupies real positions in the conditioning sequence and must be tagged
        VIDEO downstream; see :meth:`encode_with_tags`.
        """
        rows, _ = self.encode_with_tags(prompt, images)
        return rows

    def to(self, device: torch.device | str) -> "MiniMaxH3TextEncoder":
        """Move the encoder and remember where it is."""
        self.model = self.model.to(device)
        self.device = torch.device(device)
        return self

    @contextlib.contextmanager
    def resident_on(self, device: torch.device | str):
        """Hold the encoder on ``device`` for one encode, then send it back.

        H3 encodes once per request and then never touches the encoder again,
        but at ~67 GB it is the single largest resident tensor on the main
        rank. Leaving it on the GPU alongside the DiT overflows a 192 GB card
        during denoise -- measured, not predicted: the first served request
        died with 182 GiB already allocated.

        The round trip costs a couple of seconds against a multi-minute
        generation, and it is what makes a 4-GPU replica fit at all.
        """
        origin = self.device
        try:
            yield self.to(device)
        finally:
            self.to(origin)
            if torch.cuda.is_available():
                # Return the block to the allocator, not just to this module:
                # otherwise the DiT's next allocation still sees a full cache.
                torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a prebuilt presentation through Qwen3-VL -> ``[T, 5120]`` rows.

        The caller owns the token stream (see
        :mod:`atom.diffusion.models.minimax_h3.presentation`), which is the
        whole point: fl2va and ref2va differ only in how the stream is built.
        """
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be given together"
            )

        ids = input_ids.to(device=self.device, dtype=torch.long)
        batch: dict[str, Any] = {"input_ids": ids.unsqueeze(0)}
        if pixel_values is not None or pixel_values_videos is not None:
            # Hand-built input_ids means no mm_token_type_ids from the
            # processor, and Qwen3-VL raises rather than guess. Mark exactly the
            # pad positions -- vision_start/end are ordinary text tokens.
            from atom.diffusion.models.minimax_h3.presentation import (
                IMAGE_PAD,
                VIDEO_PAD,
            )

            mm = torch.zeros_like(ids)
            for token in (IMAGE_PAD, VIDEO_PAD):
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id is not None:
                    mm |= (ids == token_id).to(torch.long)
            batch["mm_token_type_ids"] = mm.unsqueeze(0)
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            batch["image_grid_thw"] = image_grid_thw.to(self.device, torch.long)
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos.to(
                self.device, torch.bfloat16
            )
            batch["video_grid_thw"] = video_grid_thw.to(self.device, torch.long)

        out = self.model(**batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states
        if len(hidden) <= MINIMAX_H3_SELECTED_LM_LAYER:
            raise ValueError(
                f"encoder returned {len(hidden)} hidden states; need at least "
                f"{MINIMAX_H3_SELECTED_LM_LAYER + 1} to select layer "
                f"{MINIMAX_H3_SELECTED_LM_LAYER}"
            )
        rows = hidden[MINIMAX_H3_SELECTED_LM_LAYER][0]
        if int(rows.shape[-1]) != MINIMAX_H3_TEXT_DIM:
            raise ValueError(
                f"text embeddings are {int(rows.shape[-1])} wide, expected "
                f"{MINIMAX_H3_TEXT_DIM}"
            )
        return rows

    def image_token_counts(self, images: list) -> tuple[dict, list[int]]:
        """Preprocess images and report Qwen's per-image token count."""
        if self.processor is None:
            raise RuntimeError(
                "images supplied but no processor is loaded; conditioning on "
                "images needs AutoProcessor for the vision tower"
            )
        vision = self.processor.image_processor(images=images, return_tensors="pt")
        grid = vision["image_grid_thw"]
        merge = int(self.processor.image_processor.merge_size) ** 2
        counts = [int(grid[i].prod().item()) // merge for i in range(len(images))]
        return vision, counts

    @torch.no_grad()
    def encode_with_tags(
        self, prompt: str, images: Any | list | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rows [T, 5120], token_tags [T])`` for t2va / fl2va.

        Tags mark vision blocks as VIDEO so the DiT's AdaLN gather treats those
        positions as image, not text.
        """
        from atom.diffusion.models.minimax_h3.presentation import (
            multi_image_presentation,
            text_only_presentation,
        )

        if not prompt:
            raise ValueError("prompt must be non-empty")

        if images is None:
            ids, tags = text_only_presentation(self.tokenizer, prompt=prompt)
            rows = self.encode_ids(ids)
        else:
            if not isinstance(images, list):
                images = [images]
            vision, counts = self.image_token_counts(images)
            ids, tags = multi_image_presentation(
                self.tokenizer, prompt=prompt, image_token_counts=counts
            )
            rows = self.encode_ids(
                ids,
                pixel_values=vision["pixel_values"],
                image_grid_thw=vision["image_grid_thw"],
            )

        if int(tags.numel()) != int(rows.shape[0]):
            raise ValueError(
                f"presentation produced {int(tags.numel())} tags for "
                f"{int(rows.shape[0])} rows"
            )
        return rows, tags
