# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Image input for MiniMax-M3.1: what the API server builds for an image request.

Kept apart from the vision stack (``vl``) so the server imports neither the
model nor AITER to render a prompt.
"""

from typing import TYPE_CHECKING, Any

from atom.model_engine.multimodal_runtime import multimodal_cache_seed

if TYPE_CHECKING:
    from atom.config import Config

IMAGE_TOKEN = "]<]image[>["
DEFAULT_IMAGE_TOKEN_ID = 200025


def image_token_id(hf_config) -> int:
    """The id of the ``]<]image[>[`` placeholder, as either key names it."""
    return int(
        getattr(hf_config, "image_token_index", None)
        or getattr(hf_config, "image_token_id", None)
        or DEFAULT_IMAGE_TOKEN_ID
    )


def image_input_refusal(atom_config: "Config") -> str | None:
    """Why this deployment takes no image input, or None."""
    if getattr(atom_config.hf_config, "vision_config", None) is None:
        return "the checkpoint has no vision_config (a text-only export)"
    mode = atom_config.compilation_config.cudagraph_mode
    if not atom_config.enforce_eager and mode is not None and mode.uses_tr_hybrid():
        # Image input needs the compiled model to always take inputs_embeds,
        # embedded eagerly before it, and TR_HYBRID replays token graphs
        # recorded on input_ids alone (`_run_capture_prefill`).
        return f"--cudagraph-mode {mode.name} (image input needs FULL or eager)"
    return None


def _image_token_spans(input_ids: list[int], token_id: int) -> list[tuple]:
    spans, start = [], None
    for index, token in enumerate(input_ids + [None]):
        if token == token_id and start is None:
            start = index
        elif token != token_id and start is not None:
            spans.append((start, index - start))
            start = None
    return spans


def build_minimax_m3_inputs(
    atom_config: "Config",
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
) -> tuple[list[int], dict]:
    """Render the chat template, then let ``MiniMaxVLProcessor`` expand images.

    The template emits one ``]<]image[>[`` per image; the processor wraps each
    in start/end-of-image tokens and repeats it once per merged patch
    (``t*h*w // 4``). Each run of image tokens becomes one embedding span, so
    image prefills can be chunked and their embeddings cached per request.
    """
    why = image_input_refusal(atom_config)
    if why is not None:
        raise ValueError(f"This MiniMax-M3 deployment takes no image input: {why}")
    token_id = image_token_id(atom_config.hf_config)
    template_kwargs = dict(chat_template_kwargs)
    template_kwargs.pop("tokenize", None)
    template_kwargs.pop("add_generation_prompt", None)
    if tools:
        template_kwargs["tools"] = tools
    # The checkpoint ships its template for the tokenizer only; the server's
    # --chat-template, when given, is installed on the processor.
    text = processor.apply_chat_template(
        messages,
        chat_template=processor.chat_template or processor.tokenizer.chat_template,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    num_placeholders = text.count(IMAGE_TOKEN)
    if num_placeholders != len(images):
        raise ValueError(
            f"Prompt has {num_placeholders} image placeholders but {len(images)} images"
        )
    # The rendered template already carries its own BOS.
    processed = processor(
        text=[text], images=images, add_special_tokens=False, return_tensors="pt"
    )
    input_ids = processed["input_ids"][0].tolist()
    grid_thw = processed["image_grid_thw"]
    spans = _image_token_spans(input_ids, token_id)
    merge_length = processor.image_processor.merge_size**2
    expected = [int(n) // merge_length for n in grid_thw.prod(-1).tolist()]
    if [count for _, count in spans] != expected:
        raise ValueError(
            f"Image token runs {[c for _, c in spans]} disagree with the "
            f"processed grids {expected}"
        )
    data = {
        "pixel_values": processed["pixel_values"],
        "image_grid_thw": grid_thw,
        "embedding_spans": tuple(spans),
    }
    data["cache_seed"] = multimodal_cache_seed(data)
    return input_ids, data
