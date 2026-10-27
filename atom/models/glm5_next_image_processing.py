# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build GLM-5 image inputs with its native chat template placeholders."""

from typing import Any


def build_glm5_image_inputs(
    atom_config: Any,
    processor: Any,
    prompt: str | list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
) -> tuple[list[int], dict]:
    """Prepare image embeddings while preserving GLM's image token format."""
    if isinstance(prompt, str):
        text = prompt
    else:
        messages = []
        for message in prompt:
            content = message.get("content")
            if isinstance(content, list):
                image_parts = [part for part in content if part["type"] == "image"]
                text_parts = [
                    part["text"] for part in content if part["type"] == "text"
                ]
                content = image_parts
                if text_parts:
                    content.append({"type": "text", "text": "\n".join(text_parts)})
            messages.append({**message, "content": content})

        template_kwargs = dict(chat_template_kwargs)
        template_kwargs.pop("tokenize", None)
        template_kwargs.pop("add_generation_prompt", None)
        if tools is not None:
            template_kwargs["tools"] = tools
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    num_placeholders = text.count("<|image|>") + text.count("<|image_pad|>")
    if num_placeholders != len(images):
        raise ValueError(
            f"Prompt has {num_placeholders} image placeholders but {len(images)} images"
        )

    processed = processor(text=[text], images=images, return_tensors="pt")
    return processed["input_ids"][0].tolist(), {
        "pixel_values": processed["pixel_values"],
        "image_grid_thw": processed["image_grid_thw"],
    }
