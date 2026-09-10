# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Multimodal helpers shared by the OpenAI server and the offline examples.

The model-specific hooks live here:

* :func:`get_mrope_input_positions` — request-level MRoPE positions, for models
  whose language side consumes 3D positions (Qwen3.5).
* :func:`build_multimodal_inputs` — turning chat messages + images into
  ``(input_ids, multimodal_data, placeholders)``. Most Hugging Face processors
  follow the Qwen convention (``processor(text=..., images=...)`` returning
  already expanded image placeholders), served by
  :func:`build_default_multimodal_inputs`; models that deviate register a
  builder below.
* :class:`MediaPlaceholder` — where each media item's tokens landed in the
  prompt, and what image put them there.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xxhash

from atom.config import Config
from atom.utils import resolve_obj_by_qualname

_MULTIMODAL_ARCH_TO_MODEL: dict[str, str] = {
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": (
        "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel"
    ),
}

_MULTIMODAL_ARCH_TO_INPUT_BUILDER: dict[str, str] = {
    "KimiK3ForConditionalGeneration": (
        "atom.model_engine.multimodal.build_kimi_k3_inputs"
    ),
}

# Architecture -> the attribute on the full multimodal config that names the
# token id repeated once per media embedding. Qwen3.5 and Kimi-K3 spell it
# differently, and neither id is readable without the model unless it is looked
# up here (the model reads the same attribute onto an instance field).
_MULTIMODAL_ARCH_TO_PLACEHOLDER_ATTR: dict[str, str] = {
    "Qwen3_5ForConditionalGeneration": "image_token_id",
    "Qwen3_5MoeForConditionalGeneration": "image_token_id",
    "KimiK3ForConditionalGeneration": "media_placeholder_token_id",
}


@dataclass(frozen=True)
class MediaPlaceholder:
    """Where one media item's placeholder tokens sit in a prompt, and what it is.

    ``offset`` and ``length`` index into the prompt token ids *after* the
    processor has expanded the placeholders, so
    ``token_ids[offset : offset + length]`` is exactly this item's run.

    ``identifier`` is a content hash of the source media, and it is the reason
    this type exists: a prompt's placeholder tokens carry the same token id no
    matter which image they stand for, so without the identifier in the block
    hash "same text, different image" collides in the prefix cache and the
    second request silently reuses the first one's KV. See
    :meth:`BlockManager._media_extra_keys`.

    Kept on the Sequence in its own field rather than folded into
    ``multimodal_data``: the scheduler clears that dict when it builds the first
    batch (``ScheduledBatch.__init__``), while block hashes are published
    afterwards in ``Scheduler.postprocess`` — placeholders stored there would be
    gone by publish time, and the blocks would go into the cache under
    text-only hashes that admission never looks up.
    """

    identifier: int
    modality: str
    offset: int
    length: int


def hash_media(image: Any) -> int:
    """Content hash of one media item, stable across requests and processes.

    Hashes the decoded pixels rather than the source bytes: every caller has
    already decoded to RGB, two encodings of the same picture should share
    cache, and it puts data URLs, HTTP downloads, local files and the offline
    example's ``Image.open`` on one path.

    Shape and dtype join the digest because ``tobytes`` drops them, and two
    differently-shaped images can hold the same bytes.
    """
    array = np.asarray(image)
    digest = xxhash.xxh3_128()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.intdigest()


def resolve_placeholder_token_id(atom_config: Config) -> int:
    """The token id an architecture repeats once per media embedding.

    Read off the full multimodal config rather than the model, because the
    callers (API server, offline example) hold a Config and no model.

    Raises rather than defaulting to a literal: a wrong id yields media ranges
    that point at text, and the prefix cache would then key blocks on offsets
    that describe nothing.
    """
    architectures = getattr(atom_config.hf_config, "architectures", None) or []
    arch = architectures[0] if architectures else None
    attr = _MULTIMODAL_ARCH_TO_PLACEHOLDER_ATTR.get(arch) if arch else None
    if attr is None:
        raise ValueError(
            f"no media placeholder token registered for architecture {arch!r}; "
            "add it to _MULTIMODAL_ARCH_TO_PLACEHOLDER_ATTR"
        )

    multimodal_config = getattr(atom_config, "multimodal_config", None)
    if multimodal_config is None:
        raise ValueError(
            f"{arch} media requests need the full HF config (the one carrying "
            "vision_config); start the server with --trust-remote-code."
        )

    token_id = getattr(multimodal_config, attr, None)
    if token_id is None:
        raise ValueError(
            f"the multimodal config has no `{attr}`, which {arch} needs to "
            "locate its media placeholders"
        )
    return int(token_id)


def locate_media_placeholders(
    input_ids: Sequence[int],
    placeholder_token_id: int,
    num_media: int,
) -> list[tuple[int, int]]:
    """Find each media item's ``(offset, length)`` run of placeholder tokens.

    For processors that expand the placeholders themselves (the Qwen
    convention), which leaves the prompt as the only record of where each item
    landed. Runs come back in prompt order, which is the order the encoder
    receives the items in.

    Runs are maximal, so this relies on consecutive items being separated by at
    least one other token — true for the templates ATOM serves, which wrap every
    item in vision start/end markers. A count that disagrees with ``num_media``
    means that assumption broke; raise rather than guess, because two items
    merged into one run would hand the prefix cache a range covering an image it
    does not name.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, token in enumerate(input_ids):
        if token == placeholder_token_id:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(input_ids) - start))

    if len(runs) != num_media:
        raise ValueError(
            f"prompt has {len(runs)} media placeholder run(s) but {num_media} "
            "media item(s) were preprocessed; the chat template and the "
            "processor disagree about placeholder expansion"
        )
    return runs


def build_media_placeholders(
    ranges: Sequence[tuple[int, int]],
    images: Sequence[Any],
    modality: str = "image",
) -> list[MediaPlaceholder]:
    """Pair each item's prompt range with a content hash of its source media."""
    if len(ranges) != len(images):
        raise ValueError(
            f"{len(ranges)} media range(s) for {len(images)} media item(s)"
        )
    return [
        MediaPlaceholder(
            identifier=hash_media(image),
            modality=modality,
            offset=offset,
            length=length,
        )
        for (offset, length), image in zip(ranges, images)
    ]


def get_mrope_input_positions(
    atom_config: Config,
    input_tokens: list[int],
    multimodal_data: dict,
) -> tuple[np.ndarray | None, int]:
    """Return request-level MRoPE positions via the model's MRoPE interface."""

    architectures = getattr(atom_config.hf_config, "architectures", None) or []
    if not architectures:
        return None, 0

    model_qualname = _MULTIMODAL_ARCH_TO_MODEL.get(architectures[0])
    if model_qualname is None:
        return None, 0

    model_cls = resolve_obj_by_qualname(model_qualname)
    mrope_getter = getattr(model_cls, "get_mrope_input_positions", None)
    if mrope_getter is None:
        return None, 0

    return mrope_getter(atom_config, input_tokens, multimodal_data)


def build_multimodal_inputs(
    atom_config: Config,
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
) -> tuple[list[int], dict, list[MediaPlaceholder]]:
    """Tokenize a chat + its images with the architecture's own processor API.

    Returns ``(input_ids, multimodal_data, placeholders)``. Architectures whose
    processor deviates from the Qwen convention register a builder above;
    everything else goes through :func:`build_default_multimodal_inputs`.
    """
    hf_config = getattr(atom_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []

    builder_qualname = None
    if architectures:
        builder_qualname = _MULTIMODAL_ARCH_TO_INPUT_BUILDER.get(architectures[0])

    if builder_qualname is None:
        return build_default_multimodal_inputs(
            atom_config,
            processor,
            messages,
            images,
            chat_template_kwargs,
        )

    builder: Callable = resolve_obj_by_qualname(builder_qualname)
    return builder(
        atom_config,
        processor,
        messages,
        images,
        chat_template_kwargs,
        tools=tools,
    )


def _images_before_text(
    processor_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hoist image parts ahead of the text within each message.

    Qwen3.5's template only reliably emits <|image_pad|> when image entries
    precede the text.
    """
    reordered: list[dict[str, Any]] = []
    for message in processor_messages:
        content = message["content"]
        if not isinstance(content, list):
            reordered.append(message)
            continue
        parts = [part for part in content if part["type"] == "image"]
        texts = [part["text"] for part in content if part["type"] == "text"]
        if texts:
            parts.append({"type": "text", "text": "\n".join(texts)})
        reordered.append({"role": message["role"], "content": parts})
    return reordered


def build_default_multimodal_inputs(
    atom_config: Config,
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
) -> tuple[list[int], dict, list[MediaPlaceholder]]:
    """Build inputs with the Qwen convention: ``processor(text=..., images=...)``.

    The processor expands the placeholders itself, so the item ranges have to be
    recovered by scanning the prompt for them.
    """
    template_kwargs = dict(chat_template_kwargs)
    template_kwargs.pop("tokenize", None)
    template_kwargs.pop("add_generation_prompt", None)
    text = processor.apply_chat_template(
        _images_before_text(messages),
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if images and "<|image_pad|>" not in text:
        raise ValueError("Multimodal chat template did not emit image placeholders")

    inputs = processor(text=[text], images=images, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()
    multimodal_data = {
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
    }
    ranges = locate_media_placeholders(
        input_ids,
        resolve_placeholder_token_id(atom_config),
        len(images),
    )
    return input_ids, multimodal_data, build_media_placeholders(ranges, images)


def expand_media_placeholders(
    input_ids: Sequence[int],
    tokens_per_media: Sequence[int],
    placeholder_token_id: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Repeat each single placeholder token into its media item's token run.

    Processors that leave the expansion to the model (Kimi-K3) emit exactly one
    placeholder per image, but ATOM needs one token per image embedding: the
    scheduler allocates KV blocks and positions from the token count, and the
    prefill scatter matches embeddings against placeholder positions.

    Returns the expanded ids and each item's ``(offset, length)`` within them.
    The ranges fall out of the expansion itself, so they cost no second scan.
    """
    num_placeholders = sum(1 for token in input_ids if token == placeholder_token_id)
    if num_placeholders != len(tokens_per_media):
        raise ValueError(
            f"prompt has {num_placeholders} media placeholder tokens but "
            f"{len(tokens_per_media)} media items were preprocessed"
        )

    expanded: list[int] = []
    ranges: list[tuple[int, int]] = []
    media_index = 0
    for token in input_ids:
        if token == placeholder_token_id:
            length = tokens_per_media[media_index]
            ranges.append((len(expanded), length))
            expanded.extend([token] * length)
            media_index += 1
        else:
            expanded.append(token)
    return expanded, ranges


def _as_pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


def kimi_k3_tokens_per_image(grid_thws, merge_kernel_size) -> list[int]:
    """Image-token count per grid after the ``sd2_tpool`` merge.

    The merge pools the temporal axis away and downsamples each spatial axis by
    the merge kernel, so a ``(t, h, w)`` patch grid yields ``(h // kh) * (w //
    kw)`` tokens regardless of ``t``.
    """
    kernel_h, kernel_w = _as_pair(merge_kernel_size)
    grids = grid_thws.tolist() if hasattr(grid_thws, "tolist") else grid_thws
    return [(int(h) // kernel_h) * (int(w) // kernel_w) for _, h, w in grids]


def build_kimi_k3_inputs(
    atom_config: Config,
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
) -> tuple[list[int], dict, list[MediaPlaceholder]]:
    """Build Kimi-K3 inputs via ``KimiK3Processor``.

    The K3 processor takes messages plus a separate ``medias`` list (its chat
    encoder is Python, not Jinja), returns ``grid_thws`` rather than
    ``image_grid_thw``, and emits a single ``<|media_pad|>`` per image that the
    reference model expands while merging embeddings. Normalize all three so the
    engine sees the same contract as every other multimodal model.
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

    medias = [{"type": "image", "image": image} for image in images]
    inputs = processor(
        messages=messages,
        medias=medias,
        return_tensors="pt",
        **template_kwargs,
    )

    grid_thws = inputs["grid_thws"]
    input_ids = inputs["input_ids"][0].tolist()
    placeholder_token_id = resolve_placeholder_token_id(atom_config)
    tokens_per_image = kimi_k3_tokens_per_image(
        grid_thws, multimodal_config.vision_config.merge_kernel_size
    )
    input_ids, ranges = expand_media_placeholders(
        input_ids, tokens_per_image, placeholder_token_id
    )

    multimodal_data = {
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": grid_thws,
    }
    return input_ids, multimodal_data, build_media_placeholders(ranges, images)
