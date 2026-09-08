# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The contract an architecture's multimodal input builder implements.

Most checkpoints follow the Qwen convention — ``processor(text=..., images=...)``
returns already-expanded placeholders — and the server handles them inline. A
model whose processor deviates registers a builder in
:mod:`atom.multimodal.registry` instead; this is the shape that builder has.
"""

from typing import Any, Protocol

from atom.config import Config


class MultiModalInputBuilder(Protocol):
    """Turn chat messages plus already-loaded media into engine inputs.

    Returns ``(input_ids, multimodal_data)``, or ``None`` when the builder does
    not apply to this checkpoint and the caller should fall back to its text
    path — a registry key is an architecture string, which for some families is
    shared by a text-only and a multimodal checkpoint.

    ``multimodal_data`` carries whatever the model's ``get_vision_embeddings``
    consumes. The engine only requires the two keys it marshals itself:

    * ``pixel_values``    — media tensor, concatenated over items
    * ``image_grid_thw``  — ``(t, h, w)`` per item, so the model can split it
    """

    def __call__(
        self,
        atom_config: Config,
        processor: Any,
        messages: list[dict],
        images: list,
        chat_template_kwargs: dict,
        tools: Any = None,
    ) -> tuple[list[int], dict] | None: ...
