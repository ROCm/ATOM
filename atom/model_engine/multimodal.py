# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import numpy as np

from atom.config import Config
from atom.utils import resolve_obj_by_qualname

_MULTIMODAL_ARCH_TO_MODEL: dict[str, str] = {
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": (
        "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel"
    ),
}


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
