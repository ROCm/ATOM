# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Checkpoint architecture -> diffusion pipeline class.

The LLM side resolves a checkpoint to an implementation by looking up
``hf_config.architectures[0]`` in a dict of dotted paths
(``model_runner.support_model_arch_dict``, ``multimodal._MULTIMODAL_ARCH_TO_MODEL``).
Diffusion checkpoints carry the same information under a different key:
``model_index.json`` names the pipeline that produced them in ``_class_name``.

Mapping it here means ``--model <root>`` alone identifies the pipeline, and an
unrecognised checkpoint fails with the list of what is supported rather than by
running the wrong stages. An explicit ``--pipeline`` still wins, which is what
out-of-tree pipelines and the tests use.
"""

import json
import os

from atom.utils import resolve_obj_by_qualname

_MINIMAX_H3 = "atom.diffusion.models.minimax_h3.pipeline.MiniMaxH3Pipeline"

_PIPELINE_ARCH_TO_CLASS: dict[str, str] = {
    # What the released checkpoint root declares.
    "MiniMaxH3ModularPipeline": _MINIMAX_H3,
    # What a partition under that root declares. ``--model <root>/FL2VA`` is
    # the documented way to serve one variant, so the manifest the server
    # actually reads is usually this one, not the root's.
    "MiniMaxH3Pipeline": _MINIMAX_H3,
}

MODEL_INDEX_FILENAME = "model_index.json"


def resolve_pipeline_class(dotted: str):
    """Import ``pkg.module.Class`` for a configured pipeline."""
    if "." not in dotted:
        raise ValueError(f"pipeline_class must be a dotted path, got {dotted!r}")
    return resolve_obj_by_qualname(dotted)


def checkpoint_architecture(model_path: str) -> str | None:
    """``_class_name`` from a checkpoint's ``model_index.json``, if readable.

    Returns None rather than raising: a missing or malformed index is a reason
    to fall back to an explicit ``--pipeline``, not to refuse to start.
    """
    index_path = os.path.join(model_path, MODEL_INDEX_FILENAME)
    try:
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    name = index.get("_class_name")
    return str(name) if name else None


def pipeline_class_for_checkpoint(model_path: str) -> str | None:
    """Dotted path of the pipeline serving ``model_path``, or None if unknown."""
    architecture = checkpoint_architecture(model_path)
    if architecture is None:
        return None
    return _PIPELINE_ARCH_TO_CLASS.get(architecture)


def supported_architectures() -> tuple[str, ...]:
    """Checkpoint architectures with a registered pipeline."""
    return tuple(sorted(_PIPELINE_ARCH_TO_CLASS))
