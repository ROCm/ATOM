# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat-template dispatch for the OpenAI chat endpoint.

Some models (e.g. DeepSeek V4) ship a Python encoder under ``<model>/encoding/``
instead of a Jinja ``chat_template``. This module discovers such encoders at
server startup and provides :func:`apply_chat_template`, a single entry point
that the request handler calls — it transparently routes to the custom encoder
when one was found, or to ``tokenizer.apply_chat_template`` otherwise.
"""

import glob
import importlib.util
import logging
import os
from typing import Any, Callable, List, Optional

from huggingface_hub import snapshot_download

logger = logging.getLogger("atom")

MessageEncoder = Callable[..., str]


def _resolve_model_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    try:
        return snapshot_download(model, local_files_only=True, allow_patterns=[])
    except Exception:
        return model


def _load_encoder_from_dir(model_path: str) -> Optional[MessageEncoder]:
    """Look for ``<model>/encoding/encoding_*.py`` and load ``encode_messages``.

    Returns ``None`` when the directory or matching file is absent (model uses
    the standard Jinja path). Returns ``None`` and warns on ambiguity (multiple
    matches) or load failures.

    """
    enc_dir = os.path.join(model_path, "encoding")
    if not os.path.isdir(enc_dir):
        return None

    candidates = sorted(glob.glob(os.path.join(enc_dir, "encoding_*.py")))
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(
            f"Multiple encoding_*.py found in {enc_dir}, refusing to guess: "
            f"{[os.path.basename(p) for p in candidates]}"
        )
        return None

    enc_path = candidates[0]
    module_name = os.path.splitext(os.path.basename(enc_path))[0]
    try:
        spec = importlib.util.spec_from_file_location(module_name, enc_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw = getattr(mod, "encode_messages")
    except Exception as e:
        logger.warning(f"Failed to load encoder from {enc_path}: {e}")
        return None

    # also valid is "chat" (non-thinking short-form). May need to add as an option.
    # Revisit when a second model ships an encode_*.py — the default may need to be per-model.
    def encode(messages, **kwargs):
        kwargs.setdefault("thinking_mode", "thinking")
        return raw(messages, **kwargs)

    logger.info(f"Loaded message encoder from {enc_path}")
    return encode


def load_custom_message_encoder(model_path: str) -> Optional[MessageEncoder]:
    """Probe ``model_path`` once at startup for a custom message encoder.

    Returns the encoder, or ``None`` when the model uses the standard Jinja
    ``chat_template`` path. Result should be cached by the caller — this does
    filesystem IO and a Python import.
    """
    return _load_encoder_from_dir(_resolve_model_path(model_path))


def apply_chat_template(
    tokenizer: Any,
    custom_encoder: Optional[MessageEncoder],
    messages: List[dict],
    *,
    tools: Optional[List[dict]] = None,
    **kwargs: Any,
) -> str:
    """Render ``messages`` to a prompt string.

    Dispatches to ``custom_encoder`` if one was discovered for this model,
    otherwise to ``tokenizer.apply_chat_template``. Jinja-only kwargs
    (``tokenize``, ``add_generation_prompt``) are stripped on the custom
    path; ``tools`` are forwarded only on the Jinja path (custom encoders
    don't currently have a tools API — caller is warned and tools are
    dropped).
    """
    if custom_encoder is not None:
        for k in ("tokenize", "add_generation_prompt"):
            kwargs.pop(k, None)
        if tools:
            logger.warning(
                "tools= is not supported with the custom message encoder; ignoring."
            )
        return custom_encoder(messages, **kwargs)

    kwargs["tokenize"] = False
    kwargs["add_generation_prompt"] = True
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)
