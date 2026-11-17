# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Discovery metadata for the running deployment, not the hosted offering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chat_encoder_adapters import MessageEncoderAdapter
    from .reasoning_controls import ReasoningRules


def deployment_metadata(config, encoder):
    hf = getattr(config, "hf_config", None)
    text = getattr(hf, "text_config", None) or hf
    limit = getattr(config, "max_model_len", None)
    declared = getattr(text, "max_position_embeddings", None)
    limits = [v for v in (limit, declared) if type(v) is int and v > 0]
    result = {"output_modalities": ["text"]}
    if limits:
        result["context_window"] = min(limits)
        result["max_output_tokens"] = min(limits)
    method = getattr(getattr(config, "speculative_config", None), "method", None)
    if (method or getattr(config, "method", None)) == "dspark":
        result["input_modalities"] = ["text"]
    from .api_compatibility import api_compatibility_for

    contract = api_compatibility_for(encoder)
    if contract.display_name is not None:
        result["name"] = contract.display_name
    if contract.effort_levels:
        result["effort"] = {
            "supported_levels": list(contract.effort_levels),
            "default_level": contract.default_effort,
        }
    if contract.output_token_limit is not None:
        result["max_output_tokens"] = min(
            result.get("max_output_tokens", contract.output_token_limit),
            contract.output_token_limit,
        )
    return result


def chat_rendering(
    defaults: dict[str, Any],
    toggle: tuple[str, Any, Any] | None,
    encoder: MessageEncoderAdapter | None,
    rules: ReasoningRules,
) -> dict[str, Any]:
    """The template kwargs the chat endpoint renders a chat with, as data a
    router can apply to place chats by the prompt the engine sees.

    The server's ``defaults``, the request's `chat_template_kwargs` over them,
    then its reasoning controls under ``rules`` (named; ATOM's code), which set
    ``toggle`` and, under the ``chat`` rules, name an effort its templates
    read. A custom ``encoder`` then takes them through
    `MessageEncoderAdapter.render_kwargs`.
    """
    from .reasoning import VALID_TEMPLATE_EFFORTS

    if toggle is not None:
        name, off, on = toggle
        toggle = {"name": name, "off": off, "on": on}
    return {
        "default_kwargs": dict(defaults),
        "reasoning_toggle": toggle,
        "reasoning_rules": rules.name,
        "template_efforts": sorted(VALID_TEMPLATE_EFFORTS),
        "encoder": None if encoder is None else encoder.rendering(),
    }
