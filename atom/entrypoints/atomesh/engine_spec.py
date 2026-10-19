"""The `EngineSpec` atomesh's Rust runtime starts from.

Its shape is `atom/mesh/src/atom_engine/spec.rs`. Python launches the
EngineCores and hands this over once; Rust is their only client from then on.
"""

from __future__ import annotations

import json
from typing import Any

from atom.entrypoints.openai.chat_encoders import (
    chat_template_source,
    load_custom_message_encoder,
    render_probe_prompt,
)
from atom.entrypoints.openai.protocol import DEFAULT_MAX_TOKENS
from atom.entrypoints.openai.reasoning import template_opens_reasoning_implicitly
from atom.entrypoints.openai.reasoning_dialects import THINK_END_MARKER, resolve_dialect
from atom.utils import envs


def check_standalone_model(model: str) -> None:
    """Refuse, before any weights load, a model the Rust frontend cannot
    prompt: one with a custom message encoder instead of a chat template."""
    if load_custom_message_encoder(model) is not None:
        raise ValueError(
            f"{model} renders prompts with a custom message encoder, which the "
            "atomesh standalone server does not support yet; serve it with "
            "USE_ATOMESH_ENTRYPOINTS=0."
        )


def default_reasoning_parser(tokenizer: Any) -> str | None:
    """The mesh reasoning parser that splits what the Python server splits,
    read off the chat template as the Python server reads it. Mesh otherwise
    picks one by model name, and a path like `/models/GLM-5.3` names none.

    `qwen3` is mesh's plain `<think>...</think>` parser; it starts inside the
    block itself when the rendered prompt opened it. Mesh has no parser for
    the other dialects, so they get none, as before.
    """
    source = chat_template_source(tokenizer)
    if template_opens_reasoning_implicitly(source):
        return "deepseek_r1"
    probe = render_probe_prompt(tokenizer, None, tools=False) or ""
    dialect, _ = resolve_dialect(source, probe)
    return "qwen3" if dialect.think_end_marker == THINK_END_MARKER else None


def build_engine_spec(
    engine: Any, served_name: str, default_chat_template_kwargs: dict[str, Any]
) -> str:
    config = engine.config
    return json.dumps(
        {
            "model": {
                "served_name": served_name,
                "tokenizer_path": config.model,
                "default_chat_template_kwargs": default_chat_template_kwargs,
                "max_model_len": config.max_model_len,
                "default_max_tokens": DEFAULT_MAX_TOKENS,
                "speculative": engine.io_processor.num_speculative_tokens > 0,
            },
            "balance": {
                "strategy": config.dp_load_balance,
                "req_equiv": envs.ATOM_DP_LB_REQ_EQUIV,
                "session_affinity": envs.ATOM_DP_SESSION_AFFINITY,
            },
            "engines": engine.core_mgr.engine_endpoints,
        }
    )
