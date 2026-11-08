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
    resolve_reasoning_toggle,
)
from atom.entrypoints.openai.model_capabilities import deployment_metadata
from atom.entrypoints.openai.protocol import DEFAULT_MAX_TOKENS
from atom.entrypoints.openai.reasoning import template_opens_reasoning_implicitly
from atom.entrypoints.openai.reasoning_dialects import THINK_END_MARKER, resolve_dialect
from atom.entrypoints.openai.tool_parser.registry import (
    required_call_channel,
    resolve_tool_call_parser,
)
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


def default_reasoning_parser(tokenizer: Any) -> str:
    """The mesh reasoning parser that splits what the Python server splits,
    read off the chat template as the Python server reads it. Mesh otherwise
    picks one by model name, and a path like `/models/GLM-5.3` names none.

    `qwen3` is mesh's plain `<think>...</think>` parser; it starts inside the
    block itself when the rendered prompt opened it. Mesh has no parser for
    the other dialects (MiniMax's `<mm:think>`, Kimi-K3's channels), whose
    reasoning would reach the client as the answer, so they are refused.
    """
    source = chat_template_source(tokenizer)
    if template_opens_reasoning_implicitly(source):
        return "deepseek_r1"
    probe = render_probe_prompt(tokenizer, None, tools=False) or ""
    dialect, _ = resolve_dialect(source, probe)
    if dialect.think_end_marker != THINK_END_MARKER:
        raise ValueError(
            f"this model reasons in a dialect ending on {dialect.think_end_marker!r}, "
            "which the atomesh standalone server cannot separate yet; serve it "
            "with USE_ATOMESH_ENTRYPOINTS=0, or name a --reasoning-parser."
        )
    return "qwen3"


def required_channel(tokenizer: Any, config: Any) -> dict[str, Any] | None:
    """The channel a `tool_choice="required"` request is held to, as the
    Python server holds it (`required_call_channel`, then the engine's
    `ending_on` its end ids); None where the tool-call format has none."""
    parser = resolve_tool_call_parser(
        None,
        tokenizer,
        model=config.model,
        rendered_probe=render_probe_prompt(tokenizer, None, tools=True),
    )
    channel = required_call_channel(parser, tokenizer)
    if channel is None:
        return None
    channel = channel.ending_on(config.end_token_ids)
    return {
        "after": list(channel.after),
        "opening": list(channel.opening),
        "opens_at": channel.opens_at,
        "end_ids": list(channel.end_ids),
    }


def build_engine_spec(
    engine: Any,
    tokenizer: Any,
    served_name: str,
    default_chat_template_kwargs: dict[str, Any],
) -> str:
    config = engine.config
    toggle = resolve_reasoning_toggle(tokenizer)
    return json.dumps(
        {
            "model": {
                "served_name": served_name,
                "tokenizer_path": config.model,
                "default_chat_template_kwargs": default_chat_template_kwargs,
                "max_model_len": config.max_model_len,
                "default_max_tokens": DEFAULT_MAX_TOKENS,
                "max_output_tokens": deployment_metadata(config, None).get(
                    "max_output_tokens"
                ),
                "speculative": engine.io_processor.num_speculative_tokens > 0,
                "required_channel": required_channel(tokenizer, config),
                "reasoning_toggle": (
                    None
                    if toggle is None
                    else dict(zip(("name", "off", "on"), toggle, strict=True))
                ),
            },
            "balance": {
                "strategy": config.dp_load_balance,
                "req_equiv": envs.ATOM_DP_LB_REQ_EQUIV,
                "session_affinity": envs.ATOM_DP_SESSION_AFFINITY,
            },
            "engines": engine.core_mgr.engine_endpoints,
        }
    )
