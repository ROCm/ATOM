"""The `EngineSpec` atomesh's Rust runtime starts from.

Its shape is `atom/mesh/src/atom_engine/spec.rs`. Python launches the
EngineCores and hands this over once; Rust is their only client from then on.
What needs ATOM's own machinery -- the chat template, the reasoning dialect
and switch, the tool-call format -- is decided here, once, the way the Python
API server decides it.
"""

from __future__ import annotations

import json
import os
from typing import Any, NamedTuple

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
    PARSERS_BY_NAME,
    required_call_channel,
    resolve_tool_call_parser,
)
from atom.utils import envs

# Each ATOM tool-call format (`PARSERS_BY_NAME`) by mesh's name for the same
# format. The names differ where mesh's vocabulary means something else by
# the same word: its `qwen` is Qwen3's JSON-in-tag, ATOM's is the XML one.
MESH_TOOL_PARSERS = {
    "qwen": "qwen_xml",
    "kimi": "kimi",
    "kimi_k3": "kimi_k3",
    "glm": "glm",
    "minimax": "minimax",
    "dsml": "dsml",
    "dsml_v41": "dsml_v41",
}

# kv_transfer roles that make a P/D worker, which standalone cannot serve yet.
_PD_ROLES = frozenset({"kv_producer", "kv_consumer", "kv_both"})


def check_standalone_model(engine_args: Any) -> None:
    """Refuse, before any weights load, what the Rust frontend cannot serve:
    a model with a custom message encoder instead of a chat template, and a
    P/D (prefill/decode) worker."""
    if load_custom_message_encoder(engine_args.model) is not None:
        raise ValueError(
            f"{engine_args.model} renders prompts with a custom message encoder, "
            "which the atomesh standalone server does not support yet; serve it "
            "with USE_ATOMESH_ENTRYPOINTS=0."
        )
    transfer = json.loads(getattr(engine_args, "kv_transfer_config", None) or "{}")
    if transfer.get("kv_role") in _PD_ROLES:
        raise ValueError(
            f"kv_role {transfer['kv_role']!r} makes a P/D worker, which the "
            "atomesh standalone server does not support yet; serve it with "
            "USE_ATOMESH_ENTRYPOINTS=0."
        )


def apply_chat_template(
    tokenizer: Any, override: str | None, directory: str
) -> str | None:
    """Apply `--chat-template` as the Python server does (a file, else the
    template text), and write the template every probe below reads to
    `directory`, so Rust renders with exactly it. None: the tokenizer has no
    single template, and Rust finds the model's own."""
    if override:
        if os.path.isfile(override):
            with open(override, encoding="utf-8") as f:
                tokenizer.chat_template = f.read()
        else:
            tokenizer.chat_template = override
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        return None
    path = os.path.join(directory, "chat_template.jinja")
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)
    return path


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


class ToolCallFormat(NamedTuple):
    # ATOM's parser, which the required channel comes from; None for a format
    # only mesh knows, or none at all.
    parser: type | None
    # What mesh is told; None leaves mesh to pick by model name.
    mesh_name: str | None


def tool_call_format(
    requested: str | None, tokenizer: Any, model: str, mesh_names: set[str]
) -> ToolCallFormat:
    """The tool-call format both halves use, decided once.

    A name ATOM knows binds ATOM's parser and mesh's parser for the same
    format; a name only mesh knows binds mesh's alone; any other name is
    refused here, before the weights load. With no name the format is read
    off the chat template as the Python server reads it; when the template
    names none, mesh keeps its own guess by model name.
    """
    if requested and requested != "auto" and requested not in PARSERS_BY_NAME:
        if requested in mesh_names:
            return ToolCallFormat(None, requested)
        raise ValueError(
            f"--tool-call-parser {requested!r} is unknown; choose one of "
            f"{sorted({*PARSERS_BY_NAME, *mesh_names})} or 'auto'"
        )
    parser = resolve_tool_call_parser(
        requested,
        tokenizer,
        model=model,
        rendered_probe=render_probe_prompt(tokenizer, None, tools=True),
    )
    if parser is None:
        return ToolCallFormat(None, None)
    mesh_name = MESH_TOOL_PARSERS.get(parser.NAME)
    if mesh_name not in mesh_names:
        raise ValueError(
            f"tool-call format {parser.NAME!r} has no parser in the atomesh "
            "standalone server; serve it with USE_ATOMESH_ENTRYPOINTS=0."
        )
    return ToolCallFormat(parser, mesh_name)


def required_channel(parser: type | None, tokenizer: Any, config: Any) -> dict | None:
    """The channel a `tool_choice="required"` request is held to, as the
    Python server holds it (`required_call_channel`, then the engine's
    `ending_on` its end ids); None where the tool-call format has none."""
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
    tool_parser: type | None,
    chat_template: str | None,
) -> str:
    config = engine.config
    toggle = resolve_reasoning_toggle(tokenizer)
    speculative = getattr(config, "speculative_config", None)
    metadata = deployment_metadata(config, None)
    return json.dumps(
        {
            "model": {
                "served_name": served_name,
                "tokenizer_path": config.model,
                "chat_template": chat_template,
                "default_chat_template_kwargs": default_chat_template_kwargs,
                "max_model_len": config.max_model_len,
                "default_max_tokens": DEFAULT_MAX_TOKENS,
                "max_output_tokens": metadata.get("max_output_tokens"),
                "model_card": metadata,
                # Forced acceptance has no real logprobs to return
                # (`SamplingParams.validate_logprob_options`).
                "synthetic_acceptance": getattr(
                    speculative, "synthetic_acceptance_rates", None
                )
                is not None,
                "required_channel": required_channel(tool_parser, tokenizer, config),
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
