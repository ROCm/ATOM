# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pydantic request/response models for the OpenAI-compatible API."""

import json
import re
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Constants
# ============================================================================

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = -1
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 8192
CHAT_COMPLETION_OBJECT = "chat.completion"
CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"
TEXT_COMPLETION_OBJECT = "text_completion"
STREAM_DONE_MESSAGE = "data: [DONE]\n\n"


# Valid OpenAI ``tool_choice`` string values and the function-name constraint.
# Spec-level (not model-specific): the same for every model served.
TOOL_CHOICE_VALUES = frozenset({"auto", "none", "required"})


def openai_stop_reason(finish_reason: str | None) -> str | None:
    """The engine's leave reason as OpenAI spells it.

    The engine says `eos` / `max_tokens` / `stop_sequence` / `stop_<token_id>`
    / `aborted` / `unschedulable: ...`; OpenAI clients understand only `stop` /
    `length` / `tool_calls`. `stop_<token_id>` is an ordinary end of turn --
    any model declaring more than one EOS reaches it in normal operation.

    Named for the vocabulary it maps *into*, and paired with
    `api_server.anthropic_stop_reason`. Two functions rather than one with a
    mode: the two vocabularies share no member, so chaining them would send
    every reason to the other's default.
    """
    if finish_reason is None:
        return None
    if finish_reason in ("stop", "length", "tool_calls"):
        return finish_reason
    if finish_reason in ("max_tokens", "max_new_tokens"):
        return "length"
    return "stop"


def openai_stop_reason_with_calls(engine_reason: str | None, has_calls: bool) -> str:
    """The reason to report when a call was parsed and the engine had its own.

    `length` outranks `tool_calls`, because they answer different questions
    and only one of them is a warning: `tool_calls` says "act on this", and
    `length` says "this is not all of it". A response cut off mid-call parses
    to a call with a silently truncated argument value -- every format's
    unclosed-region branch exists to salvage exactly that -- and reporting
    `tool_calls` for it told the client to run a tool with half its arguments
    and no indication anything was missing. OpenAI reports `length` for a
    truncated response whatever else is in it.
    """
    normalized = openai_stop_reason(engine_reason)
    if normalized == "length":
        return "length"
    return "tool_calls" if has_calls else (normalized or "stop")


TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


# ============================================================================
# Request Models
# ============================================================================


def _fix_invalid_json_escapes(s: str) -> str:
    """Fix invalid JSON escapes in model-generated tool-call arguments.

    Models occasionally produce invalid escape sequences like ``\\k`` or
    ``\\p`` in function.arguments JSON. ``json.loads`` rejects these. This
    helper doubles any backslash not followed by a valid JSON escape char.
    """
    _VALID = frozenset('"\\bfnrtu/')
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\":
            if i + 1 >= len(s):
                out.append("\\\\")
                i += 1
            elif s[i + 1] == "\\":
                out.append("\\\\")
                i += 2
            elif s[i + 1] in _VALID:
                out.append("\\")
                out.append(s[i + 1])
                i += 2
            else:
                out.append("\\\\")
                out.append(s[i + 1])
                i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _normalize_tool_call_arguments(tool_calls: Any) -> Any:
    """Deserialize ``function.arguments`` from a JSON string to a mapping.

    OpenAI clients send tool-call arguments as a JSON *string*, but chat
    templates (Qwen3 qwen3_coder/qwen3_xml, Hermes, etc.) iterate
    ``tool_call.arguments.items()`` and require a mapping. Mirrors how vLLM and
    SGLang deserialize arguments before applying the chat template.
    """
    if not isinstance(tool_calls, list):
        return tool_calls
    normalized = []
    for tc in tool_calls:
        if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
            fn = dict(tc["function"])
            if isinstance(fn.get("arguments"), str):
                raw = fn["arguments"]
                try:
                    fn["arguments"] = json.loads(raw)
                except (ValueError, TypeError):
                    try:
                        fn["arguments"] = json.loads(_fix_invalid_json_escapes(raw))
                    except (ValueError, TypeError):
                        fn["arguments"] = {"_raw": raw}
            tc = {**tc, "function": fn}
        normalized.append(tc)
    return normalized


class ChatMessage(BaseModel):
    """Represents a single chat message."""

    role: str
    content: str | list[dict[str, Any]] | None = None

    model_config = ConfigDict(extra="allow")

    def get_content_text(self) -> str:
        """Extract text content, handling both string and multimodal content parts."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # OpenAI multimodal format: [{"type": "text", "text": "..."}, ...]
        parts = []
        for part in self.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)

    def to_template_dict(self) -> dict[str, Any]:
        """Convert to dict for chat template, preserving tool-related fields.

        Returns a dict with role, content, and any extra fields (tool_calls,
        tool_call_id, name, reasoning_content, tools) that the chat template needs.
        """
        d: dict[str, Any] = {"role": self.role, "content": self.get_content_text()}
        # Preserve extra fields needed by chat templates (e.g. Kimi-K2/K3).
        # "tools" carries K3 dynamically-loaded tools declared inside a system
        # message; encoding_k3.build_chat_segments renders them per-message.
        extras = self.model_extra or {}
        for key in ("tool_calls", "tool_call_id", "name", "reasoning_content", "tools"):
            if key in extras:
                d[key] = (
                    _normalize_tool_call_arguments(extras[key])
                    if key == "tool_calls"
                    else extras[key]
                )
        return d


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions (OpenAI-compatible)."""

    model_config = {"extra": "ignore"}

    model: str | None = None
    messages: list[ChatMessage] | None = None
    prompt: list[ChatMessage] | None = None  # Accept 'prompt' as alias
    temperature: float | None = DEFAULT_TEMPERATURE
    top_k: int | None = DEFAULT_TOP_K
    top_p: float | None = DEFAULT_TOP_P
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    max_completion_tokens: int | None = None
    stop: list[str] | None = None
    ignore_eos: bool | None = False
    stream: bool | None = False
    seed: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    # Tool calling
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None  # "auto", "none", "required", or {function: {name}}
    # Structured output: {"type": "text"|"json_object"|"json_schema", ...}
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None  # "low"|"high"|"max"
    # K3 thinking control (sent by clients via extra_body):
    # {"type": "enabled"|"disabled", "keep": "all", "effort": "low"|"high"|"max"}.
    # Without this field pydantic (extra="ignore") silently drops it, so effort
    # never reaches the template and the streaming reasoning gate never fires.
    thinking: dict[str, Any] | None = None
    # Accepted for compatibility, not actively used:
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    n: int | None = 1
    # Optional KV-transfer metadata for P/D disaggregation.
    kv_transfer_params: dict[str, Any] | None = None
    data_parallel_rank: int | None = None

    def get_max_tokens(self) -> int:
        """Return the effective generation cap for OpenAI chat requests."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return DEFAULT_MAX_TOKENS

    def get_messages(self) -> list[ChatMessage]:
        """Get messages from either 'messages' or 'prompt' field."""
        if self.messages is not None:
            return self.messages
        elif self.prompt is not None:
            return self.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' field is required")


class CompletionRequest(BaseModel):
    """Request model for text completions (OpenAI-compatible)."""

    model_config = {"extra": "ignore"}

    model: str | None = None
    prompt: str
    temperature: float | None = DEFAULT_TEMPERATURE
    top_k: int | None = DEFAULT_TOP_K
    top_p: float | None = DEFAULT_TOP_P
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    max_completion_tokens: int | None = None
    stop: list[str] | None = None
    ignore_eos: bool | None = False
    stream: bool | None = False
    # Optional KV-transfer metadata for P/D disaggregation.
    kv_transfer_params: dict[str, Any] | None = None
    # Optional DPA routing hint inserted by atomesh for DP-aware workers.
    data_parallel_rank: int | None = None
    n: int | None = 1

    def get_max_tokens(self) -> int:
        """Return the effective generation cap for completion requests."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return DEFAULT_MAX_TOKENS


# ============================================================================
# Response Models
# ============================================================================


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""

    id: str
    object: str = CHAT_COMPLETION_OBJECT
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any]
    kv_transfer_params: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class CompletionResponse(BaseModel):
    """Response model for text completions."""

    id: str
    object: str = TEXT_COMPLETION_OBJECT
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any]
    # Optional KV-transfer metadata returned for P/D disaggregation.
    kv_transfer_params: dict[str, Any] | None = None


class ModelCard(BaseModel):
    """Model card for /v1/models endpoint."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "atom"


class ModelList(BaseModel):
    """Response for /v1/models endpoint."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """OpenAI-format error response."""

    error: dict[str, Any]
