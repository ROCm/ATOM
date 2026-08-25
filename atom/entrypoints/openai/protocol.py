# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pydantic request/response models for the OpenAI-compatible API."""

import json
import time
from typing import Any, Dict, List, Optional, Union

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


# ============================================================================
# Finish reasons
# ============================================================================


def openai_finish_reason(engine_reason: Optional[str]) -> Optional[str]:
    """Map ATOM's engine termination reason onto OpenAI's ``finish_reason``.

    The scheduler reports its own vocabulary — ``eos``, ``max_tokens``,
    ``stop_sequence``, ``stop_<token_id>``, ``aborted``,
    ``unschedulable: ...`` — but OpenAI clients switch on exactly
    ``{stop, length, tool_calls, content_filter}``, and an unknown string sends
    them down their error path. Only a length cap maps to ``length``; every other
    way a sequence can end is a stop from the client's point of view.
    """
    if engine_reason is None:
        return None
    return "length" if engine_reason == "max_tokens" else "stop"


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
    content: Union[str, List[Dict[str, Any]], None] = None

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

    def to_template_dict(self) -> Dict[str, Any]:
        """Convert to dict for chat template, preserving tool-related fields.

        Returns a dict with role, content, and any extra fields (tool_calls,
        tool_call_id, name, reasoning_content) that the chat template needs.
        """
        d: Dict[str, Any] = {"role": self.role, "content": self.get_content_text()}
        # Preserve extra fields needed by chat templates (e.g. Kimi-K2)
        extras = self.model_extra or {}
        for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
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

    model: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    prompt: Optional[List[ChatMessage]] = None  # Accept 'prompt' as alias
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    top_k: Optional[int] = DEFAULT_TOP_K
    top_p: Optional[float] = DEFAULT_TOP_P
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False
    seed: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = (
        None  # "auto", "none", "required", or {function: {name}}
    )
    # MiniMax-M2/M3 reasoning toggle: {"type": "enabled"} / {"type": "disabled"}.
    # Resolved by chat_request.resolve_thinking(); a plain bool is also accepted.
    thinking: Optional[Any] = None
    # Accepted for compatibility, not actively used:
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    # Optional KV-transfer metadata for P/D disaggregation.
    kv_transfer_params: Optional[Dict[str, Any]] = None

    def get_max_tokens(self) -> int:
        """Return the effective generation cap for OpenAI chat requests."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return DEFAULT_MAX_TOKENS

    def get_stop(self) -> Optional[List[str]]:
        """Normalize ``stop`` to a list (OpenAI accepts a bare string too)."""
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop

    def get_messages(self) -> List[ChatMessage]:
        """Get messages from either 'messages' or 'prompt' field.

        Raises:
            ValueError: when neither field is present, or the conversation is
                empty. An empty conversation still renders a valid generation
                prompt, so without this check the model would answer a request
                that carries no instruction at all (HTTP 200 instead of 400).
        """
        messages = self.messages or self.prompt
        if messages:
            return messages
        if self.messages is None and self.prompt is None:
            raise ValueError("Either 'messages' or 'prompt' field is required")
        raise ValueError("'messages' must contain at least one message")


class CompletionRequest(BaseModel):
    """Request model for text completions (OpenAI-compatible)."""

    model_config = {"extra": "ignore"}

    model: Optional[str] = None
    prompt: str
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    top_k: Optional[int] = DEFAULT_TOP_K
    top_p: Optional[float] = DEFAULT_TOP_P
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False
    seed: Optional[int] = None
    # Optional KV-transfer metadata for P/D disaggregation.
    kv_transfer_params: Optional[Dict[str, Any]] = None
    # Optional DPA routing hint inserted by atomesh for DP-aware workers.
    data_parallel_rank: Optional[int] = None
    n: Optional[int] = 1

    def get_max_tokens(self) -> int:
        """Return the effective generation cap for completion requests."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return DEFAULT_MAX_TOKENS

    def get_stop(self) -> Optional[List[str]]:
        """Normalize ``stop`` to a list (OpenAI accepts a bare string too)."""
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop


# ============================================================================
# Response Models
# ============================================================================


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""

    id: str
    object: str = CHAT_COMPLETION_OBJECT
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]
    kv_transfer_params: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class CompletionResponse(BaseModel):
    """Response model for text completions."""

    id: str
    object: str = TEXT_COMPLETION_OBJECT
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]
    # Optional KV-transfer metadata returned for P/D disaggregation.
    kv_transfer_params: Optional[Dict[str, Any]] = None


class ModelCard(BaseModel):
    """Model card for /v1/models endpoint."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "atom"


class ModelList(BaseModel):
    """Response for /v1/models endpoint."""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """OpenAI-format error response."""

    error: Dict[str, Any]
