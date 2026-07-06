# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pydantic request/response models for the OpenAI-compatible API."""

import json
import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
# Request Models
# ============================================================================


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
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except (ValueError, TypeError):
                    pass
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


class FunctionDefinition(BaseModel):
    """OpenAI function definition for tool calling."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ChatCompletionToolsParam(BaseModel):
    """A tool the model may call (OpenAI-compatible)."""

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(BaseModel):
    """Reference to a specific function by name for tool_choice."""

    name: str


class ChatCompletionNamedToolChoiceParam(BaseModel):
    """Force the model to call a specific named tool."""

    function: ChatCompletionNamedFunction
    type: str = "function"


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
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False
    seed: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    # Tool calling
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[str, ChatCompletionNamedToolChoiceParam]] = None
    # Accepted for compatibility, not actively used:
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    # Optional KV-transfer metadata for P/D disaggregation.
    kv_transfer_params: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Reject empty tools array (matches OpenAI API behaviour)
        if data.get("tools") == []:
            raise ValueError(
                "`tools` must not be an empty array. "
                "Either provide at least one tool or omit the field entirely."
            )

        # Default tool_choice to "auto" when tools are provided
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"

        # "none" needs no further validation
        if data.get("tool_choice") == "none":
            return data

        # If tool_choice is set, tools must be present
        if data.get("tool_choice") is not None:
            if not data.get("tools"):
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set."
                )

            choice = data["tool_choice"]
            # Must be a known string or a named-tool dict
            if choice not in ("auto", "required") and not isinstance(choice, dict):
                raise ValueError(
                    f"Invalid value for `tool_choice`: {choice!r}! "
                    'Only named tools, "none", "auto" or "required" '
                    "are supported."
                )

            # Validate named tool choice matches a provided tool
            if isinstance(choice, dict):
                fn = choice.get("function")
                if not isinstance(fn, dict) or "name" not in fn:
                    raise ValueError(
                        "Invalid `tool_choice`: expected "
                        '`{"type": "function", "function": {"name": "my_function"}}`'
                    )
                fn_name = fn["name"]
                tool_names = {
                    (t.get("function") or {}).get("name")
                    for t in (data.get("tools") or [])
                    if isinstance(t, dict)
                }
                if fn_name not in tool_names:
                    raise ValueError(
                        "The tool specified in `tool_choice` does not match "
                        "any of the specified `tools`."
                    )

        return data

    def get_max_tokens(self) -> int:
        """Return the effective generation cap for OpenAI chat requests."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return DEFAULT_MAX_TOKENS

    def get_messages(self) -> List[ChatMessage]:
        """Get messages from either 'messages' or 'prompt' field."""
        if self.messages is not None:
            return self.messages
        elif self.prompt is not None:
            return self.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' field is required")

    def get_tools_as_dicts(self) -> Optional[List[Dict[str, Any]]]:
        """Return tools as plain dicts for downstream consumers."""
        if self.tools is None:
            return None
        return [t.model_dump() for t in self.tools]

    def get_tool_choice_value(self) -> Optional[Union[str, Dict[str, Any]]]:
        """Return tool_choice as a JSON-serializable value for chat templates."""
        if self.tool_choice is None:
            return None
        if isinstance(self.tool_choice, str):
            return self.tool_choice
        return self.tool_choice.model_dump()


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
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False
    # Optional KV-transfer metadata for P/D disaggregation.
    kv_transfer_params: Optional[Dict[str, Any]] = None
    n: Optional[int] = 1

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
