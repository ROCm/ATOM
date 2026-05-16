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
RESPONSES_OBJECT = "response"


# ============================================================================
# Request Models
# ============================================================================


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

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> Any:
        """Decode OpenAI JSON-string tool arguments for chat templates."""
        if not isinstance(tool_calls, list):
            return tool_calls

        normalized = []
        for item in tool_calls:
            if not isinstance(item, dict):
                raise ValueError(
                    f"tool_calls entries must be dicts, got {type(item)!r}"
                )

            call = dict(item)
            function_value = call.get("function") or {}
            if not isinstance(function_value, dict):
                raise ValueError(
                    f"tool_calls function must be a dict, got {type(function_value)!r}"
                )

            function = dict(function_value)
            arguments = function.get("arguments")
            if arguments is None or arguments == "":
                function["arguments"] = {}
            elif isinstance(arguments, str):
                try:
                    decoded_arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "tool_calls function.arguments must be a valid JSON object"
                    ) from exc
                if not isinstance(decoded_arguments, dict):
                    raise ValueError(
                        "tool_calls function.arguments must decode to a JSON object"
                    )
                function["arguments"] = decoded_arguments
            elif not isinstance(arguments, dict):
                raise ValueError(
                    "tool_calls function.arguments must be a dict or JSON object string"
                )
            call["function"] = function
            normalized.append(call)
        return normalized

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
                value = extras[key]
                if key == "tool_calls":
                    value = self._normalize_tool_calls(value)
                d[key] = value
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
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False
    seed: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = (
        None  # "auto", "none", "required", or {function: {name}}
    )
    # Accepted for compatibility, not actively used:
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1

    def get_messages(self) -> List[ChatMessage]:
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

    model: Optional[str] = None
    prompt: str
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    top_k: Optional[int] = DEFAULT_TOP_K
    top_p: Optional[float] = DEFAULT_TOP_P
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False
    # Optional KV-transfer metadata for P/D disaggregation.
    kv_transfer_params: Optional[Dict[str, Any]] = None
    n: Optional[int] = 1


class ResponsesRequest(BaseModel):
    """Request model for the OpenAI Responses API.

    This is a compatibility subset that maps Responses input onto ATOM's
    existing chat-template and generation path.
    """

    model_config = {"extra": "allow"}

    model: Optional[str] = None
    input: Union[str, List[Any]]
    instructions: Optional[str] = None
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    top_k: Optional[int] = DEFAULT_TOP_K
    top_p: Optional[float] = DEFAULT_TOP_P
    max_output_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False
    seed: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    store: Optional[bool] = False

    def get_max_tokens(self) -> int:
        """Return the output token budget using Responses naming first."""
        if self.max_output_tokens is not None:
            return self.max_output_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return DEFAULT_MAX_TOKENS

    @staticmethod
    def _normalize_response_content(content: Any) -> Any:
        """Map Responses content part names onto Chat Completions names."""
        if not isinstance(content, list):
            return content
        parts: List[Any] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(part)
                continue
            normalized = dict(part)
            if normalized.get("type") in ("input_text", "output_text"):
                normalized["type"] = "text"
            parts.append(normalized)
        return parts

    def to_chat_messages(self) -> List[ChatMessage]:
        """Convert Responses ``input`` into chat messages."""
        messages: List[ChatMessage] = []
        if self.instructions:
            messages.append(ChatMessage(role="system", content=self.instructions))

        if isinstance(self.input, str):
            messages.append(ChatMessage(role="user", content=self.input))
            return messages

        for item in self.input:
            if isinstance(item, ChatMessage):
                messages.append(item)
                continue
            if isinstance(item, dict):
                item_type = item.get("type")
                role = item.get("role")
                content = item.get("content")
                if item_type == "message" or role is not None:
                    msg = dict(item)
                    msg.pop("type", None)
                    msg.setdefault("role", role or "user")
                    msg["content"] = self._normalize_response_content(
                        msg.get("content")
                    )
                    messages.append(ChatMessage.model_validate(msg))
                elif item_type == "input_text":
                    messages.append(ChatMessage(role="user", content=item.get("text", "")))
                elif item_type == "output_text":
                    messages.append(
                        ChatMessage(role="assistant", content=item.get("text", ""))
                    )
                elif "text" in item:
                    messages.append(ChatMessage(role="user", content=item.get("text", "")))
        if not messages or all(m.role == "system" for m in messages):
            raise ValueError("Responses input must contain at least one user or assistant message")
        return messages


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


class ResponsesResponse(BaseModel):
    """Response model for /v1/responses."""

    id: str
    object: str = RESPONSES_OBJECT
    created_at: int
    status: str
    model: str
    output: List[Dict[str, Any]]
    output_text: str
    usage: Dict[str, Any]

    model_config = ConfigDict(extra="allow")
