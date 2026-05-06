# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pydantic request/response models for the OpenAI-compatible API."""

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
                d[key] = extras[key]
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
