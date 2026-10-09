# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OpenAI Responses API adapter for ATOM, including Codex tool shapes.

Translates ``POST /v1/responses`` (and Codex's custom / namespace /
tool_search tools) to ATOM's internal chat format, then restores the
Responses wire shape on the way out. Same role as ``serving_anthropic.py``
for ``/v1/messages``.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from .protocol import STREAM_DONE_MESSAGE
from .sse import event_frame

logger = logging.getLogger("atom")

DEFAULT_MAX_OUTPUT_TOKENS = 4096
_BACKGROUND_MESSAGE = (
    "Background mode is not supported. Please set 'background' to false or omit it."
)
_PREVIOUS_RESPONSE_MESSAGE = (
    "previous_response_id is not supported on this server. "
    "Include conversation history in 'input'."
)
_CONVERSATION_MESSAGE = (
    "conversation is not supported on this server. "
    "Include conversation history in 'input'."
)


class ResponsesRequest(BaseModel):
    """Loose Responses body. Tools/input stay untyped so Codex shapes parse."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: Any = ""
    instructions: str | None = None
    stream: bool = False
    background: bool | None = None
    previous_response_id: str | None = None
    conversation: Any = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: Any = None
    stop: list[str] | None = None
    store: bool | None = None


# ── Codex tool-shape normalize / restore ────────────────────────────────


@dataclass
class CodexToolContext:
    kinds: dict[str, str] = field(default_factory=dict)
    namespaces: dict[str, str] = field(default_factory=dict)
    original_tools: Any = None

    def is_empty(self) -> bool:
        return not self.kinds and self.original_tools is None

    def kind(self, name: str) -> str:
        return self.kinds.get(name, "function")

    def namespace(self, name: str) -> str | None:
        return self.namespaces.get(name)

    def rewrite_response(self, response: dict) -> None:
        if self.original_tools is not None:
            response["tools"] = copy.deepcopy(self.original_tools)
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    self.rewrite_output_item(item)

    def rewrite_output_item(self, item: dict) -> None:
        if item.get("type") != "function_call":
            return
        name = item.get("name") or ""
        kind = self.kind(name)
        if kind == "custom":
            item["type"] = "custom_tool_call"
            item["input"] = custom_tool_input(item.get("arguments") or "")
            item.pop("arguments", None)
            namespace = self.namespace(name)
            if namespace:
                item["namespace"] = namespace
        elif kind == "tool_search":
            item["type"] = "tool_search_call"
            item["execution"] = "client"
            item["arguments"] = parse_tool_search_arguments(item.get("arguments") or "")
            item.pop("name", None)
        else:
            namespace = self.namespace(name)
            if namespace:
                item["namespace"] = namespace


def unsupported_responses_parameter(body: dict) -> str | None:
    """Why this request cannot run on the Python Responses path, or None."""
    if body.get("background") is True:
        return _BACKGROUND_MESSAGE
    if body.get("previous_response_id"):
        return _PREVIOUS_RESPONSE_MESSAGE
    if body.get("conversation"):
        return _CONVERSATION_MESSAGE
    return None


def normalize_codex_request(body: dict) -> tuple[dict, CodexToolContext]:
    """Rewrite Codex tool/history shapes into function calls.

    Mutates and returns ``body``. Caller keeps ``CodexToolContext`` for the
    response rewrite; it is not stuffed into metadata the way the mesh path
    has to, because nothing here goes through a typed protocol crate.
    """
    if not isinstance(body, dict):
        raise ValueError("Responses request must be a JSON object")  # noqa: TRY004

    original_tools = copy.deepcopy(body["tools"]) if "tools" in body else None
    context = CodexToolContext(original_tools=original_tools)
    all_tools = list(body["tools"]) if isinstance(body.get("tools"), list) else []

    if isinstance(body.get("input"), list):
        body["input"] = _normalize_input_items(body["input"], all_tools)

    normalized_tools = _normalize_tools(all_tools, context)
    if normalized_tools or "tools" in body:
        body["tools"] = normalized_tools

    return body, context


def _normalize_input_items(items: list, all_tools: list) -> list:
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        item_type = item.get("type") or ""
        if item_type == "custom_tool_call":
            call_id = _string_field(item, "call_id")
            name = _string_field(item, "name")
            raw_input = item.get("input", "")
            input_text = raw_input if isinstance(raw_input, str) else ""
            normalized.append(
                {
                    "type": "function_call",
                    "id": item.get("id") or call_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps({"input": input_text}),
                }
            )
        elif item_type == "custom_tool_call_output":
            normalized.append(
                {
                    "type": "function_call_output",
                    "id": item.get("id"),
                    "call_id": _string_field(item, "call_id"),
                    "output": _value_as_text(item.get("output")),
                }
            )
        elif item_type == "tool_search_call":
            call_id = _string_field(item, "call_id")
            normalized.append(
                {
                    "type": "function_call",
                    "id": item.get("id") or call_id,
                    "call_id": call_id,
                    "name": "tool_search",
                    "arguments": _value_as_text(item.get("arguments")),
                }
            )
        elif item_type == "tool_search_output":
            tools = item.get("tools")
            if isinstance(tools, list):
                all_tools.extend(tools)
            normalized.append(
                {
                    "type": "function_call_output",
                    "id": item.get("id"),
                    "call_id": _string_field(item, "call_id"),
                    "output": json.dumps({"tools": tools if tools is not None else []}),
                }
            )
        elif item_type == "additional_tools":
            tools = item.get("tools")
            if isinstance(tools, list):
                all_tools.extend(tools)
        else:
            normalized.append(item)
    return normalized


def _normalize_tools(tools: list, context: CodexToolContext) -> list[dict]:
    normalized: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function":
            normalized.append(copy.deepcopy(tool))
        elif tool_type == "custom":
            name = tool.get("name")
            if isinstance(name, str):
                context.kinds[name] = "custom"
                normalized.append(_custom_as_function(tool, None))
        elif tool_type == "namespace":
            namespace = tool.get("name") or "namespace"
            for child in tool.get("tools") or []:
                if not isinstance(child, dict) or not isinstance(
                    child.get("name"), str
                ):
                    continue
                name = child["name"]
                context.namespaces[name] = namespace
                if child.get("type") == "function":
                    child = copy.deepcopy(child)
                    description = child.get("description") or ""
                    child["description"] = f"[{namespace}] {description}".rstrip()
                    normalized.append(child)
                elif child.get("type") == "custom":
                    context.kinds[name] = "custom"
                    normalized.append(_custom_as_function(child, namespace))
        elif tool_type == "tool_search" and tool.get("execution") == "client":
            context.kinds["tool_search"] = "tool_search"
            normalized.append(
                {
                    "type": "function",
                    "name": "tool_search",
                    "description": tool.get("description")
                    or "Search for and load deferred client tools.",
                    "parameters": tool.get("parameters") or {"type": "object"},
                }
            )
        elif tool_type in ("web_search_preview", "code_interpreter", "mcp"):
            normalized.append(copy.deepcopy(tool))
    return normalized


def _custom_as_function(tool: dict, namespace: str | None) -> dict:
    description = tool.get("description") or ""
    if namespace:
        description = f"[{namespace}] {description}".rstrip()
    fmt = tool.get("format")
    if isinstance(fmt, dict) and fmt.get("type") == "grammar":
        syntax = fmt.get("syntax") or "grammar"
        definition = fmt.get("definition") or ""
        description += (
            f"\nReturn raw input matching this {syntax} grammar:\n{definition}"
        )
    return {
        "type": "function",
        "name": tool.get("name") or "",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Raw freeform input for this tool.",
                }
            },
            "required": ["input"],
            "additionalProperties": False,
        },
    }


def custom_tool_input(arguments: str) -> str:
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments
    if isinstance(parsed, dict):
        value = parsed.get("input")
        return value if isinstance(value, str) else arguments
    if isinstance(parsed, str):
        return parsed
    return arguments


def parse_tool_search_arguments(arguments: str) -> Any:
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {"query": arguments}


def _string_field(value: dict, key: str) -> str:
    field = value.get(key)
    return field if isinstance(field, str) else ""


def _value_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


# ── Responses → internal chat ───────────────────────────────────────────


def responses_tools_to_openai(tools: list | None) -> list[dict] | None:
    """Function tools in the nested shape ``apply_chat_template`` expects."""
    if not tools:
        return None
    result = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            result.append({"type": "function", "function": fn})
        elif isinstance(tool.get("name"), str):
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description") or "",
                        "parameters": tool.get("parameters") or {},
                    },
                }
            )
    return result or None


def responses_to_openai_messages(request: ResponsesRequest) -> list[dict]:
    """Convert a (already-normalized) Responses request to OpenAI messages."""
    messages: list[dict] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    raw_input = request.input
    if isinstance(raw_input, str) or raw_input is None:
        messages.append({"role": "user", "content": raw_input or ""})
        return messages
    if not isinstance(raw_input, list):
        raise ValueError("input must be a string or an array")  # noqa: TRY004

    for item in raw_input:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in (None, "message") and "role" in item:
            messages.append(
                _role_message(
                    item.get("role") or "user", _content_text(item.get("content"))
                )
            )
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or ""
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "",
                            },
                        }
                    ],
                }
            )
            if item.get("output") is not None:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _value_as_text(item.get("output")),
                    }
                )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "",
                    "content": _value_as_text(item.get("output")),
                }
            )
        elif item_type == "reasoning":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": _reasoning_text(item.get("content")),
                }
            )
        elif "role" in item:
            messages.append(
                _role_message(
                    item.get("role") or "user", _content_text(item.get("content"))
                )
            )
    if not messages:
        raise ValueError("Request must contain at least one message")
    return messages


def _role_message(role: str, text: str) -> dict:
    if role in ("user", "assistant", "system"):
        return {"role": role, "content": text}
    return {"role": "user", "content": text}


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in (
                "input_text",
                "output_text",
                "text",
                None,
            ):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _reasoning_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text") or ""
            for part in content
            if isinstance(part, dict)
            and part.get("type") in ("reasoning_text", "summary_text", None)
        )
    return ""


# ── Non-streaming response ──────────────────────────────────────────────


def build_responses_response(
    request_id: str,
    model: str,
    events: list,
    *,
    context: CodexToolContext,
    request: ResponsesRequest,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    created_at: int | None = None,
) -> dict:
    """One Responses object from ordered engine events, then Codex rewrite."""
    output = _output_items_from_events(events, request_id, context)
    response = {
        "id": request_id,
        "object": "response",
        "created_at": created_at if created_at is not None else int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": cache_read_input_tokens},
        },
        "parallel_tool_calls": (
            True if request.parallel_tool_calls is None else request.parallel_tool_calls
        ),
        "store": False,
        "tools": request.tools or [],
        "metadata": request.metadata or {},
        "tool_choice": (
            request.tool_choice if request.tool_choice is not None else "auto"
        ),
    }
    if request.instructions is not None:
        response["instructions"] = request.instructions
    if request.max_output_tokens is not None:
        response["max_output_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        response["temperature"] = request.temperature
    if request.top_p is not None:
        response["top_p"] = request.top_p
    context.rewrite_response(response)
    return response


def _output_items_from_events(
    events: list, request_id: str, context: CodexToolContext
) -> list[dict]:
    output: list[dict] = []
    pending: dict | None = None
    for etype, data in events:
        if not data and etype in ("reasoning", "content"):
            continue
        if etype == "reasoning":
            output.append(
                {
                    "id": f"reasoning_{request_id}",
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": data}],
                    "status": "completed",
                }
            )
        elif etype == "content":
            if output and output[-1].get("type") == "message":
                parts = output[-1].setdefault("content", [])
                if parts and parts[-1].get("type") in ("output_text", "text"):
                    parts[-1]["text"] += data
                else:
                    parts.append({"type": "output_text", "text": data})
            else:
                output.append(
                    {
                        "id": f"msg_{request_id}",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": data}],
                        "status": "completed",
                    }
                )
        elif etype == "tool_call_start":
            fn = data.get("function") or {}
            pending = {
                "id": data.get("id") or "",
                "name": fn.get("name") or "",
            }
        elif etype == "tool_call_args" and pending is not None:
            item = {
                "id": pending["id"],
                "type": "function_call",
                "call_id": pending["id"],
                "name": pending["name"],
                "arguments": (data.get("function") or {}).get("arguments") or "",
                "status": "completed",
            }
            context.rewrite_output_item(item)
            output.append(item)
            pending = None
    return output


# ── Streaming ───────────────────────────────────────────────────────────


@dataclass
class _ToolCallState:
    output_index: int = -1
    item_id: str = ""
    call_id: str = ""
    name: str = ""
    accumulated_args: str = ""
    added: bool = False


class ResponseStreamEmitter:
    """Responses SSE sequence, including Codex custom / tool_search shapes."""

    def __init__(
        self,
        response_id: str,
        model: str,
        *,
        context: CodexToolContext | None = None,
        request: ResponsesRequest | None = None,
        created_at: int | None = None,
    ) -> None:
        self.response_id = response_id
        self.model = model
        self.created_at = created_at if created_at is not None else int(time.time())
        self.context = context or CodexToolContext()
        self.request = request
        self.sequence_number = 0
        self.message_id = f"msg_{uuid.uuid4().hex}"
        self.accumulated_text = ""
        self.reasoning_buffer = ""
        self.has_emitted_output_item_added = False
        self.has_emitted_content_part_added = False
        self.current_message_output_index: int | None = None
        self.current_item_id: str | None = None
        self.next_output_index = 0
        self.completed_items: list[dict] = []
        self.tool_call_items: list[_ToolCallState] = []

    def opening_frames(self) -> list[str]:
        return [
            self._event(
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": self._next_sequence(),
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "created_at": self.created_at,
                        "status": "in_progress",
                        "model": self.model,
                        "output": [],
                    },
                },
            ),
            self._event(
                "response.in_progress",
                {
                    "type": "response.in_progress",
                    "sequence_number": self._next_sequence(),
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "status": "in_progress",
                    },
                },
            ),
        ]

    def frames_for_events(self, events: list) -> list[str]:
        frames: list[str] = []
        for etype, data in events:
            frames.extend(self._on_event(etype, data))
        return frames

    def finish_frames(
        self,
        *,
        usage: dict | None = None,
        has_tool_calls: bool = False,
    ) -> list[str]:
        frames: list[str] = []
        frames.extend(self._flush_reasoning())
        if has_tool_calls or any(state.added for state in self.tool_call_items):
            frames.extend(self._finish_tool_calls())
        else:
            frames.extend(self._finish_message())
        frames.append(self._completed_frame(usage))
        frames.append(STREAM_DONE_MESSAGE)
        return frames

    def error_frame(self, message: str, error_type: str = "api_error") -> str:
        return self._event(
            "error",
            {
                "type": "error",
                "code": error_type,
                "message": message,
                "param": None,
                "sequence_number": self._next_sequence(),
            },
        )

    def _on_event(self, etype: str, data: Any) -> list[str]:
        if etype == "reasoning":
            if not data:
                return []
            self.reasoning_buffer += data
            return []
        if etype == "content":
            if not data:
                return []
            frames = self._flush_reasoning()
            frames.extend(self._on_text(data))
            return frames
        if etype == "tool_call_start":
            frames = self._flush_reasoning()
            frames.extend(self._close_message_if_open())
            frames.extend(self._on_tool_start(data or {}))
            return frames
        if etype == "tool_call_args":
            return self._on_tool_args(data or {})
        return []

    def _on_text(self, text: str) -> list[str]:
        frames: list[str] = []
        if self.current_item_id is None:
            output_index, item_id = self._allocate("msg")
            item = {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [],
            }
            frames.append(self._output_item_added(output_index, item))
            self.has_emitted_output_item_added = True
            self.current_item_id = item_id
            self.current_message_output_index = output_index
        output_index = self.current_message_output_index
        item_id = self.current_item_id
        assert output_index is not None and item_id is not None
        if not self.has_emitted_content_part_added:
            frames.append(
                self._event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "sequence_number": self._next_sequence(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "part": {"type": "text", "text": ""},
                    },
                )
            )
            self.has_emitted_content_part_added = True
        self.accumulated_text += text
        frames.append(
            self._event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": self._next_sequence(),
                    "output_index": output_index,
                    "item_id": item_id,
                    "content_index": 0,
                    "delta": text,
                },
            )
        )
        return frames

    def _on_tool_start(self, data: dict) -> list[str]:
        index = int(data.get("index") or 0)
        while len(self.tool_call_items) <= index:
            self.tool_call_items.append(_ToolCallState())
        state = self.tool_call_items[index]
        fn = data.get("function") or {}
        if data.get("id"):
            state.call_id = data["id"]
        if fn.get("name"):
            state.name = fn["name"]
        return self._maybe_add_tool_item(state)

    def _on_tool_args(self, data: dict) -> list[str]:
        index = int(data.get("index") or 0)
        while len(self.tool_call_items) <= index:
            self.tool_call_items.append(_ToolCallState())
        state = self.tool_call_items[index]
        args = (data.get("function") or {}).get("arguments") or ""
        if args:
            state.accumulated_args += args
        frames = self._maybe_add_tool_item(state)
        if state.added and self.context.kind(state.name) == "function" and args:
            frames.append(
                self._event(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": self._next_sequence(),
                        "output_index": state.output_index,
                        "item_id": state.item_id,
                        "delta": args,
                    },
                )
            )
        return frames

    def _maybe_add_tool_item(self, state: _ToolCallState) -> list[str]:
        if state.added or not state.call_id or not state.name:
            return []
        kind = self.context.kind(state.name)
        prefix = {"custom": "ctc", "tool_search": "tsc"}.get(kind, "fc")
        output_index, item_id = self._allocate(prefix)
        state.output_index = output_index
        state.item_id = item_id
        state.added = True
        if kind == "custom":
            item = {
                "id": item_id,
                "type": "custom_tool_call",
                "call_id": state.call_id,
                "name": state.name,
                "input": "",
                "status": "in_progress",
            }
        elif kind == "tool_search":
            item = {
                "id": item_id,
                "type": "tool_search_call",
                "call_id": state.call_id,
                "execution": "client",
                "arguments": {},
                "status": "in_progress",
            }
        else:
            item = {
                "id": item_id,
                "type": "function_call",
                "call_id": state.call_id,
                "name": state.name,
                "arguments": "",
                "status": "in_progress",
            }
        namespace = self.context.namespace(state.name)
        if namespace:
            item["namespace"] = namespace
        frames = [self._output_item_added(output_index, item)]
        if kind == "function" and state.accumulated_args:
            frames.append(
                self._event(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": self._next_sequence(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "delta": state.accumulated_args,
                    },
                )
            )
        return frames

    def _finish_tool_calls(self) -> list[str]:
        frames: list[str] = []
        frames.extend(self._close_message_if_open())
        for state in self.tool_call_items:
            if not state.added:
                logger.warning(
                    "Skipping incomplete streamed tool call without id or name"
                )
                continue
            kind = self.context.kind(state.name)
            namespace = self.context.namespace(state.name)
            if kind == "custom":
                input_text = custom_tool_input(state.accumulated_args)
                if input_text:
                    frames.append(
                        self._event(
                            "response.custom_tool_call_input.delta",
                            {
                                "type": "response.custom_tool_call_input.delta",
                                "sequence_number": self._next_sequence(),
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "call_id": state.call_id,
                                "delta": input_text,
                            },
                        )
                    )
                frames.append(
                    self._event(
                        "response.custom_tool_call_input.done",
                        {
                            "type": "response.custom_tool_call_input.done",
                            "sequence_number": self._next_sequence(),
                            "output_index": state.output_index,
                            "item_id": state.item_id,
                            "call_id": state.call_id,
                            "input": input_text,
                        },
                    )
                )
                item = {
                    "id": state.item_id,
                    "type": "custom_tool_call",
                    "call_id": state.call_id,
                    "name": state.name,
                    "input": input_text,
                    "status": "completed",
                }
            elif kind == "tool_search":
                item = {
                    "id": state.item_id,
                    "type": "tool_search_call",
                    "call_id": state.call_id,
                    "execution": "client",
                    "arguments": parse_tool_search_arguments(state.accumulated_args),
                    "status": "completed",
                }
            else:
                frames.append(
                    self._event(
                        "response.function_call_arguments.done",
                        {
                            "type": "response.function_call_arguments.done",
                            "sequence_number": self._next_sequence(),
                            "output_index": state.output_index,
                            "item_id": state.item_id,
                            "arguments": state.accumulated_args,
                        },
                    )
                )
                item = {
                    "id": state.item_id,
                    "type": "function_call",
                    "call_id": state.call_id,
                    "name": state.name,
                    "arguments": state.accumulated_args,
                    "status": "completed",
                }
            if namespace:
                item["namespace"] = namespace
            frames.append(self._output_item_done(state.output_index, item))
        return frames

    def _finish_message(self) -> list[str]:
        frames: list[str] = []
        if self.current_item_id is None:
            return frames
        output_index = self.current_message_output_index
        item_id = self.current_item_id
        assert output_index is not None and item_id is not None
        if self.has_emitted_content_part_added:
            frames.append(
                self._event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "sequence_number": self._next_sequence(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "text": self.accumulated_text,
                    },
                )
            )
            frames.append(
                self._event(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "sequence_number": self._next_sequence(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "part": {"type": "text", "text": self.accumulated_text},
                    },
                )
            )
        if self.has_emitted_output_item_added:
            item = {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.accumulated_text}],
                "status": "completed",
            }
            frames.append(self._output_item_done(output_index, item))
        self.current_item_id = None
        return frames

    def _close_message_if_open(self) -> list[str]:
        if self.current_item_id is None:
            return []
        return self._finish_message()

    def _flush_reasoning(self) -> list[str]:
        if not self.reasoning_buffer:
            return []
        text = self.reasoning_buffer
        self.reasoning_buffer = ""
        output_index, item_id = self._allocate("rs")
        item = {
            "id": item_id,
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": text}],
            "status": "completed",
        }
        return [
            self._output_item_added(output_index, item),
            self._output_item_done(output_index, item),
        ]

    def _completed_frame(self, usage: dict | None) -> str:
        output = list(self.completed_items)
        if not output:
            output = [
                {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self.accumulated_text}],
                }
            ]
        response_obj: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": output,
            "store": False,
            "parallel_tool_calls": True,
        }
        if usage is not None:
            response_obj["usage"] = usage
        req = self.request
        if req is not None:
            if req.instructions is not None:
                response_obj["instructions"] = req.instructions
            if req.max_output_tokens is not None:
                response_obj["max_output_tokens"] = req.max_output_tokens
            if req.temperature is not None:
                response_obj["temperature"] = req.temperature
            if req.top_p is not None:
                response_obj["top_p"] = req.top_p
            response_obj["parallel_tool_calls"] = (
                True if req.parallel_tool_calls is None else req.parallel_tool_calls
            )
            response_obj["tools"] = req.tools or []
            response_obj["metadata"] = req.metadata or {}
            response_obj["tool_choice"] = (
                req.tool_choice if req.tool_choice is not None else "auto"
            )
        self.context.rewrite_response(response_obj)
        return self._event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": self._next_sequence(),
                "response": response_obj,
            },
        )

    def _allocate(self, prefix: str) -> tuple[int, str]:
        index = self.next_output_index
        self.next_output_index += 1
        return index, f"{prefix}_{uuid.uuid4().hex}"

    def _output_item_added(self, output_index: int, item: dict) -> str:
        return self._event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": self._next_sequence(),
                "output_index": output_index,
                "item": item,
            },
        )

    def _output_item_done(self, output_index: int, item: dict) -> str:
        self.completed_items.append(copy.deepcopy(item))
        return self._event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": self._next_sequence(),
                "output_index": output_index,
                "item": item,
            },
        )

    def _event(self, event: str, payload: dict) -> str:
        return event_frame(event, payload)

    def _next_sequence(self) -> int:
        seq = self.sequence_number
        self.sequence_number += 1
        return seq


def stream_failure_frames(
    exc: BaseException, emitter: ResponseStreamEmitter, *, opened: bool
):
    if not opened:
        yield from emitter.opening_frames()
    yield emitter.error_frame(str(exc))
    yield STREAM_DONE_MESSAGE


def completes_a_tool_call(events) -> bool:
    return any(etype == "tool_call_args" for etype, _ in events)
