# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Responses API serving helpers.

This implements a compatibility subset of OpenAI's Responses API on top of
ATOM's existing chat-completions generation path. The streaming event names
and payload shapes follow vLLM/OpenAI closely enough for SDK clients that
consume text, reasoning, and function-call deltas.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from .protocol import STREAM_DONE_MESSAGE, ResponsesResponse
from .reasoning import ReasoningFilter, separate_reasoning
from .tool_parser import ToolCallStreamParser, parse_tool_calls


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _response_base(request_id: str, model: str, status: str = "in_progress") -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": [],
        "usage": None,
    }


def _event(event_type: str, request_id: str, model: str, **extra: Any) -> Dict[str, Any]:
    event = {
        "type": event_type,
        "sequence_number": -1,
        "response": _response_base(request_id, model),
    }
    event.update(extra)
    return event


def _completed_event(
    request_id: str,
    model: str,
    output: List[Dict[str, Any]],
    output_text: str,
    usage: Dict[str, Any],
) -> Dict[str, Any]:
    response = _response_base(request_id, model, status="completed")
    response["output"] = output
    response["output_text"] = output_text
    response["usage"] = usage
    return {"type": "response.completed", "sequence_number": -1, "response": response}


def _message_item(item_id: str, status: str, content: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content or [],
    }


def _reasoning_item(item_id: str, status: str, text: str = "") -> Dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}] if text else [],
    }


def _function_call_item(call_id: str, name: str, arguments: str, status: str) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


class _ResponsesStreamState:
    def __init__(self) -> None:
        self.next_output_index = 0
        self.message_output_index: Optional[int] = None
        self.reasoning_output_index: Optional[int] = None
        self.message_content_index = 0
        self.reasoning_content_index = 0
        self.message_item_id = ""
        self.reasoning_item_id = ""
        self.sent_message_item = False
        self.sent_reasoning_item = False
        self.content_text = ""
        self.reasoning_text = ""
        self.tool_calls: Dict[int, Dict[str, Any]] = {}

    def ensure_message(self) -> List[Dict[str, Any]]:
        if self.sent_message_item:
            return []
        self.sent_message_item = True
        self.message_item_id = f"msg_{uuid.uuid4().hex}"
        self.message_output_index = self.next_output_index
        self.next_output_index += 1
        return [
            {
                "type": "response.output_item.added",
                "sequence_number": -1,
                "output_index": self.message_output_index,
                "item": _message_item(self.message_item_id, "in_progress"),
            },
            {
                "type": "response.content_part.added",
                "sequence_number": -1,
                "output_index": self.message_output_index,
                "item_id": self.message_item_id,
                "content_index": self.message_content_index,
                "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
            },
        ]

    def ensure_reasoning(self) -> List[Dict[str, Any]]:
        if self.sent_reasoning_item:
            return []
        self.sent_reasoning_item = True
        self.reasoning_item_id = f"rs_{uuid.uuid4().hex}"
        self.reasoning_output_index = self.next_output_index
        self.next_output_index += 1
        return [
            {
                "type": "response.output_item.added",
                "sequence_number": -1,
                "output_index": self.reasoning_output_index,
                "item": _reasoning_item(self.reasoning_item_id, "in_progress"),
            },
            {
                "type": "response.reasoning_part.added",
                "sequence_number": -1,
                "output_index": self.reasoning_output_index,
                "item_id": self.reasoning_item_id,
                "content_index": self.reasoning_content_index,
                "part": {"type": "reasoning_text", "text": ""},
            },
        ]

    def emit_content_delta(self, delta: str) -> List[Dict[str, Any]]:
        events = self.ensure_message()
        self.content_text += delta
        events.append(
            {
                "type": "response.output_text.delta",
                "sequence_number": -1,
                "output_index": self.message_output_index,
                "item_id": self.message_item_id,
                "content_index": self.message_content_index,
                "delta": delta,
                "logprobs": [],
            }
        )
        return events

    def emit_reasoning_delta(self, delta: str) -> List[Dict[str, Any]]:
        events = self.ensure_reasoning()
        self.reasoning_text += delta
        events.append(
            {
                "type": "response.reasoning_text.delta",
                "sequence_number": -1,
                "output_index": self.reasoning_output_index,
                "item_id": self.reasoning_item_id,
                "content_index": self.reasoning_content_index,
                "delta": delta,
            }
        )
        return events

    def emit_tool_event(self, event_type: str, data: Any) -> List[Dict[str, Any]]:
        if event_type == "tool_call_start":
            idx = int(data.get("index", len(self.tool_calls)))
            function = data.get("function", {})
            call = {
                "id": data.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                "name": function.get("name", ""),
                "arguments": "",
            }
            self.tool_calls[idx] = call
            output_index = self.next_output_index
            self.next_output_index += 1
            call["output_index"] = output_index
            return [
                {
                    "type": "response.output_item.added",
                    "sequence_number": -1,
                    "output_index": output_index,
                    "item": _function_call_item(call["id"], call["name"], "", "in_progress"),
                }
            ]
        if event_type == "tool_call_args":
            idx = int(data.get("index", 0))
            call = self.tool_calls.setdefault(
                idx,
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "",
                    "arguments": "",
                    "output_index": self.next_output_index,
                },
            )
            if call["output_index"] == self.next_output_index:
                self.next_output_index += 1
            delta = (data.get("function") or {}).get("arguments", "")
            call["arguments"] += delta
            return [
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": -1,
                    "output_index": call["output_index"],
                    "item_id": call["id"],
                    "delta": delta,
                }
            ]
        return []

    def done_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if self.sent_reasoning_item:
            events.append(
                {
                    "type": "response.reasoning_text.done",
                    "sequence_number": -1,
                    "output_index": self.reasoning_output_index,
                    "item_id": self.reasoning_item_id,
                    "content_index": self.reasoning_content_index,
                    "text": self.reasoning_text,
                }
            )
            events.append(
                {
                    "type": "response.output_item.done",
                    "sequence_number": -1,
                    "output_index": self.reasoning_output_index,
                    "item": _reasoning_item(self.reasoning_item_id, "completed", self.reasoning_text),
                }
            )
        if self.sent_message_item:
            part = {
                "type": "output_text",
                "text": self.content_text,
                "annotations": [],
                "logprobs": [],
            }
            events.append(
                {
                    "type": "response.output_text.done",
                    "sequence_number": -1,
                    "output_index": self.message_output_index,
                    "item_id": self.message_item_id,
                    "content_index": self.message_content_index,
                    "text": self.content_text,
                }
            )
            events.append(
                {
                    "type": "response.content_part.done",
                    "sequence_number": -1,
                    "output_index": self.message_output_index,
                    "item_id": self.message_item_id,
                    "content_index": self.message_content_index,
                    "part": part,
                }
            )
            events.append(
                {
                    "type": "response.output_item.done",
                    "sequence_number": -1,
                    "output_index": self.message_output_index,
                    "item": _message_item(self.message_item_id, "completed", [part]),
                }
            )
        for idx, call in sorted(self.tool_calls.items()):
            events.append(
                {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": -1,
                    "output_index": call["output_index"],
                    "item_id": call["id"],
                    "arguments": call["arguments"],
                }
            )
            events.append(
                {
                    "type": "response.output_item.done",
                    "sequence_number": -1,
                    "output_index": call["output_index"],
                    "item": _function_call_item(
                        call["id"], call["name"], call["arguments"], "completed"
                    ),
                }
            )
        return events

    def output_items(self) -> List[Dict[str, Any]]:
        indexed_items: List[tuple[int, Dict[str, Any]]] = []
        if self.reasoning_text:
            indexed_items.append(
                (
                    self.reasoning_output_index or 0,
                    _reasoning_item(
                        self.reasoning_item_id or f"rs_{uuid.uuid4().hex}",
                        "completed",
                        self.reasoning_text,
                    ),
                )
            )
        if self.content_text:
            part = {"type": "output_text", "text": self.content_text, "annotations": [], "logprobs": []}
            indexed_items.append(
                (
                    self.message_output_index or 0,
                    _message_item(
                        self.message_item_id or f"msg_{uuid.uuid4().hex}",
                        "completed",
                        [part],
                    ),
                )
            )
        for call in self.tool_calls.values():
            indexed_items.append(
                (
                    call["output_index"],
                    _function_call_item(call["id"], call["name"], call["arguments"], "completed"),
                )
            )
        return [item for _, item in sorted(indexed_items, key=lambda entry: entry[0])]


def build_responses_response(
    request_id: str,
    model: str,
    raw_text: str,
    final_output: Dict[str, Any],
) -> ResponsesResponse:
    reasoning_content, content_with_tools = separate_reasoning(raw_text)
    content, tool_calls = parse_tool_calls(content_with_tools)
    output: List[Dict[str, Any]] = []
    if reasoning_content:
        output.append(_reasoning_item(f"rs_{uuid.uuid4().hex}", "completed", reasoning_content))
    if content:
        part = {"type": "output_text", "text": content, "annotations": [], "logprobs": []}
        output.append(_message_item(f"msg_{uuid.uuid4().hex}", "completed", [part]))
    for tool_call in tool_calls:
        output.append(
            _function_call_item(
                tool_call.id,
                tool_call.function.get("name", ""),
                tool_call.function.get("arguments", ""),
                "completed",
            )
        )
    return ResponsesResponse(
        id=request_id,
        created_at=int(time.time()),
        status="completed",
        model=model,
        output=output,
        output_text=content,
        usage={
            "input_tokens": final_output["num_tokens_input"],
            "output_tokens": final_output["num_tokens_output"],
            "total_tokens": final_output["num_tokens_input"] + final_output["num_tokens_output"],
        },
    )


async def stream_responses_response(
    request_id: str,
    model: str,
    prompt: str,
    stream_queue: asyncio.Queue,
    seq_id: int,
    tokenizer,
    cleanup_fn,
) -> AsyncGenerator[str, None]:
    num_tokens_input = len(tokenizer.encode(prompt))
    num_tokens_output = 0
    state = _ResponsesStreamState()
    reasoning_filter = ReasoningFilter()
    tool_parser = ToolCallStreamParser()

    yield _sse(_event("response.created", request_id, model))
    yield _sse(_event("response.in_progress", request_id, model))

    while True:
        chunk_data = await stream_queue.get()
        new_text = chunk_data["text"]
        num_tokens_output += len(chunk_data.get("token_ids", []))
        segments = reasoning_filter.process(new_text)
        if chunk_data.get("finished", False):
            segments.extend(reasoning_filter.flush())

        for field, text in segments:
            if field == "reasoning_content" and text:
                for event in state.emit_reasoning_delta(text):
                    yield _sse(event)
            elif field == "content" and text:
                for event_type, data in tool_parser.process(text):
                    if event_type == "content" and data:
                        for event in state.emit_content_delta(data):
                            yield _sse(event)
                    elif event_type.startswith("tool_call"):
                        for event in state.emit_tool_event(event_type, data):
                            yield _sse(event)

        if chunk_data.get("finished", False):
            for event_type, data in tool_parser.flush():
                if event_type == "content" and data:
                    for event in state.emit_content_delta(data):
                        yield _sse(event)
                elif event_type.startswith("tool_call"):
                    for event in state.emit_tool_event(event_type, data):
                        yield _sse(event)
            break

    cleanup_fn(request_id, seq_id)

    for event in state.done_events():
        yield _sse(event)

    usage = {
        "input_tokens": num_tokens_input,
        "output_tokens": num_tokens_output,
        "total_tokens": num_tokens_input + num_tokens_output,
    }
    yield _sse(_completed_event(request_id, model, state.output_items(), state.content_text, usage))
    yield STREAM_DONE_MESSAGE
