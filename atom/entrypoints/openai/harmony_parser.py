# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Token-level streaming parser for GPT-OSS Harmony output.

Wraps ``openai_harmony.StreamableParser`` to process raw token IDs and emit
structured events compatible with ATOM's tool-call streaming protocol:

- ``("reasoning", text)``   — chain-of-thought (analysis channel)
- ``("content", text)``     — final response content
- ``("tool_call_start", {"id": ..., "function": {"name": ..., "arguments": ""}})``
- ``("tool_call_args", {"function": {"arguments": chunk}})``
- ``("tool_call_end", None)``

Adapted from vLLM's ``vllm/parser/harmony.py``.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openai_harmony import HarmonyError

from .harmony_utils import (
    extract_function_from_recipient,
    get_streamable_parser_for_assistant,
    is_function_recipient,
)

logger = logging.getLogger("atom")


def _make_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


class HarmonyStreamParser:
    """Stateful token-level parser for GPT-OSS Harmony output.

    Feed raw token IDs via :meth:`process_tokens` and call :meth:`flush` at
    end-of-stream.  Events match the ``ToolCallStreamParser`` protocol so the
    same downstream SSE formatting code can consume them.
    """

    def __init__(self) -> None:
        self._parser = get_streamable_parser_for_assistant()
        self._prev_recipient: Optional[str] = None
        self._tool_call_index: int = 0
        self._num_processed_messages: int = 0
        self._has_tool_calls: bool = False

    @property
    def has_tool_calls(self) -> bool:
        return self._has_tool_calls

    def _normalize_recipient(self, recipient: Optional[str]) -> Optional[str]:
        """Remove constrained-format suffixes misparsed by older Harmony."""
        if recipient is None:
            return None
        idx = recipient.find("<|constrain|>")
        if idx == -1:
            return recipient
        return recipient[:idx].rstrip() or None

    def _poll_completed_message(self):
        """Check if a new message has been completed by the parser."""
        messages = self._parser.messages
        if len(messages) <= self._num_processed_messages:
            return None
        msg = messages[self._num_processed_messages]
        msg.recipient = self._normalize_recipient(msg.recipient)
        self._num_processed_messages += 1
        return msg

    def _classify(self, channel, recipient):
        """Route a segment to reasoning, content, tool, or ignore."""
        if recipient and is_function_recipient(recipient):
            return "tool"
        if channel == "analysis":
            return "reasoning"
        if channel == "final" or (channel == "commentary" and recipient is None):
            return "content"
        return "ignore"

    def process_tokens(self, token_ids: List[int]) -> List[Tuple[str, Any]]:
        """Process a batch of raw token IDs and return structured events."""
        if not token_ids:
            return []

        events: List[Tuple[str, Any]] = []

        for token_id in token_ids:
            self._parser.process(token_id)
            channel = self._parser.current_channel
            recipient = self._normalize_recipient(self._parser.current_recipient)
            delta = self._parser.last_content_delta or ""
            completed = self._poll_completed_message()

            # A completed message resets the recipient tracking
            if completed is not None:
                self._prev_recipient = None
                continue

            if not delta:
                continue

            seg_type = self._classify(channel, recipient)

            if seg_type == "reasoning":
                events.append(("reasoning", delta))

            elif seg_type == "content":
                events.append(("content", delta))

            elif seg_type == "tool":
                self._has_tool_calls = True
                if self._prev_recipient != recipient:
                    # New tool call starting
                    tool_name = extract_function_from_recipient(recipient)
                    tool_id = _make_tool_call_id()
                    events.append((
                        "tool_call_start",
                        {
                            "index": self._tool_call_index,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": ""},
                        },
                    ))
                    # Also emit the first args delta
                    events.append((
                        "tool_call_args",
                        {
                            "index": self._tool_call_index,
                            "function": {"arguments": delta},
                        },
                    ))
                    self._tool_call_index += 1
                    self._prev_recipient = recipient
                else:
                    # Continuing same tool call
                    events.append((
                        "tool_call_args",
                        {
                            "index": self._tool_call_index - 1,
                            "function": {"arguments": delta},
                        },
                    ))

        return events

    def flush(self) -> List[Tuple[str, Any]]:
        """Finalize parsing at end-of-stream."""
        events: List[Tuple[str, Any]] = []
        try:
            self._parser.process_eos()
            msg = self._poll_completed_message()
            if msg is not None:
                # Process the final completed message
                text = ""
                if msg.content:
                    text = msg.content[0].text if hasattr(msg.content[0], "text") else str(msg.content[0])
                seg_type = self._classify(msg.channel, msg.recipient)
                if seg_type == "content" and text:
                    events.append(("content", text))
                elif seg_type == "tool" and msg.recipient:
                    self._has_tool_calls = True
                    tool_name = extract_function_from_recipient(msg.recipient)
                    content_type = msg.content_type
                    if content_type is not None and "json" not in str(content_type):
                        arguments = text
                    else:
                        try:
                            arguments = json.dumps(json.loads(text))
                        except (json.JSONDecodeError, TypeError):
                            arguments = text
                    tool_id = _make_tool_call_id()
                    events.append((
                        "tool_call_start",
                        {
                            "index": self._tool_call_index,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": ""},
                        },
                    ))
                    events.append((
                        "tool_call_args",
                        {
                            "index": self._tool_call_index,
                            "function": {"arguments": arguments},
                        },
                    ))
                    self._tool_call_index += 1
        except HarmonyError:
            logger.warning(
                "Harmony parser ended in a non-terminal state; "
                "raw unparsed output may be incomplete."
            )

        if self._has_tool_calls:
            events.append(("tool_call_end", None))

        return events

    def parse_full(self, token_ids: List[int]) -> Tuple[Optional[str], Optional[str], list]:
        """Non-streaming parse: returns (reasoning, content, tool_calls).

        ``tool_calls`` is a list of dicts with ``id``, ``type``, ``function``
        keys (OpenAI format).
        """
        events = self.process_tokens(token_ids)
        events.extend(self.flush())

        reasoning_parts: List[str] = []
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        current_tool: Optional[Dict[str, Any]] = None

        for etype, edata in events:
            if etype == "reasoning":
                reasoning_parts.append(edata)
            elif etype == "content":
                content_parts.append(edata)
            elif etype == "tool_call_start":
                if current_tool is not None:
                    tool_calls.append(current_tool)
                current_tool = {
                    "id": edata["id"],
                    "type": "function",
                    "function": {
                        "name": edata["function"]["name"],
                        "arguments": "",
                    },
                }
            elif etype == "tool_call_args" and current_tool is not None:
                current_tool["function"]["arguments"] += edata["function"]["arguments"]
            elif etype == "tool_call_end":
                if current_tool is not None:
                    tool_calls.append(current_tool)
                    current_tool = None

        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        return reasoning, content, tool_calls
