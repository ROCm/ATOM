# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tool call parser for models that output tool calls via special tokens.

Parses tool call special tokens (e.g., from Kimi-K2) into OpenAI-compatible
tool_calls format.

Model output format:
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>ARGS_JSON<|tool_call_end|>
    <|tool_calls_section_end|>

OpenAI format:
    {"tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "NAME", "arguments": "ARGS_JSON"}}]}
"""

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ToolCall:
    """Parsed tool call in OpenAI format."""

    id: str
    type: str
    function: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type, "function": self.function}


def parse_tool_calls(text: str) -> Tuple[str, List[ToolCall]]:
    """Parse tool calls from model output text.

    Args:
        text: Raw model output that may contain tool call special tokens.

    Returns:
        Tuple of (content_text, list_of_tool_calls).
        content_text has tool call sections removed.
        list_of_tool_calls contains parsed ToolCall objects.
    """
    # Check for tool call section
    section_match = re.search(
        r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>",
        text,
        flags=re.DOTALL,
    )
    if not section_match:
        # Check for unclosed section
        unclosed = re.search(
            r"<\|tool_calls_section_begin\|>(.*?)$", text, flags=re.DOTALL
        )
        if unclosed:
            content = text[: unclosed.start()]
            tool_calls = _parse_tool_call_entries(unclosed.group(1))
            return content.strip(), tool_calls
        return text, []

    content = text[: section_match.start()]
    tool_calls = _parse_tool_call_entries(section_match.group(1))

    return content.strip(), tool_calls


def _parse_tool_call_entries(section_text: str) -> List[ToolCall]:
    """Parse individual tool call entries from the section content."""
    tool_calls = []
    pattern = re.compile(
        r"<\|tool_call_begin\|>"
        r"functions\.(\w+):(\d+)"
        r"<\|tool_call_argument_begin\|>"
        r"(.*?)"
        r"<\|tool_call_end\|>",
        re.DOTALL,
    )
    for match in pattern.finditer(section_text):
        name = match.group(1)
        _index = match.group(2)  # noqa: F841 (captured but not used in ID generation)
        arguments = match.group(3).strip()
        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function={"name": name, "arguments": arguments},
            )
        )
    return tool_calls


@dataclass
class ToolCallStreamParser:
    """Stateful streaming parser for tool call special tokens.

    Processes text chunks and emits structured events:
    - ("content", text) — regular content before tool calls
    - ("tool_call_start", {"index": N, "id": ..., "function": {"name": ..., "arguments": ""}})
    - ("tool_call_args", {"index": N, "function": {"arguments": chunk}})
    - ("tool_call_end", None) — all tool calls complete

    States:
        0 = normal content (no tool call tokens seen)
        1 = inside tool_calls_section (buffering)
        2 = done (after tool_calls_section_end)
    """

    state: int = 0
    buf: str = ""
    current_index: int = 0
    _emitted_calls: int = 0

    def process(self, text: str) -> list:
        """Process a text chunk and return list of (event_type, data) tuples."""
        results = []

        if self.state == 0:
            self.buf += text
            if "<|tool_calls_section_begin|>" in self.buf:
                before = self.buf.split("<|tool_calls_section_begin|>")[0]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf.split("<|tool_calls_section_begin|>", 1)[1]
                # Process any complete tool calls already in buffer
                results.extend(self._process_buffer())
            elif "<|tool" not in self.buf and len(self.buf) > 30:
                # Safe to emit as content
                results.append(("content", self.buf))
                self.buf = ""

        elif self.state == 1:
            self.buf += text
            if "<|tool_calls_section_end|>" in self.buf:
                # Process remaining before end
                remaining = self.buf.split("<|tool_calls_section_end|>")[0]
                self.buf = remaining
                results.extend(self._process_buffer())
                results.append(("tool_call_end", None))
                self.state = 2
                self.buf = ""
            else:
                results.extend(self._process_buffer())

        # state 2: done, ignore further input

        return results

    def _process_buffer(self) -> list:
        """Extract complete tool call entries from the buffer."""
        results = []
        while "<|tool_call_begin|>" in self.buf and "<|tool_call_end|>" in self.buf:
            # Extract one complete tool call
            match = re.search(
                r"<\|tool_call_begin\|>"
                r"functions\.(\w+):(\d+)"
                r"<\|tool_call_argument_begin\|>"
                r"(.*?)"
                r"<\|tool_call_end\|>",
                self.buf,
                re.DOTALL,
            )
            if not match:
                break

            name = match.group(1)
            index = int(match.group(2))
            arguments = match.group(3).strip()

            call_id = f"call_{uuid.uuid4().hex[:8]}"
            results.append(
                (
                    "tool_call_start",
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    },
                )
            )
            if arguments:
                results.append(
                    (
                        "tool_call_args",
                        {"index": index, "function": {"arguments": arguments}},
                    )
                )

            self.buf = self.buf[match.end() :]
            self._emitted_calls += 1

        return results

    def flush(self) -> list:
        """Flush remaining buffer content."""
        results = []
        if self.state == 0 and self.buf:
            results.append(("content", self.buf))
            self.buf = ""
        elif self.state == 1:
            # Unclosed tool calls section — try to parse what we have
            results.extend(self._process_buffer())
            if self._emitted_calls > 0:
                results.append(("tool_call_end", None))
        return results
