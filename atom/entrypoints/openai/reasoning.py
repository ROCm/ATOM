# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reasoning/thinking content separation for thinking models (e.g., Kimi-K2, DeepSeek-R1).

This module provides utilities to separate <think>...</think> reasoning blocks
from the final answer, following the SGLang/vLLM reasoning_content pattern.
Also strips raw tool call tokens that the model may output.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple


def separate_reasoning(text: str) -> Tuple[Optional[str], str]:
    """Separate reasoning content from the final answer.

    Args:
        text: Raw model output that may contain <think>...</think> blocks.

    Returns:
        Tuple of (reasoning_content, content). reasoning_content is None if
        no thinking block was found.
    """
    # Check for closed thinking block
    match = re.match(r"<think>(.*?)</think>\s*(.*)", text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        content = match.group(2).strip()
        return (reasoning if reasoning else None, content)

    # Check for unclosed thinking block (truncated response)
    match = re.match(r"<think>(.*)", text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        return (reasoning if reasoning else None, "")

    # No thinking block — return content as-is (tool calls parsed separately)
    return (None, text)


@dataclass
class ReasoningFilter:
    """Stateful streaming filter that separates reasoning from content.

    Processes tokens one chunk at a time and yields (field, text) tuples
    where field is either "reasoning_content" or "content".

    States:
        0 = before <think> (buffering to detect)
        1 = inside <think> (emitting as reasoning_content)
        2 = after </think> (emitting as content)
    """

    state: int = 0
    buf: str = ""

    def process(self, text: str) -> list:
        """Process a chunk of text and return list of (field, text) tuples.

        Args:
            text: New text chunk from the model.

        Returns:
            List of (field_name, text) tuples where field_name is
            "reasoning_content" or "content".
        """
        results = []

        if self.state == 0:
            self.buf += text
            if "<think>" in self.buf:
                before = self.buf.split("<think>")[0]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf.split("<think>", 1)[1]
                # Check if </think> is already in buffer
                if "</think>" in self.buf:
                    reasoning = self.buf.split("</think>", 1)[0]
                    after = self.buf.split("</think>", 1)[1].lstrip("\n")
                    if reasoning:
                        results.append(("reasoning_content", reasoning))
                    self.state = 2
                    self.buf = ""
                    if after:
                        results.extend(self._process_content(after))
                elif self.buf:
                    results.append(("reasoning_content", self.buf))
                    self.buf = ""
            elif len(self.buf) > 7 and "<" not in self.buf:
                results.append(("content", self.buf))
                self.buf = ""

        elif self.state == 1:
            self.buf += text
            if "</think>" in self.buf:
                reasoning = self.buf.split("</think>", 1)[0]
                after = self.buf.split("</think>", 1)[1].lstrip("\n")
                if reasoning:
                    results.append(("reasoning_content", reasoning))
                self.state = 2
                self.buf = ""
                if after:
                    results.extend(self._process_content(after))
            else:
                results.append(("reasoning_content", self.buf))
                self.buf = ""

        else:  # state == 2
            results.extend(self._process_content(text))

        return results

    def _process_content(self, text: str) -> list:
        """Process content after thinking. Tool calls are handled by ToolCallStreamParser."""
        if text:
            return [("content", text)]
        return []

    def flush(self) -> list:
        """Flush any remaining buffered content."""
        results = []
        if self.buf:
            if self.state == 0:
                results.append(("content", self.buf))
            elif self.state == 1:
                results.append(("reasoning_content", self.buf))
            self.buf = ""
        return results
