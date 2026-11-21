# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Adapted from TorchSpec's data processing pipeline. The upstream attribution
# and license must be retained; see atomrl/THIRD_PARTY_NOTICES.md.

"""Kimi-K2.5 chat template parser and loss mask utilities.

Ported from TorchSpec's data processing pipeline to ensure identical
tokenization and assistant-only loss masking for speculative distillation.
"""

import json
import re

import torch
from transformers import PreTrainedTokenizer

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}

_HAS_THINKING_RE = re.compile(r"<think>(?!\s*</think>)")


def has_thinking_content(conversation: list[dict]) -> bool:
    """Detect whether any assistant message contains real thinking content.

    Checks for non-empty <think> blocks and for separate thinking/reasoning
    fields. Must be called BEFORE formatting, since format() injects empty
    <think></think> tags.
    """
    for msg in conversation:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and _HAS_THINKING_RE.search(content):
            return True
        for field in ("thinking", "thinking_content", "reasoning_content", "reasoning"):
            if msg.get(field):
                return True
    return False


def normalize_conversation(conversation: list[dict]) -> list[dict]:
    """Normalize ShareGPT format (from/value) to standard (role/content).

    Preserves reasoning_content from thinking/reasoning fields.
    """
    if not conversation:
        return conversation
    first = conversation[0]
    if "role" in first and "content" in first:
        return conversation
    if "from" in first and "value" in first:
        normalized = []
        for msg in conversation:
            role = ROLE_MAPPING.get(msg["from"], msg["from"])
            entry = {"role": role, "content": msg["value"]}
            for field in (
                "thinking",
                "thinking_content",
                "reasoning_content",
                "reasoning",
            ):
                if msg.get(field):
                    entry["reasoning_content"] = msg[field]
                    break
            normalized.append(entry)
        return normalized
    return conversation


class KimiK25Parser:
    """Parser for Kimi-K2.5 model with manual string formatting.

    Handles:
    - Converting <|image|> placeholders to Kimi media token structure
    - Preserving <think>...</think> blocks in assistant responses
    - Generating loss mask for assistant content only
    """

    MEDIA_TOKEN = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    USER_HEADER = "<|im_user|>user<|im_middle|>"
    ASSISTANT_HEADER = "<|im_assistant|>assistant<|im_middle|>"
    SYSTEM_HEADER = "<|im_system|>system<|im_middle|>"
    TOOL_HEADER = "<|im_system|>tool<|im_middle|>"
    END_TOKEN = "<|im_end|>"
    IMAGE_PLACEHOLDER = "<|image|>"
    TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
    TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
    TOOL_CALL_BEGIN = "<|tool_call_begin|>"
    TOOL_CALL_END = "<|tool_call_end|>"
    TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

    THINK_PATTERN = re.compile(r"<think>[\s\S]*?</think>")

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def _format_content(self, content: str, role: str) -> str:
        if role == "user":
            return content.replace(self.IMAGE_PLACEHOLDER, self.MEDIA_TOKEN + "\n")
        return content

    def _flatten_multimodal_list(self, content: list) -> str:
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") in ("image", "image_url"):
                text_parts.append(self.MEDIA_TOKEN + "\n")
        return "".join(text_parts)

    def _try_parse_multimodal_string(self, content: str):
        """Parse stringified JSON multimodal content (e.g. llava_instruct).

        Returns flattened text with MEDIA_TOKEN placeholders, or None if
        the string is not parseable multimodal content.
        """
        if not content.startswith("["):
            return None
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, list) or not parsed:
            return None
        if not isinstance(parsed[0], dict) or "type" not in parsed[0]:
            return None
        return self._flatten_multimodal_list(parsed)

    def _strip_thinking(self, content: str) -> str:
        return self.THINK_PATTERN.sub("", content)

    def _format_tool_calls(self, tool_calls: list) -> str:
        tc_parts = []
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            tc_args = tc.get("function", {}).get("arguments", "")
            tc_parts.append(
                f"{self.TOOL_CALL_BEGIN}{tc_id}"
                f"{self.TOOL_CALL_ARGUMENT_BEGIN}{tc_args}"
                f"{self.TOOL_CALL_END}"
            )
        return (
            self.TOOL_CALLS_SECTION_BEGIN
            + "".join(tc_parts)
            + self.TOOL_CALLS_SECTION_END
        )

    def format(
        self, conversation: list[dict], add_generation_prompt: bool = False
    ) -> str:
        """Build conversation string with Kimi-K2.5 special tokens.

        Strips <think>...</think> from all assistant turns except the last one.
        Injects <think></think> if assistant content doesn't start with <think>.
        """
        parts = []

        last_assistant_idx = max(
            (i for i, msg in enumerate(conversation) if msg["role"] == "assistant"),
            default=-1,
        )

        for idx, msg in enumerate(conversation):
            role = msg["role"]
            content = msg.get("content", "")

            if not isinstance(content, str):
                if isinstance(content, list):
                    content = self._flatten_multimodal_list(content)
                else:
                    content = str(content)
            else:
                parsed = self._try_parse_multimodal_string(content)
                if parsed is not None:
                    content = parsed
                else:
                    content = self._format_content(content, role)

            if role == "assistant" and idx != last_assistant_idx:
                content = self._strip_thinking(content)

            if role == "system":
                parts.append(f"{self.SYSTEM_HEADER}{content}{self.END_TOKEN}")
            elif role == "user":
                parts.append(f"{self.USER_HEADER}{content}{self.END_TOKEN}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    content += self._format_tool_calls(tool_calls)
                if not content.startswith("<think>"):
                    content = "<think></think>" + content
                parts.append(f"{self.ASSISTANT_HEADER}{content}{self.END_TOKEN}")
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if tool_call_id:
                    content = f"## Return of {tool_call_id}\n{content}"
                parts.append(f"{self.TOOL_HEADER}{content}{self.END_TOKEN}")

        if add_generation_prompt:
            parts.append(self.ASSISTANT_HEADER)

        return "".join(parts)

    def parse(
        self,
        conversation: list[dict],
        max_length: int,
        last_turn_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Format, tokenize, and compute assistant-only loss mask.

        Returns (input_ids, loss_mask) — both 1-D tensors.
        """
        text = self.format(conversation)
        return self._tokenize_with_loss_mask(text, max_length, last_turn_only)

    def _tokenize_with_loss_mask(
        self,
        text: str,
        max_length: int,
        last_turn_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text and compute loss mask via encode-prefix character mapping.

        Only content between ASSISTANT_HEADER and END_TOKEN gets loss_mask=1.
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        assistant_pattern = (
            re.escape(self.ASSISTANT_HEADER) + r"([\s\S]*?)" + re.escape(self.END_TOKEN)
        )
        matches = list(re.finditer(assistant_pattern, text))
        if last_turn_only and matches:
            matches = matches[-1:]

        for match in matches:
            content_start_char = match.start(1)
            content_end_char = match.end(1)

            prefix_ids = self.tokenizer.encode(
                text[:content_start_char], add_special_tokens=False
            )
            full_ids = self.tokenizer.encode(
                text[:content_end_char], add_special_tokens=False
            )

            start_token_idx = len(prefix_ids)
            end_token_idx = len(full_ids)

            actual_start = min(start_token_idx, len(input_ids))
            actual_end = min(end_token_idx, len(input_ids))

            if actual_start < actual_end:
                loss_mask[actual_start:actual_end] = 1

        return input_ids, loss_mask


# ---------------------------------------------------------------------------
# Packed loss mask utilities (run-length encoding)
# ---------------------------------------------------------------------------


def pack_loss_mask(loss_mask: torch.Tensor) -> list[int]:
    """Run-length encode a loss_mask tensor.

    Returns alternating [prompt_len, response_len, prompt_len, ...],
    always starting with prompt (0-valued segment).

    Example:
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0] -> [2, 3, 2, 2, 1]
    """
    if loss_mask.dim() > 1:
        loss_mask = loss_mask.squeeze()
    if len(loss_mask) == 0:
        return []

    lengths = []
    mask_list = loss_mask.tolist()
    current_val = 0
    current_len = 0

    for val in mask_list:
        if val == current_val:
            current_len += 1
        else:
            lengths.append(current_len)
            current_val = val
            current_len = 1

    lengths.append(current_len)
    return lengths


def unpack_loss_mask(packed: list[int] | str) -> torch.Tensor:
    """Reconstruct loss_mask tensor from packed form.

    Example:
        [2, 3, 2, 2, 1] -> tensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    """
    if isinstance(packed, str):
        packed = deserialize_packed_loss_mask(packed)
    if not packed:
        return torch.tensor([], dtype=torch.long)

    total = sum(packed)
    loss_mask = torch.zeros(total, dtype=torch.long)
    pos = 0

    for i, length in enumerate(packed):
        if i % 2 == 1:
            loss_mask[pos : pos + length] = 1
        pos += length

    return loss_mask


def serialize_packed_loss_mask(packed: list[int]) -> str:
    return ",".join(str(x) for x in packed)


def deserialize_packed_loss_mask(s: str) -> list[int]:
    if not s:
        return []
    return [int(x) for x in s.split(",")]
