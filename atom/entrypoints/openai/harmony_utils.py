# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Harmony encoding utilities for GPT-OSS tool calling.

GPT-OSS models use OpenAI's proprietary Harmony format for structured output
(tool calls, reasoning, content). This module wraps the ``openai_harmony``
library to build Harmony messages and render them to token IDs for prompt
construction, bypassing ``tokenizer.apply_chat_template`` entirely.

Adapted from vLLM's ``vllm/entrypoints/openai/parser/harmony_utils.py``.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    Role,
    StreamableParser,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)

logger = logging.getLogger("atom")

_harmony_encoding = None


# ── Encoding ──────────────────────────────────────────────────────────


def get_encoding():
    """Return the singleton Harmony encoding for GPT-OSS."""
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def get_streamable_parser_for_assistant() -> StreamableParser:
    """Create a StreamableParser for decoding assistant output tokens."""
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)


# ── Recipient helpers ─────────────────────────────────────────────────


def is_function_recipient(recipient: str) -> bool:
    """Check whether *recipient* refers to a function tool call."""
    if not recipient:
        return False
    if recipient.startswith("<|"):
        return False
    if recipient.startswith("functions."):
        return len(recipient) > len("functions.")
    if recipient == "assistant":
        return False
    return True


def extract_function_from_recipient(recipient: str) -> str:
    """Strip the ``functions.`` prefix from a recipient name."""
    return recipient.removeprefix("functions.")


# ── Message building ─────────────────────────────────────────────────


def _create_tool_description(tool: Dict[str, Any]) -> ToolDescription:
    """Create a Harmony ToolDescription from an OpenAI-format tool dict."""
    fn = tool.get("function", tool)
    return ToolDescription.new(
        name=fn.get("name", ""),
        description=fn.get("description", ""),
        parameters=fn.get("parameters"),
    )


def get_system_message() -> Message:
    """Build the Harmony system message."""
    sys_content = SystemContent.new()
    return Message.from_role_and_content(Role.SYSTEM, sys_content)


def get_developer_message(
    instructions: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Message:
    """Build the Harmony developer message with optional tools."""
    dev_content = DeveloperContent.new()
    if instructions:
        dev_content = dev_content.with_instructions(instructions)
    if tools:
        fn_tools = [t for t in tools if t.get("type") == "function"]
        if fn_tools:
            descriptions = [_create_tool_description(t) for t in fn_tools]
            dev_content = dev_content.with_function_tools(descriptions)
    return Message.from_role_and_content(Role.DEVELOPER, dev_content)


def build_harmony_preamble(
    *,
    instructions: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Message]:
    """Build the standard Harmony system + developer prefix."""
    messages = [get_system_message()]
    if instructions or tools:
        messages.append(get_developer_message(instructions=instructions, tools=tools))
    return messages


def _flatten_content(content: Any) -> Optional[str]:
    """Extract text from string or multimodal content list."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(text)
        return "".join(parts) if parts else None
    return None


def extract_instructions_from_messages(
    messages: List[Dict[str, Any]],
) -> tuple:
    """Peel a leading system/developer message and return (instructions, rest)."""
    if not messages:
        return None, messages
    first = messages[0]
    if first.get("role") not in ("system", "developer"):
        return None, messages
    instructions = _flatten_content(first.get("content"))
    return instructions, messages[1:]


def _parse_single_message(
    chat_msg: Dict[str, Any],
    tool_id_names: Dict[str, str],
) -> List[Message]:
    """Convert one OpenAI-format message dict to Harmony Message(s)."""
    role = chat_msg.get("role", "")
    msgs: List[Message] = []
    tool_calls = chat_msg.get("tool_calls", [])

    # Assistant message with tool calls
    if role == "assistant" and tool_calls:
        content = _flatten_content(chat_msg.get("content"))
        if content:
            commentary = Message.from_role_and_content(Role.ASSISTANT, content)
            commentary = commentary.with_channel("commentary")
            msgs.append(commentary)

        reasoning = chat_msg.get("reasoning") or chat_msg.get("reasoning_content")
        if reasoning:
            analysis = Message.from_role_and_content(Role.ASSISTANT, reasoning)
            analysis = analysis.with_channel("analysis")
            msgs.append(analysis)

        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "") or ""
            # arguments may be a dict (deserialized by to_template_dict);
            # Harmony requires a string
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
            msg = msg.with_channel("commentary")
            msg = msg.with_recipient(f"functions.{name}")
            msg = msg.with_content_type("json")
            msgs.append(msg)
        return msgs

    # Tool result message
    if role == "tool":
        tool_call_id = chat_msg.get("tool_call_id", "")
        name = tool_id_names.get(tool_call_id, "")
        content = _flatten_content(chat_msg.get("content")) or ""
        msg = (
            Message.from_author_and_content(
                Author.new(Role.TOOL, f"functions.{name}"), content
            )
            .with_channel("commentary")
            .with_recipient("assistant")
        )
        return [msg]

    # Non-tool reasoning content
    reasoning = chat_msg.get("reasoning") or chat_msg.get("reasoning_content")
    if role == "assistant" and reasoning:
        analysis = Message.from_role_and_content(Role.ASSISTANT, reasoning)
        analysis = analysis.with_channel("analysis")
        msgs.append(analysis)

    # Default: user/assistant/system messages
    text = _flatten_content(chat_msg.get("content")) or ""

    _role_map = {"user": Role.USER, "assistant": Role.ASSISTANT,
                 "system": Role.SYSTEM, "developer": Role.DEVELOPER}

    if role == "assistant" and text:
        msg = Message.from_role_and_content(Role.ASSISTANT, text)
        msg = msg.with_channel("final")
        msgs.append(msg)
    elif role in ("system", "developer"):
        if text:
            dev_msg = get_developer_message(instructions=text)
            msgs.append(dev_msg)
    elif role != "assistant":
        hr = _role_map.get(role, Role.USER)
        msg = Message.from_role_and_content(hr, text)
        msgs.append(msg)

    return msgs


def parse_chat_inputs_to_harmony_messages(
    chat_msgs: List[Dict[str, Any]],
) -> List[Message]:
    """Convert OpenAI-format message dicts to Harmony messages."""
    tool_id_names: Dict[str, str] = {}
    for msg in chat_msgs:
        for tc in msg.get("tool_calls", []):
            tool_id_names[tc.get("id", "")] = tc.get("function", {}).get("name", "")

    result: List[Message] = []
    for msg in chat_msgs:
        result.extend(_parse_single_message(msg, tool_id_names))
    return result


def auto_drop_analysis_messages(msgs: List[Message]) -> List[Message]:
    """Drop analysis (reasoning) messages before the last assistant final message."""
    last_final_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].author.role == "assistant" and msgs[i].channel == "final":
            last_final_idx = i
            break
    if last_final_idx == -1:
        return msgs
    return [
        m for i, m in enumerate(msgs) if not (i < last_final_idx and m.channel == "analysis")
    ]


def render_for_completion(messages: List[Message]) -> List[int]:
    """Render Harmony messages to prompt token IDs."""
    messages = auto_drop_analysis_messages(messages)
    # Validate each message can serialize before building Conversation
    for i, msg in enumerate(messages):
        try:
            msg.to_dict()
        except NotImplementedError:
            logger.error(
                "Harmony message %d failed to_dict: role=%s channel=%s "
                "recipient=%s content_types=%s",
                i,
                getattr(msg.author, "role", "?"),
                getattr(msg, "channel", None),
                getattr(msg, "recipient", None),
                [type(c).__name__ for c in (msg.content or [])],
            )
            raise
    conversation = Conversation.from_messages(messages)
    return get_encoding().render_conversation_for_completion(
        conversation,
        Role.ASSISTANT,
        config=RenderConversationConfig(auto_drop_analysis=False),
    )
