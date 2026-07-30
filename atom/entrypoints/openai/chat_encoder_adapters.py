# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Model-scoped adapters for dynamically loaded chat encoders."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

MessageEncoder = Callable[..., str]
MessagePreparer = Callable[[List[dict], Optional[List[dict]]], List[dict]]


def _copy_messages(
    messages: List[dict], _tools: Optional[List[dict]] = None
) -> List[dict]:
    """Return shallow message copies without model-specific rewriting."""
    return [dict(message) for message in messages]


def _prepare_deepseek_v4_messages(
    messages: List[dict], tools: Optional[List[dict]]
) -> List[dict]:
    """Prepare the internal message shape expected by DSV4 ``encode_messages``.

    The DeepSeek-V4 reference encoder reads tool schemas from a system message's
    ``tools`` field. Match vLLM's model-specific tokenizer wrapper by prepending
    a synthetic tool-carrying system message, without reordering or merging the
    caller's existing messages.
    """
    prepared = _copy_messages(messages)
    if tools:
        prepared.insert(0, {"role": "system", "tools": tools})
    return prepared


@dataclass(frozen=True)
class MessageEncoderAdapter:
    """A raw model encoder plus its model-specific message preparation."""

    name: str
    encode: MessageEncoder
    prepare_messages: MessagePreparer
    supports_tools: bool = False

    def __call__(self, messages: List[dict], **kwargs: Any) -> str:
        """Preserve the callable behavior of the former encoder return value."""
        return self.encode(messages, **kwargs)


_PREPARERS: dict[str, tuple[MessagePreparer, bool]] = {
    "encoding_dsv4": (_prepare_deepseek_v4_messages, True),
}


def build_message_encoder_adapter(
    module_name: str, encoder: MessageEncoder
) -> MessageEncoderAdapter:
    """Build an adapter registered for ``module_name`` or an identity adapter."""
    prepare_messages, supports_tools = _PREPARERS.get(
        module_name, (_copy_messages, False)
    )
    return MessageEncoderAdapter(
        name=module_name,
        encode=encoder,
        prepare_messages=prepare_messages,
        supports_tools=supports_tools,
    )
