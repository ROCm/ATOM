# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Model-scoped adapters for dynamically loaded chat encoders."""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import deepseek_v41_encoder

logger = logging.getLogger("atom")

MessageEncoder = Callable[..., str]
MessagePreparer = Callable[[list[dict], list[dict] | None], list[dict]]
KwargsPreparer = Callable[[dict[str, Any]], dict[str, Any]]


def _identity_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return kwargs


def _copy_messages(
    messages: list[dict], _tools: list[dict] | None = None
) -> list[dict]:
    """Return shallow message copies without model-specific rewriting."""
    return [dict(message) for message in messages]


def _prepare_deepseek_v4_messages(
    messages: list[dict], tools: list[dict] | None
) -> list[dict]:
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
    # The file the encoder was loaded from. `encode` is a closure defined in
    # `chat_encoders`, so `inspect.getmodule(encode)` names *that* module --
    # reading the model's own source needs the path carried here.
    source_path: str = ""
    # What this encoder's signature will accept, or ``None`` when it takes
    # `**kwargs` and so accepts anything. See `__call__`.
    accepts: frozenset[str] | None = field(default=None, compare=False)
    # Values this model's loader wants set when the caller did not. Applied
    # after the filter and subject to it, so a default the encoder cannot take
    # is dropped like any other kwarg rather than raising.
    defaults: dict[str, Any] = field(default_factory=dict, compare=False)
    prepare_kwargs: KwargsPreparer = _identity_kwargs
    preserve_content: bool = False

    def __call__(self, messages: list[dict], **kwargs: Any) -> str:
        """Render, passing on only the kwargs this encoder can take.

        A Jinja template silently ignores a kwarg it does not read; a Python
        encoder raises `TypeError`. The request handler forwards template
        controls it cannot know the model reads -- `response_format`,
        `tool_choice`, `thinking_effort`, and whatever a client puts in
        `chat_template_kwargs` -- so a request carrying any of them was a 500
        on every model that ships an encoder instead of a template, which on
        this box is DeepSeek-V4 and Kimi-K3. `tool_choice: "none"` reached it
        that way the moment the Anthropic endpoint began honouring the field.

        Dropped rather than raised, so the two rendering paths behave the same
        way: unread means unread. Logged at debug, because a kwarg a model
        cannot take is worth seeing when a template control appears not to
        work.
        """
        for name, value in self.defaults.items():
            kwargs.setdefault(name, value)
        kwargs = self.prepare_kwargs(kwargs)
        if self.accepts is not None:
            unread = [name for name in kwargs if name not in self.accepts]
            for name in unread:
                logger.debug(
                    "chat encoder %s does not take %r; ignoring it", self.name, name
                )
                kwargs.pop(name)
        return self.encode(messages, **kwargs)


@dataclass(frozen=True)
class _EncoderPreparation:
    messages: MessagePreparer = _copy_messages
    kwargs: KwargsPreparer = _identity_kwargs
    supports_tools: bool = False
    preserve_content: bool = False


_PREPARERS = {
    "encoding_dsv4": _EncoderPreparation(
        messages=_prepare_deepseek_v4_messages, supports_tools=True
    ),
    "encoding_dsv41": _EncoderPreparation(
        messages=deepseek_v41_encoder.prepare_messages,
        kwargs=deepseek_v41_encoder.prepare_kwargs,
        supports_tools=True,
        preserve_content=True,
    ),
}

# A generic filename does not identify a protocol. Only the checkpoint's
# explicit model_type selects the V4.1 hooks.
_MODEL_ENCODERS = {"deepseek_v41": "encoding_dsv41"}


def _accepted_kwargs(encoder: MessageEncoder) -> frozenset[str] | None:
    """The keyword names ``encoder`` takes, or ``None`` for "anything".

    ``None`` also when the signature cannot be read at all -- a C callable, a
    functools wrapper without `__wrapped__`. Passing everything through is
    what happened before this existed, so an unreadable signature is no worse
    than it was, and refusing to load the encoder over it would be.
    """
    try:
        params = inspect.signature(encoder).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return frozenset(
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


def build_message_encoder_adapter(
    module_name: str,
    encoder: MessageEncoder,
    source_path: str = "",
    accepts_from: MessageEncoder | None = None,
    defaults: dict[str, Any] | None = None,
    model_type: str | None = None,
) -> MessageEncoderAdapter:
    """Build an adapter registered for ``module_name`` or an identity adapter.

    ``accepts_from`` is the callable whose signature says what may be passed,
    when that is not ``encoder`` itself. The loader wraps the model's function
    in a closure taking ``**kwargs``, and reading the *wrapper* would answer
    "anything" for every encoder there is.
    """
    if module_name == "encoding":
        module_name = _MODEL_ENCODERS.get(model_type, module_name)
    preparation = _PREPARERS.get(module_name, _EncoderPreparation())
    return MessageEncoderAdapter(
        name=module_name,
        encode=encoder,
        prepare_messages=preparation.messages,
        prepare_kwargs=preparation.kwargs,
        supports_tools=preparation.supports_tools,
        preserve_content=preparation.preserve_content,
        source_path=source_path,
        accepts=_accepted_kwargs(accepts_from or encoder),
        defaults=dict(defaults or {}),
    )
