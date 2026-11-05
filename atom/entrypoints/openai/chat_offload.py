# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Chat completions prepared in the tokenize workers (ATOM_API_OFFLOAD_PREPROCESS).

The API server is one Python process and its event loop is one thread. An
agentic turn resends the whole conversation, ~300 KB of JSON and ~100k tokens,
and before the engine sees it the loop parsed the body, validated it with
pydantic and rendered the chat template -- milliseconds a request, all of it
on that one thread, while the threads beside it (preprocessing executor,
engine output) wait on the same GIL. That capped the server at ~180
requests/s whatever the GPUs could do.

With the offload on, the loop reads the body bytes and hands them to a
tokenize worker (`tokenize_pool.TokenizePool`), which does what the handler
did up to the engine request: parse, validate, normalize the tools, merge the
template kwargs, render, encode, check whether the prompt opens the reasoning
channel, and hash the prompt's prefix blocks (`prefix_hash`, the chain
`ship_prompt_hash_chain` would otherwise hash in the preprocessing executor).
It returns a `PreparedChat`: the ids, the hashes, and the request without its
messages. The handler then continues from the sampling parameters exactly as
before.

`prepare_chat` is the whole of that, and the workers and the in-process check
both run it. Anything it does not take -- invalid, multimodal, pre-tokenized
or n > 1 requests -- comes back as None and goes to the in-process path,
which also produces every error response.

The first few results are compared with `prepare_chat` run in the server's own
process, on the server's tokenizer and template state; a difference turns the
offload off for the life of the process.
"""

import array
import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, NamedTuple

from atom.entrypoints.chat_utils import has_multimodal_content
from atom.model_engine.prefix_block_hashes import PrefixBlockHashes
from atom.model_engine.prefix_hash import chained_block_hashes

from .chat_encoders import (
    PROBE_MESSAGES,
    apply_chat_template,
    load_custom_message_encoder,
)
from .protocol import ChatCompletionRequest
from .reasoning import prompt_starts_in_reasoning
from .serving_chat import (
    merge_chat_template_kwargs,
    normalize_chat_tools,
    validate_chat_request,
)

logger = logging.getLogger("atom")

# Results checked against the in-process path before the offload is trusted.
_VERIFY_FIRST = 8

# `Sequence.cache_seed` of a text prompt: the seed its prefix-hash chain starts
# from. `ship_prompt_hash_chain` rehashes a chain minted from another seed.
TEXT_PROMPT_CACHE_SEED = -1


class ChatPrepareSettings(NamedTuple):
    """The server's startup state a worker renders with, beyond the tokenizer."""

    # `--chat-template` as the server applied it (None: the tokenizer's own).
    chat_template: str | None
    default_chat_template_kwargs: dict[str, Any]
    reasoning_toggle: tuple[str, Any, Any] | None
    # `InputOutputProcessor.prefix_hash_block_size` when the frontend ships
    # prompt hash chains (ATOM_FRONTEND_PREFIX_HASH), else 0: hash nothing.
    prefix_hash_block_size: int


class PreparedChat(NamedTuple):
    """A chat completion request, ready for the engine."""

    # Validated, tools normalized; `messages` and `prompt` dropped (None).
    request: ChatCompletionRequest
    template_kwargs: dict[str, Any]
    prompt_ids: array.array  # "i"
    prompt_opens_reasoning: bool
    # The prompt's full-block hash chain, for `ship_prompt_hash_chain`.
    block_hashes: PrefixBlockHashes | None


def prepare_chat(
    body: bytes,
    tokenizer: Any,
    custom_encoder: Any,
    settings: ChatPrepareSettings,
) -> PreparedChat | None:
    """What `/v1/chat/completions` does with ``body`` before the engine request.

    The same steps in the same order as the handler, on the same functions.
    None for a request the handler must see itself: one it would reject (the
    error response is the handler's), multimodal, pre-tokenized, or n > 1.
    """
    try:
        request = ChatCompletionRequest.model_validate(json.loads(body))
        if request.n is not None and request.n != 1:
            return None
        request.tools = normalize_chat_tools(request.tools)
        validate_chat_request(request)
        messages = request.get_messages()
        template_kwargs = merge_chat_template_kwargs(
            request,
            settings.default_chat_template_kwargs,
            settings.reasoning_toggle,
        )
        if has_multimodal_content([{"content": m.content} for m in messages]):
            return None
        if request.get_prompt_token_ids() is not None:
            return None
        prompt = apply_chat_template(
            tokenizer,
            custom_encoder,
            messages,
            tools=request.tools,
            **template_kwargs,
        )
        prompt_ids = array.array("i", tokenizer.encode(prompt))
    except Exception:  # noqa: BLE001  # the in-process path answers it
        return None
    block_hashes = None
    block_size = settings.prefix_hash_block_size
    if block_size > 0:
        num_tokens = len(prompt_ids)
        packed = array.array(
            "Q",
            chained_block_hashes(
                prompt_ids,
                block_size,
                0,
                num_tokens // block_size,
                TEXT_PROMPT_CACHE_SEED,
            ),
        )
        block_hashes = PrefixBlockHashes(
            block_size, TEXT_PROMPT_CACHE_SEED, num_tokens, packed.tobytes()
        )
    return PreparedChat(
        request.model_copy(update={"messages": None, "prompt": None}),
        template_kwargs,
        prompt_ids,
        prompt_starts_in_reasoning(prompt),
        block_hashes,
    )


# ---- worker side ----

_worker_state: tuple[Any, Any, ChatPrepareSettings] | None = None


def init_chat_worker(tokenizer: Any, model: str, settings: ChatPrepareSettings):
    """Build a worker's render state from the server's startup configuration."""
    global _worker_state
    if settings.chat_template is not None:
        tokenizer.chat_template = settings.chat_template
    custom_encoder = load_custom_message_encoder(model)
    _worker_state = (tokenizer, custom_encoder, settings)
    # Compile the template now rather than on the worker's first request.
    try:
        apply_chat_template(
            tokenizer,
            custom_encoder,
            [dict(m) for m in PROBE_MESSAGES],
            **settings.default_chat_template_kwargs,
        )
    except Exception:
        logger.debug("chat offload: probe render failed", exc_info=True)


def prepare_chat_in_worker(body: bytes) -> PreparedChat | None:
    tokenizer, custom_encoder, settings = _worker_state
    return prepare_chat(body, tokenizer, custom_encoder, settings)


# ---- server side ----


def _outcome(fn: Callable[[Any], Any], arg: Any) -> Any:
    try:
        return fn(arg)
    except Exception as exc:  # noqa: BLE001  # the same rejection is a match
        return repr(exc)


def difference(
    offloaded: PreparedChat | None,
    reference: PreparedChat | None,
    sampling_params_of: Callable[[ChatCompletionRequest], Any],
) -> str | None:
    """What differs between two preparations of one body; None if nothing."""
    if offloaded is None or reference is None:
        if offloaded is reference:
            return None
        return "the worker's fallback" if offloaded is None else "the fallback"
    for name in PreparedChat._fields:
        if getattr(offloaded, name) != getattr(reference, name):
            return name
    if _outcome(sampling_params_of, offloaded.request) != _outcome(
        sampling_params_of, reference.request
    ):
        return "sampling params"
    return None


class ChatOffload:
    """`await prepare(body)` -> `PreparedChat` from a tokenize worker, or None
    to serve the request on the in-process path."""

    def __init__(
        self,
        pool: Any,
        reference: Callable[[bytes], PreparedChat | None],
        sampling_params_of: Callable[[ChatCompletionRequest], Any],
    ):
        # `pool`: a `TokenizePool` built with `chat_settings`. `reference`:
        # `prepare_chat` on the server's own tokenizer and template state.
        self._pool = pool
        self._reference = reference
        self._sampling_params_of = sampling_params_of
        self._verify_left = _VERIFY_FIRST
        self._on = True

    @property
    def enabled(self) -> bool:
        # The workers' tokenizer failing its own check rules this out too.
        return self._on and self._pool.enabled

    async def prepare(self, body: bytes) -> PreparedChat | None:
        try:
            prepared = await self._pool.run(prepare_chat_in_worker, body)
        except Exception:
            logger.warning(
                "chat offload: a tokenize worker failed; preparing chat "
                "completions in process from now on",
                exc_info=True,
            )
            self._on = False
            return None
        if self._verify_left > 0:
            self._verify_left -= 1
            loop = asyncio.get_running_loop()
            reference = await loop.run_in_executor(None, self._reference, body)
            differs = difference(prepared, reference, self._sampling_params_of)
            if differs is not None:
                logger.warning(
                    "chat offload: %s differs from the in-process path; "
                    "preparing chat completions in process from now on",
                    differs,
                )
                self._on = False
                return reference
        return prepared
