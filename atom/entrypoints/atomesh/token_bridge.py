# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM standalone's Rust frontend, down to LLMEngine.

The Rust server (``atom/mesh/src/routers/atom_bridge``) serves HTTP, renders
the chat template, tokenizes, detokenizes and writes the SSE. What it needs
from Python is this, and only this:

* ``submit`` takes a prompt's token ids through
  ``io_processor.preprocess_fanout`` (stop strings, prefix hash chain,
  structured admission -- whatever the engine does with a prompt) and
  ``CoreManager.add_request`` (DP routing, P/D, back-pressure).
* ``abort`` stops what a request left running.
* Each engine step's tokens go back to Rust in one ``StepSink.push`` from
  CoreManager's output thread, the way the frontend workers' hub flushes a
  step per worker (``frontend_workers.py``).
* ``render_chat`` renders the chat prompt of a model whose template Rust
  cannot render: a Python message encoder (DeepSeek-V4), or a tokenizer
  whose template is code rather than Jinja (Kimi-K3).

A request is known by the key Rust gives it, not by sequence id: Rust has
registered the key before ``submit`` is called, so no output can arrive for a
request Rust does not know yet.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from atom import SamplingParams
from atom.entrypoints.openai.api_server import (
    _coerce_n,
    _validate_context_length,
    _validate_pool_capacity,
)
from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    chat_template_source,
    load_custom_message_encoder,
    render_probe_prompt,
    resolve_reasoning_toggle,
)
from atom.entrypoints.openai.model_capabilities import deployment_metadata
from atom.entrypoints.openai.protocol import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from atom.entrypoints.openai.reasoning import (
    template_opens_reasoning_implicitly,
    thinking_switched_off,
)
from atom.entrypoints.openai.reasoning_dialects import resolve_dialect
from atom.entrypoints.openai.tool_parser.registry import (
    required_call_channel,
    resolve_tool_call_parser,
)

logger = logging.getLogger("atom")


class ChatRenderer:
    """A chat prompt as ATOM's own server renders it, for a model whose
    template the router cannot render (``needed``): one with a Python message
    encoder, or a tokenizer whose ``chat_template`` is code, not Jinja."""

    def __init__(self, tokenizer: Any, model: str) -> None:
        self.tokenizer = tokenizer
        self.encoder = load_custom_message_encoder(model)
        template = getattr(tokenizer, "chat_template", None)
        self.needed = self.encoder is not None or not isinstance(template, (str, dict))
        if not self.needed:
            return
        source = chat_template_source(tokenizer, self.encoder)
        self.dialect, _ = resolve_dialect(
            source, render_probe_prompt(tokenizer, self.encoder, tools=False) or ""
        )
        self.model_starts_in_reasoning = template_opens_reasoning_implicitly(source)
        self.reasoning_toggle = resolve_reasoning_toggle(tokenizer, self.encoder)

    def render(self, request: dict[str, Any]) -> tuple[str, str | None]:
        """The prompt, and the channel opener it ends on (or ``None``).

        The opener is what the reply starts inside, with none of its own: the
        dialect's reasoning opener when the prompt opens reasoning, or its
        answer opener (Kimi-K3's response channel with thinking off).
        """
        messages = request.get("messages")
        if not messages:
            raise ValueError("messages is required")
        kwargs = dict(request.get("chat_template_kwargs") or {})
        prompt = apply_chat_template(
            self.tokenizer,
            self.encoder,
            [_template_message(message) for message in messages],
            tools=request.get("tools"),
            **kwargs,
        )
        return prompt, self._opener(prompt, kwargs)

    def _opener(self, prompt: str, kwargs: dict[str, Any]) -> str | None:
        tail = prompt.rstrip()
        for marker in (
            self.dialect.prompt_open_marker,
            self.dialect.answer_open_marker,
        ):
            if marker and tail.endswith(marker):
                return marker
        # As the OpenAI server's `_reasoning_channel`: a template that opens
        # reasoning without writing the opener, unless the render switched
        # reasoning off.
        if self.model_starts_in_reasoning and not thinking_switched_off(
            kwargs, self.reasoning_toggle
        ):
            return self.dialect.prompt_open_marker
        return None


def _template_message(message: dict[str, Any]) -> dict[str, Any]:
    """A chat message as the template takes it: text parts joined, as the
    Python standalone service sends them."""
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    elif content is None:
        content = ""
    template_message = {"role": message.get("role"), "content": content}
    for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
        if message.get(key) is not None:
            template_message[key] = message[key]
    return template_message


class TokenBridge:
    """The Python half of ``AtomBridgeRouter``; see the module docstring."""

    def __init__(
        self,
        engine: Any,
        sink: Any,
        tokenizer: Any,
        model: str,
        tool_call_parser: str | None = None,
    ) -> None:
        self.engine = engine
        # Read by Rust (`AtomBridgeRuntime::from_bridge`) to find its sink.
        self.sink = sink
        self._renderer = ChatRenderer(tokenizer, model)
        # Read by Rust: render every chat prompt through `render_chat`.
        self.renders_chat = self._renderer.needed
        # What `tool_choice: "required"` holds a chat's output to, as the
        # Python standalone service derives it: the tool-call channel of the
        # model's format, where the format defines one (Kimi-K3's).
        self._required_channel = required_call_channel(
            resolve_tool_call_parser(
                tool_call_parser, tokenizer, self._renderer.encoder, model=model
            ),
            tokenizer,
        )
        config = getattr(engine, "config", None)
        self._max_model_len = getattr(config, "max_model_len", None)
        self._output_limit = deployment_metadata(config, None).get("max_output_tokens")
        self._lock = threading.Lock()
        # CoreManager has one request thread (the OpenAI server's event loop):
        # its DP and prefix routers are not written for concurrent
        # `add_request`, and before `_input_send_lock` two sends interleaved
        # their ZMQ frames. Rust submits from many blocking threads, so every
        # call into CoreManager goes through this. Preprocessing stays outside,
        # as the OpenAI server runs it on its executor.
        self._engine_lock = threading.Lock()
        # key -> the request's sequence ids, until its last sibling finishes.
        self._seq_ids: dict[int, list[int]] = {}
        # Each CoreManager output thread buffers its own step.
        self._local = threading.local()
        engine.core_mgr._flush_stream_batch_fn = self._flush

    # ------------------------------------------------------------------
    # Rust -> Python
    # ------------------------------------------------------------------

    def submit(self, key: int, token_ids: list[int], params: dict[str, Any]) -> int:
        """Hand one tokenized request to the engine; how many sequences it
        made of it (``n``, or 1 where greedy sampling collapsed ``n``).

        A ``ValueError`` is the request's fault and reaches the client as 400;
        anything else is the server's.
        """
        sampling = self._sampling_params(params)
        seqs = self.engine.io_processor.preprocess_fanout(
            token_ids,
            sampling,
            stream_callbacks=[self._callback(key, i) for i in range(sampling.n)],
            parent_request_id=params.get("request_id"),
            data_parallel_rank=params.get("data_parallel_rank"),
            dp_session_id=params.get("dp_session_id"),
        )
        try:
            for seq in seqs:
                self._fit_context(seq, sampling)
            with self._lock:
                self._seq_ids[key] = [seq.id for seq in seqs]
            with self._engine_lock:
                self.engine.core_mgr.add_request(seqs)
        except BaseException:
            with self._lock:
                self._seq_ids.pop(key, None)
            self._forget([seq.id for seq in seqs])
            raise
        return len(seqs)

    def render_chat(self, request_json: str) -> tuple[str, list[int], str | None]:
        """A chat request's prompt, its token ids as the engine would encode
        the text, and the channel opener it ends on (``ChatRenderer.render``)."""
        prompt, opener = self._renderer.render(json.loads(request_json))
        encoded = self.engine.io_processor.encode_prompt(prompt)
        token_ids = getattr(encoded, "token_ids", encoded)
        return prompt, list(token_ids), opener

    def abort(self, keys: list[int]) -> None:
        """Stop the requests behind ``keys``; a finished or unknown key is a no-op."""
        for key in keys:
            with self._lock:
                seq_ids = self._seq_ids.pop(key, None)
            if not seq_ids:
                continue
            for seq_id in seq_ids:
                try:
                    with self._engine_lock:
                        self.engine.core_mgr.abort_request(seq_id)
                except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
                    logger.debug("abort_request(%s): %s", seq_id, exc)
            self._forget(seq_ids)

    def is_healthy(self) -> bool:
        return bool(self.engine.core_mgr.is_healthy())

    def close(self) -> None:
        self.engine.core_mgr._flush_stream_batch_fn = None

    # ------------------------------------------------------------------
    # Engine -> Rust
    # ------------------------------------------------------------------

    def _callback(self, key: int, index: int):
        local = self._local

        def record(request_output: Any) -> None:
            pending = getattr(local, "pending", None)
            if pending is None:
                pending = local.pending = []
            pending.append(
                (
                    key,
                    index,
                    list(request_output.output_tokens or ()),
                    bool(request_output.finished),
                    request_output.finish_reason,
                    int(getattr(request_output, "num_cached_tokens", 0) or 0),
                )
            )
            if request_output.finished:
                self._finish(key, request_output.request_id)

        return record

    def _flush(self) -> None:
        """One push per output thread per step (CoreManager calls this)."""
        pending = getattr(self._local, "pending", None)
        if not pending:
            return
        self._local.pending = None
        self.sink.push(pending)

    def _finish(self, key: int, seq_id: int) -> None:
        with self._lock:
            seq_ids = self._seq_ids.get(key)
            if seq_ids is not None and seq_id in seq_ids:
                seq_ids.remove(seq_id)
                if not seq_ids:
                    del self._seq_ids[key]
        self._forget([seq_id])

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def _sampling_params(self, params: dict[str, Any]) -> SamplingParams:
        max_tokens = params.get("max_tokens")
        max_tokens_is_default = max_tokens is None
        if max_tokens_is_default:
            max_tokens = DEFAULT_MAX_TOKENS
        if self._output_limit is not None and max_tokens > self._output_limit:
            if not max_tokens_is_default:
                raise ValueError(
                    f"max_tokens must be <= {self._output_limit} for this deployment"
                )
            max_tokens = self._output_limit
        temperature = _or(params.get("temperature"), DEFAULT_TEMPERATURE)
        return SamplingParams(
            temperature=temperature,
            n=_coerce_n(params.get("n"), temperature),
            top_k=_or(params.get("top_k"), DEFAULT_TOP_K),
            top_p=_or(params.get("top_p"), DEFAULT_TOP_P),
            max_tokens=max_tokens,
            max_tokens_is_default=max_tokens_is_default,
            stop_strings=params.get("stop") or None,
            ignore_eos=bool(params.get("ignore_eos", False)),
            required_channel=(
                self._required_channel if params.get("tool_choice_required") else None
            ),
        )

    def _fit_context(self, seq: Any, sampling: SamplingParams) -> None:
        """As the OpenAI server's ``_validate_sequence_context_length``."""
        limit = self._max_model_len
        if sampling.max_tokens_is_default and limit is not None:
            remaining = limit - seq.num_prompt_tokens
            if remaining <= 0:
                raise ValueError("The prompt leaves no context space for output tokens")
            seq.max_tokens = min(seq.max_tokens, remaining)
        _validate_context_length(seq.num_prompt_tokens, seq.max_tokens, limit)
        _validate_pool_capacity(
            seq.num_prompt_tokens,
            getattr(self.engine.core_mgr, "max_pool_tokens", None),
        )

    def _forget(self, seq_ids: list[int]) -> None:
        requests = getattr(self.engine.io_processor, "requests", None)
        if requests is None:
            return
        for seq_id in seq_ids:
            requests.pop(seq_id, None)


def _or(value: Any, default: Any) -> Any:
    return default if value is None else value
