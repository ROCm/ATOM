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

import itertools
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from atom import SamplingParams, structured_output
from atom.entrypoints.openai.api_compatibility import api_compatibility_for
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
    template_needs_tool_call_argument_strings,
)
from atom.entrypoints.openai.model_capabilities import deployment_metadata
from atom.entrypoints.openai.protocol import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    ChatCompletionRequest,
)
from atom.entrypoints.openai.reasoning import (
    template_opens_reasoning_implicitly,
    thinking_switched_off,
)
from atom.entrypoints.openai.reasoning_controls import (
    CHAT_REASONING_RULES,
    apply_reasoning_controls,
    resolve_reasoning_controls,
)
from atom.entrypoints.openai.reasoning_dialects import resolve_dialect
from atom.entrypoints.openai.serving_chat import (
    normalize_chat_tools,
    request_tools,
    validate_chat_request,
)
from atom.entrypoints.openai.tool_parser.registry import (
    required_call_channel,
    resolve_tool_call_parser,
    template_keeps_tool_call_arguments,
)
from atom.model_engine.segment_tokenizer import PrefixEncoding
from atom.structured_output.validation import tool_call_grammar_format
from atom.utils import envs

logger = logging.getLogger("atom")

# Rendered prompts whose encoding waits for its submit. A request refused
# between the two never collects its entry, so an entry older than this, or
# the oldest beyond the count, goes; a submit normally follows in ms.
_RENDERED_MAX = 1024
_RENDERED_TTL_S = 30.0


class ChatRenderer:
    """A chat prompt as ATOM's own server renders it, for a model whose
    template the router cannot render (``needed``): one with a Python message
    encoder, or a tokenizer whose ``chat_template`` is code, not Jinja. Also
    the model's tool-call format, as the OpenAI server resolves it."""

    def __init__(
        self, tokenizer: Any, model: str, tool_call_parser: str | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = load_custom_message_encoder(model)
        self.tool_parser_cls = resolve_tool_call_parser(
            tool_call_parser, tokenizer, self.encoder, model=model
        )
        # A Jinja template (one string, or named variants) is the router's
        # to render; an encoder, or a template that is code, is not.
        self.needed = self.encoder is not None or not chat_template_source(tokenizer)
        if not self.needed:
            return
        source = chat_template_source(tokenizer, self.encoder)
        self.dialect, _ = resolve_dialect(
            source, render_probe_prompt(tokenizer, self.encoder, tools=False) or ""
        )
        self.model_starts_in_reasoning = template_opens_reasoning_implicitly(source)
        self.reasoning_toggle = resolve_reasoning_toggle(tokenizer, self.encoder)
        # As the OpenAI server decides it at startup: history tool-call
        # arguments go to the template as sent when it keeps or needs them so.
        keeps = template_keeps_tool_call_arguments(
            render_probe_prompt(tokenizer, self.encoder, tools=True)
        )
        self.preserve_tool_call_arguments = (
            keeps or template_needs_tool_call_argument_strings(tokenizer, self.encoder)
        )

    def render(self, request: dict[str, Any]) -> tuple[str, str | None]:
        """The prompt, and the channel opener it ends on (or ``None``).

        Rendered as the OpenAI server's `_chat_completions` renders it: the
        request read into its `ChatCompletionRequest`, tools normalized and
        checked, messages as `ChatMessage`s (`to_template_dict` keeps a
        content list for an encoder that joins it, DeepSeek-V4.1's).

        The opener is what the reply starts inside, with none of its own: the
        dialect's reasoning opener when the prompt opens reasoning, or its
        answer opener (Kimi-K3's response channel with thinking off).
        """
        chat = ChatCompletionRequest(
            **{
                key: request[key]
                for key in (
                    "messages",
                    "tools",
                    "tool_choice",
                    "thinking",
                    "reasoning_effort",
                )
                if request.get(key) is not None
            }
        )
        if not chat.messages:
            raise ValueError("messages is required")
        chat.tools = normalize_chat_tools(chat.tools)
        validate_chat_request(chat)
        kwargs = self._template_kwargs(request)
        compatibility = api_compatibility_for(self.encoder)
        if not compatibility.chat_forced_tools_with_thinking:
            compatibility.validate_chat_tool_choice(
                chat.tool_choice, thinking_enabled=self._thinking_enabled(kwargs)
            )
        prompt = apply_chat_template(
            self.tokenizer,
            self.encoder,
            chat.get_messages(),
            tools=chat.tools,
            preserve_tool_call_arguments=self.preserve_tool_call_arguments,
            **kwargs,
        )
        opener = self._opener(prompt, kwargs)
        # Every tool the request declares, K3's in-message ones included, as
        # the OpenAI server hands them to `configure_tool_call_grammar`.
        self._refuse_unheld_tool_calls(
            request_tools(chat), request.get("tool_choice"), opener
        )
        return prompt, opener

    def _template_kwargs(self, request: dict[str, Any]) -> dict[str, Any]:
        """The request's template kwargs, with what the OpenAI server's
        `_chat_completions` threads through besides: a string `tool_choice`
        (Kimi-K3's template reads it) and `thinking` / `reasoning_effort`,
        resolved by the encoder's own rules (DeepSeek-V4.1's effort aliases
        and budgets)."""
        kwargs = dict(request.get("chat_template_kwargs") or {})
        if isinstance(request.get("tool_choice"), str):
            kwargs["tool_choice"] = request["tool_choice"]
        rules = getattr(self.encoder, "reasoning_rules", None) or CHAT_REASONING_RULES
        controls = resolve_reasoning_controls(
            thinking=request.get("thinking"),
            effort=request.get("reasoning_effort"),
            rules=rules,
        )
        return apply_reasoning_controls(
            kwargs, self.reasoning_toggle, controls, rules=rules
        )

    def _thinking_enabled(self, kwargs: dict[str, Any]) -> bool:
        """As the OpenAI server's `_effective_thinking_enabled`: the template
        kwargs read through the encoder's defaults and normalization."""
        effective = dict(getattr(self.encoder, "defaults", {}))
        effective.update(kwargs)
        if self.encoder is not None:
            effective = self.encoder.prepare_kwargs(effective)
        if self.reasoning_toggle is None:
            return True
        name, off_value, on_value = self.reasoning_toggle
        return effective.get(name, on_value) != off_value

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

    def _refuse_unheld_tool_calls(self, tools, tool_choice, opener) -> None:
        """Refuse a chat whose tool calls the OpenAI server would hold to their
        schemas, decided as `configure_tool_call_grammar` decides it (Kimi-K3,
        with ATOM_TOOL_CALL_GRAMMAR and the structured-output extras): the
        bridge carries no grammar yet, and unconstrained K3 can copy an empty
        call out of its history."""
        if not tools or self.tool_parser_cls is None:
            return
        if not envs.ATOM_TOOL_CALL_GRAMMAR or not structured_output.available():
            return
        starts_open = opener is not None and opener == self.dialect.prompt_open_marker
        try:
            fmt = tool_call_grammar_format(
                self.tool_parser_cls, tools, tool_choice, starts_open
            )
        except ValueError:  # a schema the server leaves unconstrained
            return
        if fmt is not None:
            raise ValueError(
                "tool calls held to their schemas (ATOM_TOOL_CALL_GRAMMAR) are not "
                "supported by the ATOM Rust frontend yet; use the OpenAI server "
                "(USE_ATOMESH_ENTRYPOINTS=0)"
            )


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
        self._renderer = ChatRenderer(tokenizer, model, tool_call_parser)
        # Read by Rust: render every chat prompt through `render_chat`.
        self.renders_chat = self._renderer.needed
        # What `tool_choice: "required"` holds a chat's output to: the
        # tool-call channel of the model's format, where it defines one (K3's).
        self._required_channel = required_call_channel(
            self._renderer.tool_parser_cls, tokenizer
        )
        config = getattr(engine, "config", None)
        # Read by Rust: every id that ends an output, which its decode leaves
        # out of the text as the OpenAI server's does.
        self.end_token_ids = list(getattr(config, "end_token_ids", ()) or ())
        self._max_model_len = getattr(config, "max_model_len", None)
        # The OpenAI server's cap, its message encoder's included (V4.1's).
        self._output_limit = deployment_metadata(config, self._renderer.encoder).get(
            "max_output_tokens"
        )
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
        # handle -> the PrefixEncoding `render_chat` built (token ids with the
        # tokenizer cache's block-hash chain), until its request's `submit`.
        self._rendered: OrderedDict[int, tuple[float, PrefixEncoding]] = OrderedDict()
        self._render_handles = itertools.count(1)
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
        prompt = self._rendered_encoding(params.get("prompt_handle"), token_ids)
        # The session lineage the OpenAI server reads from the same headers
        # (`_get_dp_session_affinity_ids`), and only with the same switch.
        affinity = envs.ATOM_DP_SESSION_AFFINITY
        seqs = self.engine.io_processor.preprocess_fanout(
            prompt,
            sampling,
            stream_callbacks=[self._callback(key, i) for i in range(sampling.n)],
            parent_request_id=params.get("request_id"),
            data_parallel_rank=params.get("data_parallel_rank"),
            dp_session_id=params.get("dp_session_id") if affinity else None,
            dp_parent_session_id=(
                params.get("dp_parent_session_id") if affinity else None
            ),
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

    def render_chat(
        self, request_json: str
    ) -> tuple[str, list[int], str | None, int | None]:
        """A chat request's prompt, its token ids as the engine would encode
        the text, the channel opener it ends on (``ChatRenderer.render``), and
        a handle to the encoding for ``submit``, or None.

        The encoding keeps the block-hash chain the tokenizer cache built with
        the ids (extended turn by turn in a session); ``submit`` hands it to
        ``preprocess_fanout`` so the prompt is not hashed a second time.
        """
        prompt, opener = self._renderer.render(json.loads(request_json))
        encoded = self.engine.io_processor.encode_prompt(prompt)
        if not isinstance(encoded, PrefixEncoding):
            return prompt, list(encoded), opener, None
        handle = next(self._render_handles)
        now = time.monotonic()
        with self._lock:
            rendered = self._rendered
            rendered[handle] = (now, encoded)
            # Oldest first: drop what aged out, then what is over the count.
            while rendered and (
                len(rendered) > _RENDERED_MAX
                or now - next(iter(rendered.values()))[0] > _RENDERED_TTL_S
            ):
                rendered.popitem(last=False)
        return prompt, list(encoded.token_ids), opener, handle

    def _rendered_encoding(
        self, handle: int | None, token_ids: list[int]
    ) -> PrefixEncoding | list[int]:
        """The encoding `render_chat` kept for these ids, else the ids."""
        if handle is None:
            return token_ids
        with self._lock:
            entry = self._rendered.pop(handle, None)
        if entry is None or entry[1].token_ids != token_ids:
            return token_ids
        return entry[1]

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

    def close(self) -> bool:
        self.engine.core_mgr._flush_stream_batch_fn = None
        return True

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
                    request_output.num_cached_tokens or 0,
                )
            )
            if request_output.finished:
                self._finish(key, request_output.request_id)

        return record

    def _flush(self) -> None:
        """One push per output thread per step (CoreManager calls this).

        A push that fails would lose the step for good -- and with it any
        finish, leaving those streams open forever -- since CoreManager only
        logs the error. Its requests are ended with an error instead, and
        stopped in the engine.
        """
        pending = getattr(self._local, "pending", None)
        if not pending:
            return
        self._local.pending = None
        try:
            self.sink.push(pending)
        except Exception as exc:
            keys = sorted({output[0] for output in pending})
            logger.exception("TokenBridge: lost an engine step for keys %s", keys)
            self.sink.fail(keys, f"lost an engine step: {exc}")
            # `abort` forgets the keys before it stops anything, so a flush the
            # abort itself triggers finds nothing more of theirs to stop.
            self.abort(keys)

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
        """Drop what preprocessing recorded of the sequences: the request, and
        the request-id maps a request id given for ``n == 1`` fills (only the
        offline ``postprocess`` pops those)."""
        io = self.engine.io_processor
        to_external = getattr(io, "_internal_to_external", {})
        to_internal = getattr(io, "_external_to_internal", {})
        for seq_id in seq_ids:
            io.requests.pop(seq_id, None)
            external = to_external.pop(seq_id, None)
            if external is not None and to_internal.get(external) == seq_id:
                del to_internal[external]


def _or(value: Any, default: Any) -> Any:
    return default if value is None else value
