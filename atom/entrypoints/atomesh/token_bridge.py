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
* A P/D leg's ``kv_transfer_params`` go to the engine with its prompt, and a
  prefill's handoff (``kv_transfer_params_output``) comes back with its
  finish, for the Atomesh router to hand to decode; ``kv_transfer_info`` is
  what the router reads of this worker at registration (``pd_handoff``).
* Each engine step's tokens go back to Rust in one ``StepSink.push`` from
  CoreManager's output thread, the way the frontend workers' hub flushes a
  step per worker (``frontend_workers.py``).

Rust renders every chat prompt, Jinja templates and the encoders of
DeepSeek-V4.1 and Kimi-K3 alike (``atom/mesh/src/routers/prepare/chat_render.rs``);
Python takes token ids only.

A request is known by the key Rust gives it, not by sequence id: Rust has
registered the key before ``submit`` is called, so no output can arrive for a
request Rust does not know yet.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from atom import SamplingParams, structured_output
from atom.entrypoints.openai.api_server import (
    _coerce_n,
    _validate_context_length,
    _validate_pool_capacity,
)
from atom.entrypoints.openai.chat_encoders import (
    chat_template_source,
    load_custom_message_encoder,
    render_probe_prompt,
)
from atom.entrypoints.openai.model_capabilities import deployment_metadata
from atom.entrypoints.openai.pd_handoff import (
    chunked_receive_refusal,
    is_chunked_receive,
    kv_transfer_info,
)
from atom.entrypoints.openai.protocol import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    ChatCompletionRequest,
    check_media_handoff,
)
from atom.entrypoints.openai.reasoning import template_opens_reasoning_implicitly
from atom.entrypoints.openai.reasoning_dialects import resolve_dialect
from atom.entrypoints.openai.serving_chat import normalize_chat_tools, request_tools
from atom.entrypoints.openai.tool_parser.registry import (
    required_call_channel,
    resolve_tool_call_parser,
)
from atom.structured_output.validation import tool_call_grammar_format
from atom.utils import envs

logger = logging.getLogger("atom")


class ChatFormat:
    """What admission needs to know of the model's chats, read at startup as
    the OpenAI server reads it: its message encoder (the output cap
    `deployment_metadata` takes), its tool-call format, and the reasoning
    opener a prompt can end on. Rust renders the chats themselves."""

    def __init__(
        self, tokenizer: Any, model: str, tool_call_parser: str | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = load_custom_message_encoder(model)
        self.tool_parser_cls = resolve_tool_call_parser(
            tool_call_parser, tokenizer, self.encoder, model=model
        )
        self.checks_tool_call_grammar = self.holds_tool_calls(self.tool_parser_cls)
        if not self.checks_tool_call_grammar:
            return
        source = chat_template_source(tokenizer, self.encoder)
        self.dialect, _ = resolve_dialect(
            source, render_probe_prompt(tokenizer, self.encoder, tools=False) or ""
        )
        self.model_starts_in_reasoning = template_opens_reasoning_implicitly(source)

    @staticmethod
    def holds_tool_calls(tool_parser_cls: Any) -> bool:
        """Whether the OpenAI server can hold a chat's tool calls to their
        schemas: a tool-call format, ATOM_TOOL_CALL_GRAMMAR and the
        structured-output extras (`configure_tool_call_grammar`)."""
        return bool(
            tool_parser_cls is not None
            and envs.ATOM_TOOL_CALL_GRAMMAR
            and structured_output.available()
        )

    def refuse_unheld_tool_calls(self, chat: dict[str, Any], token_ids: list[int]):
        """Refuse a chat whose tool calls the OpenAI server would hold to their
        schemas, decided as `configure_tool_call_grammar` decides it (Kimi-K3,
        with ATOM_TOOL_CALL_GRAMMAR and the structured-output backend): the
        bridge carries no grammar yet, and unconstrained K3 can copy an empty
        call out of its history. Whether the reply starts inside reasoning is
        read off the prompt's tail, as `_chat_completions` reads a prompt given
        as token ids."""
        if not self.checks_tool_call_grammar:
            return
        request = ChatCompletionRequest(
            **{
                key: chat[key]
                for key in ("messages", "tools", "tool_choice")
                if chat.get(key) is not None
            }
        )
        request.tools = normalize_chat_tools(request.tools)
        tools = request_tools(request)
        if not tools:
            return
        tail = self.tokenizer.decode(token_ids[-32:]).rstrip()
        marker = self.dialect.prompt_open_marker
        starts_open = bool(marker and tail.endswith(marker))
        starts_open = starts_open or self.model_starts_in_reasoning
        try:
            fmt = tool_call_grammar_format(
                self.tool_parser_cls, tools, chat.get("tool_choice"), starts_open
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
        self._format = ChatFormat(tokenizer, model, tool_call_parser)
        # Read by Rust: hand `submit` a chat's body to check its tool calls.
        self.checks_tool_call_grammar = self._format.checks_tool_call_grammar
        # What `tool_choice: "required"` holds a chat's output to: the
        # tool-call channel of the model's format, where it defines one (K3's).
        self._required_channel = required_call_channel(
            self._format.tool_parser_cls, tokenizer
        )
        config = getattr(engine, "config", None)
        # Read by Rust: every id that ends an output, which its decode leaves
        # out of the text as the OpenAI server's does.
        self.end_token_ids = list(getattr(config, "end_token_ids", ()) or ())
        self._max_model_len = getattr(config, "max_model_len", None)
        # The OpenAI server's cap, its message encoder's included (V4.1's).
        self._output_limit = deployment_metadata(config, self._format.encoder).get(
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
        if params.get("chat"):
            self._format.refuse_unheld_tool_calls(json.loads(params["chat"]), token_ids)
        sampling = self._sampling_params(params)
        kv_transfer_params = self._kv_transfer_params(params.get("kv_transfer_params"))
        # The session lineage the OpenAI server reads from the same headers
        # (`_get_dp_session_affinity_ids`), and only with the same switch.
        affinity = envs.ATOM_DP_SESSION_AFFINITY
        seqs = self.engine.io_processor.preprocess_fanout(
            token_ids,
            sampling,
            stream_callbacks=[self._callback(key, i) for i in range(sampling.n)],
            kv_transfer_params=kv_transfer_params,
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

    def kv_transfer_info(self) -> str:
        """``GET /kv_transfer_info``, as JSON."""
        return json.dumps(kv_transfer_info(self.engine.config))

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
                    _handoff(request_output),
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

    def _kv_transfer_params(self, sent: str | None) -> dict[str, Any] | None:
        """A P/D leg's handoff as JSON, refused as the OpenAI server refuses
        it (`_validate_kv_transfer_params`) before it is enqueued. Rust has
        taken the prompt's ids out of it (they are the prompt it submits)."""
        if not sent:
            return None
        params = json.loads(sent)
        if is_chunked_receive(params):
            refusal = chunked_receive_refusal(params, self.engine.config)
            if refusal is not None:
                raise ValueError(refusal)
        check_media_handoff(params, self._max_model_len)
        return params

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


def _handoff(request_output: Any) -> str | None:
    """A finished prefill's handoff for its decode, as JSON; None otherwise."""
    if not request_output.finished:
        return None
    handoff = getattr(request_output, "kv_transfer_params_output", None)
    return json.dumps(handoff) if handoff else None
