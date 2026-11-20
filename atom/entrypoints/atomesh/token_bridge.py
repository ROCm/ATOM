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

A request is known by the key Rust gives it, not by sequence id: Rust has
registered the key before ``submit`` is called, so no output can arrive for a
request Rust does not know yet.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from atom import SamplingParams
from atom.entrypoints.openai.api_server import (
    _validate_context_length,
    _validate_pool_capacity,
)
from atom.entrypoints.openai.model_capabilities import deployment_metadata
from atom.entrypoints.openai.protocol import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)

logger = logging.getLogger("atom")


class TokenBridge:
    """The Python half of ``AtomBridgeRouter``; see the module docstring."""

    def __init__(self, engine: Any, sink: Any) -> None:
        self.engine = engine
        # Read by Rust (`AtomBridgeRuntime::from_bridge`) to find its sink.
        self.sink = sink
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

    def submit(self, key: int, token_ids: list[int], params: dict[str, Any]) -> None:
        """Hand one tokenized request to the engine.

        A ``ValueError`` is the request's fault and reaches the client as 400;
        anything else is the server's.
        """
        sampling = self._sampling_params(params)
        seqs = self.engine.io_processor.preprocess_fanout(
            token_ids,
            sampling,
            stream_callback=self._callback(key),
            parent_request_id=params.get("request_id"),
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

    def _callback(self, key: int):
        local = self._local

        def record(request_output: Any) -> None:
            pending = getattr(local, "pending", None)
            if pending is None:
                pending = local.pending = []
            pending.append(
                (
                    key,
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
        return SamplingParams(
            temperature=_or(params.get("temperature"), DEFAULT_TEMPERATURE),
            top_k=_or(params.get("top_k"), DEFAULT_TOP_K),
            top_p=_or(params.get("top_p"), DEFAULT_TOP_P),
            max_tokens=max_tokens,
            max_tokens_is_default=max_tokens_is_default,
            stop_strings=params.get("stop") or None,
            ignore_eos=bool(params.get("ignore_eos", False)),
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
