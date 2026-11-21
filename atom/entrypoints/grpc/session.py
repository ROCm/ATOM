# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Session RPC's state: each Session's seq ids in a window of their own, and per
engine step one StreamBatch per Session, then an id-free item per request it ended."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, replace

from atom.entrypoints.grpc import frames
from atom.entrypoints.openai.frontend_workers import worker_seq_id_base
from atom.model_engine.request import RequestOutput
from atom.model_engine.zmq_protocol import engine_v1_pb2 as v1

logger = logging.getLogger("atom")

SEQ_IDS = worker_seq_id_base(0)  # a window's size: a client's seq ids lie below it
WINDOWS = 127  # window k holds worker_seq_id_base(k) | seq_id; 127 stay inside int32


@dataclass(eq=False)
class _Request:
    seq: object  # its Sequence, under the engine's id for it
    external_request_id: str | None
    first_token_time: float = 0.0


class Sessions:
    """The windows of the open Sessions; a freed one goes to the back of the line."""

    def __init__(self, api):
        self.api = api
        self.free = deque(range(WINDOWS))

    def open(self) -> "Session | None":
        """A Session in the next free window; None while every window is held."""
        return Session(self, self.free.popleft()) if self.free else None


class Session:
    """One Session call: its requests by client seq_id and its messages to the client.
    Its window returns once the call has ended and its last request has too."""

    def __init__(self, sessions: Sessions, window: int):
        self.sessions, self.api = sessions, sessions.api
        self.window, self.base = window, worker_seq_id_base(window)
        self.requests: dict[int, _Request] = {}
        self.pending: list[RequestOutput] = []  # the step being delivered
        self.put = self.pending.append  # where each output goes as a step is delivered
        self.outbox: asyncio.Queue = asyncio.Queue()  # message frames; None ends them
        self.open = True

    def seq_id(self, client_id: int) -> int:
        """The engine's id for `client_id`, which is below SEQ_IDS and not in flight."""
        if client_id >= SEQ_IDS:
            raise ValueError(f"seq_id {client_id} is not below {SEQ_IDS}")
        if client_id in self.requests:
            raise ValueError(f"seq_id {client_id} is already in flight")
        return self.base | client_id

    def add(self, client_id: int, seq, external_request_id: str | None) -> None:
        self.requests[client_id] = _Request(seq, external_request_id)

    def drop(self, client_id: int) -> None:
        """Forgets a request whose submission failed."""
        request = self.requests.pop(client_id)
        self.api.engine.io_processor.requests.pop(request.seq.id, None)
        self._release()

    def reject(self, refused: list[frames.Rejection]) -> None:
        if refused and self.open:
            self.outbox.put_nowait(frames.encode_rejected(refused))

    def abort(self, client_ids) -> None:
        """Aborts those in flight as a dropped HTTP stream is aborted; each still ends
        with its Completions item."""
        for client_id in client_ids:
            if client_id in self.requests:
                seq_id = self.requests[client_id].seq.id
                self.api.cleanup_stream(seq_id, aborted=True)

    def close(self) -> None:
        """The call has ended: every request in flight is aborted."""
        self.open = False
        self.abort(list(self.requests))
        self._release()

    def flush(self) -> None:
        """The step's outputs: one StreamBatch of those carrying ids, then a Completions
        item, or a Rejected one, for each request the step ended."""
        pending = self.pending[:]
        self.pending.clear()  # the list `put` appends to
        requests, base, now = self.requests, self.base, time.time()
        entries, done = [], []
        for output in pending:
            client_id = output.request_id - base  # the engine's id is base | client id
            request = requests.get(client_id)
            if request is None:  # dropped when its submission failed
                continue
            if output.output_tokens:
                if not request.first_token_time:
                    request.first_token_time = now
                streamed = output
                if output.kv_transfer_params_output is not None:
                    streamed = _streamed(output)
                entries.append((client_id, streamed))
            if output.finished:
                done.append(_completion(client_id, request, output, now))
        if self.open and entries:
            self.outbox.put_nowait(frames.encode_stream(entries))
        if self.open and done:
            try:
                self.outbox.put_nowait(frames.encode_completions(done))
            except TypeError:  # kv_transfer_params engine_v1 cannot carry
                for completion in done:
                    self.outbox.put_nowait(_completed_or_rejected(completion))
        for completion in done:
            request = self.requests.pop(completion.seq_id)
            self.api.cleanup_stream(request.seq.id)
        self._release()

    def _release(self) -> None:
        if not self.open and not self.requests and self.window is not None:
            self.sessions.free.append(self.window)
            self.window = None


def _streamed(output: RequestOutput) -> RequestOutput:
    """`output` for a StreamBatch: its kv_transfer_params go in its Completions item."""
    return replace(output, kv_transfer_params_output=None)


def _completion(client_id, request, output, now: float) -> frames.Completion:
    """Its Completions item: no ids, which its StreamBatch entries carried."""
    seq = request.seq
    return frames.Completion(
        seq_id=client_id,
        completion_token_ids=[],
        leave_reason=output.finish_reason or "",
        num_prompt_tokens=seq.num_prompt_tokens,
        num_cached_tokens=seq.num_prompt_tokens if request.first_token_time else 0,
        arrive_time=seq.arrive_time,
        first_token_time=request.first_token_time,
        leave_time=now,
        kv_transfer_params_output=output.kv_transfer_params_output,
        external_request_id=request.external_request_id,
    )


def _completed_or_rejected(completion: frames.Completion) -> list[bytes]:
    """Its Completions item, or a Rejected INTERNAL one as Generate answers a 500."""
    try:
        return frames.encode_completions([completion])
    except TypeError as error:
        logger.error("Session seq_id %d: %s", completion.seq_id, error)
        code = v1.RejectedRequest.INTERNAL
        refused = frames.Rejection(completion.seq_id, str(error), code)
        return frames.encode_rejected([refused])
