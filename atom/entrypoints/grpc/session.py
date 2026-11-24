# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The Session RPC's state: each Session's engine ids in a window of their own, and per
engine step one StreamBatch per Session, then an id-free item per request it ended."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, replace

from atom.entrypoints.grpc import frames
from atom.entrypoints.openai.frontend_workers import worker_seq_id_base, worker_seq_ids
from atom.model_engine.request import RequestOutput
from atom.model_engine.zmq_protocol import engine_v1_pb2 as v1

logger = logging.getLogger("atom")

SEQ_IDS = worker_seq_id_base(0)  # a window's size: a client's seq ids lie below it
WINDOWS = 127  # window k's engine ids are worker_seq_ids(k)'s; 127 stay inside int32
# Messages a client may leave unread; one more fails its Session, which bounds the
# memory a client that stops reading holds in the API process.
OUTBOX = 4096
STRUCTURED_ERROR = "structured_output_error"  # a grammar failure's finish reason
STRUCTURED_FAILED = "Structured decoding failed for this request"  # HTTP chat's 500


@dataclass(eq=False)
class _Request:
    seq: object  # its Sequence, under the engine's id for it
    external_request_id: str | None
    arrive_time: float  # when the Session read it, before admission and preprocess
    first_token_time: float = 0.0


class Sessions:
    """The windows of the open Sessions; a freed one goes to the back of the line."""

    def __init__(self, api):
        self.api = api
        self.free = deque(range(WINDOWS))
        # Kept across a window's Sessions: an engine id returns only after 2^24 newer
        # requests, by when CoreManager has long forgotten it, as for HTTP's ids.
        self.engine_ids = [worker_seq_ids(window) for window in range(WINDOWS)]

    def open(self) -> "Session | None":
        """A Session in the next free window; None while every window is held."""
        return Session(self, self.free.popleft()) if self.free else None


class Session:
    """One Session call: its requests by client seq_id and its messages to the client.
    Its window returns once the call has ended and its last request has too."""

    def __init__(self, sessions: Sessions, window: int):
        self.sessions, self.api = sessions, sessions.api
        self.window, self.engine_ids = window, sessions.engine_ids[window]
        self.requests: dict[int, _Request] = {}
        self.client_ids: dict[int, int] = {}  # by engine id, each request's seq_id
        self.preparing = 0  # submissions in preprocess: each holds the window too
        # client seq_id -> its pending submission; held until admitted or ended
        self.admitting: dict[int, object] = {}
        self.aborted_while_admitting: set[int] = set()  # aborted once it is added
        # The step being delivered: each output with when the engine delivered it.
        self.pending: list[tuple[RequestOutput, float]] = []
        self.put = self.pending.append  # where each output goes as a step is delivered
        self.outbox: asyncio.Queue = asyncio.Queue()  # message frames; None ends them
        self.open = True
        self.failed: str | None = None  # why the Session failed its client's call
        self.reader: asyncio.Task | None = None  # admits the client's messages

    def seq_id(self, client_id: int) -> int:
        """A new engine id for `client_id`, which is below SEQ_IDS and not in flight."""
        if client_id >= SEQ_IDS:
            raise ValueError(f"seq_id {client_id} is not below {SEQ_IDS}")
        if client_id in self.requests:
            raise ValueError(f"seq_id {client_id} is already in flight")
        return next(self.engine_ids)

    def add(
        self, client_id: int, seq, external_request_id: str | None, arrive_time: float
    ) -> None:
        self.requests[client_id] = _Request(seq, external_request_id, arrive_time)
        self.client_ids[seq.id] = client_id

    def _pop(self, client_id: int) -> _Request:
        request = self.requests.pop(client_id)
        del self.client_ids[request.seq.id]
        return request

    def abandon(self, preprocessing: asyncio.Future) -> None:
        """A submission ended before `add`: the seq its preprocess made is dropped,
        then its hold on the window ends."""
        if not preprocessing.cancelled() and preprocessing.exception() is None:
            self.api.engine.io_processor.requests.pop(preprocessing.result().id, None)
        self.preparing -= 1
        self._release()

    def drop(self, client_id: int, seq) -> None:
        """Forgets a request whose submission failed before it reached the engine, unless
        it ended first and `client_id` names another request now."""
        request = self.requests.get(client_id)
        if request is None or request.seq is not seq:
            return
        self._pop(client_id)
        self.api.engine.io_processor.requests.pop(request.seq.id, None)
        self._release()

    def reject(self, refused: list[frames.Rejection]) -> None:
        if refused and self.open:
            self.send(frames.encode_rejected(refused))

    def abort(self, client_ids) -> None:
        """Aborts those in flight as a dropped HTTP stream is aborted; each still ends
        with its Completions item."""
        for client_id in client_ids:
            if client_id in self.admitting:  # aborted before the engine may have it
                self.aborted_while_admitting.add(client_id)
            elif client_id in self.requests:
                seq_id = self.requests[client_id].seq.id
                self.api.cleanup_stream(seq_id, aborted=True)

    def close(self) -> None:
        """The call has ended: every request in flight is aborted."""
        if self.open:
            self.open = False
            self.abort(list(self.requests))
        self._release()

    def send(self, message: list[bytes]) -> None:
        """Queues `message` for the client; past OUTBOX unread ones, the Session fails
        instead: its messages dropped, its requests aborted, its call ended."""
        if not self.open:
            return
        if self.outbox.qsize() < OUTBOX:
            self.outbox.put_nowait(message)
            return
        self.failed = f"the client left {OUTBOX} Session messages unread"
        logger.error("Session window %s: %s", self.window, self.failed)
        while not self.outbox.empty():
            self.outbox.get_nowait()
        self.close()
        if self.reader is not None:  # admits nothing after close() aborted what it held
            self.reader.cancel()
        self.outbox.put_nowait(None)

    def flush(self) -> None:
        """The step's outputs: one StreamBatch of those carrying ids, then a Completions
        item, or a Rejected one, for each request the step ended."""
        pending = self.pending[:]
        self.pending.clear()  # the list `put` appends to
        requests = self.requests
        entries, done = [], []
        for output, delivered in pending:
            client_id = self.client_ids.get(output.request_id)
            if client_id is None:  # dropped when its submission failed
                continue
            request = requests[client_id]
            if output.output_tokens:
                if not request.first_token_time:
                    request.first_token_time = delivered
                streamed = output
                if output.kv_transfer_params_output is not None:
                    streamed = _streamed(output)
                entries.append((client_id, streamed))
            if output.finished:
                done.append(_completion(client_id, request, output, delivered))
        if self.open and entries:
            self.send(frames.encode_stream(entries))
        failed = [c for c in done if c.leave_reason == STRUCTURED_ERROR]
        completed = [c for c in done if c.leave_reason != STRUCTURED_ERROR]
        if self.open and failed:  # HTTP chat answers a grammar failure with a 500
            code = v1.RejectedRequest.INTERNAL
            refused = [
                frames.Rejection(c.seq_id, STRUCTURED_FAILED, code) for c in failed
            ]
            self.send(frames.encode_rejected(refused))
        if self.open and completed:
            try:
                self.send(frames.encode_completions(completed))
            except TypeError:  # kv_transfer_params engine_v1 cannot carry
                for completion in completed:
                    self.send(_completed_or_rejected(completion))
        for completion in done:
            request = self._pop(completion.seq_id)
            # Ended while its admission settles: its seq_id is the client's again.
            self.admitting.pop(completion.seq_id, None)
            self.aborted_while_admitting.discard(completion.seq_id)
            self.api.cleanup_stream(request.seq.id)
        self._release()

    def _release(self) -> None:
        held = self.requests or self.preparing
        if not self.open and not held and self.window is not None:
            self.sessions.free.append(self.window)
            self.window = None


def _streamed(output: RequestOutput) -> RequestOutput:
    """`output` for a StreamBatch: its kv_transfer_params go in its Completions item."""
    return replace(output, kv_transfer_params_output=None)


def _completion(client_id, request, output, delivered: float) -> frames.Completion:
    """Its Completions item: no ids, which its StreamBatch entries carried."""
    seq = request.seq
    return frames.Completion(
        seq_id=client_id,
        completion_token_ids=[],
        leave_reason=output.finish_reason or "",
        num_prompt_tokens=seq.num_prompt_tokens,
        num_cached_tokens=seq.num_prompt_tokens if request.first_token_time else 0,
        arrive_time=request.arrive_time,
        first_token_time=request.first_token_time,
        leave_time=delivered,
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
