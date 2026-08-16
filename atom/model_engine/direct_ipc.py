"""Versioned MessagePack ingress owned by an EngineCore."""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import msgspec
import zmq

from atom.utils import get_open_zmq_ipc_path, make_zmq_socket

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class EngineCoreIpcEndpoint:
    address: str
    dp_rank: int
    pp_rank: int
    protocol_version: int = PROTOCOL_VERSION

    def as_dict(self) -> dict[str, int | str]:
        return self.__dict__


def allocate_endpoint(dp_rank: int, pp_rank: int) -> EngineCoreIpcEndpoint:
    return EngineCoreIpcEndpoint(get_open_zmq_ipc_path(), dp_rank, pp_rank)


@dataclass(frozen=True)
class DirectEngineRequest:
    request_id: str
    token_ids: list[int]
    sampling: dict[str, Any]
    stop_token_sequences: list[list[int]]
    kv_transfer_params: dict[str, Any] | None
    num_draft_tokens: int
    n: int
    data_parallel_rank: int | None

    @classmethod
    def from_frame(cls, frame: dict[str, Any]) -> "DirectEngineRequest":
        if frame.get("version") != PROTOCOL_VERSION or frame.get("type") != "submit":
            raise ValueError("unsupported direct EngineCore request")
        token_ids = frame.get("token_ids")
        request_id = frame.get("request_id")
        if not isinstance(request_id, str) or not request_id or not isinstance(token_ids, list) or not token_ids:
            raise ValueError("request_id and non-empty token_ids are required")
        data_parallel_rank = frame.get("data_parallel_rank")
        if data_parallel_rank is not None and (
            not isinstance(data_parallel_rank, int) or data_parallel_rank < 0
        ):
            raise ValueError("data_parallel_rank must be a non-negative integer")
        return cls(
            request_id, token_ids, frame.get("sampling") or {},
            frame.get("stop_token_sequences") or [], frame.get("kv_transfer_params"),
            int(frame.get("num_draft_tokens", 0)), int(frame.get("n", 1)),
            data_parallel_rank,
        )


class DirectEngineServer:
    """Owns one ROUTER socket; EngineCore threads enqueue output only."""

    def __init__(
        self, endpoint: EngineCoreIpcEndpoint, submit: Callable, abort: Callable
    ) -> None:
        self.endpoint, self._submit, self._abort = endpoint, submit, abort
        self._seq_routes: dict[int, tuple[bytes, str]] = {}
        self._requests: dict[str, list[int]] = {}
        self._outgoing: queue.Queue = queue.Queue()
        self._closed = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._serve, daemon=True).start()

    def close(self) -> None:
        self._closed.set()
        for request_id in self._requests:
            self._abort(request_id)

    def publish_stream(self, seq_id: int, token_ids: list[int], finished: bool,
                       finish_reason: str | None, num_cached_tokens: int,
                       kv_transfer_params: dict | None) -> None:
        route = self._seq_routes.get(seq_id)
        if route is None:
            return
        identity, request_id = route
        self._outgoing.put((identity, {
            "version": PROTOCOL_VERSION, "type": "token", "seq_id": seq_id,
            "token_ids": token_ids, "finished": finished,
            "finish_reason": finish_reason, "num_cached_tokens": num_cached_tokens,
            "kv_transfer_params": kv_transfer_params,
        }))
        if finished:
            self._seq_routes.pop(seq_id, None)
            seqs = self._requests.get(request_id, [])
            if seq_id in seqs:
                seqs.remove(seq_id)
            if not seqs:
                self._requests.pop(request_id, None)

    def _serve(self) -> None:
        with zmq.Context() as context, make_zmq_socket(
            context, self.endpoint.address, zmq.ROUTER, bind=True
        ) as socket:
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            while not self._closed.is_set():
                for readable, _ in poller.poll(50):
                    identity, payload = readable.recv_multipart()
                    self._handle(identity, payload)
                while not self._outgoing.empty():
                    identity, frame = self._outgoing.get_nowait()
                    socket.send_multipart([identity, msgspec.msgpack.encode(frame)])

    def _handle(self, identity: bytes, payload: bytes) -> None:
        request_id: str | None = None
        try:
            frame = msgspec.msgpack.decode(payload)
            if isinstance(frame, dict):
                request_id = frame.get("request_id")
            if frame.get("type") == "abort":
                self._abort(frame["request_id"])
                return
            request = DirectEngineRequest.from_frame(frame)
            if (
                request.data_parallel_rank is not None
                and request.data_parallel_rank != self.endpoint.dp_rank
            ):
                raise ValueError(
                    f"request targets DP rank {request.data_parallel_rank}, "
                    f"but endpoint belongs to rank {self.endpoint.dp_rank}"
                )
            seq_ids = self._submit(request)
            self._requests[request.request_id] = seq_ids
            for seq_id in seq_ids:
                self._seq_routes[seq_id] = (identity, request.request_id)
            self._outgoing.put((identity, {
                "version": PROTOCOL_VERSION, "type": "accepted",
                "request_id": request.request_id, "seq_ids": seq_ids,
            }))
        except Exception as error:
            self._outgoing.put((identity, {
                "version": PROTOCOL_VERSION, "type": "error",
                "request_id": request_id, "message": str(error),
            }))
