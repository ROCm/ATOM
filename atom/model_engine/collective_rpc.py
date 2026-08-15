# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Transport-independent protocol types for collective worker RPCs."""

from __future__ import annotations

import queue
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any

COLLECTIVE_RPC_COMMAND = "__atom_collective_rpc__"
COLLECTIVE_RPC_UTILITY = "collective_rpc"


@dataclass(frozen=True, slots=True)
class RPCErrorInfo:
    type: str
    message: str
    traceback: str = ""

    @classmethod
    def from_exception(cls, exc: Exception) -> RPCErrorInfo:
        return cls(type(exc).__name__, str(exc), traceback.format_exc())

    @classmethod
    def transport(cls, message: str) -> RPCErrorInfo:
        return cls("CollectiveRPCTransportError", message)

    @classmethod
    def protocol(cls, message: str) -> RPCErrorInfo:
        return cls("CollectiveRPCProtocolError", message)


@dataclass(frozen=True, slots=True)
class CollectiveRPCRequest:
    request_id: str
    method: str
    args: tuple
    kwargs: dict
    deadline: float | None

    @classmethod
    def create(
        cls,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> CollectiveRPCRequest:
        if not isinstance(method, str) or not method:
            raise TypeError("collective RPC method must be a non-empty string")
        if not isinstance(args, tuple):
            raise TypeError("collective RPC args must be a tuple")
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise TypeError("collective RPC kwargs must be a dict")
        if timeout is not None and timeout < 0:
            raise ValueError("collective RPC timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        return cls(uuid.uuid4().hex, method, args, kwargs, deadline)

    def remaining_timeout(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())


@dataclass(frozen=True, slots=True)
class WorkerRPCResponse:
    request_id: str
    tp_rank: int
    ok: bool
    result: Any = None
    error: RPCErrorInfo | None = None

    @classmethod
    def success(cls, request_id: str, tp_rank: int, result: Any) -> WorkerRPCResponse:
        return cls(request_id=request_id, tp_rank=tp_rank, ok=True, result=result)

    @classmethod
    def failure(
        cls, request_id: str, tp_rank: int, error: RPCErrorInfo
    ) -> WorkerRPCResponse:
        return cls(request_id=request_id, tp_rank=tp_rank, ok=False, error=error)


@dataclass(frozen=True, slots=True)
class EngineCoreRPCResponse:
    request_id: str
    method: str
    tp_world_size: int
    tp_responses: tuple[WorkerRPCResponse, ...]
    error: RPCErrorInfo | None = None

    @classmethod
    def success(
        cls,
        request: CollectiveRPCRequest,
        tp_world_size: int,
        tp_responses: list[WorkerRPCResponse],
    ) -> EngineCoreRPCResponse:
        return cls(
            request_id=request.request_id,
            method=request.method,
            tp_world_size=tp_world_size,
            tp_responses=tuple(tp_responses),
        )

    @classmethod
    def failure(
        cls,
        request_id: str,
        method: str,
        error: RPCErrorInfo,
    ) -> EngineCoreRPCResponse:
        return cls(request_id, method, 0, (), error)


@dataclass(frozen=True, slots=True)
class RankedRPCFailure:
    dp_rank: int
    tp_rank: int | None
    error: RPCErrorInfo


class CollectiveRPCError(RuntimeError):
    """Raised when one or more DP/TP workers fail a collective RPC."""

    def __init__(self, method: str, failures: list[RankedRPCFailure]):
        self.method = method
        self.failures = failures
        details = "; ".join(
            f"DP{item.dp_rank}"
            f"{'/TP' + str(item.tp_rank) if item.tp_rank is not None else ''}: "
            f"{item.error.type}: {item.error.message}"
            for item in failures
        )
        super().__init__(f"collective RPC {method!r} failed: {details}")


def validate_worker_responses(response: EngineCoreRPCResponse) -> RPCErrorInfo | None:
    ranks = sorted(item.tp_rank for item in response.tp_responses)
    if response.tp_world_size < 1 or ranks != list(range(response.tp_world_size)):
        return RPCErrorInfo.protocol(
            "invalid TP response set: "
            f"world_size={response.tp_world_size}, ranks={ranks}"
        )
    return None


class CollectiveRPCResponseRouter:
    """Route request-scoped EngineCore responses to concurrent callers."""

    def __init__(self) -> None:
        self._queues: dict[str, queue.Queue] = {}
        self._lock = Lock()

    @contextmanager
    def register(self, request_id: str) -> Iterator[queue.Queue]:
        response_queue: queue.Queue = queue.Queue()
        with self._lock:
            if request_id in self._queues:
                raise RuntimeError(
                    f"collective RPC request already registered: {request_id}"
                )
            self._queues[request_id] = response_queue
        try:
            yield response_queue
        finally:
            with self._lock:
                self._queues.pop(request_id, None)

    def route(self, request_id: str, response: Any) -> bool:
        with self._lock:
            response_queue = self._queues.get(request_id)
            if response_queue is None:
                return False
            response_queue.put_nowait(response)
            return True
