"""Experimental counter tokens; only the bounded probe owns channel lifetime."""

import ctypes
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

LIMIT = (1 << 63) - 1


def preflight():
    # Ordinary HostRegister memory is incompatible with CP signal-memory waits.
    if os.environ.get("GPU_STREAMOPS_CP_WAIT", "0") != "0":
        raise RuntimeError("GPU_STREAMOPS_CP_WAIT must be unset or 0 for this probe")


class Native:
    def __init__(self, build):
        preflight()
        self.lib = ctypes.CDLL(str(build / "timeline.so"))
        pointer = ctypes.c_void_p
        u64 = ctypes.c_uint64
        self.lib.k3_error.restype = ctypes.c_char_p
        signatures = {
            "k3_open": [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_size_t,
                ctypes.POINTER(pointer),
            ],
            "k3_record": [pointer, ctypes.c_size_t, ctypes.POINTER(u64)],
            "k3_wait": [pointer, ctypes.c_size_t, u64],
            "k3_query": [pointer, ctypes.POINTER(u64)],
            "k3_close_after_drain": [pointer, u64],
        }
        for name, signature in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = signature
            function.restype = ctypes.c_int

    def call(self, name, *args):
        if getattr(self.lib, name)(*args):
            raise RuntimeError(self.lib.k3_error().decode())


class Channel:
    def __init__(self, native, epoch, rank, direction, owner, stream):
        if not re.fullmatch(r"[0-9a-f]{32}", epoch):
            raise ValueError("Invalid epoch")
        if type(rank) is not int or not 0 <= rank < 8 or direction not in ("p", "c"):
            raise ValueError("Invalid channel identity")
        self.native, self.epoch, self.rank = native, epoch, rank
        self.direction, self.owner = direction, owner
        self.path = Path(f"/dev/shm/k3-timeline-{epoch}-{rank}-{direction}")
        self.pointer = ctypes.c_void_p()
        self.lock = threading.Lock()
        self.closed = False
        self.records = self.waits = self.queries = 0
        self._stream_id(stream)
        native.call(
            "k3_open",
            str(self.path).encode(),
            epoch.encode(),
            rank,
            int(owner),
            stream.cuda_stream,
            ctypes.byref(self.pointer),
        )

    def _stream_id(self, stream):
        if stream.device.index != self.rank:
            raise ValueError("Stream device differs from channel")
        return stream.cuda_stream

    def _open(self):
        if self.closed:
            raise RuntimeError("Channel already closed")

    def record(self, stream):
        with self.lock:
            self._open()
            generation = ctypes.c_uint64()
            self.native.call(
                "k3_record",
                self.pointer,
                self._stream_id(stream),
                ctypes.byref(generation),
            )
            self.records += 1
            return Token(self, generation.value)

    def import_token(self, handle):
        if (
            type(handle) is not tuple
            or len(handle) != 5
            or type(handle[0]) is not int
            or type(handle[2]) is not int
            or handle[:4] != (1, self.epoch, self.rank, self.direction)
        ):
            raise ValueError("Counter handle identity mismatch")
        return Token(self, handle[4])

    def wait(self, stream, generation):
        with self.lock:
            self._open()
            self.native.call(
                "k3_wait", self.pointer, self._stream_id(stream), generation
            )
            self.waits += 1

    def value(self):
        with self.lock:
            self._open()
            result = ctypes.c_uint64()
            self.native.call("k3_query", self.pointer, ctypes.byref(result))
            self.queries += 1
            return result.value

    def close_after_drain(self, expected):
        # Caller must have completed the two-sided GPU drain handshake.
        with self.lock:
            self._open()
            self.native.call("k3_close_after_drain", self.pointer, expected)
            self.closed = True
            self.pointer = None

    def stats(self):
        return {
            "records": self.records,
            "waits": self.waits,
            "queries": self.queries,
            "registrations": 1,
            "closed": self.closed,
        }


@dataclass(frozen=True, slots=True)
class Token:
    channel: Channel
    generation: int

    def __post_init__(self):
        if type(self.generation) is not int or not 0 < self.generation <= LIMIT:
            raise ValueError("Invalid generation")
        self.channel._open()

    def ipc_handle(self):
        channel = self.channel
        channel._open()
        return (1, channel.epoch, channel.rank, channel.direction, self.generation)

    def wait(self, stream):
        self.channel.wait(stream, self.generation)

    def query(self):
        return self.channel.value() >= self.generation
