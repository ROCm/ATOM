# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Owned and borrowed prompt buffers, independent of the inference runtime."""

import array
import sys
import threading
from dataclasses import dataclass
from typing import Any


class HostIdentity:
    @staticmethod
    def current():
        try:
            with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as source:
                return source.read().strip()
        except OSError:
            return ""


@dataclass(frozen=True, slots=True)
class BufferDescriptor:
    arena_id: str
    epoch: int
    allocation_id: int
    generation: int
    offset: int
    nbytes: int
    dtype: str = "int32_le"

    @property
    def arena_epoch(self):
        return self.epoch

    @property
    def allocation_generation(self):
        return self.generation

    @property
    def offset_bytes(self):
        return self.offset

    @property
    def token_count(self):
        return self.nbytes // 4


class _PromptBuffer:
    def __init__(self, source=None, view=None, owner=None, on_release=None):
        self.source = source
        self.view = view
        self.owner = owner
        self.descriptor = None
        self.on_release = on_release
        self.refs = 1
        self.lock = threading.RLock()

    def release(self):
        callback = None
        with self.lock:
            self.refs -= 1
            if self.refs == 0:
                if self.view is not None:
                    self.view.release()
                    self.view = None
                self.owner = self.source = None
                callback, self.on_release = self.on_release, None
        if callback is not None:
            callback()


class PromptHandle:
    """A shared source sealed once into a contiguous, read-only int32 buffer.

    Borrowers own independent handles. Exported views retain their buffer owner;
    arena mappings additionally keep the allocation leased until all views die.
    """

    def __init__(self, tokens=(), *, owned=False):
        self._buffer = _PromptBuffer(source=tokens if owned else tuple(tokens))
        self._released = False

    @classmethod
    def from_tokens(cls, tokens, *, owned=False):
        return cls(tokens, owned=owned)

    @classmethod
    def from_buffer(cls, buffer, *, owner=None, on_release=None):
        if sys.byteorder != "little" or array.array("i").itemsize != 4:
            raise RuntimeError("prompt buffers require little-endian int32")
        view = memoryview(buffer)
        if not view.c_contiguous or view.nbytes % 4:
            view.release()
            raise ValueError("prompt buffer must be contiguous and int32 aligned")
        try:
            readonly = view.cast("B").cast("i").toreadonly()
        finally:
            view.release()
        result = cls.__new__(cls)
        result._buffer = _PromptBuffer(
            view=readonly,
            owner=buffer if owner is None else owner,
            on_release=on_release,
        )
        result._released = False
        return result

    def _check(self):
        if self._released:
            raise RuntimeError("prompt handle has been released")

    @property
    def sealed(self):
        self._check()
        return self._buffer.view is not None

    @property
    def descriptor(self):
        self._check()
        return self._buffer.descriptor

    @property
    def buffer_identity(self):
        self._check()
        return id(self._buffer)

    @property
    def token_count(self):
        self._check()
        with self._buffer.lock:
            return (
                len(self._buffer.source)
                if self._buffer.view is None
                else len(self._buffer.view)
            )

    @property
    def nbytes(self):
        return self.token_count * 4

    @property
    def view(self):
        self.seal()
        with self._buffer.lock:
            return memoryview(self._buffer.view)

    def seal(self, arena=None):
        self._check()
        shared = self._buffer
        with shared.lock:
            if shared.view is not None:
                if arena is not None and shared.descriptor is None:
                    raise ValueError("prompt was already sealed for raw transport")
                if arena is not None and shared.descriptor.arena_id != arena.arena_id:
                    raise ValueError("prompt belongs to a different arena")
                return self
            if arena is None:
                tokens = shared.source
                if not (
                    isinstance(tokens, array.array)
                    and tokens.typecode == "i"
                    and tokens.itemsize == 4
                ):
                    source = tokens
                    tokens = array.array("i")
                    for start in range(0, len(source), 16384):
                        tokens.extend(array.array("i", source[start : start + 16384]))
                if sys.byteorder != "little" or tokens.itemsize != 4:
                    raise RuntimeError("prompt buffers require little-endian int32")
                shared.owner = tokens
                shared.view = memoryview(tokens).toreadonly()
            else:
                allocation = arena.reserve(len(shared.source))
                try:
                    allocation.write(shared.source)
                    published = arena.publish(allocation)
                except BaseException:
                    allocation.abort()
                    raise
                sealed = published._buffer
                shared.owner, shared.view = sealed.owner, sealed.view
                shared.descriptor = sealed.descriptor
                shared.on_release = sealed.on_release
                sealed.owner = sealed.view = sealed.on_release = None
                published.release()
            shared.source = None
        return self

    seal_for_targets = seal

    def acquire(self):
        with self._buffer.lock:
            self._check()
            self._buffer.refs += 1
            child = type(self).__new__(type(self))
            child._buffer = self._buffer
            child._released = False
            return child

    borrow = acquire

    def release(self):
        with self._buffer.lock:
            if not self._released:
                self._released = True
                self._buffer.release()

    close = release

    def __len__(self):
        return self.token_count

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        if hasattr(self, "_released"):
            self.release()


ReadOnlyPromptHandle = PromptHandle


class BufferLease:
    """An idempotent send or consumer lease."""

    def __init__(self, release, owner: Any = None):
        self._release = release
        self._owner = owner
        self._lock = threading.Lock()

    def release(self):
        with self._lock:
            callback, self._release = self._release, None
            owner, self._owner = self._owner, None
        if callback is not None:
            try:
                callback()
            finally:
                if isinstance(owner, PromptHandle):
                    owner.release()

    close = release

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
