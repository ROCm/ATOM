# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded tmpfs arena with generation checks and explicit consumer leases."""

import array
import hashlib
import mmap
import os
import re
import secrets
import struct
import threading
import uuid
import weakref
from dataclasses import dataclass, field

from .buffer_types import BufferDescriptor, BufferLease, PromptHandle

_HEADER_BYTES = 64
_ARENA_HEADER = struct.Struct("<8sQQ")
_ALLOCATION_HEADER = struct.Struct("<8sQQQQQ")
_ARENA_MAGIC = b"ATOMA001"
_ALLOCATION_MAGIC = b"ATOMP001"


class ArenaCapacityError(BufferError):
    pass


class _PromptMapping(mmap.mmap):
    """A separate mapping pins a slice even when derived views outlive handles."""


@dataclass
class _Allocation:
    descriptor: BufferDescriptor
    extent: int
    published: bool = False
    frontends: set = field(default_factory=set)
    consumers: set = field(default_factory=set)
    send_pins: set = field(default_factory=set)
    writer: object = None


class WritableAllocation:
    def __init__(self, owner, descriptor, mapping, delta):
        self.owner = owner
        self.descriptor = descriptor
        self._mapping = mapping
        self._delta = delta
        self._finished = False

    @property
    def view(self):
        if self._finished:
            raise RuntimeError("allocation is no longer writable")
        return memoryview(self._mapping)[
            self._delta : self._delta + self.descriptor.nbytes
        ].cast("i")

    def write(self, tokens):
        if len(tokens) != self.descriptor.token_count:
            raise ValueError("token count does not match allocation")
        destination = self.view
        source = None
        try:
            try:
                source = memoryview(tokens)
            except TypeError:
                pass
            if (
                source is not None
                and source.c_contiguous
                and source.itemsize == 4
                and source.format in ("i", "<i", "=i")
            ):
                destination.cast("B")[:] = source.cast("B")
            else:
                # Bounded conversion avoids a second complete packed prompt.
                for start in range(0, len(tokens), 16384):
                    chunk = array.array("i", tokens[start : start + 16384])
                    destination[start : start + len(chunk)] = chunk
        finally:
            if source is not None:
                source.release()
            destination.release()

    def abort(self):
        self.owner.abort(self)


class PromptArenaOwner:
    @staticmethod
    def capacity_for_requests(max_requests: int, max_tokens: int) -> int:
        if max_requests <= 0 or max_tokens <= 0:
            raise ValueError("request count and context length must be positive")
        prompt_bytes = max_tokens * 4
        extent = _HEADER_BYTES + max(_HEADER_BYTES, (prompt_bytes + 63) // 64 * 64)
        # The startup probe occupies one small allocation for the pool's life.
        return 2 * _HEADER_BYTES + max_requests * extent

    def __init__(self, root_dir="/dev/shm", capacity_bytes=256 * 1024 * 1024):
        if not isinstance(capacity_bytes, int) or capacity_bytes < _HEADER_BYTES:
            raise ValueError("arena capacity must be at least 64 bytes")
        self.root_dir = os.path.abspath(os.fspath(root_dir))
        self.capacity_bytes = capacity_bytes
        self.arena_id = uuid.uuid4().hex
        self.epoch = secrets.randbits(64) or 1
        self._directory = os.path.join(self.root_dir, "atom-ipc-" + self.arena_id)
        os.mkdir(self._directory, 0o700)
        self._path = os.path.join(self._directory, "arena.bin")
        self._fd = -1
        self._mapping = None
        try:
            self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            total_bytes = capacity_bytes + _HEADER_BYTES
            os.ftruncate(self._fd, total_bytes)
            # Reserve backing now: exhaustion must fail admission, not SIGBUS.
            os.posix_fallocate(self._fd, 0, total_bytes)
            self._mapping = mmap.mmap(self._fd, total_bytes, access=mmap.ACCESS_WRITE)
            _ARENA_HEADER.pack_into(
                self._mapping, 0, _ARENA_MAGIC, self.epoch, capacity_bytes
            )
        except BaseException:
            if self._mapping is not None:
                self._mapping.close()
            if self._fd != -1:
                os.close(self._fd)
            if os.path.exists(self._path):
                os.unlink(self._path)
            os.rmdir(self._directory)
            raise
        self._lock = threading.RLock()
        self._allocations = {}
        self._generations = {}
        self._free = [(_HEADER_BYTES, capacity_bytes)]
        self._consumer_allocations = {}
        self._lease_id = 0
        self._probe = None
        self._closing = False
        self._closed = False

    @property
    def registration(self):
        return {
            "root_dir": self.root_dir,
            "arena_id": self.arena_id,
            "epoch": self.epoch,
            "capacity_bytes": self.capacity_bytes,
        }

    @property
    def used_bytes(self):
        with self._lock:
            return sum(allocation.extent for allocation in self._allocations.values())

    @property
    def allocation_count(self):
        with self._lock:
            return len(self._allocations)

    def _check_open(self):
        if self._closed or self._closing:
            raise RuntimeError("prompt arena is closing or closed")

    @staticmethod
    def _key(descriptor):
        return descriptor.epoch, descriptor.allocation_id, descriptor.generation

    def _record(self, descriptor):
        if not isinstance(descriptor, BufferDescriptor):
            raise TypeError("expected BufferDescriptor")
        record = self._allocations.get(self._key(descriptor))
        if record is None or record.descriptor != descriptor:
            raise ValueError("unknown or stale arena allocation")
        return record

    def _new_lease(self):
        self._lease_id += 1
        return self._lease_id

    def _map_slice(self, descriptor, *, writable=False):
        start = descriptor.offset - descriptor.offset % mmap.ALLOCATIONGRANULARITY
        delta = descriptor.offset - start
        mapping = _PromptMapping(
            self._fd,
            max(1, delta + descriptor.nbytes),
            access=mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ,
            offset=start,
        )
        return mapping, delta

    def reserve(self, token_count):
        if not isinstance(token_count, int) or token_count < 0:
            raise ValueError("token_count must be a nonnegative integer")
        nbytes = token_count * 4
        extent = _HEADER_BYTES + max(_HEADER_BYTES, (nbytes + 63) // 64 * 64)
        with self._lock:
            self._check_open()
            for index, (offset, available) in enumerate(self._free):
                if available >= extent:
                    break
            else:
                raise ArenaCapacityError("shared prompt arena capacity exhausted")
            generation = self._generations.get(offset, 0) + 1
            self._generations[offset] = generation
            descriptor = BufferDescriptor(
                self.arena_id,
                self.epoch,
                offset,
                generation,
                offset + _HEADER_BYTES,
                nbytes,
            )
            mapping, delta = self._map_slice(descriptor, writable=True)
            writer = WritableAllocation(self, descriptor, mapping, delta)
            if available == extent:
                self._free.pop(index)
            else:
                self._free[index] = offset + extent, available - extent
            record = _Allocation(descriptor, extent, writer=writer)
            self._allocations[self._key(descriptor)] = record
            self._write_header(record)
            return writer

    def _write_header(self, record):
        descriptor = record.descriptor
        _ALLOCATION_HEADER.pack_into(
            self._mapping,
            descriptor.allocation_id,
            _ALLOCATION_MAGIC,
            descriptor.epoch,
            descriptor.allocation_id,
            descriptor.generation,
            descriptor.nbytes,
            int(record.published),
        )

    def publish(self, allocation):
        with self._lock:
            self._check_open()
            record = self._record(allocation.descriptor)
            if allocation.owner is not self or allocation._finished:
                raise ValueError("allocation is not owned by this writer")
            # mmap.close refuses publication while writable views remain exported.
            allocation._mapping.close()
            allocation._finished = True
            record.writer = None
            try:
                mapping, delta = self._map_slice(record.descriptor)
            except BaseException:
                self._reclaim(record)
                raise
            lease_id = self._new_lease()
            record.frontends.add(lease_id)
            weakref.finalize(
                mapping, self._drop_reference, record.descriptor, "frontends", lease_id
            )
            view = memoryview(mapping)[delta : delta + record.descriptor.nbytes]
            try:
                handle = PromptHandle.from_buffer(view, owner=mapping)
            finally:
                view.release()
            handle._buffer.descriptor = record.descriptor
            record.published = True
            self._write_header(record)
            return handle

    def abort(self, allocation):
        with self._lock:
            if allocation._finished:
                return
            record = self._record(allocation.descriptor)
            if allocation.owner is not self:
                raise ValueError("allocation belongs to another arena")
            allocation._mapping.close()
            allocation._finished = True
            record.writer = None
            self._reclaim(record)
            self._finish_close()

    def acquire_consumer(self, handle, consumer_key):
        descriptor = handle.descriptor
        hash(consumer_key)
        with self._lock:
            self._check_open()
            record = self._record(descriptor)
            if not record.published:
                raise ValueError("allocation has not been published")
            record.consumers.add(consumer_key)
            self._consumer_allocations.setdefault(consumer_key, set()).add(
                self._key(descriptor)
            )
        return BufferLease(lambda: self.release_consumer(consumer_key, descriptor))

    def release_consumer(self, consumer_key, descriptor=None):
        with self._lock:
            keys = self._consumer_allocations.get(consumer_key)
            if not keys:
                return False
            selected = (
                set(keys) if descriptor is None else {self._key(descriptor)} & keys
            )
            released = False
            for key in selected:
                record = self._allocations.get(key)
                if record is None or (
                    descriptor is not None and descriptor != record.descriptor
                ):
                    continue
                record.consumers.discard(consumer_key)
                keys.discard(key)
                self._maybe_reclaim(record)
                released = True
            if not keys:
                self._consumer_allocations.pop(consumer_key, None)
            return released

    def acquire_send_pin(self, handle):
        descriptor = handle.descriptor
        child = handle.acquire()
        if descriptor is None:
            return BufferLease(lambda: None, child)
        try:
            with self._lock:
                self._check_open()
                record = self._record(descriptor)
                lease_id = self._new_lease()
                record.send_pins.add(lease_id)
        except BaseException:
            child.release()
            raise
        return BufferLease(
            lambda: self._drop_reference(descriptor, "send_pins", lease_id), child
        )

    def release_frontend(self, handle):
        handle.release()

    def _drop_reference(self, descriptor, collection, lease_id):
        with self._lock:
            if self._closed:
                return
            record = self._allocations.get(self._key(descriptor))
            if record is not None and record.descriptor == descriptor:
                getattr(record, collection).discard(lease_id)
                self._maybe_reclaim(record)

    def _maybe_reclaim(self, record):
        if (
            record.writer is None
            and not record.frontends
            and not record.consumers
            and not record.send_pins
        ):
            self._reclaim(record)
        self._finish_close()

    def _reclaim(self, record):
        descriptor = record.descriptor
        self._allocations.pop(self._key(descriptor), None)
        self._mapping[descriptor.allocation_id : descriptor.allocation_id + 8] = (
            b"\0" * 8
        )
        self._free.append((descriptor.allocation_id, record.extent))
        self._free.sort()
        merged = []
        for offset, extent in self._free:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                previous, size = merged[-1]
                merged[-1] = previous, size + extent
            else:
                merged.append((offset, extent))
        self._free = merged

    def reclaim_dead_engine(self, engine_instance_id):
        """Call only after the exact process instance is confirmed stopped."""
        with self._lock:
            keys = [
                key
                for key in self._consumer_allocations
                if (
                    getattr(key, "engine_instance", None) == engine_instance_id
                    or isinstance(key, tuple)
                    and len(key) > 1
                    and key[1] == engine_instance_id
                )
            ]
            for key in keys:
                self.release_consumer(key)

    def create_probe(self):
        with self._lock:
            if self._probe is None:
                contents = array.array("i")
                contents.frombytes(secrets.token_bytes(32))
                self._probe = PromptHandle.from_tokens(contents, owned=True).seal(self)
            view = self._probe.view
            try:
                return self._probe.descriptor, hashlib.sha256(view).digest()
            finally:
                view.release()

    def release_probe(self):
        with self._lock:
            if self._probe is not None:
                self._probe.release()
                self._probe = None

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closing = True
            self.release_probe()
            if not self._finish_close():
                raise BufferError("arena still has writers, consumers, or send pins")

    def _finish_close(self):
        if not self._closing:
            return False
        if self._closed:
            return True
        if any(
            record.writer is not None or record.consumers or record.send_pins
            for record in self._allocations.values()
        ):
            return False
        self._closed = True
        self._mapping.close()
        os.close(self._fd)
        os.unlink(self._path)
        os.rmdir(self._directory)
        return True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PromptArenaReader:
    def __init__(self, registration, on_release=None):
        self.root_dir = os.path.abspath(os.fspath(registration["root_dir"]))
        self.arena_id = registration["arena_id"]
        self.epoch = registration["epoch"]
        self.capacity_bytes = registration["capacity_bytes"]
        if not isinstance(self.arena_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}", self.arena_id
        ):
            raise ValueError("invalid registered arena ID")
        if not isinstance(self.epoch, int) or not 0 < self.epoch < 2**64:
            raise ValueError("invalid registered arena epoch")
        if (
            not isinstance(self.capacity_bytes, int)
            or self.capacity_bytes < _HEADER_BYTES
        ):
            raise ValueError("invalid registered arena capacity")
        directory = os.path.join(self.root_dir, "atom-ipc-" + self.arena_id)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self._fd = os.open(
                "arena.bin", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
            )
        finally:
            os.close(directory_fd)
        try:
            if os.fstat(self._fd).st_size != self.capacity_bytes + _HEADER_BYTES:
                raise ValueError("registered arena size does not match backing file")
            header = os.pread(self._fd, _HEADER_BYTES, 0)
            if _ARENA_HEADER.unpack_from(header) != (
                _ARENA_MAGIC,
                self.epoch,
                self.capacity_bytes,
            ):
                raise ValueError("registered arena header mismatch")
        except BaseException:
            os.close(self._fd)
            raise
        self._lock = threading.RLock()
        self._mappings = {}
        self._consumers = {}
        self._pending_releases = []
        self._on_release = on_release
        self._closed = False

    def _validate(self, descriptor):
        if not isinstance(descriptor, BufferDescriptor):
            raise TypeError("expected BufferDescriptor")
        if descriptor.arena_id != self.arena_id or descriptor.epoch != self.epoch:
            raise ValueError("buffer refers to an unregistered arena or epoch")
        for value in (
            descriptor.allocation_id,
            descriptor.generation,
            descriptor.offset,
            descriptor.nbytes,
        ):
            if not isinstance(value, int) or not 0 <= value < 2**64:
                raise ValueError("buffer descriptor integer is out of range")
        if (
            descriptor.dtype != "int32_le"
            or descriptor.generation == 0
            or descriptor.allocation_id < _HEADER_BYTES
            or descriptor.allocation_id % _HEADER_BYTES
            or descriptor.offset != descriptor.allocation_id + _HEADER_BYTES
            or descriptor.offset % 4
            or descriptor.nbytes % 4
            or descriptor.offset + descriptor.nbytes
            > self.capacity_bytes + _HEADER_BYTES
        ):
            raise ValueError("invalid shared prompt layout")
        header = os.pread(self._fd, _HEADER_BYTES, descriptor.allocation_id)
        if len(header) != _HEADER_BYTES or _ALLOCATION_HEADER.unpack_from(header) != (
            _ALLOCATION_MAGIC,
            descriptor.epoch,
            descriptor.allocation_id,
            descriptor.generation,
            descriptor.nbytes,
            1,
        ):
            raise ValueError(
                "allocation is unpublished, stale, or has invalid metadata"
            )

    def _attach(self, descriptor):
        if self._closed:
            raise RuntimeError("prompt arena reader is closed")
        self._validate(descriptor)
        start = descriptor.offset - descriptor.offset % mmap.ALLOCATIONGRANULARITY
        delta = descriptor.offset - start
        mapping = _PromptMapping(
            self._fd,
            max(1, delta + descriptor.nbytes),
            access=mmap.ACCESS_READ,
            offset=start,
        )
        try:
            self._validate(descriptor)
        except BaseException:
            mapping.close()
            raise
        return mapping, delta

    def acquire(self, descriptor, consumer_key, on_release=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("prompt arena reader is closed")
            self._validate(descriptor)
            previous = self._consumers.get(consumer_key)
            if previous is not None and previous[0] != descriptor:
                raise ValueError("consumer key already refers to another allocation")
            entry = self._mappings.get(descriptor)
            mapping = None if entry is None else entry[0]()
            if mapping is None:
                mapping, delta = self._attach(descriptor)
                entry = (weakref.ref(mapping), delta, set())
                self._mappings[descriptor] = entry
                weakref.finalize(mapping, self._mapping_released, descriptor, entry[2])
            else:
                delta = entry[1]
            if previous is None:
                previous = (
                    descriptor,
                    weakref.WeakSet(),
                    on_release or self._on_release,
                )
                self._consumers[consumer_key] = previous
            entry[2].add(consumer_key)
            view = memoryview(mapping)[delta : delta + descriptor.nbytes]
            try:
                handle = PromptHandle.from_buffer(view, owner=mapping)
            finally:
                view.release()
            handle._buffer.descriptor = descriptor
            previous[1].add(handle)
            return handle

    def _mapping_released(self, descriptor, consumer_keys):
        callbacks = []
        with self._lock:
            entry = self._mappings.get(descriptor)
            if entry is not None and entry[2] is consumer_keys:
                self._mappings.pop(descriptor, None)
            for key in consumer_keys:
                consumer = self._consumers.pop(key, None)
                if consumer is not None:
                    if consumer[2] is None:
                        self._pending_releases.append((descriptor, key))
                    else:
                        callbacks.append((consumer[2], key))
        for callback, key in callbacks:
            callback(descriptor, key)

    def release(self, consumer_key):
        with self._lock:
            consumer = self._consumers.get(consumer_key)
            handles = () if consumer is None else list(consumer[1])
        for handle in handles:
            handle.release()
        with self._lock:
            for index, event in enumerate(self._pending_releases):
                if event[1] == consumer_key:
                    return self._pending_releases.pop(index)
        return None

    def take_releases(self):
        with self._lock:
            events, self._pending_releases = self._pending_releases, []
            return events

    def probe(self, descriptor):
        with self._lock:
            mapping, delta = self._attach(descriptor)
        try:
            view = memoryview(mapping)[delta : delta + descriptor.nbytes]
            try:
                return hashlib.sha256(view).digest()
            finally:
                view.release()
        finally:
            mapping.close()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
