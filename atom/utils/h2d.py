# SPDX-License-Identifier: MIT
"""Checked publication of persistent host buffers on an owner's stream.

The owner supplies its existing reuse event; this module owns no ring. CPU
writes must follow begin() or acquire_write(), before publish(). Tensor aliases
can still write around this protocol: producers must migrate with their buffers.
"""

from __future__ import annotations

import operator
import threading
from functools import wraps

import torch


class PublicationError(RuntimeError):
    """A publication violates ownership or asynchronous source lifetime."""


def h2d_producer(*group_names, runner=None):
    """Declare source groups a method writes, checking before its body runs.

    ``runner`` names the attribute holding the runner, or None for its own
    methods. Resolve the current group on every call so PP slot rotation is
    respected. Standalone producers without a publication registry retain
    their unregistered behavior; a registered runner must provide the group.
    This neither publishes data nor grants permission to republish it.
    """
    get_runner = operator.attrgetter(runner) if runner else lambda instance: instance

    def decorate(produce):
        @wraps(produce)
        def checked(instance, *args, **kwargs):
            groups = getattr(get_runner(instance), "h2d_groups", None)
            if groups is not None:
                for name in group_names:
                    groups[name].check_writable()
            return produce(instance, *args, **kwargs)

        return checked

    return decorate


def _count(value, capacity, name):
    if isinstance(value, bool):
        raise TypeError(f"{name}: count must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name}: count must be an integer") from exc
    if not 0 <= value <= capacity:
        raise ValueError(f"{name}: count {value} is outside [0, {capacity}]")
    return value


def _range(tensor):
    # Reserve the whole span for strided views; overlapping holes cannot be
    # registered as independent buffers. Direct still supports their layout.
    start = tensor.data_ptr()
    if not tensor.numel():
        return start, start
    span = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return start, start + span * tensor.element_size()


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _reason(reason):
    return isinstance(reason, str) and bool(reason.strip())


class PublicationRegistry:
    """Registration-time alias checks shared by a runner's owners and slots."""

    def __init__(self):
        self._bindings = []

    def add(self, binding):
        for prior in self._bindings:
            if prior.device == binding.device and _overlaps(
                prior.destination_range, binding.destination_range
            ):
                raise ValueError(
                    f"{binding.name}: destination overlaps registered {prior.name}"
                )
            if _overlaps(prior.source_range, binding.source_range):
                raise ValueError(
                    f"{binding.name}: source overlaps registered {prior.name}"
                )
        self._bindings.append(binding)


class PublicationOwner:
    """One existing owner/slot, with an epoch shared by all its publish groups.

    finish() seals host preparation, retaining the ledger until the next begin.
    A later host phase must resume the SAME epoch, preserving duplicate checks.
    begin() starts a forward, initialization or capture-preparation epoch;
    an auxiliary group in the same forward must not reset it.
    """

    def __init__(self, device, completion=None, *, registry=None):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type == "cuda" and completion is None:
            raise ValueError("GPU publication requires the owner's reuse event")
        self.completion = completion
        self.registry = registry if registry is not None else PublicationRegistry()
        self.epoch = 0
        self._state = "idle"
        self._stream = None
        self._thread = None
        self._stream_id = None
        # Compare PyTorch stream IDs without constructing a Stream wrapper.
        # Native handles can alias distinct PyTorch stream identities.
        self._get_current_stream = getattr(torch._C, "_cuda_getCurrentStream", None)
        self._bindings = []
        self._groups = []

    def bind(self, buffer, name, *, unit="rows"):
        if self._state != "idle":
            raise PublicationError("bind buffers before starting publication")
        if getattr(buffer, "_publication", None) is not None:
            raise ValueError(f"{name}: buffer is already bound")
        if any(b.name == name for b in self._bindings):
            raise ValueError(f"duplicate logical name: {name}")
        binding = BufferPublication(self, buffer, name, unit)
        self.registry.add(binding)
        self._bindings.append(binding)
        buffer._publication = binding
        binding._single = self.group(name, (binding,))
        return binding

    def group(self, name, members):
        if self._state != "idle":
            raise PublicationError("create groups during initialization")
        members = tuple(members)
        if any(b.owner is not self for b in members):
            raise ValueError("a group must belong to one owner/device")
        if len({id(b) for b in members}) != len(members):
            raise ValueError("a group cannot publish a destination twice")
        group = PublicationGroup(self, name, members)
        self._groups.append(group)
        return group

    def use_packed_transport(self):
        """Configure maximal eligible groups once; covered groups stay direct.

        The runner defines consumer boundaries. This layer chooses storage,
        avoiding duplicate arenas for producer groups covered by a boundary.
        Standalone calls to those groups still use checked direct copies.
        """
        if self._state != "idle":
            raise PublicationError("choose the transport during initialization")
        packed_members = []
        configured = []
        for group in sorted(self._groups, key=lambda g: -len(g.members)):
            members = set(group.members)
            if len(members) < 2 or any(members <= packed for packed in packed_members):
                continue
            group.use_transport("packed")
            if group.transport == "packed":
                packed_members.append(members)
            configured.append(group)
        return configured

    def _current_stream(self):
        if self.device.type != "cuda":
            return None
        if torch.cuda.current_device() != self.device.index:
            raise PublicationError("publication on the wrong device")
        if torch.cuda.is_current_stream_capturing():
            raise PublicationError("publish metadata before actual graph capture")
        return torch.cuda.current_stream(self.device)

    def _check_active(self):
        if self._state != "active":
            raise PublicationError(f"owner is {self._state}; acquire sources first")
        if threading.get_ident() != self._thread:
            raise PublicationError("publish on the owner thread")
        if self.device.type == "cuda":
            if torch.cuda.current_device() != self.device.index:
                raise PublicationError("publication on the wrong device")
            if torch.cuda.is_current_stream_capturing():
                raise PublicationError("publish metadata before actual graph capture")
            if self._get_current_stream is not None:
                same_stream = (
                    self._get_current_stream(self.device.index)[0] == self._stream_id
                )
            else:
                same_stream = torch.cuda.current_stream(self.device) == self._stream
            if not same_stream:
                raise PublicationError("publish on the owner's current compute stream")

    def begin(self):
        """Wait for source reuse BEFORE producer writes; start a new epoch."""
        if self._state not in ("idle", "sealed"):
            raise PublicationError(f"cannot begin: owner is {self._state}")
        stream = self._current_stream()
        if self.completion is not None:
            # The caller records constructor uploads/slot clones before first use.
            try:
                self.completion.synchronize()
            except BaseException:
                self._state = "failed"
                raise
        self.epoch += 1
        self._stream = stream
        self._stream_id = None if stream is None else stream.stream_id
        self._thread = threading.get_ident()
        for binding in self._bindings:
            binding._writable = True
        for group in self._groups:
            group._header_busy = False
        self._state = "active"

    def finish(self):
        """Seal preparation and cover every queued source read."""
        self._check_active()
        try:
            if self.completion is not None:
                self.completion.record(self._stream)
        except BaseException:
            self._state = "failed"
            raise
        self._state = "sealed"

    def resume(self):
        """Reopen a later phase without resetting the forward's ledger."""
        if self._state != "sealed":
            raise PublicationError(f"cannot resume: owner is {self._state}")
        if threading.get_ident() != self._thread:
            raise PublicationError("resume on the original owner thread")
        if self._current_stream() != self._stream:
            raise PublicationError("resume on the original owner stream")
        self._state = "active"
        # Sources stay protected; acquire_write must precede their next write.

    def _wait_sources(self):
        self._check_active()
        try:
            if self.completion is not None:
                self.completion.record(self._stream)
                self.completion.synchronize()
        except BaseException:
            self._state = "failed"
            raise
        for group in self._groups:
            group._header_busy = False

    def fail(self):
        """Retain storage and reject reuse after partially submitted failures."""
        self._state = "failed"

    def drain(self):
        """Drain failed work; the owner remains failed and must be rebuilt."""
        if self._state != "failed":
            raise PublicationError("drain is only for failed publication owners")
        if self._stream is not None:
            self._stream.synchronize()


class BufferPublication:
    def __init__(self, owner, buffer, name, unit):
        self.owner, self.buffer, self.name = owner, buffer, name
        self.source, self.destination = buffer.cpu, buffer.gpu
        self.device = self.destination.device
        if self.device != owner.device or self.source.device.type != "cpu":
            raise ValueError(f"{name}: incorrect source/destination device")
        if (
            self.source.shape != self.destination.shape
            or self.source.dtype != self.destination.dtype
        ):
            raise ValueError(f"{name}: publication cannot reshape or cast")
        if unit not in ("rows", "elements", "bytes"):
            raise ValueError(f"{name}: unknown count unit {unit}")
        if unit != "rows" and not (
            self.source.is_contiguous() and self.destination.is_contiguous()
        ):
            raise ValueError(f"{name}: flat/byte publication must be contiguous")
        self.unit = unit
        self.source_range = _range(self.source)
        self.destination_range = _range(self.destination)
        self.row_capacity = self.source.shape[0] if self.source.ndim else 1
        itemsize = self.source.element_size()
        if unit == "rows":
            self.capacity = self.row_capacity
            self.bytes_per_count = itemsize
            for width in self.source.shape[1:]:
                self.bytes_per_count *= width
            self._source_view, self._destination_view = self.source, self.destination
        else:
            self.capacity = self.source.numel() * (itemsize if unit == "bytes" else 1)
            self.bytes_per_count = 1 if unit == "bytes" else itemsize
            src, dst = self.source.reshape(-1), self.destination.reshape(-1)
            if unit == "bytes":
                src, dst = src.view(torch.uint8), dst.view(torch.uint8)
            self._source_view, self._destination_view = src, dst
        self._epoch = -1
        self._writable = False
        self._single = None
        # At most four private prefix pairs per binding, with FIFO eviction.
        self._prefixes = {}

    def acquire_write(self, *, republish_reason=None):
        """Wait before rewriting a borrowed source, never afterwards."""
        self.owner._check_active()
        if self._epoch == self.owner.epoch and not _reason(republish_reason):
            raise PublicationError(
                f"{self.name}: repeated write needs republish_reason"
            )
        if not self._writable:
            self.owner._wait_sources()
            self._writable = True
        return self.source

    def _validate(self, count, reason):
        if (
            self.buffer.cpu is not self.source
            or self.buffer.gpu is not self.destination
        ):
            raise PublicationError(
                f"{self.name}: binding changed; rebuild the buffer and owner"
            )
        if self._epoch == self.owner.epoch and not _reason(reason):
            raise PublicationError(
                f"{self.name}: repeated publish needs republish_reason"
            )
        if not self._writable:
            raise PublicationError(
                f"{self.name}: acquire_write before modifying source"
            )
        return _count(count, self.capacity, self.name)

    def _copy(self, count):
        if count == self.capacity:
            self._destination_view.copy_(self._source_view, non_blocking=True)
        elif count:
            prefix = self._prefixes.get(count)
            if prefix is None:
                prefix = (self._source_view[:count], self._destination_view[:count])
                if len(self._prefixes) == 4:
                    del self._prefixes[next(iter(self._prefixes))]
                self._prefixes[count] = prefix
            prefix[1].copy_(prefix[0], non_blocking=True)

    def copy_to_gpu(self, n=None, *, republish_reason=None):
        # Legacy n always means rows, including on a flat registered region.
        if n is None:
            count = self.capacity
        else:
            rows = _count(n, self.row_capacity, self.name)
            count = (
                rows
                if self.unit == "rows"
                else (
                    rows * self.capacity // self.row_capacity
                    if self.row_capacity
                    else 0
                )
            )
        self._single.publish((count,), republish_reasons=(republish_reason,))
        return self.destination if n is None else self.destination[:n]


class PublicationGroup:
    """Fixed members validated together before any direct or kernel enqueue."""

    def __init__(self, owner, name, members):
        self.owner, self.name, self.members = owner, name, members
        self._single_member = members[0] if len(members) == 1 else None
        # CPU-only producer scratch; it is never borrowed by the GPU.
        self.counts = [None] * len(members)
        self.indices = {member.name: i for i, member in enumerate(members)}
        self._buffer_indices = {member.buffer: i for i, member in enumerate(members)}
        self._counts = [None] * len(members)
        self._header_busy = False
        self.transport = "direct"
        self.fallback_reason = None
        self._backend = None

    def set_count(self, buffer, count):
        """Record a producer's count without copying data or granting write access.

        This only fills CPU count scratch. The producer must acquire sources
        before writing them; publish validates the entire group before enqueue.
        Lookup by buffer identity keeps binding names private to registration.
        """
        self.counts[self._buffer_indices[buffer]] = count

    def check_writable(self):
        """Check fresh sources before a producer writes any group member.

        The owner's begin() already acquired these regions. Check its common
        thread/device/stream once, then each binding without another GPU API
        query. This does not grant republish permission or wait for a reused
        source; explicit rewrites still use BufferPublication.acquire_write.
        """
        self.owner._check_active()
        for binding in self.members:
            binding._validate(0, None)

    def use_transport(self, transport):
        """Choose direct or packed at initialization, before sources are used."""
        if self.owner._state != "idle":
            raise PublicationError("choose the transport during initialization")
        if transport == "packed":
            if self.owner.device.type != "cuda":
                backend, reason = None, "packing requires GPU destinations"
            else:
                from atom.utils.packed_h2d import PackedCopy

                backend, reason = PackedCopy.create(self.members, self.owner.device)
        elif transport == "direct":
            backend, reason = None, None
        else:
            raise ValueError("H2D transport must be direct or packed")
        self._backend, self.fallback_reason = backend, reason
        self.transport = "packed" if backend is not None else "direct"
        return self.transport

    def publish(self, counts, *, republish_reasons=None):
        self.owner._check_active()
        backend = self._backend
        if len(counts) != len(self.members):
            raise ValueError(f"{self.name}: one count required per member")
        if republish_reasons is not None and len(republish_reasons) != len(counts):
            raise ValueError(f"{self.name}: one reason entry required per member")
        # Legacy wrapper uploads use a fixed single-member group. Preserve its
        # checks, ledger and failure semantics without three general loops.
        if backend is None and self._single_member is not None:
            count = counts[0]
            if count is None:
                self._counts[0] = None
                return
            binding = self._single_member
            count = binding._validate(
                count, None if republish_reasons is None else republish_reasons[0]
            )
            self._counts[0] = count
            binding._epoch = self.owner.epoch
            binding._writable = count == 0
            try:
                if count:
                    binding._copy(count)
            except BaseException:
                self.owner.fail()
                raise
            return
        active_members = 0
        for i, (binding, count) in enumerate(zip(self.members, counts)):
            self._counts[i] = (
                None
                if count is None
                else binding._validate(
                    count, None if republish_reasons is None else republish_reasons[i]
                )
            )
            active_members += self._counts[i] is not None and self._counts[i] > 0
        if backend is not None and active_members > 1 and self._header_busy:
            raise PublicationError(
                "transport counts are in flight; acquire sources first"
            )
        # No GPU-visible descriptor or ledger mutation until the WHOLE group
        # validates. Explicit zero counts count; omitted members do not.
        for binding, count in zip(self.members, self._counts):
            if count is not None:
                binding._epoch = self.owner.epoch
                binding._writable = count == 0
        try:
            if backend is not None and active_members:
                # Direct fallback leaves any earlier packed read in flight.
                self._header_busy |= backend.submit(self._counts)
            elif backend is None:
                for binding, count in zip(self.members, self._counts):
                    if count:
                        binding._copy(count)
        except BaseException:
            self.owner.fail()
            raise
