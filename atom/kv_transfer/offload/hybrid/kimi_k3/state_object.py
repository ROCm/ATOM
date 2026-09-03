# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One state checkpoint, one opaque object, keyed by ATOM's own hash.

LMCache's token-chunk database is bypassed: state bytes cannot be sliced by
token (there is no "first three chunks hit"), so chunking would produce N keys
useful only together, at the cost of LMCache's chunk-alignment loss.
"""

from __future__ import annotations

import logging
import struct
import zlib
from typing import Any

import torch

logger = logging.getLogger("atom")

MAGIC = b"K3S1"
LAYOUT_VERSION = 1
HEADER_BYTES = 32

_FLAGS_NONE = 0
_UINT32_MAX = (1 << 32) - 1

# Same framing as ``dsv4/codec.py``'s AOS1 sidecar, minus the fields the key
# already carries. Wire offsets are stable:
#   magic[0:4], version[4:8], flags[8:12], payload_bytes[12:20],
#   payload_crc32[20:24], reserved[24:32].
_HEADER_PREFIX = struct.Struct("<4sIIQI")
_RESERVED_OFFSET = _HEADER_PREFIX.size
assert _RESERVED_OFFSET == 24
assert _RESERVED_OFFSET <= HEADER_BYTES


def _memory_object_as_uint8(memory_obj: Any, nbytes: int) -> torch.Tensor:
    """The leading ``nbytes`` of an LMCache MemoryObj as a flat uint8 view."""
    tensor = getattr(memory_obj, "tensor", None)
    if tensor is None and hasattr(memory_obj, "get_tensor"):
        tensor = memory_obj.get_tensor(0)
    if tensor is None:
        raise RuntimeError("K3 state codec: invalid MemoryObj tensor")
    if tensor.dtype != torch.uint8:
        raise TypeError(f"K3 state codec: MemoryObj must be uint8, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise RuntimeError("K3 state codec: MemoryObj tensor not contiguous")
    flat = tensor.reshape(-1)
    if int(flat.numel()) < int(nbytes):
        raise ValueError(
            f"K3 state codec: MemoryObj holds {int(flat.numel())} bytes, "
            f"needs {int(nbytes)}"
        )
    return flat[: int(nbytes)]


def _payload_view(blob: torch.Tensor) -> memoryview | None:
    """Zero-copy byte view of the payload, or None when it is not host memory."""
    if blob.device.type != "cpu":
        return None
    return memoryview(blob[HEADER_BYTES:].numpy())


def _encode_header(payload: memoryview) -> bytes:
    prefix = _HEADER_PREFIX.pack(
        MAGIC,
        LAYOUT_VERSION,
        _FLAGS_NONE,
        len(payload),
        zlib.crc32(payload) & _UINT32_MAX,
    )
    return prefix + b"\x00" * (HEADER_BYTES - len(prefix))


def _header_failure(blob: torch.Tensor, payload: memoryview) -> str | None:
    """Reason this blob is not a valid image, or None when it checks out.

    Integrity only: versioning and TP skew already live in the key, so this is
    here for a truncated or torn write and for an xxh64 collision that happened
    to land on the right size.
    """
    head = bytes(blob[:HEADER_BYTES].numpy())
    magic, version, flags, payload_bytes, payload_crc32 = _HEADER_PREFIX.unpack_from(
        head
    )
    if magic != MAGIC:
        return f"bad magic {magic!r}"
    if version != LAYOUT_VERSION:
        return f"bad layout version {version}"
    if flags != _FLAGS_NONE:
        return f"unsupported flags {flags:#x}"
    if any(head[_RESERVED_OFFSET:HEADER_BYTES]):
        return "reserved header bytes are not zero"
    # Size before checksum: a corrupt length field must not drive any work
    # proportional to it.
    if payload_bytes != len(payload):
        return f"payload_bytes={payload_bytes}, blob carries {len(payload)}"
    actual = zlib.crc32(payload) & _UINT32_MAX
    if actual != payload_crc32:
        return f"CRC mismatch: stored={payload_crc32:#010x}, actual={actual:#010x}"
    return None


class StateByteCodec:
    """Pack/unpack one entry's state and move it through ``storage_manager``.

    An opaque flat uint8 blob: the x-packed / strided / multi-plane state
    layouts cannot be expressed in LMCache's token-major model at all.

    The directions read different things by design -- a store gathers the
    checkpoint's PAGE units, a load scatters into the Active Slot the resuming
    forward reads -- but the payload is the same ordered byte stream either way.
    """

    def __init__(
        self,
        backend,
        staged,
        entry_bytes: int,
        *,
        model_name: str,
        world_size: int,
        worker_id: int,
        layout_id: str,
    ) -> None:
        self._backend = backend
        self._staged = staged
        self.entry_bytes = int(entry_bytes)
        if self.entry_bytes <= 0:
            raise ValueError("state entry bytes must be > 0")
        if not isinstance(layout_id, str) or not layout_id:
            raise ValueError("a state entry key needs a non-empty layout id")
        self._model_name = model_name
        self._world_size = int(world_size)
        self._worker_id = int(worker_id)
        self._layout_id = layout_id
        self._misfit_reads = 0
        self._corrupt_reads = 0
        self._storage = None
        # Never hard-code a size: V4 keeps six compressor fields across
        # n_csa/n_hca layers plus an optional window; GDN keeps
        # 2 * num_gdn_attn_state * (1 + num_spec) slots. MB-scale, not KB.
        logger.info(
            "state offload: entry_bytes=%d (%.2f MiB) + %d header bytes per request",
            self.entry_bytes,
            self.entry_bytes / (1 << 20),
            HEADER_BYTES,
        )

    @property
    def object_bytes(self) -> int:
        """Bytes of one stored object: header plus payload."""
        return HEADER_BYTES + self.entry_bytes

    @property
    def misfit_reads(self) -> int:
        return self._misfit_reads

    @property
    def corrupt_reads(self) -> int:
        return self._corrupt_reads

    def bind_storage_manager(self, storage_manager) -> None:
        self._storage = storage_manager

    def key(self, h: int):
        """ATOM's hash, bound to the geometry the bytes were written under.

        One prefix hash maps to a different image under a different ``num_spec``,
        TP size or conv/ssm dtype, and the same size in a different order reads
        back silently wrong; ``layout_id`` names all of that. Folding it in also
        makes the KV and state key spaces disjoint, which matters because they
        share one pool and the key has no field saying what an entry IS.

        xxh64, not ``hash((h, layout_id))``: Python salts ``hash`` of a str per
        process, so a restart would silently orphan every prior entry.
        """
        import xxhash
        from lmcache.utils import CacheEngineKey

        digest = xxhash.xxh64()
        # Unsigned, and it must be: an ATOM block hash spans the full
        # 0..2**64-1, and signed=True raises OverflowError on about half of them.
        digest.update(int(h).to_bytes(8, "little", signed=False))
        digest.update(self._layout_id.encode())
        return CacheEngineKey(
            self._model_name,
            self._world_size,
            self._worker_id,
            digest.intdigest(),
            torch.uint8,
        )

    def put(self, h: int, unit_ids, on_source_released=None) -> bool:
        """Store one checkpoint image. False when nothing was stored.

        A refusal is not an error: allocation returns None under CPU pressure,
        and a whole image is refused sooner than a KV chunk.

        ``on_source_released`` fires once ``pack`` has synchronized the gather,
        i.e. once the GPU has stopped reading the units. It is separate from the
        return value on purpose: the units belong to the KV pool, and holding
        them across ``batched_put`` would keep an image out of the pool for a CPU
        operation that cannot touch them.
        """
        if self._storage is None:
            return False
        # Computed before the allocation: ``key`` can raise, and a raise from
        # inside the ``batched_put`` argument list would strand the MemoryObj.
        key = self.key(h)
        obj = self._allocate(self.object_bytes)
        if obj is None:
            return False
        # ``batched_put`` discharges the reference it is handed, but only in its
        # terminal tail loop, which runs after every step that can raise. So on
        # ANY exception below we still own the reference and down it exactly
        # once -- and never on the success path, where the tail loop already did.
        # Do not mark the object handed off before the call: a pre-adoption raise
        # then strands one entry-sized allocation at ref_count=1 and shrinks the
        # shared CPU pool by one entry per failure.
        try:
            blob = _memory_object_as_uint8(obj, self.object_bytes)
            payload = _payload_view(blob)
            if payload is None:
                raise RuntimeError(
                    "K3 state codec: storage handed back a non-CPU MemoryObj; "
                    "the header cannot be framed on device memory"
                )
            self._staged.pack(
                self._backend.page_unit_views(unit_ids), blob[HEADER_BYTES:]
            )
            # Source first: ``pack`` has synchronized the stream that reads the
            # units, so nothing on the device touches them from here.
            if on_source_released is not None:
                on_source_released()
            blob[:HEADER_BYTES].copy_(
                torch.frombuffer(bytearray(_encode_header(payload)), dtype=torch.uint8)
            )
            self._storage.batched_put([key], [obj])
        except Exception:
            obj.ref_count_down()
            raise
        return True

    def get(self, h: int, slot: int) -> bool:
        """Load one image back into Active Slot ``slot``. False on a miss.

        The reference must be discharged here: the fetch does a ``ref_count_up()``
        for the caller, and without the matching down LRU drops the block from
        the index but never returns it to the pinned allocator -- an entry-sized
        leak per hit. ``finally``, so a throwing unpack does not leak either.
        """
        if self._storage is None:
            return False
        obj = self._storage.get(self.key(h))
        if obj is None:
            return False
        try:
            size = self._object_bytes(obj)
            if size != self.object_bytes:
                # Unreachable while ``layout_id`` is in the key, which is the
                # point: a wrong-size hit means two things collided in the shared
                # pool, and unpacking would write another entry's bytes over live
                # state. An unmeasurable object counts here too -- skipping the
                # check for it would be the same unpack with less evidence.
                self._misfit_reads += 1
                logger.warning(
                    "state offload: hash %d came back %s bytes, expected %d; "
                    "treating as a miss (misfit_reads=%d)",
                    h,
                    size,
                    self.object_bytes,
                    self._misfit_reads,
                )
                return False
            blob = _memory_object_as_uint8(obj, self.object_bytes)
            payload = _payload_view(blob)
            reason = (
                "MemoryObj is not host memory"
                if payload is None
                else _header_failure(blob, payload)
            )
            if reason is not None:
                # Never raise into the transfer path: the caller disowns the
                # hash and recomputes, which is always correct.
                self._corrupt_reads += 1
                logger.warning(
                    "state offload: hash %d failed header validation (%s); "
                    "treating as a miss (corrupt_reads=%d)",
                    h,
                    reason,
                    self._corrupt_reads,
                )
                return False
            self._staged.unpack(
                blob[HEADER_BYTES:], self._backend.state_entry_views(slot)
            )
        finally:
            obj.ref_count_down()
        return True

    @staticmethod
    def _object_bytes(obj) -> int | None:
        """Bytes in a MemoryObj, or None when it will not say.

        None means "cannot measure", never "size 0", and the caller degrades it
        to a miss. Neither accessor is wrapped: an allocator that raises when
        asked its own object's size is broken in a way this must not paper over.
        """
        get_size = getattr(obj, "get_size", None)
        if callable(get_size):
            size = get_size()
            if size is not None:
                return int(size)
        tensor = getattr(obj, "tensor", None)
        if tensor is not None:
            return int(tensor.numel()) * tensor.element_size()
        return None

    def _allocate(self, nbytes: int) -> Any:
        from lmcache.v1.memory_management import MemoryFormat

        # ``fmt`` is inert (shape/dtype force a flat blob); passed only because
        # MixedMemoryAllocator.allocate rejects anything outside its tensor
        # formats. busy_loop=False because this is a *store*: LMCache warns
        # busy_loop "should only be used for retrieve" (concurrent stores
        # deadlock) yet defaults it True, under which a full pool spins forever
        # instead of returning the None this caller handles.
        return self._storage.allocate(
            torch.Size([nbytes]),
            torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
            busy_loop=False,
        )
