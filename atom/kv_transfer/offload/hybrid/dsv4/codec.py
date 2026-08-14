# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 PAGE/SLOT GPU layout and checkpoint persistence codecs.

This module is the single DSV4 data-plane implementation.  It intentionally
keeps three responsibilities in separate public classes while sharing one
file: :class:`DSV4PageSlotCodec` owns GPU geometry and movement,
:class:`DSV4CheckpointCodec` owns AOS1 framing, and
:class:`DSV4CheckpointStore` owns LMCache ``MemoryObj`` lifetimes.

Triton and LMCache are imported lazily so CPU-only format users do not need
either runtime.
"""

from __future__ import annotations

import hashlib
import logging
import operator
import struct
import threading
import zlib
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion

logger = logging.getLogger("atom")


@dataclass(frozen=True)
class _RegionSnapshot:
    """Validated immutable address geometry for one registered region."""

    base_addr: int
    total_bytes: int
    unit_bytes: int
    reverse_indexed: bool
    semantic_role: str

    def unit_addr(self, item_id: int) -> int:
        if not self.reverse_indexed:
            return self.base_addr + item_id * self.unit_bytes
        return self.base_addr + self.total_bytes - (item_id + 1) * self.unit_bytes


@dataclass(frozen=True)
class _RegionSet:
    item_count: int
    regions: tuple[_RegionSnapshot, ...]
    bytes_per_item: int


class DSV4PayloadKind(str, Enum):
    PAGE = "page"
    SLOT = "slot"


@dataclass(frozen=True)
class DSV4PayloadSection:
    """One semantic PAGE or SLOT section in a caller-owned byte buffer."""

    kind: DSV4PayloadKind
    item_ids: tuple[int, ...]
    buffer_offset: int
    nbytes: int


@dataclass(frozen=True)
class DSV4CopyPlan:
    """Typed DSV4 copy transaction; sections execute in tuple order."""

    sections: tuple[DSV4PayloadSection, ...]
    payload_bytes: int
    required_buffer_bytes: int


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    value_type = type(value)
    if isinstance(value, (bool, torch.Tensor)) or (
        value_type.__module__.split(".", 1)[0] == "numpy"
        and value_type.__name__ in {"bool", "bool_"}
    ):
        raise ValueError(f"{name} must be an integer")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if normalized < minimum:
        relation = "> 0" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {relation}, got {normalized}")
    return normalized


def _snapshot_region_set(
    regions: Sequence[KVTransferRegion],
    *,
    kind: DSV4PayloadKind,
    item_count: int,
    semantic_roles: Sequence[str] | None = None,
) -> _RegionSet:
    raw_regions = tuple(regions)
    if not raw_regions:
        raise ValueError(
            f"DSV4PageSlotCodec: at least one {kind.value.upper()} region is required"
        )
    expected_reverse = kind is DSV4PayloadKind.SLOT
    if semantic_roles is not None and len(semantic_roles) != len(raw_regions):
        raise ValueError(
            f"DSV4PageSlotCodec: {kind.value.upper()} semantic role count "
            "must match the region count"
        )
    snapshots: list[_RegionSnapshot] = []
    for region_index, region in enumerate(raw_regions):
        prefix = f"DSV4PageSlotCodec: {kind.value.upper()} region {region_index}"
        base_addr = _integer(f"{prefix} base_addr", region.base_addr, minimum=1)
        unit_bytes = _integer(f"{prefix} unit_bytes", region.unit_bytes, minimum=1)
        total_bytes = _integer(f"{prefix} total_bytes", region.total_bytes, minimum=1)
        reverse = region.reverse_indexed
        if type(reverse) is not bool:
            raise ValueError(f"{prefix} reverse_indexed must be a boolean")
        if reverse != expected_reverse:
            raise ValueError(f"{prefix} must have reverse_indexed={expected_reverse}")
        role = (
            semantic_roles[region_index]
            if semantic_roles is not None
            else getattr(region, "semantic_role", None)
        )
        if role is None:
            if kind is DSV4PayloadKind.SLOT:
                raise ValueError(
                    f"{prefix} semantic_role is required for stable fingerprints"
                )
            role = f"{kind.value}-region-{region_index}"
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{prefix} semantic_role must be a non-empty string")
        required_bytes = item_count * unit_bytes
        if total_bytes < required_bytes:
            raise ValueError(
                f"{prefix} total_bytes is too small; got {total_bytes}, "
                f"need {required_bytes}"
            )
        snapshots.append(
            _RegionSnapshot(
                base_addr=base_addr,
                total_bytes=total_bytes,
                unit_bytes=unit_bytes,
                reverse_indexed=reverse,
                semantic_role=role,
            )
        )
    roles = [region.semantic_role for region in snapshots]
    if len(set(roles)) != len(roles):
        raise ValueError(
            f"DSV4PageSlotCodec: duplicate {kind.value.upper()} semantic roles"
        )
    return _RegionSet(
        item_count=item_count,
        regions=tuple(snapshots),
        bytes_per_item=sum(region.unit_bytes for region in snapshots),
    )


class DSV4PageSlotCodec:
    """One DSV4 raw-byte codec for forward PAGE and reverse SLOT regions.

    The codec owns immutable geometry and lazily compiled region metadata, but
    never owns staging buffers, streams, events, or LMCache objects.
    """

    def __init__(
        self,
        page_regions: Sequence[KVTransferRegion],
        slot_regions: Sequence[KVTransferRegion],
        *,
        num_blocks: int,
        num_slots: int,
        device: torch.device | str,
        page_region_roles: Sequence[str] | None = None,
        slot_region_roles: Sequence[str] | None = None,
    ) -> None:
        self.num_blocks = _integer("num_blocks", num_blocks, minimum=0)
        self.num_slots = _integer("num_slots", num_slots, minimum=0)
        if not page_regions and not slot_regions:
            raise ValueError(
                "DSV4PageSlotCodec requires at least one PAGE or SLOT region"
            )
        if page_regions and self.num_blocks <= 0:
            raise ValueError("num_blocks must be > 0 when PAGE regions are present")
        if slot_regions and self.num_slots <= 0:
            raise ValueError("num_slots must be > 0 when SLOT regions are present")
        self._page = (
            _snapshot_region_set(
                page_regions,
                kind=DSV4PayloadKind.PAGE,
                item_count=self.num_blocks,
                semantic_roles=page_region_roles,
            )
            if page_regions
            else None
        )
        self._slot = (
            _snapshot_region_set(
                slot_regions,
                kind=DSV4PayloadKind.SLOT,
                item_count=self.num_slots,
                semantic_roles=slot_region_roles,
            )
            if slot_regions
            else None
        )
        self.device = torch.device(device)
        if (
            self.device.type == "cuda"
            and self.device.index is None
            and torch.cuda.is_available()
        ):
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._compiled_region_plans: dict[DSV4PayloadKind, object] = {}
        self._compiled_region_plans_lock = threading.Lock()
        self._triton_page_slot = None
        if self.device.type == "cuda":
            try:
                from atom.kv_transfer.offload.hybrid.dsv4 import triton_page_slot

                self._triton_page_slot = triton_page_slot
            except Exception:
                logger.warning(
                    "DSV4PageSlotCodec: Triton PAGE/SLOT staging unavailable",
                    exc_info=True,
                )

    @property
    def page_regions(self) -> tuple[_RegionSnapshot, ...]:
        return () if self._page is None else self._page.regions

    @property
    def slot_regions(self) -> tuple[_RegionSnapshot, ...]:
        return () if self._slot is None else self._slot.regions

    @property
    def page_bytes_per_block(self) -> int:
        return 0 if self._page is None else self._page.bytes_per_item

    @property
    def bytes_per_block(self) -> int:
        """LMCache PAGE width; SLOT bytes are deliberately excluded."""

        return self.page_bytes_per_block

    @property
    def slot_bytes(self) -> int:
        return 0 if self._slot is None else self._slot.bytes_per_item

    @property
    def has_fused_chunk_major_staging(self) -> bool:
        return self._triton_page_slot is not None

    def _require_regions(self, kind: DSV4PayloadKind) -> _RegionSet:
        region_set = self._page if kind is DSV4PayloadKind.PAGE else self._slot
        if region_set is None:
            raise ValueError(f"DSV4PageSlotCodec has no {kind.value.upper()} regions")
        return region_set

    def _normalize_ids(
        self,
        values: Sequence[int],
        *,
        kind: DSV4PayloadKind,
        allow_empty: bool,
    ) -> tuple[int, ...]:
        region_set = self._require_regions(kind)
        item_name = "block id" if kind is DSV4PayloadKind.PAGE else "group id"
        normalized = tuple(_integer(item_name, value) for value in values)
        if not normalized and not allow_empty:
            raise ValueError(f"{kind.value} item ids must not be empty")
        for item_id in normalized:
            if item_id >= region_set.item_count:
                raise ValueError(
                    f"DSV4PageSlotCodec: {item_name} {item_id} outside "
                    f"pool [0, {region_set.item_count})"
                )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "DSV4PageSlotCodec: duplicate "
                f"{'block ids' if kind is DSV4PayloadKind.PAGE else 'group ids'} "
                "are not supported"
            )
        return normalized

    def _flatten_block_ids(
        self, values: Sequence[int] | Sequence[Sequence[int]]
    ) -> tuple[int, ...]:
        raw = list(values)
        if not raw:
            return ()
        try:
            for value in raw:
                operator.index(value)
        except TypeError:
            flattened = []
            for group in raw:
                try:
                    flattened.extend(group)
                except TypeError as exc:
                    raise ValueError(
                        "block_ids must be integers or groups of integers"
                    ) from exc
            return tuple(flattened)
        # page_plan is the single normalization/duplicate-validation owner.
        return tuple(raw)

    @staticmethod
    def _offset(value: object) -> int:
        return _integer("buffer_offset", value)

    def page_plan(
        self,
        block_ids: Sequence[int],
        *,
        buffer_offset: int = 0,
    ) -> DSV4CopyPlan:
        offset = self._offset(buffer_offset)
        ids = self._normalize_ids(
            block_ids,
            kind=DSV4PayloadKind.PAGE,
            allow_empty=True,
        )
        nbytes = len(ids) * self.page_bytes_per_block
        sections = (
            (DSV4PayloadSection(DSV4PayloadKind.PAGE, ids, offset, nbytes),)
            if ids
            else ()
        )
        return DSV4CopyPlan(sections, nbytes, offset + nbytes)

    def slot_plan(
        self,
        group: int,
        *,
        buffer_offset: int = 0,
    ) -> DSV4CopyPlan:
        offset = self._offset(buffer_offset)
        ids = self._normalize_ids(
            (group,),
            kind=DSV4PayloadKind.SLOT,
            allow_empty=False,
        )
        section = DSV4PayloadSection(
            DSV4PayloadKind.SLOT,
            ids,
            offset,
            self.slot_bytes,
        )
        return DSV4CopyPlan((section,), self.slot_bytes, offset + self.slot_bytes)

    def _validate_plan(
        self,
        plan: DSV4CopyPlan,
    ) -> tuple[tuple[DSV4PayloadKind, tuple[int, ...], int], ...]:
        """Revalidate public plan DTOs before any raw-pointer GPU launch."""

        if not isinstance(plan, DSV4CopyPlan):
            raise TypeError("plan must be a DSV4CopyPlan")
        payload_bytes = _integer("plan.payload_bytes", plan.payload_bytes)
        required_buffer_bytes = _integer(
            "plan.required_buffer_bytes", plan.required_buffer_bytes
        )
        expected_payload = 0
        expected_required = 0
        occupied: list[tuple[int, int]] = []
        seen_ids = {
            DSV4PayloadKind.PAGE: set(),
            DSV4PayloadKind.SLOT: set(),
        }
        validated_sections: list[tuple[DSV4PayloadKind, tuple[int, ...], int]] = []
        for section_index, section in enumerate(plan.sections):
            if not isinstance(section, DSV4PayloadSection):
                raise TypeError(
                    f"plan section {section_index} must be a DSV4PayloadSection"
                )
            if not isinstance(section.kind, DSV4PayloadKind):
                raise TypeError(
                    f"plan section {section_index} has invalid payload kind"
                )
            region_set = self._require_regions(section.kind)
            ids = self._normalize_ids(
                section.item_ids,
                kind=section.kind,
                allow_empty=False,
            )
            duplicate_ids = seen_ids[section.kind].intersection(ids)
            if duplicate_ids:
                raise ValueError(
                    f"plan repeats {section.kind.value} item ids across sections"
                )
            seen_ids[section.kind].update(ids)
            offset = _integer(
                f"plan section {section_index} buffer_offset",
                section.buffer_offset,
            )
            nbytes = _integer(
                f"plan section {section_index} nbytes",
                section.nbytes,
                minimum=1,
            )
            expected_nbytes = len(ids) * region_set.bytes_per_item
            if nbytes != expected_nbytes:
                raise ValueError(
                    f"plan section {section_index} nbytes={nbytes} does not match "
                    f"geometry bytes={expected_nbytes}"
                )
            end = offset + nbytes
            if any(
                offset < prior_end and prior_offset < end
                for prior_offset, prior_end in occupied
            ):
                raise ValueError(
                    f"plan section {section_index} overlaps another section"
                )
            occupied.append((offset, end))
            expected_payload += nbytes
            expected_required = max(expected_required, end)
            validated_sections.append((section.kind, ids, offset))

        if payload_bytes != expected_payload:
            raise ValueError(
                f"plan.payload_bytes={payload_bytes} does not match section bytes="
                f"{expected_payload}"
            )
        if plan.sections and required_buffer_bytes != expected_required:
            raise ValueError(
                "plan.required_buffer_bytes="
                f"{required_buffer_bytes} does not match section end={expected_required}"
            )
        return tuple(validated_sections)

    def _matches_device(self, device: torch.device) -> bool:
        candidate = torch.device(device)
        return candidate.type == self.device.type and (
            self.device.index is None or candidate.index == self.device.index
        )

    def _validate_buffer(
        self, buffer: torch.Tensor, plan: DSV4CopyPlan, *, name: str
    ) -> None:
        if not isinstance(buffer, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if buffer.dtype is not torch.uint8:
            raise TypeError(f"{name} must have dtype torch.uint8")
        if not self._matches_device(buffer.device):
            raise TypeError(f"{name} must be on {self.device}, got {buffer.device}")
        if not buffer.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if int(buffer.numel()) < plan.required_buffer_bytes:
            raise ValueError(
                f"{name} is too small; need {plan.required_buffer_bytes} bytes, "
                f"got {int(buffer.numel())}"
            )

    def _region_plan(
        self,
        kind: DSV4PayloadKind,
        *,
        stream: torch.cuda.Stream | None,
    ):
        cached = self._compiled_region_plans.get(kind)
        if cached is not None:
            return cached
        triton_page_slot = self._triton_page_slot
        if triton_page_slot is None:
            raise RuntimeError("DSV4 PAGE/SLOT Triton staging is unavailable")

        # PAGE save workers and SLOT load/save streams share this codec. Build
        # each immutable device plan exactly once so a losing first-use race
        # cannot drop metadata tensors while another stream's Triton kernel is
        # still reading them.
        with self._compiled_region_plans_lock:
            cached = self._compiled_region_plans.get(kind)
            if cached is not None:
                return cached
            region_set = self._require_regions(kind)
            compiled = triton_page_slot.build_region_plan(
                region_set.regions,
                item_count=region_set.item_count,
                device=self.device,
                reverse=kind is DSV4PayloadKind.SLOT,
                stream=stream,
            )
            self._compiled_region_plans[kind] = compiled
            return compiled

    def _copy(
        self,
        buffer: torch.Tensor,
        plan: DSV4CopyPlan,
        *,
        gather: bool,
        stream: torch.cuda.Stream | None,
    ) -> None:
        sections = self._validate_plan(plan)
        self._validate_buffer(buffer, plan, name="dst" if gather else "src")
        if not sections:
            return
        if self.device.type != "cuda":
            raise RuntimeError("DSV4PageSlotCodec GPU movement requires CUDA/HIP")
        triton_page_slot = self._triton_page_slot
        if triton_page_slot is None:
            raise RuntimeError("DSV4 PAGE/SLOT Triton staging is unavailable")
        if stream is not None and not self._matches_device(torch.device(stream.device)):
            raise ValueError("stream device does not match the DSV4 codec device")

        with (
            torch.cuda.device(self.device)
            if self.device.index is not None
            else _NullCtx()
        ):
            for kind, item_ids, buffer_offset in sections:
                region_plan = self._region_plan(kind, stream=stream)
                copy = (
                    triton_page_slot._gather_region_items_unchecked
                    if gather
                    else triton_page_slot._scatter_region_items_unchecked
                )
                copy(
                    region_plan,
                    item_ids,
                    buffer,
                    buffer_offset=buffer_offset,
                    stream=stream,
                )

    def gather(
        self,
        plan: DSV4CopyPlan,
        dst: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._copy(dst, plan, gather=True, stream=stream)

    def scatter(
        self,
        src: torch.Tensor,
        plan: DSV4CopyPlan,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._copy(src, plan, gather=False, stream=stream)

    def gpu_to_chunk_major_device_buffer(
        self,
        device_buf: torch.Tensor,
        block_id_groups: Sequence[int] | Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Gather PAGE blocks through the shared ``BlockByteCodec`` contract."""

        flattened = self._flatten_block_ids(block_id_groups)
        self.gather(self.page_plan(flattened), device_buf, stream=stream)

    def chunk_major_device_buffer_to_gpu(
        self,
        device_buf: torch.Tensor,
        block_id_groups: Sequence[int] | Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        flattened = self._flatten_block_ids(block_id_groups)
        self.scatter(self.page_plan(flattened), device_buf, stream=stream)

    def gather_slot(
        self,
        group: int,
        dst: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.gather(self.slot_plan(group), dst, stream=stream)

    def scatter_slot(
        self,
        src: torch.Tensor,
        group: int,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.scatter(src, self.slot_plan(group), stream=stream)


MAGIC = b"AOS1"
LAYOUT_VERSION = 1
HEADER_BYTES = 128

_FLAGS_NONE = 0
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_STORAGE_HASH_MASK = (1 << 63) - 1
_FINGERPRINT_BYTES = 16
_REQUIRED = object()

# Wire offsets are intentionally stable:
#   magic[0:4], version[4:8], flags[8:12], boundary_tokens[12:20],
#   boundary_block_hash[20:28], payload_bytes[28:36], payload_crc32[36:40],
#   fingerprint[40:56], tp_size[56:60], tp_rank[60:64], reserved[64:128].
_HEADER_PREFIX = struct.Struct("<4sIIQQQI16sII")
_RESERVED_OFFSET = _HEADER_PREFIX.size
assert _RESERVED_OFFSET == 64
assert _RESERVED_OFFSET <= HEADER_BYTES


class DSV4CheckpointError(ValueError):
    """Raised when a sidecar key, header, or payload fails validation."""


def _is_boolean_scalar(value: object) -> bool:
    value_type = type(value)
    return isinstance(value, bool) or (
        value_type.__module__.split(".", 1)[0] == "numpy"
        and value_type.__name__ in {"bool", "bool_"}
    )


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if _is_boolean_scalar(value):
        raise DSV4CheckpointError(f"{name} must be an integer, not a boolean scalar")
    if not isinstance(value, Integral):
        raise DSV4CheckpointError(f"{name} must be an integer")
    normalized = int(value)
    if not minimum <= normalized <= maximum:
        raise DSV4CheckpointError(
            f"{name} must be in [{minimum}, {maximum}], got {normalized}"
        )
    return normalized


def _fingerprint_bytes(value: object, *, name: str = "fingerprint") -> bytes:
    view = _contiguous_byte_view(value, name=name)
    if len(view) != _FINGERPRINT_BYTES:
        raise DSV4CheckpointError(
            f"{name} must be exactly {_FINGERPRINT_BYTES} bytes, got {len(view)}"
        )
    return bytes(view)


def _contiguous_byte_view(value: object, *, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except (TypeError, ValueError) as exc:
        raise DSV4CheckpointError(f"{name} must be bytes-like") from exc
    if not view.c_contiguous:
        raise DSV4CheckpointError(f"{name} must be a contiguous bytes-like object")
    try:
        return view if view.format == "B" and view.ndim == 1 else view.cast("B")
    except TypeError as exc:
        raise DSV4CheckpointError(
            f"{name} must expose a contiguous byte representation"
        ) from exc


def _snapshot_bytes(view: memoryview) -> bytes:
    return bytes(view)


def _stable_entry_byte_view(value: object, *, name: str) -> memoryview:
    view = _contiguous_byte_view(value, name=name)
    if not isinstance(value, bytes):
        view = memoryview(_snapshot_bytes(view))
    return view


def _tp_geometry(
    tp_size: object, tp_rank: object, *, prefix: str = ""
) -> tuple[int, int]:
    size_name = f"{prefix}tp_size"
    rank_name = f"{prefix}tp_rank"
    normalized_size = _require_int(
        size_name,
        tp_size,
        minimum=1,
        maximum=_UINT32_MAX,
    )
    normalized_rank = _require_int(
        rank_name,
        tp_rank,
        minimum=0,
        maximum=_UINT32_MAX,
    )
    if normalized_rank >= normalized_size:
        raise DSV4CheckpointError(
            f"{rank_name} must be smaller than {size_name}; "
            f"got rank={normalized_rank}, size={normalized_size}"
        )
    return normalized_size, normalized_rank


@dataclass(frozen=True)
class DSV4CheckpointKey:
    """Content-addressed identity for one rank's SLOT sidecar."""

    boundary_block_hash: int
    fingerprint: bytes
    tp_size: int
    tp_rank: int

    def __post_init__(self) -> None:
        boundary_block_hash = _require_int(
            "boundary_block_hash",
            self.boundary_block_hash,
            minimum=0,
            maximum=_UINT64_MAX,
        )
        fingerprint = _fingerprint_bytes(self.fingerprint)
        tp_size, tp_rank = _tp_geometry(self.tp_size, self.tp_rank)
        object.__setattr__(self, "boundary_block_hash", boundary_block_hash)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "tp_rank", tp_rank)

    def canonical_string(self) -> str:
        """Return the stable text hashed for LMCache's ``chunk_hash``."""
        return (
            f"atom-slot-v1:{self.tp_size}:{self.tp_rank}:"
            f"{self.boundary_block_hash:016x}:{self.fingerprint.hex()}"
        )

    def storage_hash(self) -> int:
        """Return PR #1683's BLAKE2b-8 digest masked to nonnegative 63 bits."""
        digest = hashlib.blake2b(
            self.canonical_string().encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "little") & _STORAGE_HASH_MASK

    def __str__(self) -> str:
        return self.canonical_string()


@dataclass(frozen=True)
class DSV4CheckpointHeader:
    """Logical AOS1 header.

    ``payload_bytes`` and ``payload_crc32`` may be ``None`` when passed to
    :func:`encode_checkpoint`; encoding always derives both from the payload.
    Decoded headers always contain concrete integer values.

    ``fingerprint`` is an opaque compatibility identifier supplied by the
    caller. This format validates equality but does not infer what it covers.
    """

    boundary_tokens: int
    boundary_block_hash: int
    payload_bytes: int | None
    payload_crc32: int | None
    fingerprint: bytes
    tp_size: int
    tp_rank: int

    def __post_init__(self) -> None:
        boundary_tokens = _require_int(
            "boundary_tokens",
            self.boundary_tokens,
            minimum=1,
            maximum=_UINT64_MAX,
        )
        boundary_block_hash = _require_int(
            "boundary_block_hash",
            self.boundary_block_hash,
            minimum=0,
            maximum=_UINT64_MAX,
        )
        payload_bytes = self.payload_bytes
        if payload_bytes is not None:
            payload_bytes = _require_int(
                "payload_bytes",
                payload_bytes,
                minimum=1,
                maximum=_UINT64_MAX,
            )
        payload_crc32 = self.payload_crc32
        if payload_crc32 is not None:
            payload_crc32 = _require_int(
                "payload_crc32",
                payload_crc32,
                minimum=0,
                maximum=_UINT32_MAX,
            )
        fingerprint = _fingerprint_bytes(self.fingerprint)
        tp_size, tp_rank = _tp_geometry(self.tp_size, self.tp_rank)

        object.__setattr__(self, "boundary_tokens", boundary_tokens)
        object.__setattr__(self, "boundary_block_hash", boundary_block_hash)
        object.__setattr__(self, "payload_bytes", payload_bytes)
        object.__setattr__(self, "payload_crc32", payload_crc32)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "tp_rank", tp_rank)


def _encoded_header(
    header: DSV4CheckpointHeader,
    payload_view: memoryview,
) -> bytes:
    if not isinstance(header, DSV4CheckpointHeader):
        raise DSV4CheckpointError("header must be a DSV4CheckpointHeader")
    if not payload_view:
        raise DSV4CheckpointError("payload must contain at least one byte")

    payload_bytes = len(payload_view)
    payload_crc32 = zlib.crc32(payload_view) & _UINT32_MAX
    if header.payload_bytes is not None and header.payload_bytes != payload_bytes:
        raise DSV4CheckpointError(
            f"header payload_bytes={header.payload_bytes} does not match "
            f"payload size={payload_bytes}"
        )
    if header.payload_crc32 is not None and header.payload_crc32 != payload_crc32:
        raise DSV4CheckpointError(
            f"header payload CRC={header.payload_crc32:#010x} does not match "
            f"computed CRC={payload_crc32:#010x}"
        )

    prefix = _HEADER_PREFIX.pack(
        MAGIC,
        LAYOUT_VERSION,
        _FLAGS_NONE,
        header.boundary_tokens,
        header.boundary_block_hash,
        payload_bytes,
        payload_crc32,
        header.fingerprint,
        header.tp_size,
        header.tp_rank,
    )
    encoded_header = prefix + b"\x00" * (HEADER_BYTES - len(prefix))
    if len(encoded_header) != HEADER_BYTES:
        raise AssertionError("internal AOS1 header size mismatch")
    return encoded_header


def encode_checkpoint(
    header: DSV4CheckpointHeader,
    payload: bytes | bytearray | memoryview,
) -> bytes:
    """Encode one immutable AOS1 byte string, snapshotting mutable input."""

    payload_view = _stable_entry_byte_view(payload, name="payload")
    encoded_header = _encoded_header(header, payload_view)
    return b"".join((encoded_header, payload_view))


def _cpu_uint8_tensor_view(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DSV4CheckpointError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.uint8:
        raise DSV4CheckpointError(f"{name} tensor must have dtype torch.uint8")
    if value.device.type != "cpu":
        raise DSV4CheckpointError(f"{name} tensor must be on the CPU")
    if not value.is_contiguous():
        raise DSV4CheckpointError(f"{name} tensor must be contiguous")
    return value.reshape(-1)


def finalize_checkpoint_tensor_(
    framed: torch.Tensor,
    header: DSV4CheckpointHeader,
) -> torch.Tensor:
    """Compute payload CRC and write only the AOS1 header into ``framed``."""

    flat = _cpu_uint8_tensor_view(framed, name="framed")
    if flat.numel() <= HEADER_BYTES:
        raise DSV4CheckpointError("framed tensor must contain a nonempty payload")
    payload_view = memoryview(flat[HEADER_BYTES:].numpy())
    encoded_header = _encoded_header(header, payload_view)
    flat[:HEADER_BYTES].copy_(
        torch.frombuffer(bytearray(encoded_header), dtype=torch.uint8)
    )
    return framed


def _decode_checkpoint_view(
    blob: bytes | bytearray | memoryview,
    expected_fingerprint: bytes | bytearray | memoryview,
    expected_tp_size: int,
    expected_tp_rank: int,
    *,
    expected_boundary_tokens: object = _REQUIRED,
    expected_boundary_block_hash: object = _REQUIRED,
    expected_payload_bytes: object = _REQUIRED,
    snapshot_payload: bool = False,
) -> tuple[DSV4CheckpointHeader, bytes | memoryview]:
    """Validate and decode one AOS1 sidecar, failing closed on any mismatch.

    Boundary identity and payload size expectations are runtime-mandatory. The
    exact expected size bounds work before checksum work or payload allocation,
    so a corrupt size field cannot amplify memory use.

    CRC32 detects accidental corruption in trusted LMCache storage. It is not
    authentication and does not protect against adversarial tampering.
    """
    blob_view = _contiguous_byte_view(blob, name="blob")
    if len(blob_view) < HEADER_BYTES:
        raise DSV4CheckpointError(
            f"sidecar size {len(blob_view)} is smaller than header size {HEADER_BYTES}"
        )
    header_snapshot = _snapshot_bytes(blob_view[:HEADER_BYTES])

    (
        magic,
        layout_version,
        flags,
        boundary_tokens,
        boundary_block_hash,
        payload_bytes,
        payload_crc32,
        fingerprint,
        tp_size,
        tp_rank,
    ) = _HEADER_PREFIX.unpack_from(header_snapshot)

    if magic != MAGIC:
        raise DSV4CheckpointError(f"bad sidecar magic {magic!r}; expected {MAGIC!r}")
    if layout_version != LAYOUT_VERSION:
        raise DSV4CheckpointError(
            f"bad sidecar layout version {layout_version}; expected {LAYOUT_VERSION}"
        )
    if flags != _FLAGS_NONE:
        raise DSV4CheckpointError(f"unsupported sidecar flags {flags:#x}")
    if any(header_snapshot[_RESERVED_OFFSET:HEADER_BYTES]):
        raise DSV4CheckpointError("sidecar reserved header bytes must all be zero")

    for name, value in (
        ("expected_boundary_tokens", expected_boundary_tokens),
        ("expected_boundary_block_hash", expected_boundary_block_hash),
        ("expected_payload_bytes", expected_payload_bytes),
    ):
        if value is _REQUIRED:
            raise DSV4CheckpointError(f"{name} is required")

    expected_fingerprint = _fingerprint_bytes(
        expected_fingerprint,
        name="expected_fingerprint",
    )
    expected_tp_size, expected_tp_rank = _tp_geometry(
        expected_tp_size,
        expected_tp_rank,
        prefix="expected_",
    )
    expected_boundary_tokens = _require_int(
        "expected_boundary_tokens",
        expected_boundary_tokens,
        minimum=1,
        maximum=_UINT64_MAX,
    )
    expected_boundary_block_hash = _require_int(
        "expected_boundary_block_hash",
        expected_boundary_block_hash,
        minimum=0,
        maximum=_UINT64_MAX,
    )
    expected_payload_bytes = _require_int(
        "expected_payload_bytes",
        expected_payload_bytes,
        minimum=1,
        maximum=_UINT64_MAX,
    )

    header = DSV4CheckpointHeader(
        boundary_tokens=boundary_tokens,
        boundary_block_hash=boundary_block_hash,
        payload_bytes=payload_bytes,
        payload_crc32=payload_crc32,
        fingerprint=fingerprint,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    if boundary_tokens != expected_boundary_tokens:
        raise DSV4CheckpointError(
            "sidecar boundary_tokens mismatch: "
            f"stored={boundary_tokens}, expected={expected_boundary_tokens}"
        )
    if boundary_block_hash != expected_boundary_block_hash:
        raise DSV4CheckpointError(
            "sidecar boundary_block_hash mismatch: "
            f"stored={boundary_block_hash:#018x}, "
            f"expected={expected_boundary_block_hash:#018x}"
        )
    if payload_bytes != expected_payload_bytes:
        raise DSV4CheckpointError(
            "sidecar payload_bytes mismatch: "
            f"stored={payload_bytes}, expected={expected_payload_bytes}"
        )
    if fingerprint != expected_fingerprint:
        raise DSV4CheckpointError("sidecar fingerprint mismatch")
    if tp_size != expected_tp_size or tp_rank != expected_tp_rank:
        raise DSV4CheckpointError(
            "sidecar TP geometry mismatch: "
            f"stored=({tp_size}, {tp_rank}), "
            f"expected=({expected_tp_size}, {expected_tp_rank})"
        )

    expected_total_bytes = HEADER_BYTES + expected_payload_bytes
    if len(blob_view) != expected_total_bytes:
        raise DSV4CheckpointError(
            f"sidecar size {len(blob_view)} does not match expected framed size "
            f"{expected_total_bytes}"
        )

    payload_view = blob_view[HEADER_BYTES:]
    payload = _snapshot_bytes(payload_view) if snapshot_payload else payload_view
    actual_crc32 = zlib.crc32(payload) & _UINT32_MAX
    if actual_crc32 != payload_crc32:
        raise DSV4CheckpointError(
            f"sidecar payload CRC mismatch: stored={payload_crc32:#010x}, "
            f"actual={actual_crc32:#010x}"
        )

    return header, payload


def decode_checkpoint(
    blob: bytes | bytearray | memoryview,
    expected_fingerprint: bytes | bytearray | memoryview,
    expected_tp_size: int,
    expected_tp_rank: int,
    *,
    expected_boundary_tokens: object = _REQUIRED,
    expected_boundary_block_hash: object = _REQUIRED,
    expected_payload_bytes: object = _REQUIRED,
) -> tuple[DSV4CheckpointHeader, bytes]:
    """Validate AOS1 and return an ownership-independent payload snapshot."""

    header, payload = _decode_checkpoint_view(
        blob,
        expected_fingerprint,
        expected_tp_size,
        expected_tp_rank,
        expected_boundary_tokens=expected_boundary_tokens,
        expected_boundary_block_hash=expected_boundary_block_hash,
        expected_payload_bytes=expected_payload_bytes,
        snapshot_payload=True,
    )
    if not isinstance(payload, bytes):
        # An internal invariant, not user-facing type validation.
        raise AssertionError("AOS1 decode did not snapshot payload")  # noqa: TRY004
    return header, payload


def decode_checkpoint_tensor(
    framed: torch.Tensor,
    expected_fingerprint: bytes | bytearray | memoryview,
    expected_tp_size: int,
    expected_tp_rank: int,
    *,
    expected_boundary_tokens: object = _REQUIRED,
    expected_boundary_block_hash: object = _REQUIRED,
    expected_payload_bytes: object = _REQUIRED,
) -> tuple[DSV4CheckpointHeader, torch.Tensor]:
    """Validate a CPU uint8 frame and return its zero-copy payload view."""

    flat = _cpu_uint8_tensor_view(framed, name="framed")
    header, _ = _decode_checkpoint_view(
        memoryview(flat.numpy()),
        expected_fingerprint,
        expected_tp_size,
        expected_tp_rank,
        expected_boundary_tokens=expected_boundary_tokens,
        expected_boundary_block_hash=expected_boundary_block_hash,
        expected_payload_bytes=expected_payload_bytes,
    )
    return header, flat[HEADER_BYTES:]


class DSV4CheckpointCodec:
    """Bind AOS1 framing and validation to one DSV4 TP shard geometry."""

    def __init__(self, *, fingerprint: bytes, tp_size: int, tp_rank: int) -> None:
        self.fingerprint = _fingerprint_bytes(fingerprint)
        self.tp_size, self.tp_rank = _tp_geometry(tp_size, tp_rank)

    def make_key(self, *, boundary_block_hash: int) -> DSV4CheckpointKey:
        return DSV4CheckpointKey(
            boundary_block_hash=boundary_block_hash,
            fingerprint=self.fingerprint,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )

    @staticmethod
    def frame_size(*, payload_bytes: int) -> int:
        payload_bytes = _require_int(
            "payload_bytes", payload_bytes, minimum=1, maximum=_UINT64_MAX
        )
        return HEADER_BYTES + payload_bytes

    def header(
        self,
        *,
        boundary_tokens: int,
        boundary_block_hash: int,
        payload_bytes: int | None = None,
        payload_crc32: int | None = None,
    ) -> DSV4CheckpointHeader:
        return DSV4CheckpointHeader(
            boundary_tokens=boundary_tokens,
            boundary_block_hash=boundary_block_hash,
            payload_bytes=payload_bytes,
            payload_crc32=payload_crc32,
            fingerprint=self.fingerprint,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )

    def finalize_tensor_(
        self,
        framed: torch.Tensor,
        *,
        boundary_tokens: int,
        boundary_block_hash: int,
    ) -> torch.Tensor:
        return finalize_checkpoint_tensor_(
            framed,
            self.header(
                boundary_tokens=boundary_tokens,
                boundary_block_hash=boundary_block_hash,
            ),
        )

    def decode_tensor(
        self,
        framed: torch.Tensor,
        *,
        expected_boundary_tokens: int,
        expected_boundary_block_hash: int,
        expected_payload_bytes: int,
    ) -> tuple[DSV4CheckpointHeader, torch.Tensor]:
        header, payload = decode_checkpoint_tensor(
            framed,
            self.fingerprint,
            self.tp_size,
            self.tp_rank,
            expected_boundary_tokens=expected_boundary_tokens,
            expected_boundary_block_hash=expected_boundary_block_hash,
            expected_payload_bytes=expected_payload_bytes,
        )
        return header, payload

    def encode(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        boundary_tokens: int,
        boundary_block_hash: int,
    ) -> bytes:
        return encode_checkpoint(
            self.header(
                boundary_tokens=boundary_tokens,
                boundary_block_hash=boundary_block_hash,
            ),
            payload,
        )


class DSV4CheckpointCorruptionError(RuntimeError):
    """LMCache returned a malformed DSV4 checkpoint object."""


class DSV4CheckpointStore:
    """Persist rank-local AOS1 bytes through an LMCache StorageManager.

    The adapter deliberately treats the allocation as opaque ``uint8`` bytes;
    ``KV_2LTD`` is only the allocator format accepted by LMCache.
    """

    def __init__(
        self,
        engine,
        *,
        checkpoint_codec: DSV4CheckpointCodec | None = None,
        model_name: str | None = None,
        world_size: int | None = None,
        worker_id: int | None = None,
    ) -> None:
        storage_manager = getattr(engine, "storage_manager", None)
        if storage_manager is None:
            raise ValueError("engine.storage_manager must not be None")

        # LMCache is optional until an actual store is constructed.
        from lmcache.utils import CacheEngineKey
        from lmcache.v1.memory_management import MemoryFormat

        if checkpoint_codec is not None and not isinstance(
            checkpoint_codec, DSV4CheckpointCodec
        ):
            raise TypeError("checkpoint_codec must be a DSV4CheckpointCodec")
        if world_size is None and checkpoint_codec is not None:
            world_size = checkpoint_codec.tp_size
        if worker_id is None and checkpoint_codec is not None:
            worker_id = checkpoint_codec.tp_rank
        if model_name is None:
            model_name = getattr(engine, "model_name", None)
        if model_name is None:
            raise ValueError("model_name must be provided")
        if world_size is None or worker_id is None:
            raise ValueError("world_size and worker_id must be provided")
        normalized_world_size, normalized_worker_id = _tp_geometry(
            world_size,
            worker_id,
            prefix="store_",
        )
        if checkpoint_codec is not None and (
            normalized_world_size != checkpoint_codec.tp_size
            or normalized_worker_id != checkpoint_codec.tp_rank
        ):
            raise ValueError("checkpoint store TP geometry must match checkpoint_codec")

        self._storage_manager = storage_manager
        self._store_location = getattr(engine, "store_location", None)
        retrieve_locations = getattr(engine, "retrieve_locations", None)
        self._retrieve_locations = (
            None if retrieve_locations is None else list(retrieve_locations)
        )
        self._model_name = str(model_name)
        self._world_size = normalized_world_size
        self._worker_id = normalized_worker_id
        self._cache_engine_key_type = CacheEngineKey
        self._memory_format = MemoryFormat.KV_2LTD
        self._corruption_lock = threading.Lock()
        self._unresolved_corrupt_keys: set[DSV4CheckpointKey] = set()

    def put(
        self,
        key: DSV4CheckpointKey,
        framed: torch.Tensor | bytes | bytearray | memoryview,
    ) -> bool:
        checkpoint_key = self._require_key(key)
        payload = self._payload_tensor(framed)
        if not self._prepare_republication(checkpoint_key):
            return False
        cache_key = self._cache_key(checkpoint_key)
        memory_obj = None
        try:
            memory_obj = self._storage_manager.allocate(
                torch.Size((1, 1, payload.numel())),
                torch.uint8,
                fmt=self._memory_format,
                busy_loop=False,
            )
            if memory_obj is None:
                return False
            target = self._memory_tensor(memory_obj)
            if not isinstance(target, torch.Tensor):
                # The storage manager violated its allocation contract.
                raise RuntimeError(  # noqa: TRY004
                    "LMCache allocation did not expose a tensor"
                )
            if target.dtype is not torch.uint8:
                raise RuntimeError("LMCache allocation did not preserve uint8 dtype")
            if target.device.type != "cpu":
                raise RuntimeError("LMCache allocation is not on the CPU")
            if target.numel() != payload.numel():
                raise RuntimeError(
                    "LMCache allocation size does not match the checkpoint payload"
                )
            target.reshape(-1).copy_(payload)
            self._storage_manager.batched_put(
                [cache_key], [memory_obj], location=self._store_location
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LMCache DSV4 checkpoint put failed error_type=%s",
                type(exc).__name__,
            )
            if memory_obj is not None:
                self._ref_count_down(memory_obj)
            return False
        # StorageManager owns the MemoryObj after a successful batched_put.
        return True

    def get(self, key: DSV4CheckpointKey) -> torch.Tensor | None:
        """Return an ownership-independent checkpoint clone."""

        cache_key = self._cache_key(self._require_key(key))
        try:
            location = self._locate(cache_key)
        except RuntimeError:
            return None
        if location is None:
            return None
        try:
            memory_obj = self._storage_manager.get(cache_key, location=location)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LMCache DSV4 checkpoint get failed error_type=%s",
                type(exc).__name__,
            )
            return None
        if memory_obj is None:
            return None
        result = None
        try:
            result = self._validated_checkpoint_tensor(memory_obj).reshape(-1).clone()
        except DSV4CheckpointCorruptionError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LMCache DSV4 checkpoint decode failed error_type=%s",
                type(exc).__name__,
            )
        finally:
            if not self._ref_count_down(memory_obj):
                result = None
        return result

    @contextmanager
    def borrow(self, key: DSV4CheckpointKey):
        """Borrow storage-owned bytes until the caller completes its H2D copy."""

        cache_key = self._cache_key(self._require_key(key))
        location = self._locate(cache_key)
        try:
            memory_obj = (
                None
                if location is None
                else self._storage_manager.get(cache_key, location=location)
            )
        except Exception as exc:
            raise RuntimeError("LMCache DSV4 checkpoint get failed") from exc
        if memory_obj is None:
            yield None
            return
        active_exception = False
        try:
            tensor = self._validated_checkpoint_tensor(memory_obj)
            yield tensor.reshape(-1)
        except BaseException:
            active_exception = True
            raise
        finally:
            if not self._ref_count_down(memory_obj) and not active_exception:
                raise RuntimeError("LMCache DSV4 checkpoint release failed")

    def contains(self, key: DSV4CheckpointKey) -> bool:
        return self._locate(self._cache_key(self._require_key(key))) is not None

    def invalidate(self, key: DSV4CheckpointKey) -> bool:
        checkpoint_key = self._require_key(key)
        with self._corruption_lock:
            return self._invalidate_locked(checkpoint_key)

    def _prepare_republication(self, key: DSV4CheckpointKey) -> bool:
        with self._corruption_lock:
            if key not in self._unresolved_corrupt_keys:
                return True
            return self._invalidate_locked(key)

    def _invalidate_locked(self, key: DSV4CheckpointKey) -> bool:
        self._unresolved_corrupt_keys.add(key)
        cache_key = self._cache_key(key)
        try:
            removed = bool(self._storage_manager.remove(cache_key, locations=None))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LMCache DSV4 checkpoint invalidation failed error_type=%s",
                type(exc).__name__,
            )
            return False
        if removed:
            self._unresolved_corrupt_keys.discard(key)
            return True
        # A concurrent eviction can make remove legitimately return zero. Only
        # keep the corruption fence when any tier still exposes the old key;
        # absence everywhere is already a successful invalidation.
        try:
            still_present = (
                self._storage_manager.contains(
                    cache_key,
                    search_range=None,
                    pin=False,
                )
                is not None
            )
        except Exception:  # noqa: BLE001  # fail closed across storage tiers
            still_present = True
        if not still_present:
            self._unresolved_corrupt_keys.discard(key)
            return True
        logger.warning("LMCache DSV4 checkpoint invalidation removed no stored copy")
        return False

    def _locate(self, cache_key) -> str | None:
        try:
            return self._storage_manager.contains(
                cache_key, search_range=self._retrieve_locations, pin=False
            )
        except Exception as exc:
            raise RuntimeError("LMCache SLOT sidecar visibility probe failed") from exc

    def _cache_key(self, key: DSV4CheckpointKey):
        return self._cache_engine_key_type(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=self._worker_id,
            chunk_hash=key.storage_hash(),
            dtype=torch.uint8,
        )

    @staticmethod
    def _require_key(key: object) -> DSV4CheckpointKey:
        if not isinstance(key, DSV4CheckpointKey):
            raise TypeError("key must be a DSV4CheckpointKey")
        return key

    @staticmethod
    def _payload_tensor(blob: object) -> torch.Tensor:
        if isinstance(blob, torch.Tensor):
            if blob.dtype is not torch.uint8:
                raise ValueError("blob tensor must have dtype torch.uint8")
            if blob.device.type != "cpu":
                raise ValueError("blob tensor must be on the CPU")
            if not blob.is_contiguous():
                raise ValueError("blob tensor must be contiguous")
            if blob.numel() == 0:
                raise ValueError("blob must be nonempty")
            return blob.reshape(-1)
        try:
            view = memoryview(blob)
        except (TypeError, ValueError) as exc:
            raise TypeError("blob must be a torch.Tensor or bytes-like object") from exc
        if not view.c_contiguous:
            raise ValueError("blob bytes-like object must be contiguous")
        try:
            byte_view = (
                view if view.format == "B" and view.ndim == 1 else view.cast("B")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "blob must expose a contiguous byte representation"
            ) from exc
        if not byte_view:
            raise ValueError("blob must be nonempty")
        return torch.frombuffer(bytearray(byte_view), dtype=torch.uint8)

    @staticmethod
    def _memory_tensor(memory_obj):
        tensor = getattr(memory_obj, "tensor", None)
        if tensor is None:
            get_tensor = getattr(memory_obj, "get_tensor", None)
            if callable(get_tensor):
                tensor = get_tensor(0)
        return tensor

    @classmethod
    def _validated_checkpoint_tensor(cls, memory_obj) -> torch.Tensor:
        try:
            tensor = cls._memory_tensor(memory_obj)
        except Exception as exc:
            raise DSV4CheckpointCorruptionError(
                "LMCache checkpoint object did not expose a readable tensor"
            ) from exc
        if not isinstance(tensor, torch.Tensor):
            raise DSV4CheckpointCorruptionError(
                "LMCache checkpoint object did not expose a tensor"
            )
        if tensor.dtype is not torch.uint8:
            raise DSV4CheckpointCorruptionError(
                "LMCache checkpoint object must have dtype torch.uint8"
            )
        if tensor.device.type != "cpu":
            raise DSV4CheckpointCorruptionError(
                "LMCache checkpoint object must be on the CPU"
            )
        if tensor.numel() == 0:
            raise DSV4CheckpointCorruptionError(
                "LMCache checkpoint object must be nonempty"
            )
        if not tensor.is_contiguous():
            raise DSV4CheckpointCorruptionError(
                "LMCache checkpoint object must be contiguous"
            )
        return tensor

    @staticmethod
    def _ref_count_down(memory_obj) -> bool:
        try:
            memory_obj.ref_count_down()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LMCache DSV4 checkpoint release failed: %s", exc)
            return False
        return True


__all__ = [
    "HEADER_BYTES",
    "LAYOUT_VERSION",
    "MAGIC",
    "DSV4CheckpointCodec",
    "DSV4CheckpointCorruptionError",
    "DSV4CheckpointError",
    "DSV4CheckpointHeader",
    "DSV4CheckpointKey",
    "DSV4CheckpointStore",
    "DSV4CopyPlan",
    "DSV4PageSlotCodec",
    "DSV4PayloadKind",
    "DSV4PayloadSection",
    "decode_checkpoint",
    "decode_checkpoint_tensor",
    "encode_checkpoint",
    "finalize_checkpoint_tensor_",
]
