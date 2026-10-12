# SPDX-License-Identifier: MIT
"""Checkpoint images for DeepSeek-V4.1, in blocks vLLM owns and hashes.

The gap this closes. V4.1 resumes at a boundary only if two things come
back: the paged history, which vLLM's own block table already caches, and
the request's STATE slot -- the sliding-window ring, the compressor rings and
the Engram cursor. On the plugin path the slot lives in a per-request area no
vLLM block covers, so a block-table hit restores the pages and leaves the
slot holding the previous tenant's bytes. That is not an accuracy risk to be
weighed; it is the hard failure `_report_stale_state` raises on, which is why
`cap_local_hit` has been refusing every local hit outright and paying for the
whole prefix over PCIe instead.

The fix is the one native ATOM already uses: at a boundary, copy the slot's
checkpoint image into PAGE units, and on a hit copy it back. What this module
adds is that the units are **vLLM blocks**, so vLLM files them under the
prefix hash, evicts them, and reports a hit only when the proxy blocks and
the image blocks are all cached. Then a local hit is a complete resume and
may be served.

Two things make V4.1 simpler than the DeepSeek-V4 work this follows
(ROCm/ATOM#2503, read as a reference implementation rather than merged):

  * One PAGE unit is exactly one proxy block. V4.1's unit is 739,840 B and so
    is its block at 256 tokens, where V4's image needs 11 blocks of a
    different size. `units_per_checkpoint` therefore sizes the image layers
    directly, with no second geometry to keep in step.
  * vLLM 0.31 carries the scheduling half upstream. `MambaSpec` has
    `mamba_cache_mode="align"`, `num_prefill_checkpoint_blocks` and
    `prefill_checkpoint_alignment`, and `get_mamba_prefill_checkpoint_position`
    picks the boundary. This module does not re-implement any of that; it
    describes the image to it.

The copy itself is native's. `StateCopies` / `execute_paged_state_copies` cut
the image at the byte ranges the native checkpoint uses, through the same
descriptor kernel, so an image written here is the image native would write.
Nothing in this module moves a byte of its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Layer names are positional: unit *i* of the image is layer *i*. The order
#: is the key space, so a renumbering makes every stored image unreadable.
IMAGE_LAYER_PREFIX = "atom.dsv41.state_image"

#: Flipped by the wiring that writes and reads images. Registration alone is
#: not enough to admit a prefix hit: vLLM will allocate and hash the image
#: blocks whether or not anything fills them, and a hit served from unwritten
#: blocks is the stale cursor the refusal exists to prevent. Measured the
#: hard way -- opening the gate on registration alone killed the engine with
#: `needs state at ...` on the second send of one prompt.
_IMAGES_WIRED = False


def images_registered() -> bool:
    """Whether the layers are declared to vLLM."""
    import os

    return os.environ.get("ATOM_V41_STATE_IMAGES") in ("1", "true", "True")


def images_can_serve() -> bool:
    """Whether a prefix hit may be served from an image.

    Registration is necessary and not sufficient. This is what the prefix
    caching gates ask, and it stays False until the store and the restore are
    both installed.
    """
    return images_registered() and _IMAGES_WIRED


def image_layer_name(index: int) -> str:
    return f"{IMAGE_LAYER_PREFIX}.{index}"


def is_image_layer(name: str) -> bool:
    return str(name).startswith(IMAGE_LAYER_PREFIX + ".")


@dataclass(frozen=True)
class V41ImageSizing:
    """How many blocks an image takes, and how big each one is.

    Derived from the native checkpoint spec rather than recomputed, because
    the copier validates against that spec: a sizing worked out here that
    disagrees with it fails at the first store, in the worker, with the
    geometry already allocated.
    """

    unit_bytes: int
    image_bytes: int
    slot_bytes: int
    units: int
    layout_id: str

    @property
    def padded_bytes(self) -> int:
        """What the units hold, which is at least the image.

        The last unit is partly unused whenever the image is not a whole
        number of units. Reported rather than hidden: it is real pool bytes,
        and `check()` is what keeps it from growing unnoticed.
        """
        return self.unit_bytes * self.units

    def check(self) -> None:
        if self.units <= 0:
            raise ValueError("a checkpoint image needs at least one unit")
        if self.image_bytes > self.slot_bytes:
            raise ValueError(
                f"image {self.image_bytes} exceeds the {self.slot_bytes} a slot holds"
            )
        if self.padded_bytes < self.image_bytes:
            raise ValueError(
                f"{self.units} units of {self.unit_bytes} hold "
                f"{self.padded_bytes} B, short of the image's {self.image_bytes}"
            )
        # The slack is what one unit can hold, minus one byte, or the sizing
        # used more units than the image needs.
        if self.padded_bytes - self.image_bytes >= self.unit_bytes:
            raise ValueError(
                f"{self.units} units is one too many for a {self.image_bytes} B "
                f"image in {self.unit_bytes} B units"
            )


def sizing_from_spec(spec) -> V41ImageSizing:
    """Read the sizing off native's own `PagedStateCheckpointSpec`."""
    sizing = V41ImageSizing(
        unit_bytes=int(spec.page_unit_bytes),
        image_bytes=int(spec.image_bytes),
        slot_bytes=int(spec.slot_bytes),
        units=int(spec.units_per_checkpoint),
        layout_id=str(spec.layout_id),
    )
    sizing.check()
    return sizing


def sizing_from_geometry(geometry, image_bytes: int | None = None) -> V41ImageSizing:
    """The same sizing before a spec exists, from the pool geometry.

    The scheduler half needs the unit count to register layers, and that runs
    before the worker builds its checkpoint spec. Both must agree, which
    `assert_agrees_with_spec` is for -- the agreement is checked rather than
    assumed, because the two are computed in different processes from
    different objects.
    """
    unit = int(geometry.paged_bytes)
    slot = int(geometry.state_bytes)
    img = slot if image_bytes is None else int(image_bytes)
    sizing = V41ImageSizing(
        unit_bytes=unit,
        image_bytes=img,
        slot_bytes=slot,
        units=(img + unit - 1) // unit,
        layout_id="",
    )
    sizing.check()
    return sizing


def assert_agrees_with_spec(sizing: V41ImageSizing, spec) -> None:
    """Refuse a disagreement between the two halves, naming the field.

    A mismatch here is a mismatch about what a stored image means, and the
    symptom without this is a copier exception in a worker long after the
    pool was sized -- or, if the sizes happen to be compatible, an image cut
    at the wrong byte ranges and no exception at all.
    """
    theirs = sizing_from_spec(spec)
    for field in ("unit_bytes", "image_bytes", "slot_bytes", "units"):
        mine, other = getattr(sizing, field), getattr(theirs, field)
        if mine != other:
            raise ValueError(
                f"V4.1 state image {field} disagrees: the scheduler half says "
                f"{mine}, the runtime's checkpoint spec says {other}"
            )


# --- the layers vLLM sees ------------------------------------------------
#
# One per image unit. They exist so vLLM allocates, hashes and evicts the
# image blocks beside the proxy's, which is the whole mechanism: a hit is
# reported only when the proxy blocks AND the image blocks are all cached, so
# a prefix that comes back comes back whole.
#
# They are declared as `MambaSpec`, not because V4.1 has a Mamba state, but
# because that is the spec vLLM's scheduler and KV-cache manager already
# treat as "one block per request, checkpointed at a prefill boundary,
# reusable by hash". `num_prefill_checkpoint_blocks` and
# `prefill_checkpoint_alignment` are theirs; `compute_mamba_prefill_checkpoints`
# decides the boundary. Declaring anything else would mean re-implementing
# that, and the scheduler would not know these blocks are checkpointable.


def image_kv_cache_spec(sizing: V41ImageSizing, block_size: int, alignment: int):
    """The spec for one image unit.

    `shapes`/`dtypes` are a flat byte run rather than a tensor shape with
    meaning: the copier addresses the image by byte range through its own
    descriptor kernel and never reads a shape, so giving one here would be a
    second description of the same bytes for nothing to check against.
    """
    import torch
    from vllm.v1.kv_cache_interface import MambaSpec

    return MambaSpec(
        shapes=((sizing.unit_bytes,),),
        dtypes=(torch.uint8,),
        block_size=block_size,
        # One checkpoint block per request: an image is written at a boundary
        # and read at a hit, never accumulated.
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=alignment,
        # The slot is per rank, not replicated: each TP rank holds its own
        # ring and its own compressor tails, and an image is that rank's.
        tp_replicated=False,
    )


def expected_layer_names(sizing: V41ImageSizing) -> list[str]:
    return [image_layer_name(i) for i in range(sizing.units)]


def check_registration(specs: dict, sizing: V41ImageSizing) -> None:
    """Refuse a registration that cannot hold an image, naming what is wrong.

    Three ways it goes wrong quietly, and all three end as bytes read from
    somewhere nothing wrote: too few layers (the tail of every image is
    missing), a layer sized under a unit (every image is truncated), and a
    gap in the numbering (unit *i* is written to layer *i*, so a hole shifts
    every unit after it).
    """
    found = sorted(name for name in specs if is_image_layer(name))
    want = expected_layer_names(sizing)
    if found != sorted(want):
        missing = sorted(set(want) - set(found))
        extra = sorted(set(found) - set(want))
        raise ValueError(
            f"V4.1 state image needs {sizing.units} layers; "
            f"missing={missing or 'none'} unexpected={extra or 'none'}"
        )
    for name in want:
        got = int(specs[name].page_size_bytes)
        if got < sizing.unit_bytes:
            raise ValueError(
                f"{name} holds {got} B, short of the {sizing.unit_bytes} a "
                "PAGE unit needs; every image would be truncated"
            )


# --- turning vLLM block ids into native checkpoint ops ------------------


def _as_unit_ids(block_ids, sizing: V41ImageSizing) -> tuple[int, ...]:
    """The ordered PAGE units an image occupies, from vLLM's block ids.

    One block IS one unit here -- V4.1's block and its PAGE unit are both
    739,840 B -- so this is a rename, and the work is refusing the shapes
    that are not an image. Each of them otherwise reaches the copier, which
    validates and raises in a worker, or worse does not: a duplicated id
    passes every size check and writes two units over one, leaving the other
    holding whatever was there.
    """
    ids = tuple(int(b) for b in block_ids)
    if len(ids) != sizing.units:
        raise ValueError(
            f"a V4.1 image is {sizing.units} units; got {len(ids)} block ids"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"image block ids repeat: {ids}. Two units would address one "
            "block and a third would keep its previous contents"
        )
    if any(i < 0 for i in ids):
        raise ValueError(f"image block ids must be non-negative: {ids}")
    return ids


def store_op(slot: int, block_ids, sizing: V41ImageSizing, layout_id: str):
    """`CheckpointStoreOp` for one request's image, for native's copier.

    Built here rather than in the caller so the store and the restore cannot
    disagree about the unit order: they share `_as_unit_ids`, and the order
    is the image's byte order.
    """
    from atom.model_engine.page_unit_checkpoint import CheckpointStoreOp

    return CheckpointStoreOp(
        src_slot=int(slot),
        unit_ids=_as_unit_ids(block_ids, sizing),
        total_bytes=sizing.image_bytes,
        layout_id=layout_id,
    )


def restore_op(slot: int, block_ids, sizing: V41ImageSizing, layout_id: str):
    from atom.model_engine.page_unit_checkpoint import CheckpointRestoreOp

    return CheckpointRestoreOp(
        dst_slot=int(slot),
        unit_ids=_as_unit_ids(block_ids, sizing),
        total_bytes=sizing.image_bytes,
        layout_id=layout_id,
    )


class V41ImageExporter:
    """Write images at the boundaries vLLM picked. Implements vLLM's ABC.

    vLLM decides WHERE: `MambaPrefillCheckpointBuilder` turns a step into
    per-row checkpoint offsets and block-table columns, declining in lockstep
    with the scheduler and the KV-cache manager -- which matters, because a
    checkpoint block allocated and not written leaves the prefix cache
    serving uninitialized state. This decides nothing about placement; it
    moves the bytes vLLM asked for, through native's copier, so an image
    written here is cut at the ranges a native checkpoint uses.

    `NULL_BLOCK_ID` is how a row says it has no checkpoint this step, and it
    is the common case rather than an error.
    """

    def __init__(self, sizing: V41ImageSizing, layout_id: str, copier):
        self.sizing = sizing
        self.layout_id = layout_id
        self._copier = copier
        self.stored = 0
        self.declined_null = 0

    def plan(self, rows) -> list:
        """`[(slot, [block_ids...])]` -> store ops, skipping the null rows."""
        ops = []
        for slot, block_ids in rows:
            ids = [int(b) for b in block_ids]
            if not ids or any(b <= 0 for b in ids):
                # vLLM's NULL_BLOCK_ID is 0 and a column of -1 is "no
                # checkpoint": either way there is nothing to write, and
                # writing anyway would scribble on block 0.
                self.declined_null += 1
                continue
            ops.append(store_op(slot, ids, self.sizing, self.layout_id))
        return ops

    def export(self, checkpoint, *args, **kwargs) -> None:
        rows = kwargs.get("rows")
        if rows is None:
            raise TypeError(
                "V41ImageExporter.export needs `rows=[(slot, block_ids)]`; "
                "the checkpoint metadata names block-table columns, not slots"
            )
        ops = self.plan(rows)
        if not ops:
            return
        self._copier(ops, [])
        self.stored += len(ops)

    def restore(self, rows) -> int:
        """The other direction, for a hit. Returns how many images were read."""
        ops = []
        for slot, block_ids in rows:
            ids = [int(b) for b in block_ids]
            if not ids or any(b <= 0 for b in ids):
                self.declined_null += 1
                continue
            ops.append(restore_op(slot, ids, self.sizing, self.layout_id))
        if not ops:
            return 0
        self._copier([], ops)
        return len(ops)


# --- registration, and reading back what vLLM did with it ---------------


_IMAGE_LAYER_CLS = None


def image_layer_class():
    """Build the layer class lazily, with vLLM's bases.

    Lazily because this module is imported by tests that never touch vLLM,
    and a real `nn.Module` + `AttentionLayerBase` subclass is required rather
    than optional: vLLM finds layers by `isinstance` over
    `static_forward_context`, so a duck-typed class is registered, ignored,
    and leaves no trace -- the pool is sized without the image blocks and the
    first store writes somewhere nothing reserved.
    """
    global _IMAGE_LAYER_CLS
    if _IMAGE_LAYER_CLS is not None:
        return _IMAGE_LAYER_CLS

    import torch
    from torch import nn
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.attention.backend import AttentionBackend, AttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    class _V41ImageMetadataBuilder(AttentionMetadataBuilder):
        """Builds nothing. The image layers run no attention.

        It exists because vLLM reads class attributes off every group's
        builder -- `requires_block_table_width` first -- and a backend whose
        `get_builder_cls` returns None takes the engine down at startup
        naming the attribute rather than the layer.
        """

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kw):
            self.kv_cache_spec = kv_cache_spec
            self.layer_names = layer_names
            self.vllm_config = vllm_config
            self.device = device

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return None

        def build_for_cudagraph_capture(self, common_attn_metadata):
            return None

    class _V41ImageBackend(AttentionBackend):
        """A backend that answers questions and runs nothing.

        vLLM asks every layer's backend whether it is an SSM, to decide which
        block sizes it may pick for the others
        (`_find_non_ssm_backends`). The image layers are state, not
        attention, so they say so -- and a layer returning None here takes
        the engine down at startup with `'NoneType' object has no attribute
        'is_ssm'`, which names the caller rather than the layer.
        """

        @staticmethod
        def get_name() -> str:
            return "ATOM_DEEPSEEK_V41_STATE_IMAGE"

        @classmethod
        def is_ssm(cls) -> bool:
            return True

        @staticmethod
        def get_impl_cls():
            return nn.Identity

        @staticmethod
        def get_builder_cls():
            return _V41ImageMetadataBuilder

        @classmethod
        def full_cls_name(cls):
            return (cls.__module__, cls.__qualname__)

    class _V41ImageLayer(nn.Module, AttentionLayerBase):
        """One image unit, as vLLM sees it.

        Deliberately inert. vLLM allocates for it and this records what it
        got; whether those are the bytes an image is written to is NOT
        assumed here. `describe_binding` reports the allocation so the
        question can be settled by reading it rather than by modelling it.
        """

        def __init__(self, prefix, part, unit_bytes, block_size, alignment):
            super().__init__()
            self.prefix = prefix
            self.part = int(part)
            self.unit_bytes = int(unit_bytes)
            self.image_block_size = int(block_size)
            self.alignment = alignment
            self.kv_cache = torch.tensor([])
            self.impl = nn.Identity()
            self._atom_v41_image_layer = True

        def get_attn_backend(self):
            return _V41ImageBackend

        def get_kv_cache_spec(self, vllm_config):
            return MambaSpec(
                shapes=((self.unit_bytes,),),
                dtypes=(torch.uint8,),
                block_size=self.image_block_size,
                num_prefill_checkpoint_blocks=1,
                prefill_checkpoint_alignment=self.alignment,
                tp_replicated=False,
            )

        def bind_kv_cache(self, kv_cache) -> None:
            self.kv_cache = (kv_cache.reshape(kv_cache.shape[0], -1),)

        def describe_binding(self) -> dict:
            t = self.kv_cache
            t = t[0] if isinstance(t, tuple) else t
            try:
                blocks = int(t.shape[0])
                return {
                    "layer": self.prefix,
                    "blocks": blocks,
                    "bytes_per_block": int(t.numel() * t.element_size() // max(1, blocks)),
                    "base": int(t.data_ptr()),
                    "device": str(t.device),
                }
            except Exception as exc:
                return {"layer": self.prefix, "error": f"{type(exc).__name__}: {exc}"}

    _IMAGE_LAYER_CLS = _V41ImageLayer
    return _IMAGE_LAYER_CLS


def register_image_layers(vllm_config, sizing: V41ImageSizing, block_size: int,
                          alignment: int) -> list[str]:
    """Put the image layers in vLLM's static forward context. Idempotent."""
    cls = image_layer_class()
    sfc = vllm_config.compilation_config.static_forward_context
    names = expected_layer_names(sizing)
    for i, name in enumerate(names):
        existing = sfc.get(name)
        if existing is not None:
            if not getattr(existing, "_atom_v41_image_layer", False):
                raise ValueError(f"Duplicate layer name: {name}")
            continue
        sfc[name] = cls(name, i, sizing.unit_bytes, block_size, alignment)
    logger.info(
        "DeepSeek-V4.1 state images: %d layers of %d B registered "
        "(image %d B of a %d B slot, %d B padding)",
        len(names), sizing.unit_bytes, sizing.image_bytes, sizing.slot_bytes,
        sizing.padded_bytes - sizing.image_bytes,
    )
    return names


def describe_registration(vllm_config, sizing: V41ImageSizing) -> list[dict]:
    """What vLLM gave each image layer, for reading rather than assuming."""
    sfc = vllm_config.compilation_config.static_forward_context
    out = []
    for name in expected_layer_names(sizing):
        layer = sfc.get(name)
        out.append({"layer": name, "error": "absent"} if layer is None
                   else layer.describe_binding())
    return out
