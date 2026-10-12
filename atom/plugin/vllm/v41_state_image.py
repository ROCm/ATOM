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

#: False because the store and the restore never execute: `wired=4` on four
#: ranks, and `log_stats` -- called on the same line that records this step's
#: slots -- emitted nothing across a whole run. So
#: `_v41_images_around_forward` is not reaching a runtime or a `prepared`,
#: and every fix made downstream of it so far has been made on a path that
#: does not run. That is the next thing to find, and until it is found
#: flipping this admits prefix hits that resume from whatever the slot held.
_RESTORE_IMPLEMENTED = False


def images_registered() -> bool:
    """Whether the layers are declared to vLLM."""
    import os

    return os.environ.get("ATOM_V41_STATE_IMAGES") in ("1", "true", "True")


def images_can_serve() -> bool:
    """Whether a prefix hit may be served from an image.

    Registration, and only registration -- because this is asked at CONFIG
    time, before any pool is bound, and whether a runtime got installed is a
    fact that does not exist yet. An earlier version asked for the wiring too
    and therefore always said no.

    What stops a hit being served from blocks nothing wrote is not this
    predicate: it is that installing the runtime at bind time is MANDATORY
    once the layers are registered. If it cannot be built the engine fails to
    start, which is the same protection moved to a place that can actually
    see the answer. `_IMAGES_WIRED` stays as an observable, not as a gate.

    `_RESTORE_IMPLEMENTED` is False, so this is False, and a configuration
    that turns images on is refused at startup rather than dying mid-run.
    The store side works -- the byte round trip is covered -- but the restore
    reads the wrong column: `MambaPrefillCheckpointBuilder` reports where a
    checkpoint is WRITTEN, and a resuming request reads a different column of
    the block table. Measured as `needs state at 2304, found 3` with the
    gate open, which is what this constant now prevents.
    """
    return images_registered() and _RESTORE_IMPLEMENTED


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


def _as_block_id(block_ids, sizing: V41ImageSizing) -> int:
    """The ONE vLLM block an image occupies, from what the caller was given.

    Measured, not assumed: the eight image layers carry identical specs, so
    vLLM collects them into ONE kv cache group -- the startup line reads
    `kv cache group sizes [256, 256]`, the proxy's and theirs. A single block
    id therefore names one block in EACH of the eight layers,
    8 x 739,840 B, which is exactly one image.

    An earlier version of this file read the id list as eight separate blocks
    and would have scattered one image across eight unrelated prefixes. The
    shape is checked here because nothing downstream distinguishes the two
    readings: eight ids and one id are both "a list of ints".
    """
    ids = [int(b) for b in block_ids] if not isinstance(block_ids, int) else [int(block_ids)]
    if len(ids) != 1:
        raise ValueError(
            f"a V4.1 image is one vLLM block across {sizing.units} layers; "
            f"got {len(ids)} block ids. Eight ids was the earlier, wrong "
            "reading of the same list."
        )
    # A negative id is not an error: vLLM writes -1 into a checkpoint column
    # that has none this step, and NULL_BLOCK_ID is 0. Both mean "nothing to
    # copy" and the caller declines them. Only the SHAPE is an error here,
    # because a wrong shape is a wrong image and a null row is no image.
    return ids[0]


class V41ImageCopier:
    """Move an image between a STATE slot and the image layers' blocks.

    Native's `StateCopies` cannot be used unchanged: it addresses the
    destination as `proxy_base + unit * unit_bytes`, and the image blocks are
    not in the proxy arena -- vLLM allocated a tensor per image layer. What
    IS reused is everything below that: `plan_segmented_copy` intersects the
    two byte streams, `write_descriptor` fills the spans and
    `launch_copy_descriptor` runs the same kernel a native checkpoint runs.
    Only the destination addresses differ, which is the whole of the
    difference between the two.
    """

    def __init__(self, sizing: V41ImageSizing, state_bytes, layer_tensors,
                 max_batch: int = 64):
        from atom.model_ops.attentions.pool_layout.paged_state_copy import (
            DescriptorStaging,
            plan_segmented_copy,
        )

        if len(layer_tensors) != sizing.units:
            raise ValueError(
                f"{len(layer_tensors)} image layers bound, {sizing.units} needed"
            )
        self.sizing = sizing
        self._state = state_bytes
        self._layers = list(layer_tensors)
        for i, t in enumerate(self._layers):
            per_block = t.numel() * t.element_size() // max(1, t.shape[0])
            if per_block < sizing.unit_bytes:
                raise ValueError(
                    f"image layer {i} holds {per_block} B per block, short of "
                    f"the {sizing.unit_bytes} a unit needs"
                )
        self.plan = plan_segmented_copy(
            [sizing.slot_bytes],
            [sizing.unit_bytes] * sizing.units,
            sizing.image_bytes,
        )
        self.staging = DescriptorStaging(
            2 * max(1, max_batch) * self.plan.num_spans, self._state.device
        )
        self.staging.reserve([0])
        self.stored = 0
        self.restored = 0

    def _slot_base(self, slot: int) -> int:
        if not 0 <= slot < self._state.shape[0]:
            raise IndexError(f"STATE slot {slot} is out of range")
        return int(self._state[slot].data_ptr())

    def _image_bases(self, block_id: int) -> list[int]:
        out = []
        for t in self._layers:
            if not 0 <= block_id < t.shape[0]:
                raise IndexError(
                    f"image block {block_id} is outside the {t.shape[0]} "
                    "blocks vLLM allocated for an image layer"
                )
            out.append(int(t[block_id].data_ptr()))
        return out

    def _run_on_host(self, stores, restores) -> None:
        for ops, storing in ((stores, True), (restores, False)):
            for slot, block in ops:
                if not 0 <= slot < self._state.shape[0]:
                    raise IndexError(f"STATE slot {slot} is out of range")
                entry = self._state[slot]
                at = 0
                for t in self._layers:
                    if not 0 <= block < t.shape[0]:
                        raise IndexError(
                            f"image block {block} is outside the {t.shape[0]} "
                            "blocks vLLM allocated for an image layer"
                        )
                    take = min(self.sizing.unit_bytes, entry.numel() - at)
                    if take <= 0:
                        break
                    state_span, unit_span = entry[at : at + take], t[block][:take]
                    (unit_span if storing else state_span).copy_(
                        state_span if storing else unit_span
                    )
                    at += take
        self.stored += len(stores)
        self.restored += len(restores)

    def run(self, stores, restores, descriptor_slot: int = 0) -> None:
        """`stores`/`restores` are `[(slot, block_id)]`."""
        import numpy as np
        from atom.model_ops.attentions.pool_layout.paged_state_copy import (
            launch_copy_descriptor,
        )

        if not stores and not restores:
            return
        if not self._state.is_cuda:
            # The descriptor kernel is Triton and refuses host pointers.
            # Native's own copier carries the same fallback, and it is worth
            # carrying here for the same reason: it is what lets the byte
            # round trip -- the only check that distinguishes a working
            # checkpoint from one moving plausible garbage -- run without a
            # GPU.
            self._run_on_host(stores, restores)
            return
        rows = self.staging.rows(descriptor_slot)
        total = (len(stores) + len(restores)) * self.plan.num_spans
        if total > rows.shape[0]:
            raise ValueError("image copy batch exceeds descriptor capacity")
        at = 0
        for ops, storing in ((stores, True), (restores, False)):
            if not ops:
                continue
            end = at + len(ops) * self.plan.num_spans
            slot_bases = np.asarray(
                [[self._slot_base(slot)] for slot, _ in ops], dtype=np.int64
            )
            image_bases = np.asarray(
                [self._image_bases(b) for _, b in ops], dtype=np.int64
            )
            self.plan.write_descriptor(
                rows[at:end], slot_bases, image_bases, forward=storing
            )
            at = end
        descriptor = self.staging.upload(descriptor_slot, total)
        cut = len(stores) * self.plan.num_spans
        # A restore may read an image this same batch stored, so the two
        # launches are ordered rather than merged.
        launch_copy_descriptor(descriptor[:cut], self.plan)
        launch_copy_descriptor(descriptor[cut:], self.plan)
        self.stored += len(stores)
        self.restored += len(restores)


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

    def __init__(self, sizing: V41ImageSizing, copier):
        self.sizing = sizing
        self._copier = copier
        self.stored = 0
        self.restored = 0
        self.declined_null = 0

    def plan(self, rows) -> list:
        """`[(slot, block_id)]` -> pairs to copy, skipping the null rows."""
        ops = []
        for slot, block_id in rows:
            b = _as_block_id(block_id, self.sizing)
            if b <= 0:
                # vLLM's NULL_BLOCK_ID is 0 and a column of -1 is "no
                # checkpoint this step": the common case. Writing anyway
                # would scribble on block 0, which belongs to somebody.
                self.declined_null += 1
                continue
            ops.append((int(slot), b))
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
        self._copier.run(ops, [])
        self.stored += len(ops)

    def restore(self, rows) -> int:
        """The other direction, for a hit. Returns how many images were read."""
        ops = self.plan(rows)
        if not ops:
            return 0
        self._copier.run([], ops)
        self.restored += len(ops)
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

    class _V41ImageBackend(AttentionBackend):
        """A backend that answers questions and runs nothing.

        vLLM asks every layer's backend whether it is an SSM, to decide which
        block sizes the others may pick (`_find_non_ssm_backends`). The image
        layers are state, not attention, so they say so -- and returning None
        takes the engine down naming the caller, `'NoneType' object has no
        attribute 'is_ssm'`, rather than the layer.
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
            self._ckpt = None

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            # vLLM decides which rows checkpoint, in lockstep with the
            # scheduler; this only reads the answer out of THIS group's
            # block table, which is the half the proxy's builder cannot see.
            try:
                self._publish(common_attn_metadata)
            except Exception:
                # A step without images is worse than a step without this
                # bookkeeping, and the counters say when it happened.
                logger.warning("V4.1 image marks failed for a step", exc_info=True)
                _MARK_FAILURES.append(1)
            return None

        def _publish(self, m):
            """Publish both columns: the one to read, and the one to write.

            They are NOT the same column, which is the mistake this replaces.
            vLLM's `MambaPrefillCheckpointBuilder` reports where a checkpoint
            is WRITTEN this step. What a resuming request READS is
            `mamba_get_block_table_tensor`'s rule for align mode:
            `start = (seq_len - 1) // block_size`, the last state block of
            what the request already has. Using the write column for both
            resumed requests from a block nothing had filled -- measured as
            `needs state at 2304, found 3`.
            """
            from vllm.model_executor.layers.mamba.checkpoint import (
                MambaPrefillCheckpointBuilder,
            )

            if m is None:
                return
            step = _step_id(m)

            # --- the read column, for a restore ---
            read_blocks = self._read_columns(m)
            if read_blocks is not None:
                offer_restore_half(step, blocks=read_blocks)

            # --- the write column, for the store after the forward ---
            if self._ckpt is None:
                self._ckpt = MambaPrefillCheckpointBuilder(
                    self.vllm_config, self.kv_cache_spec
                )
            meta = self._ckpt.build(m, list(range(int(m.num_reqs))))
            _LAST_STORE_MARKS.clear()
            if meta is not None:
                _LAST_STORE_MARKS.extend(
                    int(b) for b in meta.state_indices.tolist()
                )

        def _read_columns(self, m):
            """The block each row would resume its state from, or None.

            Taken through vLLM's own rule rather than re-derived: a column
            picked differently here than the one the scheduler filled is a
            restore from somebody else's state, and nothing downstream
            compares the two.
            """
            import torch
            from vllm.v1.attention.backends.utils import (
                mamba_get_block_table_tensor,
            )

            table = getattr(m, "block_table_tensor", None)
            seq_lens = getattr(m, "seq_lens", None)
            if table is None or seq_lens is None:
                return None
            cols = mamba_get_block_table_tensor(
                table, seq_lens, self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )
            if cols.dim() == 2:
                cols = cols[:, 0]
            return [int(b) for b in cols.tolist()]

    class _V41ImageLayer(nn.Module, AttentionLayerBase):
        """One image unit, as vLLM sees it.

        Deliberately inert. vLLM allocates for it and this records what it
        got; whether those are the bytes an image is written to is not
        assumed -- `describe_binding` reports the allocation so the question
        is settled by reading it.
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
                    "bytes_per_block": int(
                        t.numel() * t.element_size() // max(1, blocks)
                    ),
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


# --- wiring: where a store and a restore actually happen ----------------


class V41ImageRuntime:
    """The worker-side half: a copier, an exporter, and vLLM's own builder.

    Built once the proxy pool is bound, because until then there is no state
    arena to copy out of and vLLM has not handed the image layers their
    tensors. `install` is what flips `_IMAGES_WIRED`, so the prefix-caching
    gates open on the wiring existing rather than on the layers being
    declared -- registration alone served hits from blocks nothing had
    written, which killed the engine.
    """

    def __init__(self, sizing: V41ImageSizing, copier, builder=None):
        self.sizing = sizing
        self.copier = copier
        self.exporter = V41ImageExporter(sizing, copier)
        self.vllm_builder = builder
        self.restored = 0
        self.stored = 0
        self.restore_declined = 0

    # -- restore, before the forward -------------------------------------
    def restore_for(self, rows) -> int:
        """`[(slot, block_id)]` for rows resuming at a hit boundary."""
        n = self.exporter.restore(rows)
        self.restored += n
        return n

    # -- store, after the forward ----------------------------------------
    def store_for(self, rows) -> int:
        before = self.exporter.stored
        self.exporter.export(None, rows=rows)
        n = self.exporter.stored - before
        self.stored += n
        return n

    def stats(self) -> dict:
        d = {
            "image_stored": self.stored,
            "image_restored": self.restored,
            "image_declined_null": self.exporter.declined_null,
        }
        d.update(rendezvous_stats())
        d["image_mark_failures"] = mark_failures()
        return d

    def log_stats(self, interval_s: float = 20.0) -> None:
        """Say what the two directions did, including when they did nothing.

        Unconditional and periodic for the reason every counter on this
        branch is: a mechanism that stored exactly zero images otherwise
        produces exactly zero lines saying so, and a by-construction null
        reads as a quiet success.
        """
        import time

        now = time.monotonic()
        if now - getattr(self, "_last_log", 0.0) < interval_s:
            return
        self._last_log = now
        logger.info(
            "V4.1 state images: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(self.stats().items())),
        )


_RUNTIME = None


def runtime():
    return _RUNTIME


def install_runtime(rt: V41ImageRuntime) -> None:
    """Make the runtime current and open the prefix-caching gates.

    The flag and the runtime are set together on purpose: they are the same
    claim -- that something will fill the image blocks -- and splitting them
    is how the gates once opened over blocks nothing wrote.
    """
    global _RUNTIME, _IMAGES_WIRED
    _RUNTIME = rt
    _IMAGES_WIRED = True
    logger.info(
        "DeepSeek-V4.1 state images wired: %d units of %d B, prefix hits may "
        "now be served from an image",
        rt.sizing.units,
        rt.sizing.unit_bytes,
    )


def build_runtime(vllm_config, cache, state_bytes, max_batch: int = 64):
    """Assemble the runtime from the bound pool. None when images are off."""
    if not images_registered():
        return None
    sizing = sizing_from_geometry(cache.geometry)
    sfc = vllm_config.compilation_config.static_forward_context
    tensors = []
    for name in expected_layer_names(sizing):
        layer = sfc.get(name)
        if layer is None:
            raise ValueError(
                f"image layer {name} is missing at bind time; the pool was "
                "sized without it and a store would write unreserved bytes"
            )
        t = layer.kv_cache
        t = t[0] if isinstance(t, tuple) else t
        if getattr(t, "numel", None) is None or t.numel() == 0:
            raise ValueError(
                f"image layer {name} has no tensor bound; vLLM allocates "
                "these, so an empty one means the layer was not in the "
                "kv-cache config it built"
            )
        tensors.append(t)
    copier = V41ImageCopier(sizing, state_bytes, tensors, max_batch=max_batch)
    return V41ImageRuntime(sizing, copier)


# --- the two halves of a step meet here ---------------------------------
#
# A restore needs a STATE slot and an image block id, and they are built in
# different places: the slot by the proxy's metadata builder, which stages
# the batch, and the block by the image group's, which holds that group's
# block table. vLLM calls builders per group and does not promise an order,
# so neither can assume it runs second.
#
# So both publish, and whoever completes the pair performs the restore. The
# alternative -- picking one and hoping -- fails by doing nothing, which is a
# prefix served from an unrestored slot: the silent half of the failure this
# whole mechanism exists to prevent.


class _StepRendezvous:
    def __init__(self):
        self.slots = None
        self.blocks = None
        self.step = -1
        self.completed = 0
        self.half_steps = 0

    def begin(self, step: int) -> None:
        if step != self.step:
            if self.slots is not None or self.blocks is not None:
                # One half arrived last step and the other never did.
                self.half_steps += 1
            self.slots = self.blocks = None
            self.step = step

    def offer(self, step: int, *, slots=None, blocks=None):
        self.begin(step)
        if slots is not None:
            self.slots = list(slots)
        if blocks is not None:
            self.blocks = list(blocks)
        if self.slots is None or self.blocks is None:
            return None
        n = min(len(self.slots), len(self.blocks))
        rows = [(int(self.slots[i]), int(self.blocks[i])) for i in range(n)]
        self.slots = self.blocks = None
        self.completed += 1
        return rows


_RENDEZVOUS = _StepRendezvous()
#: Blocks vLLM marked this step, for the store that follows the forward.
_LAST_STORE_MARKS: list = []
_MARK_FAILURES: list = []


def _step_id(m) -> int:
    """Something that changes every step and nothing else. The query-start
    tensor is rebuilt per step, so its identity serves without adding a
    counter two builders would have to agree on."""
    return id(getattr(m, "query_start_loc", m))


def take_store_marks() -> list:
    marks = list(_LAST_STORE_MARKS)
    _LAST_STORE_MARKS.clear()
    return marks


def mark_failures() -> int:
    return len(_MARK_FAILURES)


def rendezvous():
    return _RENDEZVOUS


def offer_restore_half(step: int, *, slots=None, blocks=None) -> int:
    """Publish one half; perform the restore when both are in. Returns rows done."""
    rt = runtime()
    rows = _RENDEZVOUS.offer(step, slots=slots, blocks=blocks)
    if rows is None or rt is None:
        return 0
    return rt.restore_for(rows)


def rendezvous_stats() -> dict:
    return {
        "image_steps_paired": _RENDEZVOUS.completed,
        # Non-zero means one half of a step never arrived. Not fatal, and not
        # harmless: that step's hit resumed from whatever the slot held.
        "image_steps_half": _RENDEZVOUS.half_steps,
    }
