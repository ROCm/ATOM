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
