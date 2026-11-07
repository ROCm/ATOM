# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The block interface: what P2 streams build and P3 composes (frozen at the end of P1).

A **block** is one reusable piece of a layer (a linear, a norm, an attention
mixer). It is plain Python on the plan side and produces ``Stage``s:

    class MyBlock(Block):
        def stages(self, sc: Scenario) -> list[Stage]: ...   # tasks, regions written, args, emit
        def pack(self, layer) -> dict[str, torch.Tensor]: ...  # its pointer-table tensors from a QuantizedLayer

Blocks talk to each other only through:

- **channels** (``Channel``): per-row tensors handed over as tagged pairs in
  scratch, in one of the formats below; the producer declares the regions
  (``Channel.regions``), consumers poll them with ``device.gfx950.channels``;
- **args**: plain device memory named in the pointer table (layer input / output,
  weights, caches, states). Names are global within a layer program, so a block
  takes its arg and channel names as constructor parameters.

Rows: a step has ``sc.rows = batch * spec`` rows; row ``r`` is request
``r // spec`` at offset ``r % spec``. Every channel is laid out row-major over
all rows, so a block never needs to know who produced a row.

Formats (one pair = 8 bytes = (32-bit word, tag)):

- ``BF16``: two bf16 per pair, ``width / 2`` pairs per row (element ``c`` of row
  ``r`` is half ``c % 2`` of pair ``(r * width + c) // 2``);
- ``F32``: one f32 per pair, ``width`` pairs per row;
- ``I32``: one int32 per pair, ``width`` pairs per row (ids, indices, counts: integer
  payloads never take an f32 round trip, which would not keep every bit pattern);
- ``MXFP8``: two regions: ``<name>.q`` with four E4M3 bytes per pair (element
  ``c`` is byte ``c % 4`` of pair ``(r * width + c) // 4``) and ``<name>.s`` with
  one E8M0 scale code per pair (group ``g`` of row ``r`` is pair
  ``r * width / 32 + g``, the code in the low byte).

Protocols (``Proto``; a channel property,
chosen per edge and scenario by the recipe):

- ``TAGGED`` (default): every 32-bit word travels with the launch tag as above;
  a consumer polls the words themselves (one round trip, 2x the bytes);
- ``FLAG`` (bulk + flag): the same words *dense* (word ``i`` of a region at byte
  ``4 i``, no tags: BF16 word = pair index above, MXFP8 ``.q`` word / ``.s`` code
  likewise), stored at device scope; each producer task drains its stores and then
  puts one tagged flag in ``<name>.f`` (slot = the producer's flag index, ``flags``
  slots). A consumer polls every flag, then loads the words (device scope): half the
  bytes, one more round trip (the producer's drain before its flag, the consumer's
  flag poll before its loads). Device helpers: ``device.gfx950.channels``.
  Where it wins and why (mostly: the consumer's input attempt holds fewer registers,
  so a Linear's weight prefetch gets deeper): the layer measurements
  L19-L27.

Block emit code follows (free ``@traced`` functions, explicit values;
see ``bench.toy`` for the pattern) and the knob discipline of
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

from ..plan.program import Region, Scenario, Stage


class Fmt(Enum):
    BF16 = "bf16"
    F32 = "f32"
    I32 = "i32"
    MXFP8 = "mxfp8"


class Proto(Enum):
    TAGGED = "tagged"  # (word, tag) pairs, polled word by word
    FLAG = "flag"  # dense words + one tagged flag per producer task (bulk + flag)


@dataclass(frozen=True)
class Channel:
    name: str
    width: int  # elements per row
    fmt: Fmt
    proto: Proto = Proto.TAGGED
    # "numa2": every region a copy on each IOD (``Region.home``); "": one copy. Not in the repr (old build keys stay):
    # the regions carry it into ``Program.key``.
    home: str = field(default="", repr=False)

    @property
    def flagged(self) -> bool:
        """Has a flag region (``FLAG``)."""
        return self.proto is not Proto.TAGGED

    @property
    def dense(self) -> bool:
        """Payload words without tags (``FLAG``)."""
        return self.proto is Proto.FLAG

    @property
    def flag_region(self) -> str:
        return f"{self.name}.f"


    def pairs_per_row(self) -> int:
        per = {Fmt.BF16: self.width // 2, Fmt.F32: self.width, Fmt.I32: self.width, Fmt.MXFP8: self.width // 4}
        return per[self.fmt]

    def words_per_row(self) -> int:
        """32-bit payload words per row (``.q`` words for MXFP8)."""
        return self.pairs_per_row()

    def region_names(self) -> tuple[str, ...]:
        names = (f"{self.name}.q", f"{self.name}.s") if self.fmt is Fmt.MXFP8 else (self.name,)
        return names + ((self.flag_region,) if self.flagged else ())

    def regions(self, rows: int, writers: tuple[str, ...], flags: int = 1) -> tuple[Region, ...]:
        """The scratch regions holding this channel for ``rows`` rows, put by ``writers``;
        ``flags``: the producer's flag slots (``FLAG`` only: one per producer task)."""
        if self.fmt is Fmt.MXFP8:
            assert self.width % 32 == 0, self
            words = ((f"{self.name}.q", rows * self.width // 4), (f"{self.name}.s", rows * self.width // 32))
        else:
            words = ((self.name, rows * self.pairs_per_row()),)
        h = self.home
        if not self.flagged:
            return tuple(Region(n, w, writers, home=h, rows=rows) for n, w in words)
        assert flags >= 1, (self, flags)
        if self.dense:  # two words per pair-sized slot of the layout
            words = tuple((n, -(-w // 2)) for n, w in words)
        return tuple(Region(n, w, writers, home=h) for n, w in words) + (
            Region(self.flag_region, flags, writers, home=h),
        )


@dataclass(frozen=True)
class Search:
    """A searchable constructor parameter of a block, as the tuner reads it: ``values`` (the
    first: the old default), ``claim`` "bitwise" (same sums, same bits) or "d10" (another summation order; or a
    callable of the block when the class depends on its configuration, e.g. a sliding vs a full attention),
    ``cost`` (knob, shape, placement, structural, frame), the ``evidence`` that made it searchable, and ``rows``
    (the row counts it applies to). Values the block's constructor refuses for this instance (a task group that
    does not divide its width) are dropped when the levers are derived."""

    values: tuple
    claim: str | Callable = "bitwise"
    cost: str = "knob"
    evidence: str = ""
    rows: tuple[int, int] = (1, 1 << 20)
    targets: tuple[str, ...] = ()  # a recipe knob's stages (``KNOB_LEVERS``; a block's parameter: the block's own)
    routing: bool = False  # its cost follows the routing (MoE groups, router slices): judged on real routing only
    # values whose format for the op is lossier than ATOM's default for it (merge rule 4): never searched unless the
    # case admits the lever (``tune.levers.Lever.below_floor``)
    below_floor: tuple = ()
    # (block) -> whether this instance uses the parameter at all (e.g. a shared expert's task shape where the block
    # has one): None, always. A parameter an instance ignores is no lever there (its arms would be no-ops)
    when: Callable | None = None
    # another SEARCH parameter of the same block this one is coupled with: also searched as a pair (both values in
    # one arm), for parameters that only pay together (MiniMax-M3's routed ug_g 6 with the shared sh_g 2: each alone
    # flat, the pair -0.7 µs)
    pair: str = ""
    # other parameters of the same block a value past ``values[0]`` needs set with it ((param, value) pairs): the arm
    # sets them too where the instance has them otherwise (Kimi K3's MLADecode kswz: only with lean and vtr), so the
    # lever applies to an instance without them and its arm is the combination
    needs: tuple = ()
    # values that deadlock from a batch size on ((value, batch) pairs): a bug to fix, never an arm there (the sweep's
    # hang watchdog found it; e.g. MiniMax-M3's BlockIndexer select "wave" at two requests)
    hangs: tuple = ()


class Block:
    """Base class of blocks. Subclasses set ``name`` and implement ``stages``
    (and ``pack`` when they read weights).

    A block is described by its constructor arguments alone (recorded at construction: ``rebuilt``), so a composed
    layer can be rebuilt with other parameters without the recipe (``plan.overrides``). ``SEARCH``: the
    parameters the tuner may change, {param: :class:`Search`}."""

    name: str = "block"
    SEARCH: dict[str, Search] = {}

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self._ctor = (args, kwargs)
        return self


    def stages(self, sc: Scenario) -> list[Stage]:
        raise NotImplementedError

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """Device tensors this block's args need, from a ``QuantizedLayer``."""
        return {}

