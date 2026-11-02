# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Architecture contracts used while porting mono device code.

These are ISA/topology facts, not deployment support.  In particular, naming
``gfx1250`` here does not make any mono kernel runnable there: the persistent
grid, LDS layout, matrix fragments, weight layouts and stage schedules must
each opt into a compatible execution model before deployment may admit it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MatrixIsa(str, Enum):
    """The matrix instruction family used by an architecture."""

    MFMA = "mfma"
    WMMA = "wmma"


@dataclass(frozen=True)
class Bf16MmaContract:
    """One wave's BF16 16x16 matrix operation and accumulator fragment."""

    isa: MatrixIsa
    m: int
    n: int
    k: int
    accumulator_elements_per_lane: int


@dataclass(frozen=True)
class Architecture:
    """Wave and matrix-ISA facts needed by shared mono primitives."""

    gfx: str
    wave_size: int
    bf16_mma: Bf16MmaContract
    native_permlane_exchange: bool
    max_workgroup_lds_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.wave_size < 16 or self.wave_size & (self.wave_size - 1):
            raise ValueError(f"{self.gfx}: wave size must be a power of two >= 16")
        accumulator_elements = (
            self.wave_size * self.bf16_mma.accumulator_elements_per_lane
        )
        if accumulator_elements != self.bf16_mma.m * self.bf16_mma.n:
            raise ValueError(
                f"{self.gfx}: {accumulator_elements} distributed accumulator "
                f"elements != {self.bf16_mma.m}x{self.bf16_mma.n}"
            )
        if (
            self.max_workgroup_lds_bytes is not None
            and self.max_workgroup_lds_bytes <= 0
        ):
            raise ValueError(f"{self.gfx}: LDS capacity must be positive")

    @property
    def reduction_offsets(self) -> tuple[int, ...]:
        """Descending XOR offsets that reduce exactly one wave."""

        return tuple(1 << bit for bit in range(self.wave_size.bit_length() - 2, -1, -1))


GFX950 = Architecture(
    gfx="gfx950",
    wave_size=64,
    bf16_mma=Bf16MmaContract(
        isa=MatrixIsa.MFMA,
        m=16,
        n=16,
        k=32,
        accumulator_elements_per_lane=4,
    ),
    native_permlane_exchange=True,
    # The existing production execution contract allocates one 160-KiB CTA.
    max_workgroup_lds_bytes=160 * 1024,
)


GFX1250 = Architecture(
    gfx="gfx1250",
    wave_size=32,
    bf16_mma=Bf16MmaContract(
        isa=MatrixIsa.WMMA,
        m=16,
        n=16,
        k=32,
        accumulator_elements_per_lane=8,
    ),
    # The gfx950 reduction path's permlane32 exchange crosses halves of a
    # wave64.  A wave32 has no corresponding 32-lane partner; shared code uses
    # shuffle_xor for its one cross-row (offset 16) exchange instead.
    native_permlane_exchange=False,
    # Checked-in AITER gfx1250 sources consistently enforce 327680 bytes:
    # pa_decode_kernel.py and pa_decode/traits.py use that exact limit, while
    # fmha_gfx1250 documents it as the 320-KiB workgroup budget.
    max_workgroup_lds_bytes=327680,
)


ARCHITECTURES = {architecture.gfx: architecture for architecture in (GFX950, GFX1250)}


def architecture_for(gfx: str) -> Architecture:
    """Return a known architecture contract, refusing an inferred fallback."""

    try:
        return ARCHITECTURES[gfx]
    except KeyError as error:
        known = ", ".join(ARCHITECTURES)
        raise ValueError(
            f"unsupported mono architecture {gfx!r}; known: {known}"
        ) from error
