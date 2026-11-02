# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Collective boundaries for the gfx1250 Kimi packed Monokernel path.

The gfx950 monokernel publishes TP partials through peer-mapped memory and
reduces them from inside its persistent K2 launch.  That transport is both the
least portable part of the kernel and inherently single-host: HIP IPC cannot
map a buffer from another host.

The gfx1250 product path keeps rank-local compute fused around four ordered
boundaries: three all-reduces and one packed router/latent all-gather. No peer
address, remote mailbox or system-scope XGMI fence is part of this contract.

The routed latent reduction cannot be packed with the final hidden reduction:
RMSNorm and the routed up projection lie between them.  Keeping that third
collective also lets every rank retain only its shipped 1/TP routed-up row
shard.  Replicating the full routed-up weight would save one launch but add
roughly 49 MiB per layer and rank, which is not a product path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT, SUPPORTED_TP
from atom.models.kimi_k3.mono.kernels.token_tiling import (
    GFX1250_MAX_TOKENS,
)


class ExternalCollective(str, Enum):
    """A collective submitted by the host runner between local kernels."""

    OPROJ = "oproj"
    FRONT_PAYLOAD = "front_payload"
    ROUTED_LATENT = "routed_latent"
    FINAL_HIDDEN = "final_hidden"


class CollectiveKind(str, Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"


@dataclass(frozen=True)
class CollectiveBoundary:
    """One rank-local partial and the full tensor produced by its all-reduce."""

    name: ExternalCollective
    rows: int
    columns: int
    local_columns: int | None = None
    dtype: str = "bf16"
    kind: CollectiveKind = CollectiveKind.ALL_REDUCE
    local_rows: int | None = None

    def __post_init__(self) -> None:
        local_columns = (
            self.columns if self.local_columns is None else self.local_columns
        )
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError(
                f"{self.name.value}: invalid shape ({self.rows}, {self.columns})"
            )
        if local_columns <= 0 or self.columns % local_columns:
            raise ValueError(f"{self.name.value}: invalid local columns")
        local_rows = self.rows if self.local_rows is None else self.local_rows
        if local_rows <= 0 or self.rows % local_rows:
            raise ValueError(f"{self.name.value}: invalid local rows")
        if self.dtype not in ("bf16", "f32", "u8"):
            raise ValueError(f"{self.name.value}: unsupported dtype {self.dtype}")
        if self.kind is CollectiveKind.ALL_REDUCE and (
            local_columns != self.columns or local_rows != self.rows
        ):
            raise ValueError("all-reduce local/global shapes must match")

    @property
    def elements(self) -> int:
        return self.rows * self.columns

    @property
    def bytes(self) -> int:
        if self.dtype == "f32":
            element_bytes = 4
        elif self.dtype == "bf16":
            element_bytes = 2
        else:
            element_bytes = 1
        return self.elements * element_bytes

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.columns

    @property
    def local_shape(self) -> tuple[int, int]:
        return (
            self.rows if self.local_rows is None else self.local_rows,
            self.local_columns or self.columns,
        )

    @property
    def global_shape(self) -> tuple[int, int]:
        return self.shape


@dataclass(frozen=True)
class ExternalCollectivePlan:
    """Ordered host/device contract for one gfx1250 Kimi layer.

    K1 is rank-local.  K2 is split into local launchable regions around the
    boundaries in :attr:`collectives`:

    ``K1 -> o_proj AR -> local router/latent -> packed all-gather
       -> local experts -> routed-latent AR
       -> routed norm/sharded up + shared partial -> final-hidden AR``.

    The last local stage adds this rank's routed-up row band to its full
    shared-down partial.  The final all-reduce both assembles the routed bands
    and sums the K-sharded shared branch.
    """

    rows: int
    tp: int = SUPPORTED_TP[0]
    hidden: int = HIDDEN
    latent: int = LATENT

    def __post_init__(self) -> None:
        if not 1 <= self.rows <= GFX1250_MAX_TOKENS:
            raise ValueError(
                f"rows must be in [1, {GFX1250_MAX_TOKENS}], got {self.rows}"
            )
        if self.tp not in SUPPORTED_TP:
            raise ValueError(
                f"gfx1250 Kimi external collectives require TP{SUPPORTED_TP[0]}, "
                f"got TP{self.tp}"
            )
        if self.hidden != HIDDEN or self.latent != LATENT:
            raise ValueError(
                "gfx1250 Kimi external collectives only support the shipped "
                f"shape hidden={HIDDEN}, latent={LATENT}"
            )

    @property
    def collectives(self) -> tuple[CollectiveBoundary, ...]:
        payload = FrontPayloadLayout(self.rows, self.tp)
        return (
            CollectiveBoundary(ExternalCollective.OPROJ, self.rows, self.hidden),
            CollectiveBoundary(
                ExternalCollective.FRONT_PAYLOAD,
                self.tp,
                payload.local_bytes,
                payload.local_bytes,
                "u8",
                CollectiveKind.ALL_GATHER,
                1,
            ),
            CollectiveBoundary(
                ExternalCollective.ROUTED_LATENT, self.rows, self.latent
            ),
            CollectiveBoundary(ExternalCollective.FINAL_HIDDEN, self.rows, self.hidden),
        )

    def boundary(self, name: ExternalCollective) -> CollectiveBoundary:
        try:
            return next(item for item in self.collectives if item.name is name)
        except StopIteration as error:
            raise ValueError(f"unknown collective boundary {name!r}") from error


@dataclass(frozen=True)
class FrontPayloadLayout:
    """Peer-major heterogeneous router/latent all-gather payload.

    Each TP rank contributes one contiguous byte row:
    ``router[S,112] f32 || latent[S,448] bf16``. The gathered tensor is
    ``[TP, local_bytes] u8`` in TP-rank order. Router values are never cast.
    """

    rows: int
    tp: int = SUPPORTED_TP[0]

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("front payload rows must be positive")
        if self.tp not in SUPPORTED_TP:
            raise ValueError(f"front payload requires TP{SUPPORTED_TP[0]}")

    @property
    def router_columns(self) -> int:
        return 896 // self.tp

    @property
    def latent_columns(self) -> int:
        return LATENT // self.tp

    @property
    def router_bytes(self) -> int:
        return self.rows * self.router_columns * 4

    @property
    def latent_bytes(self) -> int:
        return self.rows * self.latent_columns * 2

    @property
    def latent_offset(self) -> int:
        return self.router_bytes

    @property
    def local_bytes(self) -> int:
        return self.router_bytes + self.latent_bytes

    @property
    def local_shape(self) -> tuple[int, int]:
        return 1, self.local_bytes

    @property
    def gathered_shape(self) -> tuple[int, int]:
        return self.tp, self.local_bytes
