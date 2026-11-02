# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure gfx950/gfx1250 contracts for Kimi K2's post-expert tail."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    GFX1250_GEMV,
    LDS_PAD,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import (
    GFX1250_MAX_TOKENS,
    Gfx1250TokenTiling,
)
from atom.mono.plan.architecture import GFX950, GFX1250, Architecture

TP = 8
TOKENS_MAX = GFX1250_MAX_TOKENS
ROWS = 16
HIDDEN = 7168
LATENT = 3584
SHARED_INTERMEDIATE = 768
UP_ROWS = HIDDEN // TP
SHARED_DOWN_TASKS = HIDDEN // ROWS
LNORM_TASKS = LATENT // 128
UP_TASKS = UP_ROWS // ROWS
FINAL_TASKS = HIDDEN // 128
ALIGNMENT = 16


@dataclass(frozen=True)
class TailDeviceEvidence:
    primitive: str
    source: str
    contract: str


GFX1250_TAIL_EVIDENCE = (
    TailDeviceEvidence(
        "BF16 WMMA GEMV",
        "stages/gemv_gfx1250.py and AITER pa_decode/fmha gfx1250",
        "v16 BF16 operands, v8 f32 accumulator, wave32 output ownership",
    ),
    TailDeviceEvidence(
        "wave32 XOR reduction",
        "atom/mono/device/ops.py and AITER fmha_gfx1250",
        "shuffle_xor offsets 16,8,4,2,1 over an explicit wave32",
    ),
    TailDeviceEvidence(
        "rank-order fold",
        "atom/mono/device/ranks.py:sum_partials",
        "BF16 rank partials accumulate in fp32 from rank 0 through rank 7",
    ),
)


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@dataclass(frozen=True)
class TailReductionContract:
    architecture: Architecture
    numerical_order: str

    def wave64_pair_tree(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """The exact gfx950 64-lane XOR tree, represented by lane pairs."""

        groups = tuple((lane,) for lane in range(64))
        levels = []
        for offset in (32, 16, 8, 4, 2, 1):
            levels.append(tuple((lane, lane ^ offset) for lane in range(64)))
            groups = tuple(groups[lane] + groups[lane ^ offset] for lane in range(64))
        assert set(groups[0]) == set(range(64))
        return tuple(levels)

    def reduce_norm_pairs(self, values: tuple[float, ...]) -> float:
        """Emulate lane zero's f32 result with the production XOR order."""

        if len(values) != 64:
            raise ValueError("one norm task needs 64 column-pair values")
        lanes = tuple(f32(value) for value in values)
        for offset in (32, 16, 8, 4, 2, 1):
            previous = lanes
            lanes = tuple(
                f32(previous[lane] + previous[lane ^ offset]) for lane in range(64)
            )
        return lanes[0]

    def reduce_norm_pairs_wave32(self, values: tuple[float, ...]) -> float:
        """gfx1250 implementation: offset-32 in LDS, then wave32 XOR."""

        if self.architecture is not GFX1250:
            raise ValueError("wave32 norm reduction requires gfx1250")
        if len(values) != 64:
            raise ValueError("one norm task needs 64 column-pair values")
        lanes = tuple(f32(values[lane] + values[lane + 32]) for lane in range(32))
        for offset in (16, 8, 4, 2, 1):
            previous = lanes
            lanes = tuple(
                f32(previous[lane] + previous[lane ^ offset]) for lane in range(32)
            )
        return lanes[0]

    @staticmethod
    def rank_order(values: tuple[float, ...]) -> float:
        if len(values) != TP:
            raise ValueError(f"rank reduction needs {TP} partials")
        total = f32(0.0)
        for value in values:
            total = f32(total + f32(value))
        return total


GFX950_TAIL_REDUCTION = TailReductionContract(
    GFX950,
    "wave64 XOR 32,16,8,4,2,1; rank order 0..7",
)
GFX1250_TAIL_REDUCTION = TailReductionContract(
    GFX1250,
    (
        "offset-32 pair is staged in LDS, then wave32 XOR 16,8,4,2,1; "
        "bitwise-equivalent f32 addition tree to gfx950"
    ),
)


@dataclass(frozen=True)
class TailTransportContract:
    """Buffers the local tail consumes/produces without asserting visibility."""

    tp: int = TP
    latent_input: str = "latr_partials[rank][token][latent]"
    final_input: str = "rank_final[rank][token][hidden]"
    required_order: str = "rank-major 0..7"
    capability: str = "transport publication/readiness unsupported"

    def latent_elements(self, tokens: int) -> int:
        self._tokens(tokens)
        return self.tp * tokens * LATENT

    def final_elements(self, tokens: int) -> int:
        self._tokens(tokens)
        return self.tp * tokens * HIDDEN

    @staticmethod
    def _tokens(tokens: int) -> None:
        if not 1 <= tokens <= TOKENS_MAX:
            raise ValueError(f"tokens must be in [1, {TOKENS_MAX}]")


@dataclass(frozen=True)
class TailLdsRegion:
    name: str
    offset: int
    bytes: int

    @property
    def end(self) -> int:
        return self.offset + self.bytes


@dataclass(frozen=True)
class TailLdsPlan:
    tokens: int
    gemv_regions: tuple[TailLdsRegion, ...]
    norm_regions: tuple[TailLdsRegion, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.tokens <= TOKENS_MAX:
            raise ValueError("tail LDS tokens outside supported range")
        for regions in (self.gemv_regions, self.norm_regions):
            previous = 0
            for region in regions:
                if region.offset % ALIGNMENT:
                    raise ValueError(f"{region.name}: unaligned LDS offset")
                if region.offset < previous:
                    raise ValueError(f"{region.name}: overlapping LDS region")
                previous = region.end

    @property
    def required_bytes(self) -> int:
        return max(
            max((region.end for region in self.gemv_regions), default=0),
            max((region.end for region in self.norm_regions), default=0),
        )

    @property
    def token_tiles(self):
        return Gfx1250TokenTiling(self.tokens).tiles

    @property
    def lds_rows(self) -> int:
        return Gfx1250TokenTiling(self.tokens).lds_rows


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def tail_lds_plan(tokens: int, waves: int) -> TailLdsPlan:
    if waves <= 0:
        raise ValueError("tail LDS plan needs waves")
    tile_rows = Gfx1250TokenTiling(tokens).lds_rows
    activation = tile_rows * (LATENT + LDS_PAD) * 2
    accumulator = (
        waves * GFX1250.wave_size * GFX1250_GEMV.accumulator_elements_per_lane * 4
    )
    accumulator_offset = _align(activation)
    gemv = (
        TailLdsRegion("tail_gemv_rows", 0, activation),
        TailLdsRegion(
            "tail_gemv_accumulator",
            accumulator_offset,
            accumulator,
        ),
    )
    norm = (TailLdsRegion("tail_norm_tree", 0, tokens * 64 * 4),)
    return TailLdsPlan(tokens, gemv, norm)


@dataclass(frozen=True)
class TailOwnershipContract:
    architecture: Architecture
    threads: int
    waves: int
    tokens: int

    def __post_init__(self) -> None:
        if self.architecture is GFX1250:
            if self.threads != self.waves * GFX1250.wave_size:
                raise ValueError("gfx1250 tail threads/waves disagree")
        elif self.architecture is GFX950:
            if self.threads != self.waves * GFX950.wave_size:
                raise ValueError("gfx950 tail threads/waves disagree")
        else:
            raise ValueError(f"unsupported tail architecture {self.architecture.gfx}")
        TailTransportContract._tokens(self.tokens)

    @property
    def matrix_numerical_order(self) -> str:
        if self.architecture is GFX950:
            return (
                "gfx950 MFMA K128 chunks accumulate per wave, then waves "
                "fold in ascending order"
            )
        return (
            "gfx1250 WMMA preserves K128 chunk and ascending-wave fold order; "
            "WMMA-vs-MFMA intrinsic rounding is not claimed bitwise identical"
        )

    @property
    def matrix_bitwise_equivalence_proven(self) -> bool:
        return self.architecture is GFX950

    @staticmethod
    def up_owner_rank(hidden_column: int) -> int:
        if not 0 <= hidden_column < HIDDEN:
            raise ValueError("hidden column outside output")
        return hidden_column // UP_ROWS

    @staticmethod
    def final_contribution_order(hidden_column: int) -> tuple[str, ...]:
        """Per-rank BF16 values consumed by the final rank-order fold."""

        owner = TailOwnershipContract.up_owner_rank(hidden_column)
        return tuple(
            ("bf16(shared_down + routed_up)" if rank == owner else "bf16(shared_down)")
            for rank in range(TP)
        )

    def thread_pair_work(
        self, pairs_per_token: int, tid: int
    ) -> tuple[tuple[int, int], ...]:
        if not 0 <= tid < self.threads:
            raise ValueError("tail thread outside CTA")
        total = self.tokens * pairs_per_token
        return tuple(
            divmod(index, pairs_per_token) for index in range(tid, total, self.threads)
        )

    def norm_wave_tokens(self, wave: int) -> tuple[int, ...]:
        if self.architecture is not GFX1250:
            raise ValueError("two-wave norm ownership is gfx1250-only")
        if not 0 <= wave < self.waves:
            raise ValueError("tail wave outside CTA")
        return tuple(range(wave, self.tokens, self.waves))

    @staticmethod
    def shared_down_rows(task: int) -> tuple[int, ...]:
        if not 0 <= task < SHARED_DOWN_TASKS:
            raise ValueError("shared-down task outside output")
        return tuple(range(task * ROWS, (task + 1) * ROWS))

    @staticmethod
    def lnorm_columns(task: int) -> tuple[int, ...]:
        if not 0 <= task < LNORM_TASKS:
            raise ValueError("lnorm task outside latent")
        return tuple(range(task * 128, (task + 1) * 128))

    @staticmethod
    def up_rows(rank: int, task: int) -> tuple[int, ...]:
        if not 0 <= rank < TP or not 0 <= task < UP_TASKS:
            raise ValueError("up task outside rank shard")
        first = rank * UP_ROWS + task * ROWS
        return tuple(range(first, first + ROWS))

    @staticmethod
    def final_columns(task: int) -> tuple[int, ...]:
        if not 0 <= task < FINAL_TASKS:
            raise ValueError("final task outside hidden")
        return tuple(range(task * 128, (task + 1) * 128))


def tail_contract_for(
    architecture: Architecture,
    *,
    threads: int,
    waves: int,
    tokens: int,
) -> TailOwnershipContract:
    return TailOwnershipContract(architecture, threads, waves, tokens)


def tail_backend_for(
    architecture: Architecture,
    *,
    threads: int,
    waves: int,
    tokens: int,
):
    ownership = tail_contract_for(
        architecture,
        threads=threads,
        waves=waves,
        tokens=tokens,
    )
    if architecture is not GFX1250:
        raise ValueError("production gfx950 tail remains stages/moe.py")
    from atom.models.kimi_k3.mono.kernels.stages.moe_tail_gfx1250 import (
        TailBackend,
    )

    return TailBackend(ownership)
