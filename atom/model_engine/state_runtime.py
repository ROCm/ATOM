# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import inf

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)

FORK = "fork"
COPY = "copy"
NONE = "none"
# No state is carried or checkpointed: a forward resuming at a prefix hit
# rebuilds it by recomputing the hash block below the hit
# (`BlockManager.replay_start`), writing none of what the cache holds there.
REPLAY = "replay"


@dataclass(frozen=True)
class StateTransfer:
    """How a backend transfers one request's state to another slot."""

    kind: str
    fork_tokens: int = 0
    paged_layout_id: str | None = None
    #: Whether this backend can extract a state snapshot at a position *inside*
    #: a forward, rather than only at the forward's last token.
    #:
    #: A separate axis from `kind`, and it must stay one. `kind` answers how a
    #: slot is handed to *another request*; this answers where within one
    #: forward a snapshot can be taken at all. GDN is `fork(1)` and — because
    #: its chunk kernel's per-chunk intermediates are copied out —
    #: midstep-readable; DeepSeek-V4 is `copy()` and is not, because its
    #: compressor ring is not materialized at interior boundaries the way a
    #: chunk kernel's `h` is. Deriving either from the other would silently
    #: turn the prefill chunk cut off for a backend that still needs it.
    #:
    #: Default False, so every backend keeps today's behavior until it has
    #: actually ported the copy-out.
    readable_midstep: bool = False
    #: Token spacing of the interior states a readable backend exposes, counted
    #: from the *start of the forward*, not from position 0. GDN's chunk kernel
    #: materializes `h` every 64 tokens of the step, so a forward over
    #: `(start, end]` can be read at `start + 64k` and nowhere else inside it.
    #: 0 when not readable.
    midstep_grid: int = 0
    #: Tokens of this forward a snapshot needs behind it: GDN's conv window
    #: (`conv_kernel - 1`) is sliced out of the step's own conv input. 0 when
    #: not readable.
    midstep_min_offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.fork_tokens, int) or isinstance(self.fork_tokens, bool):
            raise TypeError("fork_tokens must be an integer")
        if not isinstance(self.readable_midstep, bool):
            raise TypeError("readable_midstep must be a bool")
        if self.kind in (NONE, REPLAY) and self.readable_midstep:
            raise ValueError(f"{self.kind} transfer has no state to read midstep")
        for name in ("midstep_grid", "midstep_min_offset"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(f"{name} must be a non-negative integer")
        if self.readable_midstep and self.midstep_grid <= 0:
            raise ValueError("a midstep-readable transfer must declare its grid")
        if not self.readable_midstep and (self.midstep_grid or self.midstep_min_offset):
            raise ValueError("midstep geometry on a transfer that is not readable")
        if self.kind == COPY:
            if self.fork_tokens != 0:
                raise ValueError("copy transfer cannot bind successor tokens")
            if not isinstance(self.paged_layout_id, str) or not self.paged_layout_id:
                raise ValueError("copy transfer requires a non-empty PAGE layout id")
            return
        if self.kind == FORK:
            if self.fork_tokens <= 0:
                raise ValueError("fork transfer requires positive fork_tokens")
            if self.paged_layout_id is not None:
                raise ValueError("fork transfer cannot declare a PAGE layout")
            return
        if self.kind in (NONE, REPLAY):
            if self.fork_tokens != 0 or self.paged_layout_id is not None:
                raise ValueError(
                    f"{self.kind} transfer cannot carry tokens or a PAGE layout"
                )
            return
        raise ValueError(f"unknown state transfer kind {self.kind!r}")

    @classmethod
    def none(cls) -> "StateTransfer":
        return cls(NONE)

    @classmethod
    def replay(cls) -> "StateTransfer":
        return cls(REPLAY)

    @classmethod
    def fork(
        cls,
        tokens: int,
        readable_midstep: bool = False,
        midstep_grid: int = 0,
        midstep_min_offset: int = 0,
    ) -> "StateTransfer":
        return cls(
            FORK,
            tokens,
            readable_midstep=readable_midstep,
            midstep_grid=midstep_grid,
            midstep_min_offset=midstep_min_offset,
        )

    @classmethod
    def copy(
        cls,
        layout_id: str,
        readable_midstep: bool = False,
        midstep_grid: int = 0,
        midstep_min_offset: int = 0,
    ) -> "StateTransfer":
        return cls(
            COPY,
            paged_layout_id=layout_id,
            readable_midstep=readable_midstep,
            midstep_grid=midstep_grid,
            midstep_min_offset=midstep_min_offset,
        )

    def to_wire(self) -> dict[str, str | int | bool | None]:
        return {
            "kind": self.kind,
            "fork_tokens": self.fork_tokens,
            "paged_layout_id": self.paged_layout_id,
            "readable_midstep": self.readable_midstep,
            "midstep_grid": self.midstep_grid,
            "midstep_min_offset": self.midstep_min_offset,
        }

    @classmethod
    def from_wire(cls, wire: object) -> "StateTransfer":
        if not isinstance(wire, Mapping):
            raise TypeError("state transfer capability must be a mapping")
        expected = {
            "kind",
            "fork_tokens",
            "paged_layout_id",
            "readable_midstep",
            "midstep_grid",
            "midstep_min_offset",
        }
        if set(wire) != expected:
            raise ValueError(
                "invalid state transfer capability fields: "
                f"expected={sorted(expected)}, got={sorted(wire)}"
            )
        return cls(
            kind=wire["kind"],  # type: ignore[arg-type]
            fork_tokens=wire["fork_tokens"],  # type: ignore[arg-type]
            paged_layout_id=wire["paged_layout_id"],  # type: ignore[arg-type]
            readable_midstep=wire["readable_midstep"],  # type: ignore[arg-type]
            midstep_grid=wire["midstep_grid"],  # type: ignore[arg-type]
            midstep_min_offset=wire["midstep_min_offset"],  # type: ignore[arg-type]
        )

    @property
    def copies(self) -> bool:
        return self.kind == COPY

    @property
    def forks(self) -> bool:
        return self.kind == FORK

    @property
    def replays(self) -> bool:
        return self.kind == REPLAY

    @property
    def successor_room(self) -> float:
        return inf if self.kind in (NONE, REPLAY) else float(self.fork_tokens)


@dataclass(frozen=True)
class StateRuntime:
    """Validated state transfer and optional checkpoint geometry."""

    transfer: StateTransfer = field(default_factory=StateTransfer.none)
    checkpoint_spec: PagedStateCheckpointSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transfer, StateTransfer):
            raise TypeError("state runtime transfer must be a StateTransfer")
        if self.checkpoint_spec is not None and not isinstance(
            self.checkpoint_spec, PagedStateCheckpointSpec
        ):
            raise TypeError(
                "state runtime checkpoint_spec must be a PagedStateCheckpointSpec"
            )
        if self.transfer.copies:
            if self.checkpoint_spec is None:
                raise ValueError(
                    "StateTransfer.copy(layout_id) requires a PAGE checkpoint spec"
                )
            if self.transfer.paged_layout_id != self.checkpoint_spec.layout_id:
                raise ValueError(
                    "state runtime PAGE layout mismatch: "
                    f"transfer={self.transfer.paged_layout_id!r}, "
                    f"spec={self.checkpoint_spec.layout_id!r}"
                )
        elif self.checkpoint_spec is not None:
            raise ValueError(
                f"StateTransfer.{self.transfer.kind} cannot carry a PAGE checkpoint spec"
            )

    def to_wire(self) -> dict[str, object]:
        return {
            "transfer": self.transfer.to_wire(),
            "checkpoint_spec": (
                None if self.checkpoint_spec is None else self.checkpoint_spec.to_wire()
            ),
        }

    @classmethod
    def from_wire(cls, wire: object) -> "StateRuntime":
        if not isinstance(wire, Mapping):
            raise TypeError("state runtime must be a mapping")
        expected = {"transfer", "checkpoint_spec"}
        if set(wire) != expected:
            raise ValueError(
                "invalid state runtime fields: "
                f"expected={sorted(expected)}, got={sorted(wire)}"
            )
        checkpoint_wire = wire["checkpoint_spec"]
        checkpoint_spec = (
            None
            if checkpoint_wire is None
            else PagedStateCheckpointSpec.from_wire(checkpoint_wire)
        )
        return cls(
            transfer=StateTransfer.from_wire(wire["transfer"]),
            checkpoint_spec=checkpoint_spec,
        )


DEFAULT_STATE_RUNTIME = StateRuntime()


@dataclass(frozen=True)
class StateMaintenanceOps:
    """State movement drained once before a model batch."""

    relocations: tuple[tuple[int, int], ...] = ()
    checkpoint_stores: tuple[CheckpointStoreOp, ...] = ()
    checkpoint_restores: tuple[CheckpointRestoreOp, ...] = ()

    @property
    def empty(self) -> bool:
        return not (
            self.relocations or self.checkpoint_stores or self.checkpoint_restores
        )


# ── midstep checkpoint geometry ──────────────────────────────────────────────
#
# One predicate for both ends of a midstep checkpoint. The scheduler reserves a
# position and publishes its hash after the forward; the runner writes the
# bytes during it. If the two disagree on which positions are writable, the
# scheduler files a hash over a destination nobody wrote, and a request that
# resumes there continues from the previous tenant's state with no error
# anywhere. So the scheduler filters with this and the runner asserts it.


def midstep_target_ok(
    pos: int, start: int, end: int, grid: int, min_offset: int
) -> bool:
    """Whether a forward over `(start, end]` can snapshot the state at `pos`.

    Interior positions must sit on the forward's own grid, `start + grid*k`:
    that is where the chunk kernel left a state in `h`, and a position between
    two of them reads the earlier one while being filed under the later hash.
    `end` is always readable, from the runtime slot the forward leaves behind.

    `min_offset` tokens of this forward must precede `pos`, because the conv
    window is sliced out of this forward's conv input and has no earlier rows.
    """
    if not start < pos <= end or pos - start < min_offset:
        return False
    return pos == end or (pos - start) % grid == 0


def snap_midstep_position(pos: int, start: int, end: int, grid: int) -> int:
    """`pos` moved back onto the grid of a forward over `(start, end]`.

    `end` stays where it is: the runtime slot holds it exactly. Anything else
    goes back to the last state the chunk kernel materialized at or before it,
    which costs the next resume at most `grid - 1` tokens of recompute.
    """
    if pos == end:
        return pos
    return start + (pos - start) // grid * grid


def midstep_checkpoint_targets(
    state_rows: list[int],
    num_cached_tokens,
    num_scheduled_tokens,
    runtime_slots: list[int],
    saves: list[list[tuple[int, int]]],
    num_slots: int,
    grid: int,
    min_offset: int,
) -> list[tuple[int, int, int, int, int]]:
    """`(row, slot, offset, is_end, runtime_slot)` for every reservation.

    `saves`, `runtime_slots` and `state_rows` are indexed by state-holding
    sequence; `state_rows[j]` is that sequence's row in the batch, which is
    what the token counts and the kernel's `cu_seqlens` are indexed by. The
    two only coincide when every row holds state.

    Raises rather than skipping. Every reservation here has already been
    promised to `commit_midstep`, which publishes its hash after the forward;
    a skipped write is a findable checkpoint over unwritten bytes.
    """
    if not (len(state_rows) == len(runtime_slots) == len(saves)):
        raise RuntimeError(
            "midstep targets: state rows, runtime slots and saves disagree "
            f"({len(state_rows)}, {len(runtime_slots)}, {len(saves)})"
        )
    out = []
    for j, reservations in enumerate(saves):
        row = state_rows[j]
        start = int(num_cached_tokens[row])
        end = start + int(num_scheduled_tokens[row])
        for dst, pos in reservations:
            dst, pos = int(dst), int(pos)
            if not 0 <= dst < num_slots:
                raise RuntimeError(
                    f"midstep checkpoint slot {dst} outside this rank's "
                    f"{num_slots}-slot state pool"
                )
            if not midstep_target_ok(pos, start, end, grid, min_offset):
                raise RuntimeError(
                    f"midstep checkpoint at {pos} is not writable by a forward "
                    f"over ({start}, {end}] (grid {grid}, min offset {min_offset})"
                )
            out.append((row, dst, pos - start, int(pos == end), runtime_slots[j]))
    return out
