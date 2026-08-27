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

    def __post_init__(self) -> None:
        if not isinstance(self.fork_tokens, int) or isinstance(self.fork_tokens, bool):
            raise TypeError("fork_tokens must be an integer")
        if not isinstance(self.readable_midstep, bool):
            raise TypeError("readable_midstep must be a bool")
        if self.kind == NONE and self.readable_midstep:
            raise ValueError("none transfer has no state to read midstep")
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
        if self.kind == NONE:
            if self.fork_tokens != 0 or self.paged_layout_id is not None:
                raise ValueError("none transfer cannot carry tokens or a PAGE layout")
            return
        raise ValueError(f"unknown state transfer kind {self.kind!r}")

    @classmethod
    def none(cls) -> "StateTransfer":
        return cls(NONE)

    @classmethod
    def fork(cls, tokens: int, readable_midstep: bool = False) -> "StateTransfer":
        return cls(FORK, tokens, readable_midstep=readable_midstep)

    @classmethod
    def copy(cls, layout_id: str, readable_midstep: bool = False) -> "StateTransfer":
        return cls(COPY, paged_layout_id=layout_id, readable_midstep=readable_midstep)

    def to_wire(self) -> dict[str, str | int | bool | None]:
        return {
            "kind": self.kind,
            "fork_tokens": self.fork_tokens,
            "paged_layout_id": self.paged_layout_id,
            "readable_midstep": self.readable_midstep,
        }

    @classmethod
    def from_wire(cls, wire: object) -> "StateTransfer":
        if not isinstance(wire, Mapping):
            raise TypeError("state transfer capability must be a mapping")
        expected = {"kind", "fork_tokens", "paged_layout_id", "readable_midstep"}
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
        )

    @property
    def copies(self) -> bool:
        return self.kind == COPY

    @property
    def forks(self) -> bool:
        return self.kind == FORK

    @property
    def successor_room(self) -> float:
        return inf if self.kind == NONE else float(self.fork_tokens)


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
