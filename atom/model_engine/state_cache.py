# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Protocol, runtime_checkable

from atom.model_engine.sequence import Sequence


@runtime_checkable
class StateCache(Protocol):
    """One `Pool.STATE` cache class's checkpoint lifecycle.

    Pool.STATE holds several such classes (see `sub_pool_spec.py`): the sliding
    window, the DeepSeek-V4 compressor ring, GDN/Mamba recurrence. They have
    three things in common, and this protocol is exactly those three:

    - each scales with in-flight requests, so a boundary is only resumable if
      somebody deliberately kept its state there (`checkpoint`);
    - each can therefore veto a prefix-cache hit, by answering how far back the
      nearest boundary it *can* resume from is (`resumable_hit`);
    - *where* keeping one is worth its cost is the same question for all of
      them, so the ladder lives in `BlockManager`, not here.

    Vocabulary: *checkpoint*, noun and verb, is this — a boundary kept
    resumable, and the act of keeping it. *Publish* means something else and is
    never a synonym here: a block entering the content-addressed KV index.

    What differs between classes is only *how* a boundary is kept, and that
    follows from one property — mutability:

      immutable   a filled SWA block is never written again, so keeping it is
                  one extra ref; a reader shares it and needs nothing else.
      copyable    the DeepSeek-V4 compressor entry is a contiguous byte range,
                  so keeping it is a duplicate handed to the index and the owner
                  is never disturbed; the reader is handed a duplicate too.
      rolling     GDN recurrence is still being written by its owner and is not
                  one range to duplicate, so keeping it means handing it over
                  and taking a fresh one; the reader forks, and the forward
                  right after the hand-over has to refill the replacement.

    `successor_room` is that property, quantified — and it is the only thing the
    ladder needs to know about a class, which is why the rest of the difference
    can stay inside `checkpoint`.
    """

    #: False when this class has nothing to say about any seq (not sized, or
    #: prefix caching off). Callers still invoke the methods — they are
    #: identity/no-op — so no `if enabled` appears at the call sites.
    enabled: bool

    #: Tokens the forward *after* a checkpoint must carry for that checkpoint to
    #: come out whole. Three regimes, one comparison:
    #:
    #:   0     immutable or copyable — nothing is handed over, so no successor
    #:         is needed.
    #:   n>0   rolling — the successor has to refill the replacement group, and
    #:         this is how many committed tokens that takes.
    #:   inf   the class cannot be checkpointed at all, so no position ever
    #:         qualifies. Distinct from 0, and a backend says which by declaring
    #:         a `StateTransfer` rather than a bare token count.
    successor_room: float

    #: Whether this class can snapshot a boundary *inside* a forward instead of
    #: only at the forward's last token.
    #:
    #: True changes where the ladder's cost is paid, not what a checkpoint is
    #: worth. A class that answers false forces the scheduler to end a forward
    #: exactly on each rung (`BlockManager.checkpoint_cut`), so a prompt with N
    #: rungs on it is N shortened prefill chunks; one that answers true reads
    #: each rung out of intermediates its kernel already materialized, so the
    #: same N rungs are N copies inside one full-length forward.
    #:
    #: Independent of `successor_room`. That says what the forward *after* a
    #: checkpoint owes it; this says where within *this* forward it can be taken
    #: at all. A rolling class can be either — GDN forks and is readable — and so
    #: can a copying one, so neither implies the other.
    readable_midstep: bool

    def applies(self, seq: Sequence) -> bool:
        """Whether this class gates or checkpoints anything for `seq`."""
        ...

    def reserve_midstep(
        self, seq: Sequence, positions: list[tuple[int, int]]
    ) -> list[tuple]:
        """Take a destination for each `(position, hash)` the next forward covers.

        Returns `(destination, position, hash)` per reservation made, for the
        runner to write into and `publish_midstep` to file. Empty — not an
        error — for a class that is not `readable_midstep`, so the call site
        needs no branch.

        Best-effort and order-preserving: reservations stop at the first one
        capacity cannot fill, so the earliest position survives a shortage.
        """
        ...

    def publish_midstep(self, reservations: list[tuple], seq: Sequence = None) -> None:
        """File reservations whose bytes the completed forward has now written.

        The other half of `reserve_midstep`, and the split is the point:
        publishing before the forward would index a boundary over bytes nobody
        had written, and a request resuming there would read the destination's
        previous tenant.

        `seq` lets an implementation tell this prompt's own end — the position
        the next turn actually resumes at — from the ladder and demand rungs,
        which only guess. Optional: a class that ranks its checkpoints equally
        ignores it.
        """
        ...

    def cancel_midstep(self, reservations: list[tuple]) -> None:
        """Release reservations whose forward never ran, holding nothing."""
        ...

    def resumable_hit(
        self,
        seq: Sequence,
        hit: int,
        block_hashes: list[int],
        assume_checkpointed: bool = False,
    ) -> int:
        """Largest boundary `L <= hit` (in blocks) this class can resume from.

        Scanned right-to-left so the hit is cut as little as possible. 0 is
        always valid — a request starting from scratch needs no prior state.
        Identity when the class does not apply.

        `assume_checkpointed` asks the counterfactual instead: the answer this
        class would give if a checkpoint sat at every boundary. Whatever still
        cuts the hit under that assumption is a limit no amount of
        checkpointing can lift, which is what separates reuse worth arranging a
        checkpoint for from reuse that is simply gone.
        """
        ...

    def checkpoint(self, seq: Sequence, boundary_blocks: int, h: int) -> None:
        """Keep `seq`'s state as of `boundary_blocks`, filed under hash `h`.

        Called only at a ladder position `BlockManager` has already vetted
        against `successor_room`. Best-effort: a class out of room keeps
        nothing, and the hit it later declines is the only consequence.
        """
        ...


class StateCheckpointCache(StateCache, Protocol):
    """State cache lifecycle owned by BlockManager."""

    def unindex(self, h: int) -> None: ...

    def clear_index(self) -> None: ...

    def checkpoint_fates(self) -> dict[str, int]: ...
