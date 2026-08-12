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

    def applies(self, seq: Sequence) -> bool:
        """Whether this class gates or checkpoints anything for `seq`."""
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
