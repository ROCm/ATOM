# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The seam between ATOM and whatever manages its paged KV blocks.

Two interfaces, one per process that touches the pool:

`KVBlockPool`   the scheduler's side. Which block to write next, which blocks
                hold which content, when content is destroyed. `BlockManager`
                calls nothing else.
`KVPoolMemory`  the model runner's side. Where the bytes of the pool live,
                and when the device has finished writing them.

ATOM's own implementation (`block_pool.BlockPool` over a tensor this process
allocates) is one backend. Another keeps both in an external process, so the
memory and its management can outlive, and be shared between, engine instances
(`umbp_block_pool`). `BlockManager` and `ModelRunner` only ever see the
interfaces; `ATOM_KV_POOL_BACKEND` picks the backend.
"""

from __future__ import annotations

import array
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from atom.model_engine.kv_block import Block
from atom.utils import envs

if TYPE_CHECKING:
    import torch

    from atom.model_engine.block_pool import BlockRetirement

LOCAL_BACKEND = "local"
UMBP_BACKEND = "umbp"


class KVBlockPool(ABC):
    """Paged KV blocks with ref counts and a content-addressed index.

    A block is free or held. A held block has `ref_count` holders. A free block
    may still carry content reachable by its hash until the pool hands it out
    again, so `allocate` — not `free` — is the eviction event, and `on_evict`
    fires there.

    Implementations expose these attributes:

      `num_blocks`      blocks currently in the pool
      `max_blocks`      how far `extend` may grow it
      `index_version`   bumped when a hash leaves the index or moves to another
                        block id; ids resolved at version v hold while it is v
      `blocks_evicted`, `blocks_retired`   content destroyed, by cause
    """

    num_blocks: int
    max_blocks: int
    index_version: int
    blocks_evicted: int
    blocks_retired: int

    #: Whether content published by another pool can become resident here
    #: (`fetch`). `BlockManager` skips the work of asking when it cannot.
    shares_content: bool = False

    # ------------------------------- counts -------------------------------- #
    @property
    @abstractmethod
    def num_free(self) -> int: ...

    @property
    @abstractmethod
    def num_used(self) -> int: ...

    @property
    @abstractmethod
    def num_indexed(self) -> int:
        """Blocks reachable by content hash, held or merely not yet overwritten."""

    @property
    @abstractmethod
    def num_reusable_free(self) -> int:
        """Free blocks still holding content a prefix hit could claim."""

    @abstractmethod
    def has_free(self, n: int) -> bool: ...

    @abstractmethod
    def is_used(self, block_id: int) -> bool: ...

    @abstractmethod
    def block(self, block_id: int) -> Block: ...

    @abstractmethod
    def eviction_stats(self) -> dict[str, int]: ...

    # ------------------------------- index --------------------------------- #
    @abstractmethod
    def lookup(self, h: int) -> int:
        """Block id indexed under content hash `h`, or -1."""

    @property
    @abstractmethod
    def index_get(self) -> Callable[[int], int | None]:
        """A `dict.get`-like lookup of one hash: block id or None. Read it per
        walk, not once per pool."""

    @abstractmethod
    def count_used(self, block_ids: list[int], n: int) -> int:
        """How many of the first `n` of `block_ids` (distinct) are held."""

    @abstractmethod
    def publish(self, block_id: int, h: int, token_ids: array.array) -> None:
        """Index `block_id` under the content hash of the tokens it now holds."""

    @abstractmethod
    def clear_index(self) -> None:
        """Drop every content-hash entry, keeping blocks live sequences hold."""

    # ---------------------------- allocation ------------------------------- #
    @abstractmethod
    def pop(self) -> int:
        """A free block id for fresh content; raises when none is free."""

    def prepare(self, count: int) -> None:
        """`count` calls to `pop` are about to follow.

        A hint, for a backend where getting blocks one at a time is a round
        trip each. It changes nothing a caller can observe.
        """

    @abstractmethod
    def allocate(self, block_id: int) -> Block:
        """Take `block_id` (from `pop`) for fresh content, evicting what it held."""

    @abstractmethod
    def claim(self, block_id: int) -> Block:
        """Take a share of `block_id` for the content it already holds."""

    @abstractmethod
    def claim_many(self, block_ids: Iterable[int]) -> None:
        """`claim` each of `block_ids`, in order."""

    @abstractmethod
    def free(self, block_id: int) -> None:
        """Drop one share of `block_id`."""

    # ------------------------------ raw units ------------------------------ #
    @abstractmethod
    def reserve_units(self, count: int, owner: Hashable) -> list[int] | None: ...

    @abstractmethod
    def release_units(self, unit_ids: Iterable[int], owner: Hashable) -> None: ...

    @abstractmethod
    def rekey_units(
        self, unit_ids: Iterable[int], old_owner: Hashable, new_owner: Hashable
    ) -> None: ...

    # ------------------------------ resizing ------------------------------- #
    @abstractmethod
    def extend(self, count: int) -> int:
        """Grow the pool by up to `count` blocks; returns how many it took."""

    @abstractmethod
    def retire_top(self) -> BlockRetirement | None:
        """Take the highest block id out of the pool, or None if it cannot."""

    # ------------------------------- sharing ------------------------------- #
    def fetch(
        self, hashes: Sequence[int], tokens_of: Callable[[int], array.array]
    ) -> int:
        """Make the longest leading run of `hashes` resident in this pool.

        `hashes` is a chain no local block is indexed under. A backend that
        shares content copies what other pools hold into free blocks of this
        one and indexes them, so the next `lookup` hits; the bytes are local
        before this returns. `tokens_of(i)` is the tokens `hashes[i]` stands
        for. Returns how many leading hashes are now resident.
        """
        return 0

    def flush(self) -> None:
        """Push any deferred bookkeeping to the backend. Cheap when idle."""

    def close(self) -> None:
        """Release what the pool holds outside this process."""


class KVPoolMemory(ABC):
    """Where the bytes of the paged pool come from."""

    #: Whether `allocate` shows up in this process's torch allocator statistics.
    #: `ModelRunner` checks its sizing against those only when it does.
    counted_by_torch: bool = True

    @abstractmethod
    def allocate(self, nbytes: int, device: torch.device) -> torch.Tensor:
        """A zero-filled `uint8` buffer of `nbytes` on `device`."""

    def bind_layout(
        self, buf: torch.Tensor, block_regions: Sequence[Any], num_blocks: int
    ) -> None:
        """Declare how `buf` divides into blocks.

        `block_regions` are the pool's `KVTransferRegion`s: block `i` of a
        region is `unit_bytes` at `base_addr + i * unit_bytes`. A backend that
        moves blocks itself needs this; one that does not ignores it.
        """

    def writes_issued(self) -> None:
        """Every write of one forward into the pool has been issued.

        Called by the runner on the thread and stream it issues them on, once
        per forward. Issued is not finished: the forward returns before the
        device is done. A backend that lets another process read or fill
        blocks needs to know when they are, and this is where it can find out
        (a device event recorded here completes after every earlier write).
        """

    def close(self) -> None:
        """Release the buffer."""


class LocalPoolMemory(KVPoolMemory):
    """The pool is a tensor this process allocates."""

    def allocate(self, nbytes: int, device: torch.device) -> torch.Tensor:
        import torch

        return torch.zeros(nbytes, dtype=torch.uint8, device=device)


def kv_pool_backend() -> str:
    backend = envs.ATOM_KV_POOL_BACKEND.strip().lower()
    if backend not in (LOCAL_BACKEND, UMBP_BACKEND):
        raise ValueError(
            f"unknown ATOM_KV_POOL_BACKEND {backend!r}; "
            f"expected {LOCAL_BACKEND!r} or {UMBP_BACKEND!r}"
        )
    return backend


def make_block_pool(
    config,
    num_blocks: int,
    *,
    on_evict: Callable[[int], None] | None = None,
    on_remap: Callable[[int], None] | None = None,
    cache_policy: str = "lru",
    protected_ratio: float = 0.5,
) -> KVBlockPool:
    """The scheduler-side pool for the configured backend."""
    if kv_pool_backend() == UMBP_BACKEND:
        from atom.model_engine.umbp_block_pool import make_umbp_block_pool

        return make_umbp_block_pool(
            config, num_blocks, on_evict=on_evict, on_remap=on_remap
        )
    from atom.model_engine.block_pool import BlockPool

    return BlockPool(
        num_blocks,
        on_evict=on_evict,
        cache_policy=cache_policy,
        protected_ratio=protected_ratio,
        on_remap=on_remap,
    )


def make_pool_memory(config, rank: int = 0) -> KVPoolMemory:
    """The runner-side memory provider for the configured backend."""
    if kv_pool_backend() == UMBP_BACKEND:
        from atom.model_engine.umbp_block_pool import make_umbp_pool_memory

        return make_umbp_pool_memory(config, rank)
    return LocalPoolMemory()
