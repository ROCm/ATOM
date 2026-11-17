# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The paged KV pool, with the part MORI-UMBP manages shared across ranks.

`UmbpBlockPool` is what the `umbp_l1` connector hands the scheduler: this
process's own `BlockPool` for block ids `[0, P)` and a `UmbpL1Pool` for
`[P, N)`, where `P = split_point(N, share_ratio)` -- 0 at a ratio of 1, when
UMBP manages every block. Placement is private first; a private block whose
content is about to be overwritten is demoted into the UMBP part rather than
dropped, so other ranks can still fetch it. Kernels see one arena either way;
only who manages a block differs.

`UmbpL1Pool` is the scheduler's view of the UMBP part. Every change to it is
one this process asked for, so the view here is exact without a notification
channel:

  * hold / release / publish are applied locally at once and sent to UMBP in
    order, in a batch, before the next call that needs a decision from it;
  * `pop` (which block next, what gets evicted) and `fetch` (what other pools
    hold) are decided by UMBP.

UMBP owns the free list, eviction and the content index of that part. Pools of
one group share the index, so a prefix another engine instance (another DP
rank) already computed is copied into this rank's pool on demand and then read
locally.

`UmbpPoolMemory` is the model runner's side: the pool is this process's own
allocation, and UMBP adopts the `[P, N)` part of it through its IPC handle. It
also tells UMBP how far the GPU has got.

That last part is what makes sharing safe. The scheduler publishes a block when
the forward that fills it is issued, and frees one while a forward may still be
writing it; a forward returns before the device is done. Within this engine
that is fine, because its readers queue behind its writers on the device. UMBP
copies from another process and is not in that queue. So every batch of
updates carries a timestamp, the runner records a device event after each
forward, and when the event completes it tells UMBP that everything stamped up
to then has landed. UMBP reads or fills a block from outside only after that.

Built by the `umbp_l1` KV connector (`connector.py`), which reads its settings
off `kv_transfer_config` (`settings.py`). Needs the `mori_umbp_l1` client
module and a running UMBP standalone server.
"""

from __future__ import annotations

import array
import hashlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Sequence
from itertools import islice, pairwise
from typing import Any

from atom.kv_transfer.l1.umbp_l1.settings import UmbpL1Settings
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.kv_block import Block
from atom.model_engine.kv_block_pool import KVBlockPool, KVPoolMemory
from atom.utils import envs

logger = logging.getLogger("atom")

_KEY_BYTES = 8


def _load_client_module():
    try:
        import mori_umbp_l1
    except ImportError as exc:  # pragma: no cover - deployment error
        raise RuntimeError(
            "the umbp_l1 connector needs the mori_umbp_l1 module on "
            "PYTHONPATH (built from mori with -DBUILD_UMBP_L1_PYTHON=ON)"
        ) from exc
    return mori_umbp_l1


def _connect(settings: UmbpL1Settings):
    module = _load_client_module()
    return module, module.L1Client(settings.address)


def pool_id_for(config, settings: UmbpL1Settings, rank: int = 0) -> str:
    """The name both processes of one engine rank know their pool by."""
    parallel = getattr(config, "parallel_config", None)
    dp_rank = getattr(parallel, "data_parallel_rank", 0) or 0
    return f"{settings.namespace}:dp{dp_rank}:tp{rank}"


def share_group_for(config, settings: UmbpL1Settings) -> str:
    """Pools exchange blocks only inside one group.

    The id has to change whenever a block's bytes would mean something else:
    another model, another KV or index dtype, another block size. Built from
    plain values only, so every rank of an engine derives the same one. UMBP
    additionally refuses a pool whose per-region block sizes differ from its
    group's.
    """
    if settings.group:
        return settings.group
    spec = getattr(config, "speculative_config", None)
    facts = {
        "model": str(getattr(config, "model", "")),
        "kv_cache_dtype": str(getattr(config, "kv_cache_dtype", "")),
        "index_cache_dtype": str(getattr(config, "index_cache_dtype", "")),
        "block_size": int(getattr(config, "kv_cache_block_size", 0) or 0),
        "online_quant": json.dumps(
            getattr(
                getattr(config, "quant_config", None), "online_quant_config_raw", None
            ),
            sort_keys=True,
            default=str,
        ),
        "draft": (
            None
            if spec is None
            else [
                str(spec.get(k) if isinstance(spec, dict) else getattr(spec, k, None))
                for k in ("method", "model")
            ]
        ),
    }
    digest = hashlib.sha1(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    return f"atom-{digest[:16]}"


def _key(h: int) -> bytes:
    return h.to_bytes(_KEY_BYTES, "little")


def _hash(key: bytes) -> int:
    return int.from_bytes(key, "little")


class UmbpL1Pool(KVBlockPool):
    """`KVBlockPool` over the blocks UMBP manages (the `[P, N)` part)."""

    shares_content = True

    def __init__(
        self,
        client,
        ops,
        pool_id: str,
        num_blocks: int,
        on_evict: Callable[[int], None] | None = None,
        on_remap: Callable[[int], None] | None = None,
        grant_batch: int = 32,
    ):
        self._client = client
        self._op_hold, self._op_release, self._op_publish = ops
        self.pool_id = pool_id
        opened = client.open(pool_id)
        if opened != num_blocks:
            raise ValueError(
                f"UMBP pool {pool_id} has {opened} blocks but the scheduler "
                f"addresses {num_blocks}"
            )
        self.num_blocks = num_blocks
        self.max_blocks = num_blocks
        self._on_evict = on_evict
        self._on_remap = on_remap
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self._hash_to_block_id: dict[int, int] = {}
        self.index_version = 0
        self._used: set[int] = set()
        # Free blocks whose content is still indexed, i.e. what a prefix hit
        # could claim and what the next grant may destroy.
        self._cached_free = 0
        # Blocks UMBP has granted for fresh content: `_stash` not yet handed to
        # the manager, `_granted` handed out by `pop` and awaiting `allocate`.
        self._grant_batch = max(1, grant_batch)
        self._stash: deque[int] = deque()
        self._granted: set[int] = set()
        # Hold / release / publish transitions UMBP has not been told yet.
        self._op_kinds: list[int] = []
        self._op_slots: list[int] = []
        self._op_keys: list[bytes] = []
        self.blocks_evicted = 0
        self.blocks_retired = 0
        self.blocks_fetched = 0
        self.bytes_fetched = 0
        self.fetch_seconds = 0.0
        # owner -> (future, block ids, hashes, tokens_of, start) of a copy
        # into held blocks (`fetch_async`).
        self._fetches: dict = {}
        self._executor = None
        self.async_fetch = self._probe_fetch_into()

    # ------------------------------- counts -------------------------------- #
    @property
    def num_free(self) -> int:
        return self.num_blocks - len(self._used)

    @property
    def num_used(self) -> int:
        return len(self._used)

    @property
    def num_indexed(self) -> int:
        return len(self._hash_to_block_id)

    @property
    def num_reusable_free(self) -> int:
        return self._cached_free

    def has_free(self, n: int) -> bool:
        return self.num_free >= n

    def is_used(self, block_id: int) -> bool:
        return block_id in self._used

    def block(self, block_id: int) -> Block:
        return self.blocks[block_id]

    def eviction_stats(self) -> dict[str, int]:
        return {
            "blocks_evicted": self.blocks_evicted,
            "blocks_retired": self.blocks_retired,
            "blocks_total": self.num_blocks,
            "blocks_used": self.num_used,
            "blocks_free": self.num_free,
            "blocks_free_reusable": self.num_reusable_free,
            "blocks_indexed": self.num_indexed,
            "blocks_fetched": self.blocks_fetched,
        }

    # ------------------------------- index --------------------------------- #
    def lookup(self, h: int) -> int:
        return self._hash_to_block_id.get(h, -1)

    @property
    def index_get(self):
        return self._hash_to_block_id.get

    def count_used(self, block_ids: list[int], n: int) -> int:
        return len(self._used.intersection(islice(block_ids, n)))

    def _indexed(self, block_id: int) -> bool:
        h = self.blocks[block_id].hash
        return h != -1 and self._hash_to_block_id.get(h) == block_id

    def publish(self, block_id: int, h: int, token_ids: array.array) -> None:
        block = self.blocks[block_id]
        index = self._hash_to_block_id
        old = block.hash
        if old not in (-1, h) and index.get(old) == block_id:
            del index[old]
            self.index_version += 1
            if self._on_evict is not None:
                self._on_evict(old)
        block.update(h, token_ids)
        prev = index.get(h)
        if prev == block_id:
            return
        index[h] = block_id
        if prev is not None:
            # The older copy stops being reachable; UMBP frees it likewise.
            if prev not in self._used:
                self._cached_free -= 1
            self.index_version += 1
            if self._on_remap is not None:
                self._on_remap(h)
        self._defer(self._op_publish, block_id, _key(h))

    def clear_index(self) -> None:
        self.flush()
        self._client.clear(self.pool_id)
        self._hash_to_block_id.clear()
        self.index_version += 1
        for block in self.blocks:
            if block.ref_count == 0:
                block.hash = -1
                block.token_ids = array.array("i")
        self._cached_free = 0

    def _forget(self, block_id: int) -> int:
        """UMBP handed out free block `block_id`: drop what it held.

        Returns the hash whose content that destroyed, or -1.
        """
        block = self.blocks[block_id]
        gone = -1
        if self._indexed(block_id):
            gone = block.hash
            del self._hash_to_block_id[gone]
            self.index_version += 1
            self.blocks_evicted += 1
            self._cached_free -= 1
            if self._on_evict is not None:
                self._on_evict(gone)
        block.hash = -1
        block.token_ids = array.array("i")
        return gone

    def _check_evicted(self, mine: list[int], theirs: list[bytes]) -> None:
        if sorted(mine) != sorted(_hash(k) for k in theirs):
            raise RuntimeError(
                f"UMBP pool {self.pool_id}: local view diverged "
                f"(evicted here {len(mine)}, by UMBP {len(theirs)})"
            )

    # --------------------------- deferred updates -------------------------- #
    def _defer(self, kind: int, block_id: int, key: bytes | None = None) -> None:
        self._op_kinds.append(kind)
        self._op_slots.append(block_id)
        if key is not None:
            self._op_keys.append(key)

    def flush(self) -> None:
        if not self._op_kinds:
            return
        kinds, slots, keys = self._op_kinds, self._op_slots, self._op_keys
        self._op_kinds, self._op_slots, self._op_keys = [], [], []
        # Stamped now, which is after every forward that could still be writing
        # these blocks was issued. UMBP leaves them alone until the runner's
        # device has caught up with this instant (`UmbpPoolMemory`).
        self._client.apply(self.pool_id, kinds, slots, keys, time.monotonic_ns())

    # ---------------------------- allocation ------------------------------- #
    def _refill(self, need: int) -> bool:
        """Get blocks from UMBP: `need` of them even if that evicts content,
        and up to a batch more as long as they cost nothing."""
        # Blocks already granted are free here but held at UMBP.
        available = self.num_free - len(self._stash) - len(self._granted)
        need = min(need, available)
        if need <= 0:
            return False
        self.flush()
        slots, evicted = self._client.allocate(
            self.pool_id, min(max(need, self._grant_batch), available), need
        )
        gone = [h for h in map(self._forget, slots) if h != -1]
        self._check_evicted(gone, evicted)
        self._stash.extend(slots)
        return bool(slots)

    def prepare(self, count: int) -> None:
        short = count - len(self._stash)
        if short > 0:
            self._refill(short)

    def pop(self) -> int:
        if not self._stash and not self._refill(1):
            raise AssertionError("No free blocks available")
        block_id = self._stash.popleft()
        self._granted.add(block_id)
        return block_id

    def allocate(self, block_id: int) -> Block:
        if block_id not in self._granted:
            raise NotImplementedError(
                "a UMBP pool allocates only the block ids `pop` handed out"
            )
        self._granted.remove(block_id)
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self._used.add(block_id)
        return block

    def claim(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        if block_id in self._used:
            block.ref_count += 1
            return block
        assert block.ref_count == 0
        assert block_id not in self._granted and block_id not in self._stash
        block.ref_count = 1
        self._used.add(block_id)
        if self._indexed(block_id):
            self._cached_free -= 1
        self._defer(self._op_hold, block_id)
        return block

    def claim_many(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            self.claim(block_id)

    def free(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count:
            return
        self._used.remove(block_id)
        if self._indexed(block_id):
            self._cached_free += 1
        self._defer(self._op_release, block_id)

    # ------------------------------ raw units ------------------------------ #
    def reserve_units(self, count: int, owner: Hashable) -> list[int] | None:
        raise NotImplementedError("raw PAGE units are not supported by the UMBP pool")

    def release_units(self, unit_ids: Iterable[int], owner: Hashable) -> None:
        raise NotImplementedError("raw PAGE units are not supported by the UMBP pool")

    def rekey_units(
        self, unit_ids: Iterable[int], old_owner: Hashable, new_owner: Hashable
    ) -> None:
        raise NotImplementedError("raw PAGE units are not supported by the UMBP pool")

    # ------------------------------ resizing ------------------------------- #
    def extend(self, count: int) -> int:
        return 0  # the pool is created at its full size

    def retire_top(self):
        return None

    # ------------------------------- sharing ------------------------------- #
    def fetch(
        self, hashes: Sequence[int], tokens_of: Callable[[int], array.array]
    ) -> int:
        if not hashes:
            return 0
        self.flush()
        started = time.perf_counter()
        slots, evicted, copied, copied_bytes, *_ = self._client.fetch(
            self.pool_id, [_key(h) for h in hashes]
        )
        index = self._hash_to_block_id
        # What UMBP destroyed to make room comes first, and on its own: a
        # fetch that stops early (a copy or a lower tier failed) has emptied
        # destinations it then did not fill, so they are not among `slots`.
        for key in evicted:
            block_id = index.get(_hash(key))
            if block_id is None or block_id in self._used:
                raise RuntimeError(
                    f"UMBP pool {self.pool_id}: local view diverged "
                    f"(UMBP evicted a block this view does not hold free)"
                )
            self._forget(block_id)
        for i, block_id in enumerate(slots):
            h = hashes[i]
            current = index.get(h)
            if current == block_id:
                continue
            if current is not None or block_id in self._used or self._indexed(block_id):
                raise RuntimeError(
                    f"UMBP pool {self.pool_id}: local view diverged on fetch"
                )
            block = self.blocks[block_id]
            block.hash = h
            block.token_ids = tokens_of(i)
            index[h] = block_id
            self._cached_free += 1
        elapsed = time.perf_counter() - started
        self.blocks_fetched += copied
        self.bytes_fetched += copied_bytes
        self.fetch_seconds += elapsed
        if copied:
            logger.info(
                "UMBP pool %s: fetched %d of %d blocks from peer pools "
                "(%.1f MiB in %.1f ms)",
                self.pool_id,
                copied,
                len(hashes),
                copied_bytes / (1 << 20),
                elapsed * 1e3,
            )
        return len(slots)

    # --------------------------- background fetch --------------------------- #
    # `fetch` copies on the scheduler's thread, and under DP-attention every
    # rank waits for the slowest one's step. Copying into blocks this pool
    # already holds instead touches nothing UMBP decides, so the copy can run
    # while the scheduler keeps allocating and the device keeps computing:
    # the blocks are granted here, filled by UMBP, then published and released
    # here like any block a forward filled.

    def _probe_fetch_into(self) -> bool:
        """Whether the server copies into held slots (`dst_slots`)."""
        try:
            self._client.fetch(self.pool_id, [], dst_slots=[], probe_only=True)
        except TypeError:
            return False
        return True

    def fetch_async(
        self,
        owner: Hashable,
        hashes: Sequence[int],
        tokens_of: Callable[[int], array.array],
    ) -> bool:
        """Start copying in the leading run of `hashes` other pools hold.

        True while a copy for `owner` is in flight (`fetch_pending`); its
        blocks are indexed by `poll_fetches` once it lands.
        """
        if owner in self._fetches:
            return True
        if not hashes or not self.async_fetch:
            return False
        keys = [_key(h) for h in hashes]
        *_, available = self._client.fetch(self.pool_id, keys, probe_only=True)
        room = self.num_free - len(self._stash) - len(self._granted)
        n = min(available, room)
        if n <= 0:
            return False
        self.prepare(n)
        n = min(n, len(self._stash))
        if n <= 0:
            return False
        ids = [self.pop() for _ in range(n)]
        for block_id in ids:
            # Held at UMBP since it granted them; nothing to tell it.
            self.allocate(block_id)
        future = self._fetch_executor().submit(
            self._client.fetch, self.pool_id, keys[:n], dst_slots=ids
        )
        self._fetches[owner] = (
            future,
            ids,
            list(hashes[:n]),
            tokens_of,
            time.perf_counter(),
        )
        return True

    def fetch_pending(self, owner: Hashable) -> bool:
        return owner in self._fetches

    def poll_fetches(self) -> None:
        """Index what finished copies brought in; release the rest."""
        if not self._fetches:
            return
        done = [o for o, entry in self._fetches.items() if entry[0].done()]
        for owner in done:
            future, ids, hashes, tokens_of, started = self._fetches.pop(owner)
            try:
                slots, _, copied, copied_bytes, *_ = future.result()
            except Exception:  # a failed copy is a miss
                logger.exception("UMBP pool %s: background fetch failed", self.pool_id)
                slots, copied, copied_bytes = [], 0, 0
            landed = len(slots)
            index = self._hash_to_block_id
            published = 0
            for i, block_id in enumerate(ids):
                h = hashes[i]
                # Computed here meanwhile: keep that copy, drop this one.
                if i < landed and h not in index:
                    self.publish(block_id, h, tokens_of(i))
                    published += 1
                self.free(block_id)
            if published:
                # A memoised walk stops at a miss it rereads; make it reread.
                self.index_version += 1
            elapsed = time.perf_counter() - started
            self.blocks_fetched += copied
            self.bytes_fetched += copied_bytes
            self.fetch_seconds += elapsed
            if copied:
                logger.info(
                    "UMBP pool %s: fetched %d of %d blocks from peer pools in the "
                    "background (%.1f MiB, %.1f ms until indexed)",
                    self.pool_id,
                    copied,
                    len(ids),
                    copied_bytes / (1 << 20),
                    elapsed * 1e3,
                )

    def _fetch_executor(self):
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            # The calls release the GIL; a few at once overlap their RPCs and
            # their copies, which UMBP runs outside its lock.
            self._executor = ThreadPoolExecutor(4, thread_name_prefix="umbp-fetch")
        return self._executor

    def close(self) -> None:
        try:
            self.flush()
        except Exception:  # noqa: BLE001 - the server may already be gone
            logger.debug("UMBP pool %s: flush on close failed", self.pool_id)
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)


class UmbpPoolMemory(KVPoolMemory):
    """The pool is this process's allocation; UMBP adopts its `[P, N)` part.

    Blocks `[0, P)` stay private, `[P, N)` are UMBP's slots `[0, N - P)`
    (`split_point`; at a share ratio of 1 that is the whole pool). UMBP maps
    the allocation through its IPC handle and manages, shares and copies those
    blocks, but never allocates, zeroes or frees the memory.
    """

    counted_by_torch = True
    shares_blocks = True

    def __init__(self, client, pool_id: str, group: str, ratio: float):
        self._client = client
        self.pool_id = pool_id
        self.group = group
        self.ratio = ratio
        self._device = None
        # Where UMBP's part was adopted from; None until `bind_layout`.
        self._mapped_base: int | None = None
        # (when it was recorded, device event) per forward, oldest first.
        self._issued: deque = deque()
        self._wake = threading.Event()
        self._watcher: threading.Thread | None = None

    def writes_issued(self) -> None:
        if self._mapped_base is None:
            return
        import torch

        # Read the clock first: whatever the scheduler stamped before this
        # instant was decided before the event below was recorded, so the
        # event completing proves those writes landed.
        stamp = time.monotonic_ns()
        event = torch.cuda.Event()
        event.record()
        self._issued.append((stamp, event))
        if self._watcher is None:
            self._watcher = threading.Thread(
                target=self._report_progress,
                name=f"umbp-l1-commit-{self.pool_id}",
                daemon=True,
            )
            self._watcher.start()
        self._wake.set()

    def _report_progress(self) -> None:
        """Tell UMBP each time the device finishes another forward's writes."""
        issued = self._issued
        failed = False
        while True:
            if not issued:
                self._wake.clear()
                if not issued:
                    self._wake.wait()
                continue
            stamp, event = issued[0]
            # Polled, not waited on: a blocking wait here could hold up the
            # thread that issues the next forward.
            if not event.query():
                time.sleep(0.0005)
                continue
            issued.popleft()
            while issued and issued[0][1].query():
                stamp = issued.popleft()[0]
            try:
                self._client.commit(self.pool_id, stamp)
                failed = False
            except Exception:  # sharing degrades, serving goes on
                if not failed:
                    logger.exception(
                        "UMBP pool %s: could not report device progress; other "
                        "pools will not see this rank's new blocks",
                        self.pool_id,
                    )
                failed = True

    def allocate(self, nbytes: int, device) -> Any:
        import torch

        if self._device is not None:
            raise NotImplementedError(
                f"UMBP pool {self.pool_id}: the runner asked for a second paged pool"
            )
        self._device = device
        return torch.zeros(nbytes, dtype=torch.uint8, device=device)

    def bind_layout(self, buf, block_regions: Sequence[Any], num_blocks: int) -> None:
        from atom.mono.runtime import hip_ipc

        offset = split_point(num_blocks, self.ratio)
        slots = num_blocks - offset
        if slots <= 0 or not block_regions:
            return
        base = buf.data_ptr()
        layout = []
        for region in block_regions:
            if region.reverse_indexed:
                raise NotImplementedError("reverse-indexed PAGE regions")
            # UMBP's blocks of each region start at block `offset`.
            layout.append(
                (
                    region.base_addr - base + offset * region.unit_bytes,
                    region.unit_bytes,
                )
            )
        # Blocks are copied one region at a time, so a block's bytes must not
        # be reachable through two regions: copying it would then also write
        # part of a neighbouring block.
        ordered = sorted(layout)
        overlaps = [(a, b) for a, b in pairwise(ordered) if a[0] + slots * a[1] > b[0]]
        end = ordered[-1][0] + slots * ordered[-1][1]
        if overlaps or ordered[0][0] < 0 or end > buf.numel():
            raise NotImplementedError(
                f"UMBP pool {self.pool_id}: PAGE regions overlap or leave the "
                f"pool ({len(overlaps)} overlapping pairs, pool {buf.numel()} B, "
                f"last region ends at {end})"
            )
        allocation = hip_ipc.get_allocation_base(base)
        handle = hip_ipc.get_ipc_handle(allocation)
        device = buf.device.index if buf.device.index is not None else 0
        self._client.attach(
            self.pool_id,
            self.group,
            device,
            buf.numel(),
            external_handle=handle,
            external_offset=base - allocation,
        )
        self._client.bind_layout(self.pool_id, slots, layout)
        self._mapped_base = base
        logger.info(
            "UMBP pool %s: adopted %d of %d blocks (ratio %.2f) of this process's "
            "%.2f GiB pool on %s",
            self.pool_id,
            slots,
            num_blocks,
            self.ratio,
            buf.numel() / (1 << 30),
            buf.device,
        )


def make_umbp_l1_pool(
    config,
    settings: UmbpL1Settings,
    num_blocks: int,
    *,
    on_evict: Callable[[int], None] | None = None,
    on_remap: Callable[[int], None] | None = None,
) -> UmbpL1Pool:
    module, client = _connect(settings)
    pool = UmbpL1Pool(
        client,
        (module.OP_HOLD, module.OP_RELEASE, module.OP_PUBLISH),
        pool_id_for(config, settings),
        num_blocks,
        on_evict=on_evict,
        on_remap=on_remap,
    )
    pool.shares_content = settings.fetch
    return pool


def make_umbp_pool_memory(
    config, settings: UmbpL1Settings, rank: int = 0
) -> UmbpPoolMemory:
    if rank != 0:
        raise NotImplementedError(
            "the UMBP pool backs one runner per engine rank; tensor-parallel "
            "attention within a rank is not supported yet"
        )
    _, client = _connect(settings)
    pool_id = pool_id_for(config, settings, rank)
    group = share_group_for(config, settings)
    return UmbpPoolMemory(client, pool_id, group, settings.share_ratio)


def split_point(num_blocks: int, ratio: float) -> int:
    """The first block id UMBP manages: blocks below it are private."""
    managed = min(num_blocks, max(0, round(num_blocks * ratio)))
    return num_blocks - managed


class UmbpBlockPool(KVBlockPool):
    """Private `BlockPool` below `offset`, `UmbpL1Pool` from `offset` on.

    Always the pool the `umbp_l1` connector provides; at a share ratio of 1
    `offset` is 0 and every call goes to the UMBP part.
    """

    def __init__(
        self,
        private: BlockPool,
        umbp: UmbpL1Pool,
        offset: int,
    ):
        self.private = private
        self.umbp = umbp
        self.offset = offset
        self.num_blocks = offset + umbp.num_blocks
        self.max_blocks = self.num_blocks
        # Private -> UMBP copies the batch being built has to run first.
        self._copies: list[tuple[int, int]] = []
        # Demoted UMBP slots whose copy is not yet in a taken batch, and
        # those whose batch was taken and awaits its postprocess.
        self._unissued: list[tuple[int, int, array.array]] = []
        self._issued: list[tuple[int, int, array.array]] = []
        self._version = 0
        self.blocks_demoted = 0
        self.demotions_dropped = 0
        self._logged_demoted = 0

    # --------------------------- routing helpers --------------------------- #
    def _is_umbp(self, block_id: int) -> bool:
        return block_id >= self.offset

    def holds(self, h: int) -> bool:
        """Whether content `h` is still in the pool, a demotion in flight
        included: losing a private copy then is a move, not an eviction."""
        if self.private.lookup(h) != -1 or self.umbp.lookup(h) != -1:
            return True
        return any(e[1] == h for e in self._unissued) or any(
            e[1] == h for e in self._issued
        )

    @property
    def shares_content(self) -> bool:
        return self.umbp.shares_content

    @property
    def async_fetch(self) -> bool:
        return getattr(self.umbp, "async_fetch", False)

    # ------------------------------- counts -------------------------------- #
    @property
    def num_free(self) -> int:
        return self.private.num_free + self.umbp.num_free

    @property
    def num_used(self) -> int:
        return self.private.num_used + self.umbp.num_used

    @property
    def num_indexed(self) -> int:
        return self.private.num_indexed + self.umbp.num_indexed

    @property
    def num_reusable_free(self) -> int:
        return self.private.num_reusable_free + self.umbp.num_reusable_free

    @property
    def index_version(self) -> int:
        return self.private.index_version + self.umbp.index_version + self._version

    @property
    def blocks_evicted(self) -> int:
        return self.private.blocks_evicted + self.umbp.blocks_evicted

    @property
    def blocks_retired(self) -> int:
        return self.private.blocks_retired + self.umbp.blocks_retired

    def has_free(self, n: int) -> bool:
        return self.num_free >= n

    def is_used(self, block_id: int) -> bool:
        if self._is_umbp(block_id):
            return self.umbp.is_used(block_id - self.offset)
        return self.private.is_used(block_id)

    def block(self, block_id: int) -> Block:
        if self._is_umbp(block_id):
            return self.umbp.block(block_id - self.offset)
        return self.private.block(block_id)

    def eviction_stats(self) -> dict[str, int]:
        """The whole pool's figures under the names every pool reports (the
        stats lines read them), then each part's under a prefix."""
        private = self.private.eviction_stats()
        umbp = self.umbp.eviction_stats()
        stats = {
            k: private.get(k, 0) + umbp.get(k, 0)
            for k in (
                "blocks_evicted",
                "blocks_retired",
                "blocks_total",
                "blocks_used",
                "blocks_free",
                "blocks_free_reusable",
                "blocks_indexed",
            )
        }
        stats.update({f"private_{k}": v for k, v in private.items()})
        stats.update({f"umbp_{k}": v for k, v in umbp.items()})
        stats["blocks_fetched"] = umbp.get("blocks_fetched", 0)
        stats["blocks_demoted"] = self.blocks_demoted
        stats["demotions_dropped"] = self.demotions_dropped
        return stats

    # ------------------------------- index --------------------------------- #
    def lookup(self, h: int) -> int:
        # The private copy first: it is the hot one, and it is not shared.
        block_id = self.private.lookup(h)
        if block_id != -1:
            return block_id
        block_id = self.umbp.lookup(h)
        return block_id + self.offset if block_id != -1 else -1

    @property
    def index_get(self):
        get_private = self.private.index_get
        get_umbp = self.umbp.index_get
        offset = self.offset

        def get(h):
            block_id = get_private(h)
            if block_id is not None:
                return block_id
            block_id = get_umbp(h)
            return None if block_id is None else block_id + offset

        return get

    def count_used(self, block_ids: list[int], n: int) -> int:
        return sum(1 for block_id in block_ids[:n] if self.is_used(block_id))

    def publish(self, block_id: int, h: int, token_ids: array.array) -> None:
        if self._is_umbp(block_id):
            self.umbp.publish(block_id - self.offset, h, token_ids)
        else:
            self.private.publish(block_id, h, token_ids)

    def clear_index(self) -> None:
        self.private.clear_index()
        self.umbp.clear_index()
        self._version += 1

    # ---------------------------- allocation ------------------------------- #
    def prepare(self, count: int) -> None:
        # Private blocks need no preparing; ask the UMBP part only for what
        # the private part cannot cover.
        short = count - self.private.num_free
        if short > 0:
            self.umbp.prepare(short)

    def pop(self) -> int:
        if self.private.num_free:
            return self.private.pop()
        return self.umbp.pop() + self.offset

    def allocate(self, block_id: int) -> Block:
        if self._is_umbp(block_id):
            return self.umbp.allocate(block_id - self.offset)
        block = self.private.block(block_id)
        if block.hash != -1 and self.private.lookup(block.hash) == block_id:
            self._demote(block_id, block.hash, block.token_ids)
        return self.private.allocate(block_id)

    def _demote(self, block_id: int, h: int, token_ids: array.array) -> None:
        """Keep the content `block_id` is about to lose in a UMBP block."""
        if self.umbp.lookup(h) != -1:
            return  # already in the UMBP part
        if not self.umbp.num_free:
            self.demotions_dropped += 1
            return
        slot = self.umbp.pop()
        self.umbp.allocate(slot)
        self._copies.append((block_id, slot + self.offset))
        self._unissued.append((slot, h, array.array("i", token_ids)))
        self.blocks_demoted += 1

    def take_block_copies(self) -> tuple[tuple[int, int], ...]:
        """The copies the batch being built runs before its forward."""
        copies = tuple(self._copies)
        self._copies.clear()
        self._issued.extend(self._unissued)
        self._unissued.clear()
        return copies

    def settle(self) -> None:
        """Publish demoted blocks whose copy has been issued (postprocess)."""
        if not self._issued:
            return
        for slot, h, token_ids in self._issued:
            if self.lookup(h) == -1:
                self.umbp.publish(slot, h, token_ids)
            self.umbp.free(slot)
        self._issued.clear()
        self._version += 1
        if self.blocks_demoted // 1024 != self._logged_demoted // 1024:
            self._logged_demoted = self.blocks_demoted
            logger.info(
                "KV pool: %d blocks demoted into the UMBP part, %d dropped "
                "(UMBP part full of held blocks)",
                self.blocks_demoted,
                self.demotions_dropped,
            )

    def claim(self, block_id: int) -> Block:
        if self._is_umbp(block_id):
            return self.umbp.claim(block_id - self.offset)
        return self.private.claim(block_id)

    def claim_many(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            self.claim(block_id)

    def free(self, block_id: int) -> None:
        if self._is_umbp(block_id):
            self.umbp.free(block_id - self.offset)
        else:
            self.private.free(block_id)

    # ------------------------------ raw units ------------------------------ #
    # PAGE checkpoints stay in the private part; the UMBP part holds only
    # blocks a prefix hit can reuse.
    def reserve_units(self, count: int, owner: Hashable) -> list[int] | None:
        return self.private.reserve_units(count, owner)

    def release_units(self, unit_ids: Iterable[int], owner: Hashable) -> None:
        self.private.release_units(unit_ids, owner)

    def rekey_units(
        self, unit_ids: Iterable[int], old_owner: Hashable, new_owner: Hashable
    ) -> None:
        self.private.rekey_units(unit_ids, old_owner, new_owner)

    # ------------------------------ resizing ------------------------------- #
    def extend(self, count: int) -> int:
        return 0

    def retire_top(self):
        return None

    # ------------------------------- sharing ------------------------------- #
    def fetch(
        self, hashes: Sequence[int], tokens_of: Callable[[int], array.array]
    ) -> int:
        return self.umbp.fetch(hashes, tokens_of)

    def fetch_async(self, owner: Hashable, hashes, tokens_of) -> bool:
        return self.umbp.fetch_async(owner, hashes, tokens_of)

    def fetch_pending(self, owner: Hashable) -> bool:
        return self.umbp.fetch_pending(owner)

    def poll_fetches(self) -> None:
        self.umbp.poll_fetches()

    def flush(self) -> None:
        self.umbp.flush()

    def close(self) -> None:
        self.umbp.close()


def make_umbp_block_pool(
    config,
    settings: UmbpL1Settings,
    num_blocks: int,
    *,
    on_evict: Callable[[int], None] | None = None,
    on_remap: Callable[[int], None] | None = None,
    cache_policy: str = "lru",
    protected_ratio: float = 0.5,
) -> UmbpBlockPool:
    ratio = settings.share_ratio
    if envs.ATOM_PREFIX_CACHE_POLICY.strip().lower() != "lru":
        logger.warning(
            "ATOM_PREFIX_CACHE_POLICY=%s applies to the private blocks only: "
            "UMBP evicts its part least recently released first",
            envs.ATOM_PREFIX_CACHE_POLICY,
        )
    offset = split_point(num_blocks, ratio)
    pool: UmbpBlockPool | None = None

    # A hash leaves the pool only when neither part still holds it: a
    # demotion moves content, it does not evict it.
    def evicted(h: int) -> None:
        if on_evict is not None and (pool is None or not pool.holds(h)):
            on_evict(h)

    private = BlockPool(
        offset,
        on_evict=evicted,
        cache_policy=cache_policy,
        protected_ratio=protected_ratio,
        on_remap=on_remap,
    )
    umbp = make_umbp_l1_pool(
        config,
        settings,
        num_blocks - offset,
        on_evict=evicted,
        on_remap=on_remap,
    )
    pool = UmbpBlockPool(private, umbp, offset)
    logger.info(
        "KV pool: %d private blocks, %d managed by UMBP (share ratio %.2f)",
        offset,
        num_blocks - offset,
        ratio,
    )
    return pool
