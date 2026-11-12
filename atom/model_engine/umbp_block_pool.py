# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The paged KV pool kept by a UMBP standalone process.

UMBP owns the GPU memory, the free list, eviction and the content index; this
process maps the memory and asks UMBP which block to use. Pools of one group
share the index, so a prefix another engine instance (another DP rank) already
computed is copied into this rank's pool on demand and then read locally.

`UmbpBlockPool` is the scheduler's view. Every change to this pool is one this
process asked for, so the view here is exact without a notification channel:

  * hold / release / publish are applied locally at once and sent to UMBP in
    order, in a batch, before the next call that needs a decision from it;
  * `pop` (which block next, what gets evicted) and `fetch` (what other pools
    hold) are decided by UMBP.

`UmbpPoolMemory` is the model runner's side: it has UMBP allocate the pool on
this rank's GPU and maps it, and it tells UMBP how far the GPU has got.

That last part is what makes sharing safe. The scheduler publishes a block when
the forward that fills it is issued, and frees one while a forward may still be
writing it; a forward returns before the device is done. Within this engine
that is fine, because its readers queue behind its writers on the device. UMBP
copies from another process and is not in that queue. So every batch of
updates carries a timestamp, the runner records a device event after each
forward, and when the event completes it tells UMBP that everything stamped up
to then has landed. UMBP reads or fills a block from outside only after that.

Needs the `mori_umbp_l1` client module and a server at `ATOM_UMBP_L1_ADDRESS`.
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
            "ATOM_KV_POOL_BACKEND=umbp needs the mori_umbp_l1 module on "
            "PYTHONPATH (built from mori with -DBUILD_UMBP_L1_PYTHON=ON)"
        ) from exc
    return mori_umbp_l1


def _connect():
    address = envs.ATOM_UMBP_L1_ADDRESS
    if not address:
        raise RuntimeError(
            "ATOM_KV_POOL_BACKEND=umbp needs ATOM_UMBP_L1_ADDRESS, the gRPC "
            "address of the UMBP standalone server"
        )
    module = _load_client_module()
    return module, module.L1Client(address)


def pool_id_for(config, rank: int = 0) -> str:
    """The name both processes of one engine rank know their pool by."""
    parallel = getattr(config, "parallel_config", None)
    dp_rank = getattr(parallel, "data_parallel_rank", 0) or 0
    return f"{envs.ATOM_UMBP_L1_NAMESPACE}:dp{dp_rank}:tp{rank}"


def share_group_for(config) -> str:
    """Pools exchange blocks only inside one group.

    The id has to change whenever a block's bytes would mean something else:
    another model, another KV or index dtype, another block size. Built from
    plain values only, so every rank of an engine derives the same one. UMBP
    additionally refuses a pool whose per-region block sizes differ from its
    group's.
    """
    if envs.ATOM_UMBP_L1_GROUP:
        return envs.ATOM_UMBP_L1_GROUP
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


class UmbpBlockPool(KVBlockPool):
    """`KVBlockPool` over a pool UMBP manages."""

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
        slots, evicted, copied, copied_bytes, _ = self._client.fetch(
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

    def close(self) -> None:
        try:
            self.flush()
        except Exception:  # noqa: BLE001 - the server may already be gone
            logger.debug("UMBP pool %s: flush on close failed", self.pool_id)


class _DeviceBuffer:
    """A raw device pointer, in the form `torch.as_tensor` accepts."""

    def __init__(self, pointer: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 2,
        }


class UmbpPoolMemory(KVPoolMemory):
    """The paged pool is memory UMBP allocated on this rank's GPU."""

    counted_by_torch = False

    def __init__(self, client, pool_id: str, group: str):
        self._client = client
        self.pool_id = pool_id
        self.group = group
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

        from atom.mono.runtime import hip_ipc

        if nbytes <= 0:
            raise NotImplementedError(
                "this model declares no paged pool for UMBP to own"
            )
        index = device.index if device.index is not None else 0
        kind, handle, got, created = self._client.attach(
            self.pool_id, self.group, index, nbytes
        )
        if kind != "hip_ipc" or got < nbytes:
            raise RuntimeError(
                f"UMBP pool {self.pool_id}: cannot map a {kind!r} handle of "
                f"{got} B for a {nbytes} B pool"
            )
        self._mapped_base = hip_ipc.open_ipc_handle(handle)
        buf = torch.as_tensor(_DeviceBuffer(self._mapped_base, nbytes), device=device)
        logger.info(
            "UMBP pool %s: %s %.2f GiB on %s, mapped at %#x (group %s)",
            self.pool_id,
            "created" if created else "re-attached",
            nbytes / (1 << 30),
            device,
            self._mapped_base,
            self.group,
        )
        return buf

    def bind_layout(self, buf, block_regions: Sequence[Any], num_blocks: int) -> None:
        if not block_regions:
            raise NotImplementedError(
                "this attention backend publishes no PAGE regions, so UMBP "
                "cannot tell where a block's bytes are"
            )
        base = buf.data_ptr()
        layout = []
        for region in block_regions:
            if region.reverse_indexed:
                raise NotImplementedError("reverse-indexed PAGE regions")
            layout.append((region.base_addr - base, region.unit_bytes))
        # Blocks are copied one region at a time, so a block's bytes must not
        # be reachable through two regions: copying it would then also write
        # part of a neighbouring block.
        ordered = sorted(layout)
        overlaps = [
            (a, b) for a, b in pairwise(ordered) if a[0] + num_blocks * a[1] > b[0]
        ]
        end = ordered[-1][0] + num_blocks * ordered[-1][1]
        if overlaps or ordered[0][0] < 0 or end > buf.numel():
            raise NotImplementedError(
                f"UMBP pool {self.pool_id}: PAGE regions overlap or leave the "
                f"pool ({len(overlaps)} overlapping pairs, pool {buf.numel()} B, "
                f"last region ends at {end})"
            )
        self._client.bind_layout(self.pool_id, num_blocks, layout)
        logger.info(
            "UMBP pool %s: %d blocks of %d B in %d regions",
            self.pool_id,
            num_blocks,
            sum(unit for _, unit in layout),
            len(layout),
        )


def make_umbp_block_pool(
    config,
    num_blocks: int,
    *,
    on_evict: Callable[[int], None] | None = None,
    on_remap: Callable[[int], None] | None = None,
) -> UmbpBlockPool:
    if envs.ATOM_PREFIX_CACHE_POLICY.strip().lower() != "lru":
        logger.warning(
            "ATOM_PREFIX_CACHE_POLICY=%s is ignored: the UMBP pool evicts the "
            "least recently released block",
            envs.ATOM_PREFIX_CACHE_POLICY,
        )
    module, client = _connect()
    pool = UmbpBlockPool(
        client,
        (module.OP_HOLD, module.OP_RELEASE, module.OP_PUBLISH),
        pool_id_for(config),
        num_blocks,
        on_evict=on_evict,
        on_remap=on_remap,
    )
    pool.shares_content = envs.ATOM_UMBP_L1_FETCH
    return pool


def make_umbp_pool_memory(config, rank: int = 0) -> UmbpPoolMemory:
    if rank != 0:
        raise NotImplementedError(
            "the UMBP pool backs one runner per engine rank; tensor-parallel "
            "attention within a rank is not supported yet"
        )
    _, client = _connect()
    return UmbpPoolMemory(client, pool_id_for(config, rank), share_group_for(config))
