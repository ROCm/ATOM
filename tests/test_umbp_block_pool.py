# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The paged pool kept by UMBP, seen from `BlockManager`'s side of the seam.

Three things are worth a test and none is visible from one pool alone:

* `UmbpBlockPool` answers exactly what `BlockPool` answers for the same calls.
  UMBP makes the decisions (which block next, what is evicted) and this process
  keeps a local view; the two have to stay the same pool.
* Content one pool published becomes resident in another on `fetch`, which is
  the reason the backend exists.
* `BlockManager` gets a prefix hit out of that without knowing which backend
  it has.

Every test runs against `FakeL1`, a Python model of the UMBP service, and
again against the real service over host memory when `mori_umbp_l1` is
importable. The second run is what keeps the model honest.
"""

import array
import os
import random
import time
from collections import OrderedDict

import pytest
from conftest import MockConfig

from atom.model_engine import block_manager as bm_module
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.kv_block_pool import KVBlockPool
from atom.model_engine.umbp_block_pool import UmbpBlockPool

try:
    import mori_umbp_l1
except ImportError:  # the client module is optional outside a mori build
    mori_umbp_l1 = None

UNIT = 64
OPS = (0, 1, 2)  # hold, release, publish


def toks(*ids: int) -> array.array:
    return array.array("i", ids)


class FakeL1:
    """What the UMBP L1 service does to slots, without memory."""

    def __init__(self):
        # Set to n to make fetch stop after n slots, as a failed copy does.
        self.fetch_keeps = None
        self.pools = {}
        self.holders = {}  # (group, key) -> [(pool_id, slot)]
        self.copies = []  # (src pool, src slot, dst pool, dst slot)

    def attach(self, pool_id, group, device, nbytes):
        created = pool_id not in self.pools
        if created:
            self.pools[pool_id] = {"group": group}
        return "host", b"", nbytes, created

    def bind_layout(self, pool_id, num_slots, regions):
        self.pools[pool_id]["n"] = num_slots
        self._reset(pool_id)

    def _reset(self, pool_id):
        pool = self.pools[pool_id]
        for key in list(pool.get("index", {})):
            self._unindex(pool_id, pool["index"][key])
        pool.update(
            state=["V"] * pool["n"],
            key=[None] * pool["n"],
            written=[0] * pool["n"],
            released=[0] * pool["n"],
            committed=0,
            lru=OrderedDict(),
            index={},
        )

    def open(self, pool_id):
        self._reset(pool_id)
        return self.pools[pool_id]["n"]

    def _unindex(self, pool_id, slot):
        pool = self.pools[pool_id]
        key = pool["key"][slot]
        if key is None:
            return
        del pool["index"][key]
        holders = self.holders[(pool["group"], key)]
        holders.remove((pool_id, slot))
        pool["key"][slot] = None

    def _readable(self, pool_id, slot):
        pool = self.pools[pool_id]
        return pool["written"][slot] <= pool["committed"]

    def _fillable(self, pool_id, slot):
        pool = self.pools[pool_id]
        return (
            pool["released"][slot] <= pool["committed"]
            and pool["written"][slot] <= pool["committed"]
        )

    def _take_free(self, pool_id, evicted, settled_only=False):
        pool = self.pools[pool_id]

        def usable(slot):
            return not settled_only or self._fillable(pool_id, slot)

        for slot, state in enumerate(pool["state"]):
            if state == "V" and usable(slot):
                return slot
        if evicted is None:
            return None
        for slot in pool["lru"]:
            if usable(slot):
                del pool["lru"][slot]
                evicted.append(pool["key"][slot])
                self._unindex(pool_id, slot)
                return slot
        return None

    def commit(self, pool_id, committed):
        pool = self.pools[pool_id]
        pool["committed"] = max(pool["committed"], committed)

    def allocate(self, pool_id, count, min_count):
        pool = self.pools[pool_id]
        slots, evicted = [], []
        while len(slots) < count:
            slot = self._take_free(pool_id, evicted if len(slots) < min_count else None)
            if slot is None:
                break
            pool["state"][slot] = "H"
            slots.append(slot)
        return slots, evicted

    def apply(self, pool_id, kinds, slots, keys, stamp=0):
        pool = self.pools[pool_id]
        keys = iter(keys)
        for kind, slot in zip(kinds, slots):
            state = pool["state"][slot]
            if kind == 0:
                if state == "C":
                    del pool["lru"][slot]
                pool["state"][slot] = "H"
            elif kind == 1:
                if state != "H":
                    raise RuntimeError(f"release of a slot that is not held: {slot}")
                pool["released"][slot] = stamp
                if pool["key"][slot] is None:
                    pool["state"][slot] = "V"
                else:
                    pool["state"][slot] = "C"
                    pool["lru"][slot] = None
            else:
                key = next(keys)
                if state != "H":
                    raise RuntimeError(f"publish of a slot that is not held: {slot}")
                if pool["key"][slot] == key:
                    continue
                self._unindex(pool_id, slot)
                old = pool["index"].get(key)
                if old is not None:
                    self._unindex(pool_id, old)
                    pool["released"][old] = stamp
                    if pool["state"][old] == "C":
                        del pool["lru"][old]
                        pool["state"][old] = "V"
                pool["key"][slot] = key
                pool["written"][slot] = stamp
                pool["index"][key] = slot
                self.holders.setdefault((pool["group"], key), []).append(
                    (pool_id, slot)
                )

    def fetch(self, pool_id, keys):
        pool = self.pools[pool_id]
        slots, evicted, copied = [], [], 0
        pinned = set()
        for key in keys:
            local = pool["index"].get(key)
            if local is not None:
                pinned.add(local)
                slots.append(local)
                continue
            sources = [
                h
                for h in self.holders.get((pool["group"], key), [])
                if h[0] != pool_id and self._readable(*h)
            ]
            if not sources:
                break
            # A slot this same call reported is not a destination.
            kept = [s for s in pinned if s in pool["lru"]]
            for s in kept:
                del pool["lru"][s]
            dst = self._take_free(pool_id, evicted, settled_only=True)
            for s in kept:
                pool["lru"][s] = None
            if dst is None:
                break
            self.copies.append((*sources[0], pool_id, dst))
            pool["key"][dst] = key
            pool["written"][dst] = pool["released"][dst] = 0
            pool["index"][key] = dst
            pool["state"][dst] = "C"
            pool["lru"][dst] = None
            self.holders[(pool["group"], key)].append((pool_id, dst))
            pinned.add(dst)
            slots.append(dst)
            copied += 1
        if self.fetch_keeps is not None:
            # A copy failed part-way: the destinations after it were emptied
            # (and their old content reported) but are left vacant.
            for dst in slots[self.fetch_keeps :]:
                self._unindex(pool_id, dst)
                del pool["lru"][dst]
                pool["state"][dst] = "V"
            copied = min(copied, self.fetch_keeps)
            slots = slots[: self.fetch_keeps]
        return slots, evicted, copied, copied * UNIT, 0

    def clear(self, pool_id):
        pool = self.pools[pool_id]
        for slot in range(pool["n"]):
            self._unindex(pool_id, slot)
            if pool["state"][slot] == "C":
                del pool["lru"][slot]
                pool["state"][slot] = "V"


@pytest.fixture(params=["model", "service"])
def l1(request, tmp_path):
    """A client of an L1 service: the Python model, then the real one."""
    if request.param == "model":
        yield FakeL1()
        return
    if mori_umbp_l1 is None:
        pytest.skip("mori_umbp_l1 is not built")
    # A unix socket path has to fit in 108 bytes; tmp_path may not.
    path = f"/tmp/atom_l1_{os.getpid()}_{random.randrange(1 << 30)}.sock"
    server = mori_umbp_l1.HostTestServer(f"unix://{path}")
    assert server.ok()
    yield mori_umbp_l1.L1Client(f"unix://{path}")
    del server
    if os.path.exists(path):
        os.unlink(path)


class _Spy:
    """A client that records which calls reach the service."""

    def __init__(self, client, calls):
        self._client, self._calls = client, calls

    def __getattr__(self, name):
        method = getattr(self._client, name)

        def call(*args):
            self._calls.append(name)
            return method(*args)

        return call


_next_device = iter(range(1 << 20))


def make_pool(client, pool_id: str, num_blocks: int, **kwargs) -> UmbpBlockPool:
    client.attach(pool_id, "g", next(_next_device), num_blocks * UNIT)
    client.bind_layout(pool_id, num_blocks, [(0, UNIT)])
    return UmbpBlockPool(client, OPS, pool_id, num_blocks, **kwargs)


def settle(pool: UmbpBlockPool) -> None:
    """What the runner reports once its device has finished the writes issued
    so far: UMBP may now read and fill the blocks this pool published or freed."""
    pool.flush()
    pool._client.commit(pool.pool_id, time.monotonic_ns())


def cache(pool: KVBlockPool, hashes) -> list[int]:
    """Fill, publish and release one block per hash: cached, settled content."""
    ids = []
    for h in hashes:
        block_id = pool.pop()
        pool.allocate(block_id)
        pool.publish(block_id, h, toks(h))
        ids.append(block_id)
    for block_id in ids:
        pool.free(block_id)
    if isinstance(pool, UmbpBlockPool):
        settle(pool)
    return ids


class TestSameAnswersAsBlockPool:
    def test_random_workload(self, l1):
        """Step for step, with UMBP asked for one block at a time so both
        pools make every eviction at the same moment."""
        rng = random.Random(7)
        n = 12
        evicted = {"local": [], "umbp": []}
        local = BlockPool(n, on_evict=evicted["local"].append)
        umbp = make_pool(
            l1, "parity", n, on_evict=evicted["umbp"].append, grant_batch=1
        )
        held: list[int] = []
        hashes = iter(range(1000, 10_000))
        seen: list[int] = []
        for _ in range(600):
            roll = rng.random()
            if roll < 0.45 and local.has_free(1):
                a, b = local.pop(), umbp.pop()
                assert a == b
                local.allocate(a)
                umbp.allocate(b)
                held.append(a)
                if rng.random() < 0.8:
                    h = next(hashes)
                    seen.append(h)
                    local.publish(a, h, toks(h))
                    umbp.publish(b, h, toks(h))
            elif roll < 0.75 and held:
                block_id = held.pop(rng.randrange(len(held)))
                local.free(block_id)
                umbp.free(block_id)
            elif seen:
                h = rng.choice(seen)
                a, b = local.lookup(h), umbp.lookup(h)
                assert a == b
                if a != -1:
                    local.claim(a)
                    umbp.claim(b)
                    held.append(a)
            assert umbp.has_free(1) == local.has_free(1)
            assert (umbp.num_free, umbp.num_used, umbp.num_indexed) == (
                local.num_free,
                local.num_used,
                local.num_indexed,
            )
            assert umbp.num_reusable_free == local.num_reusable_free
            assert umbp.index_version == local.index_version
            assert evicted["umbp"] == evicted["local"]
        assert evicted["local"], "the workload never evicted anything"
        for h in seen:
            assert umbp.lookup(h) == local.lookup(h)
        for block_id in range(n):
            assert umbp.block(block_id).ref_count == local.block(block_id).ref_count
            assert umbp.is_used(block_id) == local.is_used(block_id)

    def test_republishing_a_hash_moves_the_index(self, l1):
        remapped = []
        pool = make_pool(l1, "remap", 4, on_remap=remapped.append)
        (old,) = cache(pool, [11])
        new = pool.pop()
        pool.allocate(new)
        pool.publish(new, 11, toks(11))
        assert pool.lookup(11) == new
        assert remapped == [11]
        assert pool.num_reusable_free == 0
        # The old copy is plain free space again: nothing is evicted for it.
        pool.flush()
        taken = [pool.pop() for _ in range(3)]
        assert old in taken
        assert pool.lookup(11) == new


class TestGrants:
    def test_granted_blocks_still_count_as_free(self, l1):
        pool = make_pool(l1, "grants", 8, grant_batch=4)
        block_id = pool.pop()
        assert pool.num_free == 8
        pool.allocate(block_id)
        assert (pool.num_free, pool.num_used) == (7, 1)
        assert pool.has_free(7) and not pool.has_free(8)

    def test_every_free_block_can_be_popped(self, l1):
        pool = make_pool(l1, "all", 8, grant_batch=3)
        ids = []
        for _ in range(8):
            ids.append(pool.pop())
            pool.allocate(ids[-1])
        assert sorted(ids) == list(range(8))
        with pytest.raises(AssertionError):
            pool.pop()

    def test_a_pop_evicts_only_what_it_needs(self, l1):
        evicted = []
        pool = make_pool(l1, "evict", 4, on_evict=evicted.append, grant_batch=3)
        cache(pool, [1, 2, 3, 4])
        version = pool.index_version
        block_id = pool.pop()
        # A batch is asked for, but only the one block needed costs content.
        assert evicted == [1]
        assert pool.lookup(1) == -1 and pool.lookup(2) != -1
        assert pool.index_version > version
        assert pool.blocks_evicted == 1
        assert pool.block(block_id).hash == -1

    def test_asking_ahead_takes_vacant_blocks_without_evicting(self, l1):
        evicted = []
        pool = make_pool(l1, "ahead", 6, on_evict=evicted.append, grant_batch=4)
        cache(pool, [1, 2])
        pool.allocate(pool.pop())
        assert evicted == []
        assert pool.lookup(1) != -1 and pool.lookup(2) != -1

    def test_prepare_gets_a_known_demand_in_one_call(self, l1):
        evicted = []
        pool = make_pool(l1, "prepare", 4, on_evict=evicted.append, grant_batch=1)
        cache(pool, [1, 2, 3, 4])
        calls = []
        pool._client = _Spy(pool._client, calls)
        pool.prepare(3)
        for _ in range(3):
            pool.allocate(pool.pop())
        assert calls == ["allocate"]
        assert evicted == [1, 2, 3]

    def test_prepare_beyond_the_free_blocks_is_harmless(self, l1):
        pool = make_pool(l1, "short", 2)
        pool.prepare(5)
        for _ in range(2):
            pool.allocate(pool.pop())
        with pytest.raises(AssertionError):
            pool.pop()

    def test_only_popped_blocks_can_be_allocated(self, l1):
        pool = make_pool(l1, "strict", 4)
        with pytest.raises(NotImplementedError):
            pool.allocate(2)

    def test_raw_units_are_refused(self, l1):
        pool = make_pool(l1, "raw", 4)
        with pytest.raises(NotImplementedError):
            pool.reserve_units(1, "owner")


class TestFetch:
    def test_content_of_another_pool_becomes_resident(self, l1):
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 8)
        cache(a, [1, 2, 3])
        got = b.fetch([1, 2, 3, 99], lambda i: toks(i + 1))
        assert got == 3
        ids = [b.lookup(h) for h in (1, 2, 3)]
        assert -1 not in ids and b.lookup(99) == -1
        assert [b.block(i).token_ids for i in ids] == [toks(1), toks(2), toks(3)]
        # Fetched blocks are free, reusable content: nothing holds them yet.
        assert (b.num_free, b.num_used, b.num_reusable_free) == (8, 0, 3)
        assert b.blocks_fetched == 3
        b.claim_many(ids)
        assert b.num_used == 3
        # The source pool is untouched.
        assert [a.lookup(h) != -1 for h in (1, 2, 3)] == [True] * 3

    def test_fetch_sees_blocks_a_sequence_still_holds(self, l1):
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 8)
        block_id = a.pop()
        a.allocate(block_id)
        a.publish(block_id, 5, toks(5))
        settle(a)
        assert b.fetch([5], lambda i: toks(5)) == 1

    def test_a_block_is_shared_only_once_the_device_has_written_it(self, l1):
        """Published is not written: the forward that fills the block may
        still be running, and another pool must not copy it yet."""
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 8)
        block_id = a.pop()
        a.allocate(block_id)
        a.publish(block_id, 5, toks(5))
        assert b.fetch([5], lambda i: toks(5)) == 0
        a.flush()
        assert b.fetch([5], lambda i: toks(5)) == 0
        settle(a)
        assert b.fetch([5], lambda i: toks(5)) == 1

    def test_freeing_a_block_does_not_hide_what_it_holds(self, l1):
        """A request ends and releases its blocks; their content was written
        long ago and stays available to other pools without another commit."""
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 8)
        block_id = a.pop()
        a.allocate(block_id)
        a.publish(block_id, 5, toks(5))
        settle(a)
        a.free(block_id)
        a.flush()
        assert b.fetch([5], lambda i: toks(5)) == 1

    def test_fetch_does_not_fill_a_block_just_freed(self, l1):
        """A forward of this pool's own engine may still be writing it."""
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 1)
        cache(a, [1])
        block_id = b.pop()
        b.allocate(block_id)
        b.free(block_id)
        b.flush()
        assert b.fetch([1], lambda i: toks(1)) == 0
        settle(b)
        assert b.fetch([1], lambda i: toks(1)) == 1
        assert b.lookup(1) == block_id

    def test_fetch_stops_at_the_first_missing_hash(self, l1):
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 8)
        cache(a, [1, 3])
        assert b.fetch([1, 2, 3], lambda i: toks(i)) == 1
        assert b.lookup(3) == -1

    def test_fetch_evicts_local_content_to_make_room(self, l1):
        evicted = []
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 2, on_evict=evicted.append)
        cache(a, [1, 2])
        cache(b, [50, 51])
        assert b.fetch([1, 2], lambda i: toks(i)) == 2
        assert sorted(evicted) == [50, 51]
        assert b.lookup(50) == -1 and b.lookup(1) != -1

    def test_fetch_that_stops_early_still_reports_what_it_evicted(self):
        """UMBP empties its destinations before it fills them. When the fill
        fails part-way, the rest stay empty: their old content is gone here
        too, though no fetched block took its place."""
        l1 = FakeL1()
        evicted = []
        a = make_pool(l1, "a", 4)
        b = make_pool(l1, "b", 4, on_evict=evicted.append)
        cache(a, [1, 2, 3])
        own = cache(b, [11, 12, 13, 14])
        l1.fetch_keeps = 1
        assert b.fetch([1, 2, 3], toks) == 1
        assert sorted(evicted) == [11, 12, 13]
        assert b.index_get(1) is not None
        assert b.index_get(2) is None and b.index_get(11) is None
        assert b.index_get(14) == own[3]
        assert b.num_free == 4
        # The pool goes on working, and agrees with UMBP about what is free.
        l1.fetch_keeps = None
        assert b.fetch([1, 2, 3], toks) == 3
        assert evicted == [11, 12, 13]

    def test_fetch_takes_no_block_a_sequence_holds(self, l1):
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 2)
        cache(a, [1, 2])
        for _ in range(2):
            b.allocate(b.pop())
        assert b.fetch([1, 2], lambda i: toks(i)) == 0

    def test_fetched_content_is_evicted_like_any_other(self, l1):
        evicted = []
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 2, on_evict=evicted.append, grant_batch=1)
        cache(a, [1, 2])
        b.fetch([1, 2], lambda i: toks(i))
        b.allocate(b.pop())
        assert evicted == [1]

    def test_clear_index_hides_content_from_other_pools(self, l1):
        a = make_pool(l1, "a", 8)
        b = make_pool(l1, "b", 8)
        cache(a, [1])
        a.clear_index()
        assert a.lookup(1) == -1 and a.num_indexed == 0
        assert b.fetch([1], lambda i: toks(1)) == 0


class TestThroughBlockManager:
    """Two engine ranks, one prompt: the second rank hits what the first wrote."""

    @pytest.fixture
    def ranks(self, l1, monkeypatch):
        cfg = MockConfig(
            num_kvcache_blocks=32, kv_cache_block_size=4, enable_prefix_caching=True
        )
        made = iter(range(8))

        def make(config, num_blocks, *, on_evict=None, on_remap=None, **_):
            return make_pool(
                l1,
                f"rank{next(made)}",
                num_blocks,
                on_evict=on_evict,
                on_remap=on_remap,
            )

        monkeypatch.setattr(bm_module, "make_block_pool", make)
        return BlockManager(cfg), BlockManager(cfg)

    def test_a_prefix_computed_on_one_rank_hits_on_another(self, ranks, seq_factory):
        rank0, rank1 = ranks
        tokens = list(range(20))
        first = seq_factory(tokens)
        rank0.allocate(first, 0)
        rank0.hash_blocks(first, len(tokens))
        settle(rank0.kv)

        second = seq_factory(tokens + [100, 101])
        hit = rank1.can_allocate(second)
        assert hit == 5
        assert rank1.kv.blocks_fetched == 5
        rank1.allocate(second, hit)
        assert second.num_cached_tokens == 20
        # The hit blocks are rank 1's own: resident in its pool and held there.
        for block_id in second.block_table[:5]:
            assert rank1.kv.is_used(block_id)
            assert rank1.kv.block(block_id).hash != -1

    def test_the_pool_is_asked_once_per_sequence(self, ranks, seq_factory):
        _, rank1 = ranks
        calls = []
        fetch = rank1.kv.fetch
        rank1.kv.fetch = lambda hashes, tokens_of: (
            calls.append(len(hashes)) or fetch(hashes, tokens_of)
        )
        seq = seq_factory(list(range(20)))
        for _ in range(3):
            rank1.can_allocate(seq)
        assert calls == [4]

    def test_a_local_prefix_is_not_asked_for_again(self, ranks, seq_factory):
        rank0, _ = ranks
        tokens = list(range(20))
        first = seq_factory(tokens)
        rank0.allocate(first, 0)
        rank0.hash_blocks(first, len(tokens))
        calls = []
        rank0.kv.fetch = lambda hashes, tokens_of: calls.append(len(hashes)) or 0
        again = seq_factory(tokens)
        assert rank0.can_allocate(again) == 4
        assert calls == []

    def test_the_local_backend_never_fetches(self, seq_factory):
        cfg = MockConfig(
            num_kvcache_blocks=32, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        assert not bm.kv.shares_content
        assert bm.kv.fetch([1, 2], lambda i: toks(i)) == 0
