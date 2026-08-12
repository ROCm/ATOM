# SPDX-License-Identifier: MIT
# The worker-side driver: its own executor, and completions reported by req_id.

from atom.kv_transfer.offload.state_tier import StateOffloadTier
from atom.model_engine.state_offload import StateOffloadIndex


class FakeCodec:
    entry_bytes = 4096

    def __init__(self, put_ok=True, get_ok=True):
        self.put_ok, self.get_ok = put_ok, get_ok
        self.puts, self.gets = [], []

    def put(self, h, entry_index):
        self.puts.append((h, entry_index))
        return self.put_ok

    def get(self, h, entry_index):
        self.gets.append((h, entry_index))
        return self.get_ok


def tier(codec):
    return StateOffloadTier(codec, StateOffloadIndex(2, kv_offload_enabled=False))


def test_a_successful_spill_indexes_the_hash():
    codec = FakeCodec()
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    assert 11 in t.index.hashes


def test_the_codec_packs_the_staging_entry_not_the_pool_group():
    """The hook copied the group's bytes into staging precisely so `pop()`
    could hand the original out immediately; packing the group would race the
    new owner."""
    codec = FakeCodec()
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    assert codec.puts == [(11, 64)]


def test_a_failed_spill_does_not_index():
    """No spill acknowledgement is sent back to the scheduler, so the index is
    the only record — indexing a spill that did not land is a guaranteed
    false positive on every later request."""
    codec = FakeCodec(put_ok=False)
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    assert 11 not in t.index.hashes


def test_a_spill_always_releases_its_staging_slot():
    """A leaked slot shrinks the ring permanently and the feature quietly stops
    spilling with no error anywhere.  We must exhaust the ring via
    `request_spill` (the real allocator) so that `_free_slots` starts empty
    and only `release_staging` can refill it."""
    codec = FakeCodec(put_ok=False)
    t = tier(codec)
    # Drain both slots out of the ring the same way the real caller does.
    slot0 = t.index.request_spill(11, group=0)  # returns 0, ring now empty
    slot1 = t.index.request_spill(22, group=1)  # returns 1, ring now empty
    assert slot0 >= 0 and slot1 >= 0
    assert not t.index._free_slots  # ring genuinely empty before transfers
    t.submit_spill(11, entry_index=64, staging_slot=slot0)
    t.submit_spill(22, entry_index=65, staging_slot=slot1)
    t.drain()
    assert t.index._free_slots  # release_staging refilled the ring


def test_a_successful_load_reports_done():
    t = tier(FakeCodec())
    t.index.confirm_spill(11)
    t.submit_load("req-a", 11, group=3)
    t.drain()
    assert t.get_finished() == ({"req-a"}, set())


def test_a_failed_load_reports_failed_and_forgets_the_hash():
    """Three triggers funnel here — LMCache's own LRU, a spill that never
    landed, a transfer error — and all three are the same normal path."""
    t = tier(FakeCodec(get_ok=False))
    t.index.confirm_spill(11)
    t.submit_load("req-a", 11, group=3)
    t.drain()
    assert t.get_finished() == (set(), {"req-a"})
    assert 11 not in t.index.hashes


def test_get_finished_drains():
    t = tier(FakeCodec())
    t.submit_load("req-a", 11, group=3)
    t.drain()
    t.get_finished()
    assert t.get_finished() == (set(), set())
