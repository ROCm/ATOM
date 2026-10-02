# SPDX-License-Identifier: MIT
"""StateOffloadIndex must stop advertising images the worker codec dropped."""

from atom.model_engine.state_offload import StateOffloadIndex, state_tier_cpu_bytes


def _stored(idx, hashes):
    for h in hashes:
        idx.note_stored(h)


class TestStateOffloadIndexCap:
    def test_no_codec_bound_keeps_the_leak_cap(self):
        assert StateOffloadIndex()._hash_cap == 1 << 20

    def test_cap_tracks_codec_entries_minus_margin(self):
        assert StateOffloadIndex(max_cpu_entries=1183)._hash_cap == 1183 - 36
        assert StateOffloadIndex(max_cpu_entries=100)._hash_cap == 100 - 16
        assert StateOffloadIndex(max_cpu_entries=4)._hash_cap == 1

    def test_oldest_store_is_forgotten_past_the_cap(self):
        idx = StateOffloadIndex(max_cpu_entries=116)  # cap 100
        _stored(idx, range(150))
        assert len(idx.hashes) == 100
        assert idx.hashes_evicted == 50
        assert not idx.could_serve(0)
        assert idx.could_serve(149)

    def test_completed_load_refreshes_like_codec_get(self):
        idx = StateOffloadIndex(max_cpu_entries=116)  # cap 100
        _stored(idx, range(100))
        assert idx.request_load("r", 0)
        idx.complete_load("r")
        idx.note_stored(100)
        # Hash 1 is now least recent, as it is in the codec; 0 survives.
        assert idx.could_serve(0)
        assert not idx.could_serve(1)

    def test_failed_load_does_not_refresh(self):
        idx = StateOffloadIndex(max_cpu_entries=116)
        _stored(idx, range(10))
        assert idx.request_load("r", 3)
        idx.fail_load("r")
        assert not idx.could_serve(3)


def test_state_tier_cpu_bytes_reads_gib(monkeypatch):
    monkeypatch.delenv("OFFLOAD_STATE_CPU_SIZE", raising=False)
    assert state_tier_cpu_bytes() == 32 << 30
    monkeypatch.setenv("OFFLOAD_STATE_CPU_SIZE", "96")
    assert state_tier_cpu_bytes() == 96 << 30
