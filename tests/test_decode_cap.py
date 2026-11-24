# SPDX-License-Identifier: MIT
"""The P/D decode slot cap (`Scheduler._refresh_decode_cap`)."""

from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.scheduler import DECODE_CAP_FLOOR, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams


class _Consumer:
    """A kv_consumer that records which requests it asked the producer for."""

    is_producer = False
    is_offload = False

    def __init__(self):
        self.asked = []

    def get_num_new_matched_tokens(self, seq):
        if (seq.kv_transfer_params or {}).get("do_remote_prefill"):
            return len(seq.prompt_token_ids), True
        return 0, False

    def update_state_after_alloc(self, seq):
        params = seq.kv_transfer_params
        if params.get("do_remote_prefill"):
            self.asked.append(seq.id)
            params["do_remote_prefill"] = False

    def build_connector_meta(self):
        return None


def _decode(monkeypatch, *, cap=None, factor=None, window=None, **config):
    for name, value in (
        ("ATOM_DECODE_MAX_RUNNING", cap),
        ("ATOM_DECODE_RUN_CAP_FACTOR", factor),
        ("ATOM_DECODE_RUN_CAP_WINDOW_S", window),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))
    defaults = {
        "max_num_seqs": 8,
        "max_num_batched_tokens": 64,
        "num_kvcache_blocks": 200,
    }
    defaults.update(config)
    sched = Scheduler(MockConfig(**defaults))
    sched.kv_connector = _Consumer()
    return sched


def _remote(tokens=None, *, transfer_id=0):
    return Sequence(
        tokens or list(range(100, 108)),
        4,
        sampling_params=SamplingParams(max_tokens=64),
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": transfer_id,
            "first_token_id": 7,
        },
    )


def _arrive(sched, n, tokens=None):
    seqs = [_remote(tokens, transfer_id=i) for i in range(n)]
    for seq in seqs:
        sched.add(seq)
    sched.schedule()
    return seqs


def _land(sched, *seqs):
    for seq in seqs:
        sched._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving={seq.id})
        )


def _finish(sched, seq):
    sched.running.remove(seq)
    seq.status = SequenceStatus.FINISHED
    sched.block_manager.deallocate(seq)


def _parked(seq):
    return seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS


def test_landed_requests_beyond_the_cap_stay_parked(monkeypatch):
    sched = _decode(monkeypatch, cap=2)
    a, b, c = _arrive(sched, 3)
    _land(sched, a, b, c)
    sched.schedule()

    assert list(sched.running) == [a, b]
    assert _parked(c) and c in sched.waiting
    # Still counted against max_num_seqs, with its blocks and its report kept.
    assert sched._num_parked_remote_kv == 1
    assert sched._has_req_id(sched.finished_recving_kv_req_ids, c.id)
    assert c.block_table

    sched.schedule()  # no slot freed: nothing moves
    assert list(sched.running) == [a, b] and _parked(c)


def test_no_cap_by_default(monkeypatch):
    sched = _decode(monkeypatch)
    seqs = _arrive(sched, 3)
    _land(sched, *seqs)
    sched.schedule()
    assert list(sched.running) == seqs
    assert sched._decode_cap is None


def test_a_full_decode_set_does_not_hold_back_pushes(monkeypatch):
    """Requests arriving behind a held one are still allocated and asked for."""
    sched = _decode(monkeypatch, cap=1)
    a, b = _arrive(sched, 2)
    _land(sched, a, b)
    sched.schedule()
    assert list(sched.running) == [a] and _parked(b)

    later = _arrive(sched, 3)

    assert sched.kv_connector.asked == [a.id, b.id] + [s.id for s in later]
    assert all(_parked(s) and s.block_table for s in later)
    assert sched._num_parked_remote_kv == 4


def test_slots_go_to_the_oldest_landed_request(monkeypatch):
    """FIFO by arrival at D, whatever order the pushes land in: a held request
    is never overtaken, so it cannot starve."""
    sched = _decode(monkeypatch, cap=1)
    a, b, c, d = _arrive(sched, 4)
    _land(sched, a)
    sched.schedule()
    _land(sched, d, c, b)  # land newest first
    for expected in (b, c, d):
        sched.schedule()
        assert list(sched.running) == [sched.running[0]]
        _finish(sched, sched.running[0])
        sched.schedule()
        assert list(sched.running) == [expected]
    assert sched._num_parked_remote_kv == 0


def test_an_aborted_held_request_releases_its_reservation(monkeypatch):
    sched = _decode(monkeypatch, cap=1)
    a, b = _arrive(sched, 2)
    _land(sched, a, b)
    sched.schedule()
    free = sched.block_manager.pool_occupancy()["free"]

    assert sched.abort_request(b.id)

    assert sched.take_rejected() == [b]
    assert sched._num_parked_remote_kv == 0
    assert sched.finished_recving_kv_req_ids == []
    assert sched.block_manager.pool_occupancy()["free"] > free


def test_preemption_frees_a_slot_and_its_recompute_is_not_held(monkeypatch):
    """A preempted request recomputes locally; the cap does not hold it (it
    would otherwise queue behind every later landed request), so the decode set
    can run one over the cap until it drains."""
    sched = _decode(monkeypatch, cap=2)
    a, b, c = _arrive(sched, 3)
    _land(sched, a, b, c)
    sched.schedule()
    assert list(sched.running) == [a, b] and _parked(c)

    assert sched._preempt_one_running()
    assert b.status == SequenceStatus.WAITING and sched.waiting[0] is b
    sched.schedule()

    assert set(sched.running) == {a, b, c}
    assert not _parked(c)

    d = _arrive(sched, 1)[0]
    _land(sched, d)
    _finish(sched, a)
    sched.schedule()
    assert _parked(d)  # b and c hold both slots
    _finish(sched, b)
    sched.schedule()
    assert d in sched.running


def test_a_held_request_publishes_its_prefix_on_promotion(monkeypatch):
    """Its received blocks enter D's prefix cache once it is promoted, and a
    request reusing a decoding request's prefix still gets asked for while
    the slots are full."""
    sched = _decode(monkeypatch, cap=1, enable_prefix_caching=True)
    prompt = list(range(100, 116))
    a = _arrive(sched, 1, prompt)[0]
    _land(sched, a)
    sched.schedule()
    assert list(sched.running) == [a]

    b = _arrive(sched, 1, prompt + [1, 2, 3, 4])[0]
    assert b.id in sched.kv_connector.asked
    assert b.num_cached_tokens > 0  # D's own hit on a's published prefix
    _land(sched, b)
    sched.schedule()
    assert _parked(b) and not b.prefix_hashes_published

    _finish(sched, a)
    sched.schedule()
    assert list(sched.running) == [b] and b.prefix_hashes_published


def _demand(sched, decoding, landed):
    sched.running.clear()
    sched.running.extend(SimpleNamespace(id=-i) for i in range(1, decoding + 1))
    sched.finished_recving_kv_req_ids[:] = list(range(landed))


def test_adaptive_cap_follows_the_average_demand(monkeypatch):
    sched = _decode(
        monkeypatch,
        factor=1.25,
        window=10,
        capture_sizes=[1, 2, 4, 8, 16, 32, 48, 56, 64, 128],
    )
    _demand(sched, 40, 0)
    for t in range(101):
        sched._refresh_decode_cap(float(t))
    assert sched._decode_demand_avg == pytest.approx(40)
    assert sched._decode_cap == 56  # ceil(1.25 x 40) = 50, up to the captured 56

    # A burst of one tick (100 ms) moves the average a little, not the cap.
    _demand(sched, 56, 60)
    sched._refresh_decode_cap(100.1)
    assert sched._decode_cap == 56

    # A lasting rise counts the held requests and lifts the cap past them.
    for t in range(102, 160):
        sched._refresh_decode_cap(float(t))
    assert sched._decode_cap >= 56 + 60

    # Little demand: the floor.
    _demand(sched, 2, 0)
    for t in range(160, 400):
        sched._refresh_decode_cap(float(t))
    assert sched._decode_cap == DECODE_CAP_FLOOR


def test_a_fixed_cap_bounds_the_adaptive_one(monkeypatch):
    sched = _decode(monkeypatch, cap=48, factor=1.25, window=10)
    _demand(sched, 100, 0)
    sched._refresh_decode_cap(0.0)
    assert sched._decode_cap == 48


def test_the_cap_is_for_a_consumer_only(monkeypatch):
    sched = _decode(monkeypatch, cap=1)
    sched.kv_connector = SimpleNamespace(is_producer=False, is_offload=True)
    sched._refresh_decode_cap(0.0)
    assert sched._decode_cap is None
