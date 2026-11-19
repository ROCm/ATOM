# SPDX-License-Identifier: MIT
"""Side-stream DCP KV all-gather (``ATOM_DCP_PREFILL_KV_AG_OVERLAP``).

No device. ``torch.cuda`` is replaced by a recorder, so what is under test is
the ordering contract the overlap rests on:

  * the gate (``dcp_all_gather_ready``): None by default, None for a group that
    borrows another group's communicators, None during graph capture;
  * ``DcpAllGatherHandle``: the side stream starts after the ready event (not
    after whatever the caller queued since), runs the input's producer AND the
    collective, and ``wait()`` makes the caller's stream follow it and keep the
    result alive;
  * ``MLAAttention._dcp_start_context_gathers`` starts exactly the chunks the
    in-place loop would gather, and only when it should; ``_dcp_prefetch_next``
    hands layer l's gathers to layer l+1 (its own cache, nothing after the last
    layer, nothing across steps or layers).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="dcp_ops defines @triton.jit kernels")
pytest.importorskip("aiter", reason="attention_mla imports the AITER runtime")

from atom.model_ops import attention_mla as mla
from atom.model_ops import dcp_ops


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_event(self, event):
        self.log.append(f"{self.name}.wait_event")

    def __repr__(self):
        return self.name


@pytest.fixture
def cuda(monkeypatch):
    """A recording ``torch.cuda``: a main stream, one side stream, an event."""
    log, state = [], SimpleNamespace(current=None, capturing=False)
    main = _Stream("main", log)
    state.current = main

    class Event:
        def record(self, stream):
            log.append(f"event.record({stream.name})")

    class stream:
        def __init__(self, s):
            self.s = s

        def __enter__(self):
            self.prev, state.current = state.current, self.s

        def __exit__(self, *exc):
            state.current = self.prev

    def side_stream(priority=0):
        log.append(f"Stream(priority={priority})")
        return _Stream("side", log)

    monkeypatch.setattr(torch.cuda, "Stream", side_stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: state.current)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(dcp_ops, "_side_stream", None)
    return SimpleNamespace(log=log, state=state)


def test_gate(cuda, monkeypatch):
    group = SimpleNamespace(reuse_from=None)
    monkeypatch.setenv("ATOM_DCP_PREFILL_KV_AG_OVERLAP", "0")
    assert dcp_ops.dcp_all_gather_ready(group) is None
    monkeypatch.setenv("ATOM_DCP_PREFILL_KV_AG_OVERLAP", "1")
    assert dcp_ops.dcp_all_gather_ready(group) is not None
    assert cuda.log == ["event.record(main)"]
    # A borrowed communicator shares the TP all-reduce's pool and signals.
    assert dcp_ops.dcp_all_gather_ready(SimpleNamespace(reuse_from=object())) is None
    cuda.state.capturing = True
    assert dcp_ops.dcp_all_gather_ready(group) is None


def test_handle_starts_after_ready_not_after_the_callers_later_work(cuda, monkeypatch):
    ran_on = []
    out = SimpleNamespace(record_stream=lambda s: cuda.log.append(f"record({s.name})"))

    def gather(group, x, dim):
        ran_on.append(("gather", cuda.state.current.name, x, dim))
        return out

    monkeypatch.setattr(dcp_ops, "dcp_all_gather", gather)

    def produce():
        ran_on.append(("produce", cuda.state.current.name))
        return "local_kv"

    monkeypatch.setenv("ATOM_DCP_PREFILL_KV_AG_OVERLAP", "1")
    ready = dcp_ops.dcp_all_gather_ready(SimpleNamespace(reuse_from=None))
    handle = dcp_ops.DcpAllGatherHandle("group", produce, 0, ready)
    # Producer and collective both run on the side stream, nothing on main yet.
    assert ran_on == [("produce", "side"), ("gather", "side", "local_kv", 0)]
    # The side stream is high priority: a normal one can share the main stream's HW queue.
    assert cuda.log == [
        "event.record(main)",
        "Stream(priority=-1)",
        "side.wait_event",
        "event.record(side)",
    ]

    assert handle.wait() is out
    assert cuda.log[4:] == ["main.wait_event", "record(main)"]


def test_handles_share_one_side_stream(cuda, monkeypatch):
    monkeypatch.setattr(dcp_ops, "dcp_all_gather", lambda g, x, d: x)
    ready = torch.cuda.Event()
    dcp_ops.DcpAllGatherHandle("g", lambda: 1, 0, ready)
    first = dcp_ops._side_stream
    dcp_ops.DcpAllGatherHandle("g", lambda: 2, 0, ready)
    assert dcp_ops._side_stream is first


def _impl():
    return SimpleNamespace(dcp_group=SimpleNamespace(reuse_from=None))


def _meta(seq_tot, *, has_cached=True, is_dcp=True):
    chunks = SimpleNamespace(
        is_dcp=is_dcp,
        num_chunks=len(seq_tot),
        seq_tot=seq_tot,
        local_slot_ids=[f"slots{c}" for c in range(len(seq_tot))],
    )
    return SimpleNamespace(has_cached=has_cached, mla_chunk_meta=chunks)


@pytest.fixture
def started(cuda, monkeypatch):
    monkeypatch.setenv("ATOM_DCP_PREFILL_KV_AG_OVERLAP", "1")
    calls = []

    class Handle:
        def __init__(self, group, produce, dim, ready):
            calls.append((produce.args, dim, ready is not None))

    monkeypatch.setattr(dcp_ops, "DcpAllGatherHandle", Handle)
    return calls


_READY = object()


def _start(meta, ready=_READY):
    return mla.MLAAttention._dcp_start_context_gathers(
        _impl(), "cache", meta.mla_chunk_meta, ready
    )


def test_start_skips_empty_chunks_like_the_in_place_loop(started):
    gathers = _start(_meta([4, 0, 2]))
    assert sorted(gathers) == [0, 2]
    assert started == [
        (("cache", "slots0"), 0, True),
        (("cache", "slots2"), 0, True),
    ]


def test_start_does_nothing_without_ready(started):
    assert _start(_meta([4]), None) == {}
    assert started == []


def _prefetch_setup(monkeypatch, layers):
    kv = {f"layer_{n}": SimpleNamespace(k_cache=f"cache{n}") for n in layers}
    monkeypatch.setattr(
        mla, "get_forward_context", lambda: SimpleNamespace(kv_cache_data=kv)
    )
    monkeypatch.setattr(mla.MLAAttention, "_dcp_layers", set(layers))
    monkeypatch.setattr(mla.MLAAttention, "_dcp_prefetched", None)


def _layer(n):
    # Real methods on a stand-in layer: only what they read is provided.
    impl = SimpleNamespace(layer_num=n, dcp_group=SimpleNamespace(reuse_from=None))
    for name in (
        "_dcp_start_context_gathers",
        "_dcp_prefetch_next",
        "_dcp_take_prefetched",
    ):
        setattr(impl, name, getattr(mla.MLAAttention, name).__get__(impl))
    return impl


def test_prefetch_starts_the_next_dcp_layer_and_take_is_one_shot(started, monkeypatch):
    _prefetch_setup(monkeypatch, [4, 8, 12])
    meta = _meta([4, 2])
    l4, l8 = _layer(4), _layer(8)
    l4._dcp_prefetch_next(meta, meta.mla_chunk_meta, object())
    # Layer 8's own cache, not layer 4's and not layer 12's.
    assert started == [
        (("cache8", "slots0"), 0, True),
        (("cache8", "slots1"), 0, True),
    ]
    got = l8._dcp_take_prefetched(meta)
    assert sorted(got) == [0, 1]
    assert l8._dcp_take_prefetched(meta) is None


def test_no_prefetch_after_the_last_layer_or_without_ready(started, monkeypatch):
    _prefetch_setup(monkeypatch, [4, 8])
    meta = _meta([4])
    _layer(8)._dcp_prefetch_next(meta, meta.mla_chunk_meta, object())
    _layer(4)._dcp_prefetch_next(meta, meta.mla_chunk_meta, None)
    assert started == []
    assert mla.MLAAttention._dcp_prefetched is None


def test_prefetched_gathers_of_another_step_or_layer_are_dropped(started, monkeypatch):
    _prefetch_setup(monkeypatch, [4, 8])
    meta, other_step = _meta([4]), _meta([4])
    _layer(4)._dcp_prefetch_next(meta, meta.mla_chunk_meta, object())
    assert _layer(8)._dcp_take_prefetched(other_step) is None
    _layer(4)._dcp_prefetch_next(meta, meta.mla_chunk_meta, object())
    assert _layer(4)._dcp_take_prefetched(meta) is None  # prefetched for layer 8
    assert mla.MLAAttention._dcp_prefetched is None


def test_context_loop_joins_started_gathers_and_gathers_the_rest_in_place(monkeypatch):
    log = []

    class Handle:
        def wait(self):
            log.append("wait")
            return "ag_kv"

    monkeypatch.setattr(
        dcp_ops, "dcp_all_gather", lambda *a: log.append("in_place") or "ag_kv"
    )
    monkeypatch.setattr(dcp_ops, "dcp_gather_compressed_kv", lambda *a: "local")
    impl = SimpleNamespace(dcp_group="group")
    # total_tokens 0: every chunk gathers, then skips the (unmocked) attention.
    chunks = SimpleNamespace(
        num_chunks=3,
        seq_tot=[4, 4, 4],
        total_tokens=[0, 0, 0],
        local_slot_ids=[None] * 3,
    )
    run = mla.MLAAttention._dcp_compute_prefill_context
    gathers = {0: Handle(), 2: Handle()}
    run(impl, None, "cache", None, chunks, ctx_gathers=gathers)
    assert log == ["wait", "in_place", "wait"]
    assert gathers == {}
    log.clear()
    run(impl, None, "cache", None, chunks)
    assert log == ["in_place"] * 3


@pytest.mark.parametrize(
    "max_seqlen_q,is_dcp,expect",
    [
        (mla._DCP_AG_PREFETCH_MIN_NEW_TOKENS, True, True),
        (mla._DCP_AG_PREFETCH_MIN_NEW_TOKENS - 1, True, False),  # tail step
        (32768, False, False),
    ],
)
def test_prefetch_ready_only_for_big_dcp_steps(started, max_seqlen_q, is_dcp, expect):
    meta = SimpleNamespace(max_seqlen_q=max_seqlen_q)
    chunks = SimpleNamespace(is_dcp=is_dcp)
    ready = mla.MLAAttention._dcp_prefetch_ready(_impl(), meta, chunks)
    assert (ready is not None) is expect
