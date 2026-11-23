# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""TokenBridge, the Python half of the Rust frontend: admission as the
OpenAI server's, a request's sequences through CoreManager, its outputs back
to Rust once per step, and abort."""

import threading
import types

import pytest

from atom.entrypoints.atomesh.token_bridge import TokenBridge
from atom.entrypoints.openai.protocol import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from atom.model_engine.request import RequestOutput


class _Seq:
    _next = 100

    def __init__(self, tokens, sampling, callback):
        _Seq._next += 1
        self.id = _Seq._next
        self.num_prompt_tokens = len(tokens)
        self.max_tokens = sampling.max_tokens
        self.callback = callback


class _Engine:
    """`preprocess_fanout` and CoreManager as TokenBridge reaches them."""

    def __init__(self, fail_add=False):
        self.added, self.aborted, self.fail_add = [], [], fail_add
        self.io_processor = types.SimpleNamespace(
            requests={}, preprocess_fanout=self._fanout
        )
        self.core_mgr = types.SimpleNamespace(
            add_request=self._add,
            abort_request=self.aborted.append,
            is_healthy=lambda: True,
            max_pool_tokens=None,
            _flush_stream_batch_fn=None,
        )

    def _fanout(self, tokens, sampling, stream_callbacks, **kwargs):
        self.kwargs = kwargs
        seqs = [_Seq(tokens, sampling, cb) for cb in stream_callbacks]
        for seq in seqs:
            self.io_processor.requests[seq.id] = seq
        return seqs

    def _add(self, seqs):
        if self.fail_add:
            raise RuntimeError("engine gone")
        self.added.append(seqs)


class _Sink:
    def __init__(self):
        self.pushes = []

    def push(self, outputs):
        self.pushes.append(outputs)


def _bridge(engine=None, *, max_model_len=None, output_limit=None, channel=None):
    """A TokenBridge without a tokenizer: its admission and plumbing only."""
    bridge = TokenBridge.__new__(TokenBridge)
    bridge.engine = engine or _Engine()
    bridge.sink = _Sink()
    bridge._required_channel = channel
    bridge._max_model_len = max_model_len
    bridge._output_limit = output_limit
    bridge._lock = threading.Lock()
    bridge._engine_lock = threading.Lock()
    bridge._seq_ids = {}
    bridge._local = threading.local()
    bridge.engine.core_mgr._flush_stream_batch_fn = bridge._flush
    return bridge


# ── admission ─────────────────────────────────────────────────────────────


def test_sampling_defaults_are_the_openai_servers():
    sp = _bridge()._sampling_params({})
    assert (sp.temperature, sp.top_k, sp.top_p) == (
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_K,
        DEFAULT_TOP_P,
    )
    assert sp.max_tokens == DEFAULT_MAX_TOKENS and sp.max_tokens_is_default
    assert sp.n == 1 and sp.required_channel is None and not sp.ignore_eos


def test_the_deployment_cap_shrinks_a_default_and_refuses_an_explicit_ask():
    bridge = _bridge(output_limit=100)
    assert bridge._sampling_params({}).max_tokens == 100
    assert bridge._sampling_params({"max_tokens": 50}).max_tokens == 50
    with pytest.raises(ValueError, match="max_tokens must be <= 100"):
        bridge._sampling_params({"max_tokens": 101})


def test_greedy_sampling_collapses_n_and_the_channel_follows_the_flag():
    bridge = _bridge(channel="CHANNEL")
    assert bridge._sampling_params({"n": 3, "temperature": 0}).n == 1
    assert bridge._sampling_params({"n": 3, "temperature": 0.7}).n == 3
    assert bridge._sampling_params({}).required_channel is None
    held = bridge._sampling_params({"tool_choice_required": True})
    assert held.required_channel == "CHANNEL"


def test_context_fit_shrinks_a_default_and_refuses_an_overflow():
    bridge = _bridge(max_model_len=100)
    seq = types.SimpleNamespace(num_prompt_tokens=60, max_tokens=8192)
    bridge._fit_context(seq, bridge._sampling_params({}))
    assert seq.max_tokens == 40

    full = types.SimpleNamespace(num_prompt_tokens=100, max_tokens=8192)
    with pytest.raises(ValueError, match="no context space"):
        bridge._fit_context(full, bridge._sampling_params({}))

    asked = types.SimpleNamespace(num_prompt_tokens=60, max_tokens=50)
    with pytest.raises(ValueError, match="maximum context length"):
        bridge._fit_context(asked, bridge._sampling_params({"max_tokens": 50}))


def test_a_prompt_larger_than_the_kv_pool_is_refused():
    bridge = _bridge()
    bridge.engine.core_mgr.max_pool_tokens = 10
    seq = types.SimpleNamespace(num_prompt_tokens=11, max_tokens=1)
    with pytest.raises(ValueError):
        bridge._fit_context(seq, bridge._sampling_params({"max_tokens": 1}))


# ── submit, outputs, abort ────────────────────────────────────────────────


def _output(seq, tokens, finished=False, reason=None, cached=0):
    return RequestOutput(
        request_id=seq.id,
        output_tokens=tokens,
        finished=finished,
        finish_reason=reason,
        num_cached_tokens=cached,
    )


def test_submit_hands_the_siblings_to_core_manager_with_the_routing_fields():
    bridge = _bridge()
    params = {"n": 2, "temperature": 0.5, "request_id": "r1", "data_parallel_rank": 1}
    assert bridge.submit(7, [1, 2, 3], params) == 2
    (seqs,) = bridge.engine.added
    assert bridge._seq_ids[7] == [s.id for s in seqs]
    assert bridge.engine.kwargs["parent_request_id"] == "r1"
    assert bridge.engine.kwargs["data_parallel_rank"] == 1


@pytest.mark.parametrize("affinity", ["1", "0"])
def test_session_ids_reach_preprocess_only_with_session_affinity(monkeypatch, affinity):
    """The router reads the session headers (`x-session-id`, ...); as on the
    OpenAI server, CoreManager gets them only with ATOM_DP_SESSION_AFFINITY."""
    monkeypatch.setenv("ATOM_DP_SESSION_AFFINITY", affinity)
    bridge = _bridge()
    bridge.submit(7, [1, 2], {"dp_session_id": "s1", "dp_parent_session_id": "p0"})
    expected = ("s1", "p0") if affinity == "1" else (None, None)
    kwargs = bridge.engine.kwargs
    assert (kwargs["dp_session_id"], kwargs["dp_parent_session_id"]) == expected


def test_a_steps_outputs_reach_rust_in_one_push_and_a_finish_cleans_up():
    bridge = _bridge()
    bridge.submit(7, [1, 2, 3], {"n": 2, "temperature": 0.5})
    a, b = bridge.engine.added[0]
    a.callback(_output(a, [11]))
    b.callback(_output(b, [21], finished=True, reason="eos", cached=3))
    assert bridge.sink.pushes == []  # buffered until the step's flush
    bridge.engine.core_mgr._flush_stream_batch_fn()
    assert bridge.sink.pushes == [
        [(7, 0, [11], False, None, 0), (7, 1, [21], True, "eos", 3)]
    ]
    assert bridge._seq_ids[7] == [a.id]
    assert b.id not in bridge.engine.io_processor.requests

    a.callback(_output(a, [], finished=True, reason="max_tokens"))
    bridge._flush()
    assert 7 not in bridge._seq_ids
    assert bridge.engine.io_processor.requests == {}
    bridge._flush()  # nothing pending: no empty push
    assert len(bridge.sink.pushes) == 2


def test_abort_stops_every_sibling_once_and_forgets_the_key():
    bridge = _bridge()
    bridge.submit(7, [1, 2], {"n": 2, "temperature": 0.5})
    ids = [s.id for s in bridge.engine.added[0]]
    bridge.abort([7])
    assert bridge.engine.aborted == ids
    assert 7 not in bridge._seq_ids and bridge.engine.io_processor.requests == {}
    bridge.abort([7, 8])  # finished or unknown: nothing to stop
    assert bridge.engine.aborted == ids


def test_a_submit_core_manager_refuses_leaves_nothing_behind():
    bridge = _bridge(_Engine(fail_add=True))
    with pytest.raises(RuntimeError, match="engine gone"):
        bridge.submit(7, [1, 2], {})
    assert bridge._seq_ids == {} and bridge.engine.io_processor.requests == {}


def test_a_submit_admission_refuses_never_reaches_core_manager():
    bridge = _bridge(max_model_len=4)
    with pytest.raises(ValueError):
        bridge.submit(7, [1, 2, 3, 4], {})
    assert bridge.engine.added == [] and bridge._seq_ids == {}
    assert bridge.engine.io_processor.requests == {}
