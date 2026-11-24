# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""TokenBridge, the Python half of the Rust frontend: admission as the
OpenAI server's, a request's sequences through CoreManager, its outputs back
to Rust once per step, and abort."""

import itertools
import threading
import types
from collections import OrderedDict

import pytest

from atom.entrypoints.atomesh import token_bridge
from atom.entrypoints.atomesh.token_bridge import TokenBridge
from atom.entrypoints.openai.protocol import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from atom.model_engine.request import RequestOutput
from atom.model_engine.segment_tokenizer import PrefixEncoding


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
            requests={},
            preprocess_fanout=self._fanout,
            _external_to_internal={},
            _internal_to_external={},
        )
        self.core_mgr = types.SimpleNamespace(
            add_request=self._add,
            abort_request=self.aborted.append,
            is_healthy=lambda: True,
            max_pool_tokens=None,
            _flush_stream_batch_fn=None,
        )

    def _fanout(self, prompt, sampling, stream_callbacks, **kwargs):
        self.prompt, self.kwargs = prompt, kwargs
        tokens = getattr(prompt, "token_ids", prompt)
        seqs = [_Seq(tokens, sampling, cb) for cb in stream_callbacks]
        io, external = self.io_processor, kwargs.get("parent_request_id")
        for seq in seqs:
            io.requests[seq.id] = seq
            if len(seqs) == 1 and external is not None:  # as preprocess_fanout
                io._external_to_internal[external] = seq.id
                io._internal_to_external[seq.id] = external
        return seqs

    def _add(self, seqs):
        if self.fail_add:
            raise RuntimeError("engine gone")
        self.added.append(seqs)


class _Sink:
    def __init__(self, broken=False):
        self.pushes, self.failed, self.broken = [], [], broken

    def push(self, outputs):
        if self.broken:
            raise TypeError("cannot convert")
        self.pushes.append(outputs)

    def fail(self, keys, message):
        self.failed.append((keys, message))


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
    bridge._rendered = OrderedDict()
    bridge._render_handles = itertools.count(1)
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


def test_a_finished_or_aborted_request_leaves_no_request_id_behind():
    """`preprocess_fanout` maps a single sequence to its request id; only the
    offline `postprocess` pops those maps, so the bridge does."""
    bridge = _bridge()
    io = bridge.engine.io_processor
    bridge.submit(7, [1, 2], {"request_id": "r7"})
    bridge.submit(8, [1, 2], {"request_id": "r8"})
    (done,), _ = bridge.engine.added
    done.callback(_output(done, [5], finished=True, reason="eos"))
    bridge.abort([8])
    assert io.requests == {}
    assert io._external_to_internal == {} and io._internal_to_external == {}


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


# ── a Python-rendered prompt's encoding ───────────────────────────────────


def _rendering(bridge, encoded):
    bridge._renderer = types.SimpleNamespace(render=lambda request: ("PROMPT", None))
    bridge.engine.io_processor.encode_prompt = lambda prompt: encoded


def test_the_encoding_render_chat_built_reaches_preprocessing_once():
    """Its block-hash chain is not thrown away and computed again."""
    bridge = _bridge()
    encoded = PrefixEncoding([1, 2, 3], "HASHES")
    _rendering(bridge, encoded)
    _, ids, _, handle = bridge.render_chat('{"messages": []}')
    assert ids == [1, 2, 3] and handle is not None
    bridge.submit(7, ids, {"prompt_handle": handle})
    assert bridge.engine.prompt is encoded
    assert bridge._rendered == {}  # collected


def test_without_a_matching_encoding_the_ids_go_as_they_are():
    bridge = _bridge()
    _rendering(bridge, PrefixEncoding([1, 2, 3], "HASHES"))
    handle = bridge.render_chat("{}")[3]
    bridge.submit(7, [9, 9], {"prompt_handle": handle})  # not the rendered ids
    assert bridge.engine.prompt == [9, 9]
    bridge.submit(8, [1, 2, 3], {"prompt_handle": 12345})  # unknown handle
    assert bridge.engine.prompt == [1, 2, 3]

    _rendering(bridge, [4, 5])  # no tokenizer cache: plain ids, no handle
    assert bridge.render_chat("{}")[3] is None


def test_uncollected_encodings_are_bounded_by_count_and_age(monkeypatch):
    monkeypatch.setattr(token_bridge, "_RENDERED_MAX", 2)
    bridge = _bridge()
    _rendering(bridge, PrefixEncoding([1], None))
    handles = [bridge.render_chat("{}")[3] for _ in range(3)]
    assert list(bridge._rendered) == handles[1:]

    clock = iter([1000.0, 1000.0 + token_bridge._RENDERED_TTL_S + 1])
    monkeypatch.setattr(token_bridge.time, "monotonic", lambda: next(clock))
    old = bridge.render_chat("{}")[3]
    new = bridge.render_chat("{}")[3]  # the earlier ones have aged out
    assert old not in bridge._rendered and list(bridge._rendered) == [new]


# ── a lost step ───────────────────────────────────────────────────────────


def test_a_step_rust_could_not_take_ends_its_requests_with_an_error():
    bridge = _bridge()
    bridge.sink = _Sink(broken=True)
    bridge.submit(7, [1, 2], {"n": 2, "temperature": 0.5})
    bridge.submit(8, [3], {})
    a, b = bridge.engine.added[0]
    (c,) = bridge.engine.added[1]
    a.callback(_output(a, [11], finished=True, reason="eos"))
    c.callback(_output(c, [31]))
    bridge._flush()
    ((keys, message),) = bridge.sink.failed
    assert keys == [7, 8] and "lost an engine step" in message
    # What is still running is stopped; the finished sibling already left.
    assert sorted(bridge.engine.aborted) == sorted([b.id, c.id])
    assert bridge._seq_ids == {}


# ── tool calls the OpenAI server holds to a grammar ───────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


def _refusal(monkeypatch, parser_cls, grammar_on):
    """`ChatRenderer`'s check, for a model whose prompt opens K3's think."""
    from atom.entrypoints.openai import kimi_k3_tokens

    monkeypatch.setenv("ATOM_TOOL_CALL_GRAMMAR", "1" if grammar_on else "0")
    monkeypatch.setattr(token_bridge.structured_output, "available", lambda: True)
    renderer = token_bridge.ChatRenderer.__new__(token_bridge.ChatRenderer)
    renderer.tool_parser_cls = parser_cls
    think = kimi_k3_tokens.THINK_START
    renderer.dialect = types.SimpleNamespace(prompt_open_marker=think)
    return lambda tools, tool_choice=None: renderer._refuse_unheld_tool_calls(
        tools, tool_choice, think
    )


def test_a_k3_chat_with_tools_the_server_would_hold_is_refused(monkeypatch):
    from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser

    refuse = _refusal(monkeypatch, KimiK3Parser, grammar_on=True)
    with pytest.raises(ValueError, match="ATOM_TOOL_CALL_GRAMMAR"):
        refuse(_TOOLS)
    # tool_choice "none" or no tools: nothing to hold, nothing refused.
    refuse(_TOOLS, "none")
    refuse([])


@pytest.mark.parametrize("parser", ["k3_grammar_off", "dsml"])
def test_tool_calls_the_server_leaves_unconstrained_are_served(monkeypatch, parser):
    from atom.entrypoints.openai.tool_parser.deepseekv4_tool_parser import DsmlParser
    from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser

    if parser == "dsml":  # a format that declares no grammar
        _refusal(monkeypatch, DsmlParser, grammar_on=True)(_TOOLS)
    else:  # the server's default: ATOM_TOOL_CALL_GRAMMAR off
        _refusal(monkeypatch, KimiK3Parser, grammar_on=False)(_TOOLS)
