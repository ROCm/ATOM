# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`tool_choice="required"` on Kimi-K3: which requests get a `RequiredChannel`,
what it holds the output to, and how the host keeps each sequence's state.

The GPU half -- masking target rows and carrying the state across a deferred
step -- is `tests/test_required_channel_mask.py`, which needs Triton.
"""

import random
import types

import pytest
from conftest import MockConfig

from atom.entrypoints.openai.tool_parser import (
    DsmlParser,
    DsmlV41Parser,
    GlmParser,
    KimiK3Parser,
    KimiParser,
    MiniMaxParser,
    QwenXmlParser,
)
from atom.entrypoints.openai.tool_parser.registry import (
    channel_for_request,
    required_call_channel,
)
from atom.model_engine.scheduler import (
    ScheduledBatch,
    ScheduledBatchOutput,
    Scheduler,
)
from atom.model_engine.sequence import SequenceType
from atom.sampling_params import UNCONSTRAINED, RequiredChannel, SamplingParams

# Kimi-K3's ids for its control tokens and the words between them, as its
# tokenizer encodes them (checked against /share_nfs/models/moonshotai/Kimi-K3).
K3_IDS = {
    "[EOS]": 163585,
    "<|end_of_msg|>": 163586,
    "<|open|>": 163587,
    "<|close|>": 163588,
    "<|sep|>": 163589,
    "tools": 25385,
    "response": 12092,
    "call": 10257,
    " tool": 4453,
    '="': 878,
}
OPEN, CLOSE, SEP = K3_IDS["<|open|>"], K3_IDS["<|close|>"], K3_IDS["<|sep|>"]
TOOLS, RESPONSE = K3_IDS["tools"], K3_IDS["response"]
CALL, TOOL_ATTR, EQ_QUOTE = K3_IDS["call"], K3_IDS[" tool"], K3_IDS['="']
END_OF_MSG, EOS = K3_IDS["<|end_of_msg|>"], K3_IDS["[EOS]"]


class K3Tokenizer:
    """Encodes text the way Kimi-K3's tokenizer does, for the literals here."""

    eos_token_id = EOS

    def __init__(self, split_controls=False):
        self.split_controls = split_controls

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        pieces = sorted(K3_IDS, key=len, reverse=True)
        ids = []
        while text:
            piece = next(p for p in pieces if text.startswith(p))
            if self.split_controls and piece.startswith("<|"):
                ids.extend(ord(c) for c in piece)
            else:
                ids.append(K3_IDS[piece])
            text = text[len(piece) :]
        return ids


K3_CHANNEL = KimiK3Parser.required_call_channel(K3Tokenizer())
TOOLS_ARG = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def _toy_channel():
    """after = 10 11 12, opening = 20 21, of which 20 21 opens the channel."""
    return RequiredChannel(
        after=(10, 11, 12), opening=(20, 21), opens_at=2, end_ids=(2,)
    )


# ── the channel Kimi-K3 defines ──────────────────────────────────────────


def test_k3_opens_the_tools_channel_where_its_response_closes():
    assert K3_CHANNEL == RequiredChannel(
        after=(CLOSE, RESPONSE, SEP),
        opening=(OPEN, TOOLS, SEP, OPEN, CALL, TOOL_ATTR, EQ_QUOTE),
        opens_at=3,  # <|open|>tools<|sep|> opens the channel itself
        end_ids=(END_OF_MSG,),
    )


def _engine_io():
    """A real `InputOutputProcessor` over an engine stopping on its EOS and
    `stop_token_ids` [<|end_of_msg|>, 7]."""
    from atom.model_engine.llm_engine import InputOutputProcessor

    io = InputOutputProcessor.__new__(InputOutputProcessor)
    io.config = types.SimpleNamespace(
        hf_config=types.SimpleNamespace(),
        eos_token_id=EOS,
        stop_token_ids=[END_OF_MSG, 7],
        enable_dp_attention=False,
    )
    io.block_size = 16
    io.num_speculative_tokens = 0
    io.has_per_req_cache = False
    io.segment_tokenizer = None
    io.requests = {}
    io._format_channel = io._required_channel = None
    return io


def test_engine_holds_back_every_id_it_stops_on():
    """The format names `<|end_of_msg|>`; the engine also stops on its EOS
    and `stop_token_ids`, and a stop id left out would end the message before
    the channel opens."""
    params = SamplingParams(required_channel=K3_CHANNEL)

    (seq,) = _engine_io().preprocess_fanout([1, 2, 3], params)

    assert seq.required_channel.end_ids == (7, EOS, END_OF_MSG)
    assert seq.required_channel.after == K3_CHANNEL.after
    assert seq.required_channel.opening == K3_CHANNEL.opening
    assert params.required_channel is K3_CHANNEL  # the caller's is untouched


def test_the_engine_builds_its_channel_once_and_shares_it():
    io = _engine_io()
    params = SamplingParams(required_channel=K3_CHANNEL)
    (first,) = io.preprocess_fanout([1, 2, 3], params)
    (again,) = io.preprocess_fanout([4, 5], params)
    assert again.required_channel is first.required_channel


def test_a_request_naming_another_channel_is_refused_at_admission():
    """Refused alone, where it is admitted; the engine loop batches every
    request together and holds one channel."""
    io = _engine_io()
    io.preprocess_fanout([1, 2, 3], SamplingParams(required_channel=K3_CHANNEL))
    with pytest.raises(ValueError, match="another channel"):
        io.preprocess_fanout([4, 5], SamplingParams(required_channel=_toy_channel()))


def test_ending_on_adds_only_real_ids():
    ch = _toy_channel()
    assert ch.ending_on([None, -1, 2]) is ch
    assert ch.ending_on([5, 2, 3]).end_ids == (2, 3, 5)


def test_k3_constrains_nothing_when_its_control_tokens_split(caplog):
    """Forcing a split spelling would hand the model ids it never writes, and
    the server says so at startup."""
    tokenizer = K3Tokenizer(split_controls=True)
    with caplog.at_level("WARNING", logger="atom"):
        assert KimiK3Parser.required_call_channel(tokenizer) is None
    assert "will not hold output to the tools channel" in caplog.text


def test_k3_constrains_nothing_when_the_opening_does_not_start_with_its_channel(
    caplog,
):
    """`opens_at` counts `<|open|>tools<|sep|>` at the head of the opening; a
    tokenizer that encodes that head differently in context would misplace
    where the channel counts as open."""

    class MergingTokenizer(K3Tokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens)
            if text.startswith("<|open|>tools<|sep|><|open|>call"):
                ids[1] = 99999  # `tools` read another way in context
            return ids

    with caplog.at_level("WARNING", logger="atom"):
        assert KimiK3Parser.required_call_channel(MergingTokenizer()) is None
    assert "head of the forced opening" in caplog.text


def test_the_server_derives_k3s_channel_once_beside_its_parser():
    assert required_call_channel(KimiK3Parser, K3Tokenizer()) == K3_CHANNEL
    assert required_call_channel(None, K3Tokenizer()) is None


@pytest.mark.parametrize(
    "parser",
    [DsmlParser, DsmlV41Parser, GlmParser, KimiParser, MiniMaxParser, QwenXmlParser],
)
def test_other_formats_stay_unconstrained_even_when_a_call_is_required(parser):
    channel = required_call_channel(parser, K3Tokenizer())
    assert channel is None
    assert channel_for_request(channel, "required", TOOLS_ARG) is None


@pytest.mark.parametrize(
    "tool_choice",
    [None, "auto", "none", {"type": "function", "function": {"name": "f"}}],
)
def test_k3_constrains_only_a_required_call(tool_choice):
    assert channel_for_request(K3_CHANNEL, tool_choice, TOOLS_ARG) is None


def test_k3_required_call_needs_tools_to_call():
    assert channel_for_request(K3_CHANNEL, "required", []) is None
    assert channel_for_request(None, "required", TOOLS_ARG) is None
    assert channel_for_request(K3_CHANNEL, "required", TOOLS_ARG) is K3_CHANNEL


# ── the endpoints that build a chat request's sampling params ────────────
#
# Each chat entrypoint gates the server's channel itself; a site that stops
# passing it samples a required request unconstrained, and nothing else
# would notice.


class _Submitted(Exception):
    """Raised by a stub engine once it holds the request's sampling params."""


def _k3_chat(tool_choice):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS_ARG,
        "tool_choice": tool_choice,
    }


@pytest.mark.parametrize("tool_choice", ["required", "auto"])
def test_openai_chat_endpoint_hands_the_channel_to_sampling(monkeypatch, tool_choice):
    import asyncio

    from atom.entrypoints.openai import api_server
    from atom.entrypoints.openai.protocol import ChatCompletionRequest

    seen = []

    async def generate(_prompt, sampling_params, _request_id, **_kw):
        seen.append(sampling_params)
        yield {
            "text": "hello",
            "finish_reason": "eos",
            "num_tokens_input": 1,
            "num_tokens_output": 1,
        }

    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "reasoning_toggle", None)
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", KimiK3Parser)
    monkeypatch.setattr(api_server, "required_tool_channel", K3_CHANNEL)
    monkeypatch.setattr(api_server, "tokenizer", K3Tokenizer())
    monkeypatch.setattr(api_server, "apply_chat_template", lambda *a, **k: "prompt")
    monkeypatch.setattr(api_server, "generate_async", generate)

    request = ChatCompletionRequest(**_k3_chat(tool_choice))
    asyncio.run(api_server.chat_completions(request, None))
    (sampling_params,) = seen
    expected = K3_CHANNEL if tool_choice == "required" else None
    assert sampling_params.required_channel == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_choice", ["required", "auto"])
def test_atomesh_chat_endpoints_hand_the_channel_to_sampling(
    monkeypatch, stream, tool_choice
):
    standalone = pytest.importorskip(
        "atom.entrypoints.atomesh.atom_standalone_service", exc_type=ImportError
    )
    seen = []

    def submit(_prompt, sampling_params, *_args, **_kw):
        seen.append(sampling_params)
        raise _Submitted

    service = standalone.AtomStandaloneService.__new__(standalone.AtomStandaloneService)
    service.model_name = "m"
    service.default_chat_template_kwargs = {}
    service.custom_message_encoder = None
    service.tool_parser_cls = KimiK3Parser
    service.required_tool_channel = K3_CHANNEL
    service.tokenizer = K3Tokenizer()
    service.engine_service = types.SimpleNamespace(generate=submit, start_stream=submit)
    monkeypatch.setattr(standalone, "apply_chat_template", lambda *a, **k: "prompt")

    entry = (
        service.start_chat_completions_stream if stream else service.chat_completions
    )
    with pytest.raises(_Submitted):
        entry(_k3_chat(tool_choice))
    (sampling_params,) = seen
    expected = K3_CHANNEL if tool_choice == "required" else None
    assert sampling_params.required_channel == expected


# ── the state machine ────────────────────────────────────────────────────


def test_states_walk_the_failure_this_fixes():
    """A response closes, and the next tokens are the tools channel's."""
    ch = K3_CHANNEL
    state = ch.advance(0, [CLOSE, RESPONSE, SEP])
    assert state == len(ch.after)  # the next token is forced
    forced = []
    while state >= 0:
        forced.append(ch.table[state])
        state = ch.step(state, ch.table[state])
    assert forced == list(ch.opening)
    assert state == UNCONSTRAINED  # open: held to nothing more


def test_thinking_and_the_response_open_do_not_trigger():
    ch = K3_CHANNEL
    think_close = [CLOSE, 39964, SEP]  # <|close|>think<|sep|>
    response_open = [OPEN, RESPONSE, SEP]
    assert ch.advance(0, think_close + response_open + [123, 456]) == 0


def test_a_repeated_close_still_matches():
    """`<|close|><|close|>response<|sep|>`: the restart lands on the second."""
    ch = K3_CHANNEL
    assert ch.advance(0, [CLOSE, CLOSE, RESPONSE, SEP]) == len(ch.after)
    assert ch.advance(0, [CLOSE, RESPONSE, CLOSE, RESPONSE, SEP]) == len(ch.after)


def test_k3_opening_the_tools_channel_itself_opens_it():
    """Any way into `<|open|>tools<|sep|>` opens the channel: straight from
    thinking, or nested in the response. Its end ids are not held back after
    the call, so `<|close|>message<|sep|><|end_of_msg|>` ends it."""
    ch = K3_CHANNEL
    think_close = [CLOSE, 39964, SEP]
    tools_open = [OPEN, TOOLS, SEP]
    assert ch.advance(0, think_close + tools_open) == UNCONSTRAINED
    assert ch.advance(0, [OPEN, RESPONSE, SEP, 123] + tools_open) == UNCONSTRAINED
    # Short of the whole opening, it is still held, and the response close
    # still forces it.
    assert ch.advance(0, [OPEN, TOOLS]) >= 0
    assert ch.advance(0, [OPEN, TOOLS, CLOSE, RESPONSE, SEP]) == len(ch.after)


def _brute_force_state(ch, tokens):
    """The state by its definition, from the whole output: no automaton."""
    out, forced_left = [], 0
    own = ch.opening[: ch.opens_at]
    for t in tokens:
        if forced_left:
            forced_left -= 1
            if not forced_left:
                return UNCONSTRAINED
            continue
        out.append(t)
        if tuple(out[-len(ch.after) :]) == ch.after:
            out, forced_left = [], len(ch.opening)
        elif tuple(out[-len(own) :]) == own:
            return UNCONSTRAINED
    if forced_left:
        return len(ch.after) + len(ch.opening) - forced_left
    for k in range(len(own) - 1, 0, -1):
        if tuple(out[-k:]) == own[:k]:
            return len(ch.after) + len(ch.opening) + k - 1
    for j in range(len(ch.after) - 1, 0, -1):
        if tuple(out[-j:]) == ch.after[:j]:
            return j
    return 0


@pytest.mark.parametrize("seed", range(6))
def test_step_is_the_definition(seed):
    """`step` against the brute-force definition, over random outputs drawn
    mostly from the two patterns' ids, for K3 and for single-id openings."""
    rng = random.Random(seed)
    for ch in (
        K3_CHANNEL,
        _toy_channel(),
        RequiredChannel(after=(10,), opening=(20, 21), opens_at=1, end_ids=(2,)),
    ):
        pool = sorted(set(ch.after + ch.opening)) + [5, 6]
        for _ in range(300):
            tokens = [rng.choice(pool) for _ in range(rng.randint(0, 14))]
            assert ch.advance(0, tokens) == _brute_force_state(ch, tokens), tokens


def test_an_opened_channel_is_unconstrained_for_good():
    ch = _toy_channel()
    assert ch.advance(0, [10, 11, 12, 20, 21]) == UNCONSTRAINED
    assert ch.advance(0, [10, 11, 12, 20, 21, 10, 11, 12]) == UNCONSTRAINED
    assert ch.advance(UNCONSTRAINED, [10, 11, 12]) == UNCONSTRAINED


def test_table_indexes_waited_and_forced_tokens_by_state():
    ch = _toy_channel()
    assert ch.table == (10, 11, 12, 20, 21, 2)
    assert ch.num_states == 6  # after 0-2, forced 3-4, own `20` at 5
    assert [ch.table[s] for s in range(5)] == [10, 11, 12, 20, 21]
    assert ch.step(5, 21) == UNCONSTRAINED  # state 5 waits for opening[1]


@pytest.mark.parametrize(
    "after,opening,opens_at,end_ids",
    [
        ((), (1,), 1, ()),  # nothing to wait for
        ((1,), (), 1, ()),  # nothing to force
        ((1,), (2, 3), 0, ()),  # opens_at out of 1..len(opening)
        ((1,), (2, 3), 3, ()),
        ((7, 8, 7), (1,), 1, ()),  # after's first id recurs in it
        ((7, 8), (7, 9), 2, ()),  # both patterns start alike
        ((7, 8), (1, 7), 2, ()),  # after's first id inside the own start
        ((7, 1), (1, 9), 2, ()),  # the own start's first id inside after
        ((7, 8), (1, 9, 1), 3, ()),  # the own start's first id recurs in it
        ((7, 8), (1, 9), 2, (8,)),  # an end id the channel needs written
        ((7, 8), (1, 9), 1, (9,)),
    ],
)
def test_channel_rejects_shapes_the_matcher_cannot_follow(
    after, opening, opens_at, end_ids
):
    with pytest.raises(ValueError):
        RequiredChannel(
            after=after, opening=opening, opens_at=opens_at, end_ids=end_ids
        )


def test_an_engine_stop_id_the_channel_needs_is_refused():
    with pytest.raises(ValueError, match="part of the channel"):
        K3_CHANNEL.ending_on([SEP])


# ── the host's copy of the state ─────────────────────────────────────────


def _channel_params(channel=None, max_tokens=32):
    return SamplingParams(max_tokens=max_tokens, required_channel=channel)


def test_sequences_start_at_zero_or_unconstrained(seq_factory):
    ch = _toy_channel()
    assert (
        seq_factory([3, 4], sampling_params=_channel_params(ch)).required_channel_state
        == 0
    )
    assert seq_factory([3, 4]).required_channel_state == UNCONSTRAINED


def test_batch_stages_each_row_state_and_the_one_channel(seq_factory):
    ch = _toy_channel()
    a = seq_factory([3, 4], sampling_params=_channel_params(ch))
    b = seq_factory([5, 6])
    a.required_channel_state = 2
    for s in (a, b):
        s.type = SequenceType.DECODE
    batch = ScheduledBatch(
        seqs={s.id: s for s in (a, b)},
        num_scheduled_tokens=[1, 1],
        total_tokens_num=2,
        total_seqs_num=2,
    )
    assert batch.required_channel == ch
    assert batch.required_channel_states.tolist() == [2, UNCONSTRAINED]


def test_batch_without_a_channel_carries_none(seq_factory):
    s = seq_factory([3, 4])
    s.type = SequenceType.DECODE
    batch = ScheduledBatch(
        seqs={s.id: s}, num_scheduled_tokens=[1], total_tokens_num=1, total_seqs_num=1
    )
    assert batch.required_channel is None
    assert batch.required_channel_states is None


def test_a_batch_whose_channels_are_all_open_carries_none(seq_factory):
    """No states, no kernels, once every request has opened its channel."""
    s = seq_factory([3, 4], sampling_params=_channel_params(_toy_channel()))
    s.type = SequenceType.DECODE
    s.required_channel_state = s.required_channel.advance(0, [10, 11, 12, 20, 21])
    batch = ScheduledBatch(
        seqs={s.id: s}, num_scheduled_tokens=[1], total_tokens_num=1, total_seqs_num=1
    )
    assert batch.required_channel is None
    assert batch.required_channel_states is None


def test_batch_refuses_two_channels(seq_factory):
    """Admission refuses a second channel first; the batch asserts it."""
    other = RequiredChannel(after=(30,), opening=(31,), opens_at=1, end_ids=())
    seqs = [
        seq_factory([3, 4], sampling_params=_channel_params(_toy_channel())),
        seq_factory([5, 6], sampling_params=_channel_params(other)),
    ]
    for s in seqs:
        s.type = SequenceType.DECODE
    with pytest.raises(AssertionError):
        ScheduledBatch(
            seqs={s.id: s for s in seqs},
            num_scheduled_tokens=[1, 1],
            total_tokens_num=2,
            total_seqs_num=2,
        )


def _output(seq, tokens):
    return ScheduledBatchOutput(
        req_ids=[seq.id],
        token_ids=[tuple(tokens)],
        num_rejected=None,
        num_bonus=None,
        draft_token_ids=None,
    )


def test_postprocess_advances_over_the_tokens_it_commits(seq_factory):
    ch = _toy_channel()
    sched = Scheduler(MockConfig())
    seq = seq_factory([3, 4, 5, 6], sampling_params=_channel_params(ch))
    sched.add(seq)
    batch, _ = sched.schedule()
    assert batch.required_channel_states.tolist() == [0]

    sched.postprocess(list(sched.running), _output(seq, [10]), batch=batch)
    assert seq.required_channel_state == 1
    batch, _ = sched.schedule()
    assert batch.required_channel_states.tolist() == [1]

    sched.postprocess(list(sched.running), _output(seq, [11]), batch=batch)
    batch, _ = sched.schedule()
    sched.postprocess(list(sched.running), _output(seq, [12]), batch=batch)
    assert seq.required_channel_state == len(ch.after)


def test_pd_first_token_counts_once(seq_factory):
    """A P/D consumer appends the producer's T0 at admission, and the first
    output after it carries T0 again in `new_tokens`; it is counted at
    admission only.

    T0 completes `after`, so every token from it on moves the state: counted
    twice it would land one past where the output really is.
    """
    ch = RequiredChannel(after=(10,), opening=(20, 21, 22), opens_at=1, end_ids=(2,))
    sched = Scheduler(MockConfig())
    seq = seq_factory(
        [3, 4, 5, 6],
        sampling_params=_channel_params(ch),
        kv_transfer_params={"first_token_id": 10},
    )
    seq.num_cached_tokens = seq.num_prompt_tokens
    sched._schedule_first_decode_after_remote_kv(seq)
    assert seq.required_channel_state == 1  # the next token is forced

    batch, _ = sched.schedule()
    assert batch.required_channel_states.tolist() == [1]
    sched.postprocess(list(sched.running), _output(seq, [20]), batch=batch)
    assert seq.required_channel_state == 2  # counted twice it would be 3
    assert ch.advance(0, [10, 10, 20]) == 3
