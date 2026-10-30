# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`tool_choice="required"` on Kimi-K3: which requests get a `RequiredChannel`,
what it holds the output to, and how the host keeps each sequence's state.

The GPU half -- masking target rows and carrying the state across a deferred
step -- is `tests/test_required_channel_mask.py`, which needs Triton.
"""

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
from atom.entrypoints.openai.tool_parser.registry import required_call_channel
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
    return RequiredChannel(after=(10, 11, 12), opening=(20, 21), end_ids=(2,))


# ── the channel Kimi-K3 defines ──────────────────────────────────────────


def test_k3_opens_the_tools_channel_where_its_response_closes():
    assert K3_CHANNEL == RequiredChannel(
        after=(CLOSE, RESPONSE, SEP),
        opening=(OPEN, TOOLS, SEP, OPEN, CALL, TOOL_ATTR, EQ_QUOTE),
        end_ids=(END_OF_MSG,),
    )


def test_engine_holds_back_every_id_it_stops_on():
    """The format names `<|end_of_msg|>`; the engine also stops on its EOS
    and `stop_token_ids`, and a stop id left out would end the message before
    the channel opens."""
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
    params = SamplingParams(required_channel=K3_CHANNEL)

    (seq,) = io.preprocess_fanout([1, 2, 3], params)

    assert seq.required_channel.end_ids == (7, EOS, END_OF_MSG)
    assert seq.required_channel.after == K3_CHANNEL.after
    assert seq.required_channel.opening == K3_CHANNEL.opening
    assert params.required_channel is K3_CHANNEL  # the caller's is untouched


def test_ending_on_adds_only_real_ids():
    ch = _toy_channel()
    assert ch.ending_on([None, -1, 2]) is ch
    assert ch.ending_on([5, 2, 3]).end_ids == (2, 3, 5)


def test_k3_constrains_nothing_when_its_control_tokens_split():
    """Forcing a split spelling would hand the model ids it never writes."""
    assert KimiK3Parser.required_call_channel(K3Tokenizer(split_controls=True)) is None


def test_k3_channel_is_encoded_once_per_tokenizer():
    class CountingTokenizer(K3Tokenizer):
        calls = 0

        def encode(self, text, add_special_tokens=False):
            CountingTokenizer.calls += 1
            return super().encode(text, add_special_tokens)

    tokenizer = CountingTokenizer()
    first = required_call_channel(KimiK3Parser, tokenizer, "required", TOOLS_ARG)
    calls = CountingTokenizer.calls
    again = required_call_channel(KimiK3Parser, tokenizer, "required", TOOLS_ARG)
    assert again is first == K3_CHANNEL
    assert CountingTokenizer.calls == calls


@pytest.mark.parametrize(
    "parser",
    [DsmlParser, DsmlV41Parser, GlmParser, KimiParser, MiniMaxParser, QwenXmlParser],
)
def test_other_formats_stay_unconstrained_even_when_a_call_is_required(parser):
    assert required_call_channel(parser, K3Tokenizer(), "required", TOOLS_ARG) is None


@pytest.mark.parametrize(
    "tool_choice",
    [None, "auto", "none", {"type": "function", "function": {"name": "f"}}],
)
def test_k3_constrains_only_a_required_call(tool_choice):
    assert (
        required_call_channel(KimiK3Parser, K3Tokenizer(), tool_choice, TOOLS_ARG)
        is None
    )


def test_k3_required_call_needs_tools_to_call():
    assert required_call_channel(KimiK3Parser, K3Tokenizer(), "required", []) is None
    assert required_call_channel(None, K3Tokenizer(), "required", TOOLS_ARG) is None
    assert (
        required_call_channel(KimiK3Parser, K3Tokenizer(), "required", TOOLS_ARG)
        == K3_CHANNEL
    )


# ── the endpoints that build a chat request's sampling params ────────────
#
# Each chat entrypoint resolves the channel itself; a site that stops passing
# it samples a required request unconstrained, and nothing else would notice.


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
    while state < ch.open_state:
        forced.append(ch.table[state])
        state = ch.step(state, ch.table[state])
    assert forced == list(ch.opening)
    assert state == ch.open_state


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


def test_open_is_absorbing_and_unconstrained_never_moves():
    ch = _toy_channel()
    assert ch.advance(ch.open_state, [10, 11, 12, 20]) == ch.open_state
    assert ch.advance(UNCONSTRAINED, [10, 11, 12]) == UNCONSTRAINED


def test_table_indexes_waited_and_forced_tokens_by_state():
    ch = _toy_channel()
    assert ch.table == (10, 11, 12, 20, 21, 2)
    assert [ch.table[s] for s in range(ch.open_state)] == [10, 11, 12, 20, 21]


@pytest.mark.parametrize(
    "after,opening",
    [((), (1,)), ((1,), ()), ((7, 8, 7), (1,))],
)
def test_channel_rejects_shapes_the_matcher_cannot_follow(after, opening):
    with pytest.raises(ValueError):
        RequiredChannel(after=after, opening=opening, end_ids=())


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


def test_batch_refuses_two_channels(seq_factory):
    other = RequiredChannel(after=(30,), opening=(31,), end_ids=())
    seqs = [
        seq_factory([3, 4], sampling_params=_channel_params(_toy_channel())),
        seq_factory([5, 6], sampling_params=_channel_params(other)),
    ]
    for s in seqs:
        s.type = SequenceType.DECODE
    with pytest.raises(ValueError):
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
    admission only."""
    ch = _toy_channel()
    sched = Scheduler(MockConfig())
    seq = seq_factory(
        [3, 4, 5, 6],
        sampling_params=_channel_params(ch),
        kv_transfer_params={"first_token_id": 10},
    )
    seq.num_cached_tokens = seq.num_prompt_tokens
    sched._schedule_first_decode_after_remote_kv(seq)
    assert seq.required_channel_state == 1

    batch, _ = sched.schedule()
    assert batch.required_channel_states.tolist() == [1]
    sched.postprocess(list(sched.running), _output(seq, [11]), batch=batch)
    assert seq.required_channel_state == 2
