# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 tool calls held to their schemas by the structured-output grammar."""

import asyncio
import json
import re
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("llguidance")
torch = pytest.importorskip("torch")

from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import (
    KimiK3Parser,
)
from atom.entrypoints.openai.tool_parser.tool_parser import (
    ToolCallParser,
)
from atom.structured_output import backend_guidance as guidance
from atom.structured_output import manager, validation

# A byte-level tiktoken encoding with K3's control tokens as specials.
SPECIALS = ["<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>"]
ENCODING = NS(
    _mergeable_ranks={bytes([b]): b for b in range(256)},
    _special_tokens={name: 256 + i for i, name in enumerate(SPECIALS)},
    _pat_str=r"[\s\S]",
)
EOS = ENCODING._special_tokens["<|end_of_msg|>"]
OPEN = ENCODING._special_tokens["<|open|>"]
CLOSE = ENCODING._special_tokens["<|close|>"]
VOCAB = 256 + len(SPECIALS)
TOKENIZER = NS(model=ENCODING)

WRITE = {
    "type": "function",
    "function": {
        "name": "write",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}, "path": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}
SLEEP = {
    "type": "function",
    "function": {
        "name": "sleep",
        "parameters": {
            "type": "object",
            "properties": {"seconds": {"type": "integer"}},
            "required": ["seconds"],
        },
    },
}
THOUGHT = "plan<|close|>think<|sep|><|open|>response<|sep|>\n<|close|>response<|sep|>"
CALL_HEAD = '<|open|>tools<|sep|><|open|>call tool="write" index="1"<|sep|>'


def arg(key, value, kind="string"):
    return (
        f'<|open|>argument key="{key}" type="{kind}"<|sep|>{value}'
        "<|close|>argument<|sep|>"
    )


def ids(text):
    out = []
    for part in re.split("(" + "|".join(map(re.escape, SPECIALS)) + ")", text):
        out += (
            [ENCODING._special_tokens[part]]
            if part in SPECIALS
            else list(part.encode())
        )
    return out


def matcher(tools, choice="auto", starts_in_reasoning=True):
    fmt = validation.tool_call_grammar_format(
        KimiK3Parser, tools, choice, starts_in_reasoning
    )
    compiler = guidance.GrammarCompiler(
        guidance.grammar_tokenizer(TOKENIZER, VOCAB, [EOS])
    )
    compiled = compiler.compile_json_schema(validation.TOOL_CALL_GRAMMAR_SCHEMA, fmt)
    return guidance.GrammarMatcher(compiled)


def feed(m, text):
    return all(m.accept_token(t) for t in ids(text))


def allowed(m, token):
    mask = torch.zeros((1, (VOCAB + 31) // 32), dtype=torch.int32)
    m.fill_next_token_bitmask(mask)
    return bool((int(mask[0, token // 32]) >> (token % 32)) & 1)


def test_empty_call_cannot_close():
    # The KVV ToolCall-bench failure: the model copies `write {}` from history.
    m = matcher([WRITE])
    assert feed(m, THOUGHT + CALL_HEAD)
    assert not allowed(m, CLOSE)
    assert allowed(m, OPEN)


def test_call_cannot_close_before_every_required_argument():
    m = matcher([WRITE])
    assert feed(m, THOUGHT + CALL_HEAD + arg("path", "a.py"))
    assert not allowed(m, CLOSE)
    assert feed(m, arg("content", "x"))
    assert allowed(m, CLOSE)


@pytest.mark.parametrize("first", ["path", "content"])
def test_complete_call_in_either_argument_order(first):
    second = "content" if first == "path" else "path"
    m = matcher([WRITE])
    message = (
        THOUGHT
        + CALL_HEAD
        + arg(first, "a")
        + arg(second, "b")
        + "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
    )
    assert feed(m, message)
    assert m.accept_token(EOS) and m.is_terminated()


def test_answer_without_a_call_is_free():
    m = matcher([WRITE])
    assert feed(m, THOUGHT.replace("\n", "any text <tag>") + "<|close|>message<|sep|>")
    assert m.accept_token(EOS)


def test_undeclared_tool_and_key_are_refused():
    m = matcher([WRITE])
    assert not feed(m, THOUGHT + '<|open|>tools<|sep|><|open|>call tool="rm" ')
    m = matcher([WRITE])
    assert not feed(m, THOUGHT + CALL_HEAD + arg("mode", "w"))


def test_non_string_argument_follows_its_schema():
    m = matcher([SLEEP])
    head = THOUGHT + '<|open|>tools<|sep|><|open|>call tool="sleep" index="1"<|sep|>'
    assert not feed(m, head + arg("seconds", "soon", kind="number"))
    m = matcher([SLEEP])
    assert feed(m, head + arg("seconds", "3", kind="number") + "<|close|>call<|sep|>")


def test_named_choice_allows_only_that_tool_and_requires_a_call():
    choice = {"type": "function", "function": {"name": "sleep"}}
    m = matcher([WRITE, SLEEP], choice)
    assert feed(m, THOUGHT)
    assert not allowed(m, CLOSE)  # the message cannot close without the call
    assert not feed(m, CALL_HEAD)


def test_generation_that_starts_in_the_response_channel():
    m = matcher([WRITE], starts_in_reasoning=False)
    assert feed(m, "done<|close|>response<|sep|>" + CALL_HEAD)
    assert not allowed(m, CLOSE)


@pytest.mark.parametrize(
    "parser, tools, choice",
    [
        (KimiK3Parser, [WRITE], "none"),
        (KimiK3Parser, [], "auto"),
        (KimiK3Parser, [{"type": "custom", "name": "x"}], "auto"),
        (ToolCallParser, [WRITE], "auto"),
        (None, [WRITE], "auto"),
    ],
)
def test_no_grammar_when_no_function_call_can_be_held(parser, tools, choice):
    assert validation.tool_call_grammar_format(parser, tools, choice, True) is None


def test_tiktoken_fingerprint_tracks_the_mapping_and_stops():
    base = manager.tokenizer_fingerprint(TOKENIZER, [EOS])
    assert base == manager.tokenizer_fingerprint(TOKENIZER, [EOS])
    assert base != manager.tokenizer_fingerprint(TOKENIZER, [EOS, OPEN])
    other = NS(model=NS(**{**vars(ENCODING), "_pat_str": r"\w+"}))
    assert base != manager.tokenizer_fingerprint(other, [EOS])


def _params():
    return NS(
        json_schema=None,
        structured_tool_format=None,
        structured_reasoning_end="",
        required_channel="held",
        ignore_eos=False,
        stop_strings=None,
        logprobs=None,
    )


def _configure(params, engine, **kwargs):
    return asyncio.run(
        validation.configure_tool_call_grammar(
            engine,
            NS(kv_transfer_config={"kv_role": "offload"}),
            params,
            ReasoningChannel(starts_open=True),
            parser_cls=KimiK3Parser,
            tools=[WRITE],
            tool_choice="auto",
            **kwargs,
        )
    )


def test_configure_keys_the_request_and_admits_local_kv_offload():
    calls = []
    engine = NS(prepare_structured_request=lambda *args: calls.append(args))
    params = _params()
    assert _configure(params, engine)
    assert params.json_schema == validation.TOOL_CALL_GRAMMAR_SCHEMA
    assert json.loads(params.structured_tool_format)["kind"] == "lark"
    assert params.required_channel is None and len(calls) == 1


@pytest.mark.parametrize("case", ["multimodal", "stop", "disabled", "missing"])
def test_requests_that_cannot_run_a_grammar_stay_unconstrained(case, monkeypatch):
    calls = []
    engine = NS(prepare_structured_request=lambda *args: calls.append(args))
    params = _params()
    kwargs = {}
    if case == "multimodal":
        kwargs["multimodal"] = True
    elif case == "stop":
        params.stop_strings = ["END"]
    elif case == "disabled":
        monkeypatch.setenv("ATOM_TOOL_CALL_GRAMMAR", "0")
    else:
        monkeypatch.setattr(validation.structured_output, "available", lambda: False)
    assert not _configure(params, engine, **kwargs)
    assert params.json_schema is None and params.required_channel == "held"
    assert not calls


def test_grammar_compile_failure_falls_back_to_unconstrained():
    def refuse(*args):
        raise ValueError("unsupported schema keyword")

    params = _params()
    assert not _configure(params, NS(prepare_structured_request=refuse))
    assert params.json_schema is None and params.structured_tool_format is None
    assert params.required_channel == "held"
