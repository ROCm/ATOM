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


HEAD = THOUGHT + '<|open|>tools<|sep|><|open|>call tool="t" index="1"<|sep|>'
END_CALL = "<|close|>call<|sep|>"


def tool(parameters):
    return {"type": "function", "function": {"name": "t", "parameters": parameters}}


def test_required_key_outside_properties_stays_required():
    m = matcher([tool({"type": "object", "required": ["path"]})])
    assert feed(m, HEAD)
    assert not allowed(m, CLOSE)
    assert feed(m, arg("path", "a") + END_CALL)


def test_undeclared_key_rule_cannot_reuse_a_declared_key():
    # SLEEP leaves additionalProperties open; `seconds` must still be a number.
    m = matcher([SLEEP])
    head = THOUGHT + '<|open|>tools<|sep|><|open|>call tool="sleep" index="1"<|sep|>'
    assert not feed(m, head + arg("seconds", "soon"))
    m = matcher([SLEEP])
    assert feed(m, head + arg("seconds", "3", kind="number") + arg("note", "x"))


def test_string_enum_is_held_to_its_literals():
    m = matcher(
        [
            tool(
                {
                    "type": "object",
                    "properties": {"mode": {"enum": ["r", "w"], "type": "string"}},
                }
            )
        ]
    )
    assert not feed(m, HEAD + arg("mode", "x"))
    m = matcher(
        [
            tool(
                {
                    "type": "object",
                    "properties": {"mode": {"enum": ["r", "w"], "type": "string"}},
                }
            )
        ]
    )
    assert feed(m, HEAD + arg("mode", "w") + END_CALL)


@pytest.mark.parametrize("bare", [False, True])
def test_refs_resolve_against_the_parameters_definitions(bare):
    edit = {
        "type": "object",
        "properties": {"line": {"type": "integer"}},
        "required": ["line"],
    }
    ref = {"$ref": "#/$defs/Edit"}
    prop = ref if bare else {"type": "array", "items": ref}
    params = {"type": "object", "properties": {"edits": prop}, "$defs": {"Edit": edit}}
    kind, good, bad = (
        ("object", '{"line": 1}', '{"line": "x"}')
        if bare
        else ("array", '[{"line": 1}]', '[{"line": "x"}]')
    )
    m = matcher([tool(params)])
    assert feed(m, HEAD + arg("edits", good, kind=kind) + END_CALL)
    m = matcher([tool(params)])
    assert not feed(m, HEAD + arg("edits", bad, kind=kind))


NODE = {  # `#` is the parameters, not the argument the value compiles alone
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "node": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "next": {"$ref": "#"}},
        }
    },
}
PAIR = {  # a ref into the parameters' own structure, as in KVV TestReferences
    "type": "object",
    "properties": {
        "node": {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"$ref": "#/properties/node/properties/a"},
            },
        }
    },
}


@pytest.mark.parametrize(
    "params, good, bad",
    [
        (
            NODE,
            '{"name": "a", "next": {"node": {"name": "b"}}}',
            '{"next": {"name": "b"}}',
        ),
        (PAIR, '{"a": "x", "b": "y"}', '{"a": "x", "b": 1}'),
    ],
)
def test_refs_into_the_parameters_resolve_against_the_parameters(params, good, bad):
    m = matcher([tool(params)])
    assert feed(m, HEAD + arg("node", good, kind="object") + END_CALL)
    m = matcher([tool(params)])
    assert not feed(m, HEAD + arg("node", bad, kind="object"))


@pytest.mark.parametrize(
    "schema, good, bad",
    [
        (
            {"type": ["integer", "null"]},
            [("3", "number"), ("null", "null")],
            [("soon", "number"), ("3", "string")],
        ),
        ({"enum": [1, 2]}, [("1", "number")], [("3", "number"), ("1", "string")]),
        ({"type": "string", "enum": [""]}, [("", "string")], [("x", "string")]),
    ],
)
def test_union_and_literal_values_are_held_to_their_schema(schema, good, bad):
    params = {"type": "object", "properties": {"v": schema}, "required": ["v"]}
    for value, kind in good:
        assert feed(matcher([tool(params)]), HEAD + arg("v", value, kind) + END_CALL)
    for value, kind in bad:
        assert not feed(matcher([tool(params)]), HEAD + arg("v", value, kind))


@pytest.mark.parametrize("referenced", [False, True])
def test_parameters_given_as_a_ref_are_held_like_inline_ones(referenced):
    args = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    params = {"$ref": "#/$defs/Args", "$defs": {"Args": args}} if referenced else args
    m = matcher([tool(params)])
    assert feed(m, HEAD) and not allowed(m, CLOSE)
    assert not feed(matcher([tool(params)]), HEAD + arg("other", "x"))
    assert feed(matcher([tool(params)]), HEAD + arg("path", "a.py") + END_CALL)


def test_literal_values_are_not_read_as_schemas():
    # A "$ref" key inside a const is data; a property named "const" is a schema.
    params = {
        "type": "object",
        "properties": {
            "c": {"type": "object", "const": {"$ref": "#"}},
            "d": {"type": "object", "properties": {"const": {"$ref": "#/x"}}},
        },
        "x": {"type": "integer"},
    }
    c = arg("c", '{"$ref": "#"}', kind="object")
    assert feed(matcher([tool(params)]), HEAD + c + END_CALL)
    good = arg("d", '{"const": 1}', kind="object")
    assert feed(matcher([tool(params)]), HEAD + good + END_CALL)
    assert not feed(
        matcher([tool(params)]), HEAD + arg("d", '{"const": "a"}', "object")
    )


def test_parameters_ref_beside_other_keywords_raises():
    params = {"$ref": "#/$defs/A", "$defs": {"A": {}}, "required": ["path"]}
    with pytest.raises(ValueError, match=r"\$ref"):
        matcher([tool(params)])


def test_free_text_cannot_spell_a_delimiter_from_ordinary_tokens():
    # Byte IDs only: the text-based parser would read the spelling as structure.
    m = matcher([WRITE])
    assert not all(m.accept_token(b) for b in b"x<|open|>tools<|sep|>")
    m = matcher([WRITE])
    assert all(m.accept_token(b) for b in b"a <|op x | sep")


def test_json_block_argument_form_is_refused():
    # KimiK3Parser reads argument tags only; a JSON block would reach the client as {}.
    m = matcher([WRITE])
    assert not feed(m, THOUGHT + CALL_HEAD + '<|open|>json type="object"<|sep|>')


def test_unsafe_schema_raises_before_any_grammar():
    unsafe = {
        "type": "object",
        "properties": {
            "p": {
                "type": "array",
                "items": {"type": "string", "pattern": "(a{1,1000}){1,1000}"},
            }
        },
    }
    with pytest.raises(ValueError):
        validation.tool_call_grammar_format(KimiK3Parser, [tool(unsafe)], "auto", True)


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
        # The prompt shows `write`; the request selects it as `files::write`.
        (KimiK3Parser, [{**WRITE, "namespace": "files"}], "auto"),
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


@pytest.fixture
def grammar_on(monkeypatch):
    monkeypatch.setenv("ATOM_TOOL_CALL_GRAMMAR", "1")
    validation._uncompilable_tool_grammars.clear()


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


OFFLOAD = NS(
    kv_transfer_config={"kv_connector": "lmcache_offload", "kv_role": "kv_both"}
)


def _configure(params, engine, config=OFFLOAD, tools=(WRITE,), **kwargs):
    return asyncio.run(
        validation.configure_tool_call_grammar(
            engine,
            config,
            params,
            ReasoningChannel(starts_open=True),
            parser_cls=KimiK3Parser,
            tools=list(tools),
            tool_choice="auto",
            **kwargs,
        )
    )


def _recorder(calls):
    return NS(prepare_structured_request=lambda *args: calls.append(args))


def test_configure_keys_the_request_and_admits_kv_offload(grammar_on):
    calls = []
    params = _params()
    assert _configure(params, _recorder(calls))
    assert params.json_schema == validation.TOOL_CALL_GRAMMAR_SCHEMA
    assert json.loads(params.structured_tool_format)["kind"] == "lark"
    assert params.required_channel is None and len(calls) == 1


def test_off_by_default():
    calls = []
    params = _params()
    assert not _configure(params, _recorder(calls))
    assert params.json_schema is None and not calls


def test_offload_admits_response_format_and_p_d_still_refuses():
    params = _params()
    validation.validate_execution_options(OFFLOAD, params)
    p_d = NS(kv_transfer_config={"kv_connector": "moriio", "kv_role": "kv_producer"})
    with pytest.raises(ValueError, match="P/D"):
        validation.validate_execution_options(p_d, params)


@pytest.mark.parametrize("case", ["multimodal", "stop", "p_d", "missing", "unsafe"])
def test_requests_that_cannot_run_a_grammar_stay_unconstrained(
    case, grammar_on, monkeypatch
):
    calls = []
    params = _params()
    kwargs = {}
    if case == "multimodal":
        kwargs["multimodal"] = True
    elif case == "stop":
        params.stop_strings = ["END"]
    elif case == "p_d":
        kwargs["config"] = NS(kv_transfer_config={"kv_connector": "moriio"})
    elif case == "missing":
        monkeypatch.setattr(validation.structured_output, "available", lambda: False)
    else:
        bad = {"type": "string", "pattern": "(a{1,1000}){1,1000}"}
        kwargs["tools"] = [tool({"type": "object", "properties": {"p": bad}})]
    assert not _configure(params, _recorder(calls), **kwargs)
    assert params.json_schema is None and params.required_channel == "held"
    assert not calls


@pytest.mark.parametrize("error", [ValueError, TimeoutError])
def test_preparation_failure_falls_back_and_only_compile_errors_are_remembered(
    error, grammar_on
):
    calls = []

    def refuse(*args):
        calls.append(args)
        raise error("worker said no")

    for _ in range(2):
        params = _params()
        params.structured_reasoning_end = "kept"
        assert not _configure(params, NS(prepare_structured_request=refuse))
        assert params.json_schema is None and params.structured_tool_format is None
        assert params.required_channel == "held"
        assert params.structured_reasoning_end == "kept"
    assert len(calls) == (1 if error is ValueError else 2)


def test_stream_without_a_response_schema_turns_a_decode_error_into_an_error_event():
    # A tool-call grammar sets no response schema; the stream wrapper still
    # reports its decode error instead of ending as a truncated success.
    frames = [
        f"data: {json.dumps({'choices': [choice]})}\n\n"
        for choice in (
            {"index": 0, "delta": {"content": "a"}},
            {"index": 0, "delta": {}, "finish_reason": "structured_output_error"},
        )
    ]

    async def source():
        for frame in frames:
            yield frame

    async def collect():
        return [f async for f in validation.validate_chat_stream(source(), None)]

    out = asyncio.run(collect())
    assert out[0] == frames[0]
    assert "structured_output_error" in out[-2] and out[-1] == "data: [DONE]\n\n"
