# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek beta endpoints: chat prefix completion and fill-in-the-middle.

Set ATOM_DSV41_MODEL to a local V4.1 checkpoint for the cases compared with
the official encoder; the rest run without a checkpoint or GPU.
"""

import asyncio
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from atom.entrypoints.openai import api_server, request_timing
from atom.entrypoints.openai.api_compatibility import DEEPSEEK_API_COMPATIBILITY
from atom.entrypoints.openai.chat_encoders import (
    _load_encoder_from_dir,
    apply_chat_template,
)
from atom.entrypoints.openai.fim import KNOWN_FIM_FORMATS, resolve_fim_format
from atom.entrypoints.openai.prefix_completion import (
    AssistantPrefix,
    continue_prompt,
    take_assistant_prefix,
)
from atom.entrypoints.openai.protocol import (
    BetaChatCompletionRequest,
    BetaCompletionRequest,
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)
from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.entrypoints.openai.reasoning_dialects import DIALECTS, FALLBACK_DIALECT
from atom.entrypoints.openai.serving_completion import (
    stream_completion_response,
    stream_completion_response_fanout,
)
from atom.entrypoints.openai.sse import iter_sse_data
from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector

USER = {"role": "user", "content": "Write quick sort"}


def _messages(*dicts):
    return [ChatMessage(**d) for d in dicts]


# --- take_assistant_prefix ---------------------------------------------------


def test_no_prefix_leaves_messages_alone():
    messages = _messages(USER, {"role": "assistant", "content": "x", "prefix": False})
    assert take_assistant_prefix(messages) == (messages, None)


def test_trailing_prefix_is_split_off():
    messages = _messages(
        USER,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "```python\n"}],
            "reasoning_content": "Plan.",
            "prefix": True,
        },
    )
    rest, prefix = take_assistant_prefix(messages)
    assert rest == messages[:1]
    assert prefix == AssistantPrefix("```python\n", "Plan.")


TWO_BLOCKS = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]


def test_a_prefix_goes_through_the_encoders_own_message_processing():
    """As the rendered messages do; without one, its text is joined as a
    chat template gets it, one line apart."""
    seen = []

    def process(messages):
        seen.extend(messages)
        return [
            {**m, "content": "joined", "reasoning_content": "r2"} for m in messages
        ], []

    messages = _messages(
        USER,
        {
            "role": "assistant",
            "content": TWO_BLOCKS,
            "reasoning_content": "r",
            "prefix": True,
        },
    )
    encoder = SimpleNamespace(preserve_content=True, process_image_messages=process)
    _, prefix = take_assistant_prefix(messages, encoder)
    assert seen == [
        {"role": "assistant", "content": TWO_BLOCKS, "reasoning_content": "r"}
    ]
    assert prefix == AssistantPrefix("joined", "r2")
    _, prefix = take_assistant_prefix(messages)
    assert prefix == AssistantPrefix("first\nsecond", "r")


def test_an_unset_prefix_may_be_spelled_null():
    messages = _messages(
        USER, {"role": "assistant", "content": "x", "prefix": None}, USER
    )
    assert take_assistant_prefix(messages) == (messages, None)


@pytest.mark.parametrize(
    "messages",
    [
        # Not the last message.
        [
            USER,
            {"role": "assistant", "content": "a", "prefix": True},
            {"role": "user", "content": "b"},
        ],
        # Not an assistant message.
        [{"role": "user", "content": "a", "prefix": True}],
        # Nothing for the template to open an assistant turn after.
        [{"role": "assistant", "content": "a", "prefix": True}],
        [
            USER,
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b", "prefix": True},
        ],
        # Not a flag.
        [USER, {"role": "assistant", "content": "a", "prefix": "yes"}],
        # Not text.
        [
            USER,
            {
                "role": "assistant",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                "prefix": True,
            },
        ],
        [
            USER,
            {
                "role": "assistant",
                "content": "a",
                "prefix": True,
                "reasoning_content": 3,
            },
        ],
        [
            USER,
            {
                "role": "assistant",
                "content": None,
                "prefix": True,
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
        ],
    ],
)
def test_malformed_prefix_is_rejected(messages):
    with pytest.raises(ValueError):
        take_assistant_prefix(_messages(*messages))


# --- continue_prompt -----------------------------------------------------------

# Kimi-K3's channel framing, spelled out as its encoder writes it.
K3_THINK = "<|open|>think<|sep|>"
K3_THINK_END = "<|close|>think<|sep|>"
K3_RESPONSE = "<|open|>response<|sep|>"


THINK = FALLBACK_DIALECT
MINIMAX = next(d for d in DIALECTS if d.think_end_marker == "</mm:think>")
K3 = next(d for d in DIALECTS if d.answer_open_marker)


def _opened(dialect=THINK):
    return ReasoningChannel(dialect=dialect, starts_open=True)


def _closed(dialect=THINK):
    return ReasoningChannel(dialect=dialect, starts_open=False)


@pytest.mark.parametrize(
    "prompt, channel, prefix, expected",
    [
        # Reasoning off: the reply starts with the content.
        ("<A></think>", _closed(), AssistantPrefix("x"), ("<A></think>x", False)),
        ("<A>", _closed(), AssistantPrefix("x"), ("<A>x", False)),
        # An empty content prefix has begun nothing: an ordinary reply.
        ("<A></think>", _closed(), AssistantPrefix(""), ("<A></think>", False)),
        # Reasoning on, content only: the channel is closed empty first.
        ("<A><think>", _opened(), AssistantPrefix("x"), ("<A><think></think>x", False)),
        # A chain-of-thought prefix is continued inside the channel...
        ("<A><think>", _opened(), AssistantPrefix("", "r"), ("<A><think>r", True)),
        # ...unless the answer has begun too.
        (
            "<A><think>",
            _opened(),
            AssistantPrefix("x", "r"),
            ("<A><think>r</think>x", False),
        ),
        # Whitespace after the opener is the template's, and is kept.
        ("<A><think>\n", _opened(), AssistantPrefix("", "r"), ("<A><think>\nr", True)),
        # A model that starts in reasoning with no marker (DeepSeek-R1) is open
        # too, though its prompt shows no opener.
        ("<A>", _opened(), AssistantPrefix("x"), ("<A></think>x", False)),
        ("<A>", _opened(), AssistantPrefix("", "r"), ("<A>r", True)),
        # Other dialects close the channel with their own marker.
        (
            "<A><mm:think>",
            _opened(MINIMAX),
            AssistantPrefix("x", "r"),
            ("<A><mm:think>r</mm:think>x", False),
        ),
        # Kimi-K3 then opens the answer's own channel, as its encoder does.
        (
            f"<A>{K3_THINK}",
            _opened(K3),
            AssistantPrefix("x", "r"),
            (f"<A>{K3_THINK}r{K3_THINK_END}{K3_RESPONSE}x", False),
        ),
        (
            f"<A>{K3_THINK}",
            _opened(K3),
            AssistantPrefix("x"),
            (f"<A>{K3_THINK}{K3_THINK_END}{K3_RESPONSE}x", False),
        ),
        (
            f"<A>{K3_THINK}",
            _opened(K3),
            AssistantPrefix("", "r"),
            (f"<A>{K3_THINK}r", True),
        ),
        # With thinking off its prompt has opened the answer already.
        (
            f"<A>{K3_RESPONSE}",
            _closed(K3),
            AssistantPrefix("x"),
            (f"<A>{K3_RESPONSE}x", False),
        ),
    ],
)
def test_prefix_continues_the_rendered_turn(prompt, channel, prefix, expected):
    expected_prompt, in_reasoning = expected
    continued, reading = continue_prompt(prompt, prefix, channel)
    assert continued == expected_prompt
    assert reading.dialect is channel.dialect
    assert reading.starts_open is in_reasoning
    # A content prefix has begun the answer, with thinking on or off.
    assert reading.answer_started is (bool(prefix.content) and not in_reasoning)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_reasoning_prefix_needs_a_reply_that_begins_in_reasoning(dialect):
    # Thinking off, or a model that opens the channel itself (Qwen3).
    with pytest.raises(ValueError, match="thinking mode"):
        continue_prompt("<A>", AssistantPrefix("x", "r"), _closed(dialect))


def _read(channel, text, chunk):
    """``(reasoning, content)`` of ``text``: whole, as a JSON reply is read, or
    arriving ``chunk`` characters at a time, as a stream is."""
    if chunk is None:
        return channel.split(text, stopped=True)
    reader = channel.stream()
    events = []
    for i in range(0, len(text), chunk):
        events += reader.process(text[i : i + chunk])
    events += reader.flush(stopped=True)
    reasoning = "".join(v for k, v in events if k == "reasoning_content")
    return reasoning or None, "".join(v for k, v in events if k == "content")


@pytest.mark.parametrize("chunk", [None, 1, 4])
@pytest.mark.parametrize("reasoning", [None, "Plan."])
@pytest.mark.parametrize("dialect", [THINK, MINIMAX, K3])
def test_a_continued_answer_reads_as_the_models_own_would(dialect, reasoning, chunk):
    """The prefix closed the channel, so what follows is the answer: think
    markers in it are the model's words, as when the model closes the channel
    itself, and the format's answer framing is still removed."""
    prefix = AssistantPrefix("XML example: ", reasoning)
    words = f"<a>{dialect.output_open_marker}x{dialect.think_end_marker}</a>"
    continuation = words + "".join(dialect.answer_end_markers)
    _, channel = continue_prompt("<A>", prefix, _opened(dialect))
    whole = (
        (reasoning or "")
        + dialect.think_end_marker
        + dialect.answer_open_marker
        + prefix.content
        + continuation
    )
    assert _read(_opened(dialect), whole, chunk) == (
        reasoning,
        prefix.content + words,
    )
    assert _read(channel, continuation, chunk) == (None, words)


@pytest.mark.parametrize("chunk", [None, 1, 4])
@pytest.mark.parametrize("dialect", [THINK, MINIMAX, K3])
def test_a_continued_answer_reads_the_same_with_thinking_off(dialect, chunk):
    """The content began the answer in either mode, so the continuation is
    read alike: think markers in it are the model's words."""
    prefix = AssistantPrefix("XML example: ")
    words = f"<a>{dialect.output_open_marker}x{dialect.think_end_marker}</a>"
    continuation = words + "".join(dialect.answer_end_markers)
    _, thinking = continue_prompt("<A>", prefix, _opened(dialect))
    _, chat = continue_prompt("<A>", prefix, _closed(dialect))
    assert _read(chat, continuation, chunk) == (None, words)
    assert _read(chat, continuation, chunk) == _read(thinking, continuation, chunk)


# --- fill-in-the-middle format -------------------------------------------------


class _Tokenizer:
    """Vocabulary entries encode as themselves; other text one ID a character."""

    def __init__(self, vocab, adds_bos, splits=()):
        self._vocab = vocab
        self.bos_token = "<s>"
        self._adds_bos = adds_bos
        self._splits = splits

    def get_vocab(self):
        return self._vocab

    def encode(self, text, add_special_tokens=True):
        ids = [self._vocab["<s>"]] if self._adds_bos and add_special_tokens else []
        if text in self._vocab and text not in self._splits:
            return ids + [self._vocab[text]]
        return ids + [1000 + ord(c) for c in text]


DEEPSEEK_FIM = KNOWN_FIM_FORMATS[0]
FIM_VOCAB = {"<s>": 0, DEEPSEEK_FIM.begin: 1, DEEPSEEK_FIM.hole: 2, DEEPSEEK_FIM.end: 3}


@pytest.mark.parametrize("adds_bos, bos", [(False, "<s>"), (True, "")])
def test_fim_format_writes_bos_exactly_once(adds_bos, bos):
    fim = resolve_fim_format(_Tokenizer(FIM_VOCAB, adds_bos))
    assert fim.render("P", "S") == (
        f"{bos}{DEEPSEEK_FIM.begin}P{DEEPSEEK_FIM.hole}S{DEEPSEEK_FIM.end}"
    )


def test_fim_format_needs_every_marker_as_a_token():
    vocab = {k: v for k, v in FIM_VOCAB.items() if k != DEEPSEEK_FIM.hole}
    assert resolve_fim_format(_Tokenizer(vocab, False)) is None


def test_fim_format_needs_markers_that_encode_as_one_token():
    # In the vocabulary, but the tokenizer splits it in text.
    tokenizer = _Tokenizer(FIM_VOCAB, False, splits=(DEEPSEEK_FIM.hole,))
    assert resolve_fim_format(tokenizer) is None


def test_a_tokenizer_that_cannot_answer_leaves_fim_off_instead_of_failing():
    class Broken(_Tokenizer):
        def get_vocab(self):
            raise NotImplementedError

    assert resolve_fim_format(Broken(FIM_VOCAB, False)) is None


# --- endpoints -------------------------------------------------------------------


@pytest.fixture
def engine(monkeypatch):
    """Stub generation; `reply` is what the model writes after the prompt."""
    state = SimpleNamespace(prompts=[], rendered=[], reply="tail")

    async def generate(prompt, *_args, **_kwargs):
        state.prompts.append(prompt)
        yield {
            "text": state.reply,
            "finish_reason": "eos",
            "num_tokens_input": 4,
            "num_tokens_output": 1,
        }

    async def consume(generator, *_args):
        async for output in generator:
            pass
        return output

    for name, value in {
        "model_name": "m",
        "_request_logger": None,
        "default_chat_template_kwargs": {},
        "reasoning_toggle": None,
        "reasoning_dialect": None,
        "model_starts_in_reasoning": False,
        "custom_message_encoder": None,
        "tool_call_parser_cls": None,
        "tokenizer": None,
        "fim_format": DEEPSEEK_FIM,
        "generate_async": generate,
        "_run_nonstream_with_disconnect": consume,
    }.items():
        monkeypatch.setattr(api_server, name, value)
    return state


def _template(monkeypatch, engine, prompt):
    """Render every request as ``prompt``, recording the roles it was given."""

    def render(_tokenizer, _encoder, messages, **_kwargs):
        engine.rendered.append([m.role for m in messages])
        return prompt

    monkeypatch.setattr(api_server, "apply_chat_template", render)


def _chat(handler, messages, **kwargs):
    beta = handler is api_server.beta_chat_completions
    model = BetaChatCompletionRequest if beta else ChatCompletionRequest
    request = model(model="m", messages=messages, **kwargs)
    response = asyncio.run(handler(request, None))
    return json.loads(response.body)["choices"][0]["message"]


def _complete(handler, **kwargs):
    beta = handler is api_server.beta_completions
    model = BetaCompletionRequest if beta else CompletionRequest
    response = asyncio.run(handler(model(model="m", **kwargs), None))
    return json.loads(response.body)["choices"][0]


PREFIX = {"role": "assistant", "content": "```python\n", "prefix": True}


def test_beta_chat_continues_the_prefix(engine, monkeypatch):
    _template(monkeypatch, engine, "<A>")
    message = _chat(api_server.beta_chat_completions, [USER, PREFIX])
    # The prefix is split off the history and appended, not rendered.
    assert engine.rendered == [["user"]]
    assert engine.prompts == ["<A>```python\n"]
    # The reply carries only what the model added.
    assert message["content"] == "tail"


def test_reasoning_prefix_reply_is_split_at_the_models_close(engine, monkeypatch):
    _template(monkeypatch, engine, "<A><think>")
    engine.reply = " more.</think>answer"
    message = _chat(
        api_server.beta_chat_completions,
        [
            USER,
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Plan:",
                "prefix": True,
            },
        ],
    )
    assert engine.prompts == ["<A><think>Plan:"]
    assert message["reasoning_content"] == " more."
    assert message["content"] == "answer"


def test_content_prefix_reply_is_never_read_as_reasoning(engine, monkeypatch):
    """Even on a model whose output otherwise begins inside the channel."""
    _template(monkeypatch, engine, "<A><think>")
    monkeypatch.setattr(api_server, "model_starts_in_reasoning", True)
    message = _chat(api_server.beta_chat_completions, [USER, PREFIX])
    assert engine.prompts == ["<A><think></think>```python\n"]
    assert message["content"] == "tail"
    assert not message.get("reasoning_content")


def test_implicit_reasoning_model_closes_the_channel_before_content(
    engine, monkeypatch
):
    """DeepSeek-R1 starts in reasoning with no marker in its prompt."""
    _template(monkeypatch, engine, "<A>")
    monkeypatch.setattr(api_server, "model_starts_in_reasoning", True)
    message = _chat(api_server.beta_chat_completions, [USER, PREFIX])
    assert engine.prompts == ["<A></think>```python\n"]
    assert message["content"] == "tail"
    assert not message.get("reasoning_content")


def test_implicit_reasoning_model_continues_a_reasoning_prefix(engine, monkeypatch):
    _template(monkeypatch, engine, "<A>")
    monkeypatch.setattr(api_server, "model_starts_in_reasoning", True)
    engine.reply = " more.</think>answer"
    prefix = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Plan:",
        "prefix": True,
    }
    message = _chat(api_server.beta_chat_completions, [USER, prefix])
    assert engine.prompts == ["<A>Plan:"]
    assert (message["reasoning_content"], message["content"]) == (" more.", "answer")


@pytest.mark.parametrize(
    "rendered, starts_in_reasoning, prompt",
    [
        ("<A><think>", True, "<A><think></think>XML example: "),
        ("<A></think>", False, "<A></think>XML example: "),
    ],
    ids=["thinking", "chat"],
)
def test_think_tags_after_a_content_prefix_stay_in_the_content(
    engine, monkeypatch, rendered, starts_in_reasoning, prompt
):
    """The content prefix began the answer, so tags the model then writes are
    its answer's words, with thinking on or off."""
    _template(monkeypatch, engine, rendered)
    monkeypatch.setattr(api_server, "model_starts_in_reasoning", starts_in_reasoning)
    engine.reply = "<think>x</think>"
    prefix = {"role": "assistant", "content": "XML example: ", "prefix": True}
    message = _chat(api_server.beta_chat_completions, [USER, prefix])
    assert engine.prompts == [prompt]
    assert message["content"] == "<think>x</think>"
    assert not message.get("reasoning_content")


def test_streamed_prefix_reply_gets_the_same_prompt_and_channel(engine, monkeypatch):
    _template(monkeypatch, engine, "<A><think>")
    seen = {}

    async def setup(prompt, *_args, **_kwargs):
        seen["prompt"] = prompt
        return 7, None, 4

    def stream(*_args, reasoning, **_kwargs):
        seen["reasoning"] = reasoning

        async def frames():
            yield "data: [DONE]\n\n"

        return frames()

    monkeypatch.setattr(api_server, "setup_streaming_request", setup)
    monkeypatch.setattr(api_server, "stream_chat_response", stream)
    request = BetaChatCompletionRequest(model="m", messages=[USER, PREFIX], stream=True)
    asyncio.run(api_server.beta_chat_completions(request, None))
    assert seen["prompt"] == "<A><think></think>```python\n"
    assert seen["reasoning"].starts_open is False
    assert seen["reasoning"].answer_started is True


@pytest.mark.parametrize(
    "handler, kwargs",
    [
        # The final prompt cannot be continued when the client sent its IDs.
        (api_server.beta_chat_completions, {"prompt_token_ids": [1, 2]}),
        # A grammar would constrain text the prefix has already written.
        (
            api_server.beta_chat_completions,
            {"response_format": {"type": "json_object"}},
        ),
    ],
)
def test_prefix_rejections_are_400_before_generation(engine, handler, kwargs):
    with pytest.raises(HTTPException) as exc:
        _chat(handler, [USER, PREFIX], **kwargs)
    assert exc.value.status_code == 400
    assert not engine.prompts


def test_v1_chat_ignores_the_prefix_flag_as_before(engine, monkeypatch):
    # /v1 renders the assistant message as history, as main always has.
    _template(monkeypatch, engine, "<A>")
    message = _chat(api_server.chat_completions, [USER, PREFIX])
    assert engine.rendered == [["user", "assistant"]]
    assert engine.prompts == ["<A>"]
    assert message["content"] == "tail"


def test_beta_completions_renders_fill_in_the_middle(engine):
    choice = _complete(
        api_server.beta_completions, prompt="def f(a):", suffix="  return b"
    )
    assert engine.prompts == [DEEPSEEK_FIM.render("def f(a):", "  return b")]
    assert choice["text"] == "tail"


def test_beta_completions_without_suffix_is_plain_completion(engine):
    _complete(api_server.beta_completions, prompt="def f(a):")
    assert engine.prompts == ["def f(a):"]


@pytest.mark.parametrize(
    "handler, kwargs, fim",
    [
        (api_server.beta_completions, {"prompt": [5, 6], "suffix": "s"}, DEEPSEEK_FIM),
        (
            api_server.beta_completions,
            {"prompt": "p", "suffix": "s", "prompt_token_ids": [5]},
            DEEPSEEK_FIM,
        ),
        (api_server.beta_completions, {"prompt": "p", "suffix": "s"}, None),
        (api_server.beta_completions, {"suffix": "s"}, DEEPSEEK_FIM),
    ],
)
def test_fim_rejections_are_400_before_generation(
    engine, monkeypatch, handler, kwargs, fim
):
    monkeypatch.setattr(api_server, "fim_format", fim)
    with pytest.raises(HTTPException) as exc:
        _complete(handler, **kwargs)
    assert exc.value.status_code == 400
    assert not engine.prompts


def test_v1_completions_ignores_the_beta_parameters_as_before(engine):
    # Not declared on the /v1 model, so dropped as unknown fields, as on main.
    choice = _complete(
        api_server.completions,
        prompt="p",
        suffix="s",
        echo=True,
        logprobs=2,
        stream_options={"include_usage": True},
    )
    assert engine.prompts == ["p"]
    assert choice["text"] == "tail"


def test_v1_request_models_are_unchanged():
    for field in ("suffix", "echo", "logprobs", "stream_options"):
        assert field not in CompletionRequest.model_fields
    with pytest.raises(ValueError):
        CompletionRequest(prompt="p", stop="```")
    with pytest.raises(ValueError):
        ChatCompletionRequest(messages=[USER], stop="```")


def test_beta_stop_takes_one_string_or_a_list():
    assert BetaCompletionRequest(prompt="p", stop="```").stop == ["```"]
    assert BetaCompletionRequest(prompt="p", stop=["a", "b"]).stop == ["a", "b"]
    assert BetaChatCompletionRequest(messages=[USER], stop="```").stop == ["```"]


def test_echo_returns_the_prompt_ahead_of_the_completion(engine):
    choice = _complete(api_server.beta_completions, prompt="def f(a):", echo=True)
    assert choice["text"] == "def f(a):tail"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": "p", "suffix": "s", "echo": True},
        {"prompt": "p", "logprobs": 2},
        {"prompt": "p", "logprobs": 0},
        {"prompt": "p", "stream_options": {"include_usage": True}},
        {"prompt": "p", "stream": True, "stream_options": {"include_usage": "yes"}},
    ],
)
def test_completion_options_outside_the_contract_are_400(engine, kwargs):
    with pytest.raises(HTTPException) as exc:
        _complete(api_server.beta_completions, **kwargs)
    assert exc.value.status_code == 400
    assert not engine.prompts


@pytest.fixture
def no_generation(engine, monkeypatch):
    """What a zero-token completion uses: the prompt's tokens (one a
    character here) and the context length."""
    monkeypatch.setattr(
        api_server,
        "engine",
        SimpleNamespace(
            io_processor=SimpleNamespace(encode_prompt=lambda p: [0] * len(p))
        ),
    )
    monkeypatch.setattr(api_server, "_get_engine_max_model_len", lambda: 64)
    return engine


def _beta_complete(**kwargs):
    request = BetaCompletionRequest(model="m", **kwargs)
    response = asyncio.run(api_server.beta_completions(request, None))
    if not kwargs.get("stream"):
        return json.loads(response.body)

    async def frames():
        return "".join([frame async for frame in response.body_iterator])

    return [
        json.loads(data)
        for data in iter_sse_data(asyncio.run(frames()))
        if data != "[DONE]"
    ]


@pytest.mark.parametrize("echo, text", [(True, "return 42"), (False, "")])
def test_zero_max_tokens_generates_nothing(no_generation, echo, text):
    body = _beta_complete(prompt="return 42", echo=echo, max_tokens=0)
    assert not no_generation.prompts
    assert [(c["text"], c["finish_reason"]) for c in body["choices"]] == [
        (text, "length")
    ]
    assert body["usage"]["prompt_tokens"] == len("return 42")
    assert body["usage"]["completion_tokens"] == 0


def test_zero_max_tokens_counts_the_fill_in_the_middle_prompt(no_generation):
    body = _beta_complete(prompt="def f(a):", suffix="  return b", max_tokens=0)
    assert body["choices"][0]["text"] == ""
    rendered = DEEPSEEK_FIM.render("def f(a):", "  return b")
    assert body["usage"]["prompt_tokens"] == len(rendered)


def test_zero_max_tokens_answers_every_choice(no_generation):
    body = _beta_complete(prompt="p", echo=True, max_tokens=0, n=2, temperature=0.7)
    assert [c["text"] for c in body["choices"]] == ["p", "p"]


@pytest.mark.parametrize("n", [1, 2])
def test_zero_max_tokens_streams_the_echo_alone(no_generation, monkeypatch, n):
    monkeypatch.setattr(
        api_server, "api_compatibility_for", lambda _e: DEEPSEEK_API_COMPATIBILITY
    )
    payloads = _beta_complete(
        prompt="return 42",
        echo=True,
        max_tokens=0,
        stream=True,
        stream_options={"include_usage": True},
        n=n,
        temperature=0.7,
    )
    assert not no_generation.prompts
    texts = {}
    for payload in payloads:
        for choice in payload["choices"]:
            texts[choice["index"]] = texts.get(choice["index"], "") + choice["text"]
    assert texts == {i: "return 42" for i in range(n)}
    # DeepSeek's contract: the last chunk carries the usage.
    *before, last = payloads
    assert all(payload["usage"] is None for payload in before)
    assert last["choices"][0]["finish_reason"] == "length"
    assert last["usage"]["prompt_tokens"] == len("return 42")
    assert last["usage"]["completion_tokens"] == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("n", [1, 2])
def test_zero_max_tokens_records_no_time_to_first_token(no_generation, stream, n):
    """The echoed prompt is not a generated token."""
    observed = []
    timing = request_timing.RequestTiming(
        0.0, lambda *sample: observed.append(sample), streaming_response=stream
    )
    token = request_timing._request_timing.set(timing)
    try:
        _beta_complete(
            prompt="p", echo=True, max_tokens=0, stream=stream, n=n, temperature=0.7
        )
    finally:
        request_timing._request_timing.reset(token)
    assert observed == []


def test_zero_max_tokens_still_needs_the_prompt_to_fit(no_generation):
    with pytest.raises(HTTPException) as exc:
        _beta_complete(prompt="x" * 65, echo=True, max_tokens=0)
    assert exc.value.status_code == 400


def test_v1_completions_still_need_a_token(engine):
    with pytest.raises(HTTPException) as exc:
        _complete(api_server.completions, prompt="p", max_tokens=0)
    assert exc.value.status_code == 400
    for limit in ({"max_tokens": -1}, {"max_completion_tokens": -1}):
        with pytest.raises(ValueError):
            BetaCompletionRequest(prompt="p", **limit).get_max_tokens()
    assert (
        BetaCompletionRequest(prompt="p", max_completion_tokens=0).get_max_tokens() == 0
    )


def test_beta_logprobs_outside_deepseeks_range_is_invalid():
    for logprobs in (-1, 21, True):
        with pytest.raises(ValueError):
            BetaCompletionRequest(prompt="p", logprobs=logprobs)


def test_routes_bind_their_own_request_models():
    """/v1 keeps main's models; only the /beta routes take the new parameters."""
    routes = {
        route.path: route.endpoint
        for route in api_server.app.routes
        if hasattr(route, "endpoint")
    }
    for path, model in {
        "/v1/chat/completions": ChatCompletionRequest,
        "/v1/completions": CompletionRequest,
        "/beta/chat/completions": BetaChatCompletionRequest,
        "/beta/completions": BetaCompletionRequest,
    }.items():
        annotation = inspect.signature(routes[path]).parameters["request"].annotation
        assert annotation in (model, model.__name__), path


NO_ECHO = {"echo": "", "include_usage": False, "usage_on_finish": False}


@pytest.mark.parametrize(
    "handler, kwargs, deepseek, expected",
    [
        # /v1 streams as it always has, on every deployment.
        (api_server.completions, {}, False, NO_ECHO),
        (api_server.completions, {}, True, NO_ECHO),
        # /beta takes the deployment's streamed-usage shape, as Chat does.
        (api_server.beta_completions, {}, False, NO_ECHO),
        (
            api_server.beta_completions,
            {"echo": True, "stream_options": {"include_usage": True}},
            True,
            {"echo": "def f(a):", "include_usage": True, "usage_on_finish": True},
        ),
    ],
)
def test_completion_streams_get_their_routes_options(
    engine, monkeypatch, handler, kwargs, deepseek, expected
):
    seen = {}

    async def setup(*_args, **_kwargs):
        return 7, None, 4

    def stream(*_args, **options):
        seen.update(options)

        async def frames():
            yield "data: [DONE]\n\n"

        return frames()

    monkeypatch.setattr(api_server, "setup_streaming_request", setup)
    monkeypatch.setattr(api_server, "stream_completion_response", stream)
    if deepseek:
        monkeypatch.setattr(
            api_server,
            "api_compatibility_for",
            lambda _encoder: DEEPSEEK_API_COMPATIBILITY,
        )
    beta = handler is api_server.beta_completions
    model = BetaCompletionRequest if beta else CompletionRequest
    request = model(model="m", prompt="def f(a):", stream=True, **kwargs)
    asyncio.run(handler(request, None))
    assert {key: seen.get(key) for key in expected} == expected


def _stream_payloads(**options):
    collector = StreamOutputCollector("r")

    async def frames():
        collector.put_nowait(
            {"token_ids": [1], "text": "tail", "finished": True, "finish_reason": "eos"}
        )
        stream = stream_completion_response(
            "r",
            "m",
            collector,
            1,
            4,
            lambda *_args, **_kwargs: None,
            lambda _request_id: None,
            **options,
        )
        return "".join([frame async for frame in stream])

    return [
        json.loads(data)
        for data in iter_sse_data(asyncio.run(frames()))
        if data != "[DONE]"
    ]


def test_deepseek_stream_carries_usage_on_its_finish_chunk():
    """DeepSeek's contract: the last chunk, with `finish_reason`, carries the
    usage, and no usage-only chunk follows."""
    payloads = _stream_payloads(
        echo="def f():", include_usage=True, usage_on_finish=True
    )
    assert payloads[0]["choices"][0]["text"] == "def f():tail"
    assert all(payload["choices"] for payload in payloads)
    *before, last = payloads
    # `include_usage` puts a null usage on every chunk before the last.
    assert [payload["usage"] for payload in before] == [None]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"]["completion_tokens"] == 1


def test_deepseek_stream_without_include_usage_still_ends_with_usage():
    *before, last = _stream_payloads(usage_on_finish=True)
    assert all("usage" not in payload and payload["choices"] for payload in before)
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"]["completion_tokens"] == 1


def test_v1_stream_frames_are_unchanged():
    # /v1's defaults: no usage on the choice chunks, then a usage-only chunk,
    # whatever include_usage says -- as Chat streams without usage_on_finish.
    for options in ({}, {"include_usage": True}):
        payloads = _stream_payloads(**options)
        assert [("usage" in p, bool(p["choices"])) for p in payloads] == [
            (False, True),
            (False, True),
            (True, False),
        ]


def test_stream_sends_nothing_before_the_first_token():
    """The silence watch leaves only the wait for the first frame unwatched,
    which holds only while no stream yields before reading its collector."""
    collector = StreamOutputCollector("r")

    async def first_frame():
        stream = stream_completion_response(
            "r",
            "m",
            collector,
            1,
            4,
            lambda *_args, **_kwargs: None,
            lambda _request_id: None,
            echo="def f():",
            include_usage=True,
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stream.__anext__(), 0.05)

    asyncio.run(first_frame())


def _fanout_payloads(**options):
    collector = StreamOutputCollector("r")

    async def frames():
        for idx in (1, 0):
            collector.put_nowait(
                (
                    idx,
                    {
                        "token_ids": [1],
                        "text": f"t{idx}",
                        "finished": True,
                        "finish_reason": "eos",
                    },
                )
            )
        stream = stream_completion_response_fanout(
            "r",
            "m",
            collector,
            [5, 6],
            4,
            lambda *_args, **_kwargs: None,
            lambda _request_id: None,
            **options,
        )
        return "".join([frame async for frame in stream])

    return [
        json.loads(data)
        for data in iter_sse_data(asyncio.run(frames()))
        if data != "[DONE]"
    ]


def test_fanout_echo_leads_each_siblings_first_chunk():
    texts = {
        p["choices"][0]["index"]: p["choices"][0]["text"]
        for p in _fanout_payloads(echo="P:")
        if p["choices"] and p["choices"][0]["text"]
    }
    assert texts == {0: "P:t0", 1: "P:t1"}


def test_fanout_deepseek_usage_rides_on_the_last_finish_chunk():
    payloads = _fanout_payloads(include_usage=True, usage_on_finish=True)
    assert all(payload["choices"] for payload in payloads)
    *before, last = payloads
    assert all(payload["usage"] is None for payload in before)
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"]["completion_tokens"] == 2


# --- against the official V4.1 encoder ------------------------------------------


@pytest.fixture(scope="module")
def v41_encoder():
    directory = Path(os.environ.get("ATOM_DSV41_MODEL", "/mnt/DeepSeek-V4.1-Flash"))
    if not (directory / "encoding/encoding.py").is_file():
        pytest.skip("set ATOM_DSV41_MODEL for official-encoder differential tests")
    return _load_encoder_from_dir(str(directory))


@pytest.mark.parametrize(
    "content", [PREFIX["content"], TWO_BLOCKS], ids=["text", "blocks"]
)
@pytest.mark.parametrize(
    "mode, reasoning",
    [("chat", None), ("thinking", "Partition first."), ("thinking", None)],
    ids=["chat", "thinking", "thinking-content-only"],
)
def test_v41_prefix_matches_official_unterminated_assistant_turn(
    engine, monkeypatch, v41_encoder, mode, reasoning, content
):
    """The official encoder renders an assistant turn without its EOS when
    asked (`wo_eos`); a continued prefix must be exactly that prompt. A
    content-only prefix in thinking mode is the case where this server closes
    the reasoning channel itself."""
    monkeypatch.setattr(api_server, "custom_message_encoder", v41_encoder)
    prefix = {**PREFIX, "content": content}
    if reasoning:
        prefix["reasoning_content"] = reasoning
    history = [
        {"role": "system", "content": "Be brief."},
        USER,
    ]
    # The answer has begun in both modes: tags the model writes stay in it.
    engine.reply = "<think>x</think>"
    message = _chat(
        api_server.beta_chat_completions,
        [*history, prefix],
        chat_template_kwargs={"thinking_mode": mode},
    )
    official = apply_chat_template(
        None,
        v41_encoder,
        [
            *history,
            {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning,
                "wo_eos": True,
            },
        ],
        thinking_mode=mode,
    )
    assert engine.prompts == [official]
    assert message["content"] == "<think>x</think>"
    assert not message.get("reasoning_content")


IMAGE_TOKEN = "<\uff5cdeepseek_image\uff5c>"


@pytest.mark.parametrize(
    "fields",
    [
        {"content": f"see {IMAGE_TOKEN}"},
        {"content": [{"type": "text", "text": f"see {IMAGE_TOKEN}"}]},
        {"content": "x", "reasoning_content": f"see {IMAGE_TOKEN}"},
    ],
    ids=["content", "text-block", "reasoning"],
)
def test_v41_prefix_refuses_image_placeholders_as_the_history_does(
    engine, monkeypatch, v41_encoder, fields
):
    """The official encoder refuses an image placeholder written as text,
    and a prefix goes through the same check as the messages before it."""
    monkeypatch.setattr(api_server, "custom_message_encoder", v41_encoder)
    assistant = {"role": "assistant", **fields}
    for handler, messages in (
        (
            api_server.chat_completions,
            [USER, assistant, {"role": "user", "content": "go on"}],
        ),
        (api_server.beta_chat_completions, [USER, {**assistant, "prefix": True}]),
    ):
        with pytest.raises(HTTPException) as exc:
            _chat(handler, messages, chat_template_kwargs={"thinking_mode": "thinking"})
        assert exc.value.status_code == 400, handler.__name__
    assert not engine.prompts


STRICT_PARAMETERS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
STRICT_TOOL = {
    "type": "function",
    "function": {"name": "weather", "strict": True, "parameters": STRICT_PARAMETERS},
}


@pytest.fixture
def structured(engine, monkeypatch):
    """What the chat handler hands structured-output setup and validation."""
    seen = SimpleNamespace(configured=[], validated=[])

    async def configure(_params, _schema, _channel, **kwargs):
        seen.configured.append(kwargs["strict_schemas"])

    monkeypatch.setattr(api_server, "_configure_structured_output", configure)
    monkeypatch.setattr(
        api_server,
        "validate_tool_calls",
        lambda schemas, _calls, _reason, declared: seen.validated.append(
            (schemas, declared)
        ),
    )
    _template(monkeypatch, engine, "<A>")
    return seen


@pytest.mark.parametrize(
    "handler, tool_choice, strict",
    [
        (api_server.beta_chat_completions, "auto", True),
        (api_server.beta_chat_completions, "none", False),
        # /v1 keeps ignoring `strict`, as on main.
        (api_server.chat_completions, "auto", False),
    ],
)
def test_strict_tools_are_a_beta_feature(structured, handler, tool_choice, strict):
    _chat(handler, [USER], tools=[STRICT_TOOL], tool_choice=tool_choice)
    expected = {"weather": STRICT_PARAMETERS} if strict else {}
    assert structured.configured == [expected]
    # The non-streamed reply's calls are checked against the request's tools.
    assert structured.validated == (
        [(expected, frozenset({"weather"}))] if strict else []
    )


def test_a_prefix_cannot_be_continued_under_strict_tools(structured, engine):
    with pytest.raises(HTTPException) as exc:
        _chat(api_server.beta_chat_completions, [USER, PREFIX], tools=[STRICT_TOOL])
    assert exc.value.status_code == 400
    assert not engine.prompts and not structured.configured


def test_a_strict_reply_is_read_literally(structured, engine):
    """A reasoning marker the grammar let through, in a strict argument say,
    is the model's words, as under `response_format`."""
    engine.reply = "a <think> b"
    message = _chat(api_server.beta_chat_completions, [USER], tools=[STRICT_TOOL])
    assert message["content"] == "a <think> b"
    assert not message.get("reasoning_content")


def test_a_streamed_strict_reply_is_validated_and_read_literally(
    structured, monkeypatch
):
    seen = {}

    async def setup(*_args, **_kwargs):
        return 7, None, 4

    def stream(*_args, reasoning, **_kwargs):
        seen["reasoning"] = reasoning

        async def frames():
            yield "data: [DONE]\n\n"

        return frames()

    def validate(gen, schema, strict_schemas, declared):
        seen["validated"] = (schema, strict_schemas, declared)
        return gen

    monkeypatch.setattr(api_server, "setup_streaming_request", setup)
    monkeypatch.setattr(api_server, "stream_chat_response", stream)
    monkeypatch.setattr(api_server, "validate_chat_stream", validate)
    request = BetaChatCompletionRequest(
        model="m", messages=[USER], tools=[STRICT_TOOL], stream=True
    )
    asyncio.run(api_server.beta_chat_completions(request, None))
    assert seen["validated"] == (
        None,
        {"weather": STRICT_PARAMETERS},
        frozenset({"weather"}),
    )
    assert seen["reasoning"].literal_answer


def test_strict_tools_alone_reach_the_grammar_setup(monkeypatch):
    from atom.sampling_params import SamplingParams

    seen = []

    async def configure(*_args, **kwargs):
        seen.append(kwargs["strict_schemas"])

    monkeypatch.setattr(api_server, "configure_request", configure)
    monkeypatch.setattr(api_server, "_get_engine_config", lambda: None)
    for strict in ({"weather": STRICT_PARAMETERS}, {}):
        asyncio.run(
            api_server._configure_structured_output(
                SamplingParams(),
                None,
                ReasoningChannel(),
                tools=[STRICT_TOOL],
                tool_choice="auto",
                strict_schemas=strict,
            )
        )
    # Without a schema or strict tools there is nothing to configure.
    assert seen == [{"weather": STRICT_PARAMETERS}]


def test_a_refusal_for_strict_tools_alone_names_them(monkeypatch):
    """The refusals are worded for JSON Schema output: a client that sent
    only strict tools is told those asked for the grammar."""
    from atom.sampling_params import SamplingParams

    async def refuse(*_args, **_kwargs):
        raise ValueError("Structured output with tools requires a DSML tool parser")

    monkeypatch.setattr(api_server, "configure_request", refuse)
    monkeypatch.setattr(api_server, "_get_engine_config", lambda: None)

    def configure(schema):
        asyncio.run(
            api_server._configure_structured_output(
                SamplingParams(),
                schema,
                ReasoningChannel(),
                tools=[STRICT_TOOL],
                tool_choice="auto",
                strict_schemas={"weather": STRICT_PARAMETERS},
            )
        )

    with pytest.raises(ValueError, match="^strict function tools: Structured"):
        configure(None)
    # With `response_format` the client asked for structured output itself.
    with pytest.raises(ValueError, match="^Structured output"):
        configure(json.dumps({"type": "object"}))
