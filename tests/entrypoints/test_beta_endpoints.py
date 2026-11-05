# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek beta endpoints: chat prefix completion and fill-in-the-middle.

Set ATOM_DSV41_MODEL to a local V4.1 checkpoint for the cases compared with
the official encoder; the rest run without a checkpoint or GPU.
"""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from atom.entrypoints.openai import api_server
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
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)
from atom.entrypoints.openai.serving_completion import stream_completion_response
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


@pytest.mark.parametrize(
    "prompt, prefix, expected",
    [
        # Reasoning off: the reply starts with the content.
        ("<A></think>", AssistantPrefix("x"), ("<A></think>x", False)),
        ("<A>", AssistantPrefix("x"), ("<A>x", False)),
        # Reasoning on, content only: the channel is closed empty first.
        ("<A><think>", AssistantPrefix("x"), ("<A><think></think>x", False)),
        # A chain-of-thought prefix is continued inside the channel...
        ("<A><think>", AssistantPrefix("", "r"), ("<A><think>r", True)),
        # ...unless the answer has begun too.
        ("<A><think>", AssistantPrefix("x", "r"), ("<A><think>r</think>x", False)),
        # Whitespace after the opener is the template's, and is kept.
        ("<A><think>\n", AssistantPrefix("", "r"), ("<A><think>\nr", True)),
        # Other dialects close the channel with their own marker.
        (
            "<A><mm:think>",
            AssistantPrefix("x", "r"),
            ("<A><mm:think>r</mm:think>x", False),
        ),
        # Kimi-K3 then opens the answer's own channel, as its encoder does.
        (
            f"<A>{K3_THINK}",
            AssistantPrefix("x", "r"),
            (f"<A>{K3_THINK}r{K3_THINK_END}{K3_RESPONSE}x", False),
        ),
        (
            f"<A>{K3_THINK}",
            AssistantPrefix("x"),
            (f"<A>{K3_THINK}{K3_THINK_END}{K3_RESPONSE}x", False),
        ),
        (f"<A>{K3_THINK}", AssistantPrefix("", "r"), (f"<A>{K3_THINK}r", True)),
        # With thinking off its prompt has opened the answer already.
        (f"<A>{K3_RESPONSE}", AssistantPrefix("x"), (f"<A>{K3_RESPONSE}x", False)),
    ],
)
def test_prefix_continues_the_rendered_turn(prompt, prefix, expected):
    assert continue_prompt(prompt, prefix) == expected


def test_reasoning_prefix_needs_an_open_reasoning_channel():
    with pytest.raises(ValueError, match="thinking mode"):
        continue_prompt("<A></think>", AssistantPrefix("x", "r"))


# --- fill-in-the-middle format -------------------------------------------------


class _Tokenizer:
    def __init__(self, vocab, adds_bos):
        self._vocab = vocab
        self.bos_token = "<s>"
        self._adds_bos = adds_bos

    def get_vocab(self):
        return self._vocab

    def encode(self, _text):
        return [self._vocab["<s>"]] if self._adds_bos else []


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


# --- endpoints -------------------------------------------------------------------


@pytest.fixture
def engine(monkeypatch):
    """Stub generation; `reply` is what the model writes after the prompt."""
    state = SimpleNamespace(prompts=[], reply="tail")

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


def _chat(handler, messages, **kwargs):
    request = ChatCompletionRequest(model="m", messages=messages, **kwargs)
    response = asyncio.run(handler(request, None))
    return json.loads(response.body)["choices"][0]["message"]


def _complete(handler, **kwargs):
    response = asyncio.run(handler(CompletionRequest(model="m", **kwargs), None))
    return json.loads(response.body)["choices"][0]


PREFIX = {"role": "assistant", "content": "```python\n", "prefix": True}


def test_beta_chat_continues_the_prefix(engine, monkeypatch):
    monkeypatch.setattr(api_server, "apply_chat_template", lambda *a, **k: "<A>")
    message = _chat(api_server.beta_chat_completions, [USER, PREFIX])
    assert engine.prompts == ["<A>```python\n"]
    # The reply carries only what the model added.
    assert message["content"] == "tail"


def test_reasoning_prefix_reply_is_split_at_the_models_close(engine, monkeypatch):
    monkeypatch.setattr(api_server, "apply_chat_template", lambda *a, **k: "<A><think>")
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
    monkeypatch.setattr(api_server, "apply_chat_template", lambda *a, **k: "<A><think>")
    monkeypatch.setattr(api_server, "model_starts_in_reasoning", True)
    message = _chat(api_server.beta_chat_completions, [USER, PREFIX])
    assert engine.prompts == ["<A><think></think>```python\n"]
    assert message["content"] == "tail"
    assert not message.get("reasoning_content")


@pytest.mark.parametrize(
    "handler, kwargs",
    [
        # Prefix completion is a beta feature.
        (api_server.chat_completions, {}),
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
        (api_server.completions, {"prompt": "p", "suffix": "s"}, DEEPSEEK_FIM),
        (api_server.beta_completions, {"prompt": [5, 6], "suffix": "s"}, DEEPSEEK_FIM),
        (
            api_server.beta_completions,
            {"prompt": "p", "suffix": "s", "prompt_token_ids": [5]},
            DEEPSEEK_FIM,
        ),
        (api_server.beta_completions, {"prompt": "p", "suffix": "s"}, None),
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


def test_echo_returns_the_prompt_ahead_of_the_completion(engine):
    choice = _complete(api_server.beta_completions, prompt="def f(a):", echo=True)
    assert choice["text"] == "def f(a):tail"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": "p", "suffix": "s", "echo": True},
        {"prompt": "p", "logprobs": 2},
        {"prompt": "p", "stream_options": {"include_usage": True}},
        {"prompt": "p", "stream": True, "stream_options": {"include_usage": "yes"}},
    ],
)
def test_completion_options_outside_the_contract_are_400(engine, kwargs):
    with pytest.raises(HTTPException) as exc:
        _complete(api_server.beta_completions, **kwargs)
    assert exc.value.status_code == 400
    assert not engine.prompts


def test_stream_echoes_the_prompt_and_puts_usage_on_every_chunk():
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
            echo="def f():",
            include_usage=True,
        )
        return "".join([frame async for frame in stream])

    payloads = [
        json.loads(data)
        for data in iter_sse_data(asyncio.run(frames()))
        if data != "[DONE]"
    ]
    assert payloads[0]["choices"][0]["text"] == "def f():"
    assert all("usage" in payload for payload in payloads)
    # Null on the echo, content and finish chunks; counted on the last one.
    assert [p["usage"] for p in payloads if p["choices"]] == [None, None, None]
    assert payloads[-1]["usage"]["completion_tokens"] == 1


# --- against the official V4.1 encoder ------------------------------------------


@pytest.fixture(scope="module")
def v41_encoder():
    directory = Path(os.environ.get("ATOM_DSV41_MODEL", "/mnt/DeepSeek-V4.1-Flash"))
    if not (directory / "encoding/encoding.py").is_file():
        pytest.skip("set ATOM_DSV41_MODEL for official-encoder differential tests")
    return _load_encoder_from_dir(str(directory))


@pytest.mark.parametrize("mode", ["chat", "thinking"])
def test_v41_prefix_matches_official_unterminated_assistant_turn(
    engine, monkeypatch, v41_encoder, mode
):
    """The official encoder renders an assistant turn without its EOS when
    asked (`wo_eos`); a continued prefix must be exactly that prompt."""
    monkeypatch.setattr(api_server, "custom_message_encoder", v41_encoder)
    reasoning = "Partition first." if mode == "thinking" else None
    prefix = {**PREFIX, "reasoning_content": reasoning} if reasoning else PREFIX
    history = [
        {"role": "system", "content": "Be brief."},
        USER,
    ]
    _chat(
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
                "content": PREFIX["content"],
                "reasoning_content": reasoning,
                "wo_eos": True,
            },
        ],
        thinking_mode=mode,
    )
    assert engine.prompts == [official]
