# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Chat completions prepared in the tokenize workers (`chat_offload`).

The same requests go through `/v1/chat/completions` twice, once as the server
answers them today and once with `ATOM_API_OFFLOAD_PREPROCESS`'s route, and
what reaches the engine -- prompt ids, sampling parameters, the shipped prompt
hash chain -- and what the response is built with -- tools, tool_choice, the
reasoning channel -- must be identical, as must every error response.

A small trained BPE tokenizer with a Qwen-shaped chat template stands in for a
model. To run the same comparison on real traffic, point
``ATOM_TEST_CHAT_OFFLOAD_MODEL`` at a checkpoint and
``ATOM_TEST_CHAT_OFFLOAD_BODIES`` at request bodies (u32 little-endian length,
then the body, repeated).
"""

import array
import asyncio
import functools
import json
import os
import random
import struct
from types import SimpleNamespace

import pytest

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")
httpx = pytest.importorskip("httpx")

from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)

from atom.entrypoints.openai import api_server, chat_offload
from atom.entrypoints.openai.chat_encoders import (
    chat_template_source,
    load_custom_message_encoder,
    render_probe_prompt,
    resolve_reasoning_toggle,
)
from atom.entrypoints.openai.chat_offload import (
    ChatOffload,
    ChatPrepareSettings,
    prepare_chat,
)
from atom.entrypoints.openai.reasoning import (
    template_opens_reasoning_implicitly,
)
from atom.entrypoints.openai.reasoning_dialects import resolve_dialect
from atom.entrypoints.openai.streaming_dispatch import (
    StreamBatchDispatcher,
)
from atom.entrypoints.openai.tokenize_pool import TokenizePool
from atom.model_engine import prefix_hash
from atom.model_engine.llm_engine import InputOutputProcessor
from atom.model_engine.request import RequestOutput
from atom.model_engine.tokenizer_loader import load_tokenizer

REAL_MODEL = os.environ.get("ATOM_TEST_CHAT_OFFLOAD_MODEL")
REAL_BODIES = os.environ.get("ATOM_TEST_CHAT_OFFLOAD_BODIES")
MODEL_NAME = REAL_MODEL or "m"
BLOCK_SIZE = 16

WORDS = [
    "the",
    "order",
    "tool",
    "call",
    "result",
    "customer",
    "refund",
    "package",
    "delivery",
    "status",
    "def",
    "return",
    "value",
    "import",
    "class",
    "self",
    "print",
    "error",
    "test",
    "file",
    "path",
    "json",
    "list",
    "dict",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "kappa",
    "omega",
    "sigma",
    "theta",
    "lambda",
    "zeta",
    "héllo",
    "你好",
    "{",
    "}",
    "[",
    "]",
    '"',
    ":",
    ",",
    ".",
    "(",
    ")",
    "=",
    "+",
    "-",
    "_",
    "/",
    "\\",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]
SPECIALS = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
]
# Qwen-shaped: tools as JSON, tool calls, a reasoning block per assistant
# turn, and a generation prompt that opens `<think>` unless `enable_thinking`
# is off.
TEMPLATE = (
    "{%- if tools %}<|im_start|>system\n# Tools\n{%- for t in tools %}\n"
    "{{ t | tojson }}{%- endfor %}<|im_end|>\n{%- endif %}"
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n"
    "{%- if m.reasoning_content %}<think>\n{{ m.reasoning_content }}\n</think>\n"
    "{%- endif %}{{ m.content if m.content is string else '' }}"
    "{%- if m.tool_calls %}{%- for c in m.tool_calls %}\n<tool_call>\n"
    "{{ c.function.name }} {{ c.function.arguments | tojson }}\n</tool_call>"
    "{%- endfor %}{%- endif %}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and not enable_thinking %}"
    "<think>\n\n</think>\n\n{%- else %}<think>\n{%- endif %}{%- endif %}"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Look up the {name.split('_')[-1]} of an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
    for name in ("get_order", "track_package", "check_refund_status")
]


def _save_tokenizer(path, seed=0, vocab_size=400):
    rng = random.Random(seed)
    corpus = [" ".join(rng.choice(WORDS) for _ in range(60)) for _ in range(300)]
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    backend.add_special_tokens([AddedToken(t, special=True) for t in SPECIALS])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend)
    tok.chat_template = TEMPLATE
    tok.save_pretrained(str(path))
    return str(path)


def _text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _conversation(rng, turns, words):
    messages = [{"role": "system", "content": f"[salt: {rng.randrange(9999)}] "}]
    messages[0]["content"] += _text(rng, words)
    messages.append({"role": "user", "content": _text(rng, words)})
    for t in range(turns):
        call = {
            "id": f"call_{t}",
            "type": "function",
            "function": {
                "name": rng.choice(TOOLS)["function"]["name"],
                "arguments": json.dumps({"order_id": f"{rng.randrange(1 << 40):X}"}),
            },
        }
        messages.append(
            {
                "role": "assistant",
                "content": None if rng.random() < 0.5 else _text(rng, 8),
                "reasoning_content": _text(rng, words // 4),
                "tool_calls": [call],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{t}",
                "content": _text(rng, words),
            }
        )
    messages.append({"role": "user", "content": _text(rng, words // 2)})
    return messages


def _synthetic_bodies():
    """Agentic-shaped requests as the MLPerf endpoints client sends them, plus
    the knobs the handler reads: thinking switches, tool_choice, stop, both
    max-token spellings, non-streaming."""
    rng = random.Random(11)
    bodies = []
    for i in range(48):
        long = i in (5, 31)
        body = {
            "model": MODEL_NAME,
            "messages": _conversation(
                rng,
                turns=60 if long else rng.randrange(0, 12),
                words=400 if long else 40,
            ),
            "stream": i % 4 != 3,
            "max_completion_tokens": 8192,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "chat_template_kwargs": {"preserve_thinking": True},
            "tools": TOOLS,
        }
        variant = i % 8
        if variant == 1:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif variant == 2:
            body["thinking"] = {"type": "disabled"}
        elif variant == 3:
            body["tool_choice"] = "auto"
            body["reasoning_effort"] = "high"
        elif variant == 4:
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": "get_order"},
            }
        elif variant == 5:
            del body["max_completion_tokens"]
            body["max_tokens"] = 512
            body["stop"] = ["</s>", "\n\n\n"]
            body["top_k"] = 0
        elif variant == 6:
            del body["tools"]
            body["temperature"] = 0.0
            body["n"] = 1
        elif variant == 7:
            body["response_format"] = {"type": "json_object"}
            body["seed"] = 7
        bodies.append(json.dumps(body).encode())
    return bodies


def _real_bodies():
    with open(REAL_BODIES, "rb") as f:
        data = f.read()
    bodies, i = [], 0
    while i < len(data):
        (n,) = struct.unpack_from("<I", data, i)
        bodies.append(data[i + 4 : i + 4 + n])
        i += 4 + n
    return bodies


class _Engine:
    """The engine as far as the handler reaches it: a real
    `InputOutputProcessor` (Sequence, prompt hash chain), and a core manager
    that records each request and finishes it with one token."""

    def __init__(self, tokenizer, reply_id):
        config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
            speculative_config=None,
            enable_prefix_caching=True,
            decode_context_parallel_size=1,
            enable_dp_attention=False,
            max_model_len=1 << 22,
        )
        self.config = config
        self.io_processor = InputOutputProcessor(config, tokenizer, BLOCK_SIZE)
        self.core_mgr = self
        self.max_pool_tokens = None
        self.reply_id = reply_id
        self.seen = []
        preprocess_fanout = self.io_processor.preprocess_fanout

        def recording_preprocess_fanout(prompt, sampling_params, **kwargs):
            seqs = preprocess_fanout(prompt, sampling_params, **kwargs)
            self.seen.append(
                {
                    "sampling_params": sampling_params,
                    "worker_hashes": kwargs.get("prompt_block_hashes") is not None,
                    "seqs": seqs,
                }
            )
            return seqs

        self.io_processor.preprocess_fanout = recording_preprocess_fanout

    def add_request(self, seqs):
        for seq in seqs:
            callback, seq.stream_callback = seq.stream_callback, None
            callback(RequestOutput(seq.id, [self.reply_id], True, "eos"))
        api_server.flush_stream_batch()

    def abort_request(self, seq_id):
        pass


def _configure_server(monkeypatch, tokenizer, model_path):
    encoder = load_custom_message_encoder(model_path)
    source = chat_template_source(tokenizer, encoder)
    dialect, _ = resolve_dialect(
        source, render_probe_prompt(tokenizer, encoder, tools=False) or ""
    )
    monkeypatch.setenv("ATOM_TOKENIZE_CACHE_MB", "0")
    monkeypatch.setattr(api_server, "tokenizer", tokenizer)
    monkeypatch.setattr(api_server, "model_name", MODEL_NAME)
    monkeypatch.setattr(api_server, "custom_message_encoder", encoder)
    monkeypatch.setattr(api_server, "reasoning_dialect", dialect)
    monkeypatch.setattr(
        api_server,
        "model_starts_in_reasoning",
        template_opens_reasoning_implicitly(source),
    )
    toggle = resolve_reasoning_toggle(tokenizer, encoder)
    monkeypatch.setattr(api_server, "reasoning_toggle", toggle)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(
        api_server,
        "_stream_batch_dispatcher",
        StreamBatchDispatcher(tokenizer, synthetic_text=None),
    )
    engine = _Engine(tokenizer, reply_id=tokenizer.encode("ok")[0])
    monkeypatch.setattr(api_server, "engine", engine)
    # What the response is built with.
    built = []
    for name in ("stream_chat_response", "build_chat_response"):
        original = getattr(api_server, name)

        def recording(*args, _original=original, **kwargs):
            built.append({k: kwargs[k] for k in ("tools", "tool_choice", "reasoning")})
            return _original(*args, **kwargs)

        monkeypatch.setattr(api_server, name, recording)
    return engine, built, encoder, toggle


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    if REAL_MODEL:
        return REAL_MODEL
    return _save_tokenizer(tmp_path_factory.mktemp("tok") / "tok")


@pytest.fixture(scope="module")
def bodies():
    return _real_bodies() if REAL_BODIES else _synthetic_bodies()


@pytest.fixture(scope="module")
def pool_and_settings(model_path):
    tokenizer = load_tokenizer(model_path, True)
    settings = ChatPrepareSettings(
        chat_template=None,
        default_chat_template_kwargs={},
        reasoning_toggle=resolve_reasoning_toggle(
            tokenizer, load_custom_message_encoder(model_path)
        ),
        prefix_hash_block_size=BLOCK_SIZE,
    )
    pool = TokenizePool(
        model_path, True, 8 if REAL_MODEL else 2, tokenizer, chat_settings=settings
    )
    try:
        yield pool, settings
    finally:
        pool.shutdown()


@pytest.fixture
def server(monkeypatch, model_path, pool_and_settings):
    """The app with the offload route installed; `use(on)` picks the path."""
    pool, settings = pool_and_settings
    tokenizer = load_tokenizer(model_path, True)
    engine, built, encoder, toggle = _configure_server(
        monkeypatch, tokenizer, model_path
    )
    assert toggle == settings.reasoning_toggle
    monkeypatch.setattr(api_server, "_tokenize_pool", pool)
    offload = ChatOffload(
        pool,
        functools.partial(
            prepare_chat,
            tokenizer=tokenizer,
            custom_encoder=encoder,
            settings=settings,
        ),
        api_server._offloaded_chat_sampling_params,
    )
    route = next(
        r for r in api_server.app.router.routes if r.path == "/v1/chat/completions"
    )
    original_app = route.app
    api_server._offload_chat_completions_route()

    def use(on):
        monkeypatch.setattr(api_server, "_chat_offload", offload if on else None)

    try:
        yield SimpleNamespace(
            engine=engine, built=built, offload=offload, use=use, tokenizer=tokenizer
        )
    finally:
        route.app = original_app


async def _post(client, body, content_type="application/json"):
    response = await client.post(
        "/v1/chat/completions",
        content=body,
        headers={"content-type": content_type, "x-session-id": "s"},
    )
    return response.status_code, response.text


def _run(server, on, bodies, content_type="application/json"):
    server.use(on)
    server.engine.seen.clear()
    server.built.clear()

    async def go():
        transport = httpx.ASGITransport(app=api_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=600
        ) as client:
            return [await _post(client, b, content_type) for b in bodies]

    return asyncio.run(go())


def _answers(responses):
    """Status and answer text, without per-request ids and timings."""
    out = []
    for status, text in responses:
        if status != 200:
            out.append((status, text))
        elif text.startswith("data:"):
            frames = [
                json.loads(line[5:])
                for line in text.splitlines()
                if line.startswith("data:") and line[5:].strip() != "[DONE]"
            ]
            out.append(
                (
                    status,
                    [
                        (c["delta"], c["finish_reason"])
                        for f in frames
                        for c in f["choices"]
                    ],
                )
            )
        else:
            data = json.loads(text)
            out.append((status, data["choices"], data["usage"]["prompt_tokens"]))
    return out


def _engine_view(engine):
    """What reached the engine, per request."""
    out = []
    for item in engine.seen:
        seq = item["seqs"][0]
        out.append(
            (
                array.array("i", seq.token_ids),
                item["sampling_params"],
                seq.prompt_hash_chain,
            )
        )
    return out


def test_offloaded_requests_reach_the_engine_unchanged(server, bodies, monkeypatch):
    before = _run(server, False, bodies)
    engine_before = _engine_view(server.engine)
    built_before = list(server.built)

    hashed_here = []
    chained = prefix_hash.chained_block_hashes

    def counting(*args, **kwargs):
        hashed_here.append(1)
        return chained(*args, **kwargs)

    monkeypatch.setattr(prefix_hash, "chained_block_hashes", counting)
    after = _run(server, True, bodies)
    engine_after = _engine_view(server.engine)
    built_after = list(server.built)

    assert server.offload.enabled
    assert len(engine_before) == len(engine_after) == len(bodies)
    identical_ids = sum(a[0] == b[0] for a, b in zip(engine_before, engine_after))
    print(f"\n{len(bodies)} requests, {identical_ids} with identical prompt ids")
    assert identical_ids == len(bodies)
    for (ids_a, sp_a, chain_a), (ids_b, sp_b, chain_b) in zip(
        engine_before, engine_after
    ):
        assert sp_a == sp_b
        assert chain_a is not None and chain_a == chain_b
    assert built_before == built_after
    assert _answers(before) == _answers(after)
    # Every offloaded prompt's chain came from its worker.
    assert all(item["worker_hashes"] for item in server.engine.seen)
    assert not hashed_here


def test_fast_streaming_answers_as_before(server, bodies, monkeypatch):
    """ATOM_API_STREAM_FAST: the backend decode and one write per chunk leave
    every streamed answer as it was (events may share a write)."""
    from atom.entrypoints.openai import streaming_dispatch

    before = _answers(_run(server, True, bodies))
    monkeypatch.setattr(streaming_dispatch._BACKEND_DECODE, "decode", None)
    assert streaming_dispatch.enable_backend_decode(server.tokenizer)
    monkeypatch.setattr(
        api_server,
        "_stream_fast_kwargs",
        {"frames_per_chunk": True, "min_frame_interval": 0.001},
    )
    after = _answers(_run(server, True, bodies))
    assert before == after


HI = [{"role": "user", "content": "hi"}]
ERROR_AND_FALLBACK_BODIES = [
    b'{"model": "m", "messages": [}',
    b'{"model": "m"}',
    b"[1, 2]",
    b"",
    *(
        json.dumps({"model": "m", "messages": HI, **fields}).encode()
        for fields in (
            {"tools": [{"type": "nope"}]},
            {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
            {"model": "other"},
            {"stream": True, "return_token_ids": True},
            {"max_completion_tokens": 0},
            {"top_p": 0},
            {"reasoning_effort": 0},
            {"n": 2},
            {"prompt_token_ids": [5, 6, 7]},
            {"messages": None, "prompt": HI},
        )
    ),
]


@pytest.mark.parametrize("case", ERROR_AND_FALLBACK_BODIES)
def test_errors_and_fallbacks_answer_as_before(server, case):
    before = _answers(_run(server, False, [case]))
    after = _answers(_run(server, True, [case]))
    assert before == after
    assert server.offload.enabled


def test_a_body_fastapi_does_not_read_as_json_stays_with_fastapi(server):
    body = b'{"model": "m", "messages": [{"role": "user", "content": "hi"}]}'
    for content_type in ("text/plain", "application/x-www-form-urlencoded"):
        before = _answers(_run(server, False, [body], content_type))
        after = _answers(_run(server, True, [body], content_type))
        assert before == after


def test_prepare_chat_leaves_what_it_does_not_cover(model_path, pool_and_settings):
    _, settings = pool_and_settings
    tokenizer = load_tokenizer(model_path, True)

    def prepared(**fields):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        body.update(fields)
        return prepare_chat(json.dumps(body).encode(), tokenizer, None, settings)

    image = [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    assert prepared() is not None
    assert prepared(messages=[{"role": "user", "content": image}]) is None
    assert prepared(prompt_token_ids=[1, 2, 3]) is None
    assert prepared(n=3) is None
    assert prepared(tools="not a list") is None
    result = prepared(stream=True, tools=TOOLS)
    assert result.request.messages is None and result.request.tools == TOOLS
    assert result.block_hashes.num_tokens == len(result.prompt_ids)


def test_a_difference_turns_the_offload_off(model_path, pool_and_settings, tmp_path):
    pool, settings = pool_and_settings
    other = load_tokenizer(_save_tokenizer(tmp_path / "other", seed=5, vocab_size=330))
    offload = ChatOffload(
        pool,
        functools.partial(
            prepare_chat, tokenizer=other, custom_encoder=None, settings=settings
        ),
        api_server._offloaded_chat_sampling_params,
    )
    body = _synthetic_bodies()[0]
    result = asyncio.run(offload.prepare(body))
    assert not offload.enabled
    # The request is served from the in-process result.
    assert result == prepare_chat(body, other, None, settings)
    assert (
        result.prompt_ids
        != asyncio.run(pool.run(chat_offload.prepare_chat_in_worker, body)).prompt_ids
    )


def test_difference_names_the_field():
    request = SimpleNamespace()
    a = chat_offload.PreparedChat(request, {}, array.array("i", [1, 2]), True, None)
    b = a._replace(prompt_ids=array.array("i", [1, 3]))
    assert chat_offload.difference(a, a, lambda r: 0) is None
    assert chat_offload.difference(a, b, lambda r: 0) == "prompt_ids"
    assert chat_offload.difference(a, a._replace(prompt_opens_reasoning=False), str)
    assert chat_offload.difference(None, None, str) is None
    assert chat_offload.difference(None, a, str) is not None
