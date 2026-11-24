# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's prompt tokens for a set of requests, to compare with the router's.

Sends each request to ATOM's own /v1/chat/completions or /v1/completions,
served in this process as a front-end worker serves them, with an engine that
keeps the prompt tokens it is given and generates nothing. That needs ATOM's
Python dependencies, not a GPU. It writes as JSON the checkpoint, each request,
and the tokens ATOM runs it with, or the error ATOM rejects it with.

The router's test `test_requests_tokenize_as_on_atom` tokenizes each request as
HTTP routing does, with each tokenizer backend, and compares the tokens. CI's
mesh workflow runs the two in a row, for a revision of DeepSeek-V4.1-Flash
whose tokenizer files and chat encoder this script downloads from HuggingFace,
or $HF_ENDPOINT. For a checkpoint directory or a HuggingFace repo@revision, and
the built-in requests or recorded ones:

    python atom/mesh/scripts/atom_token_parity.py <checkpoint> \\
        [--requests recorded.jsonl] > atom.json
    cd atom/mesh && ATOM_TOKEN_PARITY=../../atom.json \\
        cargo test --release --lib test_requests_tokenize_as_on_atom -- \\
        --ignored --nocapture

--requests takes JSON Lines, one request per line, as
{"name": ..., "route": "/v1/chat/completions", "body": {...}}, or a bare body
(a chat if it has messages, a completion if it has a prompt). Without it, a
built-in set exercises reasoning controls, tool calls, special tokens and
prompts larger than 1 MiB. The large ones are built from the report's
`synthetic` lines, the same way the test builds them, and recorded as their
token count and the BLAKE2b digest of their tokens.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

CHAT = "/v1/chat/completions"
COMPLETION = "/v1/completions"

# What ATOM and the router read of a checkpoint to tokenize its prompts: the
# tokenizer, the model type, the chat template, and the chat encoder some
# models ship, which ATOM renders DeepSeek-V4.1's chats with.
CHECKPOINT_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "chat_template.jinja",
    "encoding/encoding*.py",
]

TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Reads a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Read README.md."},
]
ANSWERED = {
    "role": "assistant",
    "content": "It is ATOM's README.",
    "reasoning_content": "It has a title.",
}
NEXT_QUESTION = {"role": "user", "content": "List the files."}

# The lines the large prompts are made of: code, prose in several scripts,
# numbers, whitespace and text that looks like special tokens.
SNIPPETS = [
    "fn route(request: &Request) -> Result<Worker, RouteError> {",
    "    let tokens = tokenizer.encode(&prompt, false)?;",
    "    workers.iter().max_by_key(|worker| worker.cached_blocks(&tokens))",
    "}",
    "def render(messages: list[dict], **kwargs) -> str:",
    "    return template.render(messages=messages, **kwargs)",
    "# Placing requests by their tokens",
    "| policy | needs tokens | falls back to |",
    "The router places each request on the worker that holds most of its prefix.",
    "路由器把每个请求放到缓存了它最长前缀的 worker 上。",
    "ルーターは各リクエストを最も長い接頭辞を持つワーカーに送ります。",
    "يضع الموجه كل طلب على العامل الذي يحمل أطول بادئة له.",
    "👨‍👩‍👧 e\u0301 ñ ß Ω ∑ √ — “quoted” «guillemets»",
    "0x7f3a 1234567890 3.14159 -42 1e-9 2026-11-19T13:07:00Z",
    "\tindented\twith\ttabs  and  double  spaces   ",
    "<think> reasoning </think> </s> <|im_end|> ｜ text that looks special",
    '{"path": "src/routers/ingress.rs", "line": 42, "ok": true}',
    "SELECT id, url FROM workers WHERE healthy = 1 ORDER BY load;",
    "https://example.com/v1/chat/completions?stream=false&n=1",
    "",
]


def read(n: int, path: str) -> dict:
    """The agent's `n`th call of `Read`, as a client sends it back."""
    return {
        "id": f"call_{n}",
        "type": "function",
        "function": {"name": "Read", "arguments": json.dumps({"path": path})},
    }


def called(
    *paths: str, reasoning: str | None = None, contents: list[str] | None = None
) -> list[dict]:
    """The agent's turn reading `paths`, and their contents."""
    turn = {
        "role": "assistant",
        "content": None,
        "tool_calls": [read(n, p) for n, p in enumerate(paths)],
    }
    if reasoning is not None:
        turn["reasoning_content"] = reasoning
    results = [
        {
            "role": "tool",
            "tool_call_id": f"call_{n}",
            "content": contents[n] if contents else f"# {p}",
        }
        for n, p in enumerate(paths)
    ]
    return [turn, *results]


def synthetic_text(synthetic: dict, start: int, count: int) -> str:
    """Lines `start` to `start + count` of the large prompts' text."""
    snippets = synthetic["snippets"]
    return "".join(
        f"{n} {snippets[n % len(snippets)]}\n" for n in range(start, start + count)
    )


def synthetic_body(synthetic: dict, kind: str) -> dict:
    """A large request: an agent's history of reading `turns` files of `lines`
    lines each, or a completion of all those lines."""
    turns, lines = synthetic["turns"], synthetic["lines"]
    if kind == "completion":
        return {"prompt": synthetic_text(synthetic, 0, turns * lines)}
    messages = list(synthetic["messages"])
    for turn in range(turns):
        path = f"src/module_{turn}.rs"
        reasoning = f"Step {turn}: read {path} next." if turn % 2 else None
        contents = [synthetic_text(synthetic, turn * lines, lines)]
        messages += called(path, reasoning=reasoning, contents=contents)
    messages.append(synthetic["question"])
    return {"tools": synthetic["tools"], "messages": messages}


def synthetic_lines(size: int, turns: int) -> dict:
    """The large prompts' lines: `turns` files of as few lines each as come to
    `size` bytes in all."""
    synthetic = {
        "snippets": SNIPPETS,
        "messages": MESSAGES,
        "question": NEXT_QUESTION,
        "tools": [TOOL],
        "turns": turns,
    }

    def short(lines: int) -> bool:
        return len(synthetic_text(synthetic, 0, turns * lines).encode()) < size

    low, high = 0, 1
    while short(high):
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        low, high = (middle, high) if short(middle) else (low, middle)
    synthetic["lines"] = high
    return synthetic


def default_requests() -> list[dict]:
    reasoned_tool_history = [
        *MESSAGES,
        *called("README.md", reasoning="The user wants README.md."),
    ]
    chats = {
        "plain": {},
        "effort_low": {"reasoning_effort": "low"},
        "effort_medium": {"reasoning_effort": "medium"},
        "effort_high": {"reasoning_effort": "high"},
        "effort_max": {"reasoning_effort": "max"},
        "effort_minimal": {"reasoning_effort": "minimal"},
        "effort_xhigh": {"reasoning_effort": "xhigh"},
        "effort_none": {"reasoning_effort": "none"},
        "effort_unknown": {"reasoning_effort": "ultra"},
        "effort_over_kwargs_effort": {
            "reasoning_effort": "max",
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
        "effort_over_kwargs_mode": {
            "reasoning_effort": "high",
            "chat_template_kwargs": {"thinking_mode": "chat"},
        },
        "kwargs_thinking_on": {"chat_template_kwargs": {"enable_thinking": True}},
        "kwargs_thinking_off": {"chat_template_kwargs": {"enable_thinking": False}},
        "kwargs_thinking_mode_chat": {
            "chat_template_kwargs": {"thinking_mode": "chat"}
        },
        "thinking_enabled": {"thinking": {"type": "enabled"}},
        "thinking_enabled_effort_none": {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "none",
        },
        "tools_effort_high": {"tools": [TOOL], "reasoning_effort": "high"},
        "tool_history": {
            "tools": [TOOL],
            "messages": [*MESSAGES, *called("README.md")],
        },
        "tool_history_reasoning": {"tools": [TOOL], "messages": reasoned_tool_history},
        "tool_history_effort_none": {
            "tools": [TOOL],
            "messages": reasoned_tool_history,
            "reasoning_effort": "none",
        },
        "parallel_tool_calls": {
            "tools": [TOOL],
            "messages": [*MESSAGES, *called("README.md", "LICENSE")],
        },
        "multi_turn": {"messages": [*MESSAGES, ANSWERED, NEXT_QUESTION]},
        "multi_turn_keep_thinking": {
            "messages": [*MESSAGES, ANSWERED, NEXT_QUESTION],
            "chat_template_kwargs": {"drop_thinking": False},
        },
        "multi_turn_tool_history": {
            "tools": [TOOL],
            "messages": [*reasoned_tool_history, ANSWERED, NEXT_QUESTION],
        },
        "scripts_and_special_text": {
            "messages": [
                {
                    "role": "user",
                    "content": '你好 <think> 1234567 ｜ "quote" \\n </s> <|im_end|> '
                    "👨‍👩‍👧 e\u0301 مرحبا\t\ttabs",
                }
            ]
        },
    }
    requests = [
        {"name": name, "route": CHAT, "body": {"messages": MESSAGES, **fields}}
        for name, fields in chats.items()
    ]
    requests.append(
        {
            "name": "completion",
            "route": COMPLETION,
            "body": {"prompt": "The capital of France is"},
        }
    )
    requests += [
        {"name": "agentic_history_2mib", "route": CHAT, "synthetic": "agentic_history"},
        {"name": "completion_2mib", "route": COMPLETION, "synthetic": "completion"},
    ]
    return requests


def checkpoint_dir(model: str) -> str:
    """`model`, or for a HuggingFace repo@revision, the directory its files of
    CHECKPOINT_FILES download to."""
    if Path(model).exists() or "@" not in model:
        return model
    repo, revision = model.split("@")
    return snapshot_download(repo, revision=revision, allow_patterns=CHECKPOINT_FILES)


def recorded_requests(path: str) -> list[dict]:
    requests = []
    with open(path, encoding="utf-8") as lines:
        for n, line in enumerate(lines):
            if not line.strip():
                continue
            entry = json.loads(line)
            body = entry.get("body", entry)
            route = entry.get("route") or (CHAT if "messages" in body else COMPLETION)
            if route not in (CHAT, COMPLETION):
                raise SystemExit(
                    f"{path}:{n + 1}: {route} is not tokenized for routing"
                )
            requests.append(
                {
                    "name": entry.get("name", f"line_{n + 1}"),
                    "route": route,
                    "body": body,
                }
            )
    return requests


class Captured(Exception):
    """The prompt reached the engine, which is as far as a request goes here."""


class LeaveOutCaptured(logging.Filter):
    """Leaves out the log of a request's handler failing with `Captured`."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.exc_info and isinstance(record.exc_info[1], Captured))


def in_process(args) -> tuple[httpx.AsyncClient, str, dict]:
    """ATOM's app, set up as a front-end worker sets it up (api_server's
    _setup_frontend, then _setup_reasoning_and_tools), the model it serves, and
    the prompt tokens of the last request its engine was given."""
    from atom.entrypoints.openai import api_server

    server_args = argparse.Namespace(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        chat_template=args.chat_template,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        request_log=None,
        served_model_name=None,
        tool_call_parser=args.tool_call_parser,
    )
    api_server._setup_frontend(server_args)
    api_server._setup_reasoning_and_tools(server_args)
    api_server.logger.addFilter(LeaveOutCaptured())
    tokenizer = api_server.tokenizer
    prompt = {}

    def preprocess(prompt_or_tokens, sampling_params, **_):
        # LLMEngine.preprocess tokenizes text with encode_prompt, which is
        # tokenizer.encode without a segment tokenizer.
        if isinstance(prompt_or_tokens, str):
            prompt_or_tokens = tokenizer.encode(prompt_or_tokens)
        prompt["token_ids"] = list(
            getattr(prompt_or_tokens, "token_ids", prompt_or_tokens)
        )
        raise Captured

    api_server.engine = SimpleNamespace(
        config=SimpleNamespace(
            generation_config=SimpleNamespace(), max_model_len=args.max_model_len
        ),
        io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
    )
    transport = httpx.ASGITransport(app=api_server.app)
    client = httpx.AsyncClient(
        transport=transport, base_url="http://atom", timeout=None
    )
    return client, api_server.model_name, prompt


async def tokenize(
    client: httpx.AsyncClient, prompt: dict, model: str, route: str, body: dict
) -> dict:
    prompt.clear()
    # The handler answers a request that reached the engine with a 500, or
    # raises `Captured` itself.
    try:
        response = await client.post(route, json={**body, "model": model})
    except Captured:
        response = None
    if "token_ids" in prompt:
        return {"token_ids": prompt["token_ids"]}
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return {"error": f"{response.status_code}: {detail}"}


def digest(token_ids: list[int]) -> str:
    """The BLAKE2b digest of `token_ids` as little-endian u32s."""
    data = struct.pack(f"<{len(token_ids)}I", *token_ids)
    return hashlib.blake2b(data, digest_size=32).hexdigest()


async def tokenize_on_atom(args, requests: list[dict]) -> dict:
    # The test reads the checkpoint from atom/mesh, so a path is stored absolute.
    checkpoint = Path(args.model)
    report = {"model": str(checkpoint.resolve()) if checkpoint.exists() else args.model}
    if any("synthetic" in request for request in requests):
        report["synthetic"] = synthetic_lines(2 << 20, turns=64)
    client, model, prompt = in_process(args)
    async with client:
        for request in requests:
            synthetic = request.get("synthetic")
            body = (
                request["body"]
                if synthetic is None
                else synthetic_body(report["synthetic"], synthetic)
            )
            tokens = await tokenize(client, prompt, model, request["route"], body)
            if synthetic is not None and "token_ids" in tokens:
                token_ids = tokens.pop("token_ids")
                tokens.update(tokens=len(token_ids), blake2b=digest(token_ids))
            request.update(tokens)
            if "error" in request:
                outcome = request["error"]
            else:
                outcome = f"{request.get('tokens') or len(request['token_ids'])} tokens"
            print(f"{request['name']}: {outcome}", file=sys.stderr)
    report["requests"] = requests
    return report


def write(report: dict, out) -> None:
    """`report` as JSON, a request to a line."""
    out.write("{\n")
    for key, value in report.items():
        if key != "requests":
            out.write(
                f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},\n"
            )
    lines = [json.dumps(request, ensure_ascii=False) for request in report["requests"]]
    out.write('  "requests": [\n    ' + ",\n    ".join(lines) + "\n  ]\n}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "model",
        help="the checkpoint directory, or a HuggingFace repo@revision to download its files from",
    )
    parser.add_argument("--requests", help="recorded requests, as JSON Lines")
    parser.add_argument("--chat-template", help="as the server takes it")
    parser.add_argument("--default-chat-template-kwargs", help="as the server takes it")
    parser.add_argument(
        "--tool-call-parser", default="auto", help="as the server takes it"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1 << 24,
        help="the context the in-process server checks prompts against",
    )
    parser.add_argument("--out", help="write the JSON here rather than to stdout")
    args = parser.parse_args()
    args.model = checkpoint_dir(args.model)

    requests = recorded_requests(args.requests) if args.requests else default_requests()
    report = asyncio.run(tokenize_on_atom(args, requests))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            write(report, out)
    else:
        write(report, sys.stdout)


if __name__ == "__main__":
    main()
