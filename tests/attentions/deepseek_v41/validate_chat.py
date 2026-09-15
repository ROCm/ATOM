# SPDX-License-Identifier: MIT
"""Real-checkpoint chat/tool smoke through P09 ModelRunner and Scheduler.

Run with torchrun --nproc_per_node=4. This validates generated protocol output;
the HTTP transport is covered separately by the entrypoint tests.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from atom.config import CompilationConfig, Config, CUDAGraphMode
from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    load_custom_message_encoder,
)
from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.entrypoints.openai.tool_parser import flatten_tool_events, parse_tool_calls
from atom.entrypoints.openai.tool_parser.registry import resolve_tool_call_parser
from atom.entrypoints.openai.tool_parser.stream import ToolCallStreamParser
from atom.model_engine.model_runner import ModelRunner

from .benchmark_runtime import run_case

TOOLS = [
    {
        "type": "function",
        "namespace": "math",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def read_reply(text, thinking, parser, tools):
    channel = ReasoningChannel(starts_open=thinking)
    reasoning, content = channel.split(text)
    content, calls = parse_tool_calls(content, tools, parser)
    # Match the serving pipeline while forcing every marker across chunks.
    reason_filter = channel.stream()
    stream = ToolCallStreamParser(tools=tools, parser_cls=parser)
    reason_parts, events = [], []

    def feed(parts):
        for field, part in parts:
            if field == "reasoning_content":
                reason_parts.append(part)
            else:
                events.extend(stream.process(part))

    for char in text:
        feed(reason_filter.process(char))
    feed(reason_filter.flush())
    events.extend(stream.flush())
    stream_content, stream_calls = flatten_tool_events(events)
    assert "".join(reason_parts) == (reasoning or "")
    assert stream_content == content
    assert [c.function for c in stream_calls] == [c.function for c in calls]
    return {
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": [
            {"id": c.id, "type": c.type, "function": c.function} for c in calls
        ],
    }


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    args.add_argument("--output", required=True)
    options = args.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    encoder = load_custom_message_encoder(options.model)
    parser = resolve_tool_call_parser(None, tokenizer, encoder)
    assert encoder.name == "encoding_dsv41" and parser.NAME == "dsml_v41"
    config = Config(
        model=options.model,
        tensor_parallel_size=size,
        enable_expert_parallel=True,
        enforce_eager=False,
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.PIECEWISE, cudagraph_capture_sizes=[1, 2, 4]
        ),
        kv_cache_dtype="fp4",
        index_cache_dtype="fp4",
        max_num_batched_tokens=1024,
        max_model_len=2048,
        max_num_seqs=4,
        long_prefill_token_threshold=128,
        state_checkpoint_interval_tokens=128,
        enable_log_stats=False,
        port=port,
    )
    config.parallel_config.data_parallel_base_port = port
    runner = ModelRunner(rank, config)
    report = {
        "tp": size,
        "cache_dtype": "fp4",
        "graph": "PIECEWISE",
        "transport": "ModelRunner/Scheduler",
        "cases": [],
    }
    try:
        runner.get_num_blocks()
        pages = 512
        runner.pool_plan = runner.pool_plan.with_paged_entries(pages)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(pages)
        runner.capture_cudagraph()
        cases = [
            {
                "name": "chinese_chat",
                "thinking": False,
                "tools": None,
                "messages": [
                    {"role": "user", "content": "请用中文简短回答：中国的首都是哪里？"},
                ],
            },
            {
                "name": "tool_call",
                "thinking": False,
                "tools": TOOLS,
                "messages": [
                    {
                        "role": "user",
                        "content": "请调用 math::add 工具计算 17+25，收到工具结果后再用中文回答。",
                    },
                ],
            },
            {
                "name": "reasoning",
                "thinking": True,
                "tools": None,
                "messages": [
                    {"role": "user", "content": "What is 2 + 2? Reply briefly."},
                ],
            },
        ]

        def generate(batch, count):
            prompts = [
                apply_chat_template(
                    tokenizer,
                    encoder,
                    c["messages"],
                    tools=c["tools"],
                    thinking_mode="thinking" if c["thinking"] else "chat",
                    thinking_effort=1,
                )
                for c in batch
            ]
            result = run_case(
                runner,
                [
                    tokenizer.encode(prompt, add_special_tokens=False)
                    for prompt in prompts
                ],
                count,
            )
            hashes = [None] * size
            torch.distributed.all_gather_object(hashes, result["output_sha256"])
            assert len(set(hashes)) == 1, "TP ranks generated different tokens"
            for case, prompt, tokens in zip(batch, prompts, result["outputs"]):
                assert (
                    tokenizer.eos_token_id in tokens
                ), f"{case['name']}: no EOS within budget"
                tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(tokens)
                reply = read_reply(text, case["thinking"], parser, case["tools"])
                record = {
                    **case,
                    "prompt": prompt,
                    "output": text,
                    "reply": reply,
                    "completion_token_ids": tokens,
                }
                report["cases"].append(record)
                if rank == 0:
                    print("CHAT", json.dumps(record, ensure_ascii=False), flush=True)
            if rank == 0:
                Path(options.output).write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                )

        generate(cases, 256)
        plain, tool, reasoning = [c["reply"] for c in report["cases"]]
        assert "北京" in plain["content"] and not plain["reasoning_content"]
        assert reasoning["reasoning_content"] and "4" in reasoning["content"]
        assert len(tool["tool_calls"]) == 1
        call = tool["tool_calls"][0]
        assert call["function"]["name"] == "math::add"
        assert json.loads(call["function"]["arguments"]) == {"a": 17, "b": 25}
        followup = {
            **cases[1],
            "name": "tool_result",
            "messages": [
                *cases[1]["messages"],
                {"role": "assistant", **tool},
                {"role": "tool", "tool_call_id": call["id"], "content": "42"},
            ],
        }
        generate([followup], 64)
        assert "42" in report["cases"][-1]["reply"]["content"]
        assert not report["cases"][-1]["reply"]["tool_calls"]
        report["passed"] = True
        if rank == 0:
            Path(options.output).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            print("P06_CHAT_SMOKE_PASS", flush=True)
    finally:
        runner.exit()


if __name__ == "__main__":
    main()
