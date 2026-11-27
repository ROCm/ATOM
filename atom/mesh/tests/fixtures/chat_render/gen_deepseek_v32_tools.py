# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Writes deepseek_v32_tools.json: a chat with tools and a history tool call,
rendered by DeepSeek-V3.2's own encoder (the checkpoint's
encoding/encoding_dsv32.py, which ATOM's server loads) in each thinking mode.

    python gen_deepseek_v32_tools.py <checkpoint>/encoding/encoding_dsv32.py deepseek_v32_tools.json
"""

import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("enc", sys.argv[1])
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
    }
]
msgs = [
    {"role": "system", "content": "You are helpful.", "tools": tools},
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris", "days": 2}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
    {"role": "user", "content": "And Rome?"},
]
out = {}
for mode in ("thinking", "chat"):
    out[mode] = enc.encode_messages(msgs, thinking_mode=mode)
with open(sys.argv[2], "w") as f:
    json.dump({"messages": msgs, "prompts": out}, f, ensure_ascii=False, indent=1)
