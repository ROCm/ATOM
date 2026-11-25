#!/usr/bin/env python3
"""gate.py <base url> <served model>: is the server's output sane before a run spends an hour on it?

The MiniMax-M3 counterpart of the Kimi-K3 image's /opt/k3/gate.py: an answer that is empty, all '!' or one token
repeated fails, 17 * 23 must come back as 391, and -- the agentic dataset is tool calls -- a request offering a tool
must come back as a parsed call to it. The last line printed is GATE PASS or GATE FAIL <why>.
"""

import json
import sys
import urllib.request

URL = sys.argv[1].rstrip("/") + "/v1/chat/completions"
MODEL = sys.argv[2]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def chat(content, **kwargs):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 4096,
        **kwargs,
    }
    request = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["choices"][0]["message"]


def degenerate(text):
    words = text.split()
    return (
        not text.strip()
        or set(text.strip()) <= {"!"}
        or (len(words) >= 20 and len(set(words)) <= 2)
    )


problems = []
answer = chat("What is 17 * 23? Reply with the number only.").get("content") or ""
if degenerate(answer) or "391" not in answer:
    problems.append(f"17 * 23 -> {answer[:200]!r}")

message = chat("What is the weather in Paris right now? Use the tool.", tools=TOOLS)
calls = message.get("tool_calls") or []
try:
    call = calls[0]["function"]
    ok = call["name"] == "get_weather" and "paris" in json.loads(call["arguments"])["city"].lower()
except (IndexError, KeyError, TypeError, ValueError):
    ok = False
if not ok:
    problems.append(f"tool call -> {json.dumps(message)[:300]}")

print("GATE PASS" if not problems else "GATE FAIL " + "; ".join(problems))
sys.exit(1 if problems else 0)
