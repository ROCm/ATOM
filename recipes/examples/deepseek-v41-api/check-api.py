# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run the recipe examples against a live server and retain responses."""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def endpoint(name):
    if name == "fim":
        return "/beta/completions"
    if name in ("prefix", "strict-tools"):
        return "/beta/chat/completions"
    if name.startswith("responses") or name == "flash-search-declaration":
        return "/v1/responses"
    if name.startswith("messages"):
        return "/v1/messages"
    return "/v1/chat/completions"


def visible(response, route):
    if route == "/v1/responses":
        return "".join(
            part.get("text", "")
            for item in response.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
    if route == "/v1/messages":
        return "".join(
            part.get("text", "")
            for part in response.get("content", [])
            if part.get("type") == "text"
        )
    choice = response["choices"][0]
    return choice.get("text") or choice.get("message", {}).get("content") or ""


def check_response(name, body, raw, route):
    if body.get("stream"):
        events = [
            json.loads(line[6:])
            for line in raw.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert events, "No SSE events"
        assert not any(e.get("error") for e in events), "SSE error"
        if route == "/v1/responses":
            assert events[-1].get("type") == "response.completed", events[-1]
        elif route == "/v1/messages":
            assert events[-1].get("type") == "message_stop", events[-1]
            assert not any(
                e.get("delta", {}).get("stop_reason") == "max_tokens" for e in events
            ), "Truncated Messages stream"
        else:
            assert "data: [DONE]" in raw, "Missing Chat terminator"
            assert any(
                c.get("finish_reason") == "stop"
                for e in events
                for c in e.get("choices", [])
            ), "Chat stream did not finish normally"
        return
    response = json.loads(raw)
    assert not response.get("error"), response
    if route == "/v1/responses":
        assert response.get("status") == "completed", response.get("status")
    elif route == "/v1/messages":
        assert response.get("stop_reason") != "max_tokens", "Truncated Messages"
    else:
        assert (
            response["choices"][0].get("finish_reason") != "length"
        ), "Truncated output"
    if name in ("chat-tools", "strict-tools"):
        calls = response["choices"][0]["message"].get("tool_calls")
        assert calls, "No tool call returned"
        for call in calls:
            assert call["function"]["name"] == "get_weather", call
            arguments = json.loads(call["function"]["arguments"])
            assert isinstance(arguments.get("city"), str), arguments
            if name == "strict-tools":
                assert set(arguments) == {"city"}, arguments
        return
    text = visible(response, route)
    assert text.strip(), "No visible answer"
    if name.endswith("schema") or name == "chat-json":
        value = json.loads(text)
        assert isinstance(value, dict), value
        if name.endswith("schema"):
            assert set(value) == {"city", "temperature_c"}, value
            assert isinstance(value["city"], str), value
            assert type(value["temperature_c"]) in (int, float), value
    if name == "chat-logprobs":
        entries = response["choices"][0]["logprobs"]["content"]
        assert entries, "No token probabilities"
        assert all(math.isfinite(e["logprob"]) for e in entries), entries
        assert all(len(e["top_logprobs"]) <= 5 for e in entries), entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--images", action="store_true", help="Include data-URL images")
    parser.add_argument("--urls", action="store_true", help="Include public URL images")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path("api-check-results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(Path(__file__).parent.glob("*.json")):
        name = path.stem
        if "-image" in name and not args.images:
            continue
        if name.endswith("-url") and not args.urls:
            continue
        body = json.loads(path.read_text(encoding="utf8"))
        route = endpoint(name)
        request = urllib.request.Request(
            args.base_url.rstrip("/") + route,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )
        start = time.monotonic()
        result = {"name": name, "route": route}
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result["status"] = response.status
                raw = response.read().decode("utf8")
            (args.output_dir / f"{name}.response.txt").write_text(raw, encoding="utf8")
            check_response(name, body, raw, route)
            result["passed"] = True
        except urllib.error.HTTPError as error:
            result.update(status=error.code, passed=False, error=error.read().decode())
        except (
            AssertionError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            OSError,
        ) as error:
            result.update(passed=False, error=str(error))
        result["seconds"] = time.monotonic() - start
        results.append(result)
        print(json.dumps(result), flush=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(results, indent=2), encoding="utf8"
        )
    if not results or any(not r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
