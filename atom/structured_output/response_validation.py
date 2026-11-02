# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate final JSON answers and completed Chat streams."""

import json


def validate_output(schema, text, finish_reason, *, tool_calls=False):
    if schema is None or finish_reason in {"length", "max_tokens", "max_new_tokens"}:
        return
    if tool_calls and not text.strip():
        return  # A function-only turn has no response text to constrain.
    from .schema_validation import schema_validator

    try:
        value = json.loads(
            text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
        )
        schema_validator(json.loads(schema)).validate(value)
    except Exception as exc:
        raise RuntimeError(
            "Generated output did not satisfy the requested JSON Schema"
        ) from exc


def validate_responses_output(schema, response, finish_reason):
    visible = "".join(
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    validate_output(
        schema,
        visible,
        finish_reason,
        tool_calls=any(
            item.get("type")
            in {"function_call", "custom_tool_call", "tool_search_call"}
            for item in response.get("output", [])
        ),
    )


async def validate_chat_stream(source, schema):
    from atom.entrypoints.openai.sse import data_frame, iter_sse_data

    texts = {}
    try:
        async for frame in source:
            for data in iter_sse_data(frame):
                if data == "[DONE]":
                    continue
                payload = json.loads(data)
                for choice in payload.get("choices", []):
                    index = choice.get("index", 0)
                    texts.setdefault(index, []).append(
                        choice.get("delta", {}).get("content") or ""
                    )
                    if choice.get("finish_reason"):
                        validate_output(
                            schema,
                            "".join(texts[index]),
                            choice["finish_reason"],
                            tool_calls=choice["finish_reason"] == "tool_calls",
                        )
            yield frame
    except RuntimeError as exc:
        yield data_frame(
            {"error": {"type": "structured_output_error", "message": str(exc)}}
        )
        yield "data: [DONE]\n\n"
    finally:
        await source.aclose()
