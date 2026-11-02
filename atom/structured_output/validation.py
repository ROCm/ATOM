# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request admission, worker preparation and final response validation."""

import asyncio
import json
from collections import OrderedDict


def schema_from_format(fmt, *, responses=False):
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") not in {
        "text",
        "json_object",
        "json_schema",
    }:
        raise ValueError("format.type must be text, json_object, or json_schema")
    if fmt["type"] == "text":
        return None
    if fmt["type"] == "json_object":
        return json.dumps({"type": "object"})
    spec = fmt if responses else fmt.get("json_schema")
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
        raise ValueError("json_schema requires a schema object")  # noqa: TRY004
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("json_schema requires a nonempty name")
    schema = spec["schema"]
    from jsonschema import Draft202012Validator, SchemaError

    try:
        # Regex syntax is checked by the bounded RE2 compiler below, not re.compile.
        Draft202012Validator.check_schema(schema, format_checker=None)
    except SchemaError as exc:
        raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc

    from .schema import check_schema_safety

    check_schema_safety(schema)
    return json.dumps(schema, ensure_ascii=False)


def structured_tool_format(parser_cls, tools, tool_choice):
    """Describe the native tool alternative without accepting client grammars."""
    if not tools or tool_choice == "none":
        return None
    from atom.entrypoints.openai.tool_parser.deepseekv4_tool_parser import DsmlParser
    from atom.entrypoints.openai.tool_parser.tool_parser import qualified_tool_name

    if parser_cls is None or not issubclass(parser_cls, DsmlParser):
        raise ValueError("Structured output with tools requires a DSML tool parser")
    names = [qualified_tool_name(tool) for tool in tools]
    required = tool_choice == "required" or isinstance(tool_choice, dict)
    if isinstance(tool_choice, dict):
        name = qualified_tool_name(tool_choice)
        if name not in names:
            raise ValueError("The selected tool must be declared in tools")
        names = [name]
    return json.dumps(
        {
            "prefix": parser_cls.DSML_PREFIX,
            "section": parser_cls.SECTION,
            "names": names,
            "required": required,
        }
    )


def structured_reasoning_marker(channel, tokenizer):
    """Only suspend the JSON grammar when generation starts inside reasoning."""
    from atom.entrypoints.openai.reasoning_dialects import FALLBACK_DIALECT

    dialect = channel.dialect or FALLBACK_DIALECT
    if dialect.content_framing or len(dialect.end_markers) != 1:
        raise ValueError(
            "Structured output does not yet support framed reasoning channels"
        )
    if not channel.starts_open:
        return ()
    marker = dialect.think_end_marker
    tokens = tuple(tokenizer.encode(marker, add_special_tokens=False)) if marker else ()
    if not tokens:
        raise ValueError(
            "Structured thinking requires an explicit reasoning end marker"
        )
    return tokens


def structured_response_channel(channel, schema):
    from dataclasses import replace

    return replace(channel, literal_answer=True) if schema is not None else channel


def validate_output(schema, text, finish_reason, *, tool_calls=False):
    if schema is None or finish_reason in {"length", "max_tokens", "max_new_tokens"}:
        return
    if tool_calls and not text.strip():
        return  # A function-only turn has no response text to constrain.
    from .schema import schema_validator

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


class WorkerGrammarPreflight:
    def __init__(self, engine, tokenizer, config):
        from atom.structured_output import (
            structured_stop_token_ids,
            tokenizer_fingerprint,
        )

        self.engine = engine
        self.tokenizer = tokenizer
        hf = config.hf_config
        self.vocab_size = (getattr(hf, "text_config", None) or hf).vocab_size
        self.fingerprint = tokenizer_fingerprint(
            tokenizer, structured_stop_token_ids(config, tokenizer)
        )
        self.ready = OrderedDict()
        self.lock = asyncio.Lock()

    async def prepare(self, schema, tool_format):
        key = (schema, tool_format)
        async with self.lock:
            if key in self.ready:
                self.ready.move_to_end(key)
                return
            manager = getattr(self.engine, "core_mgr", None)
            rpc = getattr(manager, "collective_rpc", None)
            if rpc is None:
                raise ValueError(
                    "Structured output requires worker preflight RPC support"
                )
            replies = await asyncio.to_thread(
                rpc,
                "prepare_structured_output",
                args=(schema, tool_format, self.fingerprint, self.vocab_size),
                timeout=60.0,
            )
            if not replies or any(not reply.ok for reply in replies):
                errors = [reply.error for reply in replies if not reply.ok]
                raise ValueError(
                    "Worker JSON Schema preparation failed: " + str(errors)
                )
            expected = {"fingerprint": self.fingerprint, "vocab_size": self.vocab_size}
            if any(reply.value != expected for reply in replies):
                raise ValueError(
                    "Structured output configuration differs across workers"
                )
            self.ready[key] = None
            if len(self.ready) > 128:
                self.ready.popitem(last=False)
