# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Structured-output request validation and final semantic validation."""

import json


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
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc

    # Resolve only document-local references. Never fetch schemas from URLs.
    def check(value):
        if not isinstance(value, dict):
            return
        for key in ("$ref", "$dynamicRef"):
            if key in value and not value[key].startswith("#"):
                raise ValueError("Only local JSON Schema references are supported")
        # Names in these maps are user data, not schema keywords.
        for key in (
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
        ):
            for child in value.get(key, {}).values():
                check(child)
        for key in (
            "items",
            "additionalProperties",
            "contains",
            "propertyNames",
            "unevaluatedProperties",
            "unevaluatedItems",
            "not",
            "if",
            "then",
            "else",
        ):
            check(value.get(key))
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            for child in value.get(key, []):
                check(child)

    check(schema)
    return json.dumps(schema, ensure_ascii=False)


def structured_tool_format(parser_cls, tools, tool_choice):
    """Describe the native tool alternative without accepting client grammars."""
    if not tools or tool_choice == "none":
        return None
    from .tool_parser.deepseekv4_tool_parser import DsmlParser
    from .tool_parser.tool_parser import qualified_tool_name

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
    from .reasoning_dialects import FALLBACK_DIALECT

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


def validate_output(schema, text, finish_reason, *, tool_calls=False):
    if schema is None or finish_reason in {"length", "max_tokens", "max_new_tokens"}:
        return
    if tool_calls and not text.strip():
        return  # A function-only turn has no response text to constrain.
    from jsonschema import Draft202012Validator

    try:
        value = json.loads(
            text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
        )
        Draft202012Validator(json.loads(schema)).validate(value)
    except Exception as exc:
        raise RuntimeError(
            "Generated output did not satisfy the requested JSON Schema"
        ) from exc


async def validate_chat_stream(source, schema):
    from .sse import data_frame, iter_sse_data

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
