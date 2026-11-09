# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request admission, worker preparation and final response validation."""

import asyncio
import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
from threading import Lock

from atom import structured_output


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


def structured_tool_format(parser_cls, tools, tool_choice, strict_schemas=None):
    """Describe the native tool alternative without accepting client grammars.

    A function named in ``strict_schemas`` (see :func:`strict_tool_schemas`)
    also has its arguments constrained by that schema.
    """
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
            # What the parser reads as a call's start, marker or not: free
            # text before the calls may contain none of it.
            "markers": list(getattr(parser_cls, "START_MARKERS", ())),
            "tools": [
                {"name": name, "parameters": (strict_schemas or {}).get(name)}
                for name in names
            ],
            "required": required,
        }
    )


def declared_tool_names(tools):
    """The qualified names a call may carry: the request's own functions."""
    from atom.entrypoints.openai.tool_parser.tool_parser import qualified_tool_name

    return frozenset(qualified_tool_name(tool) for tool in tools or ())


def strict_tool_schemas(tools):
    """``{name: parameters}`` for each function declaring ``strict: true``.

    Keyed by the qualified name, the identity calls are parsed and validated by.
    """
    from atom.entrypoints.openai.tool_parser.tool_parser import qualified_tool_name

    schemas = {}
    for tool in tools or ():
        parameters = strict_parameters(tool["function"])
        if parameters is not None:
            schemas[qualified_tool_name(tool)] = parameters
    return schemas


# Keywords DeepSeek's strict mode rejects rather than enforces.
_STRICT_UNSUPPORTED = (
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    # Opens an object to undeclared keys.
    "patternProperties",
)
# The string formats DeepSeek documents for strict mode. The grammar
# enforces them, so arguments are not format-checked again afterwards.
_STRICT_FORMATS = frozenset({"email", "hostname", "ipv4", "ipv6", "uuid"})
# Where a schema holds other schemas.
_SCHEMA_MAPS = ("properties", "$defs", "definitions")
_SCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_SINGLES = (
    "items",
    "additionalProperties",
    "not",
    "if",
    "then",
    "else",
    "contains",
)
# What says which values a schema takes, short of a `type`.
_TYPED_BY = ("type", "$ref", "anyOf", "oneOf", "allOf", "enum", "const")
# What strict parameters may say at their top: their properties, and only
# annotations and definitions beside them.
_STRICT_ROOT_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "$defs",
        "definitions",
        "title",
        "description",
        "$comment",
        "$schema",
    }
)
# Keywords that would move what a reference resolves against.
_STRICT_NO_IDENTITY = ("$id", "$anchor", "$dynamicRef", "$dynamicAnchor")


def strict_parameters(function):
    """A strict function's ``parameters`` checked against DeepSeek's rules.

    ``None`` for a function that is not strict. Every object must list all of
    its properties in ``required`` and set ``additionalProperties: false``, so
    the arguments are exactly the declared ones, in declared order.
    """
    strict = function.get("strict")
    if strict is None:
        return None  # unset, however the client spells it
    if not isinstance(strict, bool):
        message = f"{function['name']}: strict must be a boolean"
        raise ValueError(message)  # noqa: TRY004
    if not strict:
        return None
    parameters = function.get("parameters") or {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError(f"{function['name']}: strict parameters must be an object")
    text = json.dumps(parameters)
    check = _checked_strict_parameters
    if len(text) > _CACHED_SCHEMA_CHARS:
        check = check.__wrapped__
    return json.loads(check(function["name"], text))


# An agent sends the same tools with every request, and the meta-schema check
# costs milliseconds a tool: each distinct tool is checked once.
_CACHED_SCHEMA_CHARS = 64 * 1024
# The dialects a client names that read strict parameters the same way; the
# check below is 2020-12's. OpenAI's zod helpers write draft-07.
_STRICT_DIALECTS = frozenset(
    {
        "http://json-schema.org/draft-07/schema",
        "https://json-schema.org/draft-07/schema",
        "https://json-schema.org/draft/2020-12/schema",
    }
)


@lru_cache(maxsize=256)
def _checked_strict_parameters(name, text):
    """``text``'s parameters, canonical and checked (``ValueError`` if not),
    as JSON text."""
    from .schema import canonical_defs

    parameters = json.loads(text)
    try:
        parameters = canonical_defs(parameters)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None
    dialect = parameters.get("$schema")
    if isinstance(dialect, str) and dialect.rstrip("#") in _STRICT_DIALECTS:
        parameters = {k: v for k, v in parameters.items() if k != "$schema"}
    # The grammar constrains each property; a keyword beside them would be
    # checked only after generation, too late for a 400.
    unconstrained = sorted(set(parameters) - _STRICT_ROOT_KEYS)
    if unconstrained:
        raise ValueError(
            f"{name}: strict parameters are constrained through "
            f"their properties only, not {unconstrained[0]}"
        )
    schema_from_format(
        {"type": "json_schema", "name": name, "schema": parameters},
        responses=True,
    )
    _check_strict(name, parameters, top=True)
    _check_strict_targets(name, parameters)
    # The DSML tags carry these names unescaped: the parser reads a name up
    # to its first quote, and every tag it reads -- a call's end, another
    # call -- opens with `<`.
    if any(c in n for n in (name, *parameters.get("properties", {})) for c in '"<'):
        raise ValueError(
            f"{name}: a strict tool's name and parameter names cannot contain "
            "'\"' or '<'"
        )
    return json.dumps(parameters)


def _check_strict_targets(name, parameters):
    """Hold what each ``$ref`` names to the strict rules too: it may sit
    where the walk over schema positions does not look, such as a
    definition's ``default``."""
    from referencing.exceptions import Unresolvable
    from referencing.jsonschema import DRAFT202012

    from .schema import OFFLINE_REGISTRY

    resolver = OFFLINE_REGISTRY.resolver_with_root(
        DRAFT202012.create_resource(parameters)
    )
    seen, pending = set(), list(_schema_refs(parameters))
    while pending:
        ref = pending.pop()
        if ref in seen:
            continue
        seen.add(ref)
        try:
            target = resolver.lookup(ref).contents
        except Unresolvable:
            raise ValueError(f"{name}: cannot resolve $ref {ref!r}") from None
        _check_strict(name, target)
        pending += _schema_refs(target)


def _schema_refs(schema):
    """The ``$ref``s at ``schema``'s schema positions, the ones
    ``check_schema_safety`` resolves: in `example` or an `x-*` annotation a
    `$ref` key is data."""
    from .schema import _ARRAYS, _MAPS, _SINGLES

    pending = [schema]
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            continue
        if isinstance(node.get("$ref"), str):
            yield node["$ref"]
        for key in _MAPS:
            if isinstance(node.get(key), dict):
                pending += node[key].values()
        pending += [node[key] for key in _SINGLES if key in node]
        for key in _ARRAYS:
            pending += node.get(key) or ()


def _check_strict(name, schema, *, top=False):
    if schema is False:
        return  # takes nothing, adds nothing
    if not isinstance(schema, dict):
        message = f"{name}: every strict schema must be an object"
        raise ValueError(message)  # noqa: TRY004
    if not top and any(k in schema for k in ("$def", "$defs", "definitions")):
        # References point into the top's definitions only.
        raise ValueError(
            f"{name}: definitions are supported only at the top of parameters"
        )
    for keyword in _STRICT_NO_IDENTITY:
        if keyword in schema:
            raise ValueError(f"{name}: {keyword} is not supported by strict tools")
    ref = schema.get("$ref")
    if ref is not None and not (
        isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/"))
    ):
        # Each parameter is its own grammar, so `#` would name the
        # parameter, not the whole arguments object.
        raise ValueError(
            f"{name}: a strict tool's $ref must point into its $def, not {ref!r}"
        )
    for keyword in _STRICT_UNSUPPORTED:
        if keyword in schema:
            raise ValueError(f"{name}: {keyword} is not supported by strict tools")
    fmt = schema.get("format")
    if fmt is not None and fmt not in _STRICT_FORMATS:
        raise ValueError(f"{name}: format {fmt!r} is not supported by strict tools")
    if not any(k in schema for k in _TYPED_BY):
        raise ValueError(f"{name}: every strict schema must give its type")
    types = schema.get("type")
    types = types if isinstance(types, list) else [types]
    if "array" in types and "items" not in schema:
        # `prefixItems` alone lets anything follow the prefix.
        raise ValueError(f"{name}: every strict array must give its items")
    if "object" in types or "properties" in schema:
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            raise ValueError(
                f"{name}: every strict object must set additionalProperties: false"
            )
        if set(schema.get("required", ())) != set(properties):
            raise ValueError(
                f"{name}: every property of a strict object must be required"
            )
    for key in _SCHEMA_MAPS:
        for subschema in (schema.get(key) or {}).values():
            _check_strict(name, subschema)
    for key in _SCHEMA_LISTS:
        for subschema in schema.get(key) or ():
            _check_strict(name, subschema)
    for key in _SCHEMA_SINGLES:
        if key in schema:
            _check_strict(name, schema[key])


def structured_reasoning_marker(channel, tokenizer):
    """Only suspend the JSON grammar when generation starts inside reasoning."""
    from atom.entrypoints.openai.reasoning_dialects import FALLBACK_DIALECT

    dialect = channel.dialect or FALLBACK_DIALECT
    if dialect.content_framing or len(dialect.end_markers) != 1:
        raise ValueError(
            "Structured output does not yet support framed reasoning channels"
        )
    if not channel.starts_open:
        return ""
    marker = dialect.think_end_marker
    if not marker:
        raise ValueError(
            "Structured thinking requires an explicit reasoning end marker"
        )
    return marker


def structured_response_channel(channel, schema, strict_tools=False):
    """The answer under a grammar is read literally: a reasoning marker the
    grammar let through (in a strict argument, say) is the model's words."""
    if schema is None and not strict_tools:
        return channel
    return replace(channel, literal_answer=True)


def raise_for_structured_error(finish_reason):
    if finish_reason == "structured_output_error":
        raise RuntimeError("Structured decoding failed for this request")


_TRUNCATED = {"length", "max_tokens", "max_new_tokens"}


def validate_output(schema, text, finish_reason, *, tool_calls=False):
    raise_for_structured_error(finish_reason)
    if schema is None or finish_reason in _TRUNCATED:
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


def validate_tool_calls(strict_schemas, tool_calls, finish_reason, declared=()):
    """Each call to a strict function must carry arguments its schema accepts,
    and with strict functions declared, every call must name one of
    ``declared``: what the parser read as another call is not one."""
    if not strict_schemas or finish_reason in _TRUNCATED:
        return
    from .schema import schema_validator

    for call in tool_calls or ():
        name = call["function"]["name"]
        if declared and name not in declared:
            raise RuntimeError(f"Tool call {name} is not one of the request's tools")
        if name not in strict_schemas:
            continue
        try:
            # The parser reads an overflowing number as inf and writes it back
            # as `Infinity`, which a client's JSON parser refuses.
            arguments = json.loads(
                call["function"]["arguments"], parse_constant=_not_json
            )
            schema_validator(strict_schemas[name]).validate(arguments)
        except Exception as exc:
            raise RuntimeError(
                f"Tool call {name} did not satisfy its strict parameters schema"
            ) from exc


def _not_json(constant):
    raise ValueError(f"{constant} is not JSON")


async def validate_chat_stream(source, schema, strict_schemas=None, declared=()):
    """Pass a Chat stream on, checking each choice as it finishes.

    A strict function's arguments are checked once its call is complete, so
    with strict tools the frames from the first tool call on wait for the
    end of the stream: a call that fails its schema never reaches the
    client, which gets the error a non-streamed reply gets instead. Nor does
    a call whose choice never finished, so was never checked: the stream
    ended in an error first, and that error is what goes out.
    """
    from atom.entrypoints.openai.sse import data_frame, iter_sse_data

    texts, calls = {}, {}
    held, finished = None, set()
    try:
        async for frame in source:
            carries = set()  # the choices whose calls this frame carries
            for data in iter_sse_data(frame):
                if data == "[DONE]":
                    continue
                payload = json.loads(data)
                for choice in payload.get("choices", []):
                    index = choice.get("index", 0)
                    delta = choice.get("delta", {})
                    texts.setdefault(index, []).append(delta.get("content") or "")
                    for call in delta.get("tool_calls") or ():
                        if strict_schemas and held is None:
                            held = []
                        carries.add(index)
                        fn = call.get("function") or {}
                        acc = calls.setdefault(index, {}).setdefault(
                            call.get("index", 0),
                            {"function": {"name": "", "arguments": ""}},
                        )["function"]
                        acc["name"] += fn.get("name") or ""
                        acc["arguments"] += fn.get("arguments") or ""
                    if choice.get("finish_reason"):
                        validate_output(
                            schema,
                            "".join(texts[index]),
                            choice["finish_reason"],
                            tool_calls=choice["finish_reason"] == "tool_calls",
                        )
                        validate_tool_calls(
                            strict_schemas,
                            calls.get(index, {}).values(),
                            choice["finish_reason"],
                            declared,
                        )
                        finished.add(index)
            if held is None:
                yield frame
            else:
                held.append((frame, carries))
        for frame, carries in held or ():
            if carries <= finished:
                yield frame
    except RuntimeError as exc:
        yield data_frame(
            {"error": {"type": "structured_output_error", "message": str(exc)}}
        )
        yield "data: [DONE]\n\n"
    finally:
        await source.aclose()


def validate_execution_options(
    config,
    params,
    *,
    kv_transfer=None,
    multimodal=False,
    relaxed_mtp=False,
    synthetic=False,
):
    if getattr(config, "pipeline_parallel_size", 1) > 1:
        raise ValueError("Structured output does not support pipeline parallelism")
    if relaxed_mtp:
        raise ValueError("Structured output cannot use relaxed MTP verification")
    if (
        kv_transfer
        or multimodal
        or getattr(config, "enable_rapidserve", False)
        or getattr(config, "kv_transfer_config", None)
    ):
        raise ValueError("Structured output currently requires text without P/D")
    spec = getattr(config, "speculative_config", None)
    if synthetic or getattr(spec, "synthetic_acceptance_rates", None):
        raise ValueError("Structured output cannot use synthetic forced acceptance")
    if params.ignore_eos or params.stop_strings:
        raise ValueError(
            "Structured output does not support ignore_eos or custom stop strings"
        )
    if params.logprobs and getattr(spec, "num_speculative_tokens", 0):
        raise ValueError(
            "Combining JSON Schema, speculative decoding and logprobs is not supported; disable speculative decoding or logprobs"
        )


class WorkerGrammarPreflight:
    def __init__(self, engine, tokenizer, config):
        self.engine = engine
        self.tokenizer = tokenizer
        # Constructed once with the engine; ordinary requests must not load
        # grammar dependencies or serialize/hash the tokenizer.
        self.config = config
        self.vocab_size = None
        self.fingerprint = None
        self.ready = OrderedDict()
        self.pending = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="grammar-admission"
        )

    def submit(self, schema, tool_format, reasoning_end=""):
        key = (schema, tool_format, reasoning_end)
        with self.lock:
            if self.fingerprint is None:
                hf = self.config.hf_config
                vocab_size = (getattr(hf, "text_config", None) or hf).vocab_size
                fingerprint = structured_output.tokenizer_fingerprint(
                    self.tokenizer, self.config.end_token_ids
                )
                self.vocab_size = vocab_size
                self.fingerprint = fingerprint
            if key in self.ready:
                self.ready.move_to_end(key)
                return self.ready[key]
            if key not in self.pending:
                self.pending[key] = self.executor.submit(self._prepare, key)
            future = self.pending[key]
        return future

    async def prepare(self, schema, tool_format, reasoning_end=""):
        future = self.submit(schema, tool_format, reasoning_end)
        return await asyncio.shield(asyncio.wrap_future(future))

    def prepare_requests(self, key, request_ids):
        self._prepare(key, tuple(request_ids))

    def _prepare(self, key, request_ids=()):
        # Neither the RPC nor compilation holds the ready-cache lock. A cached
        # request must not queue behind another client's slow schema.
        try:
            manager = getattr(self.engine, "core_mgr", None)
            rpc = getattr(manager, "collective_rpc", None)
            if rpc is None:
                raise ValueError(
                    "Structured output requires worker preflight RPC support"
                )
            deadline = time.monotonic() + 60
            while True:
                args = (*key, self.fingerprint, self.vocab_size)
                if request_ids:
                    args += (request_ids,)
                replies = rpc(
                    "prepare_structured_output",
                    args=args,
                    timeout=max(0.1, deadline - time.monotonic()),
                )
                if not replies or any(not reply.ok for reply in replies):
                    raise ValueError(
                        "Worker JSON Schema preparation failed: "
                        + str([reply.error for reply in replies if not reply.ok])
                    )
                if all(reply.value.get("ready", True) for reply in replies):
                    expected = {
                        "fingerprint": self.fingerprint,
                        "vocab_size": self.vocab_size,
                    }
                    if any(
                        any(reply.value.get(k) != v for k, v in expected.items())
                        for reply in replies
                    ):
                        raise ValueError(
                            "Structured output configuration differs across workers"
                        )
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Worker JSON Schema preparation timed out")
                time.sleep(0.01)
        except Exception:
            if request_ids:
                # Another rank may have prepared successfully. Release those
                # states before propagating an admission failure to the caller.
                rpc("release_structured_requests", args=(request_ids,), timeout=60)
                raise
            with self.lock:
                self.pending.pop(key, None)
            raise
        if request_ids:
            return
        with self.lock:
            self.ready[key] = self.pending.pop(key)
            if len(self.ready) > 128:
                self.ready.popitem(last=False)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


async def configure_request(
    engine,
    tokenizer,
    config,
    sampling_params,
    schema,
    channel,
    *,
    parser_cls=None,
    tools=None,
    tool_choice=None,
    kv_transfer=None,
    multimodal=False,
    relaxed_mtp=False,
    synthetic=False,
    strict_schemas=None,
):
    if schema is None and not strict_schemas:
        return
    validate_execution_options(
        config,
        sampling_params,
        kv_transfer=kv_transfer,
        multimodal=multimodal,
        relaxed_mtp=relaxed_mtp,
        synthetic=synthetic,
    )
    tool_format = structured_tool_format(parser_cls, tools, tool_choice, strict_schemas)
    reasoning_end = structured_reasoning_marker(channel, tokenizer)
    sampling_params.json_schema = schema
    sampling_params.structured_tool_format = tool_format
    sampling_params.structured_reasoning_end = reasoning_end
    # Schema's DSML grammar already enforces the tool alternative. Never apply
    # two independently stateful masks to the same request.
    sampling_params.required_channel = None
    await asyncio.to_thread(
        engine.prepare_structured_request, sampling_params, kv_transfer, multimodal
    )
