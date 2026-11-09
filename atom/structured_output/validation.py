# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request admission, worker preparation and final response validation."""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Lock

from atom import structured_output

logger = logging.getLogger("atom")


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


def selected_tools(tools, tool_choice):
    """The tools a request lets the model call, and whether it must call one."""
    from atom.entrypoints.openai.tool_parser.tool_parser import qualified_tool_name

    if not isinstance(tool_choice, dict):
        return list(tools), tool_choice == "required"
    name = qualified_tool_name(tool_choice)
    chosen = [tool for tool in tools if qualified_tool_name(tool) == name]
    if not chosen:
        raise ValueError("The selected tool must be declared in tools")
    return chosen, True


def structured_tool_format(parser_cls, tools, tool_choice):
    """Describe the native tool alternative without accepting client grammars."""
    if not tools or tool_choice == "none":
        return None
    from atom.entrypoints.openai.tool_parser.deepseekv4_tool_parser import DsmlParser
    from atom.entrypoints.openai.tool_parser.tool_parser import qualified_tool_name

    if parser_cls is None or not issubclass(parser_cls, DsmlParser):
        raise ValueError("Structured output with tools requires a DSML tool parser")
    chosen, required = selected_tools(tools, tool_choice)
    return json.dumps(
        {
            "prefix": parser_cls.DSML_PREFIX,
            "section": parser_cls.SECTION,
            "names": [qualified_tool_name(tool) for tool in chosen],
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
        return ""
    marker = dialect.think_end_marker
    if not marker:
        raise ValueError(
            "Structured thinking requires an explicit reasoning end marker"
        )
    return marker


def structured_response_channel(channel, schema):
    return replace(channel, literal_answer=True) if schema is not None else channel


def raise_for_structured_error(finish_reason):
    if finish_reason == "structured_output_error":
        raise RuntimeError("Structured decoding failed for this request")


def validate_output(schema, text, finish_reason, *, tool_calls=False):
    raise_for_structured_error(finish_reason)
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


def validate_execution_options(
    config,
    params,
    *,
    kv_transfer=None,
    multimodal=False,
    relaxed_mtp=False,
    synthetic=False,
):
    from atom.kv_transfer.disaggregation.pd_producer import disaggregated_kv_role

    if getattr(config, "pipeline_parallel_size", 1) > 1:
        raise ValueError("Structured output does not support pipeline parallelism")
    if relaxed_mtp:
        raise ValueError("Structured output cannot use relaxed MTP verification")
    # A KV offload tier (LMCache CPU) is not P/D: the request still decodes on
    # this engine, so its grammar state is never handed to another one.
    if (
        kv_transfer
        or multimodal
        or getattr(config, "enable_rapidserve", False)
        or disaggregated_kv_role(config) is not None
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


# Keys the grammar of a request whose only constraint is its tool calls: the
# slot that carries a response JSON Schema carries this marker instead, and
# no answer schema is applied (`compile_json_schema` dispatches on the format).
TOOL_CALL_GRAMMAR_SCHEMA = '{"x-atom": "tool-call-grammar"}'


def tool_call_grammar_format(parser_cls, tools, tool_choice, starts_in_reasoning):
    """The tool-call grammar the parser declares for this request, or None."""
    if parser_cls is None or not tools or tool_choice == "none":
        return None
    if any(tool.get("type", "function") != "function" for tool in tools):
        return None  # only function tools have a parameter schema to hold to
    from .schema import check_schema_safety

    chosen, required = selected_tools(tools, tool_choice)
    functions = [tool.get("function", tool) for tool in chosen]
    for function in functions:
        # The same checks a response schema passes (#2862): a schema llguidance
        # would accept but cannot match within its budget fails the request at
        # decode time, after the client has been told it succeeded.
        check_schema_safety(function.get("parameters") or {})
    grammar = parser_cls.tool_call_grammar(
        functions, required=required, starts_in_reasoning=starts_in_reasoning
    )
    if grammar is None:
        return None
    return json.dumps({"kind": "lark", "grammar": grammar})


# Tool-call grammars that failed to compile, so a repeated tool list does not
# rerun the all-rank preparation RPC to fail again.
_uncompilable_tool_grammars: OrderedDict = OrderedDict()


async def configure_tool_call_grammar(
    engine,
    config,
    sampling_params,
    channel,
    *,
    parser_cls=None,
    tools=None,
    tool_choice=None,
    kv_transfer=None,
    multimodal=False,
    relaxed_mtp=False,
    synthetic=False,
):
    """Hold the tool calls of a request without a response format to the
    declared schemas, when the parser declares a grammar for them.

    Admission never fails because of it: without the optional dependency, on
    a deployment or request that cannot run a grammar, with a schema that does
    not pass the safety checks, or when preparation fails or times out, the
    output stays unconstrained. A grammar error during decoding fails the
    request like any structured-output error.
    """
    if not tools or parser_cls is None or not structured_output.available():
        return False
    from atom.utils import envs

    if not envs.ATOM_TOOL_CALL_GRAMMAR:
        return False
    required_channel = sampling_params.required_channel
    fmt = None
    try:
        fmt = tool_call_grammar_format(
            parser_cls, tools, tool_choice, bool(getattr(channel, "starts_open", False))
        )
        if fmt is None or fmt in _uncompilable_tool_grammars:
            return False
        validate_execution_options(
            config,
            sampling_params,
            kv_transfer=kv_transfer,
            multimodal=multimodal,
            relaxed_mtp=relaxed_mtp,
            synthetic=synthetic,
        )
        sampling_params.json_schema = TOOL_CALL_GRAMMAR_SCHEMA
        sampling_params.structured_tool_format = fmt
        sampling_params.structured_reasoning_end = ""
        # The grammar already requires the tools section for a required choice.
        sampling_params.required_channel = None
        await asyncio.to_thread(
            engine.prepare_structured_request, sampling_params, kv_transfer, multimodal
        )
    # ValueError: an unsafe schema or a worker refusing the grammar;
    # TimeoutError: preparation past its deadline; RuntimeError: a worker RPC.
    except (ValueError, TimeoutError, RuntimeError) as exc:
        if sampling_params.json_schema == TOOL_CALL_GRAMMAR_SCHEMA:
            logger.warning("Tool calls left unconstrained: %s", exc)
            if isinstance(exc, ValueError):
                _uncompilable_tool_grammars[fmt] = None
                if len(_uncompilable_tool_grammars) > 128:
                    _uncompilable_tool_grammars.popitem(last=False)
        sampling_params.json_schema = None
        sampling_params.structured_tool_format = None
        sampling_params.required_channel = required_channel
        return False
    return True


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
):
    if schema is None:
        return
    validate_execution_options(
        config,
        sampling_params,
        kv_transfer=kv_transfer,
        multimodal=multimodal,
        relaxed_mtp=relaxed_mtp,
        synthetic=synthetic,
    )
    tool_format = structured_tool_format(parser_cls, tools, tool_choice)
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
