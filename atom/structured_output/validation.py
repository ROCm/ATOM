# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request admission, worker preparation and final response validation."""

import asyncio
import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
            "Structured output with speculative logprobs requires token-level logprob support"
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
