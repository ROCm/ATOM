# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat completion handler for the OpenAI-compatible API."""

import logging
import time
from collections.abc import AsyncGenerator
from functools import partial
from typing import Any

from atom.structured_output.validation import raise_for_structured_error

from .logprobs import (
    LogprobCollector,
    LogprobError,
    LogprobsProcessor,
    LogprobText,
    claim_logprobs,
)
from .protocol import (
    CHAT_COMPLETION_CHUNK_OBJECT,
    STREAM_DONE_MESSAGE,
    TOOL_CHOICE_VALUES,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ended_by_stop,
    openai_stop_reason_with_calls,
)
from .reasoning import (
    NO_REASONING,
    ReasoningChannel,
)
from .reasoning_controls import (
    CHAT_REASONING_RULES,
    ReasoningRules,
    resolve_reasoning_controls,
)
from .sse import data_frame
from .streaming_dispatch import StreamOutputCollector
from .tool_parser import ToolCallStreamParser, parse_tool_calls
from .tool_parser.registry import forbids_tool_calls
from .tool_parser.tool_parser import qualified_tool_name

logger = logging.getLogger("atom")


# ============================================================================
# Request validation & thinking control
# ============================================================================


def normalize_chat_tools(tools: Any) -> Any:
    """Accept Anthropic-style tools on the OpenAI-compatible endpoint.

    Well-formed OpenAI tools and malformed values are left unchanged so the
    existing validator remains authoritative. Only the unambiguous Anthropic
    shape (name + input_schema, without type/function) is converted.
    """
    if not isinstance(tools, list):
        return tools

    normalized = []
    for tool in tools:
        if (
            isinstance(tool, dict)
            and "type" not in tool
            and "function" not in tool
            and isinstance(tool.get("name"), str)
            and isinstance(tool.get("input_schema"), dict)
        ):
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    },
                }
            )
        else:
            normalized.append(tool)
    return normalized


def _tool_parser_for_request(parser_cls, tools):
    """Return the parser this request actually needs.

    A no-tools request cannot produce a dispatchable tool call. Disable parsers
    whose markers only open tool-call regions so an unclosed marker remains
    ordinary content instead of withholding the stream until EOS. Parsers that
    consume model-wide channel framing must remain active without tools.
    """
    if parser_cls is None or tools:
        return parser_cls
    consumes_channel_framing = any(
        not parser_cls.opens_region(marker) for marker in parser_cls.START_MARKERS
    )
    return parser_cls if consumes_channel_framing else None


def resolve_thinking(
    request: ChatCompletionRequest, *, rules: ReasoningRules = CHAT_REASONING_RULES
) -> tuple[bool | None, str | int | None]:
    """Read Chat control fields and resolve them with the selected model rules."""
    return resolve_reasoning_controls(
        thinking=request.thinking, effort=request.reasoning_effort, rules=rules
    )


def _validate_one_tool(tool: Any, index: int) -> str:
    if not isinstance(tool, dict):
        # ValueError (not TypeError) so the handler maps it to HTTP 400.
        raise ValueError(f"tools[{index}] must be an object")  # noqa: TRY004
    if tool.get("type") != "function":
        raise ValueError(f"tools[{index}].type must be 'function'")
    fn = tool.get("function")
    if not isinstance(fn, dict):
        raise ValueError(f"tools[{index}].function must be an object")  # noqa: TRY004
    try:
        return qualified_tool_name(tool)
    except ValueError as exc:
        raise ValueError(f"tools[{index}].{exc}") from exc


def validate_tool_list(tools: Any) -> None:
    if tools is None:
        return
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")  # noqa: TRY004
    seen: set[str] = set()
    for i, tool in enumerate(tools):
        name = _validate_one_tool(tool, i)
        if name in seen:
            raise ValueError(f"duplicate tool name: {name}")
        seen.add(name)


def _message_field(msg: Any, key: str) -> Any:
    return msg.get(key) if isinstance(msg, dict) else getattr(msg, key, None)


def request_tools(request: ChatCompletionRequest) -> list:
    """Every tool the request declares: the top-level list plus the tools
    Kimi-K3 lets a message declare (dynamic tools), in order."""
    messages = request.messages if request.messages is not None else request.prompt
    declared = [_message_field(m, "tools") for m in messages or []]
    return [
        *(request.tools or []),
        *(t for tools in declared if isinstance(tools, list) for t in tools),
    ]


def _validate_message_tools(messages: Any) -> None:
    """Each message's dynamic tools pass the same shape check as top-level
    tools. Names are checked for duplicates within one declaration only."""
    for i, msg in enumerate(messages or []):
        tools = _message_field(msg, "tools")
        if tools is None:
            continue
        try:
            validate_tool_list(tools)
        except ValueError as exc:
            sep = "." if str(exc).startswith("tools") else ".tools: "
            raise ValueError(f"messages[{i}]{sep}{exc}") from exc


def validate_history_tool_calls(messages: Any) -> None:
    """History tool_calls must carry a function name and renderable arguments."""
    for i, msg in enumerate(messages or []):
        tool_calls = _message_field(msg, "tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            raise ValueError(  # noqa: TRY004
                f"messages[{i}].tool_calls must be an array"
            )
        for j, tc in enumerate(tool_calls):
            where = f"messages[{i}].tool_calls[{j}]"
            if not isinstance(tc, dict):
                raise ValueError(f"{where} must be an object")  # noqa: TRY004
            # Kimi-K3's encoder also takes the flat {"name", "arguments"} form.
            fn = tc.get("function", tc)
            if not isinstance(fn, dict):
                raise ValueError(f"{where}.function must be an object")  # noqa: TRY004
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{where}.function.name must be a non-empty string")
            args = fn.get("arguments")
            if args is not None and not isinstance(args, (str, dict)):
                raise ValueError(
                    f"{where}.function.arguments must be a string or an object"
                )


def validate_stream_options(stream: bool | None, stream_options: Any) -> None:
    """``stream_options`` only with ``stream=true``; ``include_usage`` a boolean."""
    if stream_options is None:
        return
    if stream is not True:
        raise ValueError("stream_options requires stream=true")
    include_usage = stream_options.get("include_usage")
    if include_usage is not None and type(include_usage) is not bool:
        raise ValueError("stream_options.include_usage must be a boolean")


def validate_chat_request(request: ChatCompletionRequest) -> None:
    """Validate tool / tool_choice / response_format shape before dispatch.

    Raises ``ValueError`` (surfaced as HTTP 400) on malformed input so the
    engine is never handed a request the chat template cannot render.
    """
    if request.top_logprobs is not None and not request.logprobs:
        raise ValueError("top_logprobs requires logprobs=true")
    validate_stream_options(request.stream, request.stream_options)
    validate_tool_list(request.tools)
    messages = request.messages if request.messages is not None else request.prompt
    _validate_message_tools(messages)
    validate_history_tool_calls(messages)

    tool_choice = request.tool_choice
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(
                    f"tool_choice string must be one of {sorted(TOOL_CHOICE_VALUES)}"
                )
            if tool_choice == "required" and not request_tools(request):
                raise ValueError("tool_choice 'required' needs a non-empty tools list")
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") != "function":
                raise ValueError("tool_choice object must have type 'function'")
            fn = tool_choice.get("function")
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise ValueError(  # noqa: TRY004
                    "tool_choice.function.name must be a string"
                )
            # A named tool_choice must reference a declared tool, top-level or
            # declared in a message.
            names = {
                qualified_tool_name(t)
                for t in request_tools(request)
                if isinstance(t, dict) and isinstance(t.get("function"), dict)
            }
            if qualified_tool_name(tool_choice) not in names:
                raise ValueError(f"tool_choice names unknown tool: {fn['name']}")
        else:
            raise ValueError("tool_choice must be a string or an object")

    rf = request.response_format
    if rf is not None:
        if not isinstance(rf, dict):
            raise ValueError("response_format must be an object")
        rf_type = rf.get("type")
        if rf_type not in ("text", "json_object", "json_schema"):
            raise ValueError(
                "response_format.type must be 'text', 'json_object', or 'json_schema'"
            )
        if rf_type == "json_schema":
            js = rf.get("json_schema")
            if not isinstance(js, dict) or not isinstance(js.get("schema"), dict):
                raise ValueError("response_format.json_schema.schema must be an object")
            if js.get("strict") is not None and type(js["strict"]) is not bool:
                raise ValueError("response_format.json_schema.strict must be a boolean")


def create_chat_chunk(
    request_id: str,
    model: str,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    index: int = 0,
    *,
    include_usage: bool = False,
    kv_transfer_params: dict | None = None,
    with_logprobs: bool = False,
    logprob_values: dict | None = None,
) -> str:
    """Create a chat completion chunk in SSE format.

    ``index`` selects the ``choices[0].index`` field so fan-out siblings
    (SamplingParams.n>1) can be multiplexed over a single stream.
    """
    chunk = {
        "id": request_id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta if delta else {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    if with_logprobs and delta:
        if any(
            value and not isinstance(value, LogprobText)
            for key, value in delta.items()
            if key in ("content", "reasoning_content")
        ):
            raise LogprobError("Visible text has no token-byte source spans")
        probabilities = logprob_values
        if probabilities:
            chunk["choices"][0]["logprobs"] = probabilities
        chunk["choices"][0]["delta"] = {
            field: str(text) if isinstance(text, LogprobText) else text
            for field, text in delta.items()
        }
    if usage is not None or include_usage:
        chunk["usage"] = usage
    if kv_transfer_params is not None:
        chunk["kv_transfer_params"] = kv_transfer_params
    return data_frame(chunk)


def _chat_deltas(reader, parser, text, finished, finish_reason):
    segments = reader.process(text)
    if finished:
        segments.extend(reader.flush(stopped=ended_by_stop(finish_reason)))
    for field, value in segments:
        if field == "reasoning_content":
            if value:
                yield field, {field: value}
        else:
            for kind, data in parser.process(value):
                if kind == "content":
                    yield kind, {"content": data}
                elif kind in ("tool_call_start", "tool_call_args"):
                    yield kind, {"tool_calls": [data]}
    if finished:
        for kind, data in parser.flush():
            if kind == "content":
                yield kind, {"content": data}
            elif kind in ("tool_call_start", "tool_call_args"):
                yield kind, {"tool_calls": [data]}


def _logprob_frames(pending, make_chunk, request_id, model, index=0, held_text=""):
    # A tool region may hold older content while reasoning is already emitted.
    # Resolve all available spans in source order, then serialize the deltas in
    # their existing event order. A boundary token belongs to one field only.
    cutoff = (
        min(start for start, _ in held_text.source_ranges)
        if isinstance(held_text, LogprobText) and held_text
        else float("inf")
    )
    ready, held = [], []
    for delta in pending:
        spans = [
            end
            for field, text in delta.items()
            if field in ("content", "reasoning_content")
            and isinstance(text, LogprobText)
            for _, end in text.source_ranges
        ]
        (ready if all(end <= cutoff for end in spans) else held).append(delta)
    pending[:] = held
    values = claim_logprobs(ready)
    frames = [
        make_chunk(
            request_id, model, delta=delta, index=index, logprob_values=probabilities
        )
        for delta, probabilities in zip(ready, values)
    ]
    return frames


def chat_finish_frames(
    request_id,
    model,
    reasons,
    usage,
    kv_transfer_params=None,
    *,
    usage_on_finish=False,
    include_usage=False,
):
    """Serialize final choices once, with aggregate usage on the last choice."""
    frames = []
    successful = [i for i, reason in enumerate(reasons) if reason is not None]
    for index in successful:
        reason = reasons[index]
        last = usage_on_finish and index == successful[-1]
        frames.append(
            create_chat_chunk(
                request_id,
                model,
                finish_reason=reason,
                index=index,
                usage=usage if last else None,
                include_usage=usage_on_finish and include_usage,
                kv_transfer_params=kv_transfer_params if last else None,
            )
        )
    if not usage_on_finish:
        payload = {
            "id": request_id,
            "object": CHAT_COMPLETION_CHUNK_OBJECT,
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": usage,
        }
        if kv_transfer_params is not None:
            payload["kv_transfer_params"] = kv_transfer_params
        frames.append(data_frame(payload))
    return "".join(frames) + STREAM_DONE_MESSAGE


async def stream_chat_response(
    request_id: str,
    model: str,
    stream_collector: StreamOutputCollector,
    seq_id: int,
    num_prompt_tokens: int,
    cleanup_stream,
    cleanup_request,
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
    usage_on_finish: bool = False,
    include_usage: bool = False,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response with reasoning and tool calls.

    Yields SSE chunks with:
    - reasoning_content deltas during thinking phase
    - content deltas for the answer
    - tool_calls deltas when model invokes tools

    ``num_prompt_tokens`` is the engine-computed prompt length (``Sequence.
    num_prompt_tokens``); reusing it avoids re-tokenizing the prompt on the
    event loop at stream start.
    """
    make_chunk = (
        partial(create_chat_chunk, include_usage=True)
        if usage_on_finish and include_usage
        else create_chat_chunk
    )
    any_logprobs = isinstance(stream_collector, LogprobCollector)
    if any_logprobs:
        make_chunk = partial(make_chunk, with_logprobs=True)
    num_tokens_input = num_prompt_tokens
    num_tokens_output = 0
    num_cached_tokens = 0
    # The engine's own reason, kept so the final chunk can report it. The
    # streaming paths hardcoded `stop`, so a response the engine cut off at
    # `max_tokens` was reported as complete while `stream=false` reported
    # `length` for the same generation -- and a `stop_<token_id>` stop, which
    # the Anthropic path maps to `stop_sequence`, collapsed the same way.
    engine_finish_reason: str | None = None
    reasoning_filter = reasoning.stream()
    tool_parser = ToolCallStreamParser(
        tools=tools,
        parser_cls=_tool_parser_for_request(tool_parser_cls, tools),
        suppress_calls=forbids_tool_calls(tool_choice),
    )
    has_tool_calls = False

    kv_transfer_params_value = None

    # Assume abort until the engine's finished chunk arrives. A client
    # disconnect closes the generator before then, leaving this True so the
    # finally aborts the still-running seq; normal completion flips it to False.
    pending = [] if any_logprobs else None
    aborted = True
    try:
        role_sent = False
        while True:
            chunk_data = await stream_collector.get()
            if any_logprobs and chunk_data.get("finish_reason") == "logprob_error":
                yield data_frame(
                    {
                        "error": {
                            "type": "server_error",
                            "code": "logprob_error",
                            "message": chunk_data.get(
                                "logprob_error", "Token logprob processing failed"
                            ),
                            "index": 0,
                        }
                    }
                )
                yield STREAM_DONE_MESSAGE
                return
            raise_for_structured_error(chunk_data.get("finish_reason"))

            if not role_sent:
                yield make_chunk(
                    request_id, model, delta={"role": "assistant", "content": ""}
                )
                role_sent = True
            new_text = chunk_data["text"]
            if chunk_data.get("finish_reason"):
                engine_finish_reason = chunk_data["finish_reason"]
            num_tokens_output += len(chunk_data.get("token_ids", []))
            _ct = chunk_data.get("num_cached_tokens", 0)
            if _ct:
                num_cached_tokens = _ct

            if "kv_transfer_params" in chunk_data:
                kv_transfer_params_value = chunk_data["kv_transfer_params"]

            try:
                done = bool(chunk_data.get("finished"))
                for kind, delta in _chat_deltas(
                    reasoning_filter, tool_parser, new_text, done, engine_finish_reason
                ):
                    has_tool_calls |= kind == "tool_call_args"
                    if pending is None:
                        yield make_chunk(request_id, model, delta=delta)
                    else:
                        pending.append(delta)
                if pending is not None:
                    for frame in _logprob_frames(
                        pending,
                        make_chunk,
                        request_id,
                        model,
                        held_text=tool_parser.held_text if not done else "",
                    ):
                        yield frame
                    if not tool_parser.has_pending_text:
                        stream_collector.trackers[0].discard_resolved(
                            reasoning_filter.held_text
                        )
                if done:
                    break
            except LogprobError as exc:
                logger.exception("Logprob attribution failed for %s", request_id)
                yield data_frame(
                    {
                        "error": {
                            "type": "server_error",
                            "code": "logprob_error",
                            "message": str(exc),
                            "index": 0,
                        }
                    }
                )
                yield STREAM_DONE_MESSAGE
                return

        aborted = False

        # Final chunks
        finish_reason = openai_stop_reason_with_calls(
            engine_finish_reason, has_tool_calls
        )
        usage = {
            "prompt_tokens": num_tokens_input,
            "completion_tokens": num_tokens_output,
            "total_tokens": num_tokens_input + num_tokens_output,
            "prompt_tokens_details": {"cached_tokens": num_cached_tokens},
        }
        # Coalesce terminal events without reparsing already serialized deltas.
        yield chat_finish_frames(
            request_id,
            model,
            [finish_reason],
            usage,
            kv_transfer_params_value,
            usage_on_finish=usage_on_finish,
            include_usage=include_usage,
        )
    finally:
        if any_logprobs:
            for tracker in stream_collector.trackers:
                tracker.close()
        cleanup_stream(seq_id, aborted=aborted)
        cleanup_request(request_id)


def _build_chat_choice(
    raw_text: str,
    finish_reason: str | None,
    index: int = 0,
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
    logprob_codec=None,
    token_ids=(),
    token_logprobs=None,
    top_logprobs=0,
) -> dict[str, Any]:
    """Build one entry of ``choices[...]`` from a raw output string.

    Factored out of :func:`build_chat_response` so multi-sample responses
    (SamplingParams.n>1) can reuse the reasoning + tool-call separation
    without duplicating the logic.
    """
    raise_for_structured_error(finish_reason)
    if finish_reason == "logprob_error":
        raise LogprobError("Engine returned missing or misaligned token logprobs")
    tracker = None
    if logprob_codec is not None:
        tracker = LogprobsProcessor(logprob_codec, top_logprobs)
        raw_text = tracker.feed(token_ids, token_logprobs, finished=True)
    reasoning_content, content_with_tools = reasoning.split(
        raw_text, stopped=ended_by_stop(finish_reason)
    )
    content, tool_calls = parse_tool_calls(
        content_with_tools,
        tools,
        parser_cls=_tool_parser_for_request(tool_parser_cls, tools),
        suppress_calls=forbids_tool_calls(tool_choice),
    )

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [tc.to_dict() for tc in tool_calls]

    # `is not None`, not truthiness: `""` is `Sequence.leave_reason`'s initial
    # value and what the engine forwards for a response with no recorded
    # reason. A truthiness test reported `finish_reason: null` for it, which an
    # OpenAI client reads as "still in progress" and some SDKs raise on.
    effective_finish_reason = (
        openai_stop_reason_with_calls(finish_reason, bool(tool_calls))
        if (tool_calls or finish_reason is not None)
        else None
    )
    choice = {
        "index": index,
        "message": message,
        "finish_reason": effective_finish_reason,
    }
    if tracker is not None:
        fields = {"content": content}
        if reasoning_content is not None:
            fields["reasoning_content"] = reasoning_content
        choice["logprobs"] = tracker.take_fields(fields)
        message.update({field: str(value) for field, value in fields.items()})
        tracker.close()
    return choice


def build_chat_response(
    request_id: str,
    model: str,
    raw_text: str,
    final_output: dict[str, Any],
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
    logprob_codec=None,
    top_logprobs=0,
) -> ChatCompletionResponse:
    """Build a non-streaming chat completion response (single choice)."""
    response = ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            _build_chat_choice(
                raw_text,
                final_output["finish_reason"],
                index=0,
                tools=tools,
                tool_choice=tool_choice,
                reasoning=reasoning,
                tool_parser_cls=tool_parser_cls,
                logprob_codec=logprob_codec,
                token_ids=final_output.get("token_ids", ()),
                token_logprobs=(
                    final_output.get("logprobs") if logprob_codec is not None else None
                ),
                top_logprobs=top_logprobs,
            )
        ],
        usage={
            "prompt_tokens": final_output["num_tokens_input"],
            "completion_tokens": final_output["num_tokens_output"],
            "total_tokens": final_output["num_tokens_input"]
            + final_output["num_tokens_output"],
            "prompt_tokens_details": {
                "cached_tokens": final_output.get("num_cached_tokens", 0)
            },
            "ttft_s": round(final_output.get("ttft", 0.0), 4),
            "tpot_s": round(final_output.get("tpot", 0.0), 4),
            "latency_s": round(final_output.get("latency", 0.0), 4),
        },
    )
    update: dict[str, Any] = {}
    if "kv_transfer_output_meta_info" in final_output:
        update["kv_transfer_params"] = final_output["kv_transfer_output_meta_info"]
    if "prompt_token_ids" in final_output:
        update["prompt_token_ids"] = final_output["prompt_token_ids"]
    if update:
        response = response.model_copy(update=update)
    return response


def build_chat_response_multi(
    request_id: str,
    model: str,
    final_outputs: list[dict[str, Any]],
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
    logprob_codec=None,
    top_logprobs=0,
) -> ChatCompletionResponse:
    """Build a non-streaming response with one choice per fan-out sibling.

    Assumes all ``final_outputs`` share the same prompt and therefore the
    same ``num_tokens_input``. Completion-token counts are summed across
    siblings for usage; ttft/tpot/latency are reported as the max observed
    across siblings, which approximates wall-clock time to return the full
    multi-sample response to the client.
    """
    assert final_outputs, "build_chat_response_multi requires at least one output"
    choices = [
        _build_chat_choice(
            out["text"],
            out["finish_reason"],
            index=i,
            tools=tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
            tool_parser_cls=tool_parser_cls,
            logprob_codec=logprob_codec,
            token_ids=out.get("token_ids", ()),
            token_logprobs=(out.get("logprobs") if logprob_codec is not None else None),
            top_logprobs=top_logprobs,
        )
        for i, out in enumerate(final_outputs)
    ]
    prompt_tokens = final_outputs[0]["num_tokens_input"]
    completion_tokens = sum(out["num_tokens_output"] for out in final_outputs)
    response = ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=choices,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": final_outputs[0].get("num_cached_tokens", 0)
            },
            "ttft_s": round(
                max((out.get("ttft", 0.0) for out in final_outputs), default=0.0), 4
            ),
            "tpot_s": round(
                max((out.get("tpot", 0.0) for out in final_outputs), default=0.0), 4
            ),
            "latency_s": round(
                max((out.get("latency", 0.0) for out in final_outputs), default=0.0), 4
            ),
            "num_choices": len(final_outputs),
        },
    )
    # Sibling outputs share the first output's prompt IDs.
    if "prompt_token_ids" in final_outputs[0]:
        response = response.model_copy(
            update={"prompt_token_ids": final_outputs[0]["prompt_token_ids"]}
        )
    return response


async def stream_chat_response_fanout(
    request_id: str,
    model: str,
    shared_collector: StreamOutputCollector,
    seq_ids: list[int],
    num_prompt_tokens: int,
    cleanup_stream,
    cleanup_request,
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
    usage_on_finish: bool = False,
    include_usage: bool = False,
) -> AsyncGenerator[str, None]:
    """Streaming variant that multiplexes ``len(seq_ids)`` fan-out siblings
    into a single SSE stream, tagging every chunk with ``choices[0].index``.

    The shared queue receives ``(sibling_index, chunk_data)`` tuples from
    the engine callbacks registered in :func:`setup_streaming_request_fanout`.
    Reasoning + tool-call state is kept independently per sibling.

    ``num_prompt_tokens`` is the engine-computed prompt length shared by all
    siblings (they tokenize the same prompt once); reusing it avoids
    re-tokenizing on the event loop at stream start.
    """
    n = len(seq_ids)
    make_chunk = (
        partial(create_chat_chunk, include_usage=True)
        if usage_on_finish and include_usage
        else create_chat_chunk
    )
    any_logprobs = isinstance(shared_collector, LogprobCollector)
    if any_logprobs:
        make_chunk = partial(make_chunk, with_logprobs=True)
    num_tokens_input = num_prompt_tokens
    num_tokens_output = [0] * n
    # Every sibling answers the same prompt, so they start in the same state.
    reasoning_filters = [reasoning.stream() for _ in range(n)]
    tool_parsers = [
        ToolCallStreamParser(
            tools=tools,
            parser_cls=_tool_parser_for_request(tool_parser_cls, tools),
            suppress_calls=forbids_tool_calls(tool_choice),
        )
        for _ in range(n)
    ]
    has_tool_calls = [False] * n
    # See the single-choice path: the engine's reason, per sibling.
    engine_finish_reasons: list[str | None] = [None] * n
    finished = [False] * n
    kv_transfer_params_value = None
    num_cached_tokens = 0

    # Assume abort until every sibling reports finished; a client disconnect
    # closes the generator first, leaving this True so the finally aborts
    # whichever siblings are still running.
    pending = [[] for _ in range(n)] if any_logprobs else None
    aborted = True
    try:
        role_sent = [False] * n
        while not all(finished):
            idx, chunk_data = await shared_collector.get()
            if finished[idx]:
                continue
            if any_logprobs and chunk_data.get("finish_reason") == "logprob_error":
                logger.error("Logprob processing failed for choice %s", idx)
                cleanup_stream(seq_ids[idx], aborted=True)
                finished[idx] = True
                engine_finish_reasons[idx] = "error"
                yield data_frame(
                    {
                        "error": {
                            "type": "server_error",
                            "code": "logprob_error",
                            "message": chunk_data.get(
                                "logprob_error", "Token logprob processing failed"
                            ),
                            "index": idx,
                        }
                    }
                )
                continue
            raise_for_structured_error(chunk_data.get("finish_reason"))

            if not role_sent[idx]:
                yield make_chunk(
                    request_id,
                    model,
                    delta={"role": "assistant", "content": ""},
                    index=idx,
                )
                role_sent[idx] = True

            if finished[idx]:
                # Defensive: should not happen, engine emits finished once per seq.
                continue
            new_text = chunk_data["text"]
            if chunk_data.get("finish_reason"):
                engine_finish_reasons[idx] = chunk_data["finish_reason"]
            num_tokens_output[idx] += len(chunk_data.get("token_ids", []))
            _ct = chunk_data.get("num_cached_tokens", 0)
            if _ct:
                num_cached_tokens = _ct

            if "kv_transfer_params" in chunk_data:
                kv_transfer_params_value = chunk_data["kv_transfer_params"]

            try:
                done = bool(chunk_data.get("finished"))
                for kind, delta in _chat_deltas(
                    reasoning_filters[idx],
                    tool_parsers[idx],
                    new_text,
                    done,
                    engine_finish_reasons[idx],
                ):
                    has_tool_calls[idx] |= kind == "tool_call_args"
                    if pending is None:
                        yield make_chunk(request_id, model, delta=delta, index=idx)
                    else:
                        pending[idx].append(delta)
                if pending is not None:
                    for frame in _logprob_frames(
                        pending[idx],
                        make_chunk,
                        request_id,
                        model,
                        idx,
                        held_text=tool_parsers[idx].held_text if not done else "",
                    ):
                        yield frame
                    if not tool_parsers[idx].has_pending_text:
                        shared_collector.trackers[idx].discard_resolved(
                            reasoning_filters[idx].held_text
                        )
                if done:
                    finished[idx] = True
            except LogprobError as exc:
                logger.exception(
                    "Logprob attribution failed for %s choice %s", request_id, idx
                )
                cleanup_stream(seq_ids[idx], aborted=True)
                finished[idx] = True
                engine_finish_reasons[idx] = "error"
                yield data_frame(
                    {
                        "error": {
                            "type": "server_error",
                            "code": "logprob_error",
                            "message": str(exc),
                            "index": idx,
                        }
                    }
                )

        aborted = False

        usage = {
            "prompt_tokens": num_tokens_input,
            "completion_tokens": sum(num_tokens_output),
            "total_tokens": num_tokens_input + sum(num_tokens_output),
            "num_choices": n,
            "prompt_tokens_details": {"cached_tokens": num_cached_tokens},
        }
        # Coalesce terminal events without reparsing already serialized deltas.
        yield chat_finish_frames(
            request_id,
            model,
            [
                (
                    None
                    if engine_finish_reasons[i] == "error"
                    else openai_stop_reason_with_calls(
                        engine_finish_reasons[i], has_tool_calls[i]
                    )
                )
                for i in range(n)
            ],
            usage,
            kv_transfer_params_value,
            usage_on_finish=usage_on_finish,
            include_usage=include_usage,
        )
    finally:
        if any_logprobs:
            for tracker in shared_collector.trackers:
                tracker.close()
        for sid in seq_ids:
            cleanup_stream(sid, aborted=aborted)
        cleanup_request(request_id)
