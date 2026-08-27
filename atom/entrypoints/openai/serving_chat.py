# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat completion handler for the OpenAI-compatible API."""

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from .protocol import (
    CHAT_COMPLETION_CHUNK_OBJECT,
    STREAM_DONE_MESSAGE,
    TOOL_CHOICE_VALUES,
    ChatCompletionRequest,
    ChatCompletionResponse,
    openai_stop_reason_with_calls,
)
from .reasoning import (
    NO_REASONING,
    VALID_TEMPLATE_EFFORTS,
    ReasoningChannel,
)
from .sse import data_frame
from .streaming_dispatch import StreamOutputCollector
from .tool_parser import ToolCallStreamParser, parse_tool_calls
from .tool_parser.registry import forbids_tool_calls
from .tool_parser.tool_parser import usable_tool_name

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


def resolve_thinking(request: ChatCompletionRequest) -> tuple[bool | None, str | None]:
    """Resolve (enabled, effort) from the request's thinking / reasoning_effort.

    ``thinking`` (extra_body) takes precedence over ``reasoning_effort``.
    Thinking is disabled when ``thinking.type == "disabled"`` or
    ``reasoning_effort == "none"``. Effort is only returned when it is one of
    the values the template understands.

    ``enabled`` is ``None`` when the request stated no preference -- setting
    an effort is not asking for reasoning to be switched *on*. Collapsing
    that to ``True`` was harmless while the caller wrote a key no template
    read; once it wrote the template's real switch, a request carrying only
    `reasoning_effort` re-enabled reasoning over an operator's
    `--default-chat-template-kwargs '{"enable_thinking": false}'`, and over a
    client's own `chat_template_kwargs`, because the toggle is merged last.
    """
    thinking = request.thinking or {}
    enabled: bool | None = None
    if isinstance(thinking, dict) and thinking.get("type") is not None:
        enabled = thinking.get("type") != "disabled"
    if request.reasoning_effort == "none":
        enabled = False

    effort = None
    if isinstance(thinking, dict) and thinking.get("effort") is not None:
        effort = thinking.get("effort")
    elif request.reasoning_effort is not None:
        effort = request.reasoning_effort
    if effort not in VALID_TEMPLATE_EFFORTS:
        effort = None
    return enabled, effort


def _validate_one_tool(tool: Any, index: int) -> None:
    if not isinstance(tool, dict):
        # ValueError (not TypeError) so the handler maps it to HTTP 400.
        raise ValueError(f"tools[{index}] must be an object")  # noqa: TRY004
    if tool.get("type") != "function":
        raise ValueError(f"tools[{index}].type must be 'function'")
    fn = tool.get("function")
    if not isinstance(fn, dict):
        raise ValueError(f"tools[{index}].function must be an object")  # noqa: TRY004
    name = fn.get("name")
    # The same predicate the parsers apply to a name the *model* writes. The
    # second grammar that used to be here, `^[A-Za-z_][A-Za-z0-9_-]*$`, was
    # the stricter: it rejected a leading digit that OpenAI's own
    # `^[a-zA-Z0-9_-]{1,64}$` allows, every non-ASCII name, and MCP's
    # `server.tool` -- a 400 before the model ever ran, while `/v1/messages`
    # validated nothing and accepted all three.
    if not isinstance(name, str) or not usable_tool_name(name):
        raise ValueError(
            f"tools[{index}].function.name must be a dispatchable name: "
            "a word character followed by word characters, dots or dashes"
        )


def validate_tool_list(tools: Any) -> None:
    if tools is None:
        return
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")  # noqa: TRY004
    seen: set[str] = set()
    for i, tool in enumerate(tools):
        _validate_one_tool(tool, i)
        name = tool["function"]["name"]
        if name in seen:
            raise ValueError(f"duplicate tool name: {name}")
        seen.add(name)


def validate_chat_request(request: ChatCompletionRequest) -> None:
    """Validate tool / tool_choice / response_format shape before dispatch.

    Raises ``ValueError`` (surfaced as HTTP 400) on malformed input so the
    engine is never handed a request the chat template cannot render.
    """
    validate_tool_list(request.tools)

    tool_choice = request.tool_choice
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(
                    f"tool_choice string must be one of {sorted(TOOL_CHOICE_VALUES)}"
                )
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") != "function":
                raise ValueError("tool_choice object must have type 'function'")
            fn = tool_choice.get("function")
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise ValueError(  # noqa: TRY004
                    "tool_choice.function.name must be a string"
                )
            # A named tool_choice must reference a declared tool.
            names = {
                t["function"]["name"]
                for t in (request.tools or [])
                if isinstance(t, dict) and isinstance(t.get("function"), dict)
            }
            if fn["name"] not in names:
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


def create_chat_chunk(
    request_id: str,
    model: str,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    index: int = 0,
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
    if usage is not None:
        chunk["usage"] = usage
    return data_frame(chunk)


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
    aborted = True
    try:
        role_sent = False
        while True:
            chunk_data = await stream_collector.get()

            if not role_sent:
                yield create_chat_chunk(
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

            # Phase 1: Process through reasoning filter
            segments = reasoning_filter.process(new_text)
            if chunk_data.get("finished", False):
                segments.extend(reasoning_filter.flush())

            # Phase 2: For content segments, check for tool calls
            for field, text in segments:
                if field == "reasoning_content":
                    if text:
                        yield create_chat_chunk(
                            request_id, model, delta={"reasoning_content": text}
                        )
                elif field == "content":
                    # Run through tool parser
                    events = tool_parser.process(text)
                    for event_type, data in events:
                        if event_type == "content":
                            yield create_chat_chunk(
                                request_id, model, delta={"content": data}
                            )
                        elif event_type == "tool_call_start":
                            yield create_chat_chunk(
                                request_id,
                                model,
                                delta={"tool_calls": [data]},
                            )
                        elif event_type == "tool_call_args":
                            has_tool_calls = True
                            yield create_chat_chunk(
                                request_id,
                                model,
                                delta={"tool_calls": [data]},
                            )

            if chunk_data.get("finished", False):
                # Flush tool parser
                for event_type, data in tool_parser.flush():
                    if event_type == "content":
                        yield create_chat_chunk(
                            request_id, model, delta={"content": data}
                        )
                    elif event_type == "tool_call_start":
                        yield create_chat_chunk(
                            request_id, model, delta={"tool_calls": [data]}
                        )
                    elif event_type == "tool_call_args":
                        has_tool_calls = True
                        yield create_chat_chunk(
                            request_id, model, delta={"tool_calls": [data]}
                        )
                break

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
        usage_chunk = {
            "id": request_id,
            "object": CHAT_COMPLETION_CHUNK_OBJECT,
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": usage,
        }
        if kv_transfer_params_value is not None:
            usage_chunk["kv_transfer_params"] = kv_transfer_params_value
        # Coalesce finish + usage + [DONE] into one send: at a wave boundary many
        # requests finalize at once, so collapsing 3 socket writes/req to 1 cuts
        # the syscalls that saturate the API event loop.
        yield (
            create_chat_chunk(request_id, model, finish_reason=finish_reason)
            + data_frame(usage_chunk)
            + STREAM_DONE_MESSAGE
        )
    finally:
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
) -> dict[str, Any]:
    """Build one entry of ``choices[...]`` from a raw output string.

    Factored out of :func:`build_chat_response` so multi-sample responses
    (SamplingParams.n>1) can reuse the reasoning + tool-call separation
    without duplicating the logic.
    """
    reasoning_content, content_with_tools = reasoning.split(raw_text)
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
    return {
        "index": index,
        "message": message,
        "finish_reason": effective_finish_reason,
    }


def build_chat_response(
    request_id: str,
    model: str,
    raw_text: str,
    final_output: dict[str, Any],
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
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
    if "kv_transfer_output_meta_info" in final_output:
        response = response.model_copy(
            update={
                "kv_transfer_params": final_output["kv_transfer_output_meta_info"],
            }
        )
    return response


def build_chat_response_multi(
    request_id: str,
    model: str,
    final_outputs: list[dict[str, Any]],
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
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
        )
        for i, out in enumerate(final_outputs)
    ]
    prompt_tokens = final_outputs[0]["num_tokens_input"]
    completion_tokens = sum(out["num_tokens_output"] for out in final_outputs)
    return ChatCompletionResponse(
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
    aborted = True
    try:
        role_sent = [False] * n
        while not all(finished):
            idx, chunk_data = await shared_collector.get()

            if not role_sent[idx]:
                yield create_chat_chunk(
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

            segments = reasoning_filters[idx].process(new_text)
            if chunk_data.get("finished", False):
                segments.extend(reasoning_filters[idx].flush())

            for field, text in segments:
                if field == "reasoning_content":
                    if text:
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"reasoning_content": text},
                            index=idx,
                        )
                elif field == "content":
                    for event_type, data in tool_parsers[idx].process(text):
                        if event_type == "content":
                            yield create_chat_chunk(
                                request_id, model, delta={"content": data}, index=idx
                            )
                        elif event_type == "tool_call_start":
                            yield create_chat_chunk(
                                request_id,
                                model,
                                delta={"tool_calls": [data]},
                                index=idx,
                            )
                        elif event_type == "tool_call_args":
                            has_tool_calls[idx] = True
                            yield create_chat_chunk(
                                request_id,
                                model,
                                delta={"tool_calls": [data]},
                                index=idx,
                            )

            if chunk_data.get("finished", False):
                for event_type, data in tool_parsers[idx].flush():
                    if event_type == "content":
                        yield create_chat_chunk(
                            request_id, model, delta={"content": data}, index=idx
                        )
                    elif event_type == "tool_call_start":
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"tool_calls": [data]},
                            index=idx,
                        )
                    elif event_type == "tool_call_args":
                        has_tool_calls[idx] = True
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"tool_calls": [data]},
                            index=idx,
                        )
                finished[idx] = True

        aborted = False

        usage = {
            "prompt_tokens": num_tokens_input,
            "completion_tokens": sum(num_tokens_output),
            "total_tokens": num_tokens_input + sum(num_tokens_output),
            "num_choices": n,
            "prompt_tokens_details": {"cached_tokens": num_cached_tokens},
        }
        usage_chunk = {
            "id": request_id,
            "object": CHAT_COMPLETION_CHUNK_OBJECT,
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": usage,
        }
        if kv_transfer_params_value is not None:
            usage_chunk["kv_transfer_params"] = kv_transfer_params_value
        # Coalesce the per-sibling finish chunks + usage + [DONE] into one send.
        yield (
            "".join(
                create_chat_chunk(
                    request_id,
                    model,
                    finish_reason=openai_stop_reason_with_calls(
                        engine_finish_reasons[i], has_tool_calls[i]
                    ),
                    index=i,
                )
                for i in range(n)
            )
            + data_frame(usage_chunk)
            + STREAM_DONE_MESSAGE
        )
    finally:
        for sid in seq_ids:
            cleanup_stream(sid, aborted=aborted)
        cleanup_request(request_id)
