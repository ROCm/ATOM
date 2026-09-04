# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
ATOM OpenAI-compatible API Server.

FastAPI-based server implementing OpenAI-compatible endpoints for chat
completions and text completions, with reasoning content separation for
thinking models (Kimi-K2, DeepSeek-R1, Qwen3, etc.).

Usage:
    python -m atom.entrypoints.openai_server --model <model> [options]
"""

import asyncio
import base64
import binascii
import contextlib
import io
import json
import logging
import os
import time
import urllib.request
import uuid
from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoProcessor, AutoTokenizer

if TYPE_CHECKING:
    from PIL import Image

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.llm_engine import _load_tokenizer
from atom.model_engine.multimodal import build_multimodal_inputs
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import new_token_ids
from atom.utils.arg_parser import FlexibleArgumentParser
from atom.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
    tune_gc,
)

from .chat_encoders import (
    apply_chat_template,
    chat_template_source,
    load_custom_message_encoder,
    render_probe_prompt,
    resolve_reasoning_toggle,
)
from .metrics import AtomMetricsExporter
from .protocol import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
)
from .reasoning import (
    ReasoningChannel,
    prompt_starts_in_reasoning,
    prompt_tokens_start_in_reasoning,
    template_opens_reasoning_implicitly,
    thinking_switched_off,
)
from .reasoning_dialects import resolve_dialect
from .serving_anthropic import (
    AnthropicBlocks,
    AnthropicMessagesRequest,
    anthropic_to_openai_messages,
    anthropic_to_openai_tools,
    build_anthropic_response,
    completes_a_tool_call,
    read_whole_blocks,
    stream_failure_frames,
    stream_message_delta,
    stream_message_start,
    stream_message_stop,
    tool_event_frames,
)
from .serving_chat import (
    build_chat_response,
    build_chat_response_multi,
    normalize_chat_tools,
    resolve_thinking,
    stream_chat_response,
    stream_chat_response_fanout,
    validate_chat_request,
    validate_tool_list,
)
from .serving_completion import (
    build_completion_response,
    build_completion_response_multi,
    stream_completion_response,
    stream_completion_response_fanout,
)
from .sse import event_frame
from .streaming_dispatch import (
    SYNTHETIC_TOKEN_TEXT,
    FrameWait,
    StreamBatchDispatcher,
    StreamOutputCollector,
)
from .tool_parser import (
    ToolCallStreamParser,
    flatten_tool_events,
)
from .tool_parser.registry import (
    TOOL_CALL_PARSER_HELP,
    forbids_tool_calls,
    resolve_tool_call_parser,
)

# Configure logging
logger = logging.getLogger("atom")

# Constants
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


# ============================================================================
# Global State
# ============================================================================

engine = None
tokenizer: AutoTokenizer | None = None
# The tool-call format this model emits, resolved once at startup from its
# chat template. `None` means none was recognised and tool calls, if any, are
# delivered as plain text -- said out loud at startup, never discovered here.
tool_call_parser_cls: type | None = None
# Whether this model's output begins inside the reasoning channel even when
# nothing in the prompt or the output says so -- DeepSeek-R1 closes a block it
# never opens. Read from the chat template at startup, because a single
# response cannot tell you: its first token is already reasoning and reads
# like an answer.
model_starts_in_reasoning: bool = False
reasoning_dialect: Any = None
# (kwarg, off-value) the chat template reads to switch reasoning off, resolved
# at startup by asking it; None when the template offers no such switch.
reasoning_toggle: tuple[str, Any, Any] | None = None
processor: Any | None = None
model_name: str = ""
default_chat_template_kwargs: dict[str, Any] = {}
custom_message_encoder: Any | None = None
_seq_id_to_request_id: dict[int, str] = {}
_stream_loops: dict[str, AbstractEventLoop] = {}
_request_start_times: dict[str, float] = {}
_request_logger: logging.Logger | None = None
_stream_batch_dispatcher: StreamBatchDispatcher | None = None
# `SYNTHETIC_TOKEN_TEXT` while a run's own text is meaningless, None otherwise.
# One switch for both delivery modes: they decode in different places, and a
# response that read differently depending on `stream` is the asymmetry the
# reasoning and tool-call readers were unified to remove.
synthetic_token_text: str | None = None


def delivered_text(token_ids) -> str:
    """The text a non-streaming response carries for these tokens.

    Decoded even when the answer is thrown away: the runs that stand their text
    in are measuring throughput, and skipping the work the measured server does
    would flatter it. The streaming half of this lives in
    `IncrementalStreamDetokenizer.update`.
    """
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    if synthetic_token_text is None:
        return decoded
    return synthetic_token_text * len(token_ids)


def reasoning_channel(
    prompt_opens: bool, *, template_kwargs: dict[str, Any] | None
) -> ReasoningChannel:
    """How to read this request's reasoning channel.

    One place, because it is one answer and both endpoints and both delivery
    modes need the same one. The dialect is the model's, resolved at startup;
    what varies per request is whether the output begins inside the channel.

    The render is why that is not just the model-level fact. A prompt that
    switched reasoning off does not open the channel, and
    `model_starts_in_reasoning` -- which describes the template with reasoning
    *on* -- was OR-ed in regardless, so on a model that begins inside the
    channel implicitly an ordinary answer came back entirely as
    `reasoning_content` with `content` empty.

    ``template_kwargs``, not the request's own `thinking`: the server's
    defaults and the client's `chat_template_kwargs` switch it too, and only
    the merged dict has all three. Reasoning that was asked not to happen and
    happened anyway is still reasoning -- `anthropic_drop_reasoning` exists to
    withhold it, and it can only withhold what was separated.
    """
    switched_off = thinking_switched_off(template_kwargs, reasoning_toggle)
    return ReasoningChannel(
        dialect=reasoning_dialect,
        starts_open=prompt_opens or (model_starts_in_reasoning and not switched_off),
    )


def anthropic_thinking_enabled(request: Any) -> bool:
    """Did this request ask for a reasoning channel?

    `thinking` is absent on most requests and `{"type": "disabled"}` is the
    spelling for switching it off, which is a non-empty dict and therefore
    truthy -- so `bool(request.thinking)` read the standard off-switch as on.
    """
    thinking = getattr(request, "thinking", None) or {}
    return bool(thinking) and thinking.get("type") != "disabled"


def anthropic_template_kwargs(
    request: Any, toggle: tuple[str, Any, Any] | None
) -> dict:
    """How `thinking` reaches the model: by not asking it to think.

    `thinking: disabled` is answered where the answer costs nothing -- in the
    prompt. The chat template's own switch is set, so the model emits no
    reasoning, so there is none to separate, none to discard, and none for the
    tool parser to misread. This is what SGLang does with the same field and
    what vLLM gets structurally by having no such field at all.

    Everything downstream is then unconditional, which is the point. Three
    attempts to handle an unwanted chain of thought *after* generating it each
    broke something else: discarding it returned an empty message, relabelling
    it as `text` handed the client the thing it declined, and declining to
    separate it fed a chain of thought to the tool parser, which read one
    model's musing about `<function=NAME>` as a call to a tool named `NAME`.
    The reasoning that is never produced needs none of that.

    Only when the field is actually present, which is also what SGLang keys
    on. Anthropic's default is thinking-off, but reading an absent field as
    "switch this model's reasoning off" would silently change what every
    existing caller gets back from a reasoning model. Absent means unstated,
    and unstated leaves the model's own default alone.

    Both directions go through the resolved toggle. Writing a hardcoded
    `thinking=True` for the on direction is a no-op on any template that reads
    another name -- the whole Qwen family -- so an explicit opt-in was
    discarded silently against a server default of off.

    ``toggle`` is ``None`` for a model whose template has no switch. There is
    then nothing to put in the prompt and `anthropic_drop_reasoning` takes
    over.
    """
    if getattr(request, "thinking", None) is None:
        return {}
    if toggle is None:
        return {}
    name, off_value, on_value = toggle
    return {name: on_value if anthropic_thinking_enabled(request) else off_value}


def anthropic_drop_reasoning(request: Any) -> bool:
    """Must this response's reasoning be withheld from the client?

    Whenever the request did not ask for it. Answering in the prompt is
    strictly better and `anthropic_template_kwargs` does it wherever it can,
    but that only reaches models whose template has a switch -- and it only
    reaches the ones that *asked*, since an absent field leaves the model's
    default alone. This is everything else.

    Withheld, not left unseparated: the reasoning still goes through the
    reasoning filter, so the tool parser never sees a model musing about
    `<function=NAME>` and calls a tool named `NAME`. Only the `thinking`
    blocks are dropped. A response that was *nothing but* reasoning then ends
    on an empty text block, which is the honest answer to "do not think" --
    the previous three attempts to fix this downstream all failed by trying to
    salvage content out of it.

    Absent counts as off *here*, and that is deliberately not what the
    prompt-level answer does. The two are different questions. What to put in
    the prompt is about the model's own default, and overriding a default
    nobody asked about would change what every existing caller gets from a
    reasoning model. What to put in the *response* is about this protocol's
    default, and Anthropic's is thinking-off: a client that never sends the
    field has no reason to expect `thinking` blocks, and one that validates
    block types or verifies the signature rejects them. Reading absent as
    "show it" sent a random-signature `thinking` block to every Claude Code
    and Anthropic-SDK caller talking to a Qwen3 or DeepSeek deployment.

    Separated either way -- only whether the client is *shown* it is decided
    here. That is what keeps a chain of thought out of the tool parser, which
    is a second reader of the same text.
    """
    return not anthropic_thinking_enabled(request)


# The engine's `leave_reason` in Anthropic's vocabulary. The scheduler emits
# exactly seven shapes; the three here line up by name, `stop_<token_id>` is
# handled below, and `aborted` / `unschedulable: ...` / "" have no counterpart
# in Anthropic's vocabulary and keep the default.
_ANTHROPIC_STOP_REASON = {
    "eos": "end_turn",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
}


def anthropic_stop_reason_with_calls(finish_reason: Any, has_calls: bool) -> str:
    """`max_tokens` outranks `tool_use`, for the reason `length` outranks
    `tool_calls` on the other endpoint (see
    :func:`~.protocol.openai_stop_reason_with_calls`): only one of the two is a
    warning, and a response cut off mid-call parses to a call with a silently
    truncated argument value."""
    reason = anthropic_stop_reason(finish_reason)
    if reason == "max_tokens":
        return reason
    return "tool_use" if has_calls else reason


def anthropic_stop_reason(finish_reason: Any) -> str:
    """The engine's leave reason as Anthropic spells it.

    Not routed through `protocol.openai_stop_reason`: that maps into OpenAI's
    vocabulary (`stop` / `length`), which shares no member with the keys here,
    so chaining them would send every reason to the default.

    `stop_<token_id>` is *not* `stop_sequence`, though it was mapped to it.
    The two come from different branches of the scheduler and mean opposite
    things: `stop_sequence` is one of the client's own `stop_sequences`
    matching, and Anthropic pairs it with the matched string in the response;
    `stop_<id>` is a model end-of-turn token from `stop_token_ids` firing,
    which is an ordinary `end_turn`.

    `stop_token_ids` is `generation_config.eos_token_id` minus the single
    `tokenizer.eos_token_id`, so any model declaring more than one EOS reaches
    this branch in normal operation -- Qwen3, Qwen3.5, gpt-oss.
    """
    reason = finish_reason or ""
    if reason.startswith("stop_") and reason != "stop_sequence":
        return "end_turn"
    return _ANTHROPIC_STOP_REASON.get(reason, "end_turn")


# Anthropic's own keepalive. Sent in place of a reasoning segment the request
# asked not to see: the bytes are being generated either way, and with nothing
# going out the socket is silent for the whole chain of thought -- on an
# R1-shaped 5019-character trace the first client-visible frame arrived after
# 5016 of them. Long enough to trip proxy and SDK idle-read timeouts, and the
# stall watchdog can only report it, not prevent it.
_ANTHROPIC_PING_FRAME = event_frame("ping", {"type": "ping"})
# Paced by the clock, not by the model: one ping per dropped reasoning
# *segment* is one per engine chunk, i.e. O(reasoning tokens) socket writes of
# 35 discarded bytes. The keepalive only has to beat proxy and SDK idle-read
# timeouts, which are tens of seconds.
_ANTHROPIC_PING_INTERVAL_SECONDS = 5.0
_metrics_exporter = AtomMetricsExporter()
_metrics_refresh_task: asyncio.Task | None = None
_METRICS_REFRESH_INTERVAL_SECONDS = 5.0


def _get_dp_session_affinity_ids(
    raw_request: Request | None,
) -> tuple[str | None, str | None]:
    """Extract AIPerf session lineage for CoreManager's DPA router.

    AIPerf keeps ``X-Correlation-ID`` stable across the turns of one session.
    When its Dynamo-affinity header option is enabled it also sends the parent
    session ID for forked agent trees. CoreManager retains that lineage for
    observability but deliberately assigns each child correlation ID its own
    strict cache owner, matching SGLang's agentic routing. Clients without a
    session header retain normal DP load balancing.
    """
    if os.environ.get("ATOM_DP_SESSION_AFFINITY", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None, None
    if raw_request is None:
        return None, None

    session_id = raw_request.headers.get(
        "x-dynamo-session-id"
    ) or raw_request.headers.get("x-correlation-id")
    parent_id = raw_request.headers.get("x-dynamo-parent-session-id")
    return session_id, parent_id


# ============================================================================
# Request/Response Logging
# ============================================================================


def _log_request_event(event_type: str, request_id: str, data: Any) -> None:
    """Write a JSONL entry to the request log file (if enabled)."""
    if _request_logger is None:
        return
    entry = {
        "timestamp": time.time(),
        "request_id": request_id,
        "type": event_type,
        "data": data,
    }
    _request_logger.info(json.dumps(entry, default=str))


def _log_request_model(event_type: str, request_id: str, model: Any) -> None:
    """:func:`_log_request_event` for a pydantic model, dumped only if logged.

    The guard in `_log_request_event` is in the callee, so
    ``_log_request_event("request", rid, request.model_dump())`` builds the
    dump before the call can decline it: every message and every tool schema
    serialised on the event loop and thrown away, on every request, with
    request logging off. Measured 20-26 us on an agent-shaped request against
    0.07 us for the guard.

    `_log_sse` directly below already asks the question in this order. Two
    spellings of one rule in one module is what this removes.
    """
    if _request_logger is None:
        return
    _log_request_event(event_type, request_id, model.model_dump())


def _log_sse(chunk: str, request_id: str) -> None:
    """Log every SSE frame in `chunk`, and never fail the stream doing it.

    One yield can carry several frames: `serving_chat` deliberately coalesces
    finish + usage + `[DONE]` into one send, because at a wave boundary many
    requests finalize at once and collapsing three socket writes per request
    to one relieves the event loop. This used to `json.loads` the whole send
    as a single payload, which raises `Extra data:` on exactly that frame --
    out of the generator, so with `--request-log` on, the *last* frame of
    every OpenAI stream never reached the client and no `[DONE]` was sent.

    And frames are not all `data:`-first. Anthropic writes `event: NAME` on
    the line above, so a `startswith("data: ")` test skipped every frame that
    endpoint produces -- silently, which for a log is the worst failure it
    can have.

    A payload that will not parse is logged as text rather than dropped or
    raised: this is the diagnostic path, and it must not be the reason a
    response fails.
    """
    if _request_logger is None:
        return
    for frame in chunk.split("\n\n"):
        payload = None
        for line in frame.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
        if payload is None:
            continue
        if payload == "[DONE]":
            _log_request_event("stream_done", request_id, None)
            continue
        try:
            _log_request_event("stream_chunk", request_id, json.loads(payload))
        except ValueError:
            _log_request_event("stream_chunk_unparsed", request_id, payload)


async def _client_stream(
    gen: AsyncGenerator[str, None], request_id: str
) -> AsyncGenerator[str, None]:
    """Every SSE frame on its way to the client: logged, and timed.

    The last point a frame passes through before uvicorn writes it, which is
    why the silence watchdog lives here rather than at the collector it
    started at. Between the collector and this line sit the reasoning
    channel's read-ahead and the tool-call format's, and while either
    withholds, the collector keeps waking on schedule -- so the gauge read
    zero for exactly the stall it exists to report.

    Wrapping every streaming response and not two of the three: this was
    `_logged_stream`, and the Anthropic endpoint never used it. A watchdog
    with an endpoint-shaped hole in it is worse than none, because the zero it
    reports looks like an answer.
    """
    it = gen.__aiter__()
    delivered = False
    while True:
        # The first wait is the queue, not silence: every generator awaits the
        # collector before its opening frame.
        with FrameWait(request_id, armed=delivered):
            try:
                chunk = await it.__anext__()
            except StopAsyncIteration:
                return
        delivered = True
        _log_sse(chunk, request_id)
        yield chunk


# ============================================================================
# Engine Interface
# ============================================================================


def _build_sampling_params(
    temperature: float,
    max_tokens: int,
    stop_strings: list[str] | None,
    ignore_eos: bool,
    top_k: int = -1,
    top_p: float = 1.0,
    n: int = 1,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_strings=stop_strings,
        ignore_eos=ignore_eos,
        n=n,
    )


def _coerce_n(requested_n: int | None, temperature: float | None) -> int:
    """Return an effective ``n`` for a request.

    * ``None``/``<1`` coerce to ``1`` (matches OpenAI default).
    * ``n > 1`` combined with greedy sampling (``temperature <= 0``) is
      collapsed to ``1`` because all siblings would produce identical
      outputs — other runtimes (vLLM, TGI) silently do the same, and it
      avoids wasting KV cache on duplicate decodes.
    """
    n = requested_n if requested_n is not None else 1
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1
    n = max(n, 1)
    if n > 1 and (temperature is None or temperature <= 0.0):
        logger.info(
            "n=%s requested with temperature=%s; collapsing to n=1 because "
            "greedy sampling would produce identical siblings.",
            n,
            temperature,
        )
        n = 1
    return n


def _validate_context_length(
    num_prompt_tokens: int,
    max_tokens: int,
    max_model_len: int | None,
) -> None:
    if max_model_len is None:
        return

    requested_output_tokens = max(0, int(max_tokens or 0))
    total_tokens = int(num_prompt_tokens) + requested_output_tokens
    if total_tokens <= int(max_model_len):
        return

    raise ValueError(
        f"This model's maximum context length is {max_model_len} tokens. "
        f"However, you requested {requested_output_tokens} output tokens and "
        f"your prompt contains at least {num_prompt_tokens} input tokens, for "
        f"a total of at least {total_tokens} tokens. Please reduce the length "
        f"of the input prompt or the number of requested output tokens."
    )


def _get_engine_config():
    config = getattr(engine, "config", None)
    if config is None:
        config = getattr(getattr(engine, "io_processor", None), "config", None)
    return config


def _get_engine_max_model_len() -> int | None:
    return getattr(_get_engine_config(), "max_model_len", None)


def _get_engine_max_pool_tokens() -> int | None:
    """Longest prompt the KV pool can hold, as each engine rank reported it.

    None before the engine is up, or on a manager that never learned it, in
    which case the scheduler remains the only enforcer.
    """
    return getattr(getattr(engine, "core_mgr", None), "max_pool_tokens", None)


def _validate_pool_capacity(
    num_prompt_tokens: int, max_pool_tokens: int | None
) -> None:
    """Refuse a prompt whose KV cannot fit even a completely empty pool.

    `max_model_len` is a declared limit; this is a physical one, since the pool
    is sized from whatever device memory is free once the weights are loaded, so
    on a tight pool it binds first. The scheduler checks it too, but only once
    the request reaches the engine — by then the client holds a response it will
    never be answered on, so the request has to be turned away here instead.
    """
    if max_pool_tokens is None or int(num_prompt_tokens) <= int(max_pool_tokens):
        return

    raise ValueError(
        f"This server's KV cache holds at most {max_pool_tokens} tokens for a "
        f"single request, and your prompt contains at least {num_prompt_tokens} "
        f"input tokens. Please shorten the prompt, or restart the server with a "
        f"higher --gpu-memory-utilization to enlarge the cache."
    )


def _validate_sequence_context_length(seq) -> None:
    _validate_context_length(
        seq.num_prompt_tokens,
        seq.max_tokens,
        _get_engine_max_model_len(),
    )
    _validate_pool_capacity(seq.num_prompt_tokens, _get_engine_max_pool_tokens())


def _has_multimodal_content(messages: list[Any]) -> bool:
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                return True
    return False


def _load_image_from_url(url: str) -> "Image.Image":
    # Imported here, not at module scope, and this is the one place in the
    # file that needs it at runtime. Pillow is not a declared dependency, so a
    # module-scope `from PIL import Image` made the whole server module
    # unimportable wherever it is absent -- which is the non-GPU CI runner,
    # where the only test that reached this module had to wrap its import in a
    # try/except and degrade to `api_server = None`. Text-only serving does
    # not need Pillow, so it should not be a condition of importing the
    # server; a request that actually carries an image raises here, naming it.
    from PIL import Image

    if url.startswith("data:"):
        try:
            _, encoded = url.split(",", 1)
            image_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 data URL for image_url") from exc
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=30) as response:
            image_bytes = response.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    url = url.removeprefix("file://")
    return Image.open(url).convert("RGB")


def _get_multimodal_processor():
    global processor, model_name
    if processor is None:
        logger.info(f"Loading multimodal processor from {model_name}...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return processor


def _collect_multimodal_parts(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], list["Image.Image"]]:
    """Normalize chat messages into processor form, loading every image.

    Content parts keep the order the client sent them in; the images are
    returned separately in that same order.
    """
    processor_messages: list[dict[str, Any]] = []
    images: list[Image.Image] = []

    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, str) or content is None:
            processor_messages.append({"role": message.role, "content": content or ""})
            continue

        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if not url:
                    raise ValueError(
                        "image_url content part must include image_url.url"
                    )
                image = _load_image_from_url(url)
                images.append(image)
                parts.append({"type": "image", "image": image})
            elif part_type == "image":
                url = part.get("image")
                if not isinstance(url, str):
                    raise ValueError(
                        "image content part must include an image URL/path"
                    )
                image = _load_image_from_url(url)
                images.append(image)
                parts.append({"type": "image", "image": image})
        processor_messages.append({"role": message.role, "content": parts})

    return processor_messages, images


def _images_before_text(
    processor_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hoist image parts ahead of the text within each message.

    Qwen3.5's template only reliably emits <|image_pad|> when image entries
    precede the text, matching the native offline multimodal example.
    """
    reordered: list[dict[str, Any]] = []
    for message in processor_messages:
        content = message["content"]
        if not isinstance(content, list):
            reordered.append(message)
            continue
        parts = [part for part in content if part["type"] == "image"]
        texts = [part["text"] for part in content if part["type"] == "text"]
        if texts:
            parts.append({"type": "text", "text": "\n".join(texts)})
        reordered.append({"role": message["role"], "content": parts})
    return reordered


def _prepare_multimodal_inputs(
    messages: list[Any],
    chat_template_kwargs: dict[str, Any],
    tools: Any = None,
) -> tuple[list[int], dict[str, Any]]:
    mm_processor = _get_multimodal_processor()
    processor_messages, images = _collect_multimodal_parts(messages)

    if not images:
        raise ValueError("Multimodal request did not contain any images")

    # Models whose processor deviates from the Qwen convention register their
    # own builder (e.g. Kimi-K3's messages+medias API and unexpanded
    # <|media_pad|> placeholders).
    built = build_multimodal_inputs(
        _get_engine_config(),
        mm_processor,
        processor_messages,
        images,
        chat_template_kwargs,
        tools=tools,
    )
    if built is not None:
        return built

    template_kwargs = dict(chat_template_kwargs)
    template_kwargs.pop("tokenize", None)
    template_kwargs.pop("add_generation_prompt", None)
    text = mm_processor.apply_chat_template(
        _images_before_text(processor_messages),
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if images and "<|image_pad|>" not in text:
        raise ValueError("Multimodal chat template did not emit image placeholders")
    inputs = mm_processor(text=[text], images=images, return_tensors="pt")
    multimodal_data = {
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
    }
    return inputs["input_ids"][0].tolist(), multimodal_data


# ── Batched stream dispatch ──────────────────────────────────────────────


def _build_stream_chunk(request_output: RequestOutput, request_id: str) -> dict:
    """Build a raw chunk; detokenization happens once in the batch dispatcher."""
    started_at = _request_start_times.get(request_id)
    chunk_data = {
        "token_ids": request_output.output_tokens,
        "finished": request_output.finished,
        "finish_reason": request_output.finish_reason,
        "finished_at": time.time(),
        "started_at": started_at,
        "num_cached_tokens": getattr(request_output, "num_cached_tokens", 0),
    }
    if getattr(request_output, "kv_transfer_params_output", None):
        chunk_data["kv_transfer_params"] = request_output.kv_transfer_params_output
    return chunk_data


def _send_stream_chunk_direct(
    request_output: RequestOutput,
    request_id: str,
    stream_collector: StreamOutputCollector,
    loop: AbstractEventLoop,
    state: Any,
) -> None:
    """Buffer a single-request chunk for this engine step."""
    assert _stream_batch_dispatcher is not None
    _stream_batch_dispatcher.enqueue(
        loop=loop,
        collector=stream_collector,
        state=state,
        chunk=_build_stream_chunk(request_output, request_id),
    )


def flush_stream_batch() -> None:
    """Flush this output thread's engine-step batch to the stream collectors."""
    if _stream_batch_dispatcher is not None:
        _stream_batch_dispatcher.flush()


def _send_stream_chunk_tagged(
    request_output: RequestOutput,
    request_id: str,
    sibling_index: int,
    stream_collector: StreamOutputCollector,
    loop: AbstractEventLoop,
    state: Any,
) -> None:
    """Variant of :func:`_send_stream_chunk_direct` for fan-out siblings.

    Pushes ``(sibling_index, chunk_data)`` tuples into a single shared
    collector so the merge-stream consumer in :mod:`serving_chat` /
    :mod:`serving_completion` can demultiplex by index. The collector folds
    per tag, so a lagging consumer never mixes two siblings' deltas.

    This path serves ``SamplingParams.n > 1`` by tagging each sibling's chunks
    so the shared stream consumer can merge them in order.
    """
    assert _stream_batch_dispatcher is not None
    _stream_batch_dispatcher.enqueue(
        loop=loop,
        collector=stream_collector,
        state=state,
        chunk=_build_stream_chunk(request_output, request_id),
        tag=sibling_index,
    )


async def generate_async(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: dict[str, Any] | None = None,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Generate text asynchronously for non-streaming requests."""
    token_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    started_at = time.time()
    first_token_at: float | None = None
    last_token_at: float | None = None
    # An array, not a list: this grows for the whole life of the request,
    # and one boxed PyInt per token is what the collector then walks on
    # every pass. It stays an array all the way out -- the dict below is
    # an internal hand-off to `build_*_response`, which reads `text` and
    # the counters and never `token_ids`, so nothing serializes it. A
    # consumer that starts reading that key has to convert.
    all_token_ids = new_token_ids()
    finish_reason: str | None = None
    seq = None
    kv_transfer_output_meta_info = None
    num_cached_tokens_seen = 0

    def completion_callback(request_output: RequestOutput):
        nonlocal kv_transfer_output_meta_info, num_cached_tokens_seen
        kv_transfer_output_meta_info = getattr(
            request_output, "kv_transfer_params_output", None
        )
        _ct = getattr(request_output, "num_cached_tokens", 0)
        if _ct:
            num_cached_tokens_seen = _ct
        now = time.time()
        loop.call_soon_threadsafe(
            token_queue.put_nowait,
            {
                "token_ids": request_output.output_tokens,
                "finished": request_output.finished,
                "finish_reason": request_output.finish_reason,
                "ts": now,
            },
        )

    def do_preprocess():
        return engine.io_processor.preprocess(
            prompt,
            sampling_params,
            stream_callback=completion_callback,
            kv_transfer_params=kv_transfer_params,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )

    seq = await loop.run_in_executor(None, do_preprocess)
    try:
        _validate_sequence_context_length(seq)
    except Exception:
        engine.io_processor.requests.pop(seq.id, None)
        raise
    engine.core_mgr.add_request([seq])

    _finished_ok = False
    try:
        while True:
            item = await token_queue.get()
            token_ids = item.get("token_ids") or []
            if token_ids:
                if first_token_at is None:
                    first_token_at = item.get("ts", time.time())
                last_token_at = item.get("ts", time.time())
                all_token_ids.extend(token_ids)
            if item.get("finished", False):
                finish_reason = item.get("finish_reason")
                _finished_ok = True
                break
    finally:
        # Two responsibilities, on EVERY exit path:
        #   1) If we didn't finish (client disconnected / cancelled), tell the
        #      engine to stop so the seq doesn't run to max_tokens and burn GPU.
        #   2) Always drop the seq from io_processor.requests. The engine frees
        #      its own KV on finish, but this dict is only cleaned up here for
        #      non-stream requests -- without an unconditional pop, every
        #      completed non-stream request leaks a Sequence (pending grows
        #      forever). Streaming pops via cleanup_stream instead.
        if seq is not None:
            if not _finished_ok:
                with contextlib.suppress(Exception):
                    engine.core_mgr.abort_request(seq.id)
            engine.io_processor.requests.pop(seq.id, None)

    text = delivered_text(all_token_ids)
    num_tokens_input = (
        seq.num_prompt_tokens if seq is not None else len(tokenizer.encode(prompt))
    )
    num_tokens_output = len(all_token_ids)
    finished_at = time.time()
    latency = finished_at - started_at
    ttft = (first_token_at - started_at) if first_token_at is not None else 0.0
    tpot = (
        (last_token_at - first_token_at) / (num_tokens_output - 1)
        if first_token_at is not None
        and last_token_at is not None
        and num_tokens_output > 1
        else 0.0
    )

    response = {
        "text": text,
        "token_ids": all_token_ids,
        "finish_reason": finish_reason,
        "num_tokens_input": num_tokens_input,
        "num_tokens_output": num_tokens_output,
        "ttft": ttft,
        "tpot": tpot,
        "latency": latency,
        "num_cached_tokens": num_cached_tokens_seen,
    }
    if kv_transfer_output_meta_info is not None:
        response["kv_transfer_output_meta_info"] = kv_transfer_output_meta_info
    yield response


async def generate_async_multimodal(
    token_ids: list[int],
    multimodal_data: dict[str, Any],
    sampling_params: SamplingParams,
    request_id: str,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Generate text asynchronously for one multimodal request."""
    token_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    started_at = time.time()
    first_token_at: float | None = None
    last_token_at: float | None = None
    all_token_ids = new_token_ids()
    finish_reason: str | None = None
    seq = None

    def completion_callback(request_output: RequestOutput):
        now = time.time()
        loop.call_soon_threadsafe(
            token_queue.put_nowait,
            {
                "token_ids": request_output.output_tokens,
                "finished": request_output.finished,
                "finish_reason": request_output.finish_reason,
                "ts": now,
            },
        )

    def do_preprocess():
        return engine.io_processor.preprocess(
            token_ids,
            sampling_params,
            stream_callback=completion_callback,
            multimodal_data=multimodal_data,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )

    seq = await loop.run_in_executor(None, do_preprocess)
    try:
        _validate_sequence_context_length(seq)
    except Exception:
        engine.io_processor.requests.pop(seq.id, None)
        raise
    engine.core_mgr.add_request([seq])

    _finished_ok = False
    try:
        while True:
            item = await token_queue.get()
            token_ids_out = item.get("token_ids") or []
            if token_ids_out:
                if first_token_at is None:
                    first_token_at = item.get("ts", time.time())
                last_token_at = item.get("ts", time.time())
                all_token_ids.extend(token_ids_out)
            if item.get("finished", False):
                finish_reason = item.get("finish_reason")
                _finished_ok = True
                break
    finally:
        # See generate_async: abort on early exit, always pop to avoid leak.
        if seq is not None:
            if not _finished_ok:
                with contextlib.suppress(Exception):
                    engine.core_mgr.abort_request(seq.id)
            engine.io_processor.requests.pop(seq.id, None)

    text = delivered_text(all_token_ids)
    num_tokens_output = len(all_token_ids)
    finished_at = time.time()
    ttft = (first_token_at - started_at) if first_token_at is not None else 0.0
    tpot = (
        (last_token_at - first_token_at) / (num_tokens_output - 1)
        if first_token_at is not None
        and last_token_at is not None
        and num_tokens_output > 1
        else 0.0
    )

    yield {
        "text": text,
        "token_ids": all_token_ids,
        "finish_reason": finish_reason,
        "num_tokens_input": (
            seq.num_prompt_tokens if seq is not None else len(token_ids)
        ),
        "num_tokens_output": num_tokens_output,
        "ttft": ttft,
        "tpot": tpot,
        "latency": finished_at - started_at,
    }


async def generate_async_fanout(
    prompt_or_tokens: str | list[int],
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: dict[str, Any] | None = None,
    multimodal_data: dict[str, Any] | None = None,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Non-streaming n>1 path: fan out N siblings and await all of them.

    Returns a list of per-sibling output dicts in the same shape as
    :func:`generate_async` yields for n==1, so response builders can treat
    each entry the same way.
    """
    global engine, tokenizer

    n = int(sampling_params.n)
    assert n >= 1

    shared_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    started_at = time.time()
    per_tokens = [new_token_ids() for _ in range(n)]
    per_first_token_at: list[float | None] = [None] * n
    per_last_token_at: list[float | None] = [None] * n
    per_finish_reason: list[str | None] = [None] * n
    finished = [False] * n

    def make_callback(idx: int):
        def _cb(request_output: RequestOutput) -> None:
            now = time.time()
            loop.call_soon_threadsafe(
                shared_queue.put_nowait,
                (
                    idx,
                    {
                        "token_ids": request_output.output_tokens,
                        "finished": request_output.finished,
                        "finish_reason": request_output.finish_reason,
                        "ts": now,
                    },
                ),
            )

        return _cb

    stream_callbacks = [make_callback(i) for i in range(n)]

    def do_preprocess():
        return engine.io_processor.preprocess_fanout(
            prompt_or_tokens,
            sampling_params,
            stream_callbacks=stream_callbacks,
            kv_transfer_params=kv_transfer_params,
            multimodal_data=multimodal_data,
            parent_request_id=request_id,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )

    seqs = await loop.run_in_executor(None, do_preprocess)
    try:
        _validate_sequence_context_length(seqs[0])
    except Exception:
        for seq in seqs:
            engine.io_processor.requests.pop(seq.id, None)
        raise
    engine.core_mgr.add_request(seqs)
    num_tokens_input = seqs[0].num_prompt_tokens

    _all_finished = False
    try:
        while not all(finished):
            idx, item = await shared_queue.get()
            if finished[idx]:
                continue
            tokens = item.get("token_ids") or []
            if tokens:
                if per_first_token_at[idx] is None:
                    per_first_token_at[idx] = item.get("ts", time.time())
                per_last_token_at[idx] = item.get("ts", time.time())
                per_tokens[idx].extend(tokens)
            if item.get("finished", False):
                per_finish_reason[idx] = item.get("finish_reason")
                finished[idx] = True
        _all_finished = True
    finally:
        # Abort any sibling still running on early exit; always pop all seqs.
        for _seq in seqs:
            if not _all_finished:
                try:
                    engine.core_mgr.abort_request(_seq.id)
                except Exception:
                    pass
            engine.io_processor.requests.pop(_seq.id, None)

    finished_at = time.time()
    outputs: list[dict[str, Any]] = []
    for i in range(n):
        num_tokens_output = len(per_tokens[i])
        ttft = (
            per_first_token_at[i] - started_at
            if per_first_token_at[i] is not None
            else 0.0
        )
        tpot = (
            (per_last_token_at[i] - per_first_token_at[i]) / (num_tokens_output - 1)
            if per_first_token_at[i] is not None
            and per_last_token_at[i] is not None
            and num_tokens_output > 1
            else 0.0
        )
        outputs.append(
            {
                "text": delivered_text(per_tokens[i]),
                "token_ids": per_tokens[i],
                "finish_reason": per_finish_reason[i],
                "num_tokens_input": num_tokens_input,
                "num_tokens_output": num_tokens_output,
                "ttft": ttft,
                "tpot": tpot,
                "latency": finished_at - started_at,
            }
        )
    return outputs


def validate_model(requested_model: str | None) -> None:
    """Validate that the requested model matches the server's model."""
    if requested_model is None:
        return

    normalized_requested = requested_model.rstrip("/")
    normalized_served = model_name.rstrip("/")
    if normalized_requested != normalized_served:
        raise HTTPException(
            status_code=400,
            detail=f"Requested model '{requested_model}' does not match "
            f"server model '{model_name}'",
        )


async def setup_streaming_request(
    prompt_or_tokens: str | list[int],
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: dict[str, Any] | None = None,
    multimodal_data: dict[str, Any] | None = None,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
) -> tuple[int, StreamOutputCollector, int]:
    """Set up a streaming request with the engine.

    Returns ``(seq_id, stream_collector, num_prompt_tokens)``.
    ``num_prompt_tokens`` is the engine-computed prompt length so the stream
    response generator does not have to re-tokenize the prompt on the event
    loop.
    """
    stream_collector = StreamOutputCollector(request_id)
    stream_loop = asyncio.get_running_loop()
    _stream_loops[request_id] = stream_loop
    _request_start_times[request_id] = time.time()

    # The detokenizer lives in this closure, so it is freed when the engine
    # drops the callback on the stream's last chunk -- no registry, no cleanup.
    assert _stream_batch_dispatcher is not None
    detokenizer = _stream_batch_dispatcher.new_state()

    def stream_callback(request_output: RequestOutput) -> None:
        _send_stream_chunk_direct(
            request_output, request_id, stream_collector, stream_loop, detokenizer
        )

    executor_loop = asyncio.get_event_loop()

    def do_preprocess():
        seq = engine.io_processor.preprocess(
            prompt_or_tokens,
            sampling_params,
            stream_callback=stream_callback,
            kv_transfer_params=kv_transfer_params,
            multimodal_data=multimodal_data,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )
        _seq_id_to_request_id[seq.id] = request_id
        return seq

    seq = None
    try:
        seq = await executor_loop.run_in_executor(None, do_preprocess)
        _validate_sequence_context_length(seq)
    except Exception:
        _stream_loops.pop(request_id, None)
        _request_start_times.pop(request_id, None)
        if seq is not None:
            _seq_id_to_request_id.pop(seq.id, None)
            engine.io_processor.requests.pop(seq.id, None)
        raise
    seq_id = seq.id

    # debug, not info: this runs once per request, on the event loop, and
    # logging takes a lock the engine's output threads are also contending for.
    # A loop-stall watchdog at concurrency 8192 caught 26 stalls over a run and
    # 9 of them were sitting on this line, up to 3.3 s each -- long enough that
    # the server accepts no new request at all and the GPUs run dry waiting for
    # work. Anything per-request logged from here has to stay off info.
    # %-style, not an f-string: the arguments are formatted only if the
    # record is emitted, and this runs once per request with debug off.
    logger.debug("API: Created request_id=%s, seq_id=%s", request_id, seq_id)
    engine.core_mgr.add_request([seq])

    return seq_id, stream_collector, seq.num_prompt_tokens


def cleanup_stream(seq_id: int, aborted: bool = False) -> None:
    """Tear down one stream. A fan-out request runs this once per sibling.

    ``aborted`` says the stream did NOT reach its normal end (client disconnect
    or abnormal generator teardown), so the seq is likely still running in the
    engine and must be stopped. On normal completion pass False (the default):
    the engine has already dropped the seq, so an abort would be a guaranteed
    no-op that just floods the control path (one broadcast per engine core, per
    request).
    """
    _seq_id_to_request_id.pop(seq_id, None)
    if aborted:
        try:
            engine.core_mgr.abort_request(seq_id)
        except Exception:
            pass
    engine.io_processor.requests.pop(seq_id, None)


def cleanup_request(request_id: str) -> None:
    """Tear down what a request owns beyond its individual streams.

    Runs once, after every one of the request's streams has been cleaned up.
    Separate from :func:`cleanup_stream` because a fan-out has n streams but
    one request: folding both into one call meant these two pops ran n times,
    n-1 of them no-ops, and made a caller pass a seq id and a request id
    together when each half only needs one of them.
    """
    _stream_loops.pop(request_id, None)
    _request_start_times.pop(request_id, None)


class _ClientDisconnected(Exception):
    """Raised when a non-streaming client hangs up mid-generation."""

    def __init__(self, request_id: str):
        super().__init__(request_id)
        self.request_id = request_id


async def _listen_for_disconnect(request) -> None:
    """Block until the client sends an ``http.disconnect`` ASGI event.

    Unlike polling ``request.is_disconnected()`` on a timer, this awaits the
    disconnect event directly, so detection is immediate and costs nothing while
    the client stays connected.
    """
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


async def _race_disconnect(coro, raw_request, request_id):
    """Race an awaitable against client disconnect (vLLM ``with_cancellation``
    style).

    Starlette does NOT cancel a *non-streaming* request handler when the client
    goes away (unlike StreamingResponse, which is cancelled on http.disconnect).
    Without this, an abandoned non-stream request keeps ``await``-ing the engine
    until it hits ``max_tokens`` -- burning GPU on output nobody will read AND
    leaking the seq(s) in ``io_processor.requests`` (their finally never fires).

    We run ``coro`` (which produces the final result) as a task alongside a task
    that awaits the ASGI ``http.disconnect`` event. Whichever finishes first
    wins; the loser is cancelled. On disconnect, the coro's cancellation
    propagates into its ``await`` points so its own ``try/finally`` runs ->
    ``abort_request`` + ``io_processor.requests.pop`` (for fan-out, this aborts
    every sibling). We then raise ``_ClientDisconnected``.

    ``request.receive()`` is safe here because FastAPI has already parsed the
    request body into a pydantic model before this handler runs, so there is no
    unread body for ``receive()`` to race against.
    """
    handler_task = asyncio.ensure_future(coro)

    # No ASGI request object (e.g. internal call) -> just await the coro.
    if raw_request is None:
        return await handler_task

    disconnect_task = asyncio.ensure_future(_listen_for_disconnect(raw_request))

    done, pending = await asyncio.wait(
        [handler_task, disconnect_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel the loser and let its cancellation settle (drives the coro's own
    # finally -> abort_request when the handler is the loser). Only swallow the
    # expected CancelledError; log anything else, and let BaseException
    # (KeyboardInterrupt/SystemExit) propagate.
    for task in pending:
        task.cancel()
    for task in pending:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning(
                f"Error tearing down cancelled task for request {request_id}",
                exc_info=True,
            )

    if handler_task in done:
        return handler_task.result()

    logger.info(f"Client disconnected (non-stream), aborting request {request_id}")
    raise _ClientDisconnected(request_id)


async def _run_nonstream_with_disconnect(agen, raw_request, request_id):
    """Drive a non-stream ``generate_async*`` async-*generator* while watching
    for client disconnect.

    Thin wrapper over :func:`_race_disconnect` that collects the generator's
    last yielded output. Use :func:`_race_disconnect` directly for the fan-out
    path, whose ``generate_async_fanout`` is a coroutine returning a list.
    """

    async def _collect():
        final_output = None
        async for output in agen:
            final_output = output
        return final_output

    return await _race_disconnect(_collect(), raw_request, request_id)


async def setup_streaming_request_fanout(
    prompt_or_tokens: str | list[int],
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: dict[str, Any] | None = None,
    multimodal_data: dict[str, Any] | None = None,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
) -> tuple[list[int], StreamOutputCollector, int]:
    """Fan-out variant of :func:`setup_streaming_request`.

    Creates ``sampling_params.n`` sibling sequences sharing one output
    collector. Every callback pushes ``(sibling_index, chunk_data)`` tuples so
    the merge-stream consumer can rewrite ``choices[0].index`` correctly.

    Returns ``(seq_ids, shared_collector, num_prompt_tokens)``. All siblings
    tokenize the same prompt once, so ``num_prompt_tokens`` is shared and lets
    the stream response generator skip re-tokenizing on the event loop.
    """
    n = int(sampling_params.n)
    assert n >= 1

    shared_collector = StreamOutputCollector(request_id)
    stream_loop = asyncio.get_running_loop()
    _stream_loops[request_id] = stream_loop
    _request_start_times[request_id] = time.time()

    assert _stream_batch_dispatcher is not None

    def make_callback(idx: int):
        # One detokenizer per sibling, held by the closure that feeds it.
        detokenizer = _stream_batch_dispatcher.new_state()

        def _cb(request_output: RequestOutput) -> None:
            _send_stream_chunk_tagged(
                request_output,
                request_id,
                idx,
                shared_collector,
                stream_loop,
                detokenizer,
            )

        return _cb

    stream_callbacks = [make_callback(i) for i in range(n)]

    executor_loop = asyncio.get_event_loop()

    def do_preprocess():
        seqs = engine.io_processor.preprocess_fanout(
            prompt_or_tokens,
            sampling_params,
            stream_callbacks=stream_callbacks,
            kv_transfer_params=kv_transfer_params,
            multimodal_data=multimodal_data,
            parent_request_id=request_id,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )
        for seq in seqs:
            _seq_id_to_request_id[seq.id] = request_id
        return seqs

    seqs = []
    try:
        seqs = await executor_loop.run_in_executor(None, do_preprocess)
        _validate_sequence_context_length(seqs[0])
    except Exception:
        _stream_loops.pop(request_id, None)
        _request_start_times.pop(request_id, None)
        for seq in seqs:
            _seq_id_to_request_id.pop(seq.id, None)
            engine.io_processor.requests.pop(seq.id, None)
        raise
    seq_ids = [seq.id for seq in seqs]
    # debug for the same reason as its single-sequence counterpart: per-request
    # logging on the event loop stalls it under load.
    logger.debug(
        f"API: Created fan-out request_id={request_id}, n={n}, seq_ids={seq_ids}"
    )
    engine.core_mgr.add_request(seqs)
    return seq_ids, shared_collector, seqs[0].num_prompt_tokens


# ============================================================================
# FastAPI Application
# ============================================================================


async def _refresh_metrics_once() -> None:
    if engine is None:
        return
    try:
        # A local read of the snapshots EngineCore pushes, so it runs inline on
        # the loop -- no executor thread, and no writer on the control socket.
        snapshot = engine.get_metrics_statistics()
    except asyncio.CancelledError:
        raise
    except Exception:
        _metrics_exporter.record_refresh_error()
        logger.warning("Failed to refresh Prometheus metrics", exc_info=True)
    else:
        _metrics_exporter.update(snapshot)


async def _metrics_refresh_loop() -> None:
    while True:
        await asyncio.sleep(_METRICS_REFRESH_INTERVAL_SECONDS)
        await _refresh_metrics_once()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global _metrics_refresh_task
    logger.info("Server started successfully and ready to accept requests")
    tune_gc()
    maybe_attach_gc_debug_callback("api_server")
    await _refresh_metrics_once()
    _metrics_refresh_task = asyncio.create_task(_metrics_refresh_loop())
    # The engine was built in `main()`, so this is the last point before the
    # first request at which everything reachable is still startup state.
    freeze_gc_heap("api_server")
    try:
        yield
    finally:
        if _metrics_refresh_task is not None:
            _metrics_refresh_task.cancel()
            try:
                await _metrics_refresh_task
            except asyncio.CancelledError:
                pass
            _metrics_refresh_task = None
        logger.info("Server shutting down, releasing resources...")
        if engine is not None:
            engine.close()


app = FastAPI(title="ATOM OpenAI API Server", lifespan=lifespan)


# ---- Error handlers ----


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": 400,
            }
        },
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_server_error",
                "code": 500,
            }
        },
    )


# ---- Endpoints ----


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Handle chat completion requests (OpenAI-compatible)."""
    global engine, tokenizer, model_name

    validate_model(request.model)

    try:
        request.tools = normalize_chat_tools(request.tools)
        validate_chat_request(request)
        messages = request.get_messages()

        merged_kwargs = dict(default_chat_template_kwargs)
        if request.chat_template_kwargs:
            merged_kwargs.update(request.chat_template_kwargs)
        # Forward K3 template controls the chat template needs but that pydantic
        # does not otherwise thread through: structured-output response_format,
        # a string tool_choice ("auto"/"none"/"required"), and thinking/effort.
        if request.response_format is not None:
            merged_kwargs["response_format"] = request.response_format
        if isinstance(request.tool_choice, str):
            merged_kwargs["tool_choice"] = request.tool_choice
        _th_enabled, _th_effort = resolve_thinking(request)
        if request.thinking is not None or request.reasoning_effort is not None:
            # By the name this template actually reads. `thinking` was
            # hardcoded, which is right for Kimi-K3 and a silent no-op for the
            # whole Qwen family, whose templates read `enable_thinking` --
            # measured, `thinking=False` left the `<think>` prefill in place.
            # A template ignores a kwarg it does not know, so the failure was
            # invisible: the model reasoned anyway.
            # Only when the request said something about it. An effort is
            # not an opt-in, and this is merged after the server defaults and
            # after the client's own `chat_template_kwargs` -- so writing it
            # unconditionally overrode both.
            if reasoning_toggle is not None and _th_enabled is not None:
                name, off_value, on_value = reasoning_toggle
                merged_kwargs[name] = on_value if _th_enabled else off_value
            if _th_effort is not None:
                merged_kwargs["thinking_effort"] = _th_effort

        effective_n = _coerce_n(request.n, request.temperature)
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.get_max_tokens(),
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
            n=effective_n,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        dp_session_id, dp_parent_session_id = _get_dp_session_affinity_ids(raw_request)
        dp_routing = {
            "data_parallel_rank": request.data_parallel_rank,
            "dp_session_id": dp_session_id,
            "dp_parent_session_id": dp_parent_session_id,
        }

        _log_request_model("request", request_id, request)

        is_multimodal = _has_multimodal_content(messages)
        if is_multimodal:
            # Image loading (blocking network I/O, up to a 30s urlopen) plus
            # processor preprocessing are heavy and would stall the event loop;
            # run them in a worker thread. Warm the processor on the loop first
            # so concurrent cold-start requests don't race on its lazy init.
            _get_multimodal_processor()
            loop = asyncio.get_running_loop()
            token_ids, multimodal_data = await loop.run_in_executor(
                None,
                _prepare_multimodal_inputs,
                messages,
                merged_kwargs,
                request.tools,
            )
        else:
            prompt = apply_chat_template(
                tokenizer,
                custom_message_encoder,
                [msg.to_template_dict() for msg in messages],
                tools=request.tools,
                **merged_kwargs,
            )

        # The K3 template may inject the opening reasoning marker into the prompt
        # itself; if so the stream begins mid-thought and the ReasoningFilter must
        # start in the thinking state. Multimodal inputs arrive pre-tokenized.
        _reasoning = reasoning_channel(
            (
                prompt_tokens_start_in_reasoning(token_ids, tokenizer.decode)
                if is_multimodal
                else prompt_starts_in_reasoning(prompt)
            ),
            template_kwargs=merged_kwargs,
        )

        # Streaming
        if request.stream:
            stream_input = token_ids if is_multimodal else prompt
            stream_multimodal_data = multimodal_data if is_multimodal else None
            if effective_n > 1:
                seq_ids, stream_collector, num_prompt_tokens = (
                    await setup_streaming_request_fanout(
                        stream_input,
                        sampling_params,
                        request_id,
                        multimodal_data=stream_multimodal_data,
                        kv_transfer_params=request.kv_transfer_params,
                        **dp_routing,
                    )
                )
                gen = stream_chat_response_fanout(
                    request_id,
                    model_name,
                    stream_collector,
                    seq_ids,
                    num_prompt_tokens,
                    cleanup_stream,
                    cleanup_request,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    reasoning=_reasoning,
                    tool_parser_cls=tool_call_parser_cls,
                )
            else:
                seq_id, stream_collector, num_prompt_tokens = (
                    await setup_streaming_request(
                        stream_input,
                        sampling_params,
                        request_id,
                        multimodal_data=stream_multimodal_data,
                        kv_transfer_params=request.kv_transfer_params,
                        **dp_routing,
                    )
                )
                gen = stream_chat_response(
                    request_id,
                    model_name,
                    stream_collector,
                    seq_id,
                    num_prompt_tokens,
                    cleanup_stream,
                    cleanup_request,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    reasoning=_reasoning,
                    tool_parser_cls=tool_call_parser_cls,
                )
            return StreamingResponse(
                _client_stream(gen, request_id),
                media_type="text/event-stream",
            )

        # Non-streaming
        if is_multimodal and effective_n > 1:
            outputs = await _race_disconnect(
                generate_async_fanout(
                    token_ids,
                    sampling_params,
                    request_id,
                    multimodal_data=multimodal_data,
                    kv_transfer_params=request.kv_transfer_params,
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if not outputs:
                raise RuntimeError("No output generated")
            resp = build_chat_response_multi(
                request_id,
                model_name,
                outputs,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=_reasoning,
                tool_parser_cls=tool_call_parser_cls,
            )
        elif is_multimodal:
            final_output = await _run_nonstream_with_disconnect(
                generate_async_multimodal(
                    token_ids,
                    multimodal_data,
                    sampling_params,
                    request_id,
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if final_output is None:
                raise RuntimeError("No output generated")
            resp = build_chat_response(
                request_id,
                model_name,
                final_output["text"],
                final_output,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=_reasoning,
                tool_parser_cls=tool_call_parser_cls,
            )
        elif effective_n > 1:
            outputs = await _race_disconnect(
                generate_async_fanout(
                    prompt,
                    sampling_params,
                    request_id,
                    kv_transfer_params=request.kv_transfer_params,
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if not outputs:
                raise RuntimeError("No output generated")
            resp = build_chat_response_multi(
                request_id,
                model_name,
                outputs,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=_reasoning,
                tool_parser_cls=tool_call_parser_cls,
            )
        else:
            final_output = await _run_nonstream_with_disconnect(
                generate_async(
                    prompt,
                    sampling_params,
                    request_id,
                    kv_transfer_params=request.kv_transfer_params,
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if final_output is None:
                raise RuntimeError("No output generated")
            resp = build_chat_response(
                request_id,
                model_name,
                final_output["text"],
                final_output,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=_reasoning,
                tool_parser_cls=tool_call_parser_cls,
            )
        _log_request_model("response", request_id, resp)
        return resp

    except _ClientDisconnected:
        # Client hung up; seq already aborted + popped. Nothing to return.
        return JSONResponse(status_code=499, content={"detail": "client disconnected"})
    except ValueError as e:
        logger.error(f"Validation error in chat_completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest, raw_request: Request):
    """Handle text completion requests (OpenAI-compatible)."""
    global engine, tokenizer, model_name

    validate_model(request.model)

    try:
        effective_n = _coerce_n(request.n, request.temperature)
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.get_max_tokens(),
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
            n=effective_n,
        )

        request_id = f"cmpl-{uuid.uuid4().hex}"
        dp_session_id, dp_parent_session_id = _get_dp_session_affinity_ids(raw_request)
        dp_routing = {
            "data_parallel_rank": request.data_parallel_rank,
            "dp_session_id": dp_session_id,
            "dp_parent_session_id": dp_parent_session_id,
        }

        _log_request_model("request", request_id, request)

        # Streaming
        if request.stream:
            if effective_n > 1:
                seq_ids, stream_collector, num_prompt_tokens = (
                    await setup_streaming_request_fanout(
                        request.prompt,
                        sampling_params,
                        request_id,
                        kv_transfer_params=request.kv_transfer_params,
                        **dp_routing,
                    )
                )
                gen = stream_completion_response_fanout(
                    request_id,
                    model_name,
                    stream_collector,
                    seq_ids,
                    num_prompt_tokens,
                    cleanup_stream,
                    cleanup_request,
                )
            else:
                seq_id, stream_collector, num_prompt_tokens = (
                    await setup_streaming_request(
                        request.prompt,
                        sampling_params,
                        request_id,
                        kv_transfer_params=request.kv_transfer_params,
                        **dp_routing,
                    )
                )
                gen = stream_completion_response(
                    request_id,
                    model_name,
                    stream_collector,
                    seq_id,
                    num_prompt_tokens,
                    cleanup_stream,
                    cleanup_request,
                )
            return StreamingResponse(
                _client_stream(gen, request_id),
                media_type="text/event-stream",
            )

        # Non-streaming
        if effective_n > 1:
            outputs = await _race_disconnect(
                generate_async_fanout(
                    request.prompt,
                    sampling_params,
                    request_id,
                    kv_transfer_params=request.kv_transfer_params,
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if not outputs:
                raise RuntimeError("No output generated")
            resp = build_completion_response_multi(request_id, model_name, outputs)
        else:
            final_output = await _run_nonstream_with_disconnect(
                generate_async(
                    request.prompt,
                    sampling_params,
                    request_id,
                    kv_transfer_params=request.kv_transfer_params,
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )

            if final_output is None:
                raise RuntimeError("No output generated")

            resp = build_completion_response(request_id, model_name, final_output)
        _log_request_model("response", request_id, resp)
        return resp

    except _ClientDisconnected:
        # Client hung up; seq already aborted + popped. Nothing to return.
        return JSONResponse(status_code=499, content={"detail": "client disconnected"})
    except ValueError as e:
        logger.error(f"Validation error in completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages")
async def anthropic_messages(request: AnthropicMessagesRequest, raw_request: Request):
    """Handle Anthropic Messages API requests.

    Translates Anthropic format to OpenAI format internally, runs inference,
    and returns Anthropic-formatted responses. Enables Claude Code and other
    Anthropic-compatible tools to use ATOM as a backend.
    """
    # One validator over the shape both endpoints share: this path already
    # converts, so validating the conversion leaves a single rule rather than
    # a second one in Anthropic's spelling. It checked nothing before, so a
    # name `/v1/chat/completions` rejects was accepted here. Explicitly 400
    # and before the try, because the handler below turns every exception into
    # a 500 -- wrong for a malformed request, and once the response is
    # streaming it arrives after the client was told the request succeeded.
    try:
        validate_tool_list(anthropic_to_openai_tools(request.tools))
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )

    try:
        # Convert Anthropic messages to OpenAI format
        openai_messages = anthropic_to_openai_messages(request.messages, request.system)

        # Apply chat template
        from .protocol import ChatMessage

        messages = [ChatMessage(**m) for m in openai_messages]

        merged_kwargs = dict(default_chat_template_kwargs)
        # The request's `thinking` was dropped on the floor here, so the model
        # was never told not to think and the endpoint spent three attempts
        # dealing with a chain of thought it had asked for by omission.
        merged_kwargs.update(anthropic_template_kwargs(request, reasoning_toggle))
        drop_reasoning = anthropic_drop_reasoning(request)
        # Same answer-it-in-the-prompt rule as `thinking` above, and the chat
        # path already forwards its own spelling of this. `tool_choice` was
        # read off the Anthropic request and then used nowhere at all -- a
        # client that forbade tool calls got `tool_use` blocks and
        # `stop_reason: tool_use`. Translated to the string a chat template
        # expects; the object forms Anthropic has for *requiring* a call have
        # no such spelling, and are not answered here or on the chat path.
        if forbids_tool_calls(request.tool_choice):
            merged_kwargs["tool_choice"] = "none"
        prompt = apply_chat_template(
            tokenizer,
            custom_message_encoder,
            [msg.to_template_dict() for msg in messages],
            tools=anthropic_to_openai_tools(request.tools),
            **merged_kwargs,
        )

        generation_config = engine.config.generation_config
        model_temperature = getattr(generation_config, "temperature", None)
        model_top_p = getattr(generation_config, "top_p", None)
        model_top_k = getattr(generation_config, "top_k", None)
        if model_temperature is None:
            model_temperature = DEFAULT_TEMPERATURE
        if model_top_p is None:
            model_top_p = DEFAULT_TOP_P
        if model_top_k is None:
            model_top_k = DEFAULT_TOP_K

        sampling_params = _build_sampling_params(
            temperature=(
                request.temperature
                if request.temperature is not None
                else model_temperature
            ),
            max_tokens=request.max_tokens,
            stop_strings=request.stop_sequences,
            ignore_eos=False,
            top_k=(request.top_k if request.top_k is not None else model_top_k),
            top_p=(request.top_p if request.top_p is not None else model_top_p),
        )

        request_id = uuid.uuid4().hex[:24]
        input_tokens = len(tokenizer.encode(prompt))

        max_ctx = None
        for _path in (
            lambda: engine.config.max_model_len,
            lambda: engine.model_config.max_model_len,
            lambda: engine.scheduler.max_model_len,
            lambda: engine.max_model_len,
        ):
            try:
                _v = _path()
                if _v:
                    max_ctx = int(_v)
                    break
            except Exception:
                continue
        if not max_ctx:
            max_ctx = 30720
        logger.warning(f"[anthropic] resolved max_ctx={max_ctx}")
        headroom = min(request.max_tokens, max(1024, max_ctx // 8))
        max_input = max_ctx - headroom
        if input_tokens > max_input:
            logger.warning(
                f"Prompt too long ({input_tokens} > {max_input}), truncating"
            )
            token_ids = tokenizer.encode(prompt)[:max_input]
            prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
            input_tokens = max_input

        if request.stream:
            # Streaming response
            seq_id, stream_collector, _num_prompt_tokens = (
                await setup_streaming_request(prompt, sampling_params, request_id)
            )

            async def generate_anthropic_stream():
                # Unconditional, like the chat path and like both upstreams:
                # whatever reasoning arrives is separated and reported. The
                # request's `thinking` was answered in the prompt, so there is
                # nothing left here for it to decide -- and separating always
                # is what keeps the reasoning out of the tool parser, which is
                # a second reader of this same text.
                #
                # Asked of every dialect, not of one literal: the K3 template
                # opens with `<|open|>think<|sep|>`, which `.endswith("<think>")`
                # does not see, and the assignment it guarded skipped
                # `__post_init__` so the instance was in the thinking state
                # while claiming it did not start there. Which dialect closes
                # it is the model's, resolved at startup -- the filter used to
                # carry none and closed on any registered dialect's marker.
                reasoning_filter = reasoning_channel(
                    prompt_starts_in_reasoning(prompt),
                    template_kwargs=merged_kwargs,
                ).stream()
                tool_parser = ToolCallStreamParser(
                    parser_cls=tool_call_parser_cls,
                    suppress_calls=forbids_tool_calls(request.tool_choice),
                )
                tool_parser.tools = anthropic_to_openai_tools(request.tools)
                blocks = AnthropicBlocks()
                has_tool_calls = False
                output_tokens = 0
                # Overwritten by a tool call, and otherwise by whatever the
                # engine says. It used to be the constant `end_turn`, so a
                # response cut off at `max_tokens` claimed a normal ending --
                # and a reasoning model asked for no thinking, which produces
                # only reasoning and has all of it dropped, delivered an empty
                # message that also said nothing was wrong. The vocabularies
                # already line up; they were simply never connected.
                # Computed once, at the end, from the two facts that decide
                # it -- exactly as `serving_chat` does. Recomputing it at each
                # send point meant whichever fact arrived last won: a call
                # completing mid-stream froze it at `tool_use`, and the
                # `max_tokens` that arrived three chunks later could never be
                # folded in. Kimi-K2 is the one registered format whose region
                # closes mid-stream, so it is the one that reaches this.
                engine_reason: Any = None

                message_started = False
                # See `_ANTHROPIC_PING_INTERVAL_SECONDS`.
                last_ping = 0.0
                # Assume abort until we reach the normal end of the stream. If
                # the client disconnects, GeneratorExit unwinds through the
                # yields and the finally runs with this still True -> abort.
                aborted = True

                try:
                    while True:
                        chunk_data = await stream_collector.get()
                        if not message_started:
                            cache_read = chunk_data.get("num_cached_tokens", 0)
                            yield stream_message_start(
                                request_id, model_name, input_tokens, cache_read
                            )
                            message_started = True
                        new_text = chunk_data["text"]
                        if chunk_data.get("finish_reason"):
                            engine_reason = chunk_data["finish_reason"]
                        output_tokens += len(chunk_data.get("token_ids", []))
                        finished = chunk_data.get("finished", False)

                        # Phase 1: Reasoning filter. Never None --
                        # `reasoning_channel(...).stream()` always returns a
                        # `ReasoningFilter`. Do not add a skip path back: it
                        # would feed an unseparated chain of thought straight
                        # into the tool parser.
                        segments = reasoning_filter.process(new_text)
                        if finished:
                            segments.extend(reasoning_filter.flush())

                        for field, text in segments:
                            if not text:
                                continue

                            if field == "reasoning_content":
                                if drop_reasoning:
                                    now = time.monotonic()
                                    if now - last_ping >= (
                                        _ANTHROPIC_PING_INTERVAL_SECONDS
                                    ):
                                        last_ping = now
                                        yield _ANTHROPIC_PING_FRAME
                                    continue
                                for _frame in blocks.delta("thinking", text):
                                    yield _frame
                            else:
                                # Phase 2: Tool call detection on content
                                events = tool_parser.process(text)
                                has_tool_calls = has_tool_calls or (
                                    completes_a_tool_call(events)
                                )
                                for _frame in tool_event_frames(events, blocks):
                                    yield _frame

                        if finished:
                            # Flush remaining tool call events
                            events = tool_parser.flush()
                            has_tool_calls = has_tool_calls or (
                                completes_a_tool_call(events)
                            )
                            for _frame in tool_event_frames(events, blocks):
                                yield _frame

                            # A response with no tool call must end on a text
                            # block even when it produced none, because a reply
                            # of pure reasoning still has to carry a `text`
                            # block for clients that read only that.
                            if not has_tool_calls and blocks.kind != "text":
                                for _frame in blocks.open("text"):
                                    yield _frame
                            # Before the last frames, not after. The flag means
                            # "the engine sequence may still be running", and
                            # the engine is done the moment its finished chunk
                            # arrived -- a client hanging up between the two
                            # yields below fired a broadcast abort for a
                            # sequence that had already ended, which is the
                            # control-path flood the flag exists to prevent.
                            # `serving_chat` already sets it here.
                            aborted = False
                            for _frame in blocks.close():
                                yield _frame
                            stop_reason = anthropic_stop_reason_with_calls(
                                engine_reason, has_tool_calls
                            )
                            yield stream_message_delta(stop_reason, output_tokens)
                            yield stream_message_stop()
                            break
                except Exception as exc:
                    # Every block this stream opened has to be closed, and the
                    # client has to be told why. There was no `except` at all:
                    # the endpoint's own handler has already returned by the
                    # time the generator runs, so a raise from the collector,
                    # the reasoning filter or a tool parser cut the response
                    # mid-frame with an open block and no terminator.
                    logger.exception("Error streaming anthropic response")
                    for _frame in stream_failure_frames(
                        exc,
                        blocks,
                        output_tokens,
                        opening=(
                            None
                            if message_started
                            else stream_message_start(
                                request_id, model_name, input_tokens, 0
                            )
                        ),
                    ):
                        yield _frame
                finally:
                    cleanup_stream(seq_id, aborted=aborted)
                    cleanup_request(request_id)

            return StreamingResponse(
                _client_stream(generate_anthropic_stream(), request_id),
                media_type="text/event-stream",
                headers={
                    "anthropic-version": "2023-06-01",
                    "x-request-id": request_id,
                },
            )

        # Non-streaming response
        final_output = await _run_nonstream_with_disconnect(
            generate_async(prompt, sampling_params, request_id),
            raw_request,
            request_id,
        )
        if final_output is None:
            raise RuntimeError("No output generated")

        raw_text = final_output["text"]
        # Separating is unconditional -- the same call the chat path makes,
        # and what keeps a chain of thought out of the tool parser. Only
        # whether the client is shown it is a question, and the same one the
        # streaming branch above asks.
        # Both stages over one chunk, in the order the branch above streams
        # them: reasoning filter, then the tool parser on the content segments
        # it yields. Calling `.split()` here instead flattened the two into
        # `(reasoning, content)` and lost the interleaving at that line.
        events = read_whole_blocks(
            reasoning_channel(
                prompt_starts_in_reasoning(prompt),
                template_kwargs=merged_kwargs,
            ),
            tool_call_parser_cls,
            raw_text,
            anthropic_to_openai_tools(request.tools),
            suppress_calls=forbids_tool_calls(request.tool_choice),
        )
        if drop_reasoning:
            events = [e for e in events if e[0] != "reasoning"]
        _, tool_calls = flatten_tool_events(events)
        output_tokens = len(tokenizer.encode(raw_text))
        cache_read_input_tokens = final_output.get("num_cached_tokens", 0)
        return build_anthropic_response(
            request_id=request_id,
            model=model_name,
            events=events,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            # A tool call is its own ending; otherwise whatever the engine
            # said. Omitting this left the parameter's `end_turn` default in
            # place, so the same response cut off at `max_tokens` reported a
            # normal ending with `stream=false` and `max_tokens` with
            # `stream=true`.
            stop_reason=anthropic_stop_reason_with_calls(
                final_output.get("finish_reason"), bool(tool_calls)
            ),
        )

    except _ClientDisconnected:
        # Client hung up; seq already aborted + popped. Nothing to return.
        return JSONResponse(status_code=499, content={"detail": "client disconnected"})
    except Exception as e:
        logger.exception("Error in anthropic_messages")
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            },
        )


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return ModelList(data=[ModelCard(id=model_name)])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.api_route("/metrics", methods=["GET", "HEAD"], include_in_schema=False)
async def metrics():
    """Expose cached standalone-engine metrics in Prometheus text format."""
    return Response(
        content=_metrics_exporter.render(),
        headers={"Content-Type": _metrics_exporter.content_type},
    )


@app.get("/debug/mtp_stats")
async def get_mtp_stats():
    """Return current speculative decoding acceptance statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is not initialized")
    try:
        return engine.get_mtp_statistics()
    except Exception as e:
        logger.exception("Failed to get MTP statistics")
        raise HTTPException(
            status_code=500, detail=f"Failed to get MTP statistics: {e!s}"
        )


@app.get("/debug/cache_stats")
async def get_cache_stats():
    """Return cumulative prefix-cache reuse statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is not initialized")
    try:
        return engine.get_cache_statistics()
    except Exception as e:
        logger.exception("Failed to get cache statistics")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache statistics: {e}"
        ) from e


def _resolve_kv_transfer_role(kv_cfg: dict) -> tuple[str | None, int]:
    kv_role = kv_cfg.get("kv_role")
    handshake_port = kv_cfg.get("handshake_port", 6301)
    if kv_role is not None or kv_cfg.get("kv_connector") != "multi":
        return kv_role, handshake_port

    # MultiConnector wraps the real transfer connector. Surface the producer
    # role so atomesh can recognize multi[mooncake-producer + offload] as a
    # prefill node.
    fallback_role = None
    fallback_port = handshake_port
    for sub_cfg in kv_cfg.get("connectors", []):
        sub_role = sub_cfg.get("kv_role")
        if sub_role is None:
            continue
        if fallback_role is None:
            fallback_role = sub_role
            fallback_port = sub_cfg.get("handshake_port", handshake_port)
        if sub_role == "kv_producer":
            return sub_role, sub_cfg.get("handshake_port", handshake_port)
    return fallback_role, fallback_port


@app.get("/kv_transfer_info")
async def kv_transfer_info():
    cfg = engine.config
    kv_cfg = cfg.kv_transfer_config or {}
    kv_role, handshake_port = _resolve_kv_transfer_role(kv_cfg)
    return {
        "tp_size": cfg.tensor_parallel_size,
        "dp_size": cfg.parallel_config.data_parallel_size,
        "kv_role": kv_role,
        "handshake_port": handshake_port,
    }


@app.get("/server_info")
async def server_info():
    """Server metadata for the Atomesh router.

    The router's dp-aware discovery reads ``dp_size`` here to expand the
    per-DP-rank worker set and enable cache-aware routing to the rank that
    holds a request's prefix.
    """
    cfg = engine.config
    return {
        "model_id": model_name,
        "served_model_name": model_name,
        "tp_size": cfg.tensor_parallel_size,
        "dp_size": cfg.parallel_config.data_parallel_size,
    }


@app.post("/start_profile")
async def start_profile():
    """Start profiling the engine."""
    try:
        results = engine.start_profile()
    except Exception as e:
        logger.exception("Failed to start profiling")
        raise HTTPException(status_code=500, detail=f"Failed to start profiling: {e!s}")
    errors = [r["error"] for r in results if "error" in r]
    if errors:
        if len(errors) == len(results):
            detail = errors[0]
        else:
            # Every engine decides for itself and the broadcast has already
            # run, so the ones that were idle are recording by now. Say so
            # rather than letting the 409 imply nothing started.
            detail = (
                f"Profiling already in progress on {len(errors)} of "
                f"{len(results)} engines; the others have now started. "
                "Call /stop_profile before retrying."
            )
        raise HTTPException(status_code=409, detail=detail)
    message = next(
        (r["message"] for r in results if "message" in r), "Profiling started"
    )
    response = {"status": "success", "message": message}
    # Present only when a delay is configured, so a client can wait out the
    # window without parsing the message.
    armed = next(
        (r["armed_after_iters"] for r in results if "armed_after_iters" in r), None
    )
    if armed is not None:
        response["armed_after_iters"] = armed
    return response


@app.post("/stop_profile")
async def stop_profile():
    """Stop profiling the engine."""
    try:
        traces = engine.stop_profile()
        return {
            "status": "success",
            "message": "Profiling stopped. Trace files generated.",
            "traces": traces,
        }
    except Exception as e:
        logger.exception("Failed to stop profiling")
        raise HTTPException(status_code=500, detail=f"Failed to stop profiling: {e!s}")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the server."""
    global engine, tokenizer, model_name, default_chat_template_kwargs, _request_logger
    global tool_call_parser_cls, model_starts_in_reasoning, reasoning_toggle
    global reasoning_dialect, synthetic_token_text
    global custom_message_encoder, _stream_batch_dispatcher

    parser = FlexibleArgumentParser(description="ATOM OpenAI API Server")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server host")
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default="auto",
        help=TOOL_CALL_PARSER_HELP,
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port (note: --port is used for internal engine communication)",
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=5,
        help=(
            "Seconds the server holds an idle keep-alive connection. Pooling "
            "clients hold their end far longer (aiohttp 15s), so a caller that "
            "pauses longer than this reuses a socket the server already closed "
            "and has to re-send. Raise it past the caller's idle time to stop "
            "that; requests here run for minutes, so uvicorn's 5s is short."
        ),
    )
    parser.add_argument(
        "--disable-uvicorn-access-log",
        action="store_true",
        help=(
            "Stop uvicorn logging a line per HTTP request. It copies a "
            "LogRecord and writes to the same stdout the engine logs to, on "
            "the event loop, and says less than the engine's own "
            "'Request N arrived' line."
        ),
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help=(
            "Override the tokenizer's chat template. "
            "Accepts a file path to a Jinja template or an inline Jinja string. "
            "Useful for base models that have no built-in chat_template."
        ),
    )
    parser.add_argument(
        "--default-chat-template-kwargs",
        type=str,
        default=None,
        help=(
            "Default kwargs for chat template rendering (JSON string). "
            "Merged with per-request chat_template_kwargs (request wins). "
            "Example: '{\"enable_thinking\": false}'"
        ),
    )
    parser.add_argument(
        "--request-log",
        type=str,
        default=None,
        help="Path to JSONL file for logging all API requests and responses (debug)",
    )
    args = parser.parse_args()

    if args.request_log:
        _request_logger = logging.getLogger("atom.request_log")
        _request_logger.setLevel(logging.INFO)
        _request_logger.propagate = False
        fh = logging.FileHandler(args.request_log, mode="a")
        fh.setFormatter(logging.Formatter("%(message)s"))
        _request_logger.addHandler(fh)
        logger.info(f"Request logging enabled: {args.request_log}")

    if args.default_chat_template_kwargs:
        default_chat_template_kwargs = json.loads(args.default_chat_template_kwargs)
        logger.info(f"Default chat template kwargs: {default_chat_template_kwargs}")

    logger.info(f"Loading tokenizer from {args.model}...")
    tokenizer = _load_tokenizer(args.model, args.trust_remote_code)
    if args.chat_template:
        if os.path.isfile(args.chat_template):
            with open(args.chat_template, "r", encoding="utf-8") as f:
                tokenizer.chat_template = f.read()
            logger.info(f"Loaded chat template from file: {args.chat_template}")
        else:
            tokenizer.chat_template = args.chat_template
            logger.info("Using inline chat template from --chat-template argument")

    model_name = args.served_model_name if args.served_model_name else args.model
    custom_message_encoder = load_custom_message_encoder(args.model)

    logger.info(f"Initializing engine with model {args.model}...")
    engine_args = EngineArgs.from_cli_args(args)
    _template_source = chat_template_source(tokenizer, custom_message_encoder)
    reasoning_dialect, _dialect_stated = resolve_dialect(
        _template_source,
        render_probe_prompt(tokenizer, custom_message_encoder, tools=False) or "",
    )
    logger.info(
        "Reasoning channel: %s%s",
        reasoning_dialect.think_end_marker,
        "" if _dialect_stated else " (no dialect named in the chat template)",
    )
    model_starts_in_reasoning = template_opens_reasoning_implicitly(_template_source)
    if model_starts_in_reasoning:
        logger.info(
            "Chat template closes a reasoning block it never opens; treating "
            "output as reasoning until the end marker."
        )

    reasoning_toggle = resolve_reasoning_toggle(tokenizer, custom_message_encoder)
    if reasoning_toggle is not None:
        _name, _off, _on = reasoning_toggle
        logger.info(f"Reasoning switches on {_name}: {_on!r} on, {_off!r} off.")
    else:
        logger.info(
            "This chat template has no switch for reasoning, so a request "
            "asking for none cannot stop the model producing it. Any that "
            "arrives is still separated and reported, never discarded."
        )

    tool_call_parser_cls = resolve_tool_call_parser(
        args.tool_call_parser,
        tokenizer,
        custom_message_encoder,
        model=args.model,
    )

    engine = engine_args.create_engine(tokenizer=tokenizer)
    # Forced acceptance emits draft tokens that nothing verified, and once the
    # context is long enough those degenerate into the dialect's own channel
    # framing and nothing else -- read as structure, correctly, that leaves a
    # response with no content at all. The mode already announces that its text is
    # meaningless, so stand in for it rather than parse it.
    synthetic_token_text = (
        SYNTHETIC_TOKEN_TEXT
        if (
            args.spec_decode_acceptance_length is not None
            or args.spec_decode_acceptance_rate is not None
        )
        else None
    )
    if synthetic_token_text is not None:
        logger.warning(
            "Forced speculative acceptance is on, so every generated token is "
            "delivered as %r instead of its decoded text. Token counts, timings "
            "and throughput are unaffected; the text was already meaningless. "
            "Unset --spec-decode-acceptance-length / --spec-decode-acceptance-rate "
            "to read the model's own output again.",
            SYNTHETIC_TOKEN_TEXT,
        )
    _stream_batch_dispatcher = StreamBatchDispatcher(
        tokenizer, synthetic_text=synthetic_token_text
    )

    # Wire the batched stream-flush hook: per-seq stream callbacks only buffer
    # their chunks into a thread-local; the engine core manager's output thread
    # calls this flush after each step's callbacks to drain the buffer into the
    # per-request stream collectors (one call_soon_threadsafe per event loop).
    # Registered lazily here to avoid the api_server <-> engine_core_mgr import
    # cycle; the core manager leaves the hook as None until this resolves it.
    engine.core_mgr._flush_stream_batch_fn = flush_stream_batch

    import signal

    def _sigint_handler(signum, frame):
        logger.info("Received SIGINT, shutting down engine...")
        engine.close()
        import psutil

        try:
            current = psutil.Process()
            children = current.children(recursive=True)
            psutil.wait_procs(children, timeout=2)
            alive = [c for c in children if c.is_running()]
            for c in alive:
                c.kill()
        except psutil.NoSuchProcess:
            pass
        logger.info("Engine shutdown complete.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    # uvloop replaces the stdlib asyncio selector loop with a libuv-backed one,
    # which is markedly faster at the SSE socket I/O (sock.send / selector
    # register-unregister) that saturates the event loop under high streaming
    # concurrency. Fall back to the default loop if uvloop is unavailable.
    try:
        import uvloop  # noqa: F401

        loop_impl = "uvloop"
    except ImportError:
        loop_impl = "auto"
        logger.warning(
            "uvloop not installed; falling back to the default asyncio loop."
        )

    logger.info(
        f"Starting server on {args.host}:{args.server_port} (loop={loop_impl})..."
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.server_port,
        loop=loop_impl,
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    main()
