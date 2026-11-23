# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""`atom.grpc.v1.AtomEngine` beside the HTTP server, in the process that owns the
engine: a `grpc.aio` server on its own asyncio loop, in one named daemon thread."""

import asyncio
import json
import logging
import threading
import time
from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import replace
from functools import partial
from types import ModuleType

import grpc
import setproctitle
from fastapi import HTTPException
from google.protobuf.json_format import MessageToDict

from atom.entrypoints.grpc import atom_grpc_pb2 as pb
from atom.entrypoints.grpc import atom_grpc_pb2_grpc as pb_grpc
from atom.entrypoints.openai.api_compatibility import api_compatibility_for
from atom.entrypoints.openai.protocol import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    ChatCompletionRequest,
    resolve_prompt_token_ids,
    validate_max_tokens,
)
from atom.entrypoints.openai.reasoning import _REASONING_OPEN_MARKERS
from atom.entrypoints.openai.reasoning_dialects import THINK_END_MARKER
from atom.entrypoints.openai.serving_chat import (
    normalize_chat_tools,
    request_tools,
    validate_chat_request,
)
from atom.entrypoints.openai.tool_parser.registry import channel_for_request
from atom.model_engine.request import RequestOutput
from atom.model_engine.zmq_protocol import engine_v1_pb2 as v1
from atom.structured_output.validation import (
    raise_for_structured_error,
    schema_from_format,
)

logger = logging.getLogger("atom")

# Also its name in /proc, which threads it starts keep unless they name their own.
THREAD_NAME = "AtomGrpcServer"
# The SamplingParams fields a chat's structured output sets: PrepareStructuredOutput's.
STRUCTURED_FIELDS = (
    "required_channel",
    "json_schema",
    "structured_tool_format",
    "structured_reasoning_end",
)
# SamplingParams fields Generate does not serve: no logprobs in its answer, and the
# structured ones derived from structured_output.
UNSERVED_FIELDS = {
    "logprobs_flag",
    "logprobs_count",
    "top_logprobs",
    *STRUCTURED_FIELDS,
}
# n above 1 is served unstreamed and without P/D, where atomesh's placement sends it.
N_STREAMED = "a streamed n above 1 is not served over gRPC; send it unstreamed"
N_TRANSFERRED = "n above 1 with kv_transfer_params (P/D) is not served over gRPC"
# ATOM's tool-call formats by atomesh's name for the same syntax. ATOM's `qwen` is the
# XML one, atomesh's `qwen_xml`; GLM's two dialects are one ATOM parser, so atomesh's
# model-name mapping picks GLM's.
ROUTER_TOOL_CALL_PARSERS = {
    "dsml": "dsml",
    "dsml_v41": "dsml_v41",
    "kimi": "kimik2",
    "kimi_k3": "kimi_k3",
    "minimax": "minimax",
    "qwen": "qwen_xml",
}


class StepOutputs:
    """Engine outputs for gRPC calls, buffered per output thread during a step; `flush`
    hands the step to each call's loop with one `call_soon_threadsafe`."""

    def __init__(self):
        self._local = threading.local()

    def callback(self, queue: asyncio.Queue) -> Callable[[RequestOutput], None]:
        """A stream callback buffering for `queue`, which the running loop reads."""
        loop, local = asyncio.get_running_loop(), self._local

        def buffer(output: RequestOutput) -> None:
            step = getattr(local, "step", None)
            if step is None:
                step = local.step = defaultdict(list)
            step[loop].append((queue, output))

        return buffer

    def flush(self) -> None:
        step = getattr(self._local, "step", None)
        if not step:
            return
        self._local.step = None
        for loop, outputs in step.items():
            try:
                loop.call_soon_threadsafe(_deliver, outputs)
            except RuntimeError:  # closed: the server stopped after its calls ended
                pass


def _deliver(outputs: list[tuple[asyncio.Queue, RequestOutput]]) -> None:
    for queue, output in outputs:
        queue.put_nowait(output)


class _Sibling:
    """One seq of a Generate call, n of them for n above 1: what its complete reports,
    its times measured as generate_async measures them from the call's arrival."""

    def __init__(self, index: int, prompt_tokens: int, started: float):
        self.index, self.prompt_tokens, self.started = index, prompt_tokens, started
        self.ids, self.completion_tokens, self.cached = [], 0, 0
        self.first = self.last = None
        self.finished = False

    def add(self, output: RequestOutput, delivered: float, keep_ids: bool) -> None:
        if output.output_tokens:
            self.last = delivered
            if self.first is None:
                self.first = self.last
        self.completion_tokens += len(output.output_tokens)
        if keep_ids:
            self.ids += output.output_tokens
        self.cached = output.num_cached_tokens or self.cached
        self.finished = output.finished

    def complete(self, output: RequestOutput) -> pb.GenerateComplete:
        tokens = self.completion_tokens
        complete = pb.GenerateComplete(
            output_ids=self.ids,
            finish_reason=output.finish_reason,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=tokens,
            cached_tokens=self.cached,
            latency_s=time.time() - self.started,
            index=self.index,
        )
        if self.first is not None:
            complete.ttft_s = self.first - self.started
        if tokens > 1:
            complete.tpot_s = (self.last - self.first) / (tokens - 1)
        if output.kv_transfer_params_output is not None:
            kv_json = json.dumps(output.kv_transfer_params_output)
            complete.kv_transfer_params_json = kv_json
        return complete


class _Call:
    """A Generate call as Abort finds it: its seqs once made, and whether aborted."""

    def __init__(self):
        self.seq_ids: list[int] = []
        self.aborted = False


class AtomEngineServicer(pb_grpc.AtomEngineServicer):
    """Answers from `api`, the API server module running the engine, as its routes
    do; every RPC not defined here answers UNIMPLEMENTED."""

    def __init__(self, api: ModuleType, model_path: str):
        self.api = api
        self.model_path = model_path  # --model, which the tokenizer loads from
        self.steps = StepOutputs()
        self.live: dict[str, set[_Call]] = {}  # request_id: its Generate calls

    async def Generate(self, request, context):
        """Admitted as `/v1/completions` admits its ids, plus its chat's structured
        output; a refusal answered with its status; streamed one chunk per step. n above
        1, unstreamed only, answers one complete per sibling as each ends."""
        api, outputs, call, started = self.api, asyncio.Queue(), _Call(), time.time()
        seqs, siblings, submitted = [], {}, False  # siblings by seq id
        self.live.setdefault(request.request_id, set()).add(call)
        self._log(request.request_id, "request", request)
        try:
            _refuse_unserved(request.sampling)
            params, kv_transfer_params, stops = self._admission(request)
            if params.n > 1 and request.stream:
                raise ValueError(N_STREAMED)
            if params.n > 1 and kv_transfer_params is not None:
                raise ValueError(N_TRANSFERRED)
            # An empty repeated field is an absent one.
            prompt = resolve_prompt_token_ids(
                list(request.prompt_token_ids) or None, kv_transfer_params
            )
            if prompt is None:
                raise ValueError("prompt_token_ids is required")
            if request.HasField("structured_output"):
                fields = await self._structured(
                    request, params, kv_transfer_params, stops
                )
                params = replace(params, **fields)
            buffer = self.steps.callback(outputs)

            def stream_callback(output: RequestOutput) -> None:
                # Timed as the engine delivers it, as generate_async times its outputs.
                buffer((output, time.time()))

            preprocessing = asyncio.get_running_loop().run_in_executor(
                None,
                partial(
                    api.engine.io_processor.preprocess_fanout,
                    prompt,
                    params,
                    stream_callback=stream_callback,
                    kv_transfer_params=api._engine_kv_transfer_params(
                        kv_transfer_params
                    ),
                    data_parallel_rank=_optional(request, "data_parallel_rank"),
                    dp_session_id=_optional(request, "dp_session_id"),
                    dp_parent_session_id=_optional(request, "dp_parent_session_id"),
                    stop_token_sequences=stops,
                ),
            )
            try:
                seqs = await asyncio.shield(preprocessing)
            except asyncio.CancelledError:  # the seqs it makes are dropped once made
                preprocessing.add_done_callback(partial(_drop_preprocessed, api))
                raise
            call.seq_ids = [seq.id for seq in seqs]
            for index, seq in enumerate(seqs):
                api._validate_sequence_context_length(seq, params)
                siblings[seq.id] = _Sibling(index, seq.num_prompt_tokens, started)
            if call.aborted:  # before its submission: none reaches the engine
                for seq in seqs:
                    aborted = RequestOutput(seq.id, [], True, "aborted")
                    outputs.put_nowait((aborted, time.time()))
            else:
                submitted = True  # a failed submission may have dispatched some seqs
                await api._submit_engine_requests(seqs)
                if call.aborted:  # during its submission
                    for seq in seqs:
                        api.engine.core_mgr.abort_request(seq.id)
            running = len(seqs)
            while running:
                output, delivered = await outputs.get()
                sibling = siblings[output.request_id]
                if sibling.finished:
                    continue
                # A stream's tokens go out in its chunks.
                sibling.add(output, delivered, keep_ids=not request.stream)
                raise_for_structured_error(output.finish_reason)  # HTTP chat's 500
                if request.stream and output.output_tokens:
                    chunk = pb.GenerateStreamChunk(
                        token_ids=output.output_tokens,
                        prompt_tokens=sibling.prompt_tokens,
                        completion_tokens=sibling.completion_tokens,
                        cached_tokens=sibling.cached,
                    )
                    await self._write(
                        context, request, pb.GenerateResponse(chunk=chunk)
                    )
                if sibling.finished:
                    running -= 1
                    complete = sibling.complete(output)
                    response = pb.GenerateResponse(complete=complete)
                    await self._write(context, request, response)
        except Exception as error:  # noqa: BLE001 - answered as `_completions` does
            refusal = pb.GenerateResponse(error=_refusal(error, "Generate"))
            await self._write(context, request, refusal)
        finally:
            self._forget(request.request_id, call)
            for seq in seqs:
                if submitted:  # the HTTP stream teardown, per sibling
                    api.cleanup_stream(seq.id, aborted=not siblings[seq.id].finished)
                else:
                    api.engine.io_processor.requests.pop(seq.id, None)

    async def Abort(self, request, context):
        """Ends each of the request's seqs as a dropped HTTP stream ends one, or keeps
        them out of the engine; an unknown or finished request_id is no error."""
        for call in list(self.live.get(request.request_id, ())):
            call.aborted = True
            for seq_id in call.seq_ids:
                self.api.cleanup_stream(seq_id, aborted=True)
        return pb.AbortResponse()

    async def PrepareStructuredOutput(self, request, context):
        """The structured-output fields a Generate with the same fields is admitted
        with, each rank's grammar prepared; a refusal answered with its status."""
        try:
            fields = await self._structured(request, *self._admission(request))
        except Exception as error:  # noqa: BLE001 - answered as the chat route does
            error = _refusal(error, "PrepareStructuredOutput")
            return pb.PrepareStructuredOutputResponse(error=error)
        return pb.PrepareStructuredOutputResponse(sampling=_engine_v1(fields))

    def _admission(self, request):
        """The SamplingParams `_completions` builds, in its order, with the request's
        kv_transfer_params and its stop ids as given."""
        api, sampling = self.api, request.sampling
        kv_transfer_params = _kv_transfer_params(request)
        api._validate_chunked_receive(kv_transfer_params)
        temperature = _optional(sampling, "temperature", DEFAULT_TEMPERATURE)
        n = api._coerce_n(_optional(sampling, "n"), temperature)
        given = sampling.HasField("max_tokens")
        max_tokens = DEFAULT_MAX_TOKENS
        if given:
            max_tokens = validate_max_tokens(sampling.max_tokens)
        params = api._build_sampling_params(
            temperature=temperature,
            max_tokens=max_tokens,
            max_tokens_is_default=sampling.max_tokens_is_default or not given,
            stop_strings=None,  # they arrive as ids, which preprocess takes
            ignore_eos=sampling.ignore_eos,
            top_k=_optional(sampling, "top_k", DEFAULT_TOP_K),
            top_p=_optional(sampling, "top_p", DEFAULT_TOP_P),
            n=n,
        )
        stops = [list(stop.ids) for stop in request.stop_token_sequences]
        return params, kv_transfer_params, stops

    async def _structured(self, request, params, kv_transfer_params, stops) -> dict:
        """`STRUCTURED_FIELDS` as `_chat_completions` derives them for the chat the
        request carries, given its SamplingParams, kv_transfer_params and stop ids."""
        api, carried = self.api, request.structured_output
        chat = _chat_request(carried)
        validate_chat_request(chat)
        kwargs = json.loads(carried.chat_template_kwargs_json or "{}")
        if not isinstance(kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object")  # noqa: TRY004
        contract = api_compatibility_for(api.custom_message_encoder)
        if not contract.chat_forced_tools_with_thinking:
            thinking = api._effective_thinking_enabled(kwargs)
            contract.validate_chat_tool_choice(
                chat.tool_choice, thinking_enabled=thinking
            )
        tools = request_tools(chat)
        logprobs = request.sampling.WhichOneof("logprobs")
        # What the chat route checks: the stop ids count as stop strings and logprobs
        # as asked, though the SamplingParams Generate admits carry none of them.
        chat_params = replace(
            params,
            stop_strings=[str(ids) for ids in stops],
            logprobs=getattr(request.sampling, logprobs) if logprobs else False,
            top_logprobs=request.sampling.top_logprobs,  # SamplingParams checks both
            required_channel=channel_for_request(
                api.required_tool_channel, chat.tool_choice, tools
            ),
        )
        await api._configure_chat_structured_output(
            chat_params,
            schema_from_format(chat.response_format),
            api.reasoning_channel(
                carried.prompt_opens_reasoning, template_kwargs=kwargs
            ),
            tools=chat.tools,
            all_tools=tools,
            tool_choice=chat.tool_choice,
            kv_transfer=kv_transfer_params,
        )
        return {name: getattr(chat_params, name) for name in STRUCTURED_FIELDS}

    async def _write(self, context, request, response) -> None:
        kind = "stream_chunk" if response.HasField("chunk") else "response"
        self._log(request.request_id, kind, response)
        await context.write(response)

    def _log(self, request_id: str, kind: str, message) -> None:
        """`message` in --request-log's file, as the HTTP routes log theirs."""
        if self.api._request_logger is not None:
            data = MessageToDict(message, preserving_proto_field_name=True)
            self.api._log_request_event(kind, request_id, data)

    def _forget(self, request_id: str, call: _Call) -> None:
        live = self.live.get(request_id, set())
        live.discard(call)
        if not live:
            self.live.pop(request_id, None)

    async def HealthCheck(self, request, context):
        problem = self.api.health_problem()
        return pb.HealthCheckResponse(healthy=problem is None, message=problem or "")

    async def GetModelInfo(self, request, context):
        api, config = self.api, self.api.engine.config
        tool_call_parser, reasoning_parser = router_parser_names(
            api.tool_call_parser_cls, api.reasoning_dialect
        )
        return pb.GetModelInfoResponse(
            model_path=self.model_path,
            tokenizer_path=self.model_path,
            served_model_name=api.model_name,
            max_context_length=config.max_model_len,
            eos_token_ids=config.end_token_ids,
            chat_template=api.tokenizer.chat_template,
            default_chat_template_kwargs_json=json.dumps(
                api.default_chat_template_kwargs
            ),
            reasoning_open_markers=_REASONING_OPEN_MARKERS,
            tool_call_parser=tool_call_parser,  # None leaves it unset
            reasoning_parser=reasoning_parser,
            usage_on_finish=api_compatibility_for(
                api.custom_message_encoder
            ).usage_on_finish,
            implicit_reasoning_opener=(
                api.reasoning_dialect.prompt_open_marker
                if api.model_starts_in_reasoning and api.reasoning_dialect
                else None
            ),
            reasoning_toggle_json=(
                json.dumps(list(api.reasoning_toggle)) if api.reasoning_toggle else ""
            ),
        )

    async def GetServerInfo(self, request, context):
        return pb.GetServerInfoResponse(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            kv_transfer_info_json=json.dumps(self.api.kv_transfer_info_body()),
            server_info_json=json.dumps(self.api.server_info_body()),
            kv_event_sources_json=json.dumps(self.api.engine.core_mgr.kv_event_sources),
        )


def router_parser_names(tool_call_parser_cls, reasoning_dialect) -> tuple:
    """atomesh's tool-call and reasoning parser names for ATOM's, each None where atomesh
    has none. The `<think>` one, qwen3, starts outside reasoning: the prompt decides."""
    tool_call = getattr(tool_call_parser_cls, "NAME", None)
    reasoning = getattr(reasoning_dialect, "think_end_marker", None)
    return (
        ROUTER_TOOL_CALL_PARSERS.get(tool_call),
        "qwen3" if reasoning == THINK_END_MARKER else None,
    )


def _optional(message, name: str, default=None):
    return getattr(message, name) if message.HasField(name) else default


def _refuse_unserved(sampling) -> None:
    unserved = sorted(
        f.name for f, _ in sampling.ListFields() if f.name in UNSERVED_FIELDS
    )
    if unserved:
        raise ValueError(f"Generate does not serve sampling fields {unserved}")


def _drop_preprocessed(api: ModuleType, preprocessing: asyncio.Future) -> None:
    if not preprocessing.cancelled() and preprocessing.exception() is None:
        for seq in preprocessing.result():
            api.engine.io_processor.requests.pop(seq.id, None)


def _json(message, name: str, default=None):
    """`message`'s optional JSON field `name`, parsed; `default` when it is absent."""
    return json.loads(getattr(message, name)) if message.HasField(name) else default


def _kv_transfer_params(request) -> dict | None:
    """The request's kv_transfer_params, a JSON object as the HTTP routes take it."""
    if not request.HasField("kv_transfer_params_json"):
        return None
    params = json.loads(request.kv_transfer_params_json)
    if params is not None and not isinstance(params, dict):
        raise ValueError("kv_transfer_params must be a JSON object")
    return params


def _chat_request(carried) -> ChatCompletionRequest:
    """The chat a StructuredOutput carries, for the chat route's readers: its tools,
    tool_choice and response_format, and messages holding only their own tools."""
    message_tools = _json(carried, "message_tools_json", [])
    if not isinstance(message_tools, list):
        raise ValueError("message_tools must be an array")  # noqa: TRY004
    return ChatCompletionRequest.model_construct(
        messages=[{"tools": tools} for tools in message_tools],
        tools=normalize_chat_tools(_json(carried, "tools_json")),
        tool_choice=_json(carried, "tool_choice_json"),
        response_format=_json(carried, "response_format_json"),
    )


def _engine_v1(fields: dict) -> v1.SamplingParams:
    """`STRUCTURED_FIELDS` in an engine_v1 SamplingParams, its other fields unset."""
    sampling = v1.SamplingParams(
        structured_reasoning_end=fields["structured_reasoning_end"]
    )
    for name in ("json_schema", "structured_tool_format"):
        if fields[name] is not None:
            setattr(sampling, name, fields[name])
    if fields["required_channel"] is not None:
        channel = v1.RequiredChannel(**vars(fields["required_channel"]))
        sampling.required_channel.CopyFrom(channel)
    return sampling


def _refusal(error: Exception, rpc: str) -> pb.GenerateError:
    """`error` as the HTTP routes answer it: an HTTPException keeps its status, a
    ValueError is a 400 and anything else a 500."""
    if isinstance(error, HTTPException):
        status, message = error.status_code, str(error.detail)
    elif isinstance(error, ValueError):
        logger.error("Validation error in %s: %s", rpc, error)
        status, message = 400, str(error)
    else:
        logger.error("Error in %s", rpc, exc_info=error)
        status, message = 500, str(error)
    return pb.GenerateError(message=message, http_status=status)


def _then(flush: Callable[[], None] | None, more: Callable[[], None]):
    """One per-step flush: `flush`, then `more` even when `flush` fails."""

    def both() -> None:
        try:
            if flush is not None:
                flush()
        finally:
            more()

    return both


def startup_refusal(tokenizer) -> str | None:
    """Why --grpc-port cannot serve with `tokenizer`, or None: GetModelInfo carries
    one chat template, and atomesh's tokenizer reads only one."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        names = ", ".join(map(str, template))
        return (
            f"--grpc-port serves one chat template, and this tokenizer has named "
            f"ones ({names}); pick one with --chat-template"
        )
    return None


def start_grpc_server(api: ModuleType, args: Namespace) -> Callable[[], None]:
    """Serve AtomEngine on `--host`:`--grpc-port`; once it listens, return its stop,
    or raise why it cannot listen."""
    host = f"[{args.host}]" if ":" in args.host else args.host  # an IPv6 address
    address, servicer = f"{host}:{args.grpc_port}", AtomEngineServicer(api, args.model)
    # Before the listener starts: the per-step flush hands gRPC every call's outputs.
    core_mgr = api.engine.core_mgr
    flush = _then(core_mgr._flush_stream_batch_fn, servicer.steps.flush)
    core_mgr._flush_stream_batch_fn = flush
    listening = Future()
    thread = threading.Thread(
        target=_run, args=(address, servicer, listening), name=THREAD_NAME, daemon=True
    )
    thread.start()
    loop, stopping = listening.result()

    def stop() -> None:
        """Refuses new calls and ends the running ones, their seqs aborted; run before
        the engine closes, which would leave them waiting forever."""
        loop.call_soon_threadsafe(stopping.set)
        thread.join()  # server.stop returns before the calls' teardown has run

    return stop


def _run(address: str, servicer: AtomEngineServicer, listening: Future) -> None:
    setproctitle.setthreadtitle(THREAD_NAME)
    asyncio.run(_serve(address, servicer, listening))


async def _serve(address: str, servicer: AtomEngineServicer, listening: Future) -> None:
    # Without SO_REUSEPORT a second server on the port fails, not shares it; a long
    # prompt's Generate may pass gRPC's default 4 MiB, so no receive limit.
    options = [("grpc.so_reuseport", 0), ("grpc.max_receive_message_length", -1)]
    try:
        server = grpc.aio.server(options=options)
        pb_grpc.add_AtomEngineServicer_to_server(servicer, server)
        server.add_insecure_port(address)  # raises when the port is taken
        await server.start()
    except Exception as error:  # noqa: BLE001 - start_grpc_server raises it
        listening.set_exception(error)
        return
    stopping = asyncio.Event()
    listening.set_result((asyncio.get_running_loop(), stopping))
    await stopping.wait()
    # Stopped here, not from another task: asyncio.run would cancel that one.
    await server.stop(None)
