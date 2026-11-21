# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`atom.grpc.v1.AtomEngine` beside the HTTP server, in the process that owns the
engine: a `grpc.aio` server on its own asyncio loop, in one named daemon thread."""

import asyncio
import json
import logging
import threading
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

from atom.entrypoints.grpc import atom_grpc_pb2 as pb
from atom.entrypoints.grpc import atom_grpc_pb2_grpc as pb_grpc
from atom.entrypoints.grpc import frames
from atom.entrypoints.grpc.session import Sessions
from atom.entrypoints.openai.api_compatibility import api_compatibility_for
from atom.entrypoints.openai.frontend_workers import worker_seq_ids
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
from atom.entrypoints.openai.serving_chat import (
    normalize_chat_tools,
    request_tools,
    validate_chat_request,
)
from atom.entrypoints.openai.tool_parser.registry import channel_for_request
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.model_engine.zmq_protocol import engine_v1_pb2 as v1
from atom.structured_output.validation import schema_from_format

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
N_ABOVE_1 = "n above 1 is not served over gRPC; send one request each"


class StepOutputs:
    """Engine outputs for gRPC calls, buffered per output thread during a step; `flush`
    hands the step to each call's loop with one `call_soon_threadsafe`."""

    def __init__(self):
        self._local = threading.local()

    def callback(self, put, end=None) -> Callable[[RequestOutput], None]:
        """A stream callback buffering each output for `put`, called on the running loop;
        `end`, if given, runs there once after the step's last output."""
        loop, local = asyncio.get_running_loop(), self._local

        def buffer(output: RequestOutput) -> None:
            step = getattr(local, "step", None)
            if step is None:
                step = local.step = defaultdict(list)
            step[loop].append((put, end, output))

        return buffer

    def flush(self) -> None:
        step = getattr(self._local, "step", None)
        if not step:
            return
        self._local.step = None
        for loop, outputs in step.items():
            loop.call_soon_threadsafe(_deliver, outputs)


def _deliver(outputs: list[tuple[Callable, Callable | None, RequestOutput]]) -> None:
    ends = {}
    for put, end, output in outputs:
        put(output)
        ends[end] = None
    for end in ends:
        if end is not None:
            end()


class AtomEngineServicer(pb_grpc.AtomEngineServicer):
    """Answers from `api`, the API server module running the engine, as its routes
    do; every RPC not defined here answers UNIMPLEMENTED."""

    def __init__(self, api: ModuleType, model_path: str):
        self.api = api
        self.model_path = model_path  # --model, which the tokenizer loads from
        self.steps = StepOutputs()
        self.sessions = Sessions(api)
        self.live: dict[str, set[int]] = {}  # request_id: its live seq ids, for Abort

    async def Generate(self, request, context):
        """Admitted as `/v1/completions` admits its ids, plus its chat's structured
        output; a refusal answered with its status; streamed one chunk per step."""
        api, outputs = self.api, asyncio.Queue()
        seq, submitted, finished = None, False, False
        try:
            params, kv_transfer_params, stops = self._admission(request)
            prompt = resolve_prompt_token_ids(
                list(request.prompt_token_ids), kv_transfer_params
            )
            if request.HasField("structured_output"):
                fields = await self._structured(
                    request, params, kv_transfer_params, stops
                )
                params = replace(params, **fields)
            seq = await asyncio.get_running_loop().run_in_executor(
                None,
                partial(
                    api.engine.io_processor.preprocess,
                    prompt,
                    params,
                    stream_callback=self.steps.callback(outputs.put_nowait),
                    kv_transfer_params=api._engine_kv_transfer_params(
                        kv_transfer_params
                    ),
                    data_parallel_rank=_optional(request, "data_parallel_rank"),
                    stop_token_sequences=stops,
                ),
            )
            self.live.setdefault(request.request_id, set()).add(seq.id)
            api._validate_sequence_context_length(seq, params)
            await api._submit_engine_requests([seq])
            submitted = True
            ids, cached = [], 0
            while not finished:
                output = await outputs.get()
                finished = output.finished
                ids += output.output_tokens
                cached = output.num_cached_tokens or cached
                if request.stream and output.output_tokens:
                    chunk = pb.GenerateStreamChunk(
                        token_ids=output.output_tokens,
                        prompt_tokens=seq.num_prompt_tokens,
                        completion_tokens=len(ids),
                        cached_tokens=cached,
                    )
                    await context.write(pb.GenerateResponse(chunk=chunk))
            complete = pb.GenerateComplete(
                output_ids=[] if request.stream else ids,
                finish_reason=output.finish_reason,
                prompt_tokens=seq.num_prompt_tokens,
                completion_tokens=len(ids),
                cached_tokens=cached,
            )
            if output.kv_transfer_params_output is not None:
                kv_json = json.dumps(output.kv_transfer_params_output)
                complete.kv_transfer_params_json = kv_json
            await context.write(pb.GenerateResponse(complete=complete))
        except Exception as error:  # noqa: BLE001 - answered as `_completions` does
            await context.write(pb.GenerateResponse(error=_refusal(error, "Generate")))
        finally:
            if seq is not None:
                self._forget(request.request_id, seq.id)
                if submitted:  # the HTTP stream teardown
                    api.cleanup_stream(seq.id, aborted=not finished)
                else:
                    api.engine.io_processor.requests.pop(seq.id, None)

    async def Abort(self, request, context):
        """Ends the request as a dropped HTTP stream ends one; an unknown or finished
        request_id is no error."""
        for seq_id in list(self.live.get(request.request_id, ())):
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
        if sampling.n > 1:
            raise ValueError(N_ABOVE_1)
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
        # as asked, though the SamplingParams Generate admits carry neither.
        chat_params = replace(
            params,
            stop_strings=[str(ids) for ids in stops],
            logprobs=getattr(request.sampling, logprobs) if logprobs else False,
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

    async def Session(self, request_iterator, context):
        """engine_v1 messages both ways, Ready first, its seq ids in a window of its own;
        once the client's side ends, is cancelled or unreadable, its requests abort."""
        session = self.sessions.open()
        if session is None:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "no Session window free"
            )
        reader = asyncio.create_task(self._read(session, request_iterator))
        try:
            ready = frames.encode_ready(self.api._get_engine_max_pool_tokens())
            yield pb.Frames(frames=ready)
            while (message := await session.outbox.get()) is not None:
                yield pb.Frames(frames=message)
            await reader  # how the client's side ended
        except frames.FrameError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        finally:
            reader.cancel()
            session.close()

    async def _read(self, session, request_iterator) -> None:
        """Admits and aborts what the client sends, in order, until its side ends."""
        try:
            async for message in request_iterator:
                kind, items = frames.decode_client(message.frames)
                if kind == "abort":
                    session.abort(items)
                    continue
                refused = []
                for item in items:
                    if isinstance(item, frames.Rejection):  # the codec's INVALID
                        refused.append(item)
                        continue
                    try:
                        await self._submit(session, item)
                    except Exception as error:  # noqa: BLE001 - refused as Generate is
                        refused.append(_rejected(item.seq_id, error))
                session.reject(refused)
        finally:
            session.outbox.put_nowait(None)

    async def _submit(self, session, request: frames.SubmitRequest) -> None:
        """Admits one RequestSpec as Generate admits a request, under the Session's id
        for it, with the structured fields PrepareStructuredOutput answered."""
        api, given = self.api, request.sampling_params
        seq_id = session.seq_id(request.seq_id)
        if given.n > 1:
            raise ValueError(N_ABOVE_1)
        kv_transfer_params = request.kv_transfer_params
        api._validate_chunked_receive(kv_transfer_params)
        params = api._build_sampling_params(
            temperature=given.temperature,
            max_tokens=validate_max_tokens(given.max_tokens),
            max_tokens_is_default=given.max_tokens_is_default,
            stop_strings=None,  # they arrive as ids, which preprocess takes
            ignore_eos=given.ignore_eos,
            top_k=given.top_k,
            top_p=given.top_p,
        )
        structured = {name: getattr(given, name) for name in STRUCTURED_FIELDS}
        params = replace(params, **structured)
        prompt = resolve_prompt_token_ids(request.prompt_token_ids, kv_transfer_params)
        seq = await asyncio.get_running_loop().run_in_executor(
            None,
            partial(
                api.engine.io_processor.preprocess,
                prompt,
                params,
                stream_callback=self.steps.callback(session.put, session.flush),
                kv_transfer_params=api._engine_kv_transfer_params(kv_transfer_params),
                data_parallel_rank=request.data_parallel_rank,
                dp_session_id=request.dp_session_id,
                dp_parent_session_id=request.dp_parent_session_id,
                stop_token_sequences=request.stop_token_sequences,
                seq_id=seq_id,
            ),
        )
        session.add(request.seq_id, seq, request.request_id)
        try:
            api._validate_sequence_context_length(seq, params)
            await api._submit_engine_requests([seq])
        except BaseException:
            session.drop(request.seq_id)
            raise

    def _forget(self, request_id: str, seq_id: int) -> None:
        live = self.live.get(request_id, set())
        live.discard(seq_id)
        if not live:
            self.live.pop(request_id, None)

    async def HealthCheck(self, request, context):
        problem = self.api.health_problem()
        return pb.HealthCheckResponse(healthy=problem is None, message=problem or "")

    async def GetModelInfo(self, request, context):
        api, config = self.api, self.api.engine.config
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
        )

    async def GetServerInfo(self, request, context):
        return pb.GetServerInfoResponse(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            kv_transfer_info_json=json.dumps(self.api.kv_transfer_info_body()),
            server_info_json=json.dumps(self.api.server_info_body()),
        )


def _optional(message, name: str, default=None):
    return getattr(message, name) if message.HasField(name) else default


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


def _rejected(seq_id: int, error: Exception) -> frames.Rejection:
    """`error` as Generate answers it, with the code atomesh reads back as that status:
    INVALID for a 4xx, OVERLOADED for 503 and INTERNAL otherwise."""
    refusal, codes = _refusal(error, "Session"), v1.RejectedRequest
    if 400 <= refusal.http_status < 500:
        code = codes.INVALID
    elif refusal.http_status == 503:
        code = codes.OVERLOADED
    else:
        code = codes.INTERNAL
    return frames.Rejection(seq_id, refusal.message, code)


def _then(flush: Callable[[], None] | None, more: Callable[[], None]):
    """One per-step flush: `flush`, then `more` even when `flush` fails."""

    def both() -> None:
        try:
            if flush is not None:
                flush()
        finally:
            more()

    return both


def start_grpc_server(api: ModuleType, args: Namespace) -> None:
    """Serve AtomEngine on `--host`:`--grpc-port`; return once it listens, or raise
    why it cannot."""
    # The API process's own ids wrap inside [0, 2^24), window -1 in worker_seq_ids'
    # numbering: below every Session's window.
    Sequence.counter = worker_seq_ids(-1, next(Sequence.counter))
    host = f"[{args.host}]" if ":" in args.host else args.host  # an IPv6 address
    address, servicer = f"{host}:{args.grpc_port}", AtomEngineServicer(api, args.model)
    listening = Future()
    threading.Thread(
        target=_run, args=(address, servicer, listening), name=THREAD_NAME, daemon=True
    ).start()
    listening.result()
    core_mgr = api.engine.core_mgr  # its per-step flush now hands gRPC its outputs too
    flush = _then(core_mgr._flush_stream_batch_fn, servicer.steps.flush)
    core_mgr._flush_stream_batch_fn = flush


def _run(address: str, servicer: AtomEngineServicer, listening: Future) -> None:
    setproctitle.setthreadtitle(THREAD_NAME)
    asyncio.run(_serve(address, servicer, listening))


async def _serve(address: str, servicer: AtomEngineServicer, listening: Future) -> None:
    # Without SO_REUSEPORT a second server on the port fails, not shares it; a Session's
    # SubmitBatch of long prompts may pass gRPC's default 4 MiB, so no receive limit.
    options = [("grpc.so_reuseport", 0), ("grpc.max_receive_message_length", -1)]
    try:
        server = grpc.aio.server(options=options)
        pb_grpc.add_AtomEngineServicer_to_server(servicer, server)
        server.add_insecure_port(address)  # raises when the port is taken
        await server.start()
    except Exception as error:  # noqa: BLE001 - start_grpc_server raises it
        listening.set_exception(error)
        return
    listening.set_result(None)
    await server.wait_for_termination()
