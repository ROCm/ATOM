# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Encode and decode every CoreManager <-> EngineCore message.

Client side: `encode_submit` / `encode_abort` / `encode_utility` /
`encode_shutdown`, and `decode_core_message` for what comes back.

Core side: `decode_client_message`, `build_sequence` to turn a decoded
request into the `Sequence` the scheduler has always received, and an
`encode_*` for every kind of item the Core puts on its output queue.

Decoded kinds are `EngineCoreRequestType` members, so the dispatch on both
ends reads as it did when the payloads were pickled.
"""

import array
import logging
import pickle
from dataclasses import fields
from functools import cache, lru_cache
from itertools import chain

import numpy as np

from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.prefix_hash import ShippedChain
from atom.model_engine.protocol import engine_v1_pb2 as pb
from atom.model_engine.protocol.requests import CompletionResult, IngressRequest
from atom.model_engine.protocol.wire import (
    FrameReader,
    FrameWriter,
    ProtocolError,
    decode_envelope,
    decode_optional_value,
    decode_value,
    encode_optional_value,
    encode_value,
)
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import PER_REQ_CACHE_MODEL_TYPES, Sequence
from atom.observability import tracing
from atom.sampling_params import RequiredChannel, SamplingParams

logger = logging.getLogger("atom")

# Every SamplingParams field the wire carries. A field added to the dataclass
# without a wire field would silently fall back to its default in the Core, so
# this module refuses to import until the codec carries it.
SAMPLING_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "max_tokens",
    "ignore_eos",
    "stop_strings",
    "n",
    "logprobs",
    "required_channel",
    "top_logprobs",
    "json_schema",
    "structured_tool_format",
    "structured_reasoning_end",
    "max_tokens_is_default",
)
if {f.name for f in fields(SamplingParams)} != set(SAMPLING_FIELDS):
    raise ImportError(
        "SamplingParams and the engine wire protocol disagree on fields; "
        "update engine_v1.proto and codec._encode/_decode_sampling"
    )

# RequestOutput's fields in constructor order, as `_decode_stream` builds one
# positionally. A field added or reordered there must be carried here first.
STREAM_FIELDS = (
    "request_id",
    "output_tokens",
    "finished",
    "finish_reason",
    "kv_transfer_params_output",
    "num_cached_tokens",
    "logprobs",
)
if tuple(f.name for f in fields(RequestOutput)) != STREAM_FIELDS:
    raise ImportError(
        "RequestOutput and the engine wire protocol disagree on fields; "
        "update engine_v1.proto (StreamBatch) and codec.encode/_decode_stream"
    )

# A rejected admission's leave_reason starts with this.
REJECTED_REASON = "rejected"

# Why a request was rejected (RejectedRequest.Code).
REJECT_INVALID = pb.RejectedRequest.INVALID
REJECT_INTERNAL = pb.RejectedRequest.INTERNAL

# Admin commands with a typed form (AdminCommand), and the arguments one may
# carry. Anything else on the control socket travels as a pickled Utility.
ADMIN_COMMANDS = {
    "start_profile": pb.AdminCommand.START_PROFILE,
    "stop_profile": pb.AdminCommand.STOP_PROFILE,
    "get_mtp_statistics": pb.AdminCommand.GET_MTP_STATISTICS,
    "get_cache_statistics": pb.AdminCommand.GET_CACHE_STATISTICS,
    "clear_kv_cache": pb.AdminCommand.CLEAR_KV_CACHE,
}
_ADMIN_NAMES = {kind: name for name, kind in ADMIN_COMMANDS.items()}
_ADMIN_ARGS = ("request_id", "timeout", "deadline")


def _optional(msg, name):
    """A proto3 optional field's value, or None when it is absent."""
    return getattr(msg, name) if msg.HasField(name) else None


# ------------------------------------------------------------ sampling -----


def _encode_sampling(params: SamplingParams, msg, max_tokens) -> None:
    for name in ("temperature", "top_k", "top_p"):
        value = getattr(params, name)
        if value is not None:
            setattr(msg, name, value)
    if max_tokens is not None:
        msg.max_tokens = max_tokens
    msg.ignore_eos = bool(params.ignore_eos)
    stops = params.stop_strings
    if isinstance(stops, str):
        msg.stop_strings.single = stops
    elif stops is not None:
        msg.stop_strings.many.SetInParent()
        msg.stop_strings.many.items.extend(stops)
    msg.n = params.n
    logprobs = params.logprobs
    if type(logprobs) is bool:
        msg.logprobs_flag = logprobs
    elif logprobs is not None:
        msg.logprobs_count = logprobs
    channel = params.required_channel
    if channel is not None:
        msg.required_channel.after.extend(channel.after)
        msg.required_channel.opening.extend(channel.opening)
        msg.required_channel.opens_at = channel.opens_at
        msg.required_channel.end_ids.extend(channel.end_ids)
    msg.top_logprobs = params.top_logprobs
    for name in ("json_schema", "structured_tool_format"):
        value = getattr(params, name)
        if value is not None:
            setattr(msg, name, value)
    msg.structured_reasoning_end = params.structured_reasoning_end
    msg.max_tokens_is_default = params.max_tokens_is_default


_SAMPLING_DEFAULTS = SamplingParams()


def _optional_or_default(msg, name):
    return getattr(msg if msg.HasField(name) else _SAMPLING_DEFAULTS, name)


def _decode_sampling(msg) -> SamplingParams:
    stops = msg.stop_strings.WhichOneof("value")
    if stops == "single":
        stop_strings = msg.stop_strings.single
    elif stops == "many":
        stop_strings = list(msg.stop_strings.many.items)
    else:
        stop_strings = None
    logprobs = msg.WhichOneof("logprobs")
    channel = None
    if msg.HasField("required_channel"):
        wire = msg.required_channel
        channel = RequiredChannel(
            after=tuple(wire.after),
            opening=tuple(wire.opening),
            opens_at=wire.opens_at,
            end_ids=tuple(wire.end_ids),
        )
    # An absent temperature / top_k / top_p is the default, not None (which
    # would reach the sampler as NaN); see SamplingParams in engine_v1.proto.
    return SamplingParams(
        temperature=_optional_or_default(msg, "temperature"),
        top_k=_optional_or_default(msg, "top_k"),
        top_p=_optional_or_default(msg, "top_p"),
        max_tokens=_optional(msg, "max_tokens"),
        ignore_eos=msg.ignore_eos,
        stop_strings=stop_strings,
        n=msg.n,
        logprobs=None if logprobs is None else getattr(msg, logprobs),
        required_channel=channel,
        top_logprobs=msg.top_logprobs,
        json_schema=_optional(msg, "json_schema"),
        structured_tool_format=_optional(msg, "structured_tool_format"),
        structured_reasoning_end=msg.structured_reasoning_end,
        max_tokens_is_default=msg.max_tokens_is_default,
    )


@lru_cache(maxsize=256)
def _sampling_from_bytes(data: bytes) -> SamplingParams:
    msg = pb.SamplingParams()
    msg.ParseFromString(data)
    return _decode_sampling(msg)


def _decode_sampling_cached(msg) -> tuple[SamplingParams, int | None]:
    """`_decode_sampling`, shared by every request whose params encode alike,
    and the request's own `max_tokens`.

    A Core reads a request's SamplingParams only while it builds the
    `Sequence`, which copies the fields out, so one decoded object can serve
    them all -- as n > 1 siblings always shared theirs. Requests in one
    deployment repeat a handful of parameter sets, and decoding one costs as
    much as the rest of the request's spec. `max_tokens` stays out of the
    key: it varies per request (clamped defaults, benchmarks) and would make
    every lookup miss. `build_sequence` sets it on the sequence.
    """
    max_tokens = _optional(msg, "max_tokens")
    msg.ClearField("max_tokens")
    params = _sampling_from_bytes(msg.SerializeToString(deterministic=True))
    return params, max_tokens


def _encode_chain(chain: ShippedChain, msg, writer: FrameWriter) -> None:
    msg.hash_block_size = chain.hash_block_size
    if chain.cache_seed != -1:
        msg.cache_seed = chain.cache_seed
    msg.num_prompt_tokens = chain.num_prompt_tokens
    msg.fingerprint = chain.fingerprint
    # Shared: n > 1 siblings carry one chain, which goes out once.
    writer.add(chain.hashes, msg.hashes, key=chain.hashes)


def _decode_chain(msg, reader: FrameReader) -> ShippedChain:
    return ShippedChain(
        msg.hash_block_size,
        msg.cache_seed if msg.HasField("cache_seed") else -1,
        msg.num_prompt_tokens,
        msg.fingerprint,
        reader.uint64_array(msg.hashes),
    )


# ------------------------------------------------------------ client side --


def encode_submit(requests: list[IngressRequest], *, max_tokens=None) -> list:
    """One SubmitBatch. `max_tokens`, when given, replaces every request's."""
    writer = FrameWriter()
    envelope = pb.Envelope()
    batch = envelope.submit
    batch.SetInParent()
    for req in requests:
        spec = batch.requests.add()
        spec.seq_id = req.id
        # Shared: n > 1 siblings carry one prompt, which goes out once.
        writer.add(
            req.prompt_token_ids, spec.prompt_token_ids, key=req.prompt_token_ids
        )
        _encode_sampling(
            req.sampling_params,
            spec.sampling,
            req.max_tokens if max_tokens is None else max_tokens,
        )
        for stop in req.stop_token_sequences:
            spec.stop_token_sequences.add().ids.extend(stop)
        if req.external_request_id is not None:
            spec.external_request_id = req.external_request_id
        if req.parent_request_id is not None:
            spec.parent_request_id = req.parent_request_id
        spec.sibling_index = req.sibling_index
        spec.needs_independent_noise = req.needs_independent_noise
        encode_optional_value(
            req.kv_transfer_params, spec, "kv_transfer_params", writer
        )
        encode_optional_value(req.multimodal_data, spec, "multimodal_data", writer)
        if req.cache_seed is not None:
            spec.cache_seed = req.cache_seed
        encode_optional_value(req.mrope_positions, spec, "mrope_positions", writer)
        spec.mrope_position_delta = req.mrope_position_delta
        if req.data_parallel_rank is not None:
            spec.data_parallel_rank = req.data_parallel_rank
        if req.dp_session_id is not None:
            spec.dp_session_id = req.dp_session_id
        if req.dp_parent_session_id is not None:
            spec.dp_parent_session_id = req.dp_parent_session_id
        spec.arrive_time = req.arrive_time
        if req.prompt_hash_chain is not None:
            _encode_chain(req.prompt_hash_chain, spec.prompt_hash_chain, writer)
        trace = tracing.wire_context(req)
        if trace is not None:
            carrier, model, request_id = trace
            spec.trace.carrier.update(carrier)
            spec.trace.model = model
            if request_id is not None:
                spec.trace.request_id = request_id
    return writer.frames(envelope)


def encode_abort(seq_ids) -> list:
    envelope = pb.Envelope()
    envelope.abort.seq_ids.extend(seq_ids)
    return FrameWriter().frames(envelope)


def encode_command(payload: dict) -> list:
    """A control-socket command; `payload` is the dict the Core's handler
    receives. Typed (AdminCommand) when it is one of `ADMIN_COMMANDS` with
    only the arguments that form carries, a pickled Utility otherwise."""
    kind = ADMIN_COMMANDS.get(payload.get("cmd"))
    if kind is not None and _fits_admin_command(payload):
        return _encode_admin(kind, payload)
    return encode_utility(payload)


def _is_number(value) -> bool:
    return type(value) in (int, float)


def _fits_admin_command(payload: dict) -> bool:
    request_id = payload.get("request_id")
    return (
        set(payload) - {"cmd"} <= set(_ADMIN_ARGS)
        and (request_id is None or isinstance(request_id, str))
        and all(
            _is_number(payload[name])
            for name in ("timeout", "deadline")
            if payload.get(name) is not None
        )
    )


def _encode_admin(kind, payload: dict) -> list:
    envelope = pb.Envelope()
    msg = envelope.admin
    msg.kind = kind
    for name in _ADMIN_ARGS:
        if payload.get(name) is not None:
            setattr(msg, name, payload[name])
    return FrameWriter().frames(envelope)


def _decode_admin(msg) -> tuple[str, dict]:
    name = _ADMIN_NAMES.get(msg.kind)
    if name is None:
        raise ProtocolError(f"unknown admin command {msg.kind}")
    # The dict CoreManager's broadcast builds, keys in its order.
    payload = {"cmd": name}
    for field in _ADMIN_ARGS:
        if msg.HasField(field):
            payload[field] = getattr(msg, field)
    return name, payload


def encode_utility(payload: dict) -> list:
    """A Python-only admin command; `payload` is the dict the Core's handler
    receives, pickled into a frame."""
    writer = FrameWriter()
    envelope = pb.Envelope()
    envelope.utility.cmd = str(payload.get("cmd", ""))
    request_id = payload.get("request_id")
    if isinstance(request_id, str):
        envelope.utility.request_id = request_id
    writer.add(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        envelope.utility.payload,
    )
    return writer.frames(envelope)


def encode_shutdown() -> list:
    envelope = pb.Envelope()
    envelope.shutdown.SetInParent()
    return FrameWriter().frames(envelope)


def _decode_completion(msg, reader: FrameReader) -> CompletionResult:
    return CompletionResult(
        id=msg.seq_id,
        completion_token_ids=reader.int32_array(msg.completion_token_ids),
        logprobs=reader.float64_array(msg.logprobs),
        return_logprobs=msg.return_logprobs,
        num_prompt_tokens=msg.num_prompt_tokens,
        num_cached_tokens=msg.num_cached_tokens,
        arrive_time=msg.arrive_time,
        first_token_time=msg.first_token_time,
        leave_time=msg.leave_time,
        leave_reason=msg.leave_reason,
        kv_transfer_params_output=decode_optional_value(
            msg, "kv_transfer_params_output", reader
        ),
        external_request_id=_optional(msg, "external_request_id"),
        parent_request_id=_optional(msg, "parent_request_id"),
        sibling_index=msg.sibling_index,
    )


def _decode_stream(msg, reader: FrameReader) -> list[tuple[int, RequestOutput]]:
    counts = list(msg.token_counts)
    tokens = list(msg.tokens)
    if len(counts) != len(msg.seq_ids) or sum(counts) != len(tokens):
        raise ProtocolError("stream columns disagree in length")
    reasons = dict(msg.finish_reasons)
    kv_outputs = {
        i: decode_value(value, reader)
        for i, value in msg.kv_transfer_params_outputs.items()
    }
    logprobs = _decode_logprobs(msg, reader)
    out = []
    start = 0
    # A streamed output's request id is its sequence id (the Scheduler sets
    # `request_id=seq.id`), so the wire carries it once.
    for i, (seq_id, count, finished, cached) in enumerate(
        zip(
            list(msg.seq_ids),
            counts,
            list(msg.finished),
            list(msg.num_cached_tokens),
            strict=True,
        )
    ):
        end = start + count
        out.append(
            (
                seq_id,
                # Positional, in field order: this runs per sequence per step.
                RequestOutput(
                    seq_id,
                    tokens[start:end],
                    finished,
                    reasons.get(i),
                    kv_outputs.get(i),
                    cached,
                    logprobs.get(i),
                ),
            )
        )
        start = end
    return out


def decode_core_message(frames: list):
    """`(kind, data)` for one message from an EngineCore.

    ADD carries finished sequences as `CompletionResult`s, a rejected admission
    included (its `leave_reason` starts with "rejected"); STREAM carries
    `(seq_id, RequestOutput)` pairs; SHUTDOWN means the Core is gone.
    """
    envelope = decode_envelope(frames)
    reader = FrameReader(frames)
    body = envelope.WhichOneof("body")
    if body == "completions":
        return EngineCoreRequestType.ADD, [
            _decode_completion(item, reader) for item in envelope.completions.items
        ]
    if body == "stream":
        return EngineCoreRequestType.STREAM, _decode_stream(envelope.stream, reader)
    if body == "rejected":
        return EngineCoreRequestType.ADD, [
            CompletionResult(
                id=item.seq_id, leave_reason=f"{REJECTED_REASON}: {item.reason}"
            )
            for item in envelope.rejected.items
        ]
    if body == "metrics":
        return EngineCoreRequestType.METRICS, decode_optional_value(
            envelope.metrics, "snapshot", reader
        )
    if body == "utility_reply":
        return EngineCoreRequestType.UTILITY_RESPONSE, pickle.loads(
            reader.view(envelope.utility_reply.payload)
        )
    if body == "admin_reply":
        return EngineCoreRequestType.UTILITY_RESPONSE, _decode_admin_reply(
            envelope.admin_reply, reader
        )
    if body == "ready":
        ready = envelope.ready
        return EngineCoreRequestType.READY, (
            {"max_pool_tokens": ready.max_pool_tokens}
            if ready.HasField("max_pool_tokens")
            else None
        )
    if body == "stopped":
        return EngineCoreRequestType.SHUTDOWN, None
    raise ProtocolError(f"unexpected message from an EngineCore: {body}")


# ------------------------------------------------------------ core side ----


class AdmissionSpec:
    """A decoded RequestSpec: the `Sequence` constructor's arguments, plus
    what is set on a sequence after it is built."""

    __slots__ = (
        "arrive_time",
        "cache_seed",
        "kwargs",
        "max_tokens",
        "prompt_hash_chain",
        "seq_id",
        "trace",
    )

    def __init__(
        self,
        seq_id,
        arrive_time,
        cache_seed,
        kwargs,
        prompt_hash_chain=None,
        trace=None,
        max_tokens=None,
    ):
        self.seq_id = seq_id
        self.arrive_time = arrive_time
        self.cache_seed = cache_seed
        self.kwargs = kwargs
        # The request's own cap; the shared SamplingParams in `kwargs` has none.
        self.max_tokens = max_tokens
        self.prompt_hash_chain = prompt_hash_chain
        # `(carrier, model, request_id)`, as `tracing.wire_context` gave it.
        self.trace = trace


def _decode_spec(msg, reader: FrameReader) -> AdmissionSpec:
    sampling_params, max_tokens = _decode_sampling_cached(msg.sampling)
    kwargs = {
        "token_ids": reader.int32_array(msg.prompt_token_ids),
        "sampling_params": sampling_params,
        "stop_token_sequences": [list(s.ids) for s in msg.stop_token_sequences],
        "kv_transfer_params": decode_optional_value(msg, "kv_transfer_params", reader),
        "needs_independent_noise": msg.needs_independent_noise,
        "parent_request_id": _optional(msg, "parent_request_id"),
        "sibling_index": msg.sibling_index,
        "request_id": _optional(msg, "external_request_id"),
        "multimodal_data": decode_optional_value(msg, "multimodal_data", reader),
        "mrope_positions": decode_optional_value(msg, "mrope_positions", reader),
        "mrope_position_delta": msg.mrope_position_delta,
        "data_parallel_rank": _optional(msg, "data_parallel_rank"),
        "dp_session_id": _optional(msg, "dp_session_id"),
        "dp_parent_session_id": _optional(msg, "dp_parent_session_id"),
    }
    trace = None
    if msg.HasField("trace"):
        trace = (
            dict(msg.trace.carrier),
            msg.trace.model,
            _optional(msg.trace, "request_id"),
        )
    return AdmissionSpec(
        seq_id=msg.seq_id,
        arrive_time=msg.arrive_time,
        cache_seed=_optional(msg, "cache_seed"),
        kwargs=kwargs,
        prompt_hash_chain=(
            _decode_chain(msg.prompt_hash_chain, reader)
            if msg.HasField("prompt_hash_chain")
            else None
        ),
        trace=trace,
        max_tokens=max_tokens,
    )


def decode_client_message(frames: list):
    """`(kind, data, failures)` for one message from a client.

    ADD's data is the `AdmissionSpec`s that decoded; `failures` lists
    `(seq_id, reason, code)` for those that did not, each of which the Core
    must reject so its client is not left waiting. UTILITY's data is `(cmd,
    payload)`, from a Utility or an AdminCommand alike; ABORT's is the
    sequence ids.
    """
    envelope = decode_envelope(frames)
    reader = FrameReader(frames)
    body = envelope.WhichOneof("body")
    if body == "submit":
        specs, failures = [], []
        for msg in envelope.submit.requests:
            try:
                specs.append(_decode_spec(msg, reader))
            except Exception as exc:  # noqa: BLE001 - reported per request
                failures.append(
                    (
                        msg.seq_id,
                        f"{type(exc).__name__}: {exc}",
                        REJECT_INVALID,
                    )
                )
        return EngineCoreRequestType.ADD, specs, failures
    if body == "utility":
        payload = pickle.loads(reader.view(envelope.utility.payload))
        return EngineCoreRequestType.UTILITY, (envelope.utility.cmd, payload), []
    if body == "admin":
        return EngineCoreRequestType.UTILITY, _decode_admin(envelope.admin), []
    if body == "abort":
        return EngineCoreRequestType.ABORT, list(envelope.abort.seq_ids), []
    if body == "shutdown":
        return EngineCoreRequestType.SHUTDOWN, None, []
    raise ProtocolError(f"unexpected message from a client: {body}")


class AdmissionContext:
    """How a Core turns a decoded request into its `Sequence`.

    Holds what the Core derives from its own config -- the client used to
    derive the same from its copy of it -- and fills in what a client may
    leave out (the REQUESTS section of engine_v1.proto). Built once per Core,
    at startup.
    """

    def __init__(self, config):
        speculative = getattr(config, "speculative_config", None)
        self.config = config
        self.block_size = config.kv_cache_block_size
        self.num_draft_tokens = (
            speculative.num_speculative_tokens if speculative is not None else 0
        )
        self.has_per_req_cache = (
            config.hf_config.model_type in PER_REQ_CACHE_MODEL_TYPES
        )
        self.validate_request = getattr(config.hf_config, "validate_request", None)
        self.enable_dp_attention = bool(getattr(config, "enable_dp_attention", False))


def build_sequence(spec: AdmissionSpec, context: AdmissionContext):
    """The `Sequence` the client would have built for `spec`.

    Raises what the frontend would have raised building it, plus what the
    model's own request validation refuses.
    """
    kwargs = dict(spec.kwargs)
    multimodal_data = kwargs["multimodal_data"]
    if context.validate_request is not None:
        context.validate_request(
            num_draft_tokens=context.num_draft_tokens,
            multimodal_data=multimodal_data,
            enable_dp_attention=context.enable_dp_attention,
        )
    if multimodal_data is not None and kwargs["mrope_positions"] is None:
        # A client that cannot run the model's MRoPE code leaves them out.
        from atom.multimodal.registry import get_mrope_input_positions

        (
            kwargs["mrope_positions"],
            kwargs["mrope_position_delta"],
        ) = get_mrope_input_positions(
            context.config, list(kwargs["token_ids"]), multimodal_data
        )
    # The client hashed the media already; hand the seed to the constructor
    # the way it reads one, then take it back out, so the sequence's
    # multimodal_data is the dict the client sent.
    lend_seed = (
        spec.cache_seed is not None
        and multimodal_data is not None
        and "cache_seed" not in multimodal_data
    )
    if lend_seed:
        multimodal_data["cache_seed"] = spec.cache_seed
    try:
        seq = Sequence(
            block_size=context.block_size,
            num_draft_tokens=context.num_draft_tokens,
            has_per_req_cache=context.has_per_req_cache,
            id=spec.seq_id,
            **kwargs,
        )
    finally:
        if lend_seed:
            del multimodal_data["cache_seed"]
    # `Sequence` reads a falsy id as "draw one"; 0 is a real client id.
    seq.id = spec.seq_id
    seq.max_tokens = spec.max_tokens
    seq.arrive_time = spec.arrive_time
    # The BlockManager checks a shipped chain before it trusts it.
    seq.prompt_hash_chain = spec.prompt_hash_chain
    if spec.trace is not None:
        tracing.adopt_context(seq, *spec.trace)
    return seq


def rejection_code(exc: BaseException) -> int:
    """INVALID for what a bad request raises, INTERNAL for anything else."""
    if isinstance(exc, (ValueError, TypeError, IndexError, KeyError, OverflowError)):
        return REJECT_INVALID
    return REJECT_INTERNAL


def _encode_completion(seq, msg, writer: FrameWriter) -> None:
    msg.seq_id = seq.id
    writer.add(seq.completion_token_ids, msg.completion_token_ids)
    # A copy, like the token slice above: the frame outlives this call.
    writer.add(seq.logprobs[:], msg.logprobs)
    msg.return_logprobs = seq.return_logprobs
    msg.num_prompt_tokens = seq.num_prompt_tokens
    msg.num_cached_tokens = seq.num_cached_tokens
    msg.arrive_time = seq.arrive_time
    msg.first_token_time = seq.first_token_time
    msg.leave_time = seq.leave_time
    msg.leave_reason = seq.leave_reason or ""
    encode_optional_value(
        seq.kv_transfer_params_output, msg, "kv_transfer_params_output", writer
    )
    if seq.external_request_id is not None:
        msg.external_request_id = seq.external_request_id
    if seq.parent_request_id is not None:
        msg.parent_request_id = seq.parent_request_id
    msg.sibling_index = seq.sibling_index


def encode_completions(seqs) -> list:
    writer = FrameWriter()
    envelope = pb.Envelope()
    envelope.completions.SetInParent()
    for seq in seqs:
        _encode_completion(seq, envelope.completions.items.add(), writer)
    return writer.frames(envelope)


def encode_stream(outputs) -> list:
    """One step's `(seq_id, RequestOutput)` pairs, as columns."""
    writer = FrameWriter()
    envelope = pb.Envelope()
    msg = envelope.stream
    msg.SetInParent()
    results = [output for _, output in outputs]
    msg.seq_ids.extend([seq_id for seq_id, _ in outputs])
    msg.token_counts.extend([len(output.output_tokens) for output in results])
    msg.tokens.extend(chain.from_iterable(output.output_tokens for output in results))
    msg.finished.extend([output.finished for output in results])
    msg.num_cached_tokens.extend([output.num_cached_tokens or 0 for output in results])
    for i, output in enumerate(results):
        if output.finish_reason is not None:
            msg.finish_reasons[i] = output.finish_reason
        if output.kv_transfer_params_output is not None:
            try:
                encode_value(
                    output.kv_transfer_params_output,
                    msg.kv_transfer_params_outputs[i],
                    writer,
                )
            except Exception:  # one entry must not drop the whole step
                # Every other sequence keeps this step's tokens; this one
                # still ends on its Completion, which carries the params too.
                del msg.kv_transfer_params_outputs[i]
                logger.exception(
                    "dropping seq %s's unencodable kv_transfer_params_output",
                    outputs[i][0],
                )
    _encode_logprobs(results, msg, writer)
    return writer.frames(envelope)


def _encode_logprobs(results, msg, writer: FrameWriter) -> None:
    """`StreamBatch`'s logprob columns for `results`' token logprobs."""
    entries, counts, widths, floats, ints = [], [], [], [], []
    for i, output in enumerate(results):
        records = output.logprobs
        if records is None:
            continue
        width = (len(records[0]) - 2) // 2 if records else 0
        entries.append(i)
        counts.append(len(records))
        widths.append(width)
        for record in records:
            if len(record) != 2 * width + 2:
                raise ProtocolError(
                    "token logprob records of one entry differ in width"
                )
            floats.append(record[0])
            floats.extend(record[3::2])
            ints.append(record[1])
            ints.extend(record[2::2])
    if entries:
        msg.logprob_entries.extend(entries)
        msg.logprob_counts.extend(counts)
        msg.logprob_widths.extend(widths)
        # Data frames, not packed fields: a reader then views them in place
        # instead of materializing one Python number per value.
        writer.add(array.array("d", floats), msg.logprob_floats)
        writer.add(array.array("i", ints), msg.logprob_ints)


def _decode_logprobs(msg, reader: FrameReader) -> dict:
    """Entry index -> its records, from `StreamBatch`'s logprob columns."""
    counts = list(msg.logprob_counts)
    widths = list(msg.logprob_widths)
    if len(counts) != len(msg.logprob_entries) or len(widths) != len(counts):
        raise ProtocolError("logprob columns disagree in length")
    if not counts:
        return {}
    total = sum(count * (width + 1) for count, width in zip(counts, widths))
    floats = np.frombuffer(reader.view(msg.logprob_floats), dtype="<f8")
    ints = np.frombuffer(reader.view(msg.logprob_ints), dtype="<i4")
    if total != len(floats) or total != len(ints):
        raise ProtocolError("logprob columns disagree in length")
    # Entries without records (a finished one's `[]`) carry no width.
    record_widths = {width for width, count in zip(widths, counts) if count}
    if len(record_widths) <= 1:
        # One step's records share their width: build them all at once, the
        # way the runner unpacks them (`token_logprobs.unpack_token_logprobs`).
        width = record_widths.pop() if record_widths else 0
        records = _logprob_records(floats, ints, width)
    else:
        records = _logprob_records_by_entry(floats, ints, counts, widths)
    out = {}
    pos = 0
    for entry, count in zip(msg.logprob_entries, counts):
        out[entry] = records[pos : pos + count]
        pos += count
    return out


@cache
def _logprob_record_dtype(width: int) -> np.dtype:
    columns = [("logprob", "<f8"), ("rank", "<i4")]
    for i in range(width):
        columns += [(f"id{i}", "<i4"), (f"value{i}", "<f8")]
    return np.dtype(columns)


def _logprob_records(floats, ints, width: int) -> list:
    """Every record, all `width` wide, as tuples built in one `tolist`."""
    floats = floats.reshape(-1, width + 1)
    ints = ints.reshape(-1, width + 1)
    records = np.empty(len(floats), dtype=_logprob_record_dtype(width))
    records["logprob"] = floats[:, 0]
    records["rank"] = ints[:, 0]
    for i in range(width):
        records[f"id{i}"] = ints[:, i + 1]
        records[f"value{i}"] = floats[:, i + 1]
    return records.tolist()


def _logprob_records_by_entry(floats, ints, counts, widths) -> list:
    floats, ints = floats.tolist(), ints.tolist()
    records = []
    pos = 0
    for count, width in zip(counts, widths):
        for _ in range(count):
            end = pos + width + 1
            f, n = floats[pos:end], ints[pos:end]
            records.append((f[0], n[0], *chain.from_iterable(zip(n[1:], f[1:]))))
            pos = end
    return records


def encode_rejected(rejections) -> list:
    """`rejections`: `(seq_id, reason, code)`, code a RejectedRequest.Code."""
    envelope = pb.Envelope()
    envelope.rejected.SetInParent()
    for seq_id, reason, code in rejections:
        item = envelope.rejected.items.add()
        item.seq_id = seq_id
        item.reason = reason
        item.code = code
    return FrameWriter().frames(envelope)


def encode_ready(payload) -> list:
    """EngineCore's READY payload: None, or a dict of the facts in `Ready`."""
    envelope = pb.Envelope()
    envelope.ready.SetInParent()
    if payload is not None:
        unknown = set(payload) - {"max_pool_tokens"}
        if unknown:
            # A new startup fact needs a Ready field, not a silent drop.
            raise ProtocolError(f"Ready has no field for {sorted(unknown)}")
        if payload.get("max_pool_tokens") is not None:
            envelope.ready.max_pool_tokens = payload["max_pool_tokens"]
    return FrameWriter().frames(envelope)


def encode_metrics(snapshot) -> list:
    writer = FrameWriter()
    envelope = pb.Envelope()
    envelope.metrics.SetInParent()
    encode_optional_value(snapshot, envelope.metrics, "snapshot", writer)
    return writer.frames(envelope)


def encode_utility_reply(reply) -> list:
    """A utility handler's reply: an AdminReply when it answers a typed admin
    command in the shape those take, a pickled UtilityReply otherwise."""
    kind = ADMIN_COMMANDS.get(reply.get("cmd")) if isinstance(reply, dict) else None
    if kind is not None and _fits_admin_reply(reply):
        return _encode_admin_reply(kind, reply)
    writer = FrameWriter()
    envelope = pb.Envelope()
    request_id = reply.get("request_id") if isinstance(reply, dict) else None
    if isinstance(request_id, str):
        envelope.utility_reply.request_id = request_id
    writer.add(
        pickle.dumps(reply, protocol=pickle.HIGHEST_PROTOCOL),
        envelope.utility_reply.payload,
    )
    return writer.frames(envelope)


def _fits_admin_reply(reply: dict) -> bool:
    request_id = reply.get("request_id")
    return (
        set(reply) <= {"cmd", "result", "error", "request_id"}
        and ("result" in reply) != ("error" in reply)
        and isinstance(reply.get("error", ""), str)
        and (request_id is None or isinstance(request_id, str))
    )


def _encode_admin_reply(kind, reply: dict) -> list:
    writer = FrameWriter()
    envelope = pb.Envelope()
    msg = envelope.admin_reply
    msg.kind = kind
    if reply.get("request_id") is not None:
        msg.request_id = reply["request_id"]
    if "error" in reply:
        msg.error = reply["error"]
    else:
        encode_value(reply["result"], msg.result, writer)
    return writer.frames(envelope)


def _decode_admin_reply(msg, reader: FrameReader) -> dict:
    name = _ADMIN_NAMES.get(msg.kind)
    if name is None:
        raise ProtocolError(f"unknown admin command {msg.kind}")
    # The dict the handler built, keys in its order.
    reply = {"cmd": name}
    if msg.HasField("error"):
        reply["error"] = msg.error
    else:
        reply["result"] = decode_value(msg.result, reader)
    if msg.HasField("request_id"):
        reply["request_id"] = msg.request_id
    return reply


def encode_stopped() -> list:
    envelope = pb.Envelope()
    envelope.stopped.SetInParent()
    return FrameWriter().frames(envelope)
