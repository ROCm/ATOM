# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Codec for the Session's engine_v1 messages, each an Envelope then the frames its
BufferRefs name: what the server reads from its client and writes back."""

from dataclasses import dataclass, field
from itertools import chain
from typing import NamedTuple

import numpy as np
from google.protobuf.message import DecodeError

from atom.entrypoints.openai.protocol import DEFAULT_MAX_TOKENS
from atom.model_engine.request import RequestOutput
from atom.model_engine.zmq_protocol import engine_v1_pb2 as pb
from atom.sampling_params import RequiredChannel, SamplingParams

PROTOCOL_VERSION = 1  # engine_v1's: atom_engine::wire::PROTOCOL_VERSION in Rust
INVALID = pb.RejectedRequest.INVALID
_DEFAULTED = ("temperature", "top_k", "top_p")  # absent: SamplingParams' defaults
_ROUTING = ("data_parallel_rank", "dp_session_id", "dp_parent_session_id")
# RequestSpec fields the Session does not read: it stamps arrival on its own clock,
# hashes the prompt itself, takes no siblings and does not trace.
_UNREAD = ("arrive_time", "prompt_hash_chain", "parent_request_id", "trace")
_COMPLETION_SCALARS = (
    "leave_reason",
    "num_prompt_tokens",
    "num_cached_tokens",
    "arrive_time",
    "first_token_time",
    "leave_time",
)


class FrameError(ValueError):
    """A message, or one request in it, that this codec cannot read."""


@dataclass
class SubmitRequest:
    """One RequestSpec, as `InputOutputProcessor.preprocess` takes it."""

    seq_id: int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    stop_token_sequences: list[list[int]] = field(default_factory=list)
    kv_transfer_params: dict | None = None
    request_id: str | None = None
    data_parallel_rank: int | None = None
    dp_session_id: str | None = None
    dp_parent_session_id: str | None = None


@dataclass
class Completion:
    """A finished request's Completions item: ids, usage, times and leave reason."""

    seq_id: int
    completion_token_ids: list[int]
    leave_reason: str
    num_prompt_tokens: int = 0
    num_cached_tokens: int = 0
    arrive_time: float = 0.0
    first_token_time: float = 0.0
    leave_time: float = 0.0
    logprobs: list[float] | None = None
    kv_transfer_params_output: dict | None = None
    external_request_id: str | None = None


class Rejection(NamedTuple):
    seq_id: int
    reason: str
    code: int  # a RejectedRequest.Code


def decode_client(frames) -> tuple[str, list]:
    """("submit", a SubmitRequest or an INVALID Rejection per request) or ("abort",
    seq ids). Raises FrameError for any other message, or one of another version."""
    envelope = _envelope(frames)
    body = envelope.WhichOneof("body")
    if body == "abort":
        return body, list(envelope.abort.seq_ids)
    if body != "submit":
        raise FrameError(f"a Session client sends SubmitBatch or Abort, not {body}")
    items = []
    for spec in envelope.submit.requests:
        try:
            items.append(_request(spec, frames))
        except ValueError as e:  # FrameError, or a value SamplingParams refuses
            items.append(Rejection(spec.seq_id, str(e), INVALID))
    return body, items


def encode_ready(max_pool_tokens: int | None = None) -> list[bytes]:
    return _finish(pb.Envelope(ready=pb.Ready(max_pool_tokens=max_pool_tokens)), [])


def encode_stream(entries: list[tuple[int, RequestOutput]]) -> list[bytes]:
    """One StreamBatch: a step's (seq_id, RequestOutput) entries, column by column."""
    envelope, data = pb.Envelope(stream=pb.StreamBatch()), []
    msg = envelope.stream
    outputs = [output for _, output in entries]
    msg.seq_ids.extend([seq_id for seq_id, _ in entries])
    msg.token_counts.extend([len(o.output_tokens) for o in outputs])
    msg.tokens.extend(chain.from_iterable(o.output_tokens for o in outputs))
    msg.finished.extend([o.finished for o in outputs])
    msg.num_cached_tokens.extend([o.num_cached_tokens for o in outputs])
    floats, ints = [], []
    for i, o in enumerate(outputs):
        if o.finish_reason is not None:
            msg.finish_reasons[i] = o.finish_reason
        if o.kv_transfer_params_output is not None:
            _put_value(o.kv_transfer_params_output, msg.kv_transfer_params_outputs[i])
        if o.logprobs is not None:
            _put_records(i, o.logprobs, msg, floats, ints)
    if msg.logprob_entries:
        _attach(data, floats, "<f8", msg.logprob_floats)
        _attach(data, ints, "<i4", msg.logprob_ints)
    return _finish(envelope, data)


def encode_completions(items: list[Completion]) -> list[bytes]:
    envelope, data = pb.Envelope(completions=pb.Completions()), []
    for c in items:
        msg = envelope.completions.items.add(
            seq_id=c.seq_id,
            external_request_id=c.external_request_id,
            **{name: getattr(c, name) for name in _COMPLETION_SCALARS},
        )
        _attach(data, c.completion_token_ids, "<i4", msg.completion_token_ids)
        if c.logprobs is not None:
            _attach(data, c.logprobs, "<f8", msg.logprobs)
        if c.kv_transfer_params_output is not None:
            _put_value(c.kv_transfer_params_output, msg.kv_transfer_params_output)
    return _finish(envelope, data)


def encode_rejected(items: list[Rejection]) -> list[bytes]:
    rejected = [pb.RejectedRequest(seq_id=s, reason=r, code=c) for s, r, c in items]
    return _finish(pb.Envelope(rejected=pb.Rejected(items=rejected)), [])


# ---------------------------------------------------------------- frames ---


def _envelope(frames) -> pb.Envelope:
    if not frames:
        raise FrameError("a message needs its Envelope frame")
    try:
        envelope = pb.Envelope.FromString(frames[0])
    except DecodeError as e:
        raise FrameError(f"frame 0 is not an Envelope: {e}") from e
    if envelope.protocol_version != PROTOCOL_VERSION:
        raise FrameError(
            f"protocol version {envelope.protocol_version}, "
            f"expected {PROTOCOL_VERSION}"
        )
    return envelope


def _finish(envelope: pb.Envelope, data: list[bytes]) -> list[bytes]:
    envelope.protocol_version = PROTOCOL_VERSION
    return [envelope.SerializeToString(), *data]


def _attach(data: list[bytes], values, dtype: str, ref: pb.BufferRef) -> None:
    """`values` as a little-endian data frame of their own, which `ref` names."""
    data.append(np.asarray(values, dtype).tobytes())
    ref.frame, ref.nbytes = len(data), len(data[-1])


def _read(frames, parent, name: str, dtype: str) -> np.ndarray:
    """The little-endian values BufferRef `parent.name` names; none when absent."""
    if not parent.HasField(name):
        return np.empty(0, dtype)
    ref, size = getattr(parent, name), np.dtype(dtype).itemsize
    if not 0 < ref.frame < len(frames):
        raise FrameError(f"{name} names frame {ref.frame} of {len(frames)}")
    if ref.offset + ref.nbytes > len(frames[ref.frame]) or ref.nbytes % size:
        raise FrameError(f"{name} is not whole {dtype} values inside its frame")
    return np.frombuffer(frames[ref.frame], dtype, ref.nbytes // size, ref.offset)


def _is_set(msg, name: str) -> bool:
    """Whether field `name` is set: present, or not its default where it has no
    presence."""
    if msg.DESCRIPTOR.fields_by_name[name].has_presence:
        return msg.HasField(name)
    return bool(getattr(msg, name))


def _optional(msg, name: str):
    return getattr(msg, name) if msg.HasField(name) else None


# -------------------------------------------------------------- requests ---


def _request(spec: pb.RequestSpec, frames) -> SubmitRequest:
    media = ("multimodal_data", "mrope_positions", "cache_seed")
    if spec.mrope_position_delta or any(spec.HasField(name) for name in media):
        raise FrameError("the Session takes text prompts; this request carries media")
    if spec.needs_independent_noise or spec.sibling_index:
        raise FrameError("the Session takes n=1; this request is one sibling of n")
    unread = [name for name in _UNREAD if _is_set(spec, name)]
    if unread:
        raise FrameError(f"the Session does not read {', '.join(unread)}")
    prompt = _read(frames, spec, "prompt_token_ids", "<i4")
    if (prompt < 0).any():
        raise FrameError("prompt token ids must be non-negative")
    kv = _value_field(spec, "kv_transfer_params")
    if kv is not None and not isinstance(kv, dict):
        raise FrameError("kv_transfer_params must be a map")
    return SubmitRequest(
        seq_id=spec.seq_id,
        prompt_token_ids=prompt.tolist(),
        sampling_params=_sampling(spec.sampling),
        # An empty one would match at once; the HTTP path drops it as well.
        stop_token_sequences=[list(s.ids) for s in spec.stop_token_sequences if s.ids],
        kv_transfer_params=kv,
        request_id=_optional(spec, "external_request_id"),
        **{name: _optional(spec, name) for name in _ROUTING},
    )


def _sampling(msg: pb.SamplingParams) -> SamplingParams:
    """Absent fields read as the HTTP routes fill omitted ones (max_tokens their
    default, logprobs False), and n below 1 as 1, as those routes coerce it."""
    stops, which = msg.stop_strings, msg.stop_strings.WhichOneof("value")
    stop_strings = None
    if which == "single":
        stop_strings = stops.single
    elif which == "many":
        stop_strings = list(stops.many.items)
    channel = None
    if msg.HasField("required_channel"):
        c = msg.required_channel
        channel = RequiredChannel(
            tuple(c.after), tuple(c.opening), c.opens_at, tuple(c.end_ids)
        )
    logprobs = msg.WhichOneof("logprobs")
    has_max = msg.HasField("max_tokens")
    return SamplingParams(
        **{name: getattr(msg, name) for name in _DEFAULTED if msg.HasField(name)},
        max_tokens=msg.max_tokens if has_max else DEFAULT_MAX_TOKENS,
        max_tokens_is_default=msg.max_tokens_is_default or not has_max,
        n=max(msg.n, 1),
        ignore_eos=msg.ignore_eos,
        stop_strings=stop_strings,
        logprobs=False if logprobs is None else getattr(msg, logprobs),
        required_channel=channel,
        top_logprobs=msg.top_logprobs,
        json_schema=_optional(msg, "json_schema"),
        structured_tool_format=_optional(msg, "structured_tool_format"),
        structured_reasoning_end=msg.structured_reasoning_end,
    )


# --------------------------------------------------------------- outputs ---


def _put_records(entry: int, records, msg: pb.StreamBatch, floats, ints) -> None:
    """An entry's TokenLogprob records, (logprob, rank, id_0, logprob_0, ...)."""
    width = (len(records[0]) - 2) // 2 if records else 0
    for record in records:
        if len(record) != 2 * width + 2:
            raise FrameError(f"entry {entry}'s logprob records differ in width")
        floats += (record[0], *record[3::2])
        ints += (record[1], *record[2::2])
    msg.logprob_entries.append(entry)
    msg.logprob_counts.append(len(records))
    msg.logprob_widths.append(width)


# ---------------------------------------------------------------- values ---


def _put_value(value, msg: pb.Value) -> None:
    """A JSON-shaped value; tuples go as lists, as json.dumps writes them."""
    if value is None:
        msg.null_value = True
    elif isinstance(value, bool):
        msg.bool_value = value
    elif isinstance(value, int):
        msg.int_value = value
    elif isinstance(value, float):
        msg.float_value = value
    elif isinstance(value, str):
        msg.string_value = value
    elif isinstance(value, (list, tuple)):
        msg.list_value.SetInParent()
        for item in value:
            _put_value(item, msg.list_value.items.add())
    elif isinstance(value, dict) and all(isinstance(key, str) for key in value):
        msg.map_value.SetInParent()
        for key, item in value.items():
            entry = msg.map_value.entries.add()
            entry.key.string_value = key
            _put_value(item, entry.value)
    else:
        raise TypeError(f"{value!r} is not JSON-shaped")


def _get_value(msg: pb.Value):
    """`msg`'s JSON-shaped value; bytes, tuples, tensors and pickles are refused."""
    kind = msg.WhichOneof("kind")
    if kind == "null_value":
        return None
    if kind in ("bool_value", "int_value", "float_value", "string_value"):
        return getattr(msg, kind)
    if kind == "list_value":
        return [_get_value(item) for item in msg.list_value.items]
    if kind == "map_value":
        entries = msg.map_value.entries
        if all(e.key.HasField("string_value") for e in entries):
            return {e.key.string_value: _get_value(e.value) for e in entries}
    raise FrameError(f"a Value of kind {kind} is not JSON-shaped")


def _value_field(parent, name: str):
    return _get_value(getattr(parent, name)) if parent.HasField(name) else None
