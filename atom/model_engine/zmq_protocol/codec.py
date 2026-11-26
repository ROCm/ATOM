# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Encode and decode the CoreManager <-> EngineCore messages (`messages`).

Each side calls these where it called ``pickle.dumps((type, payload))`` and
``pickle.loads(frame)``, with the same ``(type, payload)`` it always had:

- the client: `encode_input`, `decode_output`;
- the EngineCore: `decode_input`, `encode_output`.

What differs from pickle is what crosses for a sequence. The client sends a
`SubmitItem` (the Core builds the `Sequence`, `admission.build_sequence`), and
the Core sends a finished sequence back as a `Completion`, which reads like
the finished `Sequence` did. Everything else is the payload the engine
pickled before, either typed (`Ready`, STREAM) or pickled as it was.

One item that will not encode does not fail its batch: a request the client
cannot encode is sent refused (`SubmitItem.refused`) and comes back rejected;
a step output or completion the Core cannot encode is sent without what did
not encode, or as a rejected completion.
"""

import array
import io
import logging
import pickle

import msgspec
import numpy as np

from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.prefix_hash import ShippedChain
from atom.model_engine.request import RequestOutput
from atom.model_engine.zmq_protocol import messages as m
from atom.observability import tracing
from atom.sampling_params import RequiredChannel, SamplingParams

logger = logging.getLogger("atom")

_PICKLE_EXT = 1  # a value msgpack cannot hold, pickled (Python only)


def _enc_hook(obj):
    return msgspec.msgpack.Ext(
        _PICKLE_EXT, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    )


def _ext_hook(code: int, data: memoryview):
    if code == _PICKLE_EXT:
        return pickle.loads(data)
    raise msgspec.DecodeError(f"unknown msgpack extension {code}")


_encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)
_head = msgspec.msgpack.Decoder(tuple[int, msgspec.Raw])
_sampling = msgspec.msgpack.Decoder(m.Sampling)
_DEFAULT_SAMPLING = msgspec.Raw(_encoder.encode(m.Sampling()))


def _decoders(bodies) -> dict:
    return {
        m.KIND[t]: (t, msgspec.msgpack.Decoder(body, ext_hook=_ext_hook))
        for t, body in bodies.items()
    }


_input_decoders = _decoders(m.INPUT_BODY)
_output_decoders = _decoders(m.OUTPUT_BODY)


def _frame(request_type, body) -> bytes:
    return _encoder.encode((m.KIND[request_type], body))


def _pickled(payload) -> m.Pickled:
    return m.Pickled(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def _decode(buf, decoders):
    kind, raw = _head.decode(buf)
    entry = decoders.get(kind)
    if entry is None:
        raise msgspec.DecodeError(f"unexpected message kind {kind}")
    request_type, decoder = entry
    return request_type, decoder.decode(raw)


def _encode_items(request_type, wrap, items, fallback) -> bytes:
    """`wrap(items)` as one message. When that does not encode, each item
    that does not alone is replaced by `fallback(item, exc)`."""
    try:
        return _frame(request_type, wrap(items))
    except Exception:  # noqa: BLE001 - an item's; the loop below finds which
        logger.warning("a %s message does not encode; item by item", request_type)
    kept = []
    for item in items:
        try:
            _encoder.encode(item)
        except Exception as exc:  # this item alone
            logger.exception("seq %s does not encode", item.seq_id)
            item = fallback(item, exc)
        kept.append(item)
    return _frame(request_type, wrap(kept))


def _token_bytes(seq, start: int, end: int) -> bytes:
    """`seq.token_ids[start:end]` as int32 bytes, in one copy."""
    return memoryview(seq.token_ids)[start:end].tobytes()


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------------ client --


def _sampling_of(seq) -> m.Sampling:
    """The `SamplingParams` a `Sequence` was built from, out of the fields it
    copied (n is 1 per sequence; `logprobs` survives as a flag)."""
    channel = seq.required_channel
    return m.Sampling(
        temperature=seq.temperature,
        top_k=seq.top_k,
        top_p=seq.top_p,
        max_tokens=seq.max_tokens,
        ignore_eos=seq.ignore_eos,
        stop_strings=seq.stop_strings,
        logprobs=True if seq.return_logprobs else None,
        required_channel=(
            None
            if channel is None
            else m.RequiredChannel(
                channel.after, channel.opening, channel.opens_at, channel.end_ids
            )
        ),
        top_logprobs=seq.top_logprobs,
        json_schema=seq.json_schema,
        structured_tool_format=seq.structured_tool_format,
        structured_reasoning_end=seq.structured_reasoning_end,
    )


class _MediaPickler(pickle.Pickler):
    """Pickles a value with its CPU tensors' and numpy arrays' bytes left out,
    in `buffers` (one contiguous byte view each), where pickle would copy them
    into its stream -- a tensor through torch's own serialization."""

    def __init__(self, file, buffers: list):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._buffers = buffers
        # id -> (object, its id): one object twice is one buffer, and comes
        # back one object, as pickle's own memo would have it.
        self._seen: dict[int, tuple] = {}

    def persistent_id(self, obj):
        seen = self._seen.get(id(obj))
        if seen is not None:
            return seen[1]
        kind = type(obj)
        if kind is np.ndarray and not obj.dtype.hasobject:
            # Not `ascontiguousarray`, which makes a 0-d array 1-d.
            raw = obj if obj.flags.c_contiguous else obj.copy(order="C")
            tag = ("ndarray", raw.dtype.str, raw.shape)
            data = raw.reshape(-1).view(np.uint8)
        elif kind.__name__ == "Tensor" and kind.__module__ == "torch":
            if obj.device.type != "cpu":
                return None
            raw = obj.detach().contiguous()
            tag = ("tensor", str(raw.dtype).removeprefix("torch."), tuple(raw.shape))
            data = raw.reshape(-1).view(_torch().uint8).numpy()
        else:
            return None
        self._buffers.append(memoryview(data))
        pid = (*tag, len(self._buffers) - 1)
        self._seen[id(obj)] = (obj, pid)
        return pid


class _MediaUnpickler(pickle.Unpickler):
    def __init__(self, data, buffers):
        super().__init__(io.BytesIO(data))
        self._buffers = buffers
        self._loaded: dict[int, object] = {}

    def persistent_load(self, pid):
        kind, dtype, shape, index = pid
        if index in self._loaded:
            return self._loaded[index]
        if kind == "ndarray":
            out = np.empty(shape, dtype=np.dtype(dtype))
            view = out.reshape(-1).view(np.uint8)
        else:
            torch = _torch()
            out = torch.empty(shape, dtype=getattr(torch, dtype))
            view = out.reshape(-1).view(torch.uint8).numpy()
        view[:] = np.frombuffer(self._buffers[index], dtype=np.uint8)
        self._loaded[index] = out
        return out


def _torch():
    import torch

    return torch


def pickle_media(value) -> tuple[bytes, list[memoryview]]:
    """`value` pickled, and the bytes of the tensors and arrays in it."""
    buffers: list[memoryview] = []
    stream = io.BytesIO()
    _MediaPickler(stream, buffers).dump(value)
    return stream.getvalue(), buffers


def unpickle_media(data: bytes, buffers) -> object:
    return _MediaUnpickler(data, buffers or []).load()


def _optional_pickle(value) -> bytes | None:
    if value is None:
        return None
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _submit_item(seq) -> m.SubmitItem:
    chain = seq.prompt_hash_chain
    trace = tracing.wire_context(seq)
    multimodal = seq.multimodal_data
    media, media_buffers = None, None
    if multimodal is not None or seq.mrope_positions is not None:
        media, media_buffers = pickle_media((multimodal, seq.mrope_positions))
    return m.SubmitItem(
        seq_id=seq.id,
        prompt=_token_bytes(seq, 0, seq.num_prompt_tokens),
        sampling=msgspec.Raw(_encoder.encode(_sampling_of(seq))),
        stop_token_sequences=[s.tolist() for s in seq.stop_token_sequences],
        kv_transfer_params=_optional_pickle(seq.kv_transfer_params),
        multimodal=media,
        multimodal_buffers=media_buffers,
        mrope_position_delta=seq.mrope_position_delta,
        cache_seed=seq.cache_seed if multimodal is not None else None,
        needs_independent_noise=seq.needs_independent_noise,
        parent_request_id=seq.parent_request_id,
        sibling_index=seq.sibling_index,
        external_request_id=seq.external_request_id,
        data_parallel_rank=seq.data_parallel_rank,
        dp_session_id=seq.dp_session_id,
        dp_parent_session_id=seq.dp_parent_session_id,
        arrive_time=seq.arrive_time,
        prompt_hash_chain=(
            None
            if chain is None
            else m.PromptHashChain(
                chain.hash_block_size,
                chain.cache_seed,
                chain.num_prompt_tokens,
                chain.fingerprint,
                chain.hashes.tobytes(),
            )
        ),
        trace=None if trace is None else m.TraceContext(*trace),
        dp_est_cached_tokens=seq.dp_est_cached_tokens,
    )


def _refused(seq_id, prompt: bytes, arrive_time, exc) -> m.SubmitItem:
    return m.SubmitItem(
        seq_id=seq_id,
        prompt=prompt,
        sampling=_DEFAULT_SAMPLING,
        arrive_time=arrive_time,
        refused=f"request not encodable: {_reason(exc)}",
    )


def _refused_item(item: m.SubmitItem, exc: BaseException) -> m.SubmitItem:
    return _refused(item.seq_id, item.prompt, item.arrive_time, exc)


def encode_submit(seqs) -> tuple[bytes, set[int]]:
    """ADD for `seqs`, and the ids of those sent refused because they do not
    encode: their Core rejects them, so they end like any refused request."""
    refused: set[int] = set()
    items = []
    for seq in seqs:
        try:
            items.append(_submit_item(seq))
        except Exception as exc:  # this request alone
            logger.exception("seq %s does not encode", seq.id)
            refused.add(seq.id)
            prompt = _token_bytes(seq, 0, seq.num_prompt_tokens)
            items.append(_refused(seq.id, prompt, seq.arrive_time, exc))

    def refuse(item, exc):
        refused.add(item.seq_id)
        return _refused_item(item, exc)

    return _encode_items(EngineCoreRequestType.ADD, m.Submit, items, refuse), refused


def encode_input(request_type: EngineCoreRequestType, payload=None) -> bytes:
    """A client's message to an EngineCore: ADD with `Sequence`s, ABORT with
    their ids, SHUTDOWN, or a UTILITY / DP_PREFIX_CREDIT payload."""
    if request_type == EngineCoreRequestType.ADD:
        return encode_submit(payload)[0]
    if request_type == EngineCoreRequestType.ABORT:
        return _frame(request_type, m.Abort(list(payload)))
    if request_type == EngineCoreRequestType.SHUTDOWN:
        return _frame(request_type, m.Empty())
    return _frame(request_type, _pickled(payload))


class StreamOutputs(list):
    """A STREAM's `(seq_id, RequestOutput)` pairs.

    `cut_short`: the sequences this client ends here, rejected, while their
    Core still runs them -- their step's KV-transfer params did not
    unpickle. The client tells the Core to stop them.
    """

    cut_short: tuple = ()


def _loaded_step_params(item, failed: list):
    """A step's KV-transfer params, unpickled; None, with the reason in
    `failed`, when they do not load -- one step's params must not lose every
    other sequence's step."""
    try:
        return pickle.loads(item.kv_transfer_params_output)
    except Exception as exc:
        logger.exception(
            "seq %s's kv_transfer_params_output does not load", item.seq_id
        )
        failed.append((item.seq_id, exc))
        return None


def _stream_outputs(items) -> StreamOutputs:
    failed: list = []
    out = StreamOutputs(
        [
            (
                item.seq_id,
                RequestOutput(
                    item.seq_id,
                    item.output_tokens,
                    item.finished,
                    item.finish_reason,
                    (
                        None
                        if item.kv_transfer_params_output is None
                        else _loaded_step_params(item, failed)
                    ),
                    item.num_cached_tokens,
                    item.logprobs,
                ),
            )
            for item in items
        ]
    )
    if failed:
        reasons = dict(failed)
        cut_short = []
        for seq_id, output in out:
            exc = reasons.get(seq_id)
            if exc is None:
                continue
            if not output.finished:
                cut_short.append(seq_id)
            output.finished = True
            output.finish_reason = (
                f"rejected: kv_transfer_params_output not loadable: {_reason(exc)}"
            )
        out.cut_short = tuple(cut_short)
    return out


def decode_output(buf):
    """`(type, payload)` for an EngineCore's message, as the client always
    had it: ADD's finished sequences (`Completion`s), STREAM's
    `(seq_id, RequestOutput)` pairs, READY's dict (or None), SHUTDOWN's None,
    and the rest as the Core pickled them."""
    request_type, body = _decode(buf, _output_decoders)
    if request_type == EngineCoreRequestType.STREAM:
        return request_type, _stream_outputs(body.items)
    if request_type == EngineCoreRequestType.ADD:
        return request_type, body.items
    if request_type == EngineCoreRequestType.READY:
        if body.protocol_version != m.PROTOCOL_VERSION:
            raise msgspec.DecodeError(
                f"EngineCore speaks protocol {body.protocol_version}, "
                f"this client {m.PROTOCOL_VERSION}"
            )
        ready = {}
        if body.max_pool_tokens is not None:
            ready["max_pool_tokens"] = body.max_pool_tokens
        if body.weight_syncs_unavailable:
            ready["weight_syncs_unavailable"] = body.weight_syncs_unavailable
        return request_type, ready or None
    if request_type == EngineCoreRequestType.SHUTDOWN:
        return request_type, None
    return request_type, pickle.loads(body.data)


# -------------------------------------------------------------- EngineCore --


def decode_input(buf):
    """`(type, payload)` for a client's message: ADD's `SubmitItem`s (for
    `admission.build_sequence`), ABORT's seq ids, SHUTDOWN's None, and a
    UTILITY / DP_PREFIX_CREDIT payload as the client pickled it."""
    request_type, body = _decode(buf, _input_decoders)
    if request_type == EngineCoreRequestType.ADD:
        return request_type, body.items
    if request_type == EngineCoreRequestType.ABORT:
        return request_type, body.seq_ids
    if request_type == EngineCoreRequestType.SHUTDOWN:
        return request_type, None
    return request_type, pickle.loads(body.data)


def sampling_params(item: m.SubmitItem) -> SamplingParams:
    s = _sampling.decode(item.sampling)
    channel = s.required_channel
    return SamplingParams(
        temperature=s.temperature,
        top_k=s.top_k,
        top_p=s.top_p,
        max_tokens=s.max_tokens,
        ignore_eos=s.ignore_eos,
        stop_strings=s.stop_strings,
        n=s.n,
        logprobs=s.logprobs,
        required_channel=(
            None
            if channel is None
            else RequiredChannel(
                channel.after, channel.opening, channel.opens_at, channel.end_ids
            )
        ),
        top_logprobs=s.top_logprobs,
        json_schema=s.json_schema,
        structured_tool_format=s.structured_tool_format,
        structured_reasoning_end=s.structured_reasoning_end,
        max_tokens_is_default=s.max_tokens_is_default,
    )


def prompt_token_ids(item: m.SubmitItem) -> array.array:
    tokens = array.array("i")
    tokens.frombytes(item.prompt)
    return tokens


def unpickled(data: bytes | None):
    return None if data is None else pickle.loads(data)


def media(item: m.SubmitItem) -> tuple:
    """`(multimodal_data, mrope_positions)` as the client sent them."""
    if item.multimodal is None:
        return None, None
    return unpickle_media(item.multimodal, item.multimodal_buffers)


def shipped_chain(item: m.SubmitItem) -> ShippedChain | None:
    chain = item.prompt_hash_chain
    if chain is None:
        return None
    hashes = array.array("Q")
    hashes.frombytes(chain.hashes)
    return ShippedChain(
        chain.hash_block_size,
        chain.cache_seed,
        chain.num_prompt_tokens,
        chain.fingerprint,
        hashes,
    )


def rejected(item: m.SubmitItem, reason: str, code: int) -> m.Completion:
    """The completion a refused request ends with."""
    return m.Completion(
        seq_id=item.seq_id,
        num_prompt_tokens=len(item.prompt) // 4,
        arrive_time=item.arrive_time,
        leave_reason=f"rejected: {reason}",
        reject_code=code,
    )


def rejection_code(exc: BaseException) -> int:
    """INVALID for what a bad request raises, INTERNAL for anything else."""
    if isinstance(exc, (ValueError, TypeError, IndexError, KeyError, OverflowError)):
        return m.REJECT_INVALID
    return m.REJECT_INTERNAL


def _completion(seq) -> m.Completion:
    if isinstance(seq, m.Completion):  # refused at admission
        return seq
    return m.Completion(
        seq_id=seq.id,
        completion_tokens=_token_bytes(seq, seq.num_prompt_tokens, seq.num_tokens),
        logprobs_bytes=seq.logprobs.tobytes() if seq.return_logprobs else None,
        return_logprobs=seq.return_logprobs,
        num_prompt_tokens=seq.num_prompt_tokens,
        num_cached_tokens=seq.num_cached_tokens,
        arrive_time=seq.arrive_time,
        first_token_time=seq.first_token_time,
        leave_reason=seq.leave_reason,
        prefill_completed=seq.prefill_completed,
        json_schema=seq.json_schema,
        structured_tool_format=seq.structured_tool_format,
    )


def _failed_completion(item: m.Completion, exc: BaseException) -> m.Completion:
    return m.Completion(
        seq_id=item.seq_id,
        num_prompt_tokens=item.num_prompt_tokens,
        arrive_time=item.arrive_time,
        leave_reason=f"rejected: completion not encodable: {_reason(exc)}",
        reject_code=m.REJECT_INTERNAL,
    )


def _step_params(seq_id, params) -> bytes | None:
    """A step's KV-transfer params, pickled; None, and only this step's
    params lost, when they do not pickle."""
    try:
        return _optional_pickle(params)
    except Exception:
        logger.exception("seq %s's kv_transfer_params_output does not pickle", seq_id)
        return None


def _bare_step(item: m.StreamItem, exc: BaseException) -> m.StreamItem:
    # Its tokens still go; the rest rides its Completion, or is lost with it.
    return m.StreamItem(item.seq_id, item.output_tokens, item.finished)


def encode_output(request_type: EngineCoreRequestType, payload=None) -> bytes:
    """An EngineCore's message: ADD with finished `Sequence`s (or refused
    `Completion`s), STREAM with `(seq_id, RequestOutput)` pairs, READY with
    its dict, SHUTDOWN, or a payload only Python reads."""
    if request_type == EngineCoreRequestType.STREAM:
        items = [
            m.StreamItem(
                seq_id,
                out.output_tokens,
                out.finished,
                out.finish_reason,
                _step_params(seq_id, out.kv_transfer_params_output),
                out.num_cached_tokens or 0,
                out.logprobs,
            )
            for seq_id, out in payload
        ]
        return _encode_items(request_type, m.Stream, items, _bare_step)
    if request_type == EngineCoreRequestType.ADD:
        items = [_completion(seq) for seq in payload]
        return _encode_items(request_type, m.Completions, items, _failed_completion)
    if request_type == EngineCoreRequestType.READY:
        payload = payload or {}
        return _frame(
            request_type,
            m.Ready(
                max_pool_tokens=payload.get("max_pool_tokens"),
                weight_syncs_unavailable=payload.get("weight_syncs_unavailable") or {},
            ),
        )
    if request_type == EngineCoreRequestType.SHUTDOWN:
        return _frame(request_type, m.Empty())
    return _frame(request_type, _pickled(payload))
