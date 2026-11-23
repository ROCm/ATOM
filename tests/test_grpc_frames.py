# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The Session's engine_v1 codec: a SubmitBatch decodes to what preprocess takes, each
buffer read through its ref, and the server's messages carry what a client reads."""

import struct

import pytest

from atom.entrypoints.grpc import frames
from atom.entrypoints.openai.protocol import DEFAULT_MAX_TOKENS
from atom.model_engine.request import RequestOutput
from atom.model_engine.zmq_protocol import engine_v1_pb2 as pb
from atom.sampling_params import RequiredChannel, SamplingParams

VERSION = frames.PROTOCOL_VERSION
# A P instance's kv_transfer_params_output shape, with every JSON kind.
KV = {
    "do_remote_prefill": True,
    "remote_block_ids": [3, 4, 5],
    "remote_engine_id": "p0",
    "ratio": 0.5,
    "first_token_id": None,
    "endpoint": {"host": "10.0.0.2", "ports": [1, -2]},
}
# What a request that sets no sampling field reads as.
DEFAULTS = SamplingParams(
    max_tokens=DEFAULT_MAX_TOKENS, max_tokens_is_default=True, logprobs=False
)


def le32(*ids: int) -> bytes:
    return struct.pack(f"<{len(ids)}i", *ids)


def ref(frame: int, nbytes: int, offset: int = 0) -> pb.BufferRef:
    return pb.BufferRef(frame=frame, offset=offset, nbytes=nbytes)


def submit(specs, *data: bytes, version: int = VERSION) -> list[bytes]:
    """A SubmitBatch built by hand: `specs`, then `data` as frames 1, 2, ..."""
    envelope = pb.Envelope(protocol_version=version, submit=pb.SubmitBatch())
    envelope.submit.requests.extend(specs)
    return [envelope.SerializeToString(), *data]


def json_value(obj) -> pb.Value:
    """`obj` as a client turning JSON into engine_v1 writes it."""
    kinds = {bool: "bool_value", int: "int_value", float: "float_value"}
    if obj is None:
        return pb.Value(null_value=True)
    if isinstance(obj, str):
        return pb.Value(string_value=obj)
    if type(obj) in kinds:
        return pb.Value(**{kinds[type(obj)]: obj})
    if isinstance(obj, list):
        return pb.Value(list_value=pb.ValueList(items=[json_value(i) for i in obj]))
    entries = [
        pb.MapEntry(key=pb.Value(string_value=k), value=json_value(v))
        for k, v in obj.items()
    ]
    return pb.Value(map_value=pb.ValueMap(entries=entries))


def envelope(message: list[bytes]) -> pb.Envelope:
    decoded = pb.Envelope.FromString(message[0])
    assert decoded.protocol_version == VERSION
    return decoded


def buffer(message: list[bytes], buffer_ref: pb.BufferRef, dtype: str) -> list:
    data = message[buffer_ref.frame][
        buffer_ref.offset : buffer_ref.offset + buffer_ref.nbytes
    ]
    return list(struct.unpack(f"<{len(data) // struct.calcsize(dtype)}{dtype}", data))


def test_a_submit_batch_decodes_to_what_preprocess_takes():
    channel = pb.RequiredChannel(after=[1, 2], opening=[3, 4], opens_at=2, end_ids=[9])
    sampling = pb.SamplingParams(
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        max_tokens=77,
        ignore_eos=True,
        logprobs_flag=True,
        top_logprobs=2,
        required_channel=channel,
        structured_reasoning_end="</think>",
    )
    sampling.stop_strings.many.items.extend(["Hello world", "zu"])
    spec = pb.RequestSpec(
        seq_id=(1 << 24) - 1,
        prompt_token_ids=ref(1, 16),
        sampling=sampling,
        stop_token_sequences=[
            pb.TokenIds(ids=[5, 6]),
            pb.TokenIds(),
            pb.TokenIds(ids=[7]),
        ],
        external_request_id="cmpl-1",
        kv_transfer_params=json_value(KV),
        data_parallel_rank=3,
        dp_session_id="s1",
        dp_parent_session_id="s0",
    )
    kind, [request] = frames.decode_client(submit([spec], le32(0, 5, 70000, 2**31 - 1)))
    assert kind == "submit"
    assert request == frames.SubmitRequest(
        seq_id=(1 << 24) - 1,
        prompt_token_ids=[0, 5, 70000, 2**31 - 1],
        sampling_params=SamplingParams(
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            max_tokens=77,
            ignore_eos=True,
            stop_strings=["Hello world", "zu"],
            logprobs=True,
            top_logprobs=2,
            required_channel=RequiredChannel((1, 2), (3, 4), 2, (9,)),
            structured_reasoning_end="</think>",
        ),
        stop_token_sequences=[[5, 6], [7]],  # an empty one would match at once
        kv_transfer_params=KV,
        request_id="cmpl-1",
        data_parallel_rank=3,
        dp_session_id="s1",
        dp_parent_session_id="s0",
    )


def test_absent_sampling_fields_read_as_the_http_routes_fill_them():
    specs = [
        pb.RequestSpec(seq_id=1, prompt_token_ids=ref(1, 4)),
        pb.RequestSpec(
            seq_id=2, prompt_token_ids=ref(1, 4), sampling=pb.SamplingParams(n=0)
        ),
    ]
    _, items = frames.decode_client(submit(specs, le32(9)))
    assert [item.sampling_params for item in items] == [DEFAULTS, DEFAULTS]
    assert all(
        item.request_id is None and item.kv_transfer_params is None for item in items
    )


def test_explicit_zero_and_false_sampling_fields_are_kept_not_defaulted():
    explicit = pb.SamplingParams(
        temperature=0.0, top_k=0, top_p=1.0, max_tokens=5, logprobs_flag=False
    )
    spec = pb.RequestSpec(seq_id=1, prompt_token_ids=ref(1, 4), sampling=explicit)
    _, [item] = frames.decode_client(submit([spec], le32(9)))
    assert item.sampling_params == SamplingParams(
        temperature=0.0, top_k=0, top_p=1.0, max_tokens=5, logprobs=False
    )
    assert not item.sampling_params.max_tokens_is_default


def test_an_abort_decodes_to_its_seq_ids():
    message = pb.Envelope(protocol_version=VERSION, abort=pb.Abort(seq_ids=[0, 7]))
    assert frames.decode_client([message.SerializeToString()]) == ("abort", [0, 7])


def test_buffers_are_read_through_their_refs_in_shared_or_separate_frames():
    shared = le32(1, 2, 3).ljust(64, b"\0") + le32(4, 5)  # the second 64-byte aligned
    specs = [
        pb.RequestSpec(seq_id=1, prompt_token_ids=ref(1, 8, offset=64)),
        pb.RequestSpec(seq_id=2, prompt_token_ids=ref(2, 4)),
        pb.RequestSpec(seq_id=3, prompt_token_ids=ref(1, 12)),
    ]
    _, items = frames.decode_client(submit(specs, shared, le32(6)))
    prompts = [(r.seq_id, r.prompt_token_ids) for r in items]
    assert prompts == [(1, [4, 5]), (2, [6]), (3, [1, 2, 3])]


@pytest.mark.parametrize("version", [0, VERSION + 1])
def test_a_message_of_another_protocol_version_is_refused(version):
    spec = pb.RequestSpec(seq_id=1, prompt_token_ids=ref(1, 4))
    with pytest.raises(frames.FrameError, match="protocol version"):
        frames.decode_client(submit([spec], le32(5), version=version))


def test_a_request_the_session_cannot_read_is_rejected_alone():
    unreadable = {
        2: {"prompt_token_ids": ref(1, 8, offset=4)},  # overruns its frame
        3: {"prompt_token_ids": ref(3, 4)},  # a frame the message lacks
        4: {"prompt_token_ids": ref(0, 4)},  # frame 0 is the Envelope
        5: {"prompt_token_ids": ref(1, 6)},  # not whole int32 values
        6: {"prompt_token_ids": ref(2, 4)},  # a negative id
        7: {"multimodal_data": pb.Value(null_value=True)},
        8: {"sibling_index": 1, "needs_independent_noise": True},
        9: {"kv_transfer_params": pb.Value(string_value="not a map")},
        10: {"kv_transfer_params": pb.Value(pickled_value=ref(1, 8))},  # never loaded
        11: {"kv_transfer_params": pb.Value(tuple_value=pb.ValueList())},
        12: {"sampling": pb.SamplingParams(top_p=2.0)},  # SamplingParams refuses it
    }
    specs = [pb.RequestSpec(seq_id=1, prompt_token_ids=ref(1, 8))] + [
        pb.RequestSpec(**{"seq_id": n, "prompt_token_ids": ref(1, 8), **fields})
        for n, fields in unreadable.items()
    ]
    _, items = frames.decode_client(submit(specs, le32(7, 8), le32(-1)))
    assert items[0] == frames.SubmitRequest(1, [7, 8], DEFAULTS)
    rejected = [(type(item), item.seq_id, item.code) for item in items[1:]]
    assert rejected == [(frames.Rejection, n, frames.INVALID) for n in unreadable]


@pytest.mark.parametrize(
    "fields",
    [
        {"arrive_time": 1.5},
        {"trace": pb.TraceContext(model="m")},
        {"prompt_hash_chain": pb.PromptHashChain()},
        {"parent_request_id": "p"},
    ],
)
def test_a_field_the_session_does_not_read_is_rejected_not_dropped(fields):
    spec = pb.RequestSpec(seq_id=2, prompt_token_ids=ref(1, 4), **fields)
    _, [item] = frames.decode_client(submit([spec], le32(7)))
    assert (type(item), item.code) == (frames.Rejection, frames.INVALID)
    assert next(iter(fields)) in item.reason


def test_a_message_that_is_no_session_request_is_refused_whole():
    for body in ({"shutdown": pb.Shutdown()}, {"ready": pb.Ready()}, {}):
        message = pb.Envelope(protocol_version=VERSION, **body)
        with pytest.raises(frames.FrameError):
            frames.decode_client([message.SerializeToString()])
    for message in ([], [b"\xff\xff"]):
        with pytest.raises(frames.FrameError):
            frames.decode_client(message)


def test_a_stream_batch_carries_a_step_column_by_column():
    step = [
        (5, RequestOutput(5, [11, 12], False, num_cached_tokens=64)),
        (6, RequestOutput(6, [], True, "abort")),
        (7, RequestOutput(7, [21], True, "stop_21", KV)),
    ]
    message = frames.encode_stream(step)
    stream = envelope(message).stream
    assert list(stream.seq_ids) == [5, 6, 7]
    assert list(stream.token_counts) == [2, 0, 1]
    assert list(stream.tokens) == [11, 12, 21]
    assert list(stream.finished) == [False, True, True]
    assert list(stream.num_cached_tokens) == [64, 0, 0]
    assert dict(stream.finish_reasons) == {1: "abort", 2: "stop_21"}
    assert list(stream.kv_transfer_params_outputs) == [2]
    assert frames._get_value(stream.kv_transfer_params_outputs[2]) == KV
    assert message[1:] == []  # no logprobs: no data frames


def test_a_stream_batch_carries_logprob_records_in_little_endian_frames():
    records = [(-0.5, 1, 258, -0.75)]  # (logprob, rank, id, logprob)
    message = frames.encode_stream(
        [(1, RequestOutput(1, [258], False, logprobs=records))]
    )
    stream = envelope(message).stream
    assert (list(stream.logprob_entries), list(stream.logprob_counts)) == ([0], [1])
    assert list(stream.logprob_widths) == [1]
    assert message[1:] == [struct.pack("<2d", -0.5, -0.75), le32(1, 258)]


def test_completions_carry_usage_times_and_their_ids_by_ref():
    done = [
        frames.Completion(7, [21, 258], "stop_21", 12, 4, 1.5, 1.75, 2.0),
        frames.Completion(6, [], "abort", kv_transfer_params_output=KV),
        frames.Completion(5, [11], "eos", external_request_id="cmpl-5"),
    ]
    message = frames.encode_completions(done)
    items = envelope(message).completions.items
    assert [buffer(message, c.completion_token_ids, "i") for c in items] == [
        [21, 258],
        [],
        [11],
    ]
    first = items[0]
    assert (first.seq_id, first.leave_reason, first.num_prompt_tokens) == (
        7,
        "stop_21",
        12,
    )
    assert (first.num_cached_tokens, first.arrive_time) == (4, 1.5)
    assert (first.first_token_time, first.leave_time) == (1.75, 2.0)
    assert frames._get_value(items[1].kv_transfer_params_output) == KV
    assert items[2].external_request_id == "cmpl-5"
    assert message[1] == le32(21, 258)  # little-endian


def test_kv_transfer_params_that_are_not_json_shaped_are_not_encoded():
    for unsent in ({"x": b"\0"}, {1: "x"}, {"x": {1, 2}}):
        done = frames.Completion(1, [], "eos", kv_transfer_params_output=unsent)
        with pytest.raises(TypeError, match="not JSON-shaped"):
            frames.encode_completions([done])


def test_ready_and_rejected_carry_their_fields():
    assert envelope(frames.encode_ready(4096)).ready.max_pool_tokens == 4096
    assert not envelope(frames.encode_ready()).ready.HasField("max_pool_tokens")
    refused = [
        frames.Rejection(3, "max_tokens must be at least 1", frames.INVALID),
        frames.Rejection(4, "the KV pool is full", pb.RejectedRequest.OVERLOADED),
    ]
    items = envelope(frames.encode_rejected(refused)).rejected.items
    assert [frames.Rejection(r.seq_id, r.reason, r.code) for r in items] == refused
