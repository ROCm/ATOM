# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The engine wire protocol: values, frames, and every message both ways.

The equivalence tests are the contract this protocol was introduced under:
the `Sequence` an EngineCore builds from the wire must be the `Sequence` the
frontend used to build and pickle to it, attribute for attribute.
"""

import array
import inspect
import pickle
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from google.protobuf.message import DecodeError

from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.multimodal_runtime import multimodal_cache_seed
from atom.model_engine.protocol import engine_v1_pb2 as pb
from atom.model_engine.protocol.codec import (
    ADMIN_COMMANDS,
    REJECT_INTERNAL,
    REJECT_INVALID,
    SAMPLING_FIELDS,
    AdmissionContext,
    build_sequence,
    decode_client_message,
    decode_core_message,
    encode_abort,
    encode_command,
    encode_completions,
    encode_metrics,
    encode_ready,
    encode_rejected,
    encode_shutdown,
    encode_stopped,
    encode_stream,
    encode_submit,
    encode_utility_reply,
)
from atom.model_engine.protocol.requests import (
    CompletionResult,
    EngineRequest,
    prompt_token_array,
)
from atom.model_engine.protocol.wire import (
    OWN_FRAME_BYTES,
    PROTOCOL_VERSION,
    FrameReader,
    FrameWriter,
    ProtocolError,
    decode_envelope,
    decode_value,
    encode_value,
)
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import BlockTable, Sequence, SequenceStatus
from atom.sampling_params import SamplingParams

PER_REQ_MODEL = "qwen3_next"  # listed in InputOutputProcessor's per-req set


def _wire(frames):
    """What a peer receives: every frame as its own writable buffer."""
    return [bytearray(bytes(memoryview(f).cast("B"))) for f in frames]


def _roundtrip_value(value):
    writer = FrameWriter()
    msg = pb.Value()
    encode_value(value, msg, writer)
    envelope = pb.Envelope()
    envelope.metrics.snapshot.CopyFrom(msg)
    frames = _wire(writer.frames(envelope))
    out = decode_envelope(frames)
    return decode_value(out.metrics.snapshot, FrameReader(frames))


def assert_same(a, b, path="value"):
    """Deep equality that also checks types, and compares buffers by content."""
    if isinstance(a, BlockTable):
        # A table's version is a process-global draw; its ids are the content.
        assert isinstance(b, BlockTable), path
        assert list(a) == list(b), path
        return
    assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert a.device == b.device and a.requires_grad == b.requires_grad, path
        assert torch.equal(a.detach().cpu(), b.detach().cpu()), path
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert np.array_equal(a, b), path
    elif isinstance(a, np.generic):
        assert a.dtype == b.dtype and a == b, path
    elif isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), path
        for key in a:
            assert_same(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif is_dataclass(a) and not isinstance(a, type):
        for f in fields(a):
            assert_same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, SamplingParams):
        assert_same(vars(a), vars(b), path)
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def assert_same_sequence(old: Sequence, new: Sequence):
    old_vars, new_vars = vars(old), vars(new)
    assert set(old_vars) == set(new_vars)
    for name in old_vars:
        assert_same(old_vars[name], new_vars[name], f"seq.{name}")


# ---------------------------------------------------------------- values ---


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -7,
        (1 << 63) - 1,
        -(1 << 63),
        (1 << 64) - 1,  # a cache seed: blake2b's 8 bytes, unsigned
        -(1 << 70),
        1.5,
        float("inf"),
        "",
        "héllo",
        b"\x00\xff",
        [],
        (),
        [1, (2, 3), [4.0, None]],
        ((0, 5), (9, 3)),  # embedding_spans
        {"a": 1, 2: "b", (3, 4): [5]},
        {"nested": {"deeper": {"k": (1,)}}},
    ],
)
def test_plain_values_roundtrip_with_their_types(value):
    assert_same(value, _roundtrip_value(value))


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.uint8,
        np.int8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
def test_numpy_arrays_and_scalars_roundtrip(dtype):
    arr = (np.arange(24) % 7).astype(dtype).reshape(2, 3, 4)
    assert_same(arr, _roundtrip_value(arr))
    assert_same(arr[0, 1, 2], _roundtrip_value(arr[0, 1, 2]))  # numpy scalar


def test_numpy_edge_layouts_roundtrip():
    base = np.arange(30, dtype=np.int32).reshape(5, 6)
    for arr in (
        base[:, ::2],  # non-contiguous
        base.T,
        np.zeros((0, 3), np.float32),
        np.array(7, dtype=np.int64),  # 0-d
        np.arange(4, dtype=">i4"),  # big-endian
    ):
        out = _roundtrip_value(arr)
        assert np.array_equal(arr, out) and arr.shape == out.shape
        assert out.dtype == arr.dtype.newbyteorder("=")


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
    ],
)
def test_torch_tensors_roundtrip(dtype):
    t = (torch.arange(24) % 5).to(dtype).reshape(4, 6)
    assert_same(t, _roundtrip_value(t))
    assert_same(t[1:, ::2], _roundtrip_value(t[1:, ::2]).contiguous())


def test_torch_edge_tensors_roundtrip():
    for t in (
        torch.tensor(3.5),
        torch.empty(0, 4, dtype=torch.bfloat16),
        torch.randn(3, 2, requires_grad=True),
    ):
        assert_same(t, _roundtrip_value(t))


class _Opaque:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return type(other) is _Opaque and other.x == self.x


def test_values_the_tree_cannot_express_are_pickled_not_converted():
    for value in (
        _Opaque(3),
        {1, 2},
        OrderedDict(a=1),  # a dict subclass keeps its type
        np.array(["a", "b"]),  # a string array has no wire dtype
        torch.nn.Parameter(torch.ones(2)),
    ):
        out = _roundtrip_value(value)
        assert type(out) is type(value)
        if isinstance(value, torch.Tensor):
            assert torch.equal(out, value)
        elif isinstance(value, np.ndarray):
            assert np.array_equal(out, value)
        else:
            assert out == value


def test_large_tensors_alias_the_received_frame():
    big = torch.arange(OWN_FRAME_BYTES, dtype=torch.int32)
    writer = FrameWriter()
    msg = pb.Value()
    encode_value({"pixel_values": big}, msg, writer)
    envelope = pb.Envelope()
    envelope.metrics.snapshot.CopyFrom(msg)
    frames = _wire(writer.frames(envelope))
    assert len(frames) == 2  # envelope + the tensor's own frame
    out = decode_value(decode_envelope(frames).metrics.snapshot, FrameReader(frames))
    tensor = out["pixel_values"]
    frames[1][:4] = (12345).to_bytes(4, "little")
    assert int(tensor[0]) == 12345  # a view of the frame, not a copy
    del frames, out
    assert int(tensor[-1]) == OWN_FRAME_BYTES - 1  # the view keeps it alive


def test_small_buffers_share_one_aligned_frame():
    writer = FrameWriter()
    refs = [pb.BufferRef() for _ in range(3)]
    for i, ref in enumerate(refs):
        writer.add(array.array("i", range(i + 1)), ref)
    frames = writer.frames(pb.Envelope())
    assert len(frames) == 2
    assert {ref.frame for ref in refs} == {1}
    assert all(ref.offset % 64 == 0 for ref in refs)
    reader = FrameReader(_wire(frames))
    assert [list(reader.int32_array(ref)) for ref in refs] == [
        [0],
        [0, 1],
        [0, 1, 2],
    ]


def test_reader_refuses_refs_outside_the_message():
    frames = _wire(encode_submit([_request(1, [1, 2, 3])]))
    reader = FrameReader(frames)
    with pytest.raises(ProtocolError):
        reader.view(pb.BufferRef(frame=5, offset=0, nbytes=4))
    with pytest.raises(ProtocolError):
        reader.view(pb.BufferRef(frame=1, offset=0, nbytes=1 << 20))


def test_a_different_protocol_version_is_refused():
    frames = encode_shutdown()
    envelope = pb.Envelope()
    envelope.ParseFromString(bytes(frames[0]))
    envelope.protocol_version = PROTOCOL_VERSION + 1
    with pytest.raises(ProtocolError):
        decode_client_message([envelope.SerializeToString()])


def test_garbage_is_a_decode_error():
    with pytest.raises(DecodeError):
        decode_client_message([b"\xff\xff\xff"])


# ------------------------------------------------- coverage of the schema --


def test_wire_carries_every_sampling_params_field():
    assert {f.name for f in fields(SamplingParams)} == set(SAMPLING_FIELDS)


def test_wire_carries_every_sequence_constructor_argument():
    """A new Sequence argument must get a wire field, or a Core-side source."""
    params = set(inspect.signature(Sequence.__init__).parameters) - {"self"}
    _, specs, _ = decode_client_message(_wire(encode_submit([_request(1, [1])])))
    on_the_wire = set(specs[0].kwargs)
    derived_in_core = {"block_size", "num_draft_tokens", "has_per_req_cache", "id"}
    # The callback stays in the client: CoreManager keeps it by seq id.
    client_only = {"stream_callback"}
    assert params == on_the_wire | derived_in_core | client_only


# ------------------------------------------------------------ equivalence --


def _config(model_type="llama", spec_tokens=0, block_size=16):
    return SimpleNamespace(
        kv_cache_block_size=block_size,
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=spec_tokens) if spec_tokens else None
        ),
        hf_config=SimpleNamespace(model_type=model_type),
    )


def _request(seq_id, tokens, sampling=None, **kwargs):
    return EngineRequest(
        id=seq_id,
        prompt_token_ids=tokens,
        sampling_params=sampling or SamplingParams(),
        **kwargs,
    )


def _old_core_sequence(req: EngineRequest, config, *, max_tokens=None):
    """What the Core received before: a frontend-built Sequence, pickled."""
    from atom.model_engine.llm_engine import InputOutputProcessor

    spec = config.speculative_config
    seq = Sequence(
        list(req.prompt_token_ids),
        config.kv_cache_block_size,
        req.sampling_params,
        req.stop_token_sequences,
        stream_callback=req.stream_callback,
        id=req.id,
        num_draft_tokens=spec.num_speculative_tokens if spec else 0,
        has_per_req_cache=config.hf_config.model_type
        in InputOutputProcessor._per_req_cache_model_types(),
        kv_transfer_params=req.kv_transfer_params,
        multimodal_data=req.multimodal_data,
        mrope_positions=req.mrope_positions,
        mrope_position_delta=req.mrope_position_delta,
        needs_independent_noise=req.needs_independent_noise,
        parent_request_id=req.parent_request_id,
        sibling_index=req.sibling_index,
        request_id=req.external_request_id,
        data_parallel_rank=req.data_parallel_rank,
        dp_session_id=req.dp_session_id,
        dp_parent_session_id=req.dp_parent_session_id,
    )
    # The old frontend drew this id from the counter itself (0 included);
    # passing a falsy one to the constructor would draw another.
    seq.id = req.id
    seq.arrive_time = req.arrive_time
    seq.stream_callback = None  # CoreManager kept it
    if max_tokens is not None:
        seq.max_tokens = max_tokens
    return pickle.loads(pickle.dumps(seq))


def _new_core_sequences(reqs, config, **encode_kwargs):
    frames = _wire(encode_submit(reqs, **encode_kwargs))
    kind, specs, failures = decode_client_message(frames)
    assert kind == EngineCoreRequestType.ADD and failures == []
    context = AdmissionContext(config)
    return [build_sequence(spec, context) for spec in specs]


def _multimodal_data():
    return {
        "pixel_values": torch.randn(64, 32, dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.int64),
        "token_types": np.asarray([0, 1, 1, 2], dtype=np.int8),
        "embedding_spans": ((1, 2),),
    }


CASES = {
    "plain": lambda: [_request(5, [11, 12, 13], arrive_time=123.5)],
    "id_zero": lambda: [_request(0, [7], arrive_time=1.0)],
    "sampling": lambda: [
        _request(
            9,
            list(range(100)),
            SamplingParams(
                temperature=0.0,
                top_k=20,
                top_p=0.8,
                max_tokens=33,
                ignore_eos=True,
                stop_strings=["\n\n", "END"],
                logprobs=True,
            ),
            stop_token_sequences=[[271], [3, 4]],
            external_request_id="req-abc",
            parent_request_id="req-abc",
            data_parallel_rank=3,
            dp_session_id="s1",
            dp_parent_session_id="s0",
            arrive_time=9.25,
        )
    ],
    "stop_string_scalar": lambda: [
        _request(4, [1, 2], SamplingParams(stop_strings="###", logprobs=2))
    ],
    "kv_transfer": lambda: [
        _request(
            12,
            [5] * 40,
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_block_ids": [1, 2, 3],
                "remote_host": "10.0.0.1",
                "prefix_cache_hit_tokens": 32,
            },
        )
    ],
    "fanout": lambda: _fanout(),
    "multimodal": lambda: _multimodal(),
    "long_prompt": lambda: [_request(77, np.arange(100_000, dtype=np.int32) % 50_000)],
}


def _fanout():
    prompt = prompt_token_array([3, 1, 4, 1, 5, 9, 2, 6])
    sampling = SamplingParams(n=3, temperature=0.7, max_tokens=8)
    return [
        _request(
            40 + i,
            prompt,
            sampling,
            needs_independent_noise=True,
            parent_request_id="parent",
            sibling_index=i,
        )
        for i in range(3)
    ]


def _multimodal():
    data = _multimodal_data()
    # Shared by the siblings, as preprocess_fanout shares them.
    prompt = prompt_token_array([101, 7, 7, 102])
    positions = np.arange(12, dtype=np.int64).reshape(3, 4)
    return [
        _request(
            60 + i,
            prompt,
            SamplingParams(n=2),
            multimodal_data=data,
            cache_seed=multimodal_cache_seed(data),
            mrope_positions=positions,
            mrope_position_delta=-2,
            needs_independent_noise=True,
            sibling_index=i,
        )
        for i in range(2)
    ]


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize(
    "config",
    [_config(), _config(PER_REQ_MODEL, block_size=64)],
    ids=["dense", "per_req_cache"],
)
def test_core_builds_the_sequence_the_frontend_used_to_send(case, config):
    reqs = CASES[case]()
    for old_req, new_seq in zip(reqs, _new_core_sequences(reqs, config)):
        assert_same_sequence(_old_core_sequence(old_req, config), new_seq)


def test_equivalence_holds_under_speculative_decoding():
    config = _config(spec_tokens=3)
    reqs = CASES["plain"]()
    (new,) = _new_core_sequences(reqs, config)
    assert new.num_draft_tokens == 3
    assert_same_sequence(_old_core_sequence(reqs[0], config), new)


def test_prefill_half_of_a_disagg_pair_gets_one_token():
    """DisaggCoreManager used to copy each sequence and set max_tokens=1."""
    config = _config()
    reqs = CASES["sampling"]()
    (new,) = _new_core_sequences(reqs, config, max_tokens=1)
    assert_same_sequence(_old_core_sequence(reqs[0], config, max_tokens=1), new)
    assert reqs[0].max_tokens == 33  # the frontend's request is untouched


def test_siblings_send_their_shared_prompt_and_media_once():
    reqs = _multimodal()
    once = encode_submit(reqs[:1])
    twice = encode_submit(reqs)
    sent_once = sum(memoryview(f).nbytes for f in once[1:])
    sent_twice = sum(memoryview(f).nbytes for f in twice[1:])
    # The second sibling adds envelope fields, not another copy of any buffer.
    assert sent_twice == sent_once


def test_cache_seed_is_lent_to_the_constructor_and_taken_back():
    reqs = _multimodal()
    seqs = _new_core_sequences(reqs, _config())
    for seq in seqs:
        assert seq.cache_seed == reqs[0].cache_seed
        assert "cache_seed" not in seq.multimodal_data


def test_a_request_that_does_not_decode_is_reported_not_dropped():
    good, bad = _request(1, [1, 2]), _request(2, [3, 4])
    frames = encode_submit([good, bad])
    envelope = pb.Envelope()
    envelope.ParseFromString(bytes(frames[0]))
    envelope.submit.requests[1].sampling.top_p = 7.0  # SamplingParams refuses it
    frames[0] = envelope.SerializeToString()
    _, specs, failures = decode_client_message(_wire(frames))
    assert [s.seq_id for s in specs] == [1]
    assert [(seq_id, code) for seq_id, _, code in failures] == [(2, REJECT_INVALID)]


# ----------------------------------------------------------------- outputs --


def _finished_sequence():
    seq = Sequence(
        [1, 2, 3],
        16,
        SamplingParams(logprobs=True),
        id=31,
        request_id="ext",
        parent_request_id="ext",
        sibling_index=0,
    )
    for token in (40, 41, 42, 43):
        seq.append_token(token)
        seq.logprobs.append(-0.5 * token)
    seq.arrive_time, seq.first_token_time, seq.leave_time = 1.0, 1.25, 2.0
    seq.leave_reason = "stop"
    seq.num_cached_tokens = 2
    seq.kv_transfer_params_output = {"remote_block_ids": [4, 5], "ok": True}
    seq.status = SequenceStatus.FINISHED
    return seq


def test_completions_carry_what_the_frontend_reads():
    seq = _finished_sequence()
    kind, (result,) = decode_core_message(_wire(encode_completions([seq])))
    assert kind == EngineCoreRequestType.ADD
    assert isinstance(result, CompletionResult)
    for name in (
        "id",
        "completion_token_ids",
        "num_completion_tokens",
        "num_prompt_tokens",
        "num_tokens",
        "num_cached_tokens",
        "logprobs",
        "return_logprobs",
        "arrive_time",
        "first_token_time",
        "leave_time",
        "leave_reason",
        "kv_transfer_params_output",
        "external_request_id",
        "parent_request_id",
        "sibling_index",
    ):
        assert_same(getattr(seq, name), getattr(result, name), name)


def test_stream_outputs_roundtrip():
    outputs = [
        (
            3,
            RequestOutput(
                request_id=3,
                output_tokens=[5, 6],
                finished=False,
                num_cached_tokens=16,
            ),
        ),
        (
            4,
            RequestOutput(
                request_id=4,
                output_tokens=[],
                finished=True,
                finish_reason="abort",
                kv_transfer_params_output={"x": (1, 2)},
            ),
        ),
    ]
    kind, decoded = decode_core_message(_wire(encode_stream(outputs)))
    assert kind == EngineCoreRequestType.STREAM
    assert decoded == outputs


def _envelope(frames):
    return decode_envelope(_wire(frames))


def test_utility_commands_stay_pickled_and_name_their_request():
    payload = {
        "cmd": "update_weights",
        "named_tensors": [("w", torch.ones(3))],
        "request_id": "abc",
    }
    frames = encode_command(payload)
    assert _envelope(frames).WhichOneof("body") == "utility"
    assert _envelope(frames).utility.request_id == "abc"
    kind, (cmd, decoded), _ = decode_client_message(_wire(frames))
    assert kind == EngineCoreRequestType.UTILITY and cmd == "update_weights"
    assert decoded["named_tensors"][0][0] == "w"
    assert torch.equal(decoded["named_tensors"][0][1], torch.ones(3))

    reply = {"cmd": "update_weights", "result": 3, "request_id": "abc"}
    frames = encode_utility_reply(reply)
    assert _envelope(frames).utility_reply.request_id == "abc"
    assert decode_core_message(_wire(frames)) == (
        EngineCoreRequestType.UTILITY_RESPONSE,
        reply,
    )


@pytest.mark.parametrize("cmd", sorted(ADMIN_COMMANDS))
def test_admin_commands_are_typed_and_decode_to_the_same_payload(cmd):
    for payload in (
        {"cmd": cmd},
        {"cmd": cmd, "request_id": "r1", "timeout": 30.0, "deadline": 1.7e9},
    ):
        frames = encode_command(payload)
        envelope = _envelope(frames)
        assert envelope.WhichOneof("body") == "admin"
        assert envelope.admin.kind == ADMIN_COMMANDS[cmd]
        assert len(frames) == 1  # nothing pickled
        decoded = decode_client_message(_wire(frames))
        assert decoded == (EngineCoreRequestType.UTILITY, (cmd, payload), [])
        assert list(decoded[1][1]) == list(payload)  # the broadcast's key order


def test_an_admin_command_with_other_arguments_falls_back_to_utility():
    payload = {"cmd": "start_profile", "trace_dir": "/tmp"}
    assert _envelope(encode_command(payload)).WhichOneof("body") == "utility"


def test_admin_replies_are_typed_values():
    for reply in (
        {"cmd": "get_mtp_statistics", "result": {"enabled": False}, "request_id": "r"},
        {"cmd": "start_profile", "result": None, "request_id": "r"},
        {"cmd": "stop_profile", "result": [{"trace": "/tmp/x"}, None]},
        {"cmd": "clear_kv_cache", "error": "RuntimeError: boom", "request_id": "r"},
    ):
        frames = encode_utility_reply(reply)
        envelope = _envelope(frames)
        assert envelope.WhichOneof("body") == "admin_reply"
        decoded = decode_core_message(_wire(frames))
        assert decoded == (EngineCoreRequestType.UTILITY_RESPONSE, reply)
        assert list(decoded[1]) == list(reply)


def test_lifecycle_messages_roundtrip():
    assert decode_client_message(_wire(encode_abort([7, 0]))) == (
        EngineCoreRequestType.ABORT,
        [7, 0],
        [],
    )
    assert decode_client_message(_wire(encode_shutdown()))[0] == (
        EngineCoreRequestType.SHUTDOWN
    )
    assert decode_core_message(_wire(encode_ready({"max_pool_tokens": 4096}))) == (
        EngineCoreRequestType.READY,
        {"max_pool_tokens": 4096},
    )
    assert decode_core_message(_wire(encode_ready(None))) == (
        EngineCoreRequestType.READY,
        None,
    )
    metrics = {"enabled": True, "kv": {"used": 3, "frac": 0.25}, "rate": np.float64(1)}
    kind, decoded = decode_core_message(_wire(encode_metrics(metrics)))
    assert kind == EngineCoreRequestType.METRICS
    assert_same(metrics, decoded)
    assert decode_core_message(_wire(encode_stopped())) == (
        EngineCoreRequestType.SHUTDOWN,
        None,
    )


def test_ready_refuses_a_fact_it_has_no_field_for():
    with pytest.raises(ProtocolError, match="no field"):
        encode_ready({"max_pool_tokens": 1, "new_fact": 2})


def test_rejections_end_their_requests_and_say_why():
    frames = encode_rejected(
        [(8, "bad top_p", REJECT_INVALID), (9, "oom", REJECT_INTERNAL)]
    )
    items = _envelope(frames).rejected.items
    assert [(i.seq_id, i.reason, i.code) for i in items] == [
        (8, "bad top_p", REJECT_INVALID),
        (9, "oom", REJECT_INTERNAL),
    ]
    kind, results = decode_core_message(_wire(frames))
    assert kind == EngineCoreRequestType.ADD
    assert [r.id for r in results] == [8, 9]
    assert results[0].num_completion_tokens == 0
    assert results[0].leave_reason == "rejected: bad top_p"


# ------------------------------------------- what the Core fills in itself --


def test_model_validation_runs_in_the_core():
    def refuse(*, num_draft_tokens, multimodal_data, enable_dp_attention):
        if multimodal_data:
            raise ValueError("text requests only")

    config = _config()
    config.hf_config.validate_request = refuse
    context = AdmissionContext(config)
    (text,) = decode_client_message(_wire(encode_submit([_request(1, [1])])))[1]
    build_sequence(text, context)
    image = _request(2, [1], multimodal_data={"x": np.zeros(2, np.int8)})
    (spec,) = decode_client_message(_wire(encode_submit([image])))[1]
    with pytest.raises(ValueError, match="text requests only"):
        build_sequence(spec, context)


def test_the_core_computes_mrope_positions_a_client_left_out(monkeypatch):
    from atom.multimodal import registry

    calls = []

    def positions(config, tokens, multimodal_data):
        calls.append(list(tokens))
        return np.arange(len(tokens) * 3).reshape(3, -1), -1

    monkeypatch.setattr(registry, "get_mrope_input_positions", positions)
    context = AdmissionContext(_config())
    media = {"pixel_values": torch.ones(2, 2)}
    left_out = _request(1, [5, 6], multimodal_data=media)
    sent = _request(
        2,
        [5, 6],
        multimodal_data=media,
        mrope_positions=np.zeros((3, 2), np.int64),
        mrope_position_delta=4,
    )
    specs = decode_client_message(_wire(encode_submit([left_out, sent])))[1]
    computed, given = (build_sequence(spec, context) for spec in specs)
    assert calls == [[5, 6]]  # only for the one that left them out
    assert np.array_equal(computed.mrope_positions, np.arange(6).reshape(3, 2))
    assert computed.mrope_position_delta == -1
    assert np.array_equal(given.mrope_positions, np.zeros((3, 2)))
    assert given.mrope_position_delta == 4


def test_the_core_hashes_media_a_client_did_not():
    media = _multimodal_data()
    (spec,) = decode_client_message(
        _wire(encode_submit([_request(1, [1, 2], multimodal_data=media)]))
    )[1]
    seq = build_sequence(spec, AdmissionContext(_config()))
    assert seq.cache_seed == multimodal_cache_seed(media)
