# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""engine_v1's codec, message by message: what a client or Core encodes is
what the other side decodes, field by field and type by type, and a bad
message fails the way the protocol says."""

import array
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.model_engine.zmq_protocol import codec, wire
from atom.model_engine.zmq_protocol import engine_v1_pb2 as pb
from atom.model_engine.zmq_protocol.requests import IngressRequest
from atom.sampling_params import RequiredChannel, SamplingParams


def on_the_wire(frames):
    """What the receiver gets: every frame as bytes."""
    return [bytes(f) for f in frames]


def submitted(params, **ingress):
    req = IngressRequest(id=7, prompt_token_ids=[1, 2, 3], sampling_params=params)
    for name, value in ingress.items():
        setattr(req, name, value)
    _, specs, failures = codec.decode_client_message(
        on_the_wire(codec.encode_submit([req]))
    )
    assert failures == []
    return specs[0]


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(),
        SamplingParams(logprobs=True, top_logprobs=3),
        SamplingParams(logprobs=0),
        SamplingParams(stop_strings="x"),
        SamplingParams(stop_strings=["a", "b"]),
        SamplingParams(json_schema='{"type":"object"}', structured_tool_format="kimi"),
        SamplingParams(structured_reasoning_end="</think>", max_tokens_is_default=True),
        SamplingParams(
            required_channel=RequiredChannel(
                after=(5, 6), opening=(7, 8), opens_at=1, end_ids=(9,)
            )
        ),
        SamplingParams(n=4, temperature=0.5, top_p=0.9, top_k=0),
    ],
    ids=repr,
)
def test_sampling_params_cross_field_by_field(params):
    spec = submitted(params)
    decoded = spec.kwargs["sampling_params"]
    # The request's own cap rides beside the parameters it shares.
    assert spec.max_tokens == params.max_tokens
    # The shared parameters keep SamplingParams' default instead.
    assert decoded == replace(params, max_tokens=SamplingParams().max_tokens)
    for name in codec.SAMPLING_FIELDS:
        if name != "max_tokens":
            assert type(getattr(decoded, name)) is type(getattr(params, name)), name


def test_requests_alike_but_for_max_tokens_share_one_decoded_params():
    first = submitted(SamplingParams(max_tokens=10))
    second = submitted(SamplingParams(max_tokens=20))
    assert first.kwargs["sampling_params"] is second.kwargs["sampling_params"]
    assert (first.max_tokens, second.max_tokens) == (10, 20)


def core_context(**config):
    """A Core's `AdmissionContext` over a minimal config; `config` overrides."""
    fields = {
        "kv_cache_block_size": 16,
        "speculative_config": None,
        "enable_dp_attention": False,
        "hf_config": SimpleNamespace(model_type="llama"),
        "end_token_ids": (2,),
        "kv_transfer_config": None,
        "enable_rapidserve": False,
        **config,
    }
    return codec.AdmissionContext(SimpleNamespace(**fields))


def test_the_sequence_gets_its_own_max_tokens():
    context = core_context()
    for cap in (10, 20):
        seq = codec.build_sequence(submitted(SamplingParams(max_tokens=cap)), context)
        assert seq.max_tokens == cap


def test_the_share_cache_estimate_reaches_the_sequence():
    spec = submitted(SamplingParams(), dp_est_cached_tokens=48)
    seq = codec.build_sequence(spec, core_context())
    assert seq.dp_est_cached_tokens == 48
    # Absent, as from a client that does not route by prefix: none.
    seq = codec.build_sequence(submitted(SamplingParams()), core_context())
    assert seq.dp_est_cached_tokens == 0


CHANNEL = RequiredChannel(after=(5, 6), opening=(7, 8), opens_at=1, end_ids=(9,))


@pytest.mark.parametrize(
    "params, ingress, config, refusal",
    [
        (
            SamplingParams(logprobs=True),
            {"kv_transfer_params": {"do_remote_decode": True}},
            {},
            "P/D transfer",
        ),
        (SamplingParams(logprobs=True), {}, {"enable_rapidserve": True}, "RapidServe"),
        (
            SamplingParams(json_schema="{}", required_channel=CHANNEL),
            {},
            {},
            "both a structured-output grammar",
        ),
        (
            SamplingParams(json_schema="{}", ignore_eos=True),
            {},
            {},
            "ignore_eos",
        ),
    ],
    ids=["logprobs-pd", "logprobs-rapidserve", "grammar-and-channel", "grammar-eos"],
)
def test_the_core_refuses_what_the_frontend_refuses(params, ingress, config, refusal):
    """A client other than this repo's frontend is held to the same admission."""
    spec = submitted(params, **ingress)
    with pytest.raises(ValueError, match=refusal):
        codec.build_sequence(spec, core_context(**config))


def test_the_core_holds_every_required_request_to_its_one_channel():
    context = core_context()
    first = codec.build_sequence(
        submitted(SamplingParams(required_channel=CHANNEL)), context
    )
    # Held back on the Core's own end ids too.
    assert first.required_channel == replace(CHANNEL, end_ids=(2, 9))
    # A client that held it back already sends the held one.
    again = codec.build_sequence(
        submitted(SamplingParams(required_channel=first.required_channel)), context
    )
    assert again.required_channel is first.required_channel
    other = replace(CHANNEL, after=(4, 6))
    with pytest.raises(ValueError, match="another channel"):
        codec.build_sequence(submitted(SamplingParams(required_channel=other)), context)


def test_absent_sampling_values_take_the_defaults_not_none():
    """A non-Python client may leave temperature / top_k / top_p out."""
    env = pb.Envelope()
    spec = env.submit.requests.add()
    writer = wire.FrameWriter()
    writer.add(array.array("i", [1, 2]), spec.prompt_token_ids)
    spec.sampling.n = 1
    _, specs, failures = codec.decode_client_message(on_the_wire(writer.frames(env)))
    assert failures == []
    params = specs[0].kwargs["sampling_params"]
    default = SamplingParams()
    assert (params.temperature, params.top_k, params.top_p) == (
        default.temperature,
        default.top_k,
        default.top_p,
    )


def value_round_trip(value):
    env = pb.Envelope()
    writer = wire.FrameWriter()
    wire.encode_value(value, env.metrics.snapshot, writer)
    frames = on_the_wire(writer.frames(env))
    env = wire.decode_envelope(frames)
    return wire.decode_value(env.metrics.snapshot, wire.FrameReader(frames))


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
            and a.requires_grad == b.requires_grad
        )
    if isinstance(a, np.ndarray):
        return a.dtype.kind == b.dtype.kind and np.array_equal(a, b)
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    return a == b


@pytest.mark.parametrize(
    "value",
    [
        None,
        1 << 64,  # past int64: no wire type, pickled
        [1, -(1 << 63), (1 << 63) - 1],  # packed
        (4, 5),  # packed, still a tuple
        [1, True, np.int64(2)],  # not plain ints: a Value each, types kept
        [1, 1 << 64],  # one past int64: a Value each
        float("nan"),
        b"\x00\x01",
        [1, [2, (3, 4)], {"a": None}],
        {1: "a", (1, 2): "b", None: 3},  # keys that are not strings
        {1, 2},  # no wire type: pickled
        np.int32(5),
        np.arange(40000, dtype=np.float32),  # its own frame
        np.arange(20).reshape(4, 5)[:, ::2],  # strided
        torch.ones(5, dtype=torch.bfloat16),
        torch.ones(3, requires_grad=True),
    ],
    ids=lambda v: type(v).__name__,
)
def test_values_cross_whole(value):
    assert same(value, value_round_trip(value))


def test_an_int_list_goes_packed():
    env = pb.Envelope()
    wire.encode_value(
        {"ids": list(range(8192))}, env.metrics.snapshot, wire.FrameWriter()
    )
    [entry] = env.metrics.snapshot.map_value.entries
    assert entry.value.WhichOneof("kind") == "int_list_value"


def nested(depth, inner=5):
    """`inner` inside `depth` dicts."""
    for _ in range(depth):
        inner = {"k": inner}
    return inner


def test_the_deepest_value_allowed_still_parses():
    """The deepest place a Value sits (a RequestSpec's) at the deepest nesting
    allowed: the Core parses it under protobuf's recursion limit."""
    value = nested(wire.MAX_VALUE_DEPTH)
    spec = submitted(SamplingParams(), kv_transfer_params=value)
    assert spec.kwargs["kv_transfer_params"] == value


def test_a_value_nested_too_deep_is_refused_by_its_sender():
    req = IngressRequest(id=7, prompt_token_ids=[1], sampling_params=SamplingParams())
    req.kv_transfer_params = nested(wire.MAX_VALUE_DEPTH + 1)
    with pytest.raises(wire.ProtocolError, match="containers deep"):
        codec.encode_submit([req])


def test_a_small_decoded_tensor_does_not_pin_the_shared_frame():
    env = pb.Envelope()
    writer = wire.FrameWriter()
    wire.encode_value(np.arange(4, dtype=np.int32), env.metrics.snapshot, writer)
    frames = writer.frames(env)
    frames[0] = bytes(frames[0])
    shared = bytearray(frames[1])
    frames[1] = shared
    out = wire.decode_value(env.metrics.snapshot, wire.FrameReader(frames))
    shared[:] = b"\xff" * len(shared)
    assert out.tolist() == [0, 1, 2, 3]


def test_a_token_id_past_int32_is_refused_not_wrapped():
    with pytest.raises(OverflowError):
        IngressRequest(
            id=1,
            prompt_token_ids=np.array([1, 1 << 31], dtype=np.int64),
            sampling_params=SamplingParams(),
        )
    req = IngressRequest(
        id=1,
        prompt_token_ids=np.array([1, 2], dtype=np.int64),
        sampling_params=SamplingParams(),
    )
    assert list(req.prompt_token_ids) == [1, 2]


def test_a_tensor_whose_buffer_does_not_match_its_shape_is_refused():
    env = pb.Envelope()
    writer = wire.FrameWriter()
    wire.encode_value(torch.arange(4, dtype=torch.int32), env.metrics.snapshot, writer)
    env.metrics.snapshot.tensor_value.shape[:] = [8]
    frames = on_the_wire(writer.frames(env))
    with pytest.raises(wire.ProtocolError, match="needs 32 bytes"):
        wire.decode_value(
            wire.decode_envelope(frames).metrics.snapshot, wire.FrameReader(frames)
        )


def test_a_stream_crosses_with_logprob_columns_of_every_width():
    outputs = [
        (
            1,
            RequestOutput(
                1,
                [10, 11],
                False,
                None,
                None,
                5,
                [(-0.5, 1, 10, -0.5, 3, -1.25), (-0.1, 2, 11, -0.1, 4, -2.0)],
            ),
        ),
        (2, RequestOutput(2, [], True, "stop_7", {"a": 1}, 0, [])),
        (3, RequestOutput(3, [12], True, "eos", None, 0, None)),
        (4, RequestOutput(4, [13], False, None, None, 0, [(-0.25, 1)])),
    ]
    _, decoded = codec.decode_core_message(on_the_wire(codec.encode_stream(outputs)))
    assert [seq_id for seq_id, _ in decoded] == [1, 2, 3, 4]
    for (_, sent), (_, got) in zip(outputs, decoded):
        assert got == replace(sent, output_tokens=list(sent.output_tokens))


def test_an_unencodable_kv_output_drops_only_its_own_entry():
    outputs = [
        (1, RequestOutput(1, [1], False)),
        (2, RequestOutput(2, [2], True, "eos", {"x": Unpicklable()})),
    ]
    _, decoded = codec.decode_core_message(on_the_wire(codec.encode_stream(outputs)))
    assert [list(out.output_tokens) for _, out in decoded] == [[1], [2]]
    assert decoded[1][1].kv_transfer_params_output is None


class Unpicklable:
    def __reduce__(self):
        raise TypeError("nope")


def finished_seq(seq_id, kv_output=None):
    seq = Sequence([1, 2, 3], 16, SamplingParams(), id=seq_id)
    seq.append_token(4)
    seq.leave_reason = "eos"
    seq.kv_transfer_params_output = kv_output
    return seq


def test_a_completion_that_does_not_encode_fails_alone():
    seqs = [finished_seq(1), finished_seq(2, {"x": Unpicklable()}), finished_seq(3)]
    _, done = codec.decode_core_message(on_the_wire(codec.encode_completions(seqs)))
    assert [d.id for d in done] == [1, 2, 3]
    assert [d.leave_reason for d in done] == ["eos", done[1].leave_reason, "eos"]
    assert done[1].leave_reason.startswith(f"{codec.REJECTED_REASON}: ")
    assert list(done[2].completion_token_ids) == [4]


def test_a_completion_that_does_not_decode_fails_alone():
    seqs = [finished_seq(1), finished_seq(2, {"x": {1, 2}}), finished_seq(3)]
    frames = codec.encode_completions(seqs)
    env = wire.decode_envelope(on_the_wire(frames))
    # Its pickled params point at a frame the message does not have.
    item = env.completions.items[1].kv_transfer_params_output
    item.map_value.entries[0].value.pickled_value.frame = 9
    frames[0] = env.SerializeToString()
    _, done = codec.decode_core_message(on_the_wire(frames))
    assert [d.id for d in done] == [1, 2, 3]
    assert done[1].leave_reason.startswith(f"{codec.REJECTED_REASON}: ")
    assert "missing frame 9" in done[1].leave_reason
    assert [d.leave_reason for d in (done[0], done[2])] == ["eos", "eos"]


def test_a_completion_carries_what_a_client_reads():
    seq = Sequence(
        [1, 2, 3],
        16,
        SamplingParams(logprobs=True),
        id=9,
        request_id="ext",
        parent_request_id="p",
        sibling_index=2,
    )
    seq.append_token(4)
    seq.logprobs.append(-0.5)
    seq.leave_reason = "eos"
    seq.arrive_time, seq.first_token_time, seq.leave_time = 1.0, 1.5, 2.0
    seq.kv_transfer_params_output = {"blocks": [1, 2]}
    seq.prefill_completed = True
    _, [done] = codec.decode_core_message(on_the_wire(codec.encode_completions([seq])))
    for name in (
        "return_logprobs",
        "num_prompt_tokens",
        "num_cached_tokens",
        "arrive_time",
        "first_token_time",
        "leave_time",
        "leave_reason",
        "kv_transfer_params_output",
        "external_request_id",
        "parent_request_id",
        "sibling_index",
        "prefill_completed",
    ):
        assert getattr(done, name) == getattr(seq, name), name
    assert done.id == 9
    assert list(done.completion_token_ids) == [4]
    assert list(done.logprobs) == [-0.5]


@pytest.mark.parametrize(
    "payload, body",
    [
        ({"cmd": "start_profile"}, "admin"),
        ({"cmd": "get_cache_statistics", "request_id": "r1", "timeout": 30.0}, "admin"),
        ({"cmd": "clear_kv_cache", "extra": 1}, "utility"),  # not an admin's shape
        ({"cmd": "abort_request", "req_id": 3}, "utility"),
    ],
)
def test_commands_go_typed_when_they_fit_and_pickled_otherwise(payload, body):
    frames = on_the_wire(codec.encode_command(payload))
    assert wire.decode_envelope(frames).WhichOneof("body") == body
    _, (_, decoded), _ = codec.decode_client_message(frames)
    assert decoded == payload


def test_siblings_share_one_prompt_frame():
    prompt = array.array("i", range(20000))  # 80 KB: a frame of its own
    reqs = [
        IngressRequest(id=i, prompt_token_ids=prompt, sampling_params=SamplingParams())
        for i in range(3)
    ]
    frames = codec.encode_submit(reqs)
    assert len(frames) == 2


def test_one_bad_request_is_rejected_alone():
    env = pb.Envelope()
    writer = wire.FrameWriter()
    for seq_id in range(2):
        spec = env.submit.requests.add()
        spec.seq_id = seq_id
        spec.sampling.n = 1
        writer.add(array.array("i", [1, 2]), spec.prompt_token_ids)
    env.submit.requests[1].prompt_token_ids.frame = 9
    _, specs, failures = codec.decode_client_message(on_the_wire(writer.frames(env)))
    assert [s.seq_id for s in specs] == [0]
    [(seq_id, reason, code)] = failures
    assert (seq_id, code) == (1, codec.REJECT_INVALID)
    assert "missing frame 9" in reason


def test_share_cache_messages_cross_both_ways():
    kind, updates, failures = codec.decode_client_message(
        on_the_wire(codec.encode_dp_prefix_credit([(3, 16), (0, 0)]))
    )
    assert (kind, updates, failures) == (
        codec.EngineCoreRequestType.DP_PREFIX_CREDIT,
        [(3, 16), (0, 0)],
        [],
    )
    for applied in ((), [(3, 16)]):
        kind, report = codec.decode_core_message(
            on_the_wire(codec.encode_prefill_backlog((5, 4096, applied)))
        )
        assert kind == codec.EngineCoreRequestType.PREFILL_BACKLOG
        assert report == (5, 4096, list(applied))
    kind, reports = codec.decode_core_message(
        on_the_wire(codec.encode_dp_reuse([(7, 32), (8, 0)]))
    )
    assert (kind, reports) == (codec.EngineCoreRequestType.DP_REUSE, [(7, 32), (8, 0)])


def test_paired_columns_of_unequal_length_are_refused():
    env = pb.Envelope()
    env.dp_reuse.seq_ids.extend([1, 2])
    env.dp_reuse.reused_tokens.append(5)
    with pytest.raises(wire.ProtocolError):
        codec.decode_core_message(on_the_wire(wire.FrameWriter().frames(env)))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"max_pool_tokens": 4096},
        {"weight_syncs_unavailable": {"weight_reload": "a runner cannot reload"}},
        {
            "max_pool_tokens": 4096,
            "weight_syncs_unavailable": {"weight_reload": "a runner cannot reload"},
        },
    ],
    ids=repr,
)
def test_ready_carries_every_startup_fact(payload):
    kind, decoded = codec.decode_core_message(on_the_wire(codec.encode_ready(payload)))
    assert kind.name == "READY"
    assert decoded == payload


def test_ready_refuses_a_fact_it_has_no_field_for():
    with pytest.raises(codec.ProtocolError):
        codec.encode_ready({"max_pool_tokens": 1, "new_fact": 2})
