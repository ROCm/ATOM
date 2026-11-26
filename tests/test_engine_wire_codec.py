# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The engine wire protocol's codec (`zmq_protocol.codec`), message by message:
what the client sends builds the sequence the client built, what the Core
sends reads as what it sent, and one item that does not encode fails alone."""

import array
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch
from engine_wire_harness import core_context as _core_context

from atom.model_engine import admission
from atom.model_engine.engine_core_protocol import EngineCoreRequestType as T
from atom.model_engine.prefix_hash import ShippedChain
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.model_engine.zmq_protocol import codec
from atom.model_engine.zmq_protocol import messages as m
from atom.sampling_params import RequiredChannel, SamplingParams

END_IDS = (99,)


def core_context(**config):
    return _core_context(**{"end_token_ids": END_IDS, **config})


def submitted(*seqs):
    kind, items = codec.decode_input(codec.encode_input(T.ADD, list(seqs)))
    assert kind == T.ADD
    return items


def built(seq, context=None):
    [item] = submitted(seq)
    return admission.build_sequence(item, context or core_context())


# Set by the frontend or the Core itself, not sent: the Core's own.
NOT_SENT = {
    "block_table",
    "stream_callback",
    "spec_token_ids",
    "offload_joint",
    "multimodal_data",
    "mrope_positions",
    "required_channel",
}


def assert_same_sequence(sent: Sequence, got: Sequence):
    for name, value in vars(sent).items():
        if name in NOT_SENT or name.startswith("_atom_trace"):
            continue
        theirs = vars(got).get(name, "<missing>")
        if name == "prompt_hash_chain" and value is not None:
            assert (
                tuple(theirs[:4]) == tuple(value[:4]) and theirs.hashes == value.hashes
            )
        elif isinstance(value, np.ndarray):
            assert np.array_equal(value, theirs), name
        else:
            assert theirs == value, name


def test_a_submitted_sequence_is_built_as_the_client_built_it(monkeypatch):
    from atom.model_engine import multimodal_runtime

    channel = RequiredChannel((1,), (2, 3), 1, (4,)).ending_on(END_IDS)
    params = SamplingParams(
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        max_tokens=33,
        stop_strings=["\n\n"],
        required_channel=channel,
    )
    media = {"pixel_values": torch.arange(6.0).reshape(2, 3), "spans": ((0, 2), (3, 4))}
    transfer = {"remote_block_ids": list(range(8192)), "pair": (1, 2)}
    seq = Sequence(
        [5, 6, 7, 8],
        16,
        params,
        [[9, 10], [11]],
        id=42,
        kv_transfer_params=transfer,
        multimodal_data=media,
        mrope_positions=np.arange(12).reshape(3, 4),
        mrope_position_delta=-2,
        needs_independent_noise=True,
        parent_request_id="p",
        sibling_index=1,
        request_id="ext",
        data_parallel_rank=0,
        dp_session_id="s",
        dp_parent_session_id="ps",
    )
    seq.arrive_time = 123.5
    seq.dp_est_cached_tokens = 16
    seq.max_tokens = 17  # what the API server clamps it to
    seq.prompt_hash_chain = ShippedChain(
        16, seq.cache_seed, 4, 99, array.array("Q", [1, 2**64 - 1])
    )
    seq._atom_trace_carrier = {"traceparent": "00-ab"}
    seq._atom_trace_model, seq._atom_trace_request_id = "model", "rid"

    def no_rehash(data):
        raise AssertionError("the client's media seed is taken, not recomputed")

    monkeypatch.setattr(multimodal_runtime, "multimodal_cache_seed", no_rehash)
    core = built(seq)
    assert_same_sequence(seq, core)
    assert core.required_channel == channel
    hash(core.required_channel)  # the scheduler keeps it in a set
    assert core.multimodal_data.keys() == media.keys()  # no seed lent into it
    assert torch.equal(core.multimodal_data["pixel_values"], media["pixel_values"])
    assert core.multimodal_data["spans"] == ((0, 2), (3, 4))
    assert np.array_equal(core.mrope_positions, seq.mrope_positions)
    assert core._atom_trace_carrier == {"traceparent": "00-ab"}
    # A seed the media itself carries is the one a sequence keeps.
    carried = Sequence(
        [1, 2, 3], 16, multimodal_data={"pixel_values": torch.zeros(1), "cache_seed": 7}
    )
    assert built(carried).cache_seed == 7


def test_media_tensors_and_arrays_arrive_exactly_and_writable():
    x = torch.randn(4, 6)
    value = {
        "bf16": torch.randn(3, 5).to(torch.bfloat16),
        "transposed": x.t(),
        "scalar": torch.tensor(3.5),
        "empty": torch.empty(0, 7),
        "grad": torch.randn(2, requires_grad=True),
        "bool": torch.tensor([True, False]),
        "f16": np.arange(6, dtype=np.float16).reshape(2, 3),
        "strided": np.arange(12).reshape(3, 4)[:, 1],
        "zero_d": np.array(5),
        "objects": np.array([None, "a"], dtype=object),
        "shared": [x, x],
    }
    data, buffers = codec.pickle_media(value)
    # As they arrive: read-only views into one received message.
    received = msgspec.msgpack.decode(
        msgspec.msgpack.encode(buffers), type=list[memoryview]
    )
    out = codec.unpickle_media(data, received)
    for name, sent in value.items():
        got = out[name]
        if isinstance(sent, torch.Tensor):
            assert (got.dtype, got.shape) == (sent.dtype, sent.shape), name
            assert torch.equal(got, sent.detach()), name
            got.copy_(got)  # its own, writable memory
        elif isinstance(sent, np.ndarray):
            assert (got.dtype, got.shape) == (sent.dtype, sent.shape), name
            assert (got == sent).all(), name
    assert out["shared"][0] is out["shared"][1]
    assert torch.equal(out["shared"][0], x)


def test_a_finished_sequence_reads_as_it_did():
    seq = Sequence([1, 2], 16, SamplingParams(logprobs=True), id=61)
    seq.arrive_time, seq.first_token_time = 123.5, 124.0
    for token, logprob in ((100, -0.5), (101, -1.0)):
        seq.append_token(token)
        seq.logprobs.append(logprob)
    seq.leave_reason = "stop"
    kind, [done] = codec.decode_output(codec.encode_output(T.ADD, [seq]))
    read = (done.id, done.completion_token_ids, done.num_completion_tokens)
    read += (done.logprobs, done.return_logprobs, done.num_prompt_tokens)
    read += (done.arrive_time, done.first_token_time, done.leave_reason)
    assert (kind, *read, done.rejected) == (
        T.ADD, 61, [100, 101], 2, [-0.5, -1.0], True, 2, 123.5, 124.0, "stop", False
    )  # fmt: skip
    done.leave_time = 125.0  # the client stamps it


@pytest.mark.parametrize(
    "kind,payload",
    [
        (
            T.STREAM,
            [
                (
                    42,
                    RequestOutput(
                        42, [100], False, None, None, 3, [(-0.5, 1, 7, -0.1)]
                    ),
                ),
                (43, RequestOutput(43, [5], True, "eos", {"k": (1, 2)}, 0, None)),
            ],
        ),
        (T.READY, {"max_pool_tokens": 4096}),
        (T.READY, {"weight_syncs_unavailable": {"ipc": "no"}}),
        (T.READY, None),
        (T.METRICS, {"mtp": {"distribution": {1: 2}}}),
        (T.DP_REUSE, [(1, 2)]),
        (T.PREFILL_BACKLOG, (1, 2, ())),
        (T.UTILITY_RESPONSE, {"cmd": "x", "results": [(1, np.int64(3))]}),
        (T.SHUTDOWN, None),
    ],
)
def test_the_cores_other_messages_read_as_they_were_sent(kind, payload):
    assert codec.decode_output(codec.encode_output(kind, payload)) == (kind, payload)


@pytest.mark.parametrize(
    "kind,payload",
    [
        (T.UTILITY, {"cmd": "abort_request", "req_id": 1}),
        (T.DP_PREFIX_CREDIT, [(1, 2)]),
        (T.ABORT, [7, 8]),
        (T.SHUTDOWN, None),
    ],
)
def test_the_clients_other_messages_read_as_they_were_sent(kind, payload):
    assert codec.decode_input(codec.encode_input(kind, payload)) == (kind, payload)


@pytest.mark.parametrize(
    "decode, message, match",
    [
        (
            codec.decode_output,
            msgspec.msgpack.encode(
                (m.KIND[T.READY], m.Ready(protocol_version=m.PROTOCOL_VERSION + 1))
            ),
            "protocol",
        ),
        (codec.decode_input, msgspec.msgpack.encode((200, [])), "kind"),
        (codec.decode_output, b"not msgpack", ""),
    ],
)
def test_a_message_of_another_version_kind_or_format_does_not_decode(
    decode, message, match
):
    with pytest.raises(msgspec.DecodeError, match=match):
        decode(message)


def test_a_request_that_does_not_encode_is_refused_alone():
    bad = Sequence([1, 2], 16, SamplingParams(stop_strings=["\ud800"]), id=50)
    bad.arrive_time = 9.0
    good = Sequence([3], 16, SamplingParams(), id=51)
    bad_item, good_item = submitted(bad, good)
    assert good_item.refused is None
    assert built(good).id == 51
    with pytest.raises(ValueError, match="not encodable") as refused:
        admission.build_sequence(bad_item, core_context())
    completion = codec.rejected(
        bad_item, str(refused.value), codec.rejection_code(refused.value)
    )
    _, [done] = codec.decode_output(codec.encode_output(T.ADD, [completion]))
    assert done.rejected and done.reject_code == m.REJECT_INVALID
    # What the client's bookkeeping reads, as for any finished request.
    assert (done.arrive_time, done.num_prompt_tokens) == (9.0, 2)
    assert done.leave_reason.startswith("rejected: request not encodable")


def test_a_step_that_does_not_encode_keeps_its_tokens():
    outputs = [
        (1, RequestOutput(1, [5], True, "\ud800", None, 0, None)),
        (2, RequestOutput(2, [6], False)),
    ]
    _, got = codec.decode_output(codec.encode_output(T.STREAM, outputs))
    assert [(o.output_tokens, o.finished) for _, o in got] == [
        ([5], True),
        ([6], False),
    ]


def test_a_completion_that_does_not_encode_ends_rejected():
    seq = Sequence([1, 2], 16, SamplingParams(), id=7)
    seq.append_token(3)
    seq.leave_reason = "\ud800"
    _, [done] = codec.decode_output(codec.encode_output(T.ADD, [seq]))
    assert done.rejected and done.reject_code == m.REJECT_INTERNAL
    assert done.num_prompt_tokens == 2


def test_a_bad_sampling_field_refuses_its_request_alone():
    item = m.SubmitItem(
        seq_id=1,
        prompt=array.array("i", [1, 2]).tobytes(),
        sampling=msgspec.Raw(msgspec.msgpack.encode(m.Sampling(top_p=2.0))),
    )
    with pytest.raises(ValueError, match="top_p"):
        admission.build_sequence(item, core_context())


def test_a_model_without_mrope_does_not_compute_it():
    media = {"pixel_values": torch.zeros(1)}
    seq = Sequence([1, 2, 3], 16, SamplingParams(), id=3, multimodal_data=media)
    context = core_context(
        hf_config=SimpleNamespace(
            model_type="deepseek_v4", architectures=["DeepseekV41ForCausalLM"]
        )
    )
    assert not context.uses_mrope
    assert built(seq, context).mrope_positions is None


def test_what_the_core_consumes_at_once_is_a_view_of_the_frame_not_a_copy():
    params = {"remote_block_ids": list(range(64)), "do_remote_prefill": True}
    seq = Sequence([1, 2, 3], 16, SamplingParams(), id=9, kv_transfer_params=params)
    frame = bytearray(codec.encode_input(T.ADD, [seq]))
    _, [item] = codec.decode_input(frame)
    assert isinstance(item.prompt, memoryview)
    assert isinstance(item.kv_transfer_params, memoryview)
    assert list(codec.prompt_token_ids(item)) == [1, 2, 3]
    # A view: the frame's own bytes, which a write to the frame shows through.
    at = frame.find(item.prompt.tobytes())
    frame[at] = 7
    assert list(codec.prompt_token_ids(item)) == [7, 2, 3]
    frame[at] = 1
    assert codec.unpickled(item.kv_transfer_params) == params
    command = bytearray(codec.encode_input(T.UTILITY, {"cmd": "x", "n": (1, 2)}))
    assert codec.decode_input(command) == (T.UTILITY, {"cmd": "x", "n": (1, 2)})
    # The client holds a Completion long after its frame: it owns its bytes.
    done = Sequence([1], 16, SamplingParams(), id=9)
    done.append_token(5)
    _, [completion] = codec.decode_output(bytearray(codec.encode_output(T.ADD, [done])))
    assert type(completion.completion_tokens) is bytes


def test_the_core_runs_the_models_own_validation():
    seen = []

    def validate_request(**kwargs):
        seen.append(kwargs)
        if kwargs["multimodal_data"] is not None:
            raise ValueError("this model takes no media")

    context = core_context(
        hf_config=SimpleNamespace(
            model_type="llama", validate_request=validate_request
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        enable_dp_attention=True,
    )
    built(Sequence([1, 2, 3], 16, SamplingParams(), id=1), context)
    assert seen == [
        {
            "num_draft_tokens": 3,
            "multimodal_data": None,
            "media_on_prefill_instance": False,
            "enable_dp_attention": True,
        }
    ]
    media = Sequence([1, 2, 3], 16, multimodal_data={"pixel_values": torch.zeros(1)})
    with pytest.raises(ValueError, match="no media"):
        built(media, context)


def test_the_core_refuses_what_the_frontend_refuses_of_a_prompt():
    """An empty prompt, and a decode leg whose prefill kept the image: its
    handed-off MRoPE delta must stay within the prompt, and RapidServe
    serves none."""
    [item] = submitted(Sequence([1, 2, 3], 16, SamplingParams(), id=60))
    empty = msgspec.structs.replace(item, prompt=memoryview(b""))
    with pytest.raises(ValueError, match="cannot be empty"):
        admission.build_sequence(empty, core_context())

    def leg(delta):
        params = {"do_remote_prefill": True, "cache_seed": 7}
        params["mrope_position_delta"] = delta
        return Sequence([1, 2, 3], 16, kv_transfer_params=params, id=61)

    seq = built(leg(-2))
    assert seq.media_on_prefill_instance and seq.cache_seed == 7
    with pytest.raises(ValueError, match="mrope_position_delta"):
        built(leg(1))
    with pytest.raises(ValueError, match="RapidServe"):
        built(leg(-2), core_context(enable_rapidserve=True))


@pytest.mark.parametrize("finished", [False, True])
def test_kv_params_that_do_not_load_end_only_their_own_sequence(finished):
    stream = m.Stream(
        [
            m.StreamItem(1, [5]),
            m.StreamItem(2, [6], finished, "eos" if finished else None, b"no pickle"),
            m.StreamItem(3, [7], True, "eos", codec._optional_pickle({"ok": 1})),
        ]
    )
    _, got = codec.decode_output(codec._frame(T.STREAM, stream))
    assert [(seq_id, o.output_tokens) for seq_id, o in got] == [
        (1, [5]),
        (2, [6]),
        (3, [7]),
    ]
    (_, ok), (_, bad), (_, done) = got
    assert (ok.finished, ok.finish_reason) == (False, None)
    assert (done.finished, done.kv_transfer_params_output) == (True, {"ok": 1})
    assert bad.finished and bad.kv_transfer_params_output is None
    assert bad.finish_reason.startswith("rejected: kv_transfer_params_output")
    # The Core still runs one it had not finished: the client stops it.
    assert got.cut_short == (() if finished else (2,))
