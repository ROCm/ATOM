# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Golden vectors: one byte-exact message per kind, and what each one means.

Each `tests/fixtures/engine_protocol/<name>.frames` file holds one multipart
message -- a little-endian uint32 frame count, then per frame a uint32 length
and the frame's bytes (see the GOLDEN VECTORS section of engine_v1.proto).
This test checks that the Python codec encodes exactly those bytes and that
they decode to what the case below states, so a codec in another language can
be checked against the same files without talking to this one.

After an intended wire change, regenerate with
    ATOM_REGEN_PROTOCOL_FIXTURES=1 pytest tests/test_engine_protocol_golden.py
and bump PROTOCOL_VERSION if an older peer could misread the result.
"""

import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from atom.model_engine.engine_core_protocol import EngineCoreRequestType as T
from atom.model_engine.protocol.codec import (
    REJECT_INTERNAL,
    REJECT_INVALID,
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
from atom.model_engine.protocol.requests import EngineRequest, prompt_token_array
from atom.model_engine.protocol.wire import OWN_FRAME_BYTES
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

FIXTURES = Path(__file__).parent / "fixtures" / "engine_protocol"
REGENERATE = bool(os.environ.get("ATOM_REGEN_PROTOCOL_FIXTURES"))
# name -> the frames read back from its fixture, for checks about layout.
FIXTURE_FRAMES: dict[str, list] = {}


def serialize(frames) -> bytes:
    out = [struct.pack("<I", len(frames))]
    for frame in frames:
        data = bytes(memoryview(frame).cast("B"))
        out += [struct.pack("<I", len(data)), data]
    return b"".join(out)


def deserialize(blob: bytes) -> list:
    (count,), offset = struct.unpack_from("<I", blob), 4
    frames = []
    for _ in range(count):
        (size,) = struct.unpack_from("<I", blob, offset)
        frames.append(bytearray(blob[offset + 4 : offset + 4 + size]))
        offset += 4 + size
    assert offset == len(blob), "trailing bytes"
    return frames


# ------------------------------------------------------------ client -> Core


def _text_requests():
    return [
        EngineRequest(
            id=0,
            prompt_token_ids=[151644, 872, 198, 9707],
            sampling_params=SamplingParams(
                temperature=0.0,
                top_k=20,
                top_p=0.95,
                max_tokens=128,
                ignore_eos=False,
                stop_strings=["</s>", "\n\n"],
                logprobs=True,
            ),
            stop_token_sequences=[[2], [271, 271]],
            external_request_id="chatcmpl-1",
            parent_request_id="chatcmpl-1",
            data_parallel_rank=1,
            dp_session_id="session-a",
            arrive_time=1700000000.25,
        ),
        EngineRequest(
            id=1,
            prompt_token_ids=[9707],
            sampling_params=SamplingParams(max_tokens=8, stop_strings="###"),
            kv_transfer_params={"do_remote_prefill": True, "remote_block_ids": [3, 4]},
            arrive_time=1700000000.5,
        ),
    ]


def _check_text(decoded):
    kind, specs, failures = decoded
    assert kind == T.ADD and failures == []
    first, second = specs
    assert first.seq_id == 0 and first.arrive_time == 1700000000.25
    kw = first.kwargs
    assert list(kw["token_ids"]) == [151644, 872, 198, 9707]
    assert kw["sampling_params"] == SamplingParams(
        temperature=0.0,
        top_k=20,
        top_p=0.95,
        max_tokens=128,
        stop_strings=["</s>", "\n\n"],
        logprobs=True,
    )
    assert kw["stop_token_sequences"] == [[2], [271, 271]]
    assert kw["request_id"] == kw["parent_request_id"] == "chatcmpl-1"
    assert kw["data_parallel_rank"] == 1 and kw["dp_session_id"] == "session-a"
    assert kw["dp_parent_session_id"] is None and kw["multimodal_data"] is None
    kw = second.kwargs
    assert second.seq_id == 1 and list(kw["token_ids"]) == [9707]
    assert kw["sampling_params"].stop_strings == "###"
    assert kw["sampling_params"].logprobs is None
    assert kw["kv_transfer_params"] == {
        "do_remote_prefill": True,
        "remote_block_ids": [3, 4],
    }


def _fanout_requests():
    # One array for both siblings, as preprocess_fanout builds it: the prompt
    # goes out once and both specs point at it.
    prompt = prompt_token_array([101, 102, 103])
    sampling = SamplingParams(temperature=0.7, max_tokens=16, n=2)
    return [
        EngineRequest(
            id=10 + i,
            prompt_token_ids=prompt,
            sampling_params=sampling,
            needs_independent_noise=True,
            parent_request_id="cmpl-n2",
            sibling_index=i,
            arrive_time=1.0,
        )
        for i in range(2)
    ]


def _check_fanout(decoded):
    _, specs, _ = decoded
    assert len(FIXTURE_FRAMES["submit_fanout"]) == 2
    assert len(FIXTURE_FRAMES["submit_fanout"][1]) == 12  # one 3-token prompt
    assert [s.seq_id for s in specs] == [10, 11]
    assert [s.kwargs["sibling_index"] for s in specs] == [0, 1]
    assert all(s.kwargs["needs_independent_noise"] for s in specs)
    assert all(list(s.kwargs["token_ids"]) == [101, 102, 103] for s in specs)


def _multimodal_request():
    # What a client that cannot run the model's MRoPE code sends: media and
    # its layout, no positions, no cache seed.
    return [
        EngineRequest(
            id=20,
            prompt_token_ids=[151652, 151655, 151655, 151653],
            sampling_params=SamplingParams(max_tokens=4),
            multimodal_data={
                "pixel_values": torch.arange(8, dtype=torch.float32).reshape(2, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
                "token_types": np.array([0, 1, 1, 0], dtype=np.int8),
                "embedding_spans": ((1, 2),),
            },
            arrive_time=2.0,
        )
    ]


def _check_multimodal(decoded):
    (spec,) = decoded[1]
    media = spec.kwargs["multimodal_data"]
    assert list(media) == [
        "pixel_values",
        "image_grid_thw",
        "token_types",
        "embedding_spans",
    ]
    assert torch.equal(
        media["pixel_values"], torch.arange(8, dtype=torch.float32).reshape(2, 4)
    )
    assert media["image_grid_thw"].dtype == torch.int64
    assert media["token_types"].tolist() == [0, 1, 1, 0]
    assert media["embedding_spans"] == ((1, 2),)
    assert spec.cache_seed is None and spec.kwargs["mrope_positions"] is None


def _long_prompt_request():
    tokens = np.arange(OWN_FRAME_BYTES // 4 + 3, dtype=np.int32) % 1000
    return [
        EngineRequest(
            id=30,
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=1),
        )
    ]


def _check_long_prompt(decoded):
    (spec,) = decoded[1]
    assert len(spec.kwargs["token_ids"]) == OWN_FRAME_BYTES // 4 + 3
    assert spec.kwargs["token_ids"][-1] == (OWN_FRAME_BYTES // 4 + 2) % 1000


ADMIN = {
    "cmd": "stop_profile",
    "request_id": "0123456789abcdef",
    "timeout": 600.0,
    "deadline": 1700000600.0,
}
UTILITY = {"cmd": "get_weight_sync_status", "request_id": "fedcba9876543210"}


CLIENT_CASES = {
    "submit_text": (lambda: encode_submit(_text_requests()), _check_text),
    "submit_fanout": (lambda: encode_submit(_fanout_requests()), _check_fanout),
    "submit_multimodal": (
        lambda: encode_submit(_multimodal_request()),
        _check_multimodal,
    ),
    "submit_long_prompt": (
        lambda: encode_submit(_long_prompt_request()),
        _check_long_prompt,
    ),
    "abort": (
        lambda: encode_abort([3, 0]),
        lambda d: d == (T.ABORT, [3, 0], []),
    ),
    "admin_stop_profile": (
        lambda: encode_command(ADMIN),
        lambda d: d == (T.UTILITY, ("stop_profile", ADMIN), []),
    ),
    "utility_pickled": (
        lambda: encode_command(UTILITY),
        lambda d: d == (T.UTILITY, ("get_weight_sync_status", UTILITY), []),
    ),
    "shutdown": (lambda: encode_shutdown(), lambda d: d == (T.SHUTDOWN, None, [])),
}


# ------------------------------------------------------------ Core -> client


def _finished_sequence():
    seq = Sequence(
        [1, 2, 3],
        16,
        SamplingParams(logprobs=True),
        id=40,
        request_id="ext-40",
        parent_request_id="ext-40",
    )
    for token, logprob in ((7, -0.5), (8, -0.25)):
        seq.append_token(token)
        seq.logprobs.append(logprob)
    seq.arrive_time, seq.first_token_time, seq.leave_time = 10.0, 10.5, 11.0
    seq.leave_reason = "stop"
    seq.num_cached_tokens = 2
    seq.kv_transfer_params_output = {"remote_block_ids": [9]}
    return seq


def _check_completion(decoded):
    kind, (result,) = decoded
    assert kind == T.ADD
    assert result.id == 40 and list(result.completion_token_ids) == [7, 8]
    assert list(result.logprobs) == [-0.5, -0.25] and result.return_logprobs
    assert result.num_prompt_tokens == 3 and result.num_cached_tokens == 2
    assert (result.arrive_time, result.first_token_time, result.leave_time) == (
        10.0,
        10.5,
        11.0,
    )
    assert result.leave_reason == "stop"
    assert result.kv_transfer_params_output == {"remote_block_ids": [9]}
    assert result.external_request_id == result.parent_request_id == "ext-40"
    return True


STREAM = [
    (50, RequestOutput(request_id=50, output_tokens=[11], finished=False)),
    (51, RequestOutput(request_id=51, output_tokens=[12, 13], finished=False)),
    (
        52,
        RequestOutput(
            request_id=52,
            output_tokens=[],
            finished=True,
            finish_reason="aborted",
            kv_transfer_params_output={"ok": True},
            num_cached_tokens=64,
        ),
    ),
]
METRICS = {"enabled": True, "running": 3, "waiting": 0, "kv": {"usage": 0.5}}
UTILITY_REPLY = {
    "cmd": "get_weight_sync_status",
    "result": {"fenced": False, "sync_id": None},
    "request_id": "fedcba9876543210",
}
ADMIN_REPLY = {
    "cmd": "get_mtp_statistics",
    "result": {"enabled": True, "acceptance_rate": 0.75, "accepted": [10, 5]},
    "request_id": "0123456789abcdef",
}


CORE_CASES = {
    "ready": (
        lambda: encode_ready({"max_pool_tokens": 131072}),
        lambda d: d == (T.READY, {"max_pool_tokens": 131072}),
    ),
    "ready_without_pool": (
        lambda: encode_ready(None),
        lambda d: d == (T.READY, None),
    ),
    "completions": (
        lambda: encode_completions([_finished_sequence()]),
        _check_completion,
    ),
    "stream": (lambda: encode_stream(STREAM), lambda d: d == (T.STREAM, STREAM)),
    "metrics": (lambda: encode_metrics(METRICS), lambda d: d == (T.METRICS, METRICS)),
    "admin_reply": (
        lambda: encode_utility_reply(ADMIN_REPLY),
        lambda d: d == (T.UTILITY_RESPONSE, ADMIN_REPLY),
    ),
    "utility_reply_pickled": (
        lambda: encode_utility_reply(UTILITY_REPLY),
        lambda d: d == (T.UTILITY_RESPONSE, UTILITY_REPLY),
    ),
    "rejected": (
        lambda: encode_rejected(
            [(60, "ValueError: top_p", REJECT_INVALID), (61, "boom", REJECT_INTERNAL)]
        ),
        lambda d: [(r.id, r.leave_reason) for r in d[1]]
        == [(60, "rejected: ValueError: top_p"), (61, "rejected: boom")],
    ),
    "stopped": (lambda: encode_stopped(), lambda d: d == (T.SHUTDOWN, None)),
}


CASES = [(name, *case, decode_client_message) for name, case in CLIENT_CASES.items()]
CASES += [(name, *case, decode_core_message) for name, case in CORE_CASES.items()]


@pytest.mark.parametrize(
    "name,encode,check,decode", CASES, ids=[case[0] for case in CASES]
)
def test_golden_vector(name, encode, check, decode):
    blob = serialize(encode())
    path = FIXTURES / f"{name}.frames"
    if REGENERATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    assert path.exists(), f"missing {path}; regenerate (see module docstring)"
    assert blob == path.read_bytes(), f"{name}: the encoding changed"
    FIXTURE_FRAMES[name] = deserialize(path.read_bytes())
    result = check(decode(deserialize(path.read_bytes())))
    assert result is None or result is True


def test_every_message_kind_has_a_vector():
    from atom.model_engine.protocol import engine_v1_pb2 as pb

    kinds = {
        field.name for field in pb.Envelope.DESCRIPTOR.oneofs_by_name["body"].fields
    }
    covered = set()
    for name, encode, *_ in CASES:
        envelope = pb.Envelope()
        envelope.ParseFromString(bytes(memoryview(encode()[0]).cast("B")))
        covered.add(envelope.WhichOneof("body"))
    assert kinds == covered, f"no golden vector for {sorted(kinds - covered)}"


def test_no_stale_vectors():
    names = {name for name, *_ in CASES}
    on_disk = {path.stem for path in FIXTURES.glob("*.frames")}
    assert on_disk <= names, f"vectors with no case: {sorted(on_disk - names)}"
