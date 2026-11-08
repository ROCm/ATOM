# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""engine_v1 across the language boundary.

The Core's messages are encoded here and decoded by the Rust client
(`atom/mesh/src/atom_engine/tests.rs`); the Rust client's are encoded there
and decoded here. Each side keeps its own messages under
`tests/fixtures/engine_v1/` (`core_*` from here, `client_*` from Rust) and
fails when they drift from what it encodes now; `ATOM_UPDATE_FIXTURES=1`
rewrites them (and fails, so a rewrite is never mistaken for a pass), and the
other side's tests then check the new ones. Each side also decodes its own
fixtures and checks what they say, so neither half guards nothing alone.
"""

import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.protocol import codec
from atom.model_engine.protocol.engine_v1_pb2 import Envelope
from atom.model_engine.request import RequestOutput
from atom.sampling_params import RequiredChannel

FIXTURES = Path(__file__).parent / "fixtures" / "engine_v1"


def write_frames(path: Path, frames: list) -> None:
    """A message as `u32le length, bytes` per frame."""
    with path.open("wb") as f:
        for frame in frames:
            data = bytes(frame)
            f.write(struct.pack("<I", len(data)))
            f.write(data)


def read_frames(path: Path) -> list[bytes]:
    data, frames, at = path.read_bytes(), [], 0
    while at < len(data):
        (n,) = struct.unpack_from("<I", data, at)
        frames.append(data[at + 4 : at + 4 + n])
        at += 4 + n
    return frames


def core_messages() -> dict[str, list]:
    """One of each message a Core sends, for the client's sequence 0."""
    completion = SimpleNamespace(
        id=0,
        completion_token_ids=np.array([5, 6], dtype=np.int32),
        logprobs=np.array([-0.5, -1.25], dtype=np.float64),
        return_logprobs=True,
        num_prompt_tokens=3,
        num_cached_tokens=2,
        arrive_time=1.0,
        first_token_time=2.0,
        leave_time=3.0,
        leave_reason="max_tokens",
        kv_transfer_params_output=None,
        external_request_id="req",
        parent_request_id="req",
        sibling_index=0,
    )
    stream = RequestOutput(
        request_id=0,
        output_tokens=[7, 8],
        finished=True,
        finish_reason="eos",
        num_cached_tokens=3,
    )
    return {
        "core_ready": codec.encode_ready({"max_pool_tokens": 4096}),
        "core_stream": codec.encode_stream([(0, stream)]),
        "core_completions": codec.encode_completions([completion]),
        "core_rejected": codec.encode_rejected([(0, "bad", codec.REJECT_INVALID)]),
        "core_stopped": codec.encode_stopped(),
    }


def same_message(a: list, b: list) -> bool:
    """Equal envelopes (map order aside) and equal data frames."""
    head_a, head_b = Envelope(), Envelope()
    head_a.ParseFromString(bytes(a[0]))
    head_b.ParseFromString(bytes(b[0]))
    return head_a == head_b and [bytes(f) for f in a[1:]] == [bytes(f) for f in b[1:]]


@pytest.mark.parametrize("name", sorted(core_messages()))
def test_core_messages_match_their_fixtures(name):
    frames = core_messages()[name]
    path = FIXTURES / f"{name}.frames"
    if os.environ.get("ATOM_UPDATE_FIXTURES"):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_frames(path, frames)
        pytest.fail(f"rewrote {path.name}; rerun without ATOM_UPDATE_FIXTURES")
    assert same_message(frames, read_frames(path)), (
        f"{path.name} is stale: rerun with ATOM_UPDATE_FIXTURES=1, then the "
        "mesh tests (cargo test --lib atom_engine) against it"
    )


def test_the_core_fixtures_say_what_the_client_reads():
    """What the Rust client reads off each `core_*` fixture, read here too."""
    ready = Envelope()
    ready.ParseFromString(read_frames(FIXTURES / "core_ready.frames")[0])
    assert ready.ready.max_pool_tokens == 4096
    _, outputs = codec.decode_core_message(read_frames(FIXTURES / "core_stream.frames"))
    [(seq_id, out)] = outputs
    assert (seq_id, list(out.output_tokens), out.finished) == (0, [7, 8], True)
    assert (out.finish_reason, out.num_cached_tokens) == ("eos", 3)
    _, [done] = codec.decode_core_message(
        read_frames(FIXTURES / "core_completions.frames")
    )
    assert (done.id, list(done.completion_token_ids)) == (0, [5, 6])
    assert list(done.logprobs) == [-0.5, -1.25]
    assert (done.num_prompt_tokens, done.leave_reason) == (3, "max_tokens")
    _, [rejected] = codec.decode_core_message(
        read_frames(FIXTURES / "core_rejected.frames")
    )
    assert (rejected.id, rejected.leave_reason) == (0, f"{codec.REJECTED_REASON}: bad")


def test_the_rust_clients_submit_decodes_as_the_core_reads_it():
    kind, specs, failures = codec.decode_client_message(
        read_frames(FIXTURES / "client_submit.frames")
    )
    assert kind == EngineCoreRequestType.ADD
    assert failures == []
    assert [s.kwargs["sibling_index"] for s in specs] == [0, 1]
    for spec in specs:
        kw = spec.kwargs
        assert list(kw["token_ids"]) == [11, 12, 13]
        assert kw["stop_token_sequences"] == [[5, 6], [7]]
        assert kw["request_id"] is None  # n > 1: only the parent id is the client's
        assert kw["parent_request_id"] == "req"
        assert kw["needs_independent_noise"] is True
        assert kw["data_parallel_rank"] == 0
        assert kw["dp_session_id"] == "sess"
        assert spec.arrive_time > 0
        params = kw["sampling_params"]
        assert params.temperature == pytest.approx(0.7)
        assert (params.top_k, params.top_p) == (20, pytest.approx(0.9))
        # `max_tokens` is the request's own, kept beside the shared params.
        assert (spec.max_tokens, params.max_tokens_is_default) == (64, False)
        assert params.stop_strings == ["\n\n"]
        assert (params.n, params.logprobs, params.top_logprobs) == (2, True, 0)
        assert params.required_channel == RequiredChannel(
            after=(1,), opening=(2, 3), opens_at=1, end_ids=(4,)
        )
        assert params.json_schema is None
    assert specs[0].seq_id != specs[1].seq_id


def test_the_rust_clients_control_messages_decode():
    kind, seq_ids, _ = codec.decode_client_message(
        read_frames(FIXTURES / "client_abort.frames")
    )
    assert (kind, seq_ids) == (EngineCoreRequestType.ABORT, [0])
    kind, _, _ = codec.decode_client_message(
        read_frames(FIXTURES / "client_shutdown.frames")
    )
    assert kind == EngineCoreRequestType.SHUTDOWN
