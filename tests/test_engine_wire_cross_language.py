# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The engine wire protocol across languages, against the atomesh client.

The Core's messages, as this side encodes them, are kept in
`tests/fixtures/atom_engine/core_*.msgpack`; the Rust client's tests
(`atom/mesh/src/atom_engine/tests.rs`) feed them to the client. The client's
own messages are kept there as `client_*.msgpack` by those tests; this file
decodes them as an EngineCore does and builds their sequences.

`ATOM_UPDATE_FIXTURES=1` rewrites the `core_*` fixtures (and fails, since a
rewrite is not a pass).
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.model_engine import admission
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.model_engine.zmq_protocol import codec
from atom.model_engine.zmq_protocol import messages as m
from atom.sampling_params import RequiredChannel, SamplingParams

FIXTURES = Path(__file__).parent / "fixtures" / "atom_engine"


def finished_with_logprobs() -> Sequence:
    """Seq 0: prompt [1, 2, 3], completion [5, 6] with its logprobs."""
    seq = Sequence([1, 2, 3], 16, SamplingParams(logprobs=True), id=0)
    for token in (5, 6):
        seq.append_token(token)
    seq.logprobs.extend([-0.5, -1.25])
    seq.leave_reason = "max_tokens"
    return seq


def refused_item() -> m.SubmitItem:
    return m.SubmitItem(seq_id=0, prompt=b"\0" * 12, sampling=codec._DEFAULT_SAMPLING)


def core_messages() -> dict[str, bytes]:
    out = EngineCoreRequestType
    return {
        "core_ready": codec.encode_output(out.READY, {"max_pool_tokens": 4096}),
        "core_stream": codec.encode_output(
            out.STREAM, [(0, RequestOutput(0, [7, 8], True, "eos", None, 3, None))]
        ),
        "core_completions": codec.encode_output(out.ADD, [finished_with_logprobs()]),
        "core_rejected": codec.encode_output(
            out.ADD,
            [codec.rejected(refused_item(), "ValueError: bad", m.REJECT_INVALID)],
        ),
        "core_stopped": codec.encode_output(out.SHUTDOWN),
        "core_metrics": codec.encode_output(out.METRICS, {"mtp": {1: 2}}),
    }


def test_core_messages_match_their_fixtures():
    messages = core_messages()
    if os.environ.get("ATOM_UPDATE_FIXTURES"):
        for name, message in messages.items():
            (FIXTURES / f"{name}.msgpack").write_bytes(message)
        pytest.fail("rewrote the core_* fixtures; rerun without ATOM_UPDATE_FIXTURES")
    for name, message in messages.items():
        assert (FIXTURES / f"{name}.msgpack").read_bytes() == message, (
            f"{name}.msgpack is stale: rerun with ATOM_UPDATE_FIXTURES=1, then "
            "atom/mesh's cargo tests against it"
        )


def test_the_core_fixtures_say_what_the_client_reads():
    def read(name):
        return codec.decode_output((FIXTURES / f"{name}.msgpack").read_bytes())

    assert read("core_ready") == (
        EngineCoreRequestType.READY,
        {"max_pool_tokens": 4096},
    )
    kind, [(seq_id, output)] = read("core_stream")
    assert (kind, seq_id, output.output_tokens, output.finish_reason) == (
        EngineCoreRequestType.STREAM,
        0,
        [7, 8],
        "eos",
    )
    kind, [done] = read("core_completions")
    assert (done.id, done.completion_token_ids, done.logprobs) == (
        0,
        [5, 6],
        [-0.5, -1.25],
    )
    assert (done.num_prompt_tokens, done.leave_reason) == (3, "max_tokens")
    kind, [refused] = read("core_rejected")
    assert refused.rejected and refused.reject_code == m.REJECT_INVALID
    assert refused.leave_reason == "rejected: ValueError: bad"
    assert read("core_stopped") == (EngineCoreRequestType.SHUTDOWN, None)
    assert read("core_metrics") == (EngineCoreRequestType.METRICS, {"mtp": {1: 2}})


def core_context():
    return admission.AdmissionContext(
        SimpleNamespace(
            kv_cache_block_size=16,
            speculative_config=None,
            enable_dp_attention=False,
            hf_config=SimpleNamespace(model_type="llama"),
            end_token_ids=(9,),
            kv_transfer_config=None,
            enable_rapidserve=False,
        )
    )


def test_the_rust_clients_submit_builds_its_sequences():
    kind, items = codec.decode_input((FIXTURES / "client_submit.msgpack").read_bytes())
    assert kind == EngineCoreRequestType.ADD
    assert [item.sibling_index for item in items] == [0, 1]
    context = core_context()
    for item in items:
        assert item.arrive_time > 0
        params = codec.sampling_params(item)
        assert params.temperature == pytest.approx(0.7)
        assert (params.top_k, params.top_p) == (20, pytest.approx(0.9))
        assert (params.max_tokens, params.max_tokens_is_default) == (64, False)
        assert params.stop_strings == ["\n\n"]
        assert (params.n, params.logprobs, params.top_logprobs) == (2, True, 0)
        assert params.required_channel == RequiredChannel(
            after=(1,), opening=(2, 3), opens_at=1, end_ids=(4,)
        )
        assert params.json_schema is None
        seq = admission.build_sequence(item, context)
        assert list(seq.token_ids) == [11, 12, 13]
        assert [list(s) for s in seq.stop_token_sequences] == [[5, 6], [7]]
        # n > 1: only the parent id is the client's.
        assert seq.external_request_id is None
        assert seq.parent_request_id == "req"
        assert seq.needs_independent_noise is True
        assert seq.data_parallel_rank == 0
        assert seq.dp_session_id == "sess"
        assert (seq.max_tokens, seq.return_logprobs) == (64, True)
        # Held to this engine's end ids too.
        assert seq.required_channel.end_ids == (4, 9)
    assert items[0].seq_id != items[1].seq_id


def test_the_rust_clients_control_messages_decode():
    def read(name):
        return codec.decode_input((FIXTURES / f"{name}.msgpack").read_bytes())

    assert read("client_abort") == (EngineCoreRequestType.ABORT, [0])
    assert read("client_shutdown") == (EngineCoreRequestType.SHUTDOWN, None)
