# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CoreManager and EngineCore talking over real ZMQ sockets, no GPU.

Each side runs its real IO code -- CoreManager's send paths and output
thread, EngineCore's input and output threads -- with only the busy loop
replaced by the test reading and writing the Core's queues directly. What the
test checks on the Core side is what the busy loop would have consumed.
"""

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zmq
from aiter_stub import stubbed_aiter

from atom.model_engine.protocol.codec import AdmissionContext, decode_client_message
from atom.model_engine.protocol.requests import CompletionResult, EngineRequest
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    get_exit_sequence,
)
from atom.sampling_params import SamplingParams
from atom.utils import get_open_zmq_inproc_path, get_open_zmq_ipc_path, make_zmq_socket

with stubbed_aiter():
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager

TIMEOUT = 10.0


def _core_config(spec_tokens=0):
    return SimpleNamespace(
        kv_cache_block_size=16,
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=spec_tokens) if spec_tokens else None
        ),
        hf_config=SimpleNamespace(model_type="llama"),
    )


def _manager_config():
    return SimpleNamespace(dp_load_balance="round_robin", tp_world_size=1)


class Link:
    """One CoreManager wired to one EngineCore's IO threads."""

    def __init__(self, core_config):
        addresses = [get_open_zmq_ipc_path() for _ in range(3)]
        in_addr, out_addr, ctrl_addr = addresses

        core = EngineCore.__new__(EngineCore)
        core.label = "Engine Core"
        core.input_queue = queue.Queue()
        core.output_queue = queue.Queue()
        core.utility_queue = queue.Queue()
        core._has_pending_utility = False
        core._admission = AdmissionContext(core_config)
        self.core = core
        self.core_threads = [
            threading.Thread(
                target=core.process_input_sockets,
                args=(in_addr, ctrl_addr),
                daemon=True,
            ),
            threading.Thread(
                target=core.process_output_sockets, args=(out_addr,), daemon=True
            ),
        ]
        for thread in self.core_threads:
            thread.start()

        mgr = CoreManager.__new__(CoreManager)
        mgr._init_shared_state(_manager_config(), label="Mgr", local_engine_count=1)
        mgr.pp_size = 1
        input_socket = make_zmq_socket(mgr.ctx, in_addr, zmq.ROUTER, bind=True)
        identity, _ = input_socket.recv_multipart()
        mgr.input_sockets.append(input_socket)
        mgr.engine_core_identities.append(identity)
        control_socket = make_zmq_socket(mgr.ctx, ctrl_addr, zmq.ROUTER, bind=True)
        identity, _ = control_socket.recv_multipart()
        mgr.control_sockets.append(control_socket)
        mgr.control_identities.append(identity)
        mgr.output_sockets.append(
            make_zmq_socket(mgr.ctx, out_addr, zmq.PULL, bind=True)
        )
        mgr.shutdown_paths.append(get_open_zmq_inproc_path())
        self.mgr = mgr

        core.output_queue.put(("READY", {"max_pool_tokens": 4096}))
        mgr._wait_for_all_ready_signals()
        thread = mgr._create_output_thread(
            0, mgr.output_sockets[0], mgr.shutdown_paths[0]
        )
        thread.start()
        mgr.output_threads.append(thread)
        mgr._finalizer = None
        mgr.async_output_queue = None

    def close(self):
        # The Core's output thread ends on the exit sentinel, after telling the
        # manager it stopped; the input thread ends on the manager's SHUTDOWN.
        self.core.output_queue.put([get_exit_sequence()])
        self.core_threads[1].join(timeout=TIMEOUT)
        self.mgr.close()
        self.core_threads[0].join(timeout=TIMEOUT)
        # close() leaves the context to the process. Its shutdown signal to an
        # output thread that already left is still queued, and a context with
        # a queued message never finishes terminating; drop it here, not in
        # whatever later test the garbage collector happens to run under.
        self.mgr.ctx.destroy(linger=0)


@pytest.fixture
def link():
    link = Link(_core_config())
    yield link
    link.close()


def _request(seq_id, tokens, **kwargs):
    sampling = kwargs.pop("sampling", None) or SamplingParams(max_tokens=4)
    return EngineRequest(
        id=seq_id,
        prompt_token_ids=tokens,
        sampling_params=sampling,
        arrive_time=time.time(),
        **kwargs,
    )


def _until(predicate):
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_ready_reports_pool_capacity(link):
    assert link.mgr.max_pool_tokens == 4096


def test_requests_reach_the_scheduler_as_sequences(link):
    pixels = torch.randn(512, 1024)  # 2 MiB: its own zero-copy frame
    reqs = [
        _request(1, [5, 6, 7]),
        _request(
            2,
            np.arange(70_000, dtype=np.int32),
            multimodal_data={"pixel_values": pixels, "spans": ((0, 3),)},
            cache_seed=99,
            kv_transfer_params={"do_remote_prefill": True},
        ),
    ]
    link.mgr.add_request(reqs)
    seqs = link.core.input_queue.get(timeout=TIMEOUT)
    assert [type(seq) for seq in seqs] == [Sequence, Sequence]
    assert [seq.id for seq in seqs] == [1, 2]
    assert list(seqs[0].token_ids) == [5, 6, 7]
    assert list(seqs[1].token_ids) == list(range(70_000))
    assert seqs[1].max_tokens == 4 and seqs[1].block_size == 16
    assert torch.equal(seqs[1].multimodal_data["pixel_values"], pixels)
    assert seqs[1].multimodal_data["spans"] == ((0, 3),)
    assert seqs[1].cache_seed == 99
    assert all(seq.status == SequenceStatus.WAITING for seq in seqs)
    # Stamped on receipt, as before.
    assert all(seq.queue_timing is not None for seq in seqs)
    assert seqs[1].queue_timing.is_pd


def test_streams_reach_their_callbacks(link):
    got = []
    link.mgr.add_request([_request(3, [1], stream_callback=got.append)])
    link.core.input_queue.get(timeout=TIMEOUT)
    outputs = [
        RequestOutput(request_id=3, output_tokens=[9], finished=False),
        RequestOutput(
            request_id=3, output_tokens=[10], finished=True, finish_reason="stop"
        ),
    ]
    link.core.output_queue.put(("STREAM", [(3, outputs[0])]))
    link.core.output_queue.put(("STREAM", [(3, outputs[1])]))
    _until(lambda: len(got) == 2)
    assert got == outputs
    assert 3 not in link.mgr._seq_id_to_callback


def test_finished_sequences_come_back_as_results(link):
    link.mgr.add_request([_request(4, [1, 2])])
    (seq,) = link.core.input_queue.get(timeout=TIMEOUT)
    for token in (7, 8):
        seq.append_token(token)
    seq.leave_reason = "eos"
    seq.first_token_time = seq.arrive_time + 0.5
    link.core.output_queue.put([seq])
    (result,) = link.mgr.get_output()
    assert isinstance(result, CompletionResult)
    assert result.id == 4 and list(result.completion_token_ids) == [7, 8]
    assert result.leave_reason == "eos" and result.num_prompt_tokens == 2


def test_a_request_the_core_cannot_build_is_rejected_to_its_client():
    # Logprobs under speculative decoding: Sequence refuses to build it.
    link = Link(_core_config(spec_tokens=2))
    try:
        got = []
        link.mgr.add_request(
            [
                _request(
                    5,
                    [1],
                    sampling=SamplingParams(logprobs=True),
                    stream_callback=got.append,
                ),
                _request(6, [2]),
            ]
        )
        (seq,) = link.core.input_queue.get(timeout=TIMEOUT)
        assert seq.id == 6  # its batch-mate is admitted
        (result,) = link.mgr.get_output()
        assert result.id == 5 and result.leave_reason.startswith("rejected: ")
        _until(lambda: len(got) == 1)
        assert got[0].finished and got[0].finish_reason.startswith("rejected: ")
    finally:
        link.close()


def test_abort_reaches_the_utility_queue_as_before(link):
    link.mgr.abort_request(11)
    assert link.core.utility_queue.get(timeout=TIMEOUT) == (
        "abort_request",
        {"cmd": "abort_request", "req_id": 11},
    )
    assert link.core._has_pending_utility


@pytest.mark.parametrize(
    "cmd,result",
    [
        # Typed: AdminCommand out, AdminReply back.
        ("get_mtp_statistics", {"enabled": False}),
        # Python-only: a pickled Utility out, a pickled UtilityReply back.
        ("get_weight_sync_status", {"fenced": False, "sync_id": None}),
    ],
)
def test_synchronous_commands_roundtrip(link, cmd, result):
    def respond():
        got, payload = link.core.utility_queue.get(timeout=TIMEOUT)
        assert got == cmd and payload["cmd"] == cmd
        assert set(payload) == {"cmd", "request_id", "timeout", "deadline"}
        link.core.output_queue.put(
            (
                "UTILITY_RESPONSE",
                {"cmd": cmd, "result": result, "request_id": payload["request_id"]},
            )
        )

    responder = threading.Thread(target=respond)
    responder.start()
    (reply,) = link.mgr.broadcast_utility_command_sync(cmd, timeout=TIMEOUT)
    responder.join()
    assert reply["result"] == result


def test_fire_and_forget_commands_reach_the_utility_queue(link):
    link.mgr.send_utility_command("get_mtp_stats")
    assert link.core.utility_queue.get(timeout=TIMEOUT) == (
        "get_mtp_stats",
        {"cmd": "get_mtp_stats"},
    )


def test_metrics_land_in_latest_metrics(link):
    snapshot = {"enabled": True, "running": 2, "kv": {"used": 0.5}}
    link.core.output_queue.put(("METRICS", snapshot))
    _until(lambda: link.mgr.latest_metrics.get(0) == snapshot)


def test_an_undecodable_message_is_dropped_and_serving_continues(link):
    link.mgr._send_request(0, [b"\xffnot an envelope"])
    link.mgr.add_request([_request(12, [3])])
    (seq,) = link.core.input_queue.get(timeout=TIMEOUT)
    assert seq.id == 12


def test_shutdown_reaches_the_input_queue(link):
    link.mgr._shutdown_engine_core_rank(0)
    (seq,) = link.core.input_queue.get(timeout=TIMEOUT)
    assert seq.status == SequenceStatus.EXIT_ENGINE
    link.core_threads[0].join(timeout=TIMEOUT)
    assert not link.core_threads[0].is_alive()


def test_disagg_prefill_half_gets_one_token():
    sent = {}
    mgr = DisaggCoreManager.__new__(DisaggCoreManager)
    mgr.label = "DisaggCoreManager"
    mgr._seq_id_to_callback = {}
    mgr._send_request = lambda rank, frames: sent.__setitem__(rank, frames)
    req = _request(20, [1, 2, 3], sampling=SamplingParams(max_tokens=50))
    mgr.add_request([req])
    (decode,) = decode_client_message(sent[1])[1]
    (prefill,) = decode_client_message(sent[0])[1]
    assert decode.kwargs["sampling_params"].max_tokens == 50
    assert prefill.kwargs["sampling_params"].max_tokens == 1
    assert req.max_tokens == 50
