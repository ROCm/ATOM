# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The coordinator routes to every global DP rank, not just its own node's.

Keeping one router over the whole group is what lets the load balancer stay a
global scheduler instead of degrading into one independent balancer per node.
So the routing arrays size to the GLOBAL engine count while `local_engine_count`
keeps meaning "engines spawned on this node".
"""

from types import SimpleNamespace

import pytest

from atom.model_engine.engine_core_mgr import CoreManager


def _bare(*, local_engine_count, global_engine_count=None, strategy="least_requests"):
    """A manager with shared state only -- no engines, no sockets."""
    mgr = CoreManager.__new__(CoreManager)
    mgr._init_shared_state(
        SimpleNamespace(dp_load_balance=strategy),
        label="test",
        local_engine_count=local_engine_count,
        global_engine_count=global_engine_count,
    )
    return mgr


class TestGlobalVsLocalCounts:
    def test_single_node_counts_coincide(self):
        mgr = _bare(local_engine_count=4)
        try:
            assert mgr.local_engine_count == 4
            assert mgr.global_engine_count == 4
            assert mgr._rank_reqs == [0] * 4
        finally:
            mgr.ctx.term()

    def test_coordinator_routes_beyond_its_own_node(self):
        mgr = _bare(local_engine_count=4, global_engine_count=8)
        try:
            assert mgr.local_engine_count == 4
            assert mgr.global_engine_count == 8
            assert mgr._rank_reqs == [0] * 8
            assert mgr._rank_tokens == [0] * 8
        finally:
            mgr.ctx.term()

    def test_global_count_defaults_to_local(self):
        """Callers that predate multi-node pass only local_engine_count."""
        mgr = _bare(local_engine_count=2)
        try:
            assert mgr.global_engine_count == 2
        finally:
            mgr.ctx.term()


class TestRoutingUsesGlobalCount:
    def test_selection_reaches_remote_ranks(self):
        mgr = _bare(local_engine_count=2, global_engine_count=8, strategy="round_robin")
        try:
            picked = {mgr._select_dp_rank_locked() for _ in range(16)}
            assert picked == set(
                range(8)
            ), "round robin must cover every global rank, not only local ones"
        finally:
            mgr.ctx.term()

    def test_least_requests_can_pick_a_remote_rank(self):
        mgr = _bare(local_engine_count=2, global_engine_count=4)
        try:
            for rank in (0, 1):
                mgr._rank_reqs[rank] = 5
            assert mgr._select_dp_rank_locked() in (2, 3)
        finally:
            mgr.ctx.term()

    def test_hint_to_a_remote_rank_is_accepted(self):
        mgr = _bare(local_engine_count=2, global_engine_count=8)
        try:
            seqs = [SimpleNamespace(data_parallel_rank=6, num_prompt_tokens=10)]
            assert mgr._resolve_and_validate_hints(seqs) == [6]
        finally:
            mgr.ctx.term()

    def test_hint_beyond_the_global_group_is_rejected(self):
        mgr = _bare(local_engine_count=2, global_engine_count=8)
        try:
            seqs = [SimpleNamespace(data_parallel_rank=8, num_prompt_tokens=10)]
            with pytest.raises(ValueError, match="Invalid data_parallel_rank=8"):
                mgr._resolve_and_validate_hints(seqs)
        finally:
            mgr.ctx.term()

    def test_reset_resizes_to_the_global_count(self):
        mgr = _bare(local_engine_count=2, global_engine_count=8)
        try:
            mgr._rank_reqs[7] = 3
            mgr.reset_dp_router()
            assert mgr._rank_reqs == [0] * 8
            assert mgr._rank_tokens == [0] * 8
        finally:
            mgr.ctx.term()


class TestOutputPullBindPolarity:
    def test_pull_socket_binds(self):
        """PULL always binds so the engine's PUSH can connect.

        The production code passes bind=True to make_zmq_socket for the PULL
        socket regardless of transport. A regression to bind=False would let
        both ends connect with nobody binding, causing _wait_for_all_ready_signals
        to hang forever. This test pins the polarity at the source level.
        """
        import inspect

        from atom.model_engine.engine_core_mgr import CoreManager

        src = inspect.getsource(CoreManager.__init__)
        # Find the make_zmq_socket call for PULL -- it must have bind=True,
        # never bind=False or bind=socket_plan is not None.
        import re

        pull_calls = re.findall(
            r"make_zmq_socket\([^)]*zmq\.PULL[^)]*\)", src, re.DOTALL
        )
        assert (
            pull_calls
        ), "Expected at least one make_zmq_socket(... zmq.PULL ...) call"
        for call in pull_calls:
            assert "bind=True" in call, (
                f"PULL socket must use bind=True (got: {call!r}). "
                "If both ends connect nobody binds and the server hangs."
            )
            assert "bind=False" not in call
            assert "bind=socket_plan" not in call


def _seq(seq_id=0, tokens=10):
    return SimpleNamespace(id=seq_id, num_prompt_tokens=tokens, stream_callback=None)


def _route(mgr, seqs):
    """Drive the real add_request and report which branch it took.

    Returns ``(shortcut_sends, dispatch_calls)`` -- the first is the rank-0
    shortcut, the second the load-balanced fan-out.
    """
    calls = {"send": [], "dispatch": 0}
    mgr.pp_size = 1
    mgr._send_request = lambda rank, payload: calls["send"].append(rank)
    mgr._dispatch_to_dp_ranks = lambda s: calls.__setitem__(
        "dispatch", calls["dispatch"] + 1
    )
    mgr.add_request(seqs)
    return calls["send"], calls["dispatch"]


class TestAddRequestRoutesOnRoutableCount:
    def test_coordinator_with_one_local_but_many_global_dispatches(self):
        """The shortcut must gate on the routable count, not the local one.

        A coordinator hosting 1 engine but routing to 4 must fan out. Gating on
        local_engine_count would short-circuit every request to rank 0 and idle
        the other 3 -- silently, at 1/4 throughput.
        """
        mgr = _bare(local_engine_count=1, global_engine_count=4)
        try:
            sends, dispatched = _route(mgr, [_seq()])
            assert dispatched == 1, "must fan out across all 4 routable ranks"
            assert sends == [], "must not take the rank-0 shortcut"
        finally:
            mgr.ctx.term()

    def test_genuinely_single_engine_still_takes_the_shortcut(self):
        """The inverse: one routable engine must skip the balancer entirely."""
        mgr = _bare(local_engine_count=1, global_engine_count=1)
        try:
            sends, dispatched = _route(mgr, [_seq()])
            assert sends == [0], "a lone engine should be sent to directly"
            assert dispatched == 0, "no fan-out when there is nowhere to fan out"
        finally:
            mgr.ctx.term()


class TestShutdownSendsToRemoteRanks:
    def test_shutdown_reaches_rank_without_local_process(self):
        """_shutdown_engine_core_rank must send SHUTDOWN to a remote rank.

        On a coordinator, ranks >= len(engine_core_processes) have no local
        process but do have an open control socket. The original code returned
        early for those ranks, leaving remote engines running and their worker
        nodes blocked in proc.join() forever.
        """

        class FakeSocket:
            def __init__(self):
                self.sent = []
                self.closed = False

            def send_multipart(self, frames, copy=False):
                self.sent.append(frames)

        mgr = _bare(local_engine_count=2, global_engine_count=4)
        try:
            # Simulate coordinator state: 2 local processes (indices 0-1),
            # 4 control sockets (indices 0-3 including remote ranks 2-3).
            # We only need to test a remote rank (dp_rank=2, no local process).
            fake_socket = FakeSocket()
            # Populate control_sockets up to rank 2 (remote rank under test).
            # Ranks 0-1 use placeholders; rank 2 is our fake.
            for _ in range(2):
                placeholder = FakeSocket()
                mgr.control_sockets.append(placeholder)
                mgr.control_identities.append(b"id")
            mgr.control_sockets.append(fake_socket)
            mgr.control_identities.append(b"remote-id")

            # _send_control uses send_multipart on control_sockets[dp_rank].
            # Patch it to avoid the lock dance and ROUTER framing complexity.
            import pickle

            from atom.model_engine.engine_core_protocol import EngineCoreRequestType

            sent_payloads = []

            def _fake_send_control(dp_rank, payload, copy=False):
                sent_payloads.append((dp_rank, pickle.loads(payload)))

            mgr._send_control = _fake_send_control

            # dp_rank=2 has no local process (engine_core_processes is empty).
            assert len(mgr.engine_core_processes) == 0
            mgr._shutdown_engine_core_rank(2)

            assert len(sent_payloads) == 1, (
                "_shutdown_engine_core_rank must send SHUTDOWN for a remote rank "
                "that has an open control socket but no local process"
            )
            assert sent_payloads[0][0] == 2
            req_type, _ = sent_payloads[0][1]
            assert req_type == EngineCoreRequestType.SHUTDOWN
        finally:
            mgr.ctx.term()

    def test_dead_local_process_still_gets_no_shutdown(self):
        """The single-node behavior the remote fix must not have changed.

        A local rank whose process already exited was never sent SHUTDOWN, and
        widening the send to cover remote ranks must not start sending to it.
        """
        mgr = _bare(local_engine_count=1, global_engine_count=1)
        try:
            mgr.engine_core_processes = [SimpleNamespace(is_alive=lambda: False)]
            mgr.control_sockets = [SimpleNamespace(closed=False)]
            mgr.control_identities = [b"id"]

            sent = []
            mgr._send_control = lambda rank, payload, copy=False: sent.append(rank)

            mgr._shutdown_engine_core_rank(0)

            assert sent == [], "a dead local process must not be sent SHUTDOWN"
        finally:
            mgr.ctx.term()


class TestLaunchEngineCoreSignature:
    def test_accepts_local_rank_and_addresses(self):
        import inspect

        from atom.model_engine.engine_core_mgr import launch_engine_core

        params = inspect.signature(launch_engine_core).parameters
        assert "local_dp_rank" in params
        assert "addresses" in params

    def test_returns_the_local_rank(self):
        """The caller uses the return value for device placement, which is local."""
        import inspect

        from atom.model_engine.engine_core_mgr import launch_engine_core

        src = inspect.getsource(launch_engine_core)
        assert "return (" in src
        assert "local_dp_rank," in src.split("return (")[-1], (
            "launch_engine_core must return the LOCAL rank; the caller feeds it "
            "to set_device_control_env_var, which indexes this node's GPUs"
        )
