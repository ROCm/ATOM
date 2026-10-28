# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the MoRIIO KV connector's transfer bookkeeping.

The connector module pulls in aiter, which resolves the gfx arch from rocminfo
while importing, so it only loads on a GPU host. The behavioural tests run
there and skip elsewhere. So that the CPU lane still catches these regressions,
the pure-Python functions are also lifted out of the source and executed on
their own, and the rest is asserted against the source.
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = ROOT / "atom/kv_transfer/disaggregation/moriio/moriio_connector.py"
COMMON = ROOT / "atom/kv_transfer/disaggregation/moriio/moriio_common.py"


def load_function(path: Path, name: str, namespace: dict | None = None):
    """Execute a single top-level function from a module we cannot import."""
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            scope: dict = dict(namespace or {})
            # Our own source, lifted so the CPU lane can run it.
            code = compile(ast.Module([node], []), str(path), "exec")
            exec(code, scope)  # noqa: S102
            return scope[name]
    raise AssertionError(f"{name} not found in {path}")


class PortOffsetTest(unittest.TestCase):
    """Side-channel ports must not collide across DP ranks."""

    @classmethod
    def setUpClass(cls):
        cls.get_port_offset = staticmethod(load_function(COMMON, "get_port_offset"))

    def test_tp_size_strides_the_dp_ranks_apart(self):
        offset = self.get_port_offset
        self.assertEqual(offset(0, 0, 8), 0)
        self.assertEqual(offset(0, 7, 8), 7)
        # Without the stride, dp1/tp0 would land on dp0/tp1's port.
        self.assertEqual(offset(1, 0, 8), 8)
        self.assertEqual(offset(2, 3, 4), 11)

    def test_every_rank_of_a_two_dp_group_is_unique(self):
        offset = self.get_port_offset
        ports = [offset(dp, tp, 8) for dp in range(2) for tp in range(8)]
        self.assertEqual(len(set(ports)), 16)

    def test_default_tp_size_still_collides(self):
        # Pinned so a caller that omits tp_size is a visible choice, not a
        # silent one.
        offset = self.get_port_offset
        self.assertEqual(offset(1, 0), offset(0, 1))


class ConnectorSourceTest(unittest.TestCase):
    """Guards that need torch or numpy to execute, pinned at the source."""

    @classmethod
    def setUpClass(cls):
        cls.source = CONNECTOR.read_text()

    def segment(self, name: str) -> str:
        tree = ast.parse(self.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(self.source, node) or ""
        raise AssertionError(f"{name} not found")

    def test_every_port_offset_call_passes_a_size(self):
        tree = ast.parse(self.source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_port_offset"
        ]
        # Side channel, read notify, early-failure notify, handshake, write
        # request, write report.
        self.assertEqual(len(calls), 6)
        for call in calls:
            self.assertEqual(len(call.args), 3, ast.unparse(call))

    def test_a_short_remote_list_is_caught_before_the_pairing(self):
        body = self.segment("_read_blocks")
        guard = body.index("self._source_covers(")
        pairing = body.index("self._group_offsets(")
        # The pairing silently drops the tail, which costs the end of the
        # prompt's KV and surfaces only as degenerate decode output.
        self.assertLess(guard, pairing)
        self.assertIn("raise ValueError", self.segment("_source_covers"))
        self.assertIn(
            "zip(local_block_ids, remote_block_ids)", self.segment("_group_offsets")
        )

    def test_a_short_source_is_caught_on_both_ends_of_a_push(self):
        # The consumer refuses before asking, the producer before writing.
        self.assertIn("self._source_covers(", self.segment("_request_write"))
        body = self.segment("_write_blocks")
        self.assertLess(
            body.index("self._source_covers("), body.index("self._group_offsets(")
        )

    def test_layer_shapes_are_checked_against_the_offset_source(self):
        self.assertIn("other.shape != cache_tensor.shape", self.segment("_kv_layout"))
        # Both directions take their offsets from the same checked layout.
        self.assertIn("self._kv_layout()", self.segment("_read_blocks"))
        self.assertIn("self._kv_layout()", self.segment("_write_blocks"))

    def test_the_listener_routes_both_write_messages(self):
        body = self.segment("_handshake_listener")
        self.assertIn("MoRIIOConstants.WRITE_REQ", body)
        self.assertIn("MoRIIOConstants.WRITE_DONE", body)
        # The listener thread must not touch the RDMA engine.
        self.assertIn("self._write_requests.put(", body)
        self.assertNotIn("_write_blocks", body)

    def test_a_bad_write_frame_cannot_kill_the_listener(self):
        body = self.segment("_handshake_listener")
        writes = body[body.index("MoRIIOConstants.WRITE_REQ") :]
        # The same thread serves every peer's handshake, so a bad frame is
        # logged and dropped instead of raised.
        self.assertIn("Dropping undecodable WRITE_REQ", writes)
        self.assertIn("Dropping undecodable WRITE_DONE", writes)

    def test_get_finished_serves_pushes_before_reporting_sends(self):
        body = self.segment("get_finished")
        # A push that completes this step should free its blocks this step.
        self.assertLess(
            body.index("self._service_write_requests()"),
            body.index("done_sending, self.done_sending = self.done_sending, set()"),
        )
        # Consumer reports feed the same verify pass as finished reads.
        self.assertLess(
            body.index("self._collect_write_reports("),
            body.index("self._verify_landed(req_id)"),
        )

    def test_start_load_kv_never_waits_for_a_handshake(self):
        body = self.segment("start_load_kv")
        # It runs on the worker's busy loop; get_finished issues the read.
        self.assertNotIn("_ready_requests.get", body)
        self.assertIn("self._awaiting_handshake[req_id] =", body)
        self.assertIn("self._issue_ready_reads()", self.segment("get_finished"))

    def test_handshake_guard_uses_the_dp_suffixed_id(self):
        body = self.segment("start_load_kv")
        # _remote_agents is keyed by the dp-suffixed id, so testing the bare
        # engine id would re-run the handshake for every request.
        self.assertIn("known = peer in self._remote_agents", body)
        self.assertNotIn("remote_engine_id in self._remote_agents", body)

    def test_handshake_wait_is_bounded(self):
        body = self.segment("_issue_ready_reads")
        self.assertIn("now >= deadline", body)
        self.assertIn("self._fail_before_transfer(req_id, meta)", body)
        self.assertIn("zmq.RCVTIMEO", self.segment("_execute_handshake"))

    def test_write_mode_refuses_a_zero_hold(self):
        body = self.segment("__init__")
        self.assertIn("ATOM_PD_FAILED_WRITE_HOLD_S <= 0", body)
        self.assertIn("raise ValueError", body)

    def test_registration_uses_the_transfer_layout(self):
        self.assertIn("self._kv_layout()", self.segment("register_kv_caches"))

    def test_done_sending_is_swapped_under_its_lock(self):
        for name in ("get_finished", "_handshake_listener", "_finish_write"):
            self.assertIn("with self._done_sending_lock:", self.segment(name), name)

    def test_unimplemented_transfer_subsets_are_not_ignored(self):
        body = self.segment("_rejects_transfer_subset")
        self.assertIn("num_computed_blocks", body)
        self.assertIn("src_block_skip_factor", body)
        for path in ("_issue_read_for_req", "_request_write"):
            caller = self.segment(path)
            self.assertIn("self._rejects_transfer_subset(", caller, path)
            self.assertIn("_fail_before_transfer", caller, path)

    def test_a_rejected_request_does_not_take_down_the_rank(self):
        body = self.segment("_issue_read_for_req")
        # start_load_kv runs on the worker's busy_loop, so an escaping exception
        # takes down the async proc of the rank, not just the request.
        self.assertIn("except Exception:", body)
        self.assertNotIn("raise ValueError", body)
        # And the failure has to reach the scheduler, or the request hangs.
        self.assertIn("_failed_before_read", self.segment("get_finished"))

    def test_remote_descriptors_are_compared_against_our_own(self):
        body = self.segment("_get_or_build_sessions")
        # Echoed-back descriptors make every read a local copy that reports
        # SUCCESS while the destination never changes.
        self.assertIn("local_metas == remote_metas", body)
        check = body[body.index("local_metas == remote_metas") :]
        self.assertIn("raise ValueError", check[:200])

    def test_env_vars_are_defined_and_documented(self):
        self.assertNotIn("os.environ", self.source)
        defined = (ROOT / "atom/utils/envs.py").read_text()
        documented = (ROOT / "docs/environment_variables.md").read_text()
        for suffix in ("MODE", "HANDSHAKE_TIMEOUT_S", "WRITE_TIMEOUT_S", "VERIFY"):
            name = f"ATOM_MORIIO_{suffix}"
            self.assertIn(f"envs.{name}", self.source)
            self.assertIn(f'"{name}"', defined)
            self.assertIn(f"**{name}**", documented)

    def test_engine_key_is_unique_per_process(self):
        # MoRI reuses the queue pairs of a key it knows, so a restarted peer
        # has to come back under a new one.
        self.assertIn("uuid.uuid4()", self.segment("__init__"))

    def test_verify_mode_is_opt_in(self):
        self.assertIn("envs.ATOM_MORIIO_VERIFY", self.source)
        body = self.segment("get_finished")
        # Digesting under the wrapper lock would serialise the step loop
        # against every in-flight transfer.
        self.assertIn("if self._verify_kv:", body)
        self.assertIn("_verify_landed", body)

    def test_verify_failure_is_reported_as_an_error(self):
        body = self.segment("_verify_landed")
        self.assertIn("logger.error", body)
        self.assertIn("unchanged", body)


def import_connector():
    """The connector class, or None where it cannot be imported.

    aiter resolves the gfx arch from rocminfo while importing, so this only
    succeeds in a container that can see the GPUs.
    """
    try:
        from atom.kv_transfer.disaggregation.moriio.moriio_connector import (
            MoRIIOConnector,
        )
    # Broad on purpose: whatever stops the native stack loading means skip,
    # and an unexpected type must not turn the CPU lane red.
    except Exception:  # noqa: BLE001
        return None
    return MoRIIOConnector


class ConnectorBehaviourTest(unittest.TestCase):
    """The same guards as above, executed rather than read."""

    @classmethod
    def setUpClass(cls):
        cls.connector = import_connector()
        if cls.connector is None:
            raise unittest.SkipTest("moriio_connector needs aiter, torch and a GPU")
        import torch

        cls.torch = torch

    def make_connector(self, layers: int = 3, blocks: int = 4, block_size: int = 2):
        """A connector with only the attributes the verify path reads."""
        import types

        conn = object.__new__(self.connector)
        names = [f"layer.{i}" for i in range(layers)]
        conn.kv_cache_block_size = block_size
        conn.layer_name_to_local_kv_cache_metadata = {name: {} for name in names}
        conn.kv_caches = {
            name: types.SimpleNamespace(
                k_cache=self.torch.zeros(
                    (blocks * block_size, 4), dtype=self.torch.uint8
                )
            )
            for name in names
        }
        conn._verify_rows = {}
        conn._verify_before = {}
        return conn, names

    def read_blocks(self, local, remote):
        object.__new__(self.connector)._read_blocks(
            local_block_ids=local,
            remote_block_ids=remote,
            dst_engine_id="engine",
            request_id="req",
            remote_host="host",
            remote_handshake_port=1,
        )

    def test_too_few_remote_blocks_is_rejected_before_anything_is_read(self):
        with self.assertRaises(ValueError) as caught:
            self.read_blocks([1, 2, 3], [1, 2])
        # A bare instance has no tp_rank, so reaching this message at all also
        # shows the guard runs before the first attribute access.
        self.assertIn(
            "3 destination blocks but only 2 source blocks", str(caught.exception)
        )

    def test_a_longer_remote_list_is_accepted(self):
        # The producer's block table also covers the token it generated, which
        # the consumer never allocates for, so this is the ordinary case.
        with self.assertRaises(AttributeError):
            self.read_blocks([1, 2], [1, 2, 3])

    def test_rows_index_blocks_directly_when_there_is_no_mla(self):
        conn, _ = self.make_connector()
        rows = conn._verify_rows_for([1, 3], is_mla=False)
        self.assertEqual(rows.tolist(), [1, 3])

    def test_mla_rows_expand_each_block_into_its_tokens(self):
        conn, _ = self.make_connector(block_size=2)
        # dim 0 is tokens under MLA, so block 1 is rows 2 and 3.
        rows = conn._verify_rows_for([1, 3], is_mla=True)
        self.assertEqual(rows.tolist(), [2, 3, 6, 7])

    def test_a_layer_that_never_changed_is_reported(self):
        conn, names = self.make_connector()
        rows = conn._verify_rows_for([0, 1], is_mla=False)
        conn._verify_rows["req"] = rows
        conn._verify_before["req"] = conn._verify_digest(rows)
        # Two layers land, the third is left as the transfer found it.
        for name in names[:2]:
            conn.kv_caches[name].k_cache[rows] = 7

        with self.assertLogs("atom", "ERROR") as logs:
            conn._verify_landed("req")
        self.assertIn("1 of 3 layers unchanged", logs.output[0])
        self.assertIn(names[2], logs.output[0])

    def test_a_fully_landed_read_is_not_reported_as_an_error(self):
        conn, names = self.make_connector()
        rows = conn._verify_rows_for([0, 1], is_mla=False)
        conn._verify_rows["req"] = rows
        conn._verify_before["req"] = conn._verify_digest(rows)
        for name in names:
            conn.kv_caches[name].k_cache[rows] = 7

        with self.assertLogs("atom", "INFO") as logs:
            conn._verify_landed("req")
        self.assertIn("all 3 layers changed", logs.output[0])
        # Both maps are popped, or a long run accumulates a digest per request.
        self.assertEqual(conn._verify_rows, {})
        self.assertEqual(conn._verify_before, {})

    def test_digest_survives_a_dtype_without_arithmetic(self):
        conn, names = self.make_connector()
        # The PD cells use an fp8 KV cache, which cannot be summed natively,
        # so the digest reads the bytes instead.
        for name in names:
            conn.kv_caches[name].k_cache = self.torch.zeros(
                (8, 4), dtype=self.torch.float8_e4m3fnuz
            )
        rows = conn._verify_rows_for([0, 1], is_mla=False)
        before = conn._verify_digest(rows)
        conn.kv_caches[names[0]].k_cache[rows] = 1.0
        after = conn._verify_digest(rows)
        self.assertNotEqual(before[names[0]], after[names[0]])
        self.assertEqual(before[names[1]], after[names[1]])

    def test_a_bad_request_is_recorded_rather_than_raised(self):
        import threading
        import types

        conn = object.__new__(self.connector)
        notified = []
        conn._failed_before_read = set()
        conn._recving_transfers = {}
        conn.request_id_to_transfer_id = {}
        conn.moriio_wrapper = types.SimpleNamespace(
            lock=threading.Lock(),
            send_notify=lambda transfer_id, host, port: notified.append(transfer_id),
        )
        conn.tp_rank = 0
        meta = types.SimpleNamespace(
            num_computed_blocks=0,
            src_block_skip_factor=1,
            remote_engine_id="engine",
            local_block_ids=[1, 2, 3],
            remote_block_ids=[1, 2],
            remote_host="host",
            remote_handshake_port=1,
            remote_dp_rank=0,
            tp_size=8,
            transfer_id=5,
        )
        # No exception: the rank survives and the scheduler hears about it.
        conn._issue_read_for_req("req", meta)
        self.assertEqual(conn._failed_before_read, {"req"})
        # And the producer is told to release the source blocks.
        self.assertEqual(notified, [5])

    def test_digest_tells_apart_payloads_with_the_same_byte_sum(self):
        conn, names = self.make_connector(layers=1)
        cache = conn.kv_caches[names[0]].k_cache
        rows = conn._verify_rows_for([0, 1], is_mla=False)
        cache[0] = self.torch.tensor([1, 2, 3, 4], dtype=self.torch.uint8)
        cache[1] = self.torch.tensor([5, 6, 7, 8], dtype=self.torch.uint8)
        before = conn._verify_digest(rows)[names[0]]
        plain_before = int(cache[rows].sum())

        # The same bytes in a different order: what a reused block looks like
        # when new KV happens to total what the old KV did.
        cache[0] = self.torch.tensor([8, 7, 6, 5], dtype=self.torch.uint8)
        cache[1] = self.torch.tensor([4, 3, 2, 1], dtype=self.torch.uint8)
        self.assertEqual(int(cache[rows].sum()), plain_before)
        self.assertNotEqual(conn._verify_digest(rows)[names[0]], before)

    def test_merge_is_callable_on_the_real_class(self):
        self.assertEqual(
            self.connector.merge_contiguous_blocks(
                None, [0, 100], [0, 100], [100, 100]
            ),
            ([0], [0], [200]),
        )


class FakeStatus:
    """Stands in for mori's TransferStatus; ``state`` is flipped by the test."""

    def __init__(self, state: str = "progress"):
        self.state = state

    def Failed(self):
        return self.state == "failed"

    def Succeeded(self):
        return self.state == "ok"

    def Message(self):
        return "fake"

    def Code(self):
        return self.state


class WritePathTest(unittest.TestCase):
    """Serial WRITE mode, both ends, with the RDMA and ZMQ edges stubbed."""

    @classmethod
    def setUpClass(cls):
        cls.connector = import_connector()
        if cls.connector is None:
            raise unittest.SkipTest("moriio_connector needs aiter, torch and a GPU")
        import msgpack
        import torch

        from atom.kv_transfer.disaggregation.moriio.moriio_common import (
            MoRIIOConstants,
        )

        cls.msgpack = msgpack
        cls.torch = torch
        cls.constants = MoRIIOConstants

    def make(self, *, producer: bool, layers: int = 2, blocks: int = 8):
        """A connector holding only what the write path reads.

        MLA layout with block_size 2 over 4-byte rows, so a block is 8 bytes,
        and 4 blocks per registered chunk so a block id can cross chunks.
        """
        import collections
        import queue
        import threading
        import types

        conn = object.__new__(self.connector)
        names = [f"layer_{i}" for i in range(layers)]
        conn.is_producer = producer
        conn.transfer_mode = "write"
        conn.tp_rank = 1
        conn.tp_size = 8
        conn.dp_rank = 0
        conn.local_ip = "10.0.0.2"
        conn.base_handshake_port = 7400
        conn.kv_cache_block_size = 2
        conn.blocks_per_chunk = 4
        conn.num_k_chunks = 2
        conn._logged_layout = True
        conn._verify_kv = False
        conn._verify_rows = {}
        conn._verify_before = {}
        conn._failed_before_read = set()
        conn._region_layers = {}
        conn.request_id_to_transfer_id = {}
        conn.layer_name_to_local_kv_cache_metadata = {name: [] for name in names}
        conn.kv_caches = {
            name: types.SimpleNamespace(
                k_cache=self.torch.zeros((blocks * 2, 1, 4), dtype=self.torch.uint8),
                v_cache=None,
            )
            for name in names
        }
        conn._write_requests = queue.Queue()
        conn._pending_writes = []
        conn._write_handshakes = {}
        conn._sending_writes = {}
        conn._awaiting_write = {}
        conn._write_reports = {}
        conn._write_reports_lock = threading.Lock()
        conn._write_timeout_s = 600.0
        conn._handshake_timeout_s = 300.0
        conn._handshake_lock = threading.RLock()
        conn._remote_agents = {}
        conn._read_handshakes = {}
        conn._awaiting_handshake = {}
        conn._ready_requests = queue.Queue()
        conn._recving_transfers = {}
        conn._recving_transfers_callback_addr = {}
        conn._held_failures = {}
        conn._built_sessions = {}
        conn._retired_sessions = []
        conn.remote_moriio_metadata = {}
        conn.layer_name_to_remote_kv_cache_metadata = {}
        conn.done_sending = set()
        conn._done_sending_lock = threading.Lock()
        conn._write_ids = {}
        conn._finished_writes = collections.deque()

        conn.sent = []
        conn.writes = []
        conn.notified = []

        def write(szs, local_offsets, remote_offsets, session):
            conn.writes.append((session, local_offsets, remote_offsets, szs))
            return FakeStatus()

        conn.moriio_wrapper = types.SimpleNamespace(
            lock=threading.Lock(),
            send_message=lambda frames, host, port: conn.sent.append(
                (frames, host, port)
            ),
            send_notify=lambda transfer_id, host, port: conn.notified.append(
                (transfer_id, host, port)
            ),
            write_remote_data_status=write,
        )
        return conn

    def recv_meta(self, local, remote):
        import types

        return types.SimpleNamespace(
            num_computed_blocks=0,
            src_block_skip_factor=1,
            local_block_ids=local,
            remote_block_ids=remote,
            remote_host="10.0.0.1",
            remote_handshake_port=7301,
            remote_dp_rank=0,
            remote_dp_size=1,
            tp_size=8,
            transfer_id=0,
        )

    def push_request(self, **overrides):
        request = {
            "transfer_id": 9,
            "decode_req_id": "d1",
            "decode_host": "10.0.0.2",
            "decode_handshake_port": 7400,
            "decode_dp_rank": 0,
            "decode_tp_size": 8,
            "dst_block_ids": [1],
            "src_block_ids": [1],
        }
        request.update(overrides)
        return request

    def test_mode_defaults_to_read(self):
        mode = self.connector._transfer_mode_from
        self.assertEqual(mode({}, env_mode=""), "read")
        self.assertEqual(mode({"moriio_mode": "WRITE"}, env_mode=""), "write")
        # The env var wins, so a run can flip modes without a new config.
        self.assertEqual(mode({"moriio_mode": "write"}, env_mode="read"), "read")
        with self.assertRaises(ValueError):
            mode({"moriio_mode": "push"}, env_mode="")

    def test_offsets_are_grouped_by_chunk_pair(self):
        conn = self.make(producer=False)
        groups = conn._group_offsets([1, 5, 6], [6, 2, 3], 10)
        self.assertEqual(
            groups,
            {
                (0, 1): ([10], [20], [10]),
                (1, 0): ([10, 20], [20, 30], [10, 10]),
            },
        )

    def test_the_consumer_asks_its_producer_rank_for_a_push(self):
        import time

        conn = self.make(producer=False)
        conn.request_id_to_transfer_id = {"d1": 77}
        conn._request_write("d1", self.recv_meta([4, 5], [10, 11, 12]))

        ((frames, host, port),) = conn.sent
        self.assertEqual(frames[0], self.constants.WRITE_REQ)
        # Rank 1 of a TP8 producer listens one port above its base.
        self.assertEqual((host, port), ("10.0.0.1", 7302))
        request = self.msgpack.loads(frames[1])
        self.assertEqual(request["transfer_id"], 77)
        self.assertEqual(request["decode_req_id"], "d1")
        self.assertEqual(request["dst_block_ids"], [4, 5])
        # The block holding the producer's generated token stays behind.
        self.assertEqual(request["src_block_ids"], [10, 11])
        self.assertEqual(
            (request["decode_host"], request["decode_handshake_port"]),
            ("10.0.0.2", 7400),
        )
        self.assertAlmostEqual(request["deadline"], time.time() + 600, delta=60)
        self.assertEqual(conn._awaiting_write["d1"][0], request["deadline"])

    def test_the_consumer_refuses_a_short_source_without_asking(self):
        conn = self.make(producer=False)
        conn._request_write("d1", self.recv_meta([4, 5, 6], [10, 11]))
        self.assertEqual(conn.sent, [])
        self.assertEqual(conn._failed_before_read, {"d1"})
        # Rank 1 of the producer still holds the source blocks; let it go.
        self.assertEqual(conn.notified, [(0, "10.0.0.1", 7302)])

    def test_an_unreachable_producer_fails_the_request_not_the_rank(self):
        conn = self.make(producer=False)

        def unreachable(*_args):
            raise RuntimeError("peer gone")

        conn.moriio_wrapper.send_message = unreachable
        conn._request_write("d1", self.recv_meta([4, 5], [10, 11]))
        self.assertEqual(conn._failed_before_read, {"d1"})
        self.assertNotIn("d1", conn._awaiting_write)

    def test_the_producer_writes_its_blocks_into_the_consumers(self):
        conn = self.make(producer=True)
        sessions = [({(0, 0): f"L{i}-00", (0, 1): f"L{i}-01"}, {}) for i in range(2)]
        conn._get_or_build_sessions = lambda engine_id: (sessions, None)
        request = self.push_request(dst_block_ids=[5, 1], src_block_ids=[2, 3, 7])

        statuses = conn._write_blocks(request, "engine")

        # src 2 -> dst 5 crosses into the consumer's second chunk; src 3 -> dst
        # 1 stays in the first. Local offsets are ours, remote theirs, and src
        # block 7 is never touched.
        self.assertEqual(len(statuses), 4)
        self.assertEqual(
            sorted((s, l, r) for s, l, r, _ in conn.writes),
            sorted(
                [
                    ("L0-01", [16], [8]),
                    ("L0-00", [24], [8]),
                    ("L1-01", [16], [8]),
                    ("L1-00", [24], [8]),
                ]
            ),
        )

    def test_the_producer_refuses_a_short_source(self):
        conn = self.make(producer=True)
        conn._get_or_build_sessions = lambda engine_id: ([], None)
        with self.assertRaises(ValueError):
            conn._write_blocks(
                self.push_request(dst_block_ids=[1, 2], src_block_ids=[1]), "engine"
            )

    def test_a_finished_push_is_reported_and_frees_the_source(self):
        conn = self.make(producer=True)
        statuses = [FakeStatus(), FakeStatus()]
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: statuses
        conn._write_requests.put(self.push_request())

        conn._service_write_requests()
        self.assertIn(9, conn._sending_writes)
        self.assertEqual(conn.done_sending, set())
        self.assertEqual(conn.sent, [])

        for status in statuses:
            status.state = "ok"
        conn._service_write_requests()
        self.assertEqual(conn.done_sending, {9})
        self.assertEqual(conn._sending_writes, {})
        ((frames, host, port),) = conn.sent
        self.assertEqual(frames[0], self.constants.WRITE_DONE)
        self.assertEqual(
            self.msgpack.loads(frames[1]), {"decode_req_id": "d1", "ok": True}
        )
        # Back to rank 1 of the consumer, the rank that asked.
        self.assertEqual((host, port), ("10.0.0.2", 7401))

    def test_a_failed_push_still_frees_the_source(self):
        conn = self.make(producer=True)
        conn._remote_agents = {"10.0.0.2:7400_dp0": {"agent"}}
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: [
            FakeStatus("ok"),
            FakeStatus("failed"),
        ]
        conn._write_requests.put(self.push_request())
        conn._service_write_requests()
        self.assertEqual(conn.done_sending, {9})
        ((frames, _host, _port),) = conn.sent
        self.assertEqual(self.msgpack.loads(frames[1])["ok"], False)
        # Re-handshaken by the next push, in case the consumer restarted.
        self.assertEqual(conn._remote_agents, {})

    def test_a_malformed_push_request_is_dropped_not_raised(self):
        conn = self.make(producer=True)
        for bad in (
            {},
            {"transfer_id": "not-an-id"},
            ["not", "a", "dict"],
            self.push_request(decode_host=None),
        ):
            conn._write_requests.put(bad)
        # Raising here would take down the producer rank's worker loop.
        with self.assertLogs("atom", "ERROR") as logs:
            conn._service_write_requests()
        self.assertEqual(len(logs.output), 4)
        self.assertEqual(conn._pending_writes, [])
        self.assertEqual(conn.sent, [])
        self.assertEqual(conn.done_sending, set())

    def test_a_push_waits_for_the_handshake_but_not_forever(self):
        import time

        conn = self.make(producer=True)
        conn._write_target_ready = lambda request: None
        conn._write_requests.put(self.push_request())
        conn._service_write_requests()
        self.assertEqual(len(conn._pending_writes), 1)
        self.assertEqual(conn.sent, [])

        request, _queued = conn._pending_writes[0]
        conn._pending_writes = [(request, time.monotonic() - 1000)]
        conn._service_write_requests()
        self.assertEqual(conn._pending_writes, [])
        self.assertEqual(conn.done_sending, {9})
        ((frames, _host, _port),) = conn.sent
        self.assertEqual(self.msgpack.loads(frames[1])["ok"], False)

    def test_the_handshake_runs_once_and_off_the_worker_thread(self):
        import types
        from concurrent.futures import Future

        conn = self.make(producer=True)
        future = Future()
        submitted = []
        conn._handshake_executor = types.SimpleNamespace(
            submit=lambda fn, *args: submitted.append(args) or future
        )
        request = self.push_request()

        self.assertIsNone(conn._write_target_ready(request))
        self.assertIsNone(conn._write_target_ready(request))
        self.assertEqual(submitted, [("10.0.0.2", 7400, 8, "10.0.0.2:7400_dp0", 0)])

        future.set_result({"agent"})
        self.assertTrue(conn._write_target_ready(request))
        self.assertEqual(conn._remote_agents["10.0.0.2:7400_dp0"], {"agent"})
        self.assertTrue(conn._write_target_ready(request))
        self.assertEqual(len(submitted), 1)

    def test_a_failed_handshake_is_retried_by_the_next_request(self):
        import types
        from concurrent.futures import Future

        conn = self.make(producer=True)
        futures = [Future(), Future()]
        conn._handshake_executor = types.SimpleNamespace(
            submit=lambda fn, *args: futures.pop(0)
        )
        request = self.push_request()
        first = futures[0]
        self.assertIsNone(conn._write_target_ready(request))
        first.set_exception(RuntimeError("consumer not up"))
        self.assertFalse(conn._write_target_ready(request))
        # Not cached as failed: the next call submits a fresh handshake.
        self.assertIsNone(conn._write_target_ready(request))
        self.assertEqual(futures, [])

    def test_the_consumer_turns_reports_into_finished_and_failed(self):
        import time

        conn = self.make(producer=False)
        meta = self.recv_meta([4], [10])
        now = time.time()
        conn._awaiting_write = {
            "a": (now + 600, meta),
            "b": (now + 600, meta),
            "late": (now - 10_000, meta),
            "just_late": (now - 1, meta),
        }
        conn._write_reports = {"a": True, "b": False, "stray": True}
        done, failed = set(), set()
        with mock.patch.dict(os.environ, {"ATOM_PD_FAILED_WRITE_HOLD_S": "60"}):
            conn._collect_write_reports(done, failed)
        self.assertEqual(done, {"a"})
        # just_late is inside the hold, so a push it started may still land.
        self.assertEqual(failed, {"late"})
        self.assertEqual(set(conn._awaiting_write), {"just_late"})
        self.assertIn("b", conn._held_failures)
        self.assertEqual(conn._write_reports, {})
        # late's WRITE_REQ may never have arrived, so the producer is told.
        self.assertEqual(len(conn.notified), 1)

    def test_a_failed_push_is_reported_only_once_no_write_is_in_flight(self):
        conn = self.make(producer=True)
        in_flight = FakeStatus("progress")
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: [
            FakeStatus("failed"),
            in_flight,
        ]
        conn._write_requests.put(self.push_request())
        conn._service_write_requests()
        self.assertEqual(conn.done_sending, set())
        self.assertEqual(conn.sent, [])

        in_flight.state = "ok"
        conn._service_write_requests()
        self.assertEqual(conn.done_sending, {9})
        ((frames, _host, _port),) = conn.sent
        self.assertEqual(self.msgpack.loads(frames[1])["ok"], False)

    def test_a_failed_read_is_reported_only_once_no_read_is_in_flight(self):
        conn = self.make(producer=False)
        in_flight = FakeStatus("progress")
        conn.request_id_to_transfer_id = {"r": 77}
        conn._recving_transfers = {"r": [FakeStatus("failed"), in_flight]}
        conn._recving_transfers_callback_addr = {"r": ("10.0.0.1", "7302", "p_dp0")}
        self.assertEqual(conn._pop_done_transfers(), (set(), set()))
        self.assertEqual(conn.notified, [])

        in_flight.state = "ok"
        self.assertEqual(conn._pop_done_transfers(), (set(), {"r"}))
        self.assertEqual(conn.notified, [(77, "10.0.0.1", "7302")])
        self.assertEqual(conn._recving_transfers, {})

    def test_a_failed_receive_is_held_before_it_is_reported(self):
        import time

        conn = self.make(producer=False)
        with mock.patch.dict(os.environ, {"ATOM_PD_FAILED_WRITE_HOLD_S": "60"}):
            conn._hold_failure("r")
        failed = set()
        conn._report_held_failures(failed)
        self.assertEqual(failed, set())

        conn._held_failures["r"] = time.monotonic() - 1
        conn._report_held_failures(failed)
        self.assertEqual(failed, {"r"})
        self.assertEqual(conn._held_failures, {})

    def test_the_producer_starts_no_push_past_the_consumers_deadline(self):
        import time

        conn = self.make(producer=True)
        pushed = []
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: pushed.append(request) or []
        conn._write_requests.put(self.push_request(deadline=time.time() - 1))
        conn._service_write_requests()
        self.assertEqual(pushed, [])
        self.assertEqual(conn.done_sending, {9})
        ((frames, _host, _port),) = conn.sent
        self.assertEqual(self.msgpack.loads(frames[1])["ok"], False)

    def test_a_failed_handshake_fails_the_read_instead_of_issuing_it(self):
        import time

        conn = self.make(producer=False)
        issued = []
        conn._issue_read_for_req = lambda req_id, meta: issued.append(req_id)
        meta = self.recv_meta([4], [10])
        meta.remote_engine_id = "10.0.0.1:7301"
        conn._awaiting_handshake = {
            "bad": (time.monotonic() + 300, meta),
            "good": (time.monotonic() + 300, meta),
        }
        conn._ready_requests.put(("bad", meta, RuntimeError("producer gone")))
        conn._ready_requests.put(("good", meta, None))
        with self.assertLogs("atom", "ERROR"):
            conn._issue_ready_reads()
        self.assertEqual(issued, ["good"])
        self.assertEqual(conn._failed_before_read, {"bad"})
        self.assertEqual(conn._awaiting_handshake, {})
        self.assertEqual(len(conn.notified), 1)

    def test_a_read_whose_handshake_never_finishes_is_failed(self):
        import time

        conn = self.make(producer=False)
        issued = []
        conn._issue_read_for_req = lambda req_id, meta: issued.append(req_id)
        conn._awaiting_handshake = {
            "r": (time.monotonic() - 1, self.recv_meta([4], [10]))
        }
        with self.assertLogs("atom", "ERROR"):
            conn._issue_ready_reads()
        self.assertEqual(conn._failed_before_read, {"r"})
        self.assertEqual(len(conn.notified), 1)
        # A handshake finishing after that must not start a read for it.
        conn._ready_requests.put(("r", self.recv_meta([4], [10]), None))
        conn._issue_ready_reads()
        self.assertEqual(issued, [])

    def test_reads_for_one_producer_share_its_handshake(self):
        import types
        from concurrent.futures import Future

        conn = self.make(producer=False)
        conn.transfer_mode = "read"
        submitted, futures = [], []

        def submit(fn, *args):
            submitted.append(args)
            futures.append(Future())
            return futures[-1]

        conn._handshake_executor = types.SimpleNamespace(submit=submit)
        conn.start_load_kv(
            types.SimpleNamespace(
                reqs_to_recv={
                    "r1": self.recv_meta([4], [10]),
                    "r2": self.recv_meta([5], [11]),
                },
                request_id_to_transfer_id={"r1": 1, "r2": 2},
            )
        )
        self.assertEqual(submitted, [("10.0.0.1", 7301, 8, "10.0.0.1:7301_dp0", 0)])
        self.assertEqual(set(conn._awaiting_handshake), {"r1", "r2"})

        futures[0].set_result({"agent"})
        issued = []
        conn._issue_read_for_req = lambda req_id, meta: issued.append(req_id)
        conn._issue_ready_reads()
        self.assertEqual(sorted(issued), ["r1", "r2"])
        self.assertIn("10.0.0.1:7301_dp0", conn._remote_agents)

    def test_a_read_that_fails_part_way_waits_for_what_it_posted(self):
        conn = self.make(producer=False)
        posted = FakeStatus("progress")

        def post_one_then_fail(**_kwargs):
            conn._recving_transfers["r"] = [posted]
            conn._recving_transfers_callback_addr["r"] = ("10.0.0.1", "7302", "p_dp0")
            raise RuntimeError("could not post the rest")

        conn._read_blocks = post_one_then_fail
        meta = self.recv_meta([4, 5], [10, 11])
        meta.remote_engine_id = "10.0.0.1:7301"
        with self.assertLogs("atom", "ERROR"):
            conn._issue_read_for_req("r", meta)
        self.assertEqual(conn._failed_before_read, set())

        posted.state = "ok"
        conn.request_id_to_transfer_id = {"r": 77}
        self.assertEqual(conn._pop_done_transfers(), (set(), {"r"}))

    def test_descriptors_identical_to_ours_are_refused(self):
        conn = self.block_peer([b"L0", b"L1"], [b"L0", b"L1"], remote_blocks=8)
        with self.assertRaises(ValueError):
            conn._get_or_build_sessions("engine")
        self.assertEqual(conn._built_sessions, {})

    def test_a_second_write_req_for_one_transfer_is_not_pushed_twice(self):
        conn = self.make(producer=True)
        pushed = []
        in_flight = FakeStatus("progress")
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: (
            pushed.append(request["decode_req_id"]) or [in_flight]
        )
        conn._write_requests.put(self.push_request())
        conn._service_write_requests()
        conn._write_requests.put(self.push_request(decode_req_id="d2"))
        with self.assertLogs("atom", "WARNING"):
            conn._service_write_requests()
        self.assertEqual(pushed, ["d1"])

        in_flight.state = "ok"
        conn._service_write_requests()
        # The source blocks are released now, so a late repeat must not push.
        conn._write_requests.put(self.push_request(decode_req_id="d3"))
        with self.assertLogs("atom", "ERROR"):
            conn._service_write_requests()
        self.assertEqual(pushed, ["d1"])
        reports = [self.msgpack.loads(frames[1]) for frames, _host, _port in conn.sent]
        self.assertEqual(
            reports,
            [
                {"decode_req_id": "d1", "ok": True},
                {"decode_req_id": "d3", "ok": False},
            ],
        )

    def test_a_push_that_fails_part_way_waits_for_what_it_posted(self):
        conn = self.make(producer=True)
        conn._remote_agents = {"10.0.0.2:7400_dp0": {"agent"}}
        sessions = [({(0, 0): "L0"}, {}), ({(0, 0): "L1"}, {})]
        conn._get_or_build_sessions = lambda engine_id: (sessions, None)
        posted = FakeStatus("progress")

        def write(szs, local_offsets, remote_offsets, session):
            if session == "L1":
                raise RuntimeError("could not post")
            return posted

        conn.moriio_wrapper.write_remote_data_status = write
        conn._write_target_ready = lambda request: True
        conn._write_requests.put(self.push_request())
        with self.assertLogs("atom", "ERROR"):
            conn._service_write_requests()
        # Layer 0's write may still be reading the source blocks.
        self.assertEqual(conn.done_sending, set())
        self.assertEqual(conn.sent, [])

        posted.state = "ok"
        with self.assertLogs("atom", "ERROR"):
            conn._service_write_requests()
        self.assertEqual(conn.done_sending, {9})
        ((frames, _host, _port),) = conn.sent
        self.assertEqual(self.msgpack.loads(frames[1])["ok"], False)
        self.assertEqual(conn._remote_agents, {})

    def test_a_read_refused_before_any_transfer_releases_the_producer(self):
        conn = self.make(producer=False)
        meta = self.recv_meta([4], [10])
        meta.src_block_skip_factor = 2
        meta.remote_engine_id = "10.0.0.1:7301"
        conn.request_id_to_transfer_id = {"r": 77}
        with self.assertLogs("atom", "ERROR"):
            conn._issue_read_for_req("r", meta)
        self.assertEqual(conn._failed_before_read, {"r"})
        self.assertEqual(conn.notified, [(77, "10.0.0.1", 7302)])

    def block_peer(self, local, remote, remote_blocks):
        """A producer with one block layer, facing a consumer's descriptors."""
        import types

        conn = self.make(producer=True)
        conn.layer_name_to_local_kv_cache_metadata = {"layer_0": local}
        conn.layer_name_to_remote_kv_cache_metadata = {"engine": {"layer_0": remote}}
        conn.remote_moriio_metadata = {
            "engine": types.SimpleNamespace(num_blocks=remote_blocks)
        }
        conn._built_sessions = {}
        conn.moriio_wrapper.get_unpack_memory_metadata = lambda packed: packed
        conn.moriio_wrapper.build_session = lambda local, remote: (local, remote)
        return conn

    def test_sessions_span_pools_of_different_sizes(self):
        # Our pool needs two chunks of 4 blocks while the consumer's three
        # blocks fit in one: P and D size their pools independently.
        conn = self.block_peer([b"L0", b"L1"], [b"R0"], remote_blocks=3)

        ((k_sessions, v_sessions),), _meta = conn._get_or_build_sessions("engine")

        self.assertEqual(k_sessions, {(0, 0): (b"L0", b"R0"), (1, 0): (b"L1", b"R0")})
        self.assertEqual(v_sessions, {})
        # Source block 5 sits in our second chunk, destination 1 in their first.
        self.assertEqual(list(conn._group_offsets([5], [1], 8)), [(1, 0)])

    def test_v_sessions_split_each_end_at_its_own_k_count(self):
        conn = self.block_peer(
            [b"K0", b"K1", b"V0", b"V1"], [b"RK0", b"RV0"], remote_blocks=3
        )

        ((k_sessions, v_sessions),), _meta = conn._get_or_build_sessions("engine")

        self.assertEqual(k_sessions, {(0, 0): (b"K0", b"RK0"), (1, 0): (b"K1", b"RK0")})
        self.assertEqual(v_sessions, {(0, 0): (b"V0", b"RV0"), (1, 0): (b"V1", b"RV0")})

    def test_descriptors_that_do_not_cover_the_peers_pool_are_refused(self):
        # Two descriptors for three blocks at four per chunk: the peer chunked
        # its pool differently, so no offset would land where it points.
        conn = self.block_peer([b"L0", b"L1"], [b"R0", b"R1"], remote_blocks=3)
        with self.assertRaises(ValueError):
            conn._get_or_build_sessions("engine")
        self.assertEqual(conn._built_sessions, {})


class WrapperMessagingTest(unittest.TestCase):
    """The wrapper's control-plane sends and the status-returning write.

    tests/test_transfer_engine.py would cover the wrapper but is skipped
    module-wide, so the socket cache shared by send_notify and send_message is
    pinned here.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from atom.kv_transfer.disaggregation.moriio import moriio_engine
        except Exception as exc:
            raise unittest.SkipTest(f"moriio_engine does not import: {exc}") from exc
        cls.module = moriio_engine

    def setUp(self):
        import types

        self.sockets = []
        self.fail_next = False

        def make_socket(**kwargs):
            sock = types.SimpleNamespace(path=kwargs["path"], frames=[])

            def send_multipart(frames):
                if self.fail_next:
                    self.fail_next = False
                    raise RuntimeError("peer gone")
                sock.frames.append(frames)

            sock.send_multipart = send_multipart
            self.sockets.append(sock)
            return sock

        self.original = self.module.make_zmq_socket
        self.module.make_zmq_socket = make_socket
        self.wrapper = object.__new__(self.module.MoRIIOWrapper)
        self.wrapper._sockets = {}

    def tearDown(self):
        self.module.make_zmq_socket = self.original

    def test_notify_and_messages_share_one_socket_per_peer(self):
        constants = self.module.MoRIIOConstants
        self.wrapper.send_notify(7, "10.0.0.1", 7302)
        self.wrapper.send_message([constants.WRITE_REQ, b"x"], "10.0.0.1", 7302)
        self.wrapper.send_message([constants.WRITE_DONE, b"y"], "10.0.0.2", 7401)
        self.assertEqual(len(self.sockets), 2)
        self.assertEqual(
            self.sockets[0].frames,
            [[constants.POP_DONE_RECV, b"7"], [constants.WRITE_REQ, b"x"]],
        )

    def test_a_failed_send_drops_the_socket_so_the_next_reconnects(self):
        self.fail_next = True
        with self.assertRaises(RuntimeError):
            self.wrapper.send_message([b"t", b"x"], "10.0.0.1", 7302)
        self.assertEqual(self.wrapper._sockets, {})
        self.wrapper.send_message([b"t", b"y"], "10.0.0.1", 7302)
        self.assertEqual(len(self.sockets), 2)

    def test_the_tracked_write_hands_back_its_status(self):
        import types

        calls = []
        session = types.SimpleNamespace(
            batch_write=lambda *args: calls.append(args) or "status"
        )
        self.wrapper.local_memory_registered = True
        self.wrapper.moriio_engine = types.SimpleNamespace(
            allocate_transfer_uid=lambda: 42
        )
        self.wrapper.transfer_status = []
        status = self.wrapper.write_remote_data_status([8], [16], [24], session)
        self.assertEqual(status, "status")
        # MoRI's order is (local offsets, remote offsets, sizes, uid).
        self.assertEqual(calls, [([16], [24], [8], 42)])
        # Tracked by the caller, not parked on the wrapper's shared list.
        self.assertEqual(self.wrapper.transfer_status, [])


class MergeContiguousBlocksTest(unittest.TestCase):
    """The offset merger is pure numpy, so it can run wherever numpy exists."""

    @classmethod
    def setUpClass(cls):
        np = __import__("importlib").util.find_spec("numpy")
        if np is None:
            raise unittest.SkipTest("numpy is not installed in this lane")
        import numpy

        tree = ast.parse(CONNECTOR.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == (
                "merge_contiguous_blocks"
            ):
                node.decorator_list = []
                node.args.args = [a for a in node.args.args if a.arg != "self"]
                scope = {"np": numpy}
                code = compile(ast.Module([node], []), "<merge>", "exec")
                exec(code, scope)  # noqa: S102
                cls.merge = staticmethod(scope["merge_contiguous_blocks"])
                return
        raise AssertionError("merge_contiguous_blocks not found")

    def test_adjacent_blocks_collapse_into_one(self):
        local, remote, sizes = self.merge([0, 100, 200], [0, 100, 200], [100, 100, 100])
        self.assertEqual((local, remote, sizes), ([0], [0], [300]))

    def test_a_gap_keeps_the_segments_apart(self):
        local, _remote, sizes = self.merge(
            [0, 100, 300], [0, 100, 300], [100, 100, 100]
        )
        self.assertEqual(local, [0, 300])
        self.assertEqual(sizes, [200, 100])

    def test_remote_discontinuity_prevents_a_merge(self):
        _local, _remote, sizes = self.merge([0, 100], [0, 500], [100, 100])
        self.assertEqual(sizes, [100, 100])

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.merge([0, 100], [0], [100, 100])

    def test_empty_input(self):
        self.assertEqual(self.merge([], [], []), ([], [], []))


def load_class_members(names: list[str], namespace: dict) -> dict:
    """Lift methods of MoRIIOConnector (and top-level classes) from the source.

    Static methods keep their decorator; everything else becomes a plain
    function, ready to be put on a stand-in class.
    """
    tree = ast.parse(CONNECTOR.read_text())
    scope: dict = dict(namespace)
    wanted = set(names)
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            nodes.append(node)
            wanted.discard(node.name)
    if wanted:
        raise AssertionError(f"not found: {sorted(wanted)}")
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = [
                d for d in node.decorator_list if ast.unparse(d) == "staticmethod"
            ]
        code = compile(ast.Module([node], []), str(CONNECTOR), "exec")
        exec(code, scope)  # noqa: S102
    return {name: scope[name] for name in names}


class RegionPseudoLayerTest(unittest.TestCase):
    """PAGE by block id, SLOT by state_slot. Lifted so the CPU lane can run it."""

    PAGE, SLOT = 0, 1

    @classmethod
    def setUpClass(cls):
        import logging

        members = load_class_members(
            [
                "_region_unit",
                "_group_offsets",
                "_check_region_layout",
                "_region_write_plan",
                "_write_blocks",
                "_source_covers",
                "_LazySessions",
            ],
            {
                "Any": object,
                "_REGION_PAGE": cls.PAGE,
                "_REGION_SLOT": cls.SLOT,
                "logger": logging.getLogger("atom"),
                "_POST_FAILED": "post-failed",
            },
        )
        cls.lazy = members.pop("_LazySessions")
        cls.Fake = type("FakeConnector", (), members)

    def make(self, local: dict, remote: dict):
        import types

        conn = self.Fake()
        conn._region_layers = local
        conn.blocks_per_chunk = None
        conn.remote_moriio_metadata = {
            "eng": types.SimpleNamespace(region_layout=remote)
        }
        conn.writes = []
        built = []

        def build_session(local_md, remote_md):
            built.append((local_md, remote_md))
            return (local_md, remote_md)

        def write(szs, l_offs, r_offs, session):
            conn.writes.append((session, l_offs, r_offs, szs))
            return "status"

        conn.moriio_wrapper = types.SimpleNamespace(
            build_session=build_session, write_remote_data_status=write
        )
        conn.built = built
        sessions = [
            (self.lazy(conn.moriio_wrapper, [(n, i) for i in range(8)], range(8)), {})
            for n in local
        ]
        conn._get_or_build_sessions = lambda engine_id: (sessions, None)
        return conn

    # [kind, unit_bytes, num_units, reverse, units_per_chunk]
    def layouts(self, slot_units_remote: int = 6):
        local = {
            "page.0.kv": [self.PAGE, 8, 10, 0, 4],
            "slot.0.kv": [self.SLOT, 16, 4, 1, 2],
        }
        remote = {
            "page.0.kv": [self.PAGE, 8, 12, 0, 4],
            "slot.0.kv": [self.SLOT, 16, slot_units_remote, 1, 2],
        }
        return local, remote

    def request(self, **overrides):
        request = {
            "transfer_id": 3,
            "dst_block_ids": [2, 9],
            "src_block_ids": [1, 5, 6],
            "src_slot": 0,
            "dst_slot": 1,
        }
        request.update(overrides)
        return request

    def test_reverse_indexed_unit_matches_the_region_address(self):
        from_end = self.Fake._region_unit([self.SLOT, 16, 4, 1, 2], 0, "x")
        # KVTransferRegion.unit_addr(0) = base + total - unit = base + 3 * unit.
        self.assertEqual(from_end, 3)
        self.assertEqual(self.Fake._region_unit([self.PAGE, 8, 10, 0, 4], 7, "x"), 7)
        with self.assertRaises(ValueError):
            self.Fake._region_unit([self.SLOT, 16, 4, 1, 2], 4, "x")
        with self.assertRaises(ValueError):
            self.Fake._region_unit([self.PAGE, 8, 10, 0, 4], -1, "x")

    def test_plan_pairs_blocks_and_slots_per_region(self):
        conn = self.make(*self.layouts())
        req = self.request()
        page, slot = conn._region_write_plan(req, "eng", [1, 5], [2, 9])
        # Blocks 1->2 share chunk 0 on both ends; 5->9 is chunk 1 -> chunk 2.
        self.assertEqual(page, {(0, 0): ([8], [16], [8]), (1, 2): ([8], [8], [8])})
        # Local slot 0 is forward unit 3 (chunk 1, +16); remote slot 1 of a
        # 6-unit region is forward unit 4 (chunk 2, +0).
        self.assertEqual(slot, {(1, 2): ([16], [0], [16])})

    def test_each_end_reverses_against_its_own_slot_count(self):
        conn = self.make(*self.layouts(slot_units_remote=4))
        _page, slot = conn._region_write_plan(self.request(), "eng", [1], [2])
        # Remote slot 1 of 4 units is forward unit 2: chunk 1, +0.
        self.assertEqual(slot, {(1, 1): ([16], [0], [16])})

    def test_a_missing_state_slot_is_refused(self):
        conn = self.make(*self.layouts())
        with self.assertRaises(ValueError) as caught:
            conn._region_write_plan(self.request(dst_slot=-1), "eng", [1], [2])
        self.assertIn("no state slot", str(caught.exception))

    def test_nothing_is_posted_when_any_region_is_out_of_range(self):
        conn = self.make(*self.layouts())
        # Slot 4 is past the producer's 4-unit region; the page region before
        # it must not have been posted either.
        with self.assertRaises(ValueError):
            conn._write_blocks(self.request(src_slot=4), "eng")
        self.assertEqual(conn.writes, [])
        with self.assertRaises(ValueError):
            conn._write_blocks(self.request(dst_block_ids=[12]), "eng")
        self.assertEqual(conn.writes, [])

    def test_write_posts_one_batch_per_region_chunk_pair(self):
        conn = self.make(*self.layouts())
        statuses = conn._write_blocks(self.request(), "eng")
        self.assertEqual(statuses, ["status"] * 3)
        sessions = [w[0] for w in conn.writes]
        # Lazily built sessions, keyed by (region, local chunk) x remote chunk.
        self.assertEqual(
            sessions,
            [(("page.0.kv", 0), 0), (("page.0.kv", 1), 2), (("slot.0.kv", 1), 2)],
        )
        self.assertEqual(conn.writes[2][1:], ([16], [0], [16]))

    def test_lazy_sessions_are_built_once(self):
        conn = self.make(*self.layouts())
        conn._write_blocks(self.request(), "eng")
        conn._write_blocks(self.request(transfer_id=4), "eng")
        self.assertEqual(len(conn.built), 3)

    def test_layout_check_allows_only_unit_count_differences(self):
        local, remote = self.layouts()
        conn = self.make(local, remote)
        self.assertIs(conn._check_region_layout("eng"), remote)

        bad = {k: list(v) for k, v in remote.items()}
        bad["slot.0.kv"][1] = 32
        conn = self.make(local, bad)
        with self.assertRaises(ValueError):
            conn._check_region_layout("eng")

        conn = self.make(local, {"page.0.kv": remote["page.0.kv"]})
        with self.assertRaises(ValueError):
            conn._check_region_layout("eng")

        conn = self.make(local, None)
        with self.assertRaises(ValueError) as caught:
            conn._check_region_layout("eng")
        self.assertIn("no region layout", str(caught.exception))

    def test_per_layer_path_keeps_its_global_chunking(self):
        conn = self.make({}, {})
        conn.blocks_per_chunk = 4
        groups = conn._group_offsets([1, 5], [2, 9], 8)
        self.assertEqual(groups, {(0, 0): ([8], [16], [8]), (1, 2): ([8], [8], [8])})


if __name__ == "__main__":
    unittest.main()
