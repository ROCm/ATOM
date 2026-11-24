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
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = ROOT / "atom/kv_transfer/disaggregation/moriio/moriio_connector.py"
COMMON = ROOT / "atom/kv_transfer/disaggregation/moriio/moriio_common.py"


def load_function(
    path: Path, name: str, namespace: dict | None = None, *, cls: str | None = None
):
    """Execute one top-level function, or one method of class `cls`, from a
    module we cannot import, as a plain function."""
    body = ast.parse(path.read_text()).body
    if cls is not None:
        classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls]
        body = classes[0].body if classes else []
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            scope: dict = dict(namespace or {})
            # Our own source, lifted so the CPU lane can run it.
            code = compile(ast.Module([node], []), str(path), "exec")
            exec(code, scope)  # noqa: S102
            return scope[name]
    raise AssertionError(f"{cls + '.' if cls else ''}{name} not found in {path}")


class RemotePrefillLookupTest(unittest.TestCase):
    """The consumer's lookup keeps answering until its receive is queued."""

    @classmethod
    def setUpClass(cls):
        import logging

        namespace = {"logger": logging.getLogger("atom")}
        scheduler = "MoRIIOConnectorScheduler"
        cls.lookup = staticmethod(
            load_function(CONNECTOR, "get_num_new_matched_tokens", cls=scheduler)
        )
        cls.after_alloc = staticmethod(
            load_function(
                CONNECTOR, "update_state_after_alloc", namespace, cls=scheduler
            )
        )

    def test_a_held_back_request_is_still_a_remote_load(self):
        import types

        # The consumer's mode and paging too, which a kept-prefix receive reads.
        sched = types.SimpleNamespace(
            is_producer=False,
            transfer_id_to_request_id={},
            request_id_to_transfer_id={},
            _reqs_need_recv={},
            transfer_mode="write",
            block_size=4,
            dcp_size=1,
            hash_block_size=4,
            _same_page_geometry=lambda params: True,
        )
        seq = types.SimpleNamespace(
            id=7,
            prompt_token_ids=[1, 2, 3],
            block_table=[4],
            state_slot=-1,
            num_cached_tokens=0,
            kv_transfer_params={"do_remote_prefill": True, "transfer_id": 7},
        )
        # Held back after the first yes (a full running set), asked again.
        self.assertEqual(self.lookup(sched, seq), (3, True))
        self.assertEqual(self.lookup(sched, seq), (3, True))
        self.after_alloc(sched, seq)
        self.assertEqual(self.lookup(sched, seq), (0, False))
        self.assertIn(7, sched._reqs_need_recv)


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

    def test_transfer_subsets_are_skipped_or_refused(self):
        body = self.segment("_rejects_transfer_subset")
        self.assertIn("src_block_skip_factor", body)
        for path in ("_issue_read_for_req", "_request_write"):
            caller = self.segment(path)
            self.assertIn("self._rejects_transfer_subset(", caller, path)
            self.assertIn("_fail_before_transfer", caller, path)
            # The consumer's local prefix is skipped on both ends of the pairing.
            self.assertIn("self._kept_prefix_blocks(meta)", caller, path)

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

    def test_per_request_state_never_joins_the_block_layout(self):
        body = self.segment("register_kv_caches")
        # Chunked as blocks, a slot-addressed tensor would be pushed to block
        # offsets, and `_kv_layout` refuses the mixed model outright.
        self.assertLess(
            body.index('getattr(kv_cache, "per_request_state", False)'),
            body.index("chunk_tensor_for_rdma("),
        )
        self.assertIn("self._register_state(layer_name, kv_cache)", body)

    def test_state_is_checked_before_any_block_is_written(self):
        body = self.segment("_write_blocks")
        self.assertLess(
            body.index("self._state_writes(request, engine_id)"),
            body.index("write_remote_data_status("),
        )


def import_connector(name: str = "MoRIIOConnector"):
    """A class from the connector module, or None where it cannot be imported.

    aiter resolves the gfx arch from rocminfo while importing, so this only
    succeeds in a container that can see the GPUs.
    """
    try:
        from atom.kv_transfer.disaggregation.moriio import moriio_connector
    # Broad on purpose: whatever stops the native stack loading means skip,
    # and an unexpected type must not turn the CPU lane red.
    except Exception:  # noqa: BLE001
        return None
    return getattr(moriio_connector, name)


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
        import threading
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
        conn._state_tensors = {}
        conn._verify_state_before = {}
        conn._verify_kept = {}
        conn._write_state_digests = {}
        conn._write_reports_lock = threading.Lock()
        return conn, names

    def add_state(self, conn, slots: int = 4):
        """One KDA layer of state on `conn`: a bf16 conv and an fp32 SSM tensor."""
        conv = self.torch.zeros((slots, 3, 4), dtype=self.torch.bfloat16)
        ssm = self.torch.zeros((slots, 2, 2, 2), dtype=self.torch.float32)
        conn._state_tensors = {
            "kda.k": (b"k", conv.stride(0) * conv.element_size(), conv),
            "kda.v": (b"v", ssm.stride(0) * ssm.element_size(), ssm),
        }
        return conv, ssm

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

    def test_state_digest_reads_one_slot_of_each_tensor(self):
        conn, _ = self.make_connector()
        conv, ssm = self.add_state(conn)
        before = conn._state_digest(1)
        conv[1] = 1.5
        after = conn._state_digest(1)
        self.assertNotEqual(before["kda.k"], after["kda.k"])
        self.assertEqual(before["kda.v"], after["kda.v"])
        # Another request's slot is no part of this one's digest.
        ssm[2] = 3.0
        self.assertEqual(conn._state_digest(1), after)

    def test_a_slot_that_matches_the_producer_is_reported_as_info(self):
        conn, _ = self.make_connector()
        conv, ssm = self.add_state(conn)
        conn._verify_state_before["req"] = (1, conn._state_digest(1))
        conv[1] = 0.25
        ssm[1] = -2.0
        # What the producer digested from its own slot before pushing it.
        conn._write_state_digests["req"] = conn._state_digest(1)

        with self.assertLogs("atom", "INFO") as logs:
            conn._verify_state_landed("req")
        self.assertIn("all 2 state tensors match the producer", logs.output[0])
        # Both maps are popped, or a long run accumulates a digest per request.
        self.assertEqual(conn._verify_state_before, {})
        self.assertEqual(conn._write_state_digests, {})

    def test_a_slot_unlike_the_producers_is_reported(self):
        conn, _ = self.make_connector()
        _conv, ssm = self.add_state(conn)
        conn._verify_state_before["req"] = (1, conn._state_digest(1))
        ssm[1] = 2.0
        conn._write_state_digests["req"] = conn._state_digest(1)
        # Something other than the producer's slot landed in the SSM state.
        ssm[1] = 1.0

        with self.assertLogs("atom", "ERROR") as logs:
            conn._verify_state_landed("req")
        self.assertIn("1 of 2 state tensors differ", logs.output[0])
        self.assertIn("kda.v", logs.output[0])

    def test_a_missing_producer_digest_is_reported(self):
        conn, _ = self.make_connector()
        self.add_state(conn)
        conn._verify_state_before["req"] = (1, conn._state_digest(1))
        with self.assertLogs("atom", "ERROR") as logs:
            conn._verify_state_landed("req")
        self.assertIn("the producer sent no state digest", logs.output[0])


class FakeEvent:
    """Stands in for torch.cuda.Event; ``done`` is flipped by the test."""

    def __init__(self, done: bool = False):
        self.done = done
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True

    def query(self):
        return self.done


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
        import sys
        import types

        import msgpack
        import torch

        from atom.kv_transfer.disaggregation.moriio.moriio_common import (
            MoRIIOConstants,
        )

        cls.msgpack = msgpack
        cls.torch = torch
        cls.types = types
        cls.constants = MoRIIOConstants
        cls.connector_module = sys.modules[cls.connector.__module__]

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
        conn._write_wakeup = threading.Event()
        conn._write_service_stop = threading.Event()
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
        conn._regions = None
        conn._region_sessions = {}
        conn._retired_sessions = []
        conn.remote_moriio_metadata = {}
        conn.layer_name_to_remote_kv_cache_metadata = {}
        conn.done_sending = set()
        conn._done_sending_lock = threading.Lock()
        conn._write_ids = {}
        conn._finished_writes = collections.deque()
        # The prefill of the transfer push_request asks for has finished.
        prefilled = FakeEvent()
        prefilled.done = True
        conn._kv_ready_events = {9: prefilled}
        conn._kv_ready_batches = collections.deque()
        conn._kv_ready_expired = {}
        conn._kv_ready_lock = threading.Lock()
        conn._diag_post_ready = False
        conn._post_ready_forwards = {}
        conn._gated_writes = collections.deque()
        conn._state_tensors = {}
        conn._num_state_slots = 0
        conn._built_state_sessions = {}
        conn._verify_state_before = {}
        conn._verify_kept = {}
        conn._write_state_digests = {}
        conn._first_transfer_slot = 0
        conn._staged_sources = {}
        conn._config = None

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
            register_local_buffer=lambda ptr, size, device: f"{ptr}+{size}".encode(),
            get_agent_metadata=lambda: b"agent",
        )
        return conn

    def recv_meta(self, local, remote, **extra):
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
            **extra,
        )

    def make_stateful(self, conn, *, slots: int = 6, conv: int = 24, ssm: int = 128):
        """Give `conn` one KDA layer of state, with its sessions stubbed.

        Two uint8 tensors of `slots` rows: a conv slot of `conv` bytes and an
        SSM slot of `ssm`. The consumer is taken to have as many slots.
        """
        import types

        conn._num_state_slots = slots
        conn._state_tensors = {
            "kda.k": (
                b"k",
                conv,
                self.torch.zeros((slots, conv), dtype=self.torch.uint8),
            ),
            "kda.v": (
                b"v",
                ssm,
                self.torch.zeros((slots, ssm), dtype=self.torch.uint8),
            ),
        }
        conn.remote_moriio_metadata = {
            "engine": types.SimpleNamespace(num_state_slots=slots)
        }
        conn._get_or_build_state_sessions = lambda engine_id: [
            ("S-k", conv),
            ("S-v", ssm),
        ]
        return conn

    @staticmethod
    def block_sessions():
        """Two layers' K sessions over two chunks, as the push tests use."""
        return [({(0, 0): f"L{i}-00", (0, 1): f"L{i}-01"}, {}) for i in range(2)]

    def hybrid_caches(self, slots: int = 6):
        """A KDA layer ahead of two MLA layers, as Kimi-K3 interleaves them."""
        import types

        state = types.SimpleNamespace(
            k_cache=self.torch.zeros((slots, 3, 8), dtype=self.torch.bfloat16),
            v_cache=self.torch.zeros((slots, 2, 4, 4), dtype=self.torch.float32),
            replay_buf_k=None,
            replay_buf_u=None,
            replay_buf_g=None,
            replay_write_pos=None,
            per_request_state=True,
        )
        caches = {"layer_0": state}
        for name in ("layer_1", "layer_2"):
            caches[name] = types.SimpleNamespace(
                k_cache=self.torch.zeros((16, 1, 4), dtype=self.torch.uint8),
                v_cache=None,
            )
        return caches

    def replay_caches(self, slots: int = 6):
        """`hybrid_caches` under ReplaySSM, plus a second KDA layer.

        Each KDA layer carries its (k, u, g) records; both share one cursor.
        """
        import types

        caches = self.hybrid_caches(slots)
        cursor = self.torch.full((slots,), -1, dtype=self.torch.int32)

        def records(state):
            state.replay_buf_k = self.torch.zeros(
                (slots, 2, 4, 8), dtype=self.torch.bfloat16
            )
            state.replay_buf_u = self.torch.zeros(
                (slots, 2, 4, 8), dtype=self.torch.bfloat16
            )
            state.replay_buf_g = self.torch.zeros(
                (slots, 2, 4, 8), dtype=self.torch.float32
            )
            state.replay_write_pos = cursor
            return state

        records(caches["layer_0"])
        caches["layer_3"] = records(
            types.SimpleNamespace(
                k_cache=self.torch.zeros((slots, 3, 8), dtype=self.torch.bfloat16),
                v_cache=self.torch.zeros((slots, 2, 4, 4), dtype=self.torch.float32),
                per_request_state=True,
            )
        )
        return caches

    def register(self, conn, kv_caches, transfer_tensors=None):
        """Run `register_kv_caches` with the listener stubbed; return its metadata."""
        conn.kv_cache_shape = None
        conn.blocks_per_chunk = None
        conn.layer_name_to_local_kv_cache_metadata = {}
        conn.engine_id = "10.0.0.2:7400"
        conn.side_channel_port = 7401
        advertised = []
        conn._handshake_listener = lambda metadata, *_rest: advertised.append(metadata)
        conn._start_write_service = lambda: None
        conn.register_kv_caches(kv_caches, transfer_tensors)
        conn._handshake_listener_thread.join(timeout=5)
        return advertised[0]

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

    def test_a_region_session_refuses_our_own_descriptor(self):
        import types

        conn = self.make(producer=True)
        conn._regions = [types.SimpleNamespace(name="page.0.kv")]
        conn.layer_name_to_local_kv_cache_metadata = {"page.0.kv": [b"ours"]}
        conn.layer_name_to_remote_kv_cache_metadata = {
            "engine": {"page.0.kv": [b"ours"]}
        }
        with self.assertRaises(ValueError):
            conn._region_session("engine", 0, 0, 0)

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

    def test_a_push_skips_the_prefix_the_consumer_holds(self):
        conn = self.make(producer=False)
        meta = self.recv_meta([4, 5, 6], [10, 11, 12])
        meta.num_computed_blocks = 1
        conn._request_write("d1", meta)
        ((frames, _host, _port),) = conn.sent
        request = self.msgpack.loads(frames[1])
        self.assertEqual(request["dst_block_ids"], [5, 6])
        self.assertEqual(request["src_block_ids"], [11, 12])

    def test_a_read_skips_the_prefix_the_consumer_holds(self):
        conn = self.make(producer=False)
        conn.transfer_mode = "read"
        reads = []
        conn._read_blocks = lambda **kwargs: reads.append(kwargs)
        meta = self.recv_meta([4, 5, 6], [10, 11, 12])
        meta.num_computed_blocks = 2
        meta.remote_engine_id = "engine"
        conn._issue_read_for_req("d1", meta)
        ((call,),) = [reads]
        self.assertEqual(call["local_block_ids"], [6])
        self.assertEqual(call["remote_block_ids"], [12])
        self.assertEqual(call["kept_block_ids"], [4, 5])

    def test_verify_reports_a_receive_that_wrote_into_the_kept_prefix(self):
        conn = self.make(producer=False)
        conn._note_kept_prefix("d1", [1], is_mla=True)
        with self.assertLogs("atom", "INFO") as logs:
            conn._verify_kept_landed("d1")
        self.assertIn("reused prefix rows unchanged", logs.output[-1])

        conn._note_kept_prefix("d2", [1], is_mla=True)
        conn.kv_caches["layer_0"].k_cache[2] = 7
        with self.assertLogs("atom", "ERROR") as logs:
            conn._verify_kept_landed("d2")
        self.assertIn("the receive wrote into blocks it kept", logs.output[0])

    def test_a_producer_over_two_hosts_is_reached_on_its_ranks_host(self):
        import types

        # A TP8 producer over two hosts (4 ranks each) reports only its first
        # host; rank 5 asks, and later notifies, rank 5 on the second.
        conn = self.make(producer=False)
        conn.tp_rank = 5
        conn._remote_tp_hosts = {"10.0.0.1": ["10.0.0.1", "10.0.0.9"]}
        conn.request_id_to_transfer_id = {"d1": 77}
        meta = self.recv_meta([4, 5], [10, 11, 12])
        conn._request_write("d1", meta)
        ((_frames, host, port),) = conn.sent
        self.assertEqual((host, port), ("10.0.0.9", 7306))
        conn._notify_producer("d1", meta)
        self.assertEqual(conn.notified, [(77, "10.0.0.9", 7306)])
        # Rank 3 is on the first host; a producer the map does not name keeps
        # its remote_host; hosts that do not split the TP group are refused.
        conn.tp_rank = 3
        self.assertEqual(conn._producer_host(meta), "10.0.0.1")
        other = types.SimpleNamespace(remote_host="10.0.0.3", tp_size=8)
        self.assertEqual(conn._producer_host(other), "10.0.0.3")
        conn._remote_tp_hosts = {"10.0.0.1": ["a", "b", "c"]}
        with self.assertRaises(ValueError):
            conn._producer_host(meta)

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

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("timed out")
            time.sleep(0.001)

    def test_the_write_service_pushes_and_reports_without_an_engine_step(self):
        conn = self.make(producer=True)
        statuses = [FakeStatus()]
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: statuses
        conn._start_write_service()
        try:
            conn._write_requests.put(self.push_request())
            conn._write_wakeup.set()
            self.wait_for(lambda: 9 in conn._sending_writes)
            self.assertEqual(conn.sent, [])
            statuses[0].state = "ok"
            self.wait_for(lambda: conn.sent)
            self.assertEqual(conn.done_sending, {9})
            self.assertEqual(conn.sent[0][0][0], self.constants.WRITE_DONE)
        finally:
            conn._write_service_stop.set()
            conn._write_wakeup.set()
            conn._write_service_thread.join(timeout=5)
        self.assertFalse(conn._write_service_thread.is_alive())

    def test_a_push_waits_for_its_prefill_to_finish_on_the_gpu(self):
        conn = self.make(producer=True)
        statuses = [FakeStatus()]
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: statuses
        event = FakeEvent()
        conn._kv_ready_events[9] = event
        conn._write_requests.put(self.push_request())

        conn._service_write_requests()
        self.assertEqual(conn._sending_writes, {})
        self.assertEqual(len(conn._pending_writes), 1)

        event.done = True
        conn._service_write_requests()
        self.assertIn(9, conn._sending_writes)
        # Another consumer rank may still ask for this transfer.
        self.assertIs(conn._kv_ready_events[9], event)

    def test_a_push_waits_until_its_prefill_records_a_forward(self):
        # The head rank returned the token, this rank's forward is not done.
        conn = self.make(producer=True)
        conn._kv_ready_events = {}
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: [FakeStatus()]
        conn._write_requests.put(self.push_request())
        conn._service_write_requests()
        self.assertEqual(conn._sending_writes, {})
        self.assertEqual(len(conn._pending_writes), 1)

        with (
            mock.patch("torch.cuda.Event", FakeEvent),
            mock.patch("torch.cuda.current_stream", lambda: None),
        ):
            conn.record_kv_cache_ready([9])
        conn._kv_ready_events[9].done = True
        conn._service_write_requests()
        self.assertIn(9, conn._sending_writes)

    def test_a_push_whose_prefill_never_reports_fails_after_the_write_timeout(self):
        conn = self.make(producer=True)
        conn._kv_ready_events = {}
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: self.fail("pushed")
        finished = []
        conn._finish_write = lambda request, ok: finished.append(ok)
        conn._pending_writes = [(self.push_request(), time.monotonic() - 601.0)]
        with self.assertLogs("atom", "ERROR") as logs:
            conn._service_write_requests()
        self.assertEqual(finished, [False])
        self.assertIn("no finished forward", logs.output[0])

    def test_old_ready_events_expire_but_their_prefills_stay_finished(self):
        conn = self.make(producer=True)
        stale, kept = FakeEvent(), FakeEvent()
        conn._kv_ready_events = {1: stale, 2: kept}
        conn._kv_ready_batches.append((time.monotonic() - 1e6, stale, [1, 2]))
        with (
            mock.patch("torch.cuda.Event", FakeEvent),
            mock.patch("torch.cuda.current_stream", lambda: None),
        ):
            conn.record_kv_cache_ready([2, 3])
        self.assertNotIn(1, conn._kv_ready_events)
        self.assertTrue(conn._source_ready(1))
        # Re-recorded since, so the stale batch must not drop it.
        self.assertIs(conn._kv_ready_events[2], conn._kv_ready_events[3])
        self.assertNotIn(2, conn._kv_ready_expired)
        self.assertFalse(conn._source_ready(2))

    def post_ready_batch(self):
        import types

        # Transfer 9 decodes 4 tokens ending at position 6 with k = 3, so the
        # draft writes through position 8: block indices 1..4 of a 3-block
        # table. Transfer 3 has recorded no KV-ready event.
        return types.SimpleNamespace(
            req_ids=[3, 9],
            num_scheduled_tokens=[5, 4],
            context_lens=[5, 7],
            num_rejected=[0, 1],
            block_tables=[[1, 2, 3], [4, 5, 6]],
            total_seqs_num_prefill=1,
            is_dummy_run=False,
        )

    def test_the_diagnostic_logs_a_forward_after_the_kv_ready_event(self):
        conn = self.make(producer=True)
        conn._diag_post_ready = True
        started = time.monotonic()
        conn._write_ids = {9: None}
        conn._sending_writes = {
            9: (
                [],
                {
                    "transfer_id": 9,
                    "src_block_ids": [4, 5],
                    "_page_stats": (started, 16, 2, 2),
                },
            )
        }
        with self.assertLogs("atom", "WARNING") as logs:
            conn.note_forward(self.post_ready_batch(), 2, 3)
        self.assertEqual(len(logs.output), 1)
        line = logs.output[0]
        self.assertIn("transfer 9: forward after its KV-ready event", line)
        self.assertIn("positions 3..6, draft through 8, num_rejected 1", line)
        self.assertIn("writes blocks [5, 6]", line)
        self.assertIn("push in flight for", line)
        self.assertIn("its 2 source blocks include [5]", line)
        self.assertEqual(list(conn._post_ready_forwards), [9])

    def test_the_diagnostic_is_silent_when_off(self):
        conn = self.make(producer=True)
        with self.assertNoLogs("atom", "WARNING"):
            conn.note_forward(self.post_ready_batch(), 2, 3)
        self.assertEqual(conn._post_ready_forwards, {})

    def test_blocks_are_described_by_the_transfer_holding_them(self):
        import types

        producer = self.make(producer=True)
        producer._sending_writes = {
            9: ([], {"transfer_id": 9, "src_block_ids": [4, 5]})
        }
        producer._pending_writes = [({"transfer_id": 3, "src_block_ids": [7]}, 0.0)]
        self.assertEqual(
            producer.describe_blocks([5, 7, 8]),
            "source of transfer 9 (pushing): [5]; source of transfer 3 (queued): [7]",
        )
        self.assertEqual(
            producer.describe_blocks([1]), "no transfer in flight holds them"
        )
        consumer = self.make(producer=False)
        consumer._awaiting_write = {
            "d1": (0.0, types.SimpleNamespace(local_block_ids=[2, 6]))
        }
        self.assertEqual(consumer.describe_blocks([6]), "target of req d1's push: [6]")

    def test_a_fabric_batch_never_writes_a_block_twice(self):
        conn = self.make(producer=True)
        conn._page_regions = [(0, 16, "k"), (1000, 4, "k_scale")]
        remote = [(64, 16), (5000, 4)]
        l_offs, r_offs, sizes = conn._page_batch(
            [2, 3, 7], [3, 4, 0], remote, into_remote=True
        )
        self.assertEqual(l_offs, [32, 112, 1008, 1028])
        self.assertEqual(r_offs, [112, 64, 5012, 5000])
        self.assertEqual(sizes, [32, 16, 8, 4])
        with self.assertRaisesRegex(ValueError, r"remote blocks \[4\] appear more"):
            conn._page_batch([2, 3, 7], [4, 5, 4], remote, into_remote=True)
        # A read writes the local blocks; a source read twice is allowed.
        conn._page_batch([2, 3, 7], [4, 5, 4], remote, into_remote=False)
        with self.assertRaisesRegex(ValueError, r"local blocks \[2\] appear more"):
            conn._page_batch([2, 3, 2], [4, 5, 6], remote, into_remote=False)

    def fabric(self, conn, *, blocks: int = 8):
        """`conn` over the fabric: two PAGE regions, the peer's at other offsets."""
        conn.use_fabric = True
        conn._page_regions = [(0, 16, "k"), (1000, 4, "k_scale")]
        conn._page_views = [
            self.torch.zeros((blocks, 1, 16), dtype=self.torch.uint8),
            self.torch.zeros((blocks, 1, 4), dtype=self.torch.uint8),
        ]
        conn._verify_page_dst = {}
        conn._verify_page_kept = {}
        conn._write_page_digests = {}
        conn._get_or_build_page_session = lambda engine_id: (
            "S-page",
            [(64, 16), (5000, 4)],
        )
        return conn

    def test_a_fabric_push_moves_the_suffix_and_a_staged_transfer_slot(self):
        # The consumer names only the blocks past its kept prefix (#3572).
        consumer = self.fabric(self.make(producer=False))
        meta = self.recv_meta([4, 5, 6], [10, 11, 12])
        meta.num_computed_blocks = 2
        consumer._request_write("d1", meta)
        ((frames, _host, _port),) = consumer.sent
        asked = self.msgpack.loads(frames[1])
        self.assertEqual((asked["dst_block_ids"], asked["src_block_ids"]), ([6], [12]))

        # The producer pushes those pages, and the state from transfer slot 5
        # only once the copy into it landed (#3571).
        conn = self.fabric(self.make_stateful(self.make(producer=True), slots=6))
        conn._first_transfer_slot = 4
        request = self.pending_push(conn, 5)
        request["dst_block_ids"], request["src_block_ids"] = [6], [12]
        conn._service_write_requests()
        self.assertEqual((conn.writes, len(conn._pending_writes)), ([], 1))
        conn._staged_sources[9] = (5, FakeEvent(done=True))
        conn._service_write_requests()
        self.assertEqual(
            conn.writes,
            [
                (
                    "S-page",
                    [12 * 16, 1000 + 12 * 4],
                    [64 + 6 * 16, 5000 + 6 * 4],
                    [16, 4],
                ),
                ("S-k", [5 * 24], [5 * 24], [24]),
                ("S-v", [5 * 128], [5 * 128], [128]),
            ],
        )

    def test_fabric_verify_reports_a_push_into_the_kept_prefix(self):
        conn = self.fabric(self.make(producer=False))
        conn._verify_kv = True
        meta = self.recv_meta([4, 5, 6], [10, 11, 12])
        meta.num_computed_blocks = 2
        conn._request_write("d1", meta)
        self.assertEqual(conn._verify_page_dst["d1"], [6])
        self.assertEqual(conn._verify_page_kept["d1"][0], [4, 5])
        # The push lands in block 6 and leaves the kept blocks alone.
        conn._page_views[0][6] = 9
        with self.assertLogs("atom", "INFO") as logs:
            conn._verify_page_kept_landed("d1")
        self.assertIn("2 reused prefix blocks unchanged", logs.output[-1])

        conn._request_write("d2", meta)
        conn._page_views[1][5] = 7
        with self.assertLogs("atom", "ERROR") as logs:
            conn._verify_page_kept_landed("d2")
        self.assertIn("the receive wrote into blocks it kept", logs.output[0])
        self.assertIn("roles ['k_scale']", logs.output[0])
        self.assertEqual(conn._verify_page_kept, {})

    def test_a_changed_source_reports_the_forwards_after_the_kv_ready_event(self):
        conn = self.make(producer=True)
        started = time.monotonic()
        request = {"transfer_id": 9, "_page_stats": (started, 16, 2, 2)}
        self.assertEqual(conn._post_ready_summary(request), "")
        conn._diag_post_ready = True
        self.assertIn("no forward ran it", conn._post_ready_summary(request))
        conn._post_ready_forwards[9] = [started + 0.002]
        self.assertIn(
            "1 forward(s) ran it after its KV-ready event on this rank, at +2.00 ms",
            conn._post_ready_summary(request),
        )

    def test_the_consumer_asks_for_a_push_once_earlier_forwards_finished(self):
        conn = self.make(producer=False)
        asked = []
        conn._request_write = lambda req_id, meta: asked.append(req_id)
        first, second = FakeEvent(), FakeEvent()
        conn._gated_writes.append((first, [("d1", None), ("d2", None)]))
        conn._gated_writes.append((second, [("d3", None)]))

        conn._request_gated_writes()
        self.assertEqual(asked, [])
        second.done = True
        conn._request_gated_writes()
        # In order: a later step's requests wait behind an earlier step's.
        self.assertEqual(asked, [])
        first.done = True
        conn._request_gated_writes()
        self.assertEqual(asked, ["d1", "d2", "d3"])
        self.assertFalse(conn._gated_writes)

    def test_get_finished_leaves_pushes_to_the_write_service(self):
        conn = self.make(producer=True)
        conn._write_service_thread = threading.current_thread()
        conn._service_write_requests = lambda: self.fail("served from the step")
        conn.get_finished()

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

    def test_state_is_registered_apart_from_the_block_layout(self):
        conn = self.make(producer=True)
        conn.transfer_mode = "write"
        advertised = self.register(conn, self.hybrid_caches())

        self.assertEqual(
            list(conn.layer_name_to_local_kv_cache_metadata), ["layer_1", "layer_2"]
        )
        self.assertEqual(list(conn._state_tensors), ["layer_0.k", "layer_0.v"])
        # The block geometry comes from the MLA layers, never from the state.
        self.assertEqual(tuple(conn.kv_cache_shape), (16, 1, 4))
        self.assertEqual(conn._kv_layout(), (True, 8))
        self.assertEqual(
            advertised.state_slot_bytes, {"layer_0.k": 48, "layer_0.v": 128}
        )
        self.assertEqual(set(advertised.state_descs), {"layer_0.k", "layer_0.v"})
        self.assertEqual(advertised.num_state_slots, 6)

    def test_a_model_without_state_advertises_none_on_the_wire(self):
        import msgspec

        conn = self.make(producer=True)
        conn.transfer_mode = "write"
        caches = self.hybrid_caches()
        del caches["layer_0"]
        advertised = self.register(conn, caches)
        # Omitted rather than empty, so the handshake is byte-for-byte what a
        # peer without state support sends.
        fields = msgspec.msgpack.decode(msgspec.msgpack.encode(advertised))
        self.assertFalse(
            {"state_descs", "state_slot_bytes", "num_state_slots"} & set(fields)
        )

    def test_read_mode_refuses_per_request_state(self):
        conn = self.make(producer=False)
        conn.transfer_mode = "read"
        with self.assertRaises(NotImplementedError):
            self.register(conn, self.hybrid_caches())

    def test_replayssm_records_and_cursor_are_registered_as_state(self):
        conn = self.make(producer=True)
        conn.transfer_mode = "write"
        advertised = self.register(conn, self.replay_caches())

        layer = ["k", "v", "replay_k", "replay_u", "replay_g"]
        self.assertEqual(
            list(conn._state_tensors),
            [f"layer_0.{part}" for part in layer]
            + [self.connector_module.REPLAY_CURSOR]
            + [f"layer_3.{part}" for part in layer],
        )
        # One int32 per slot, registered once for every layer.
        self.assertEqual(
            advertised.state_slot_bytes[self.connector_module.REPLAY_CURSOR], 4
        )
        self.assertEqual(advertised.state_slot_bytes["layer_3.replay_g"], 256)
        self.assertEqual(advertised.num_state_slots, 6)

    def test_replayssm_records_without_their_cursor_are_refused(self):
        conn = self.make(producer=True)
        conn.transfer_mode = "write"
        caches = self.replay_caches()
        for name in ("layer_0", "layer_3"):
            caches[name].replay_write_pos = None
        with self.assertRaises(ValueError) as caught:
            self.register(conn, caches)
        self.assertIn("replay_write_pos", str(caught.exception))

    def fabric_registration(self, conn):
        """Make `conn` register over the fabric: one PAGE region, and buffers
        whose packed descriptors carry a fabric handle."""
        types = self.types
        conn.use_fabric = True
        conn._verify_kv = False

        def register_local_buffer(ptr, size, device):
            fields = [b"key", 1, 0, b"bus", ptr, size, 1, b"", 0, 0, b"\x01" * 64]
            return self.msgpack.packb(fields + [16, b"ppod", 0, size, {}])

        conn.moriio_wrapper.register_local_buffer = register_local_buffer
        region = types.SimpleNamespace(
            base_addr=0x10000,
            total_bytes=8 * 16,
            unit_bytes=16,
            semantic_role="mla.kv",
            reverse_indexed=False,
        )
        view = types.SimpleNamespace(device=types.SimpleNamespace(index=0))
        return types.SimpleNamespace(
            pages=[types.SimpleNamespace(region=region, view=view)],
            swa_block_regions=[],
            num_blocks=8,
        )

    def test_the_fabric_registers_the_state_rdma_does(self):
        # ReplaySSM records and cursor included, and the same transfer slots.
        config = self.types.SimpleNamespace(state_transfer_slots=2)
        rdma = self.make(producer=True)
        rdma._config = config
        rdma_meta = self.register(rdma, self.replay_caches())

        conn = self.make(producer=True)
        conn._config = config
        tensors = self.fabric_registration(conn)
        meta = self.register(conn, self.replay_caches(), tensors)
        self.assertEqual(list(conn._state_tensors), list(rdma._state_tensors))
        self.assertEqual(meta.state_slot_bytes, rdma_meta.state_slot_bytes)
        self.assertEqual(set(meta.state_descs), set(rdma_meta.state_descs))
        self.assertEqual((meta.num_state_slots, conn._first_transfer_slot), (6, 4))
        self.assertEqual(rdma._first_transfer_slot, 4)

        caches = self.replay_caches()
        for name in ("layer_0", "layer_3"):
            caches[name].replay_write_pos = None
        conn = self.make(producer=True)
        tensors = self.fabric_registration(conn)
        with self.assertRaises(ValueError) as caught:
            self.register(conn, caches, tensors)
        self.assertIn("replay_write_pos", str(caught.exception))

    def test_the_cursor_is_pushed_and_digested_with_the_slot(self):
        conn = self.make(producer=True)
        conn.transfer_mode = "write"
        self.register(conn, self.replay_caches())
        cursor_name = self.connector_module.REPLAY_CURSOR
        _desc, _bytes, cursor = conn._state_tensors[cursor_name]
        conn.remote_moriio_metadata = {
            "engine": self.types.SimpleNamespace(num_state_slots=6)
        }
        conn._get_or_build_state_sessions = lambda engine_id: [
            (f"S-{name}", slot_bytes)
            for name, (_d, slot_bytes, _t) in conn._state_tensors.items()
        ]
        writes = conn._state_writes(
            self.push_request(src_slot_index=2, dst_slot_index=5), "engine"
        )
        self.assertIn((f"S-{cursor_name}", 4, 8, 20), writes)
        self.assertEqual(len(writes), len(conn._state_tensors))

        before = conn._state_digest(2)
        cursor[2] = 7
        after = conn._state_digest(2)
        self.assertNotEqual(before[cursor_name], after[cursor_name])
        self.assertEqual(
            {name for name in before if before[name] != after[name]}, {cursor_name}
        )

    def test_a_state_slot_that_is_not_one_run_is_refused(self):
        conn = self.make(producer=True)
        state = self.hybrid_caches()["layer_0"]
        state.k_cache = self.torch.zeros((8, 6), dtype=self.torch.uint8).t()
        with self.assertRaises(ValueError):
            conn._register_state("layer_0", state)

    def test_state_tensors_must_agree_on_the_slot_count(self):
        conn = self.make(producer=True)
        state = self.hybrid_caches(slots=6)["layer_0"]
        state.v_cache = self.torch.zeros((5, 2, 4, 4))
        with self.assertRaises(ValueError):
            conn._register_state("layer_0", state)

    def test_a_stateless_request_names_no_state_slot(self):
        conn = self.make(producer=False)
        conn._request_write("d1", self.recv_meta([4, 5], [10, 11]))
        ((frames, _host, _port),) = conn.sent
        request = self.msgpack.loads(frames[1])
        self.assertEqual(
            (request["src_slot_index"], request["dst_slot_index"]), (-1, -1)
        )

    def test_the_consumer_names_both_state_slots(self):
        conn = self.make_stateful(self.make(producer=False))
        meta = self.recv_meta([4, 5], [10, 11], local_slot_index=3, remote_slot_index=5)
        conn._request_write("d1", meta)
        ((frames, _host, _port),) = conn.sent
        request = self.msgpack.loads(frames[1])
        # The producer's slot is the source, ours the destination.
        self.assertEqual(request["src_slot_index"], 5)
        self.assertEqual(request["dst_slot_index"], 3)
        self.assertIn("d1", conn._awaiting_write)

    def test_a_stateful_request_without_a_slot_is_refused_before_asking(self):
        conn = self.make_stateful(self.make(producer=False))
        meta = self.recv_meta(
            [4, 5], [10, 11], local_slot_index=3, remote_slot_index=-1
        )
        with self.assertLogs("atom", "ERROR") as logs:
            conn._request_write("d1", meta)
        self.assertIn("refusing to move the KV without it", logs.output[0])
        self.assertEqual(conn.sent, [])
        self.assertEqual(conn._failed_before_read, {"d1"})
        self.assertNotIn("d1", conn._awaiting_write)
        # Rank 1 of the producer still holds the source blocks; let it go.
        self.assertEqual(conn.notified, [(0, "10.0.0.1", 7302)])

    def test_the_producer_writes_the_state_slot_with_the_blocks(self):
        conn = self.make_stateful(self.make(producer=True))
        sessions = self.block_sessions()
        conn._get_or_build_sessions = lambda engine_id: (sessions, None)
        request = self.push_request(
            dst_block_ids=[5, 1],
            src_block_ids=[2, 3],
            src_slot_index=2,
            dst_slot_index=5,
        )

        statuses = conn._write_blocks(request, "engine")

        # The four block writes as before, then one per state tensor at its slot.
        self.assertEqual(len(statuses), 6)
        self.assertIn(("S-k", [48], [120], [24]), conn.writes)
        self.assertIn(("S-v", [256], [640], [128]), conn.writes)

    def pending_push(self, conn, src_slot: int, *, queued_at: float | None = None):
        """Queue a push from `src_slot` whose consumer is already handshaken."""
        import time

        conn._get_or_build_sessions = lambda engine_id: (self.block_sessions(), None)
        request = self.push_request(
            src_slot_index=src_slot, dst_slot_index=5, deadline=time.time() + 60
        )
        engine_id = conn._write_engine_id(request)
        conn._remote_agents[engine_id] = set()
        conn.remote_moriio_metadata[engine_id] = conn.remote_moriio_metadata["engine"]
        conn._write_ids[9] = None
        queued = time.monotonic() if queued_at is None else queued_at
        conn._pending_writes = [(request, queued)]
        return request

    def test_a_live_slot_is_pushed_without_staging(self):
        conn = self.make_stateful(self.make(producer=True), slots=6)
        conn._first_transfer_slot = 4
        self.pending_push(conn, 2)
        conn._service_write_requests()
        self.assertIn(9, conn._sending_writes)

    def test_a_transfer_slot_is_pushed_only_once_its_copy_landed(self):
        conn = self.make_stateful(self.make(producer=True), slots=6)
        conn._first_transfer_slot = 4
        self.pending_push(conn, 5)
        conn._service_write_requests()
        # The runner has not copied the state there yet.
        self.assertEqual((conn.writes, len(conn._pending_writes)), ([], 1))

        landed = FakeEvent(done=False)
        conn._staged_sources[9] = (5, landed)
        conn._service_write_requests()
        self.assertEqual((conn.writes, len(conn._pending_writes)), ([], 1))

        landed.done = True
        conn._service_write_requests()
        self.assertEqual(conn._pending_writes, [])
        self.assertIn(9, conn._sending_writes)
        self.assertIn(("S-v", [640], [640], [128]), conn.writes)

    def test_a_push_from_a_slot_it_was_not_staged_into_fails(self):
        conn = self.make_stateful(self.make(producer=True), slots=6)
        conn._first_transfer_slot = 4
        self.pending_push(conn, 5)
        conn._staged_sources[9] = (4, FakeEvent(done=True))
        with self.assertLogs("atom", "ERROR"):
            conn._service_write_requests()
        self.assertEqual(conn.writes, [])
        self.assertEqual(conn.done_sending, {9})
        ((frames, _host, _port),) = conn.sent
        self.assertFalse(self.msgpack.loads(frames[1])["ok"])

    def test_a_copy_that_never_arrives_fails_after_the_handshake_timeout(self):
        import time

        conn = self.make_stateful(self.make(producer=True), slots=6)
        conn._first_transfer_slot = 4
        self.pending_push(conn, 5, queued_at=time.monotonic() - 301)
        with self.assertLogs("atom", "ERROR") as logs:
            conn._service_write_requests()
        self.assertIn("never reached transfer slot 5", logs.output[0])
        self.assertEqual(conn.done_sending, {9})

    def test_a_finished_or_released_transfer_forgets_its_copy(self):
        conn = self.make_stateful(self.make(producer=True), slots=6)
        conn._staged_sources = {9: (4, FakeEvent()), 10: (5, FakeEvent())}
        conn._finish_write(self.push_request(), ok=True)
        self.assertNotIn(9, conn._staged_sources)
        # A consumer that failed before asking releases the source by notify.
        conn.done_sending.add(10)
        conn.get_finished()
        self.assertEqual(conn._staged_sources, {})

    def test_one_event_marks_every_copy_of_a_step(self):
        conn = self.make(producer=True)
        meta = self.types.SimpleNamespace(state_stages={9: (1, 4), 10: (2, 5)})
        with mock.patch.object(self.torch.cuda, "Event", FakeEvent):
            conn.start_load_kv(meta)
        (slot9, event9), (slot10, event10) = (
            conn._staged_sources[9],
            conn._staged_sources[10],
        )
        self.assertEqual((slot9, slot10), (4, 5))
        self.assertIs(event9, event10)
        self.assertTrue(event9.recorded)

    def test_the_producer_refuses_a_push_that_names_no_slot(self):
        conn = self.make_stateful(self.make(producer=True))
        sessions = self.block_sessions()
        conn._get_or_build_sessions = lambda engine_id: (sessions, None)
        with self.assertRaises(ValueError):
            conn._write_blocks(self.push_request(), "engine")
        self.assertEqual(conn.writes, [])

    def test_a_slot_past_the_consumers_pool_is_refused(self):
        conn = self.make_stateful(self.make(producer=True), slots=6)
        sessions = self.block_sessions()
        conn._get_or_build_sessions = lambda engine_id: (sessions, None)
        request = self.push_request(src_slot_index=2, dst_slot_index=6)
        with self.assertRaises(ValueError):
            conn._write_blocks(request, "engine")
        self.assertEqual(conn.writes, [])

    def test_state_cannot_cross_a_tp_mismatch(self):
        conn = self.make_stateful(self.make(producer=True))
        sessions = self.block_sessions()
        conn._get_or_build_sessions = lambda engine_id: (sessions, None)
        request = self.push_request(
            src_slot_index=2, dst_slot_index=5, decode_tp_size=4
        )
        with self.assertRaises(ValueError) as caught:
            conn._write_blocks(request, "engine")
        self.assertIn("the consumer runs TP 4", str(caught.exception))
        self.assertEqual(conn.writes, [])

    def test_under_verify_the_report_carries_the_source_digest(self):
        conn = self.make_stateful(self.make(producer=True))
        conn._verify_kv = True
        _desc, _slot_bytes, conv = conn._state_tensors["kda.k"]
        conv[2] = 9
        sessions = self.block_sessions()
        conn._get_or_build_sessions = lambda engine_id: (sessions, None)
        request = self.push_request(src_slot_index=2, dst_slot_index=5)

        conn._write_blocks(request, "engine")
        # Digested from our slot before it is pushed, not from the consumer's.
        self.assertEqual(request["state_digest"], conn._state_digest(2))
        conn._finish_write(request, ok=True)
        ((frames, _host, _port),) = conn.sent
        self.assertEqual(
            self.msgpack.loads(frames[1])["state_digest"], request["state_digest"]
        )

    def stateful_peer(self, conn, **remote):
        """`conn` holding one KDA layer of state, facing a consumer's `remote`."""
        import types

        conn._state_tensors = {
            "kda.k": (b"k-ours", 24, None),
            "kda.v": (b"v-ours", 128, None),
        }
        conn.remote_moriio_metadata = {"engine": types.SimpleNamespace(**remote)}
        conn.moriio_wrapper.get_unpack_memory_metadata = lambda packed: packed
        conn.moriio_wrapper.build_session = lambda local, remote: (local, remote)
        return conn

    def test_state_sessions_pair_each_tensor_with_the_peers(self):
        conn = self.stateful_peer(
            self.make(producer=True),
            state_slot_bytes={"kda.k": 24, "kda.v": 128},
            state_descs={"kda.k": b"k-theirs", "kda.v": b"v-theirs"},
        )
        sessions = conn._get_or_build_state_sessions("engine")
        self.assertEqual(
            sessions,
            [((b"k-ours", b"k-theirs"), 24), ((b"v-ours", b"v-theirs"), 128)],
        )
        # Built once per consumer.
        conn.moriio_wrapper.build_session = None
        self.assertIs(conn._get_or_build_state_sessions("engine"), sessions)

    def test_state_sessions_refuse_a_peer_with_different_state(self):
        descs = {"kda.k": b"k-theirs", "kda.v": b"v-theirs"}
        for remote in (
            {"state_slot_bytes": None, "state_descs": None},
            {"state_slot_bytes": {"kda.k": 24, "kda.v": 64}, "state_descs": descs},
            {"state_slot_bytes": {"kda.k": 24}, "state_descs": descs},
        ):
            with self.subTest(remote=remote):
                conn = self.stateful_peer(self.make(producer=True), **remote)
                with self.assertRaises(ValueError):
                    conn._get_or_build_state_sessions("engine")
                self.assertEqual(conn._built_state_sessions, {})

    def test_a_replayssm_mismatch_names_the_record_tensor(self):
        conn = self.stateful_peer(
            self.make(producer=True),
            state_slot_bytes={"kda.k": 24, "kda.v": 128, "kda.replay_k": 64},
            state_descs={"kda.k": b"k-theirs", "kda.v": b"v-theirs"},
        )
        with self.assertRaises(ValueError) as caught:
            conn._get_or_build_state_sessions("engine")
        self.assertIn(
            "kda.replay_k: ours None B/slot, theirs 64", str(caught.exception)
        )

    def test_state_sessions_refuse_our_own_descriptors(self):
        conn = self.stateful_peer(
            self.make(producer=True),
            state_slot_bytes={"kda.k": 24, "kda.v": 128},
            state_descs={"kda.k": b"k-ours", "kda.v": b"v-theirs"},
        )
        with self.assertRaises(ValueError):
            conn._get_or_build_state_sessions("engine")


class SchedulerSlotTest(unittest.TestCase):
    """Both ends' state slots ride the scheduler's transfer params."""

    @classmethod
    def setUpClass(cls):
        cls.scheduler = import_connector("MoRIIOConnectorScheduler")
        if cls.scheduler is None:
            raise unittest.SkipTest("moriio_connector needs aiter, torch and a GPU")

    def make(self, *, producer: bool):
        """A scheduler holding only what `request_finished` and the recv path read."""
        sched = object.__new__(self.scheduler)
        sched.is_producer = producer
        sched.engine_id = "10.0.0.1:7300"
        sched.host_ip = "10.0.0.1"
        sched.handshake_port = 7300
        sched.base_handshake_port = 7300
        sched.tp_size = 8
        sched.dp_rank = 0
        sched._reqs_need_recv = {}
        sched._awaiting_send = set()
        sched.hash_block_size = 4
        sched._state_stages = {}
        sched.request_id_to_transfer_id = {}
        sched.transfer_id_to_request_id = {}
        return sched

    @staticmethod
    def seq(**attrs):
        import types

        fields = {
            "id": 7,
            "block_table": [1, 2, 3],
            "kv_transfer_params": {},
            "output_tokens": [42],
            "spec_token_ids": None,
            "num_cached_tokens": 0,
        }
        fields.update(attrs)
        return types.SimpleNamespace(**fields)

    def test_the_lookup_answers_until_the_receive_is_queued(self):
        sched = self.make(producer=False)
        seq = self.seq(
            kv_transfer_params={"do_remote_prefill": True, "transfer_id": 7},
            prompt_token_ids=[1, 2, 3],
            state_slot=-1,
        )
        # Held back after the first yes (a full running set), asked again.
        self.assertEqual(sched.get_num_new_matched_tokens(seq), (3, True))
        self.assertEqual(sched.get_num_new_matched_tokens(seq), (3, True))
        sched.update_state_after_alloc(seq)
        self.assertEqual(sched.get_num_new_matched_tokens(seq), (0, False))

    def test_the_consumer_records_the_prefix_it_holds(self):
        sched = self.make(producer=False)
        seq = self.seq(
            kv_transfer_params={"do_remote_prefill": True, "transfer_id": 7},
            block_table=[1, 2, 3],
            num_cached_tokens=8,
            state_slot=3,
        )
        sched.update_state_after_alloc(seq)
        self.assertEqual(seq.kv_transfer_params["num_computed_blocks"], 2)
        self.assertTrue(sched.receives_suffix_and_state)
        self.assertFalse(self.make(producer=True).receives_suffix_and_state)

    def test_a_staged_copy_rides_the_next_metadata_once(self):
        sched = self.make(producer=True)
        sched.stage_state_for_send(self.seq(), 1, 4)
        self.assertEqual(sched.build_connector_meta().state_stages, {7: (1, 4)})
        self.assertEqual(sched.build_connector_meta().state_stages, {})

    def test_a_send_released_before_its_copy_drops_the_copy(self):
        sched = self.make(producer=True)
        sched.stage_state_for_send(self.seq(), 1, 4)
        sched.send_finished("7")
        self.assertEqual(sched.build_connector_meta().state_stages, {})

    def test_only_a_producer_stages_state(self):
        with self.assertRaises(RuntimeError):
            self.make(producer=False).stage_state_for_send(self.seq(), 1, 4)

    def test_the_producer_publishes_its_state_slot(self):
        sched = self.make(producer=True)
        seq = self.seq(state_slot=6)
        sched.request_finished(seq)
        self.assertEqual(seq.kv_transfer_params_output["remote_slot_index"], 6)

    def test_a_model_without_state_publishes_no_slot(self):
        sched = self.make(producer=True)
        seq = self.seq()
        sched.request_finished(seq)
        self.assertEqual(seq.kv_transfer_params_output["remote_slot_index"], -1)

    def test_the_consumer_hands_both_slots_to_its_worker(self):
        sched = self.make(producer=False)
        seq = self.seq(
            state_slot=4,
            kv_transfer_params={
                "do_remote_prefill": True,
                "transfer_id": 9,
                "remote_block_ids": [5, 6, 7],
                "remote_slot_index": 6,
                "remote_host": "10.0.0.1",
                "remote_handshake_port": 7300,
                "tp_size": 8,
            },
        )
        sched.update_state_after_alloc(seq)
        meta = sched.build_connector_meta().reqs_to_recv[7]
        self.assertEqual((meta.local_slot_index, meta.remote_slot_index), (4, 6))
        self.assertEqual(meta.local_block_ids, [1, 2, 3])


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


class RegionPlanTest(unittest.TestCase):
    """Pair local and remote regions on the CPU lane."""

    @classmethod
    def setUpClass(cls):
        import dataclasses
        import json

        import numpy

        from atom.kv_transfer.disaggregation.types import KVTransferRegion

        tree = ast.parse(CONNECTOR.read_text())
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        wanted = {"_remote_regions", "_region_plan", "merge_contiguous_blocks"}
        methods = [
            n
            for n in classes["MoRIIOConnector"].body
            if isinstance(n, ast.FunctionDef) and n.name in wanted
        ]
        scope = {
            "Any": object,
            "EngineId": str,
            "KVTransferRegion": KVTransferRegion,
            "MAX_RDMA_CHUNK_BYTES": 40,
            "dataclass": dataclasses.dataclass,
            "json": json,
            "np": numpy,
        }
        module = ast.Module([classes["_Region"], *methods], [])
        exec(compile(module, "<regions>", "exec"), scope)  # noqa: S102
        cls.json = json
        cls.Region = scope["_Region"]
        cls.Planner = type("Planner", (), {name: scope[name] for name in wanted})
        cls.local = [
            cls.Region("page.0.kv", "page", 0, 80, 10, False, 40),
            cls.Region("slot.0.swa", "slot", 0, 80, 10, True, 40),
        ]

    def make(self, remote_regions):
        import types

        conn = self.Planner()
        conn._regions = self.local
        layout = self.json.dumps([r.layout() for r in remote_regions])
        conn.remote_moriio_metadata = {
            "decode": types.SimpleNamespace(region_layout=layout)
        }
        conn._region_session = lambda _engine, ri, lc, rc: (ri, lc, rc)
        return conn

    def test_pairs_page_and_reverse_slot_regions(self):
        request = {
            "src_block_ids": [3, 4],
            "dst_block_ids": [0, 5],
            "src_slot_index": 0,
            "dst_slot_index": 5,
        }
        fewer_slots = self.Region("slot.0.swa", "slot", 0, 60, 10, True, 40)
        self.assertEqual(
            self.make([self.local[0], fewer_slots])._region_plan(request, "decode"),
            [
                ((0, 0, 0), [30], [0], [10]),
                ((0, 1, 1), [0], [10], [10]),
                ((1, 1, 0), [30], [0], [10]),
            ],
        )

    def test_rejects_a_different_region_unit(self):
        other = self.Region("page.0.kv", "page", 0, 80, 20, False, 40)
        with self.assertRaises(ValueError):
            self.make([other, self.local[1]])._remote_regions("decode")

    def test_of_keeps_each_unit_inside_one_chunk(self):
        import types

        region = types.SimpleNamespace(
            unit_bytes=10, total_bytes=80, base_addr=1000, reverse_indexed=False
        )
        built = self.Region.of("page.0.kv", "page", region)
        chunks = built.chunks()
        self.assertEqual(sum(size for _, size in chunks), built.total)
        for ptr, size in chunks:
            self.assertEqual((ptr - built.base) % built.unit, 0)
            self.assertEqual(size % built.unit, 0)

    def test_of_rejects_a_unit_over_the_rdma_limit(self):
        import types

        region = types.SimpleNamespace(
            unit_bytes=41, total_bytes=41, base_addr=0, reverse_indexed=False
        )
        with self.assertRaises(ValueError):
            self.Region.of("page.0.kv", "page", region)

    def test_of_rejects_a_partial_unit(self):
        import types

        region = types.SimpleNamespace(
            unit_bytes=10, total_bytes=85, base_addr=0, reverse_indexed=True
        )
        with self.assertRaises(ValueError):
            self.Region.of("slot.0.swa", "slot", region)


if __name__ == "__main__":
    unittest.main()
