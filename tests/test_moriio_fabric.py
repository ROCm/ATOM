# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU tests for MoRIIO over the fabric (`moriio_backend` `fabric`).

`atom.kv_transfer.disaggregation.moriio.fabric` imports only torch and msgpack,
but its package imports the connector; the connector wiring is asserted
against the source.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import msgpack
import pytest
import torch
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    # `importorskip`, as in `test_pd_source_claim.py`: the package reaches
    # triton through `disaggregation.utils`, which a CPU-only runner lacks.
    _SKIP = "P/D backend deps (triton) are absent on a CPU-only runner"
    fabric = pytest.importorskip(
        "atom.kv_transfer.disaggregation.moriio.fabric", reason=_SKIP
    )
    MoRIIOAgentMetadata = pytest.importorskip(
        "atom.kv_transfer.disaggregation.moriio.moriio_common", reason=_SKIP
    ).MoRIIOAgentMetadata

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = ROOT / "atom/kv_transfer/disaggregation/moriio/moriio_connector.py"


def packed_desc(handle: bytes, vpod: int = 16) -> bytes:
    fields = [b"key", 1, 0, b"bus", 0x1000, 4096, 1, b"", 0, 0, handle, vpod]
    return msgpack.packb(fields + [b"ppod", 0, 4096, {}])


class BlockRunsTest(unittest.TestCase):
    def test_runs_contiguous_on_both_sides_merge(self):
        self.assertEqual(fabric.block_runs([4, 5, 6], [10, 11, 12]), [(4, 10, 3)])

    def test_a_gap_on_either_side_starts_a_run(self):
        self.assertEqual(
            fabric.block_runs([4, 5, 7, 8], [10, 11, 12, 13]),
            [(4, 10, 2), (7, 12, 2)],
        )
        self.assertEqual(
            fabric.block_runs([1, 2, 3], [5, 9, 10]), [(1, 5, 1), (2, 9, 2)]
        )

    def test_empty(self):
        self.assertEqual(fabric.block_runs([], []), [])


class PageOffsetsTest(unittest.TestCase):
    def test_every_region_gets_every_run_at_its_own_offsets(self):
        local = [(0, 16), (1000, 4)]
        remote = [(64, 16), (5000, 4)]
        offs = fabric.page_offsets(local, remote, [(2, 3, 2), (7, 0, 1)])
        self.assertEqual(
            offs,
            (
                [32, 112, 1008, 1028],
                [64 + 48, 64, 5012, 5000],
                [32, 16, 8, 4],
            ),
        )

    def test_regions_pair_by_position(self):
        local = [(0, 8), (100, 8)]
        remote = [(100, 8), (0, 8)]
        l, r, _ = fabric.page_offsets(local, remote, [(0, 0, 1)])
        self.assertEqual((l, r), ([0, 100], [100, 0]))


class PoolSpanTest(unittest.TestCase):
    def tearDown(self):
        fabric._pool_range = None

    def test_regions_in_the_pool_register_the_whole_pool(self):
        fabric._pool_range = (1000, 500)
        self.assertEqual(fabric.pool_span([(1100, 10), (1300, 50)]), (1000, 500))

    def test_regions_outside_any_pool_register_their_hull(self):
        self.assertEqual(fabric.pool_span([(300, 10), (100, 50)]), (100, 210))

    def test_a_region_outside_the_pool_falls_back_to_the_hull(self):
        fabric._pool_range = (1000, 500)
        with self.assertLogs("atom", "WARNING"):
            self.assertEqual(fabric.pool_span([(1100, 10), (1600, 4)]), (1100, 504))


class DescFabricInfoTest(unittest.TestCase):
    def test_a_handle_and_its_vpod(self):
        self.assertEqual(fabric.desc_fabric_info(packed_desc(b"\x01" * 64)), (True, 16))

    def test_an_all_zero_handle_is_no_handle(self):
        self.assertFalse(fabric.desc_fabric_info(packed_desc(b"\x00" * 64, -1))[0])


class PageDigestTest(unittest.TestCase):
    def views(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        return [
            torch.randint(0, 256, (8, 1, unit), dtype=torch.uint8, generator=g)
            for unit in (16, 4, 16)
        ]

    def test_same_bytes_in_other_blocks_digest_equal(self):
        src, dst = self.views(0), self.views(1)
        for s, d in zip(src, dst):
            d[[5, 6]] = s[[1, 2]]
        self.assertEqual(fabric.page_digest(src, [1, 2]), fabric.page_digest(dst, [5, 6]))

    def test_one_stale_region_changes_the_digest(self):
        src, dst = self.views(0), self.views(1)
        for s, d in zip(src[:2], dst[:2]):
            d[[5, 6]] = s[[1, 2]]
        self.assertNotEqual(
            fabric.page_digest(src, [1, 2]), fabric.page_digest(dst, [5, 6])
        )
        src_terms = fabric.page_region_digests(src, [1, 2])
        dst_terms = fabric.page_region_digests(dst, [5, 6])
        self.assertEqual(sum(src_terms), fabric.page_digest(src, [1, 2]))
        self.assertEqual([s == d for s, d in zip(src_terms, dst_terms)], [True, True, False])

    def test_swapped_regions_change_the_digest(self):
        a = torch.arange(32, dtype=torch.uint8).view(2, 1, 16)
        b = (torch.arange(32, dtype=torch.uint8) + 1).view(2, 1, 16)
        self.assertNotEqual(fabric.page_digest([a, b], [0]), fabric.page_digest([b, a], [0]))


class MetadataTest(unittest.TestCase):
    def test_page_layout_survives_the_handshake_encoding(self):
        meta = MoRIIOAgentMetadata(
            engine_id="e",
            agent_metadata=b"a",
            page_desc=b"d",
            page_regions=[[0, 16384, "mha.k.layer_0"], [16384, 512, "mha.k_scale"]],
        )
        decoded = msgspec_roundtrip(meta)
        self.assertEqual(decoded.page_desc, b"d")
        self.assertEqual(decoded.page_regions, meta.page_regions)

    def test_an_rdma_peer_decodes_without_a_page_layout(self):
        decoded = msgspec_roundtrip(MoRIIOAgentMetadata(engine_id="e", agent_metadata=b"a"))
        self.assertIsNone(decoded.page_desc)
        self.assertIsNone(decoded.page_regions)


def msgspec_roundtrip(meta):
    import msgspec

    return msgspec.msgpack.Decoder(MoRIIOAgentMetadata).decode(
        msgspec.msgpack.Encoder().encode(meta)
    )


class ConnectorWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = CONNECTOR.read_text()

    def segment(self, name: str) -> str:
        tree = ast.parse(self.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(self.source, node) or ""
        raise AssertionError(f"{name} not found")

    def test_fabric_never_creates_rdma_next_to_it(self):
        init = self.segment("__init__")
        self.assertIn("if self.use_fabric:", init)
        self.assertIn("self._create_fabric_backend", init)
        self.assertIn("else:\n            self._create_rdma_backend", init)

    def test_a_missing_copy_kernel_is_an_error(self):
        body = self.segment("_create_fabric_backend")
        self.assertIn('simplefilter("error", RuntimeWarning)', body)

    def test_the_pool_is_registered_once_and_checked_for_a_handle(self):
        body = self.segment("_register_pages")
        self.assertEqual(body.count("register_local_buffer"), 1)
        self.assertIn("if not has_handle:", body)

    def test_both_directions_take_the_page_path(self):
        self.assertIn("self._write_pages(", self.segment("_write_blocks"))
        self.assertIn("self._get_or_build_page_session(", self.segment("_read_blocks"))

    def test_the_peer_layout_is_compared_before_a_session(self):
        body = self.segment("_get_or_build_page_session")
        self.assertLess(body.index("first difference"), body.index("build_session"))
        self.assertIn("our own PAGE descriptor", body)


if __name__ == "__main__":
    unittest.main()
