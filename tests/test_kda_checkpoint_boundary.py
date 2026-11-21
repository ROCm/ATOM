"""CPU byte/control tests for the temporary fixed-request observer."""

import array
import importlib.util
import json
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

SOURCE = Path(__file__).parents[1] / "atom/model_engine/kda_checkpoint_boundary.py"
spec = importlib.util.spec_from_file_location("boundary", SOURCE)
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


class Vector(list):
    def tolist(self):
        return list(self)

    def numel(self):
        return len(self)


class PackingTest(unittest.TestCase):
    def test_preserves_typed_bytes_order_and_signed_zero(self):
        floats = array.array("f", [1.0, -0.0])
        words = array.array("H", [0x1234])
        packed = boundary.pack_bytes([memoryview(floats), memoryview(words)], 10)
        self.assertEqual(packed, struct.pack("=ffH", 1.0, -0.0, 0x1234))
        self.assertNotEqual(packed, struct.pack("=ffH", 1.0, 0.0, 0x1234))
        floats[0] = 9.0
        self.assertEqual(packed, struct.pack("=ffH", 1.0, -0.0, 0x1234))

    def test_partition_changes_do_not_change_the_byte_stream(self):
        self.assertEqual(
            boundary.pack_bytes([b"AB", b"CDEF"], 6),
            boundary.pack_bytes([b"ABCDE", b"F"], 6),
        )
        self.assertNotEqual(
            boundary.pack_bytes([b"AB", b"CDEF"], 6),
            boundary.pack_bytes([b"CDEF", b"AB"], 6),
        )

    def test_refuses_short_long_and_noncontiguous_streams(self):
        for chunks, size in (([b"a"], 2), ([b"abc"], 2)):
            with self.assertRaises(ValueError):
                boundary.pack_bytes(chunks, size)
        with self.assertRaises(TypeError):
            boundary.pack_bytes([memoryview(b"abcdef")[::2]], 3)


class ObserverTest(unittest.TestCase):
    def setUp(self):
        self.page = b"PAGE"
        self.slots = {3: b"PAGE", 4: b"BAD!"}
        self.lines = []
        self.layout = "kda-paged-state-v1:test:tp=8:spec=0:carry=all"
        cfg = NS(
            compilation_config=NS(level=0),
            speculative_config=None,
            tensor_parallel_size=8,
            tp_world_size=8,
            pipeline_parallel_size=1,
            parallel_config=NS(data_parallel_size=1),
        )
        self.runner = NS(
            config=cfg,
            enforce_eager=True,
            hf_text_config=NS(model_type="kimi_linear"),
            state_runtime=NS(checkpoint_spec=NS(image_bytes=4, layout_id=self.layout)),
            replayssm_write_pos={3: -1, 4: -1},
        )
        self.builder = NS(
            model_runner=self.runner,
            page_unit_views=lambda units: [self.page[:1], self.page[1:]],
            state_entry_views=lambda slot: [self.slots[slot]],
        )
        self.md = NS(
            non_spec_state_indices_in_tensor=Vector([3]),
            non_spec_state_indices_tensor=Vector([3]),
            has_initial_state=Vector([True]),
            non_spec_query_start_loc=Vector([0, 512]),
            num_prefills=1,
            num_actual_tokens=512,
        )
        self.metadata = NS(gdn_metadata=self.md)
        self.positions = Vector(range(512, 1024))
        self.observer = boundary.KDACheckpointBoundary(0)
        self.addCleanup(patch.stopall)
        patch.object(boundary, "IMAGE_BYTES", 4).start()
        patch.object(boundary, "image_bytes", side_effect=boundary.pack_bytes).start()
        patch.object(boundary.os, "write", side_effect=self.write).start()

    def write(self, fd, line):
        self.assertEqual(fd, 1)
        self.lines.append(line)
        return len(line)

    def batch(self, storing=False, request="warm"):
        op = NS(
            unit_ids=(29, 30),
            layout_id=self.layout,
            total_bytes=4,
            src_slot=7,
            scratch_slot=-1,
            dst_slot=3,
        )
        return NS(
            state_maintenance_ops=NS(
                checkpoint_stores=[op] if storing else [],
                checkpoint_restores=[] if storing else [op],
            ),
            is_dummy_run=False,
            total_seqs_num=1,
            req_ids=(request,),
            has_prefill=not storing,
            num_cached_tokens=[1024 if storing else 512],
            num_scheduled_tokens=[1 if storing else 512],
        )

    def observe(self, storing=False, request="warm"):
        ticket = self.observer.before(self.builder, self.batch(storing, request))
        self.observer.after(ticket, self.metadata, self.positions)
        return json.loads(self.lines[-1].decode()[len(boundary.PREFIX) :])

    def test_one_reference_then_two_restores_are_bounded(self):
        self.observe(True, "cold")
        for request in ("warm0", "warm1"):
            row = self.observe(request=request)
            for key in (
                "identity_matches",
                "page_unchanged",
                "restored_matches_incoming",
                "restored_matches_reference",
                "controls_expected",
            ):
                self.assertIs(row[key], True, key)
        self.assertEqual(self.observer.count, 3)
        self.assertLessEqual(sum(map(len, self.lines)) * 8, 65536)
        with self.assertRaisesRegex(RuntimeError, "one store then two restores"):
            self.observe(request="extra")

    def test_unequal_bytes_survive_even_when_recorded_hashes_collide(self):
        patch.object(
            boundary.hashlib, "sha256", return_value=NS(hexdigest=lambda: "0" * 64)
        ).start()
        self.observe(True)
        self.page = self.slots[3] = b"NEW!"
        changed = self.observe(request="warm0")
        self.assertFalse(changed["page_unchanged"])
        self.assertTrue(changed["restored_matches_incoming"])
        self.assertFalse(changed["restored_matches_reference"])
        self.page = b"PAGE"
        stale = self.observe(request="warm1")
        self.assertTrue(stale["page_unchanged"])
        self.assertFalse(stale["restored_matches_incoming"])
        self.assertFalse(stale["restored_matches_reference"])
        self.assertEqual(changed["page_sha256"], stale["page_sha256"])

    def test_reads_actual_slot_and_reports_wrong_controls(self):
        self.observe(True)
        self.md.non_spec_state_indices_in_tensor = Vector([4])
        row = self.observe(request="wrong-slot")
        self.assertEqual(row["controls"]["read_slot"], 4)
        self.assertFalse(row["restored_matches_incoming"])
        self.assertFalse(row["controls_expected"])
        self.md.non_spec_state_indices_in_tensor = Vector([3])
        self.md.has_initial_state = Vector([False])
        self.runner.replayssm_write_pos[3] = 0
        row = self.observe(request="wrong-flags")
        self.assertTrue(row["restored_matches_incoming"])
        self.assertFalse(row["controls_expected"])

    def test_refuses_wrong_scope_order_and_oversized_record(self):
        with self.assertRaisesRegex(RuntimeError, "order changed"):
            self.observe()
        self.runner.config.compilation_config.level = 3
        with self.assertRaisesRegex(RuntimeError, "fixed eager Kimi scope"):
            self.observe(True)
        self.runner.config.compilation_config.level = 0
        with self.assertRaisesRegex(RuntimeError, "per-record bound"):
            self.observe(True, "x" * 3000)
        self.assertEqual(self.lines, [])

    def test_dummy_and_empty_batches_do_not_observe(self):
        batch = self.batch(True)
        batch.is_dummy_run = True
        self.assertIsNone(self.observer.before(self.builder, batch))
        batch.is_dummy_run = False
        batch.state_maintenance_ops.checkpoint_stores = []
        self.assertIsNone(self.observer.before(self.builder, batch))
        self.assertEqual(self.observer.count, 0)


if __name__ == "__main__":
    unittest.main()
