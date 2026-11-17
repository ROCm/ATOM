"""ATOM_DRAFT_SLOT_CHECK: draft KV writes outside the forward's own blocks."""

import types
import unittest

import numpy as np
import torch

from atom.spec_decode.slot_check import DraftSlotCheck, foreign_writes


def batch(req_ids, tables, scheduled):
    return types.SimpleNamespace(
        req_ids=req_ids,
        block_tables=tables,
        num_scheduled_tokens=scheduled,
        total_seqs_num_prefill=len(req_ids),
        is_dummy_run=False,
    )


class ForeignWritesTest(unittest.TestCase):
    def test_token_entries_map_to_their_rows(self):
        blocks = np.array([4, 4, 9, 5, -1, 7])
        foreign = np.array([0, 0, 1, 0, 0, 1])
        index, hit, rows = foreign_writes(blocks, foreign, np.array([0, 2, 4]))
        self.assertEqual(index.tolist(), [2, 5])
        self.assertEqual(hit.tolist(), [9, 7])
        self.assertEqual(rows.tolist(), [1, 2])

    def test_row_entries_are_their_own_rows(self):
        _, _, rows = foreign_writes(np.array([1, 8]), np.array([0, 1]), None)
        self.assertEqual(rows.tolist(), [1])


class DraftSlotCheckTest(unittest.TestCase):
    def setUp(self):
        self.seen = []
        self.check = DraftSlotCheck(
            4, "cpu", lambda blocks: self.seen.append(blocks) or "held by x"
        )
        # Request 10 holds blocks 2 and 3, request 11 block 6; 3 + 2 tokens.
        self.check.begin_forward(batch([10, 11], [[2, 3], [6]], [3, 2]))

    def test_writes_into_the_batchs_blocks_pass(self):
        slots = torch.tensor([8, 13, -1, 24, 27])
        with self.assertNoLogs("atom", "ERROR"):
            self.check.check(slots, "propose step 0", per_token=True)
            self.check.poll()
        self.assertEqual((self.check.checks, self.check.reports), (1, 0))

    def test_a_write_into_another_requests_block_is_reported(self):
        # Token 4 (request 11's second) writes slot 21, block 5.
        slots = torch.tensor([8, 13, 14, 24, 21])
        with self.assertLogs("atom", "ERROR") as logs:
            self.check.check(slots, "propose step 0", per_token=True)
            self.check.poll()
        line = logs.output[0]
        self.assertIn("forward 1 propose step 0: 1 of 5 draft KV writes", line)
        self.assertIn("outside the 3 blocks this batch holds", line)
        self.assertIn("blocks [5] at entries [4], written by rows [1]", line)
        self.assertIn("requests [11]", line)
        self.assertIn("those blocks: held by x", line)
        self.assertEqual(self.seen, [[5]])

    def test_a_padded_step_with_live_slots_is_reported_per_row(self):
        # One row per sequence plus a pad row whose slot was left live.
        slots = torch.tensor([12, 26, 40])
        with self.assertLogs("atom", "ERROR") as logs:
            self.check.check(slots, "propose step 1", per_token=False)
            self.check.poll()
        self.assertIn("blocks [10] at entries [2], written by rows [2]", logs.output[0])
        self.assertIn("requests []", logs.output[0])

    def test_read_backs_stay_on_the_host_whatever_the_default_device(self):
        # The runner sets a GPU default device; "meta" stands in for it here.
        with torch.device("meta"):
            self.check.begin_forward(batch([10], [[2]], [1]))
            self.assertEqual(self.check._meta["_host"].device.type, "cpu")
        with self.assertLogs("atom", "ERROR"):
            self.check.check(torch.tensor([21]), "propose step 0", per_token=True)
            self.check.poll()


if __name__ == "__main__":
    unittest.main()
