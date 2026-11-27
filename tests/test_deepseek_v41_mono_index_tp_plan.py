# SPDX-License-Identifier: MIT
"""TP work coverage, stable key ranges and the distributed mailbox ABI."""

import unittest
from itertools import pairwise

from atom.mono.dsa_indexer.paged import tp_plan as tp


class IndexTPPlanTests(unittest.TestCase):
    def test_small_buckets_are_local_only_in_bounded_deployments(self):
        for world in (2, 4, 8):
            for rows in (1, 6, 8):
                for capacity in (8320, 32768, 163968, 262144):
                    self.assertEqual(tp.kernel_world(world, rows, capacity), 1)
                self.assertEqual(tp.kernel_world(world, rows, 262272), world)
            for rows in (9, 12, 16, 40, 48, 64):
                self.assertEqual(tp.kernel_world(world, rows, 163968), world)
        self.assertEqual(tp.kernel_world(1, 8, 163968), 1)

    def assert_coverage(self, bounds, batches, world):
        ranks = [tp.reference_work(bounds, batches, world, r) for r in range(world)]
        for row, (bound, batch) in enumerate(zip(bounds, batches)):
            work = [rank[row] for rank in ranks]
            live = [w for w in work if w.count]
            if batch < 0:
                self.assertFalse(live)
                continue
            self.assertEqual(len({w.kind for w in work}), 1)
            if work[0].kind == tp.LOCAL:
                self.assertTrue(all((w.first, w.count) == (0, bound) for w in work))
                continue
            pieces = sorted((w.first, w.first + w.count) for w in live)
            cursor = 0
            for first, last in pieces:
                self.assertEqual(first, cursor)
                self.assertLessEqual(last, bound)
                cursor = last
            self.assertEqual(cursor, bound)
            if work[0].kind == tp.OWNED:
                self.assertLessEqual(len(live), 1)

    def test_low_batch_splits_keys_with_full_coverage(self):
        for world in (2, 4, 8):
            for bound in (8193, 10240, 18432, 81920, 163840, 1048576):
                with self.subTest(world=world, bound=bound):
                    self.assert_coverage([bound], [7], world)
                    self.assertEqual(
                        tp.reference_work([bound], [7], world, 0)[0].kind, tp.SPLIT
                    )

    def test_short_decode_and_verify_stay_local(self):
        for world in (1, 2, 4, 8):
            for qlen in (1, 6):
                for requests in (1, 8):
                    bounds = [8192 + t for _ in range(requests) for t in range(qlen)]
                    batches = [i for i in range(requests) for _ in range(qlen)]
                    self.assert_coverage(bounds, batches, world)
                    self.assertTrue(
                        all(
                            w.kind == tp.LOCAL
                            for w in tp.reference_work(bounds, batches, world, 0)
                        )
                    )

    def test_large_batch_owns_whole_requests(self):
        for world in (2, 4, 8):
            for qlen in (1, 6):
                bounds = [163840 + t for _ in range(8) for t in range(qlen)]
                batches = [10 * i for i in range(8) for _ in range(qlen)]
                self.assert_coverage(bounds, batches, world)
                ranks = [
                    tp.reference_work(bounds, batches, world, r) for r in range(world)
                ]
                for request in range(8):
                    for row in range(request * qlen, (request + 1) * qlen):
                        self.assertEqual(ranks[request % world][row].kind, tp.OWNED)
                        self.assertEqual(ranks[request % world][row].count, bounds[row])

    def test_dominant_long_request_splits_while_short_ones_stay_owned(self):
        bounds = [524288, 9000, 10000, 11000, 12000, 13000, 14000, 15000]
        for world in (2, 4, 8):
            self.assert_coverage(bounds, list(range(8)), world)
            work = tp.reference_work(bounds, list(range(8)), world, 0)
            self.assertEqual(work[0].kind, tp.SPLIT)
            self.assertTrue(all(w.kind == tp.OWNED for w in work[1:]))

    def test_small_split_tail_needs_scores_when_global_topk_does(self):
        ranks = [tp.reference_work([8193], [0], 8, r)[0] for r in range(8)]
        self.assertTrue(any(0 < w.count <= 512 for w in ranks))
        self.assertTrue(all(w.scored for w in ranks if w.count))
        self.assertFalse(tp.needs_scores(512, 128))

    def test_partial_query_groups_padding_and_mixed_contexts(self):
        bounds = [8193 + i for i in range(6)] + [512, 513] + [0, 0, 0]
        batches = [31] * 6 + [45] * 2 + [-1] * 3
        for world in (1, 2, 4, 8):
            self.assert_coverage(bounds, batches, world)
            for rank in range(world):
                work = tp.reference_work(bounds, batches, world, rank)
                self.assertEqual(len({w.first for w in work[:6]}), 1)

    def test_peer_cells_are_disjoint_and_cover_counts_and_acks(self):
        for rows in (1, 6, 40, 48, 64):
            for world in (1, 2, 4, 8):
                indices = {
                    tp.row_rank_at(kind, row, source, rows, world)
                    for kind in range(2)
                    for row in range(rows)
                    for source in range(world)
                }
                layout = tp.peer_layout(rows, world)
                self.assertEqual(len(indices), layout["ix_count"][1] // 8)
                self.assertEqual(layout["ix_done"][1], layout["ix_count"][1])
                self.assertEqual(min(indices), 0)
                self.assertEqual(max(indices) + 1, len(indices))
                spans = sorted(layout.values())
                self.assertTrue(all(a + n <= b for (a, n), (b, _) in pairwise(spans)))
                self.assertEqual(
                    tp.row_rank_at(1, rows - 1, world - 1, rows, world),
                    2 * rows * world - 1,
                )


if __name__ == "__main__":
    unittest.main()
