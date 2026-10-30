"""Superseded Mamba states: the scheduler marks them, the worker removes their exact keys after the delay."""

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import BoundaryPut
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import KVCacheStoreSendingThread
    import pd_c48_worker
    from pd_c48_scheduler import track_superseded_states
except ImportError:
    BoundaryPut = None


class FakeStoreScheduler:
    _boundary_state_group_ids = frozenset({1, 2})
    _hash_block_size = 128

    def __init__(self):
        self.hit, self.metas = (0, False), []

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self.hit

    def build_connector_meta(self, scheduler_output):
        return NS(requests=self.metas)


def step(store, finished=(), **puts):
    store.metas = [NS(req_id=r, boundary_puts=p) for r, p in puts.items()]
    store.build_connector_meta(NS(finished_req_ids=set(finished)))
    return {m.req_id: getattr(m, "c48_superseded", None) for m in store.metas}


@unittest.skipIf(BoundaryPut is None, "needs the vLLM image")
class TrackTest(unittest.TestCase):
    def test_resume_then_checkpoints(self):
        store = FakeStoreScheduler()
        track_superseded_states(store)
        store.hit = (1536, True)
        store.get_num_new_matched_tokens(NS(request_id="a"), 3072)
        kv, s1, s2 = BoundaryPut(0, 7, 9216), BoundaryPut(1, 8, 9216), BoundaryPut(2, 9, 9216)
        self.assertEqual(step(store, a=[kv, s1, s2])["a"], [BoundaryPut(1, 8, 4608), BoundaryPut(2, 9, 4608)])
        self.assertEqual(step(store, a=[BoundaryPut(1, 8, 9984), BoundaryPut(1, 8, 10240)])["a"],
                         [BoundaryPut(1, 8, 9216), BoundaryPut(1, 8, 9984)])
        self.assertIsNone(step(store, a=[BoundaryPut(1, 8, 10240)])["a"])

    def test_off_grid_resume_and_finish(self):
        store = FakeStoreScheduler()
        track_superseded_states(store)
        store.get_num_new_matched_tokens(NS(request_id="b"), 3000)
        self.assertEqual(step(store, finished={"b"}, b=[BoundaryPut(1, 8, 9216)])["b"], [])
        store.hit = (None, False)
        store.get_num_new_matched_tokens(NS(request_id="b"), 3072)
        self.assertEqual(step(store, b=[BoundaryPut(1, 8, 9216)])["b"], [])


class FakeStore:
    def __init__(self):
        self.removed = []

    def batch_remove(self, keys, force):
        assert force is False
        self.removed.append(keys)
        return [0, -706][:len(keys)]


@unittest.skipIf(BoundaryPut is None, "needs the vLLM image")
class DropTest(unittest.TestCase):
    def test_removes_exact_rank_keys_after_delay(self):
        pd_c48_worker.drop_superseded_states()
        db = NS(block_size=1536, metadata=None, key_for=lambda h: f"g1@{h}",
                prepare_value_for_block=lambda block_id: ([block_id], [1]))
        thread = NS(coord=NS(hash_block_size=128), group_participates=[False, True], token_databases=[None, db],
                    group_put_steps=[1, 1], tp_rank=0, store=FakeStore())
        thread._boundary_puts = lambda meta: KVCacheStoreSendingThread._boundary_puts(thread, meta)
        offload = KVCacheStoreSendingThread._maybe_offload_boundary_states
        meta = NS(req_id="a", boundary_puts=[], block_hashes=[f"h{i}" for i in range(80)],
                  c48_superseded=[BoundaryPut(0, 7, 4608), BoundaryPut(1, 8, 4608)])
        self.assertTrue(offload(thread, meta))
        self.assertEqual(meta.boundary_puts, [])
        time.sleep(2)
        self.assertEqual(thread.store.removed, [])
        thread._c48_drop[0] = (0, thread._c48_drop[0][1])
        time.sleep(2)
        self.assertEqual(thread.store.removed, [["g1@h35"]])


if __name__ == "__main__":
    unittest.main()
