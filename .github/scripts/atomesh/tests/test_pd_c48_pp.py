"""Check CPP early release, stage depth and RPC ring widening against the image's real EngineCore and MessageQueue."""

import importlib.util
import sys
import unittest
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
try:
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.v1.engine.core import EngineCore
    import pd_c48_pp
except ImportError:
    pd_c48_pp = None


def done(value):
    f = Future()
    f.set_result(value)
    return f


def run_engine(samples, total):
    """Step the patched step_with_batch_queue; futures resolve at once, so the engine drains only when the queue rules
    block it. Returns (max in flight, max sampling in flight, update order)."""
    script, order, peak = list(enumerate(samples)), [], [0, 0]

    def schedule(_throttle):
        i, flag = script.pop(0)
        return NS(step=i, c48_samples=flag, total_num_scheduled_tokens=1, pending_structured_output_tokens=False)

    def execute_model(so, non_block):
        peak[0] = max(peak[0], len(engine.batch_queue) + 1)
        peak[1] = max(peak[1], sum(s.c48_samples for _, s, _ in engine.batch_queue) + so.c48_samples)
        return done(None)

    engine = object.__new__(EngineCore)
    engine.__dict__.update(
        batch_queue=deque(maxlen=total), batch_queue_size=total, is_mm_encoder_only=False, is_pooling_model=False,
        check_for_draft_tokens=False, _should_throttle_prefills=lambda: False,
        log_error_detail=lambda so: nullcontext(), capture_iteration_details=lambda so: nullcontext(),
        _process_aborts_queue=lambda: None, _attach_iteration_details=lambda *a: None,
        _attach_weight_version=lambda *a: None,
        model_executor=NS(execute_model=execute_model, sample_tokens=lambda g, non_block: done("out")),
        scheduler=NS(has_requests=lambda: bool(script), schedule=schedule, get_grammar_bitmask=lambda so: None,
                     update_from_output=lambda so, out: order.append(so.step)))
    while script or engine.batch_queue:
        engine.step_with_batch_queue()
    return peak[0], peak[1], order


@unittest.skipIf(pd_c48_pp is None, "needs the vLLM image")
class EarlyReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = EngineCore.step_with_batch_queue
        pd_c48_pp.early_release(3)

    @classmethod
    def tearDownClass(cls):
        EngineCore.step_with_batch_queue = cls.original

    def test_bounds_and_fifo_order(self):
        for samples in ([False] * 20, [True] * 20, [False] * 6 + [True] * 6 + [False, True] * 10):
            in_flight, sampling, order = run_engine(samples, total=8)
            self.assertLessEqual(in_flight, 8)
            self.assertLessEqual(sampling, 3)
            self.assertEqual(order, list(range(len(samples))))

    def test_middle_chunks_fill_total_and_sampling_keeps_old_slots(self):
        self.assertEqual(run_engine([False] * 20, 8)[0], 8)
        self.assertEqual(run_engine([True] * 20, 8)[:2], (3, 3))

    def test_mark_sampling(self):
        sched = NS(requests={"a": NS(is_prefill_chunk=True), "b": NS(is_prefill_chunk=False)})
        outs = iter([{"a": 6400}, {"a": 512, "b": 2000}, {}])
        sched.schedule = lambda: NS(num_scheduled_tokens=next(outs))
        pd_c48_pp.mark_sampling(sched)
        # Empty steps keep their slot.
        self.assertEqual([sched.schedule().c48_samples for _ in range(3)], [False, True, True])

    def test_cap_needs_larger_total(self):
        with self.assertRaisesRegex(RuntimeError, "C48_P_PP_INFLIGHT"):
            pd_c48_pp.enable_early_release(NS(), 9)


@unittest.skipIf(pd_c48_pp is None, "needs the vLLM image")
class StageDepthAndRingTest(unittest.TestCase):
    def test_wait_stage_depth(self):
        synced = []
        events = deque(NS(synchronize=lambda i=i: synced.append(i)) for i in range(4))
        pd_c48_pp.wait_stage_depth(events, 2)  # step N may start once step N-2 is done
        self.assertEqual((synced, len(events)), ([0, 1, 2], 1))

    def test_ring_widened_only_inside_init_executor(self):
        spec = importlib.util.spec_from_file_location("c48_pp_ring", HERE / "pp_ring" / "sitecustomize.py")
        site = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(site)
        executor = NS(MessageQueue=MessageQueue)
        executor.MultiprocExecutor = type("E", (), {"_init_executor": lambda self: setattr(
            self, "rpc_broadcast_mq", executor.MessageQueue(1, 1))})
        site.widen_rpc_ring(executor, 64)
        e = executor.MultiprocExecutor()
        e._init_executor()
        self.assertEqual(e.rpc_broadcast_mq.buffer.max_chunks, 64)
        self.assertIs(executor.MessageQueue, MessageQueue)
        self.assertEqual(MessageQueue(1, 1).buffer.max_chunks, 10)


if __name__ == "__main__":
    unittest.main()
