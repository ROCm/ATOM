# SPDX-License-Identifier: MIT
"""Foundation tests for the AF segmented cudagraph (the AF_PIECEWISE capture).

Validates the two invariants the feature relies on:
  1. A segment's output is read ZERO-COPY by a later segment across replays when
     both are captured in ONE session sharing ONE pinned pool (the coordinated
     liveness the old per-piece independent captures lacked).
  2. Structurally-identical segments across captures (a dense piece at the same
     shape across ragged buckets) share ONE hipGraphExec via the dedup registry.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AF segmented cudagraph needs a GPU"
)


def _load():
    try:
        from atom.utils.attn_ffn_segmented_cudagraph import (
            SegmentedCudaGraph,
            SegmentedCudaGraphCapture,
        )
        from atom.utils.hip_graph_dedup import HipGraphDedupRegistry
    except ImportError as e:  # off-GPU CI may lack atom.utils deps
        pytest.skip(f"cannot import AF segmented cudagraph: {e}")
    return SegmentedCudaGraph, SegmentedCudaGraphCapture, HipGraphDedupRegistry


def test_two_buckets_share_dense_execs_and_are_zero_copy_correct():
    SegmentedCudaGraph, SegmentedCudaGraphCapture, HipGraphDedupRegistry = _load()
    dev = "cuda"
    D = 16
    pool = torch.cuda.graph_pool_handle()  # ONE shared pinned pool
    cap = torch.cuda.Stream()
    dedup = HipGraphDedupRegistry()

    def build(add_const):
        # a "bucket": dense0 -> attn(bucket-specific) -> dense1, all CAPTURED
        # segments. dense segs are identical structure across buckets (dedup);
        # attn differs. Each segment reads the previous one's output zero-copy.
        src = torch.zeros(D, device=dev)
        result = torch.empty(D, device=dev)

        def dense0():
            return src * 2.0 + 1.0

        def attn(q):
            return q + add_const

        def dense1(a):
            result.copy_(a * 3.0)
            return result

        def forward(sess=None):
            if sess is None:
                return dense1(attn(dense0()))
            q = sess.run_segment(("dense0", D), dense0)
            a = sess.run_segment(("attn", add_const), attn, q)
            return sess.run_segment(("dense1", D), dense1, a)

        with torch.cuda.stream(cap):
            for _ in range(2):
                src.fill_(0.0)
                forward()
        torch.cuda.synchronize()
        g = SegmentedCudaGraph()
        with SegmentedCudaGraphCapture(
            cuda_graph=g, pool=pool, stream=cap, dedup=dedup
        ) as sess:
            forward(sess)
        torch.cuda.synchronize()
        return g, src, result

    gA, sA, rA = build(10.0)
    gB, sB, rB = build(20.0)
    # dense0 + dense1 shared by A&B (2 groups) + one attn group each = 4 groups
    assert dedup.stats()[0] == 4

    for step in range(1, 4):
        sA.fill_(float(step))
        gA.replay()
        torch.cuda.synchronize()
        refA = ((float(step) * 2 + 1) + 10) * 3
        assert torch.allclose(rA, torch.full_like(rA, refA)), (rA[0].item(), refA)
        sB.fill_(float(step))
        gB.replay()
        torch.cuda.synchronize()
        refB = ((float(step) * 2 + 1) + 20) * 3
        assert torch.allclose(rB, torch.full_like(rB, refB)), (rB[0].item(), refB)
    dedup.close()
