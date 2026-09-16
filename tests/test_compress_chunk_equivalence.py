# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compressing a prompt in two chunks must equal compressing it in one.

A chunked prefill runs the compressor once per chunk, carrying `kv_state` /
`score_state` across, so a prompt cut at a rung takes a different path through
the compressor than the same prompt fed whole. Everything downstream reads the
compressed rows for the rest of the request's life -- unlike the sliding
window, they are never rebuilt by decode -- so any inequality here is a
permanent divergence, not a transient one.

Written while chunked prefill was suspected of costing accuracy. It does not
-- that turned out to be a drafter bug, and the compressor is bit-exact here
-- so this pins the boundary rather than reproducing a known failure.

The cut is placed on a multiple of `ratio` so the compression groups
themselves are identical between the two arms; what is under test is purely
whether the state handed across the boundary reproduces what the one-shot run
had in flight at that point.
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises the compressor kernel; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.v4_kernels.compress_plan import make_compress_plans
from atom.model_ops.v4_kernels.fused_compress import (
    fused_compress_attn,
    fused_compress_attn_reference,
)
from atom.model_ops.v4_kernels.state_writes import update_compressor_states

DEV = "cuda"
HEAD_DIM = 64
ROPE_HEAD_DIM = 16
RMS_EPS = 1e-6
K_PER_BLOCK = 64
MAX_SEQ = 4096
NUM_BLOCKS = 256


class _Buf:
    """CpuGpuBuffer stand-in: `make_compress_plans` writes numpy, reads torch."""

    def __init__(self, rows):
        self.np = np.zeros((rows, 4), dtype=np.int32)
        self._t = torch.from_numpy(self.np).to(DEV)

    def copy_to_gpu(self, n=None):
        self._t.copy_(torch.from_numpy(self.np).to(DEV))
        return self._t if n is None else self._t[:n]


def _plan(extend, context, ratio, overlap, rows=8192):
    plans = make_compress_plans(
        np.asarray(extend, dtype=np.int32),
        np.asarray(context, dtype=np.int32),
        [(ratio, overlap)],
        plan_buffers={ratio: {"compress": _Buf(rows), "write": _Buf(rows)}},
    )
    return plans[ratio]


def _fixture(total, ratio, overlap, seed=0):
    """One sequence's worth of compressor inputs, shared by both arms."""
    torch.manual_seed(seed)
    dim_full = HEAD_DIM * (2 if overlap else 1)
    state_size = (2 if overlap else 1) * ratio
    return {
        "kv_in": torch.randn(total, dim_full, dtype=torch.float32, device=DEV),
        "score_in": torch.randn(total, dim_full, dtype=torch.float32, device=DEV) * 0.5,
        "kv_state": torch.randn(
            3, state_size, dim_full, dtype=torch.float32, device=DEV
        ),
        "score_state": torch.randn(
            3, state_size, dim_full, dtype=torch.float32, device=DEV
        )
        * 0.5,
        "ape": torch.randn(ratio, dim_full, dtype=torch.float32, device=DEV) * 0.5,
        "rms_weight": torch.rand(HEAD_DIM, dtype=torch.float32, device=DEV) + 0.5,
        "cos_cache": torch.randn(
            MAX_SEQ, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16, device=DEV
        ),
        "sin_cache": torch.randn(
            MAX_SEQ, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16, device=DEV
        ),
        "block_tables": torch.arange(64, dtype=torch.int32, device=DEV).view(1, 64),
        "slots": torch.zeros(1, dtype=torch.int32, device=DEV),
    }


def _common(t, plan, ratio, overlap):
    return {
        "plan": plan,
        "state_slot_mapping": t["slots"],
        "ape": t["ape"],
        "rms_weight": t["rms_weight"],
        "rms_eps": RMS_EPS,
        "cos_cache": t["cos_cache"],
        "sin_cache": t["sin_cache"],
        "block_tables": t["block_tables"],
        "k_per_block": K_PER_BLOCK,
        "overlap": overlap,
        "ratio": ratio,
        "head_dim": HEAD_DIM,
        "rope_head_dim": ROPE_HEAD_DIM,
    }


def _fresh_cache():
    return torch.zeros(
        NUM_BLOCKS, K_PER_BLOCK, HEAD_DIM, dtype=torch.bfloat16, device=DEV
    )


def _run_whole(t, total, ratio, overlap):
    """Compress `total` tokens in a single call."""
    cache = _fresh_cache()
    fused_compress_attn(
        kv_in=t["kv_in"],
        score_in=t["score_in"],
        kv_state=t["kv_state"].clone(),
        score_state=t["score_state"].clone(),
        kv_cache=cache,
        **_common(t, _plan([total], [total], ratio, overlap), ratio, overlap),
    )
    torch.cuda.synchronize()
    return cache


def _chunk(t, lo, hi, context, kv_state, score_state, cache, ratio, overlap):
    """One prefill chunk: compress its tokens, then hand its state forward.

    Both halves matter. `fused_compress_attn` only READS `kv_state`; the ring
    is written by `update_compressor_states`, which is what lets the next chunk
    reach back past its own first token. Running compress alone leaves the next
    chunk reading a stale ring -- that is a bug in the test, not the engine.
    """
    plan = _plan([hi - lo], [context], ratio, overlap)
    fused_compress_attn(
        kv_in=t["kv_in"][lo:hi],
        score_in=t["score_in"][lo:hi],
        kv_state=kv_state,
        score_state=score_state,
        kv_cache=cache,
        **_common(t, plan, ratio, overlap),
    )
    update_compressor_states(
        t["kv_in"][lo:hi],
        t["score_in"][lo:hi],
        t["ape"],
        kv_state,
        score_state,
        write_plan=plan.write_plan_gpu,
        state_slot_mapping=t["slots"],
        ratio=ratio,
        overlap=overlap,
    )


def _run_chunked(t, total, cut, ratio, overlap):
    """Compress the same tokens as two chunks, carrying state across."""
    cache = _fresh_cache()
    kv_state = t["kv_state"].clone()
    score_state = t["score_state"].clone()

    # `context` is cumulative -- what the scheduler passes for a resumed chunk.
    _chunk(t, 0, cut, cut, kv_state, score_state, cache, ratio, overlap)
    _chunk(t, cut, total, total, kv_state, score_state, cache, ratio, overlap)
    torch.cuda.synchronize()
    return cache


@pytest.mark.parametrize("ratio,overlap", [(4, True), (128, False)])
def test_two_chunks_equal_one_shot(ratio, overlap):
    """The property the chunked-prefill path depends on."""
    total, cut = 1024, 512
    assert cut % ratio == 0 and total % ratio == 0, "cut must keep groups aligned"
    t = _fixture(total, ratio, overlap)

    whole = _run_whole(t, total, ratio, overlap)
    chunked = _run_chunked(t, total, cut, ratio, overlap)

    assert whole.abs().sum().item() > 0, "one-shot arm wrote nothing; test is vacuous"
    diff = (whole.float() - chunked.float()).abs()
    n_bad = int((diff > 0).sum().item())
    assert n_bad == 0, (
        f"ratio={ratio}: chunked compression differs from one-shot in "
        f"{n_bad} of {diff.numel()} slots, max|diff|={diff.max().item():.6g}"
    )


@pytest.mark.parametrize("cut", [512, 768, 896])
def test_cut_position_does_not_change_the_result(cut):
    """Where the prompt is cut must not change what gets compressed."""
    ratio, overlap, total = 4, True, 1024
    t = _fixture(total, ratio, overlap)
    whole = _run_whole(t, total, ratio, overlap)
    chunked = _run_chunked(t, total, cut, ratio, overlap)
    diff = (whole.float() - chunked.float()).abs()
    assert (
        int((diff > 0).sum().item()) == 0
    ), f"cut={cut}: differs from one-shot, max|diff|={diff.max().item():.6g}"


def test_reference_agrees_with_itself_across_the_cut():
    """Same property against the pure-torch reference.

    Separating the two answers a question the kernel comparison cannot: if the
    reference also disagrees, the chunked path is not algorithmically
    equivalent (a design issue); if only the kernel does, it is an
    implementation bug.
    """
    ratio, overlap, total, cut = 4, True, 1024, 512
    t = _fixture(total, ratio, overlap)

    whole_rows = fused_compress_attn_reference(
        kv_in=t["kv_in"],
        score_in=t["score_in"],
        kv_state=t["kv_state"].clone(),
        score_state=t["score_state"].clone(),
        kv_cache=None,
        **_common(t, _plan([total], [total], ratio, overlap), ratio, overlap),
    )

    kv_state = t["kv_state"].clone()
    score_state = t["score_state"].clone()
    head_plan = _plan([cut], [cut], ratio, overlap)
    fused_compress_attn_reference(
        kv_in=t["kv_in"][:cut],
        score_in=t["score_in"][:cut],
        kv_state=kv_state,
        score_state=score_state,
        kv_cache=None,
        **_common(t, head_plan, ratio, overlap),
    )
    # Hand the ring forward, exactly as the engine does between chunks. The
    # ring writer has no pure-torch twin, so the kernel stands in for it here;
    # what is under test is the compressor's algebra across the boundary, not
    # the writer.
    update_compressor_states(
        t["kv_in"][:cut],
        t["score_in"][:cut],
        t["ape"],
        kv_state,
        score_state,
        write_plan=head_plan.write_plan_gpu,
        state_slot_mapping=t["slots"],
        ratio=ratio,
        overlap=overlap,
    )
    torch.cuda.synchronize()
    tail_rows = fused_compress_attn_reference(
        kv_in=t["kv_in"][cut:],
        score_in=t["score_in"][cut:],
        kv_state=kv_state,
        score_state=score_state,
        kv_cache=None,
        **_common(t, _plan([total - cut], [total], ratio, overlap), ratio, overlap),
    )

    # The resumed chunk emits the rows for tokens [cut, total); compare them
    # against the tail of the one-shot run.
    n_tail = tail_rows.shape[0]
    assert n_tail > 0, "resumed chunk emitted no rows; test is vacuous"
    ref_tail = whole_rows[-n_tail:]
    diff = (ref_tail.float() - tail_rows.float()).abs()
    assert int((diff > 0).sum().item()) == 0, (
        f"reference disagrees across the cut in {int((diff > 0).sum().item())} "
        f"slots, max|diff|={diff.max().item():.6g}"
    )
