# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`fused_compress_attn` kernel vs reference on ragged, mixed-length batches.

The compressor is what writes the per-request state a resumed prefill chunk
reads back, and it had no test at all. The shapes here are the ones the
checkpoint ladder actually produces and that a uniform batch never does: one
sequence cut at a rung sitting beside one that finishes in a single chunk, and
a resumed chunk (non-zero prefix) beside a fresh one.

The kernel reads `kv_in[ragged_id - (K - 1 - k)]` — an index into the batch's
CONCATENATED token stream — for the `k >= window_len` lanes. Only `window_len`
keeps that from reaching back into the previous sequence's tokens, so a ragged
batch is exactly where an off-by-one there would show up, and a uniform one
would hide it.
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "compares a Triton kernel against its reference; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.v4_kernels.compress_plan import make_compress_plans
from atom.model_ops.v4_kernels.fused_compress import (
    fused_compress_attn,
    fused_compress_attn_reference,
)

DEV = "cuda"
HEAD_DIM = 64
ROPE_HEAD_DIM = 16
RMS_EPS = 1e-6
K_PER_BLOCK = 64
MAX_SEQ = 4096


class _Buf:
    """CpuGpuBuffer stand-in: `make_compress_plans` writes numpy, reads torch."""

    def __init__(self, rows):
        self.np = np.zeros((rows, 4), dtype=np.int32)
        self._t = torch.from_numpy(self.np).to(DEV)

    def copy_to_gpu(self, n=None):
        self._t.copy_(torch.from_numpy(self.np).to(DEV))
        return self._t if n is None else self._t[:n]


def _plan(extend, context, ratio, overlap, rows=4096):
    plans = make_compress_plans(
        np.asarray(extend, dtype=np.int32),
        np.asarray(context, dtype=np.int32),
        [(ratio, overlap)],
        plan_buffers={ratio: {"compress": _Buf(rows), "write": _Buf(rows)}},
    )
    return plans[ratio]


def _inputs(extend, ratio, overlap, seed=0):
    torch.manual_seed(seed)
    total = int(sum(extend))
    bs = len(extend)
    dim_full = HEAD_DIM * (2 if overlap else 1)
    k_pool = (2 if overlap else 1) * ratio
    state_size = k_pool
    num_slots = bs + 2
    nb = 64
    return {
        "kv_in": torch.randn(total, dim_full, dtype=torch.float32, device=DEV),
        # Keep scores modest: softmax over K rows, and -inf padding rows must
        # stay the only thing that saturates it.
        "score_in": torch.randn(total, dim_full, dtype=torch.float32, device=DEV) * 0.5,
        "kv_state": torch.randn(
            num_slots, state_size, dim_full, dtype=torch.float32, device=DEV
        ),
        "score_state": torch.randn(
            num_slots, state_size, dim_full, dtype=torch.float32, device=DEV
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
        "kv_cache": torch.zeros(
            nb, K_PER_BLOCK, HEAD_DIM, dtype=torch.bfloat16, device=DEV
        ),
        "block_tables": torch.arange(bs * 8, dtype=torch.int32, device=DEV).view(bs, 8),
        "slots": torch.arange(bs, dtype=torch.int32, device=DEV),
    }


def _run(extend, context, ratio, overlap, seed=0):
    """Return (kernel cache, reference rows, plan) for one batch."""
    plan = _plan(extend, context, ratio, overlap)
    t = _inputs(extend, ratio, overlap, seed)
    common = {
        "kv_in": t["kv_in"],
        "score_in": t["score_in"],
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
    ref_rows = fused_compress_attn_reference(
        kv_state=t["kv_state"].clone(),
        score_state=t["score_state"].clone(),
        kv_cache=None,
        **common,
    )
    fused_compress_attn(
        kv_state=t["kv_state"].clone(),
        score_state=t["score_state"].clone(),
        kv_cache=t["kv_cache"],
        **common,
    )
    torch.cuda.synchronize()
    return t["kv_cache"], ref_rows, plan, t


def _kernel_rows(cache, plan, block_tables, ratio):
    """Gather the kernel's scattered rows back into plan order."""
    rows = []
    p = plan.compress_plan_gpu[: plan.num_compress].cpu().numpy()
    bt = block_tables.cpu().numpy()
    for _, batch_id, position, _ in p:
        comp = int(position) // ratio
        rows.append(cache[bt[batch_id, comp // K_PER_BLOCK], comp % K_PER_BLOCK])
    return torch.stack(rows) if rows else None


# (ratio, overlap): CSA is 4/overlap, HCA is 128/non-overlap. 128 needs prompts
# long enough to reach a boundary, so the HCA cases use the longer shapes.
CSA = (4, True)


@pytest.mark.parametrize(
    "extend,context,label",
    [
        # The measured-toxic composition: cut-at-rung beside finishes-here.
        ([512, 400], [512, 400], "cut+whole"),
        ([400, 512], [400, 512], "whole+cut"),
        # A resumed chunk (prefix 512) beside a fresh one — the composition the
        # isolation probe allows and which scores clean, kept as the contrast.
        ([88, 512], [600, 512], "resume+fresh"),
        # GSM8K-shaped raggedness.
        ([289, 512, 425, 512, 511], [289, 512, 425, 512, 511], "gsm8k-ragged"),
        # Adjacent short sequences: the tightest case for the backward reach of
        # `ragged_id - (K - 1 - k)` across a sequence boundary.
        ([4, 4, 4, 512], [4, 4, 4, 512], "tiny-then-long"),
    ],
)
def test_kernel_matches_reference_on_ragged_batches(extend, context, label):
    ratio, overlap = CSA
    cache, ref_rows, plan, t = _run(extend, context, ratio, overlap)
    assert plan.num_compress > 0, f"{label}: plan is empty, test proves nothing"
    got = _kernel_rows(cache, plan, t["block_tables"], ratio)
    assert got is not None
    # Arm the comparison: zeros match zeros, so prove both sides carry signal
    # before believing they agree.
    assert ref_rows.float().abs().mean() > 1e-2, f"{label}: reference is ~zero"
    assert got.float().abs().mean() > 1e-2, f"{label}: kernel wrote ~nothing"
    diff = (got.float() - ref_rows.float()).abs().max()
    # Measured agreement is ~3e-5 on values reaching 14, i.e. the two agree to
    # bf16 rounding. Keep the bound near that rather than at a comfortable 1e-2,
    # which would pass even if a lane read the wrong sequence's token.
    assert torch.allclose(got.float(), ref_rows.float(), atol=1e-3, rtol=1e-3), (
        f"{label}: kernel != reference on a ragged batch "
        f"(extend={extend}); max|diff|={diff}"
    )


def test_plan_is_nonempty_for_every_shape_used():
    """Guards the cases above from silently degenerating into no-ops."""
    for extend, context, label in [
        ([512, 400], [512, 400], "cut+whole"),
        ([88, 512], [600, 512], "resume+fresh"),
        ([4, 4, 4, 512], [4, 4, 4, 512], "tiny-then-long"),
    ]:
        plan = _plan(extend, context, *CSA)
        assert plan.num_compress > 0, f"{label} produced no compress rows"


# ---------------------------------------------------------------------------
# Batch-composition invariance — the property a kernel-vs-reference test
# cannot reach.
#
# The tests above pin the kernel to `fused_compress_attn_reference`. Both
# implement the same `in_row = ragged_id - (K - 1 - k)` back-reach into the
# batch's CONCATENATED token stream, so a wrong `window_len` contract would
# move both by the same amount and they would still agree. What no reference
# comparison can settle is whether a sequence's own compressed rows depend on
# WHO SITS BEFORE IT — and that is the only batch property with a direction to
# it, which is what the e2e evidence points at.
# ---------------------------------------------------------------------------

VICTIM_EXTEND = 200  # a prompt that finishes in this one chunk


def _victim_inputs(seed=7):
    """The victim sequence's own tensors — byte-identical across arms."""
    torch.manual_seed(seed)
    dim_full = HEAD_DIM * 2  # CSA overlaps
    return (
        torch.randn(VICTIM_EXTEND, dim_full, dtype=torch.float32, device=DEV),
        torch.randn(VICTIM_EXTEND, dim_full, dtype=torch.float32, device=DEV) * 0.5,
    )


def _run_with_predecessor(pred_extend, pred_context, victim_kv, victim_score):
    """Compress a batch whose LAST sequence is the victim; return its rows.

    The victim keeps the same state slot and the same cache blocks in every
    arm, so the returned rows are directly comparable. Only the predecessor —
    its length, its context (fresh vs resumed) and its token values — varies.
    """
    ratio, overlap = CSA
    dim_full = HEAD_DIM * 2
    extend = ([pred_extend] if pred_extend else []) + [VICTIM_EXTEND]
    context = ([pred_context] if pred_extend else []) + [VICTIM_EXTEND]
    plan = _plan(extend, context, ratio, overlap)
    bs = len(extend)
    v_idx = bs - 1

    torch.manual_seed(99)  # predecessor content: deliberately different data
    pred_kv = torch.randn(pred_extend, dim_full, dtype=torch.float32, device=DEV) * 3.0
    pred_score = (
        torch.randn(pred_extend, dim_full, dtype=torch.float32, device=DEV) * 3.0
    )
    kv_in = torch.cat([pred_kv, victim_kv]) if pred_extend else victim_kv.clone()
    score_in = (
        torch.cat([pred_score, victim_score]) if pred_extend else victim_score.clone()
    )

    torch.manual_seed(0)  # shared tensors: identical in every arm
    k_pool = 2 * ratio
    num_slots = 4
    kv_state = torch.randn(num_slots, k_pool, dim_full, dtype=torch.float32, device=DEV)
    score_state = (
        torch.randn(num_slots, k_pool, dim_full, dtype=torch.float32, device=DEV) * 0.5
    )
    # Victim always owns state slot 0 and cache blocks [0, 8); the predecessor
    # is pushed to slot 1 and blocks [8, 16) so neither can alias the victim.
    slots = torch.tensor(
        ([1] if pred_extend else []) + [0], dtype=torch.int32, device=DEV
    )
    bt_rows = ([list(range(8, 16))] if pred_extend else []) + [list(range(8))]
    block_tables = torch.tensor(bt_rows, dtype=torch.int32, device=DEV)
    kv_cache = torch.zeros(16, K_PER_BLOCK, HEAD_DIM, dtype=torch.bfloat16, device=DEV)

    fused_compress_attn(
        kv_in=kv_in,
        score_in=score_in,
        plan=plan,
        state_slot_mapping=slots,
        ape=torch.zeros(ratio, dim_full, dtype=torch.float32, device=DEV),
        rms_weight=torch.ones(HEAD_DIM, dtype=torch.float32, device=DEV),
        rms_eps=RMS_EPS,
        cos_cache=torch.zeros(
            MAX_SEQ, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16, device=DEV
        ),
        sin_cache=torch.zeros(
            MAX_SEQ, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16, device=DEV
        ),
        block_tables=block_tables,
        k_per_block=K_PER_BLOCK,
        overlap=overlap,
        ratio=ratio,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        kv_state=kv_state,
        score_state=score_state,
        kv_cache=kv_cache,
    )
    torch.cuda.synchronize()

    p = plan.compress_plan_gpu[: plan.num_compress].cpu().numpy()
    rows = [
        kv_cache[int(pos) // ratio // K_PER_BLOCK, int(pos) // ratio % K_PER_BLOCK]
        for _, batch_id, pos, _ in p
        if int(batch_id) == v_idx
    ]
    assert rows, "victim contributed no compress rows — the arm proves nothing"
    return torch.stack(rows)


@pytest.mark.parametrize(
    "pred_extend,pred_context,label",
    [
        # A sequence cut at a checkpoint rung, still mid-prompt.
        (512, 512, "cut-at-rung"),
        # The continuation of one — a resumed chunk, non-zero prefix.
        (88, 600, "resumed-chunk"),
        # A short mate, to vary the concatenation offset on its own.
        (37, 37, "short-mate"),
    ],
)
def test_victim_rows_do_not_depend_on_its_predecessor(pred_extend, pred_context, label):
    """Same sequence, different sequence in front of it: rows must be equal.

    Bitwise, not approximately — the victim's own inputs, state slot and
    cache blocks are identical across arms, so every lane it legitimately
    reads is identical too. Any difference means a lane read across the
    sequence boundary into the predecessor's tokens.
    """
    victim_kv, victim_score = _victim_inputs()
    alone = _run_with_predecessor(0, 0, victim_kv, victim_score)
    behind = _run_with_predecessor(pred_extend, pred_context, victim_kv, victim_score)

    assert alone.float().abs().mean() > 1e-2, "victim wrote ~nothing; test is vacuous"
    assert torch.equal(alone, behind), (
        f"{label}: the victim's compressed rows changed when a "
        f"{pred_extend}-token predecessor joined its batch; "
        f"max|diff|={(alone.float() - behind.float()).abs().max()}"
    )
