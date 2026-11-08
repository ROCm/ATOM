# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The verdict of a step the V4.1 mono hi hook's verify head served, in one launch (``ATOM_DSV41_MONO_HI``).

On such a step ``ModelRunner.postprocess`` has the head's id a row (``verify_token_ids``) and would run, between the
target graph and the draft: two ``index_select`` (the verify rows' ids, the bonus rows' ids), under forced acceptance
a ``torch.rand`` of the step's uniforms and the rejection sampler's kernel (the output ids and the verdict).
``verdict`` does all of it in one Triton launch (``_head_verdict_kernel``, a program a request): it reads the head's
ids through the verify / bonus row indices itself, compares or force-accepts row by row exactly as ATOM's greedy /
synthetic kernels do, and writes the same output ids and verdict ([3, requests]: accepted, rejected, anchor row). Each
request's next id stays the runner's ``gather``, from the output ids after their TP broadcast.

Under forced acceptance its uniforms are drawn in the kernel (``tl.rand``, seeded from the sampler's step counter as
ATOM's generator is: the same on every TP rank, a different stream from ``torch.rand``'s, the same schedule in
expectation); with real acceptance there is no draw and the result is ATOM's bit for bit. The device round
(``device_round``) takes the fused verdict as it takes ATOM's (``post_verify.verdict`` reports it). Steps laid out as the
in-graph post-verify assumes take its results instead (``post_verify``); this launch serves the others.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_spec_steps", "seed"])
def _head_verdict_kernel(
    out_ptr,  # int32 [requests, num_spec_steps + 1]
    verdict_ptr,  # int32 [3, requests]: accepted drafts, rejected, anchor row
    cu_num_draft_ptr,  # [requests]
    input_ids_ptr,  # the step's input ids; draft r is input_ids[target_idx[r] + 1]
    target_idx_ptr,  # [verify rows]: each verify row's row of the step
    bonus_idx_ptr,  # [requests]: each request's bonus row
    head_ids_ptr,  # [rows of the step]: the head's id a row
    uniform_ptr,  # [verify rows] f32 (SYNTHETIC and not DRAW)
    cond_rates_ptr,  # [num_spec_steps] f32 (SYNTHETIC)
    seed,
    num_spec_steps,
    SYNTHETIC: tl.constexpr,
    DRAW: tl.constexpr,
):
    """ATOM's ``rejection_greedy_sample_kernel`` / ``rejection_synthetic_sample_kernel`` + ``_finish_request`` on the
    head's ids read through the row indices (the two ``index_select``), the uniforms drawn here (``DRAW``).
    """
    req = tl.program_id(0)
    if req == 0:
        start = 0
    else:
        start = tl.load(cu_num_draft_ptr + req - 1)
    n = tl.load(cu_num_draft_ptr + req) - start
    INVALID: tl.constexpr = -1
    row_out = out_ptr + req * (num_spec_steps + 1)
    rejected = False
    last = -1
    for pos in range(n):
        if rejected:
            oid = INVALID
        else:
            r = start + pos
            vrow = tl.load(target_idx_ptr + r)
            target = tl.load(head_ids_ptr + vrow).to(tl.int32)
            draft = tl.load(input_ids_ptr + vrow + 1).to(tl.int32)
            if SYNTHETIC:
                if DRAW:
                    u = tl.rand(seed, r)
                else:
                    u = tl.load(uniform_ptr + r)
                if u < tl.load(cond_rates_ptr + pos):
                    oid = draft
                else:
                    oid = target
                    rejected = True
            else:
                oid = target
                if draft != target:
                    rejected = True
            last += 1
        tl.store(row_out + pos, oid)
    accepted = last
    brow = tl.load(bonus_idx_ptr + req)
    if rejected:
        bonus = INVALID
    else:
        bonus = tl.load(head_ids_ptr + brow).to(tl.int32)
        accepted += 1
    tl.store(row_out + n, bonus)
    for pos in range(n + 1, num_spec_steps + 1):
        tl.store(row_out + pos, INVALID)
    nreq = tl.num_programs(0)
    tl.store(verdict_ptr + req, accepted)
    tl.store(verdict_ptr + nreq + req, num_spec_steps - accepted)
    tl.store(verdict_ptr + 2 * nreq + req, brow - n + accepted)


def _launch(spec, head_ids, sampler):
    """The fused launch: (output ids [requests, spec + 1], verdict [3, requests]); under forced acceptance the
    uniforms are drawn in the kernel, seeded from the sampler's step."""
    from atom.model_ops.rejection_sampler import (
        _SYNTHETIC_RNG_BASE_SEED,
        _get_synthetic_cond_rates,
    )

    cu = spec.cu_num_draft_tokens
    nreq, steps, dev = len(cu), spec.num_spec_steps, head_ids.device
    out = torch.empty((nreq, steps + 1), dtype=torch.int32, device=dev)
    verdict = torch.empty((3, nreq), dtype=torch.int32, device=dev)
    rates = sampler.synthetic_acceptance_rates
    synthetic = rates is not None
    cond = _get_synthetic_cond_rates(rates, dev) if synthetic else out
    _head_verdict_kernel[(nreq,)](
        out, verdict, cu, spec.input_ids, spec.target_logits_indices, spec.bonus_logits_indices, head_ids,
        out, cond, _SYNTHETIC_RNG_BASE_SEED + int(sampler._synthetic_step), steps,
        SYNTHETIC=synthetic, DRAW=True, num_warps=1,
    )  # fmt: skip
    if synthetic:
        sampler._synthetic_step += 1  # as ``RejectionSampler.forward``
    return out, verdict


def verdict(spec, head_ids, sampler):
    """A head-served verify step's (output ids, verdict) (module docstring)."""
    return _launch(spec, head_ids, sampler)
