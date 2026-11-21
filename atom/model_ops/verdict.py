# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A speculative verify's verdict from the head's id a row (a head that picks
the ids itself: the V4.1 mono heads), the rejection sampler's arithmetic.

Request b of a step laid out as ``B`` requests of ``rq`` rows (``rq - 1`` draft
rows then the bonus row): a draft row's verify id is the head's id of that row;
the draft it verifies, the step's input id of the next row. Its first rejected
row is a min over its rows (``n`` when none is rejected); then, as the
rejection sampler lays them out:

- output ids [B, rq]: the kept drafts (exact match: the targets), the first
  rejected row's target, -1 past it; with no rejection, the bonus row's id last;
- verdict [3, B]: accepted, rejected (``num_spec_steps - accepted``) and the
  anchor row (``bonus row - n + accepted``: the row the next step continues
  from).

``SYNTHETIC`` (forced acceptance): a draft is kept when a uniform from ``seed``
falls below its position's conditional rate, and the kept row emits the draft.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def verdict_row(
    out_ptr, verdict_ptr, cu_ptr, input_ids_ptr, target_idx_ptr, bonus_idx_ptr,
    head_ids_ptr, cond_rates_ptr, seed, b, nreq, num_spec_steps,
    SP: tl.constexpr, SYNTHETIC: tl.constexpr,
):  # fmt: skip
    """Request ``b`` of ``nreq``: its output ids and verdict (module docstring).
    -> its anchor row."""
    if b == 0:
        start = 0
    else:
        start = tl.load(cu_ptr + b - 1)
    n = tl.load(cu_ptr + b) - start
    p = tl.arange(0, SP)
    live = p < n
    vrow = tl.load(target_idx_ptr + start + p, mask=live, other=0)
    target = tl.load(head_ids_ptr + vrow, mask=live, other=0).to(tl.int32)
    draft = tl.load(input_ids_ptr + vrow + 1, mask=live, other=0).to(tl.int32)
    if SYNTHETIC:
        u = tl.rand(seed, start + p)
        ok = u < tl.load(cond_rates_ptr + p, mask=live, other=0.0)
        kept = draft  # a forced accept emits the draft
    else:
        ok = draft == target
        kept = target
    # the first rejected row, n if none
    k = tl.min(tl.where(live & (ok == 0), p, n), axis=0)
    rejected = k < n
    brow = tl.load(bonus_idx_ptr + b)
    bonus = tl.load(head_ids_ptr + brow).to(tl.int32)
    INVALID: tl.constexpr = -1
    row = tl.where(p < n, tl.where(p < k, kept, tl.where(p == k, target, INVALID)),
                   tl.where((p == n) & (rejected == 0), bonus, INVALID))  # fmt: skip
    tl.store(out_ptr + b * (num_spec_steps + 1) + p, row, mask=p < num_spec_steps + 1)
    accepted = tl.where(rejected, k, n)
    anchor = brow - n + accepted
    tl.store(verdict_ptr + b, accepted)
    tl.store(verdict_ptr + nreq + b, num_spec_steps - accepted)
    tl.store(verdict_ptr + 2 * nreq + b, anchor)
    return anchor


@triton.jit
def _verdict_kernel(
    out_ptr, verdict_ptr, cu_ptr, input_ids_ptr, target_idx_ptr, bonus_idx_ptr,
    head_ids_ptr, cond_rates_ptr, launches_ptr, seed_base, nreq, num_spec_steps,
    SP: tl.constexpr, SYNTHETIC: tl.constexpr,
):  # fmt: skip
    """Exact match: program b, request b (``verdict_row``). ``SYNTHETIC``: one
    program, every request in turn, seeded ``seed_base`` + ``launches``, which
    it then bumps: one count every width's launches share, the same on every
    rank (each replays the same launches)."""
    if SYNTHETIC:
        launch = tl.load(launches_ptr)
        for b in range(nreq):
            verdict_row(
                out_ptr, verdict_ptr, cu_ptr, input_ids_ptr, target_idx_ptr,
                bonus_idx_ptr, head_ids_ptr, cond_rates_ptr, seed_base + launch, b,
                nreq, num_spec_steps, SP=SP, SYNTHETIC=True,
            )  # fmt: skip
        tl.store(launches_ptr, launch + 1)
    else:
        verdict_row(
            out_ptr, verdict_ptr, cu_ptr, input_ids_ptr, target_idx_ptr, bonus_idx_ptr,
            head_ids_ptr, head_ids_ptr, 0, tl.program_id(0), nreq, num_spec_steps,
            SP=SP, SYNTHETIC=False,
        )  # fmt: skip


class UniformVerdict:
    """A step of ``requests`` requests of ``rq`` rows each, its verify rows'
    layout and its outputs at fixed addresses (a captured graph replays them):
    ``launch(head_ids, input_ids)`` -> (output ids [requests, rq], verdict
    [3, requests]). ``cond_rates`` (forced acceptance: the positions'
    conditional rates, ``rejection_sampler._get_synthetic_cond_rates``): the
    ``SYNTHETIC`` verdict, its uniforms seeded from ``seed_base`` and
    ``launches`` (int64 [1] on the device), the launch count the head's widths
    share."""

    def __init__(
        self, requests: int, rq: int, device, cond_rates=None, seed_base=0,
        launches=None,
    ):  # fmt: skip
        i32 = {"dtype": torch.int32, "device": device}
        s = rq - 1
        self.requests, self.rq = requests, rq
        self.cond_rates, self.seed_base, self.launches = cond_rates, seed_base, launches
        self.out = torch.zeros(requests, rq, **i32)
        self.verdict = torch.zeros(3, requests, **i32)
        self.cu = torch.arange(1, requests + 1, **i32) * s
        self.target = torch.tensor(
            [b * rq + j for b in range(requests) for j in range(s)],
            dtype=torch.long,
            device=device,
        )
        self.bonus = torch.tensor(
            [b * rq + s for b in range(requests)], dtype=torch.long, device=device
        )

    def launch(self, head_ids: torch.Tensor, input_ids: torch.Tensor):
        synthetic = self.cond_rates is not None
        _verdict_kernel[(1 if synthetic else self.requests,)](
            self.out, self.verdict, self.cu, input_ids, self.target, self.bonus,
            head_ids, self.cond_rates if synthetic else head_ids,
            self.launches if synthetic else head_ids, self.seed_base, self.requests,
            self.rq - 1, SP=triton.next_power_of_2(self.rq), SYNTHETIC=synthetic,
            num_warps=1,
        )  # fmt: skip
        return self.out, self.verdict
