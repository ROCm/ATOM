# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A served verify step's verdict and Engram cursor commit inside the target graph (the V4.1 mono hi hook,
``ATOM_DSV41_MONO_HI``): the forward's head output.

After the target graph a verify step runs, on ATOM's path, the verdict (the rejection sampler on the logits' rows)
and Engram's cursor commit (``TentativeState.commit``: ``_commit_cursors``). On a served step both read only device
state: the head's id a row (the hook's verify head, greedy or one draw a sampled row) and the tentative staging plane.
The hook launches them as one kernel (``_post_verify_kernel``, a program a request, each request's rows at once) at the
end of its forward (``HiLayers.end_step``), so a captured graph replays it:

- the verdict on the head's ids (ATOM's greedy / synthetic rejection kernels' arithmetic), the width's verify / bonus
  rows and draft counts laid out as the served step lays them (B requests of ``rq`` rows: ``rq - 1`` drafts and a bonus
  row each), into persistent outputs a width (the output ids [B, rq] and the verdict [3, B], as the rejection sampler
  lays them out); under forced acceptance the uniforms are seeded from the hook's launch-tag step (bumped by every
  forward: the same on every TP rank);
- the commit: each live request's accepted prefix's candidate cursor (ATOM's tentative staging plane, written on
  the device on served steps) into its slot's Engram cursor (a padding request of a bs-B graph commits nothing: its
  slot entry is padding, which names some other slot).

The outputs are the forward's head output (``HiDecodeModel.compute_head`` -> ``ModelRunner._head_output``, recorded
with the forward). The runner takes them for its step's requests (the first of the width's) where they stand for it
(``ModelRunner._head_ids_serve``: every request verifying the full span, the rows' picks all it reads), else ATOM's
path runs on the logits. The commit (``commit_speculative_state``) then skips its launch exactly when the anchors it
commits are these outputs' (``PostVerify.committed``): the runner took them, and the kernel committed them; on any
other step ATOM's commit rewrites the cursor, so the in-graph write it never asked for is overwritten. The ids that
feed the next step are not staged here: the runner broadcasts the output ids over TP and gathers each request's next
id from them, and the DSpark proposer stages its anchors from those, as on ATOM's logits path. The outputs are a
width's fixed addresses, read after the step by the runner's side-stream copies (the sampled ids, the verdict counts)
with no reverse wait, as ``send_draft_token_ids_to_cpu`` reads the draft graph's: the replay that rewrites them is the
next target forward's end, a draft pass and a whole forward later on the default stream.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from atom.models.deepseek_v41.mono.hi_bridge import REQ_LIVE

logger = logging.getLogger("atom")


@triton.jit(do_not_specialize=["num_spec_steps", "seed_base"])
def _post_verify_kernel(
    out_ptr, verdict_ptr, cu_ptr, input_ids_ptr, target_idx_ptr, bonus_idx_ptr, head_ids_ptr, cond_rates_ptr,
    step_ptr, used_seed_ptr, cursor_ptr, cursor_stride, staging_ptr, staging_req_stride, staging_row_stride, starts_ptr,
    slots_ptr, live_ptr, num_spec_steps, seed_base,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr, SP: tl.constexpr, SYNTHETIC: tl.constexpr, SHADOW: tl.constexpr,
):  # fmt: skip
    """Program b, request b: the verdict (ATOM's rejection kernels' arithmetic, its rows at once: the first rejected
    position is a min over the rows) and the commit (request b's accepted prefix's candidate cursor,
    ``staging[b, anchor - start]``, into ``cursor[slot]`` when ``live_ptr[b]`` (the hook's request word: not a padding
    request); ``SHADOW``: into ``cursor[b]``, the check's rows). The uniforms' seed is ``seed_base`` + the hook's
    launch-tag step (bumped by every forward's begin-step: the same on every TP rank), recorded in
    ``used_seed_ptr``."""
    b = tl.program_id(0)
    seed = seed_base + tl.load(step_ptr).to(tl.int64)
    if b == 0:
        tl.store(used_seed_ptr, seed)
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
    nreq = tl.num_programs(0)
    tl.store(verdict_ptr + b, accepted)
    tl.store(verdict_ptr + nreq + b, num_spec_steps - accepted)
    tl.store(verdict_ptr + 2 * nreq + b, anchor)
    cols = tl.arange(0, BLOCK)
    keep = cols < WIDTH
    prefix = anchor - tl.load(starts_ptr + b)
    crow = tl.load(
        staging_ptr + b * staging_req_stride + prefix * staging_row_stride + cols,
        mask=keep,
    )
    if SHADOW:
        dst = b.to(tl.int64)
    else:
        dst = tl.load(slots_ptr + b).to(tl.int64)
    if (
        tl.load(live_ptr + b) != 0
    ):  # (a padding request's slot entry names some other request's slot, or a free one)
        tl.store(cursor_ptr + dst * cursor_stride + cols, crow, mask=keep)


class PostVerify:
    """The hook's in-graph post-verify (module docstring): a width's persistent outputs and constant row layout, its
    launch from the hook's forward, its outputs as the forward's head output and the commit's skip.
    """

    def __init__(self, widths, split, device):
        from atom.model_ops.rejection_sampler import _SYNTHETIC_RNG_BASE_SEED

        self.split = split
        self.seed_base = _SYNTHETIC_RNG_BASE_SEED
        self.used_seed = torch.zeros(1, dtype=torch.int64, device=device)
        self.bufs, self.launched = {}, set()
        i32 = {"dtype": torch.int32, "device": device}
        for rows in widths:
            nb, rq = split[rows]
            s = rq - 1
            self.bufs[rows] = {
                "out": torch.zeros(nb, rq, **i32), "verdict": torch.zeros(3, nb, **i32),
                "cu": torch.arange(1, nb + 1, **i32) * s,
                "target": torch.tensor([b * rq + j for b in range(nb) for j in range(s)], dtype=torch.long,
                                       device=device),
                "bonus": torch.tensor([b * rq + s for b in range(nb)], dtype=torch.long, device=device),
            }  # fmt: skip

    def launch(self, hook, cache, step, rows: int) -> None:
        """At the end of a served forward (``HiLayers.end_step``; a capture records it): the kernel on this
        width's fixed buffers. Every input is a fixed address of this width: the head's ids, the step's input ids,
        query starts and slots, ATOM's cursor and tentative staging plane."""
        from atom.model_ops.rejection_sampler import _get_synthetic_cond_rates

        nb, rq = self.split[rows]
        bufs = self.bufs[rows]
        sampler = (
            getattr(hook.runner, "rejection_sampler", None)
            if hook.runner is not None
            else None
        )
        if sampler is None or hook.step_ids is None:
            raise RuntimeError(
                "V4.1 mono hi post-verify: needs the runner's rejection sampler and the step's input ids"
            )
        rates = sampler.synthetic_acceptance_rates
        cond = (
            _get_synthetic_cond_rates(rates, self.used_seed.device)
            if rates is not None
            else bufs["out"]
        )
        staging, cursor = cache.tentative_staging.gpu, cache.cursor
        self.launched.add(rows)
        _post_verify_kernel[(nb,)](
            bufs["out"], bufs["verdict"], bufs["cu"], hook.step_ids, bufs["target"], bufs["bonus"], hook.head_ids,
            cond, hook.step, self.used_seed, cursor, cursor.stride(0), staging, staging.stride(0), staging.stride(1),
            step.cu_seqlens_q, step.slots, hook.req[REQ_LIVE:], rq - 1, self.seed_base, WIDTH=cache.cursor.shape[1],
            BLOCK=triton.next_power_of_2(cache.cursor.shape[1]), SP=triton.next_power_of_2(rq),
            SYNTHETIC=rates is not None, SHADOW=False, num_warps=1,
        )  # fmt: skip

    def output(self, rows: int):
        """Width ``rows``' outputs, the forward's head output (``HiLayers.head_output``): (output ids [B, rq], verdict
        [3, B])."""
        b = self.bufs[rows]
        return b["out"], b["verdict"]

    def committed(self, metadata, anchors) -> bool:
        """The builder's commit (``HiRound.committed``) at ``anchors``: a step whose anchors are a width's in-graph
        verdict's (the runner took the forward's head output) committed in the graph (``_post_verify_kernel``); here
        only the bookkeeping ``cache.commit_tentative`` does. -> committed.
        """
        cache = metadata.cache
        pending = getattr(cache, "pending", None)
        if anchors is None or pending is None or not pending.staged_on_device:
            return False
        ptr = anchors.data_ptr()
        if not any(
            ptr == self.bufs[rows]["verdict"][2].data_ptr() for rows in self.launched
        ):
            return False
        if pending.step is not metadata.step:
            raise RuntimeError("Tentative state was not prepared for this step")
        cache.pending = None
        return True
