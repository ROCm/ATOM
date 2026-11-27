# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Speculative verify in ONE launch: sample every verify row, then the verdict.

What ``ModelRunner.postprocess`` does for a speculative step with plain
temperature sampling (no top-k / top-p, logprobs, grammar or channel): gather
the target and bonus rows, draw exponential noise in torch, sample them
(``split_sample``), compare with the drafts (``rejection_sample``), then
broadcast rank 0's ids over the TP group because each rank's torch RNG stream
is its own -- ~17 launches and two collectives a step.

Here every (verify row, vocabulary chunk) is one program. The noise is drawn
in the kernel from a counter-based generator (Philox, ``tl.randint4x``) keyed by
a per-call seed every rank advances alike, so every rank samples the same ids
from its bit-identical logits and no broadcast is needed. A chunk's best
candidate goes into its row's slot as a 64-bit key (score, then lowest token
id) with ``atomic_max``; the last program to finish (a device-scope counter)
reads the rows' winners, resets the slots, and runs the verdict exactly as
``rejection_greedy_sample_kernel`` / ``rejection_synthetic_sample_kernel`` do.

Greedy rows (temperature at or below ``greedy_eps``) take the plain argmax of
the logits, lowest id on ties: bitwise the production pick. A sampled row
draws from the same distribution as production (Gumbel-max over
``logit / T``), not the same draw: the noise is a different stream.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.triton_fast_launch import FastLaunch

_BLOCK = 4096
_MAX_ROWS = 256
_KEY_INIT = -(2**63)
_KEY_INIT_C = tl.constexpr(_KEY_INIT)  # what the kernel resets a row slot to
_STATE: dict = {}


@triton.jit
def _score_key(score, idx):
    """int64 key, larger for a larger score, then for a LOWER idx."""
    b = score.to(tl.int32, bitcast=True)
    o = tl.where(b < 0, ~b, b | -2147483648)  # uint32-ordered as the float
    hi = (o ^ -2147483648).to(tl.int64)  # int32-ordered, sign-extended
    lo = (2147483647 - idx).to(tl.int64)
    return (hi << 32) | lo


@triton.jit(do_not_specialize=["num_tokens", "batch_size", "num_spec_steps", "seed"])
def _spec_verify_kernel(
    logits_ptr,  # [*, V] the step's logits, unit stride along V
    logits_stride,
    temp_ptr,  # [batch_size] fp32, pre-clamped
    target_idx_ptr,  # [num_tokens] logits row of each draft-verifying row
    bonus_idx_ptr,  # [batch_size] logits row each request's bonus is sampled at
    cu_ptr,  # [batch_size] cumulative drafts
    input_ids_ptr,  # the step's input ids; draft r = input_ids[target_idx[r] + 1]
    cond_rates_ptr,  # [num_spec_steps] forced acceptance (FORCED only)
    keys_ptr,  # [_MAX_ROWS] int64 row slots, at _KEY_INIT between launches
    counter_ptr,  # [1] int32, 0 between launches
    out_ptr,  # [batch_size, num_spec_steps + 1] int32
    verdict_ptr,  # [3, batch_size] int32
    next_ptr,  # [batch_size] int32: out[b, verdict[0, b]], the token the drafter anchors on
    num_tokens,
    batch_size,
    num_spec_steps,
    V,
    seed,
    greedy_eps,
    FORCED: tl.constexpr,  # synthetic acceptance: target rows take the argmax
    NEXT_IDS: tl.constexpr,  # write next_ptr (else the caller gathers the next ids)
    GREEDY_NO_NOISE: tl.constexpr,  # greedy rows draw no noise (else every row draws it and greedy rows drop it)
    BLOCK: tl.constexpr,
    ROWS_B: tl.constexpr,  # >= num_tokens + batch_size
    BS_B: tl.constexpr,  # >= batch_size
):
    s = tl.program_id(0)
    j = tl.program_id(1)  # verify row: drafts first, then one bonus a request
    is_target = j < num_tokens
    bb = tl.arange(0, BS_B)
    cu = tl.load(cu_ptr + bb, mask=bb < batch_size, other=2147483647)
    req = tl.where(is_target, tl.sum((cu <= j).to(tl.int32), axis=0), j - num_tokens)
    row = tl.where(
        is_target,
        tl.load(target_idx_ptr + tl.minimum(j, num_tokens - 1)),
        tl.load(bonus_idx_ptr + tl.minimum(req, batch_size - 1)),
    )
    t = tl.load(temp_ptr + req).to(tl.float32)
    greedy = t <= greedy_eps
    if FORCED:
        greedy = greedy | is_target
    offs = s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_stride + offs, mask=mask, other=0.0
    ).to(tl.float32)
    if GREEDY_NO_NOISE:
        if greedy:
            # The argmax of the logits: a greedy row draws no noise (the generator
            # is counter-based, so the sampling rows' draws do not move).
            score = x
        else:
            # Gumbel-max: argmax(x / T - log(E)), E ~ Exp(1) = -log(u), u in (0, 1).
            # Four uniforms a Philox evaluation, one counter per (row, chunk, quad).
            quad = (j * tl.num_programs(0) + s) * (BLOCK // 4) + tl.arange(0, BLOCK // 4)
            r0, r1, r2, r3 = tl.randint4x(seed, quad)
            r = tl.reshape(tl.join(tl.join(r0, r1), tl.join(r2, r3)), (BLOCK,))
            u = ((r >> 8).to(tl.float32) + 0.5) * 5.9604644775390625e-08
            score = x * (1.0 / tl.maximum(t, 1e-5)) - tl.log(-tl.log(u))
    else:
        # rF13b's: every row draws the noise, greedy rows take the logits.
        quad = (j * tl.num_programs(0) + s) * (BLOCK // 4) + tl.arange(0, BLOCK // 4)
        r0, r1, r2, r3 = tl.randint4x(seed, quad)
        r = tl.reshape(tl.join(tl.join(r0, r1), tl.join(r2, r3)), (BLOCK,))
        u = ((r >> 8).to(tl.float32) + 0.5) * 5.9604644775390625e-08
        noisy = x * (1.0 / tl.maximum(t, 1e-5)) - tl.log(-tl.log(u))
        score = tl.where(greedy, x, noisy)
    score = tl.where(mask, score, float("-inf"))
    best = tl.max(_score_key(score, offs), axis=0)
    tl.atomic_max(keys_ptr + j, best, sem="relaxed")
    done = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
    if done == tl.num_programs(0) * tl.num_programs(1) - 1:
        rows = tl.arange(0, ROWS_B)
        nrows = num_tokens + batch_size
        keys = tl.atomic_xchg(keys_ptr + rows, _KEY_INIT_C, mask=rows < nrows)
        tl.atomic_xchg(counter_ptr, 0)
        tok = (2147483647 - (keys & 0xFFFFFFFF)).to(tl.int32)
        is_t = rows < num_tokens
        tli = tl.load(target_idx_ptr + rows, mask=is_t, other=0)
        draft = tl.load(input_ids_ptr + tli + 1, mask=is_t, other=-1).to(tl.int32)
        if FORCED:
            unif = tl.rand(seed ^ 0x5EED, rows)
        INVALID: tl.constexpr = -1
        for b in range(batch_size):
            if b == 0:
                start = 0
            else:
                start = tl.load(cu_ptr + b - 1)
            n = tl.load(cu_ptr + b) - start
            out_row = out_ptr + b * (num_spec_steps + 1)
            rejected = False
            last_checked = -1
            last_id = -1
            for p in range(n):
                rr = start + p
                if rejected:
                    out_id = INVALID
                else:
                    t_id = tl.sum(tl.where(rows == rr, tok, 0), axis=0)
                    d_id = tl.sum(tl.where(rows == rr, draft, 0), axis=0)
                    if FORCED:
                        uu = tl.sum(tl.where(rows == rr, unif, 0.0), axis=0)
                        if uu < tl.load(cond_rates_ptr + p):
                            out_id = d_id
                        else:
                            out_id = t_id
                            rejected = True
                    else:
                        out_id = t_id
                        if d_id != t_id:
                            rejected = True
                    last_checked += 1
                    last_id = out_id
                tl.store(out_row + p, out_id)
            # `_finish_request`
            num_accepted = last_checked
            if rejected:
                bonus = INVALID
                next_id = last_id
            else:
                bonus = tl.sum(tl.where(rows == num_tokens + b, tok, 0), axis=0)
                num_accepted += 1
                next_id = bonus
            tl.store(out_row + n, bonus)
            if NEXT_IDS:
                tl.store(next_ptr + b, next_id)
            for pos in range(n + 1, num_spec_steps + 1):
                tl.store(out_row + pos, INVALID)
            anchor = tl.load(bonus_idx_ptr + b) - n
            tl.store(verdict_ptr + b, num_accepted)
            tl.store(verdict_ptr + batch_size + b, num_spec_steps - num_accepted)
            tl.store(verdict_ptr + 2 * batch_size + b, anchor + num_accepted)


# The largest tensor is the step's logits, [<= 256, vocab] (~100 MB).
_launch = FastLaunch(_spec_verify_kernel, small_tensors=True)


def _state(device):
    st = _STATE.get(device)
    if st is None:
        st = _STATE[device] = {
            "keys": torch.full(
                (_MAX_ROWS,), _KEY_INIT, dtype=torch.int64, device=device
            ),
            "counter": torch.zeros(1, dtype=torch.int32, device=device),
            "calls": 0,
        }
    return st


def usable(logits: torch.Tensor, num_tokens: int, batch_size: int) -> bool:
    return (
        logits.is_cuda
        and logits.dim() == 2
        and logits.stride(1) == 1
        and logits.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and 0 < batch_size
        and num_tokens + batch_size <= _MAX_ROWS
    )


def spec_verify(
    logits: torch.Tensor,  # [*, V]
    temperatures: torch.Tensor,  # [batch_size], pre-clamped
    target_logits_indices: torch.Tensor,  # [num_tokens]
    bonus_logits_indices: torch.Tensor,  # [batch_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    input_ids: torch.Tensor,
    num_spec_steps: int,
    *,
    greedy_eps: float,
    cond_rates: torch.Tensor | None = None,  # forced acceptance, conditional
    seed_base: int = 0,
    next_ids: bool = True,
    greedy_no_noise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``(output_token_ids [bs, num_spec_steps + 1], verdict [3, bs])`` as
    `rejection_sample` returns them, and each request's next token
    ``output_token_ids[b, verdict[0, b]]`` (the last one it keeps), which
    the drafter anchors on (None with ``next_ids`` off). Call on every rank
    alike: the noise seed advances by one a call. ``greedy_no_noise`` off:
    greedy rows draw the noise too and drop it (rF13b's kernel)."""
    device = logits.device
    st = _state(device)
    st["calls"] += 1
    seed = (seed_base * 1000003 + st["calls"]) & 0x7FFFFFFFFFFF
    num_tokens = target_logits_indices.shape[0]
    batch_size = cu_num_draft_tokens.shape[0]
    V = logits.shape[1]
    out = torch.empty(
        (batch_size, num_spec_steps + 1), dtype=torch.int32, device=device
    )
    verdict = torch.empty((3, batch_size), dtype=torch.int32, device=device)
    next_out = torch.empty(batch_size if next_ids else 1, dtype=torch.int32, device=device)
    forced = cond_rates is not None
    nrows = num_tokens + batch_size
    grid = (triton.cdiv(V, _BLOCK), nrows)
    _launch(
        grid,
        logits,
        logits.stride(0),
        temperatures,
        target_logits_indices,
        bonus_logits_indices,
        cu_num_draft_tokens,
        input_ids,
        cond_rates if forced else temperatures,
        st["keys"],
        st["counter"],
        out,
        verdict,
        next_out,
        num_tokens,
        batch_size,
        num_spec_steps,
        V,
        seed,
        greedy_eps,
        FORCED=forced,
        NEXT_IDS=bool(next_ids),
        GREEDY_NO_NOISE=bool(greedy_no_noise),
        BLOCK=_BLOCK,
        ROWS_B=max(16, triton.next_power_of_2(nrows)),
        BS_B=max(16, triton.next_power_of_2(batch_size)),
        num_warps=8,
    )
    return out, verdict, (next_out if next_ids else None)
