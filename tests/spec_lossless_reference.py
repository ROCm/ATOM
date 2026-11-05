# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Oracles and reference helpers for the lossless speculative-sampling tests.

Kept out of `atom/model_ops/spec_sampling/dspark_markov_sample.py` and
`atom/model_ops/spec_sampling/sparse_q_select.py` so the shipped modules hold only what
serving runs. The level grid (bins, levels per nat, the 19-nat floor) is
imported from the kernel module rather than restated, so a change there
reaches the oracle too.

Not a test file (no ``test_`` prefix): it is imported, not collected. It
imports the kernel module, so import it only below a test file's AITER / GPU
guards.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.rejection_sampler import rejection_sample
from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.dspark_markov_sample import (
    _sample_scratch,
    dspark_markov_sample,
    dspark_markov_sample_scratch,
)
from atom.model_ops.spec_sampling.ratio_target_p import (
    _RATIO_BLOCK_V,
    _ratio_stats_kernel,
    _ratio_target_p,
    target_p_scratch,
)
from atom.model_ops.spec_sampling.ratio_verify import (
    _RATIO_ACCEPT_KEY,
    _RATIO_RESIDUAL_KEY,
    RatioVerifyInputs,
    _ratio_residual_pick_kernel,
    _rejecting_row_of,
    _target_p_tile,
    ratio_verify_scratch,
)
from atom.model_ops.spec_sampling.sampling_scratch import carve, scratch_buffer
from atom.model_ops.spec_sampling.softmax_tiles import race_key, softmax_combine_kernel
from atom.model_ops.spec_sampling.sparse_q_select import (
    _Q_BINS,
    _Q_LEVELS_PER_NAT,
    _Q_MAX_GAP,
)
from atom.model_ops.spec_sampling.sparse_q_store import SparseQStore
from atom.model_ops.tile_argmax import argmax_lowest_id
from atom.model_ops.verify_kernels import _draft_token, _finish_request


def sparse_q_reference(
    s: torch.Tensor,
    store_width: int,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
) -> torch.Tensor:
    """The q the sampled op stores for scores ``s`` [B, V] fp32, densified:
    ``[B, V]``, 0 off the stored set. For tests.

    Selects by (level, id) with a stable sort rather than through per-tile
    quotas, so it checks the kernels' bookkeeping rather than repeating it.
    The normalization sums in another order, so probabilities agree with the
    kernels' to rounding, the set exactly (the top-p cut sums level masses in
    another order too, so at a top-p it may differ by a level on a near tie).
    A row with nothing to store gets the delta on its lowest-id max.
    """
    rows, vocab = s.shape
    row_max = s.max(dim=-1, keepdim=True).values
    gap = row_max - s
    level = (1 + torch.floor(gap * _Q_LEVELS_PER_NAT)).to(torch.int64)
    level = torch.where(gap == 0, 0, level)
    sink = _Q_BINS - 1
    level = torch.where((gap >= 0) & (gap <= _Q_MAX_GAP), level, sink)
    counts = torch.zeros(rows, _Q_BINS, dtype=torch.int64, device=s.device)
    counts.scatter_add_(1, level, torch.ones_like(level))
    counts[:, sink] = 0
    limit = torch.full((rows,), store_width, dtype=torch.int64, device=s.device)
    if top_k is not None:
        limit = torch.where(top_k > 0, torch.minimum(top_k.long(), limit), limit)
    n = torch.minimum(limit, counts.sum(-1))
    if top_p is not None:
        bins = torch.arange(_Q_BINS, device=s.device, dtype=torch.float32)
        mid_gap = torch.where(bins == 0, 0.0, (bins - 0.5) / _Q_LEVELS_PER_NAT)
        mass = counts.float() * torch.exp(-mid_gap)
        before = mass.cumsum(-1) - mass
        inside = before < top_p[:, None].float() * mass.sum(-1, keepdim=True)
        nucleus = torch.where(inside, counts, 0).sum(-1)
        n = torch.where(top_p < 1.0, torch.minimum(n, nucleus), n)
    order = torch.sort(level, dim=-1, stable=True).indices
    chosen = torch.arange(vocab, device=s.device)[None, :] < n[:, None]
    weights = torch.zeros_like(s)
    weights.scatter_(1, order, torch.where(chosen, 1.0, 0.0))
    weights = torch.where(weights > 0, torch.exp(s - row_max), 0.0)
    q = weights / weights.sum(-1, keepdim=True)
    delta = torch.zeros_like(s).scatter_(1, s.argmax(-1, keepdim=True), 1.0)
    return torch.where((n > 0)[:, None], q, delta)


def dspark_markov_sample_reference(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
    store_width: int,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The eager spelling of the op, for tests: ``(ids [B], q [B, V])`` with q
    densified (`sparse_q_reference`), a greedy row's the delta on its argmax.

    Takes its ``[B, V]`` Exp(1) noise as an argument rather than drawing it,
    so a test can feed it the same draws the kernel would make, or none.
    """
    markov_embed = markov_w1[prev_ids]
    bias = torch.matmul(markov_embed.float(), markov_w2.float().t())
    logits = base_logits.float() + bias
    q = sparse_q_reference(
        logits / temperatures[:, None].float(), store_width, top_k, top_p
    )
    greedy = greedy_mask.bool()
    argmax = logits.argmax(dim=-1, keepdim=True)
    q = torch.where(greedy[:, None], torch.zeros_like(q).scatter_(1, argmax, 1.0), q)
    noise = torch.where(greedy[:, None], 1.0, exp_noise)
    return (q / noise).argmax(dim=-1), q


DEV = "cuda"

ANCHOR = 0


def tables(rows, vocab, rank, seed=1201):
    torch.manual_seed(seed)
    kw = {"dtype": torch.bfloat16, "device": DEV}
    return (
        torch.randn(rows, vocab, **kw),
        torch.randint(0, vocab, (rows,), dtype=torch.int64, device=DEV),
        torch.randn(vocab, rank, **kw) * 0.1,
        torch.randn(vocab, rank, **kw) * 0.1,
    )


def draft_args(rows, temperature=1.0, greedy=False, step=0):
    return (
        torch.full((rows,), temperature, device=DEV),
        torch.full((rows,), int(greedy), dtype=torch.int32, device=DEV),
        torch.tensor([step], dtype=torch.int32, device=DEV),
    )


def filters(rows, top_k=0, top_p=1.0):
    return (
        torch.full((rows,), top_k, dtype=torch.int32, device=DEV),
        torch.full((rows,), top_p, device=DEV),
    )


def empty_store(rows, width):
    """A store every slot of which reads as "never written"."""
    store = SparseQStore.allocate(rows, width, DEV)
    for t in store:
        t.fill_(-7)
    return store


def greedy_at_eps(temps, greedy):
    """The temperatures `ModelRunner.prepare_sample` publishes: a greedy
    row's clamped to ``SAMPLER_EPS``."""
    return torch.where(greedy != 0, torch.full_like(temps, SAMPLER_EPS), temps)


def sample(
    base,
    prev,
    w1,
    w2,
    store,
    row_base,
    position,
    temps,
    greedy,
    seed,
    top_k=None,
    top_p=None,
    noise_offset=0,
    scratch=None,
):
    """One `dspark_markov_sample` call; ``(out, s)``, ``s`` a copy of the
    scores the call carved from its scratch, NaN on rows it did not write. Its
    scratch is sized to ``base``'s rows unless one is given. A ``greedy`` row
    drafts at ``SAMPLER_EPS``, the clamped temperature the op reads as
    greedy."""
    rows, vocab = base.shape
    temps = greedy_at_eps(temps, greedy)
    if scratch is None:
        scratch = dspark_markov_sample_scratch(rows, vocab, DEV)
    if top_k is None:
        top_k, _ = filters(rows)
    if top_p is None:
        _, top_p = filters(rows)
    out = torch.empty(rows, dtype=torch.int64, device=DEV)
    s = carve(scratch, _sample_scratch(rows, vocab)).s
    s.fill_(float("nan"))
    dspark_markov_sample(
        base,
        prev,
        w1,
        w2,
        out,
        *store,
        row_base,
        position,
        temps,
        top_k,
        top_p,
        seed,
        noise_offset,
        scratch,
    )
    return out, s.clone()


def stored(store, row):
    """``(ids, probs, xslot)`` of one store row, as written."""
    n, xslot = store.meta[row].tolist()
    return store.ids[row, :n], store.probs[row, :n], xslot


def assert_store_row_is_q(store, row, s_row, out_id, width, top_k=None, top_p=None):
    """The store contract for one sampled row whose scores were ``s_row``."""
    ids, probs, xslot = stored(store, row)
    n = ids.numel()
    assert 1 <= n <= width
    # Strictly ascending, as the `SparseQStore` contract says (the residual
    # race binary-searches each row), and in range.
    assert (ids[1:] > ids[:-1]).all(), "stored ids are not strictly ascending"
    assert ids[0] >= 0 and ids[-1] < s_row.numel()
    assert ids[xslot].item() == out_id, "xslot does not name the emitted id"
    assert (probs > 0).all()
    assert abs(probs.double().sum().item() - 1.0) < 1e-5
    gap = s_row.max() - s_row[ids.long()]
    assert (gap <= 19.0).all(), "an id more than 19 nats down was stored"
    # Exactly the set the (level, id) oracle picks, at its values.
    ref = sparse_q_reference(
        s_row[None],
        width,
        None if top_k is None else top_k[:1],
        None if top_p is None else top_p[:1],
    )[0]
    assert torch.equal(ids.long(), ref.nonzero().view(-1))
    torch.testing.assert_close(probs, ref[ids.long()], rtol=1e-5, atol=1e-7)


def verify_batch(drafts, width, logits, temperatures, row_base=None, scratch=None):
    """`verify` arguments for equal-width requests whose row r
    scores ``logits[r]`` and drafts ``drafts[r]``, with a ratio scratch sized
    to the batch unless one is given."""
    rows = drafts.numel()
    batch = rows // width
    ids, logit_rows = [], []
    for b in range(batch):
        start = len(ids)
        ids.append(ANCHOR)
        ids.extend(drafts[b * width : (b + 1) * width].tolist())
        logit_rows.extend(range(start, start + width))
    return {
        "input_ids": torch.tensor(ids, device=DEV, dtype=torch.int32),
        "target_logits_indices": torch.tensor(logit_rows, device=DEV),
        "num_spec_steps": width,
        "cu_num_draft_tokens": torch.arange(
            width, rows + 1, width, device=DEV, dtype=torch.int32
        ),
        "target_logits": logits.contiguous(),
        "bonus_token_ids": torch.zeros(batch, device=DEV, dtype=torch.int32),
        "bonus_logits_indices": torch.arange(
            width, (width + 1) * batch, width + 1, device=DEV
        ),
        "temperatures": temperatures,
        # Flat ragged q unless told otherwise: request b's rows start at b * width.
        "q_row_base": (
            torch.arange(batch, device=DEV, dtype=torch.int32) * width
            if row_base is None
            else row_base
        ),
        "ratio_scratch": (
            ratio_verify_scratch(batch, width, logits.shape[1], DEV)
            if scratch is None
            else scratch
        ),
    }


def verify(
    q,
    *,
    temperatures,
    q_row_base,
    ratio_scratch,
    step=0,
    top_ks=None,
    top_ps=None,
    **args,
):
    """`rejection_sample` on `verify_batch` arguments: the ratio verify of
    ``q`` at Philox ``step`` under these filters, or exact match where ``q``
    is None."""
    ratio = None
    if q is not None:
        ratio = RatioVerifyInputs(
            q,
            q_row_base,
            step,
            temperatures,
            top_ks,
            top_ps,
            ratio_scratch,
        )
    return rejection_sample(**args, ratio=ratio)


def sparse(vocab, support):
    """A distribution over ``vocab`` ids with ``support = {id: mass}``."""
    p = torch.zeros(vocab, device=DEV)
    for i, m in support.items():
        p[i] = m
    return p


def draw(dist, n):
    return torch.multinomial(dist, n, replacement=True).to(torch.int32)


def draft_q(q, drafts=None):
    """The `SparseQStore` of dense q rows: each row's support, ascending, at
    its q, with xslot naming ``drafts[r]`` (slot 0 where that is not stored,
    or with no drafts). A 1-D ``q`` is one row, or every draft's row when
    ``drafts`` is given."""
    rows = q.view(-1, q.shape[-1])
    vocab = rows.shape[1]
    present = rows > 0
    order = torch.where(present, torch.arange(vocab, device=DEV), vocab).sort(1)[0]
    n = present.sum(1)
    ids = order[:, : max(int(n.max()), 1)]
    kept = ids < vocab
    probs = torch.where(kept, rows.gather(1, ids.clamp(max=vocab - 1)), 0.0)
    ids = torch.where(kept, ids, -1)
    if q.ndim == 1 and drafts is not None:
        count = drafts.numel()
        ids, probs, n = ids.expand(count, -1), probs.expand(count, -1), n.expand(count)
    xslot = torch.zeros_like(n)
    if drafts is not None:
        hit = ids[: drafts.numel()] == drafts.view(-1, 1)
        xslot[: drafts.numel()] = hit.int().argmax(1)
    return SparseQStore(
        ids.to(torch.int32).contiguous(),
        probs.float().contiguous(),
        torch.stack((n, xslot), 1).to(torch.int32).contiguous(),
    )


# Supports spread over three vocab tiles, the last one ragged.

VOCAB = 2 * _RATIO_BLOCK_V + 1000

P = {5: 0.5, _RATIO_BLOCK_V + 7: 0.3, VOCAB - 1: 0.2}

Q = {5: 0.2, _RATIO_BLOCK_V + 7: 0.1, VOCAB - 2: 0.7}


def tied_logits(rows, vocab, low, high):
    """bf16 logits whose max is an exact tie between ids ``low < high``."""
    logits = torch.randn(rows, vocab, device=DEV).to(torch.bfloat16)
    logits[:, low] = 30.0
    logits[:, high] = 30.0
    return logits


@triton.jit(do_not_specialize=["num_spec_steps", "accept_seed"])
def _dense_accept_kernel(
    output_token_ids_ptr,
    verdict_ptr,
    residual_row_ptr,
    cu_num_draft_tokens_ptr,
    input_ids_ptr,
    target_logits_indices_ptr,
    logits_ptr,
    temperatures_ptr,
    row_max_ptr,
    row_inv_z_ptr,
    row_argmax_ptr,
    q_ptr,
    q_row_base_ptr,
    bonus_token_ids_ptr,
    bonus_logits_indices_ptr,
    num_spec_steps,
    vocab_size,
    logits_row_stride,
    greedy_temperature,
    accept_seed,
):
    """The accept walk as it was over a dense ``[rows, V]`` q, kept as the
    reference the sparse walk must reproduce verdict for verdict."""
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    greedy = temp <= greedy_temperature
    q_base = tl.load(q_row_base_ptr + req_idx)
    has_q = q_base >= 0
    out_row = output_token_ids_ptr + req_idx * (num_spec_steps + 1)
    INVALID_TOKEN: tl.constexpr = -1
    rejected = False
    reject_pos = -1
    last_checked = -1
    for pos in range(num_draft_tokens):
        if rejected:
            tl.store(out_row + pos, INVALID_TOKEN)
        else:
            row = start_idx + pos
            x = _draft_token(input_ids_ptr, target_logits_indices_ptr, row)
            target_argmax = tl.load(row_argmax_ptr + row)
            if greedy:
                accept = x == target_argmax
            else:
                logit = tl.load(logits_ptr + row.to(tl.int64) * logits_row_stride + x)
                p = tl.exp(
                    logit.to(tl.float32) / temp - tl.load(row_max_ptr + row)
                ) * tl.load(row_inv_z_ptr + row)
                q = tl.load(
                    q_ptr + (q_base + pos).to(tl.int64) * vocab_size + x,
                    mask=has_q,
                    other=0.0,
                )
                u = tl.rand(accept_seed, row)
                accept = (q > 0.0) & (q <= 1.0) & (u * q < p)
            if accept:
                tl.store(out_row + pos, tl.cast(x, tl.int32))
            else:
                rejected = True
                reject_pos = pos
                if greedy:
                    tl.store(out_row + pos, tl.cast(target_argmax, tl.int32))
            last_checked += 1
    residual_row = -1
    if rejected and not greedy:
        residual_row = start_idx + reject_pos
    tl.store(residual_row_ptr + req_idx, residual_row)
    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        last_checked,
        rejected,
        num_spec_steps,
    )


@triton.jit(do_not_specialize=["batch_size", "residual_seed"])
def _dense_residual_race_kernel(
    residual_row_ptr,  # [batch_size] int32, the rejecting row, or -1
    logits_ptr,  # [num_tokens, V] raw target logits
    temperatures_ptr,  # [batch_size] fp32
    row_max_ptr,  # [num_tokens] fp32
    row_inv_z_ptr,  # [num_tokens] fp32
    row_floor_ptr,  # [num_tokens] fp32, the lowest logit p_f keeps (FILTER)
    dense_ptr,  # [>= batch_size, V] fp32: a rejecting request's q at its row
    q_row_base_ptr,  # [batch_size] int32, this request's first q row, or < 0
    part_key_ptr,  # [n_tiles, batch_size] fp32: best residual key
    part_idx_ptr,  # [n_tiles, batch_size] int32: its lowest id
    batch_size,
    vocab_size,
    logits_row_stride,
    residual_seed,
    BLOCK_V: tl.constexpr,
    FILTER: tl.constexpr,
):
    """`_ratio_residual_race_kernel` over a dense q row per request, the
    reference the sparse race is held to bit for bit."""
    req_idx = tl.program_id(0)
    tile = tl.program_id(1)
    row, _, has_q = _rejecting_row_of(residual_row_ptr, q_row_base_ptr, req_idx)
    if row >= 0:
        temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
        offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
        vmask = offs < vocab_size
        p = _target_p_tile(
            logits_ptr,
            row,
            logits_row_stride,
            offs,
            vmask,
            temp,
            tl.load(row_max_ptr + row),
            tl.load(row_inv_z_ptr + row),
            row_floor_ptr,
            FILTER,
        )
        q = tl.load(
            dense_ptr + req_idx.to(tl.int64) * vocab_size + offs,
            mask=vmask & has_q,
            other=0.0,
        )
        # Anything that is not a probability (NaN, inf, negative) counts as 0,
        # so the residual falls back toward p rather than inheriting garbage.
        q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
        residual = tl.maximum(p - q, 0.0)
        key = race_key(
            residual, residual_seed, row.to(tl.int64) * vocab_size, offs, vmask
        )
        best, best_id = argmax_lowest_id(key, offs, vmask, 2147483647)
        part = tile * batch_size + req_idx
        tl.store(part_key_ptr + part, best)
        tl.store(part_idx_ptr + part, best_id)


def dense_verify(args, q_dense, step):
    """The verify over a dense q: the same stats, race and pick kernels the
    sparse path runs, around `_dense_accept_kernel` and
    `_dense_residual_race_kernel`, the race reading the rejecting position's
    row of ``q_dense`` copied to the request's row."""
    logits = args["target_logits"]
    cu = args["cu_num_draft_tokens"]
    steps = args["num_spec_steps"]
    temps = args["temperatures"].float().contiguous()
    base = args["q_row_base"]
    num_tokens, vocab = logits.shape
    batch_n = cu.numel()
    tiles = triton.cdiv(vocab, _RATIO_BLOCK_V)

    def f32(*shape):
        return torch.empty(shape, device=DEV)

    def i32(*shape):
        return torch.empty(shape, dtype=torch.int32, device=DEV)

    part_max, part_sum, part_val = (f32(tiles, num_tokens) for _ in range(3))
    part_idx, row_argmax = i32(tiles, num_tokens), i32(num_tokens)
    row_max, row_inv_z = f32(num_tokens), f32(num_tokens)
    out, verdict, residual = i32(batch_n, steps + 1), i32(3, batch_n), i32(batch_n)
    keys, key_idx = f32(tiles, batch_n), i32(tiles, batch_n)
    _ratio_stats_kernel[(num_tokens, tiles)](
        logits,
        cu,
        temps,
        part_max,
        part_sum,
        part_val,
        part_idx,
        num_tokens,
        vocab,
        vocab,
        batch_n,
        batch_n.bit_length() + 1,
        BLOCK_V=_RATIO_BLOCK_V,
        num_warps=8,
    )
    softmax_combine_kernel[(num_tokens,)](
        part_max,
        part_sum,
        row_max,
        row_inv_z,
        part_val,
        part_idx,
        row_argmax,
        num_tokens,
        tiles,
        BLOCK_TILE=triton.next_power_of_2(tiles),
        WITH_ARGMAX=True,
        num_warps=4,
    )
    _dense_accept_kernel[(batch_n,)](
        out,
        verdict,
        residual,
        cu,
        args["input_ids"],
        args["target_logits_indices"],
        logits,
        temps,
        row_max,
        row_inv_z,
        row_argmax,
        q_dense,
        base,
        args["bonus_token_ids"],
        args["bonus_logits_indices"],
        steps,
        vocab,
        vocab,
        SAMPLER_EPS,
        _RATIO_ACCEPT_KEY | step,
        num_warps=1,
    )
    # The rejecting position's own q row at its request's row. A request without one gets some row the race masks
    # off, since it decides from ``residual`` and ``base`` alone.
    start = torch.cat((cu.new_zeros(1), cu[:-1]))
    has_row = (residual >= 0) & (base >= 0)
    q_row = torch.where(has_row, base + residual - start, 0)
    _dense_residual_race_kernel[(batch_n, tiles)](
        residual,
        logits,
        temps,
        row_max,
        row_inv_z,
        None,  # no floor: unfiltered
        q_dense[q_row.long()],
        base,
        keys,
        key_idx,
        batch_n,
        vocab,
        vocab,
        _RATIO_RESIDUAL_KEY | step,
        BLOCK_V=_RATIO_BLOCK_V,
        FILTER=False,
        num_warps=8,
    )
    _ratio_residual_pick_kernel[(batch_n,)](
        out,
        residual,
        cu,
        keys,
        key_idx,
        logits,
        temps,
        row_max,
        row_inv_z,
        row_argmax,
        None,
        batch_n,
        tiles,
        steps,
        vocab,
        vocab,
        _RATIO_RESIDUAL_KEY | step,
        BLOCK_TILE=triton.next_power_of_2(tiles),
        BLOCK_V=_RATIO_BLOCK_V,
        FILTER=False,
        num_warps=1,
    )
    return out, verdict


def top_of(logits, k):
    """softmax over each row's k largest logits, 0 elsewhere."""
    kept = logits.topk(k, dim=-1)
    q = torch.zeros_like(logits)
    return q.scatter_(-1, kept.indices, torch.softmax(kept.values, -1))


def filtered_p(logits, temps, top_ks=None, top_ps=None, widths=None):
    """`_ratio_target_p` over requests of ``widths`` draft rows each (one by
    default), filters and temperatures per request: ``(kept, p_f)`` per row."""
    rows, vocab = logits.shape
    widths = torch.ones(rows, dtype=torch.int64) if widths is None else widths
    cu = widths.cumsum(0).to(device=DEV, dtype=torch.int32)
    assert int(cu[-1]) == rows
    spec = target_p_scratch(rows, vocab)
    work = carve(scratch_buffer(spec, DEV), spec)
    temps = temps.float().contiguous()
    assert _ratio_target_p(logits, cu, temps, top_ks, top_ps, work)
    row_max, row_inv_z, row_floor = work.row_max, work.row_inv_z, work.row_floor
    x = logits.float()
    kept = x >= row_floor[:, None]
    temps = temps.repeat_interleave(widths.to(DEV))
    p = torch.exp(x / temps[:, None] - row_max[:, None]) * row_inv_z[:, None]
    return kept, torch.where(kept, p, 0.0)


def sampler_kept(logits, temps, top_ks=None, top_ps=None):
    """The ids `Sampler`'s aiter kernels may draw, and its filtered p: fewer
    than top_k ids strictly above, whose p sums to less than top_p, both over
    the unfiltered p `Sampler` computes (`_topk_topp_sample`). Per-row tensors
    or None; -1 and 1.0 are off. Sums in double, so a row whose cut is within
    rounding of its top_p is no check of the mass rule."""
    rows, vocab = logits.shape
    probs = (logits / temps[:, None]).softmax(dim=-1, dtype=torch.float32)
    sorted_p, order = probs.sort(-1, descending=True)
    starts = torch.ones_like(sorted_p, dtype=torch.bool)
    starts[:, 1:] = sorted_p[:, 1:] != sorted_p[:, :-1]
    slot = torch.arange(vocab, device=DEV).expand(rows, vocab)
    # Ids strictly above each id: the slot its tie group starts at.
    higher = torch.where(starts, slot, 0).cummax(-1).values
    mass = (sorted_p.double().cumsum(-1) - sorted_p.double()).gather(1, higher)
    k = vocab + 1 if top_ks is None else torch.where(top_ks > 0, top_ks, vocab + 1)
    p = 2.0 if top_ps is None else torch.where(top_ps < 1, top_ps.double(), 2.0)
    k = torch.as_tensor(k, device=DEV).view(-1, 1)
    p = torch.as_tensor(p, device=DEV).view(-1, 1)
    keep = torch.zeros_like(starts).scatter_(1, order, (higher < k) & (mass < p))
    p_f = torch.where(keep, probs.double(), 0.0)
    return keep, (p_f / p_f.sum(-1, keepdim=True)).float()


def tied_rows(rows, vocab, seed, dtype=torch.bfloat16):
    """Logits, every other row on a half-integer grid (ties everywhere, so
    every cut lands in a tie group), none near 0 (where `Sampler`'s fp32 p
    could round two distinct logits onto one value)."""
    torch.manual_seed(seed)
    logits = torch.randn(rows, vocab, device=DEV) * 2
    logits[::2] = (logits[::2] * 2).round() / 2
    logits = torch.where(logits.abs() < 0.05, 0.05, logits)
    return logits.to(dtype)


def assert_draws_are(draws, kept, p_f):
    """Draws from one row follow p_f, both ways: none lands outside the kept
    set, every kept id worth 20 draws turns up (a set missing a tie group
    would not), and each at its rate."""
    n = draws.numel()
    assert kept[draws].all(), "Sampler drew an id the verify filters out"
    freq = torch.bincount(draws, minlength=kept.numel()).float() / n
    assert (freq[p_f * n > 20] > 0).all(), "a kept id Sampler never draws"
    torch.testing.assert_close(freq, p_f, rtol=0, atol=0.01)


def first_token_law(target, temps, top_ks, top_ps, width=1024, step=41):
    """Draft with each row's own filters (the draft spells "off" 0 and 1.0),
    verify against the target's filtered p; ``(first token, accepted)``."""
    n = temps.numel()
    vocab = target.shape[-1]
    _, prev, w1, w2 = tables(1, vocab, 32, seed=137)
    base = (target.float() + torch.randn(vocab, device=DEV)).to(torch.bfloat16)
    greedy = (temps <= SAMPLER_EPS).to(torch.int32)
    store = empty_store(n, width)
    drafts, _ = sample(
        base.expand(n, -1).contiguous(),
        prev.expand(n).contiguous(),
        w1,
        w2,
        store,
        torch.arange(n, dtype=torch.int32, device=DEV),
        0,
        temps,
        greedy,
        torch.tensor([step], dtype=torch.int32, device=DEV),
        top_k=top_ks.clamp(min=0),
        top_p=top_ps,
    )
    out, (accepted, _, _) = verify(
        **verify_batch(drafts.to(torch.int32), 1, target.expand(n, -1), temps),
        q=store,
        step=step,
        top_ks=top_ks,
        top_ps=top_ps,
    )
    return out[:, 0].long(), accepted
