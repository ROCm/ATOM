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

from atom.model_ops.rejection_sampler import rejection_sample
from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.dspark_markov_sample import (
    _sample_scratch,
    dspark_markov_sample,
    dspark_markov_sample_scratch,
)
from atom.model_ops.spec_sampling.ratio_target_p import (
    _RATIO_BLOCK_V,
    _ratio_target_p,
    target_p_scratch,
)
from atom.model_ops.spec_sampling.ratio_verify import (
    _RATIO_ACCEPT_KEY,
    _RATIO_RESIDUAL_KEY,
    RatioVerifyInputs,
    ratio_verify_scratch,
)
from atom.model_ops.spec_sampling.sampling_scratch import carve, scratch_buffer
from atom.model_ops.spec_sampling.sparse_q_select import (
    _Q_BINS,
    _Q_LEVELS_PER_NAT,
    _Q_MAX_GAP,
)
from atom.model_ops.spec_sampling.sparse_q_store import SparseQStore


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


# Triton's Philox4x32-10 (`triton.language.random.philox_impl`): the round
# multipliers, and the constants the key advances by after each round.
_PHILOX_ROUND = (0xD2511F53, 0xCD9E8D57)
_PHILOX_BUMP = (0x9E3779B9, 0xBB67AE85)
_PHILOX_ROUNDS = 10
_U32 = 0xFFFFFFFF
# `uint_to_uniform_float`'s scale for a 32-bit word, applied in fp32.
_RAND_SCALE = 4.6566127342e-10

# A decision of the verify is compared only where it clears its threshold by
# this much: the verify's p is fp32 (GPU exp and log, 1/Z summed in its own
# order), the reference's fp64, so a decision closer than that may go either
# way. ``u * q < p`` and a race key carry it relative to p; a top-p cut, an
# fp32 sum of at most 1 against top_p, carries it absolutely.
_P_RTOL = 1e-5
_MASS_ATOL = 2**-20


def _mul_hi_lo(a, b):
    """``(hi, lo)`` 32-bit words of ``a * b``: ``a`` an int64 tensor of uint32
    words, ``b`` a uint32 constant, multiplied by the 16-bit halves of ``b`` so
    no product leaves int64."""
    low = a * (b & 0xFFFF)
    high = a * (b >> 16) + (low >> 16)
    return high >> 16, ((high & 0xFFFF) << 16) | (low & 0xFFFF)


def philox(counter, key):
    """Philox4x32-10 as Triton runs it: four counter and two key words, int64
    tensors holding uint32 values; the four output words."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    a, b = _PHILOX_ROUND
    for _ in range(_PHILOX_ROUNDS):
        hi_a, lo_a = _mul_hi_lo(c0, a)
        hi_b, lo_b = _mul_hi_lo(c2, b)
        c0, c1, c2, c3 = hi_b ^ c1 ^ k0, lo_b, hi_a ^ c3 ^ k1, lo_a
        k0 = (k0 + _PHILOX_BUMP[0]) & _U32
        k1 = (k1 + _PHILOX_BUMP[1]) & _U32
    return c0, c1, c2, c3


def triton_rand(seed: int, offsets: torch.Tensor) -> torch.Tensor:
    """``tl.rand(seed, offsets)`` in torch, bit for bit: fp32 in [0, 1).

    Written from Triton's definition, not from ATOM's kernels: the counter is
    the offset's low and high words, the key the seed's, and the first output
    word, read as int32 and complemented if negative, is scaled in fp32.
    ``offsets`` are non-negative, so an int32 offset is the same counter as
    its int64 value."""
    offsets = offsets.to(torch.int64)
    zero = torch.zeros_like(offsets)
    word, _, _, _ = philox(
        (offsets & _U32, offsets >> 32, zero, zero),
        (zero + (seed & _U32), zero + (seed >> 32)),
    )
    word = torch.where(word >= 1 << 31, _U32 - word, word)
    scale = torch.tensor(_RAND_SCALE, dtype=torch.float32, device=word.device)
    return word.to(torch.int32).float() * scale


def target_p_reference(logits, temps, top_ks=None, top_ps=None):
    """`Sampler`'s p_f per row, in fp64: ``(p_f, undecided)``.

    softmax(logits / T) over the ids with fewer than top_k logits strictly
    above them and whose p (unfiltered) above them sums to less than top_p,
    renormalized. ``temps`` and the filters are per row, a top-k <= 0 or a
    top-p >= 1 off, or None. ``undecided`` marks a row with a tie group whose
    mass above lies within `_MASS_ATOL` of its top_p."""
    rows, vocab = logits.shape
    x = logits.double()
    p = torch.softmax(x / temps.double()[:, None], -1)
    sorted_x, order = x.sort(-1, descending=True)
    starts = torch.ones_like(sorted_x, dtype=torch.bool)
    starts[:, 1:] = sorted_x[:, 1:] != sorted_x[:, :-1]
    slot = torch.arange(vocab, device=x.device).expand(rows, vocab)
    # Ids strictly above each id: the slot its tie group starts at.
    higher = torch.where(starts, slot, 0).cummax(-1).values
    sorted_p = p.gather(1, order)
    mass = (sorted_p.cumsum(-1) - sorted_p).gather(1, higher)
    k = torch.full((rows, 1), vocab + 1, device=x.device)
    if top_ks is not None:
        k = torch.where(top_ks > 0, top_ks, vocab + 1).view(-1, 1)
    top_p = torch.full((rows, 1), 2.0, dtype=torch.float64, device=x.device)
    if top_ps is not None:
        top_p = torch.where(top_ps < 1, top_ps.double(), 2.0).view(-1, 1)
    keep = (higher < k) & (mass < top_p)
    near = (higher < k) & ((mass - top_p).abs() <= _MASS_ATOL)
    p_f = torch.zeros_like(p).scatter_(1, order, torch.where(keep, sorted_p, 0.0))
    return p_f / p_f.sum(-1, keepdim=True), near.any(-1)


def _race_reference(mass, bound, e):
    """``(winner, has_mass, near)`` of the exponential race ``argmax mass /
    e``, lowest id on a tie: whether any key is positive, and whether the
    winner fails to clear 0 and every other key by their ``bound / e``."""
    key = mass / e
    err = bound / e
    winner = int(key.argmax())
    above = key + err
    above[winner] = 0.0
    near = bool(key[winner] - err[winner] <= above.max())
    return winner, bool(key[winner] > 0), near


def _residual_reference(p_row, q_row, raw_argmax, e):
    """``(token, near)`` the residual draws at a rejecting row: the race of
    relu(p_f - q), else of p_f where the residual has no mass, else the raw
    argmax where p_f has none either."""
    bound = _P_RTOL * p_row
    token, has_mass, near = _race_reference((p_row - q_row).clamp(min=0), bound, e)
    if has_mass:
        return token, near
    # The verify's fp32 p may still leave mass where p_f is within rounding of q.
    near = bool(((p_row > 0) & (q_row - p_row <= bound)).any())
    token, has_mass, near_p = _race_reference(p_row, bound, e)
    if has_mass:
        return token, near or near_p
    return raw_argmax, near


def ratio_verify_reference(args, q, step, top_ks=None, top_ps=None):
    """The ratio verify of `verify_batch` ``args`` over the store ``q`` at
    Philox ``step``, under per-request filters (or None), in fp64 torch with
    none of the verify's code: ``(out, verdict, undecided)``, ``undecided``
    [batch] marking a request one of whose decisions lies within rounding of
    its threshold (`_P_RTOL`, `_MASS_ATOL`), which the verify may take either
    way.

    p_f is dense (`target_p_reference`); q is the store densified, an entry
    that is not a probability 0, and 0 throughout for a request whose row base
    is negative; the coins are `triton_rand` at the verify's counters, the
    accept's at the draft row and the race's at ``row * V + id``. A sampled
    request accepts x iff ``q(x) > 0`` and ``u * q(x) < p_f(x)``, a greedy one
    iff x is the raw logits' lowest-id argmax, and at the first rejection
    emits that argmax (greedy) or `_residual_reference`'s draw. Then the bonus
    or -1, the -1 tail, and the verdict `_finish_request` writes."""
    logits = args["target_logits"]
    rows, vocab = logits.shape
    steps = args["num_spec_steps"]
    cu = args["cu_num_draft_tokens"].tolist()
    batch = len(cu)
    widths = torch.tensor([0] + cu, device=DEV).diff()
    # Compared in fp32, as the verify compares its fp32 temperatures.
    temps = args["temperatures"].float()
    greedy = (temps <= SAMPLER_EPS).tolist()

    def per_row(values):
        return None if values is None else values.repeat_interleave(widths)

    row_temps = per_row(torch.where(temps > SAMPLER_EPS, temps, 1.0).double())
    p_f, cut_near = target_p_reference(
        logits, row_temps, per_row(top_ks), per_row(top_ps)
    )
    raw_argmax = logits.float().argmax(-1)
    drafts = args["input_ids"][args["target_logits_indices"] + 1].long()

    n = q.meta[:, 0].clamp(0, q.ids.shape[1])
    held = torch.arange(q.ids.shape[1], device=DEV) < n[:, None]
    probs = q.probs.double()
    probs = torch.where(held & (probs >= 0) & (probs <= 1), probs, 0.0)
    q_dense = torch.zeros(q.ids.shape[0], vocab, dtype=torch.float64, device=DEV)
    q_dense.scatter_add_(1, torch.where(held, q.ids.long(), 0), probs)

    # Every row's accept decision, as if the walk reached it.
    base = per_row(args["q_row_base"].long())
    q_row = (
        base
        + torch.arange(rows, device=DEV)
        - per_row(torch.tensor([0] + cu[:-1], device=DEV))
    )
    has_q = base >= 0
    q_x = torch.where(has_q, q_dense[q_row.clamp(min=0), drafts], 0.0)
    p_x = p_f.gather(1, drafts[:, None]).view(-1)
    coins = triton_rand(_RATIO_ACCEPT_KEY | step, torch.arange(rows, device=DEV))
    u_q = coins.double() * q_x
    sampled_accept = ((q_x > 0) & (u_q < p_x)).tolist()
    accept_near = ((q_x > 0) & ((u_q - p_x).abs() <= _P_RTOL * p_x)).tolist()
    greedy_accept = (drafts == raw_argmax).tolist()
    cut_near = cut_near.tolist()
    draft_ids, argmax_ids = drafts.tolist(), raw_argmax.tolist()

    out = torch.full((batch, steps + 1), -1, dtype=torch.int32)
    verdict = torch.zeros((3, batch), dtype=torch.int32)
    undecided = torch.zeros(batch, dtype=torch.bool)
    bonus = args["bonus_token_ids"].view(-1).tolist()
    bonus_rows = args["bonus_logits_indices"].tolist()
    start = 0
    for b, end in enumerate(cu):
        near = False
        accepted = 0
        for row in range(start, end):
            if greedy[b]:
                ok = greedy_accept[row]
            else:
                ok = sampled_accept[row]
                near = near or accept_near[row] or cut_near[row]
            if not ok:
                break
            out[b, accepted] = draft_ids[row]
            accepted += 1
        if accepted == end - start:
            out[b, accepted] = bonus[b]
        elif greedy[b]:
            out[b, accepted] = argmax_ids[row]
        else:
            race = start + accepted
            offsets = race * vocab + torch.arange(vocab, device=DEV)
            e = -triton_rand(_RATIO_RESIDUAL_KEY | step, offsets).double().log()
            q_race = q_dense[q_row[race]] if has_q[race] else torch.zeros_like(e)
            token, race_near = _residual_reference(
                p_f[race], q_race, argmax_ids[race], e
            )
            out[b, accepted] = token
            near = near or race_near
        undecided[b] = near
        anchor = bonus_rows[b] - (end - start)
        verdict[:, b] = torch.tensor([accepted, steps - accepted, anchor + accepted])
        start = end
    return out.to(DEV), verdict.to(DEV), undecided.to(DEV)


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
