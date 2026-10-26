# SPDX-License-Identifier: MIT
"""Offline distribution tests for atom.model_ops.dspark_prob_sample.

1. prob_markov_sample draws x ~ softmax(z / T) with z = base + W1[prev] W2^T
   (and greedy rows take the argmax); stored z matches a torch reference.
2. prob_rejection_sample (standard and block) emits sequences distributed as
   the target autoregressive distribution: the first L emitted tokens of each
   request, completed on the host with fresh target samples, must follow the
   joint target law (chi-square), including top-p and rows without q.

Run: python test_dspark_prob_sample.py   (needs one GPU)
"""

import itertools
import math
import sys
import threading

import numpy as np
import torch

from atom.model_ops.dspark_prob_sample import (
    ProbDraftState,
    prob_markov_sample,
    prob_rejection_sample,
)

DEV = "cuda"
torch.manual_seed(0)
np.random.seed(0)

try:
    from scipy.stats import chi2 as _chi2

    def chi2_sf(x, dof):
        return float(_chi2.sf(x, dof))

except Exception:  # Wilson-Hilferty

    def chi2_sf(x, dof):
        z = ((x / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
        return 0.5 * math.erfc(z / math.sqrt(2))


def chisq(counts, probs, min_exp=5.0):
    counts = np.asarray(counts, dtype=np.float64).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    n = counts.sum()
    exp = probs * n
    # cells with prob 0 must have count 0
    bad = counts[probs <= 0].sum()
    keep = exp >= min_exp
    # pool small cells
    c = list(counts[keep]) + [counts[~keep & (probs > 0)].sum()]
    e = list(exp[keep]) + [exp[~keep & (probs > 0)].sum()]
    if e[-1] < min_exp:
        c, e = c[:-1], e[:-1]
        # renormalize (tiny pooled mass dropped)
        s = sum(e)
        e = [x * sum(c) / s for x in e]
    stat = sum((ci - ei) ** 2 / ei for ci, ei in zip(c, e))
    dof = max(len(c) - 1, 1)
    return stat, dof, chi2_sf(stat, dof), bad


def nucleus(logits, T, top_p):
    """Target distribution: softmax(l/T) on the smallest top set with mass>=top_p."""
    p = torch.softmax(logits.double() / T, -1)
    if top_p >= 1.0:
        return p
    sp, idx = torch.sort(p, descending=True)
    cs = torch.cumsum(sp, -1)
    # keep tokens whose preceding cumulative mass < top_p
    keep_sorted = (cs - sp) < top_p
    thr = sp[keep_sorted].min()
    keep = p >= thr  # ties at the boundary kept
    out = torch.where(keep, p, torch.zeros_like(p))
    return out / out.sum()


# ---------------------------------------------------------------------------
# 1. drafting
# ---------------------------------------------------------------------------


def test_draft(V=96, r=64, B=64, iters=1500, T=0.8):
    base = (torch.randn(B, V, device=DEV) * 1.5).to(torch.bfloat16)
    base[:] = base[0]  # identical rows => B samples per launch
    w1 = (torch.randn(V, r, device=DEV) * 0.15).to(torch.bfloat16)
    w2 = (torch.randn(V, r, device=DEV) * 0.15).to(torch.bfloat16)
    prev = torch.full((B,), 7, dtype=torch.int32, device=DEV)
    temps = torch.full((B,), T, dtype=torch.float32, device=DEV)
    temps[-1] = 1e-10  # one greedy row
    q = torch.empty(B, V, dtype=torch.float32, device=DEV)
    out = torch.empty(B, dtype=torch.int32, device=DEV)
    emb = torch.empty(B, r, dtype=torch.bfloat16, device=DEV)
    seed = torch.zeros(1, dtype=torch.int64, device=DEV)
    z_ref = base.float() + w1[prev].float() @ w2.float().t()
    counts = torch.zeros(V, dtype=torch.int64, device=DEV)
    greedy_ok = True
    for it in range(iters):
        seed.fill_(it * 4)
        prob_markov_sample(base, prev, w1, w2, out, q, temps, seed, 2, emb)
        counts += torch.bincount(out[:-1].long(), minlength=V)
        greedy_ok &= bool(out[-1].item() == int(q[-1].argmax().item()))
    zerr = (q - z_ref).abs().max().item()
    assert torch.equal(emb, w1[prev]), "markov embed mismatch"
    probs = torch.softmax(q[0].double() / T, -1).cpu().numpy()
    stat, dof, pv, bad = chisq(counts.cpu().numpy(), probs)
    print(f"[draft] V={V} T={T} n={int(counts.sum())} |z-zref|max={zerr:.2e} "
          f"chi2={stat:.1f} dof={dof} p={pv:.3f} greedy_ok={greedy_ok}")
    assert zerr < 1e-3 and greedy_ok and pv > 1e-3
    return pv


def test_draft_tp(V=128, r=64, B=48, tp=4, iters=40, T=0.9):
    """Sharded drafting == whole-vocab drafting (same seed => same Gumbel
    noise per global id => same winner), and every rank agrees."""
    base = (torch.randn(B, V, device=DEV) * 1.5).to(torch.bfloat16)
    w1 = (torch.randn(V, r, device=DEV) * 0.15).to(torch.bfloat16)
    w2 = (torch.randn(V, r, device=DEV) * 0.15).to(torch.bfloat16)
    prev = torch.randint(0, V, (B,), dtype=torch.int32, device=DEV)
    temps = torch.full((B,), T, dtype=torch.float32, device=DEV)
    temps[::7] = 1e-10
    seed = torch.zeros(1, dtype=torch.int64, device=DEV)
    cols = V // tp
    same = tot = 0
    for it in range(iters):
        seed.fill_(it * 4)
        out1 = torch.empty(B, dtype=torch.int32, device=DEV)
        q1 = torch.empty(B, V, dtype=torch.float32, device=DEV)
        emb = torch.empty(B, r, dtype=torch.bfloat16, device=DEV)
        prob_markov_sample(base, prev, w1, w2, out1, q1, temps, seed, 1, emb)

        def rank_fn(rk, grp):
            o = torch.empty(B, dtype=torch.int32, device=DEV)
            q = torch.empty(B, cols, dtype=torch.float32, device=DEV)
            e = torch.empty(B, r, dtype=torch.bfloat16, device=DEV)
            prob_markov_sample(base[:, rk * cols:(rk + 1) * cols], prev, w1, w2, o, q,
                               temps, seed, 1, e, vocab_start=rk * cols, tp_group=grp)
            return o, q

        res = run_ranks(tp, rank_fn)
        for rk, (o, q) in enumerate(res):
            assert torch.equal(o, res[0][0]), "ranks disagree"
            assert (q - q1[:, rk * cols:(rk + 1) * cols]).abs().max().item() < 1e-4
        same += int((res[0][0] == out1).sum())
        tot += B
    print(f"[draft-tp] tp={tp} V={V} sharded==whole ids {same}/{tot}")
    assert same >= tot * 0.995


# ---------------------------------------------------------------------------
# 2. verification
# ---------------------------------------------------------------------------


class ARModel:
    """Prefix-dependent random logits tables for target (p) and draft (q)."""

    def __init__(self, V, n, scale_p=1.5, scale_q=1.5, mix=0.6, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.V, self.n = V, n
        self.tp = {}
        self.tq = {}
        self.sp, self.sq, self.mix, self.g = scale_p, scale_q, mix, g

    def _get(self, tab, prefix, scale):
        if prefix not in tab:
            tab[prefix] = torch.randn(self.V, generator=self.g, dtype=torch.float64) * scale
        return tab[prefix]

    def target_logits(self, prefix):
        return self._get(self.tp, prefix, self.sp)

    def draft_logits(self, prefix):
        # correlated with the target so acceptance is non-trivial
        noise = self._get(self.tq, prefix, self.sq)
        return self.mix * self.target_logits(prefix) + (1 - self.mix) * noise


class _FakeGroup:
    """All-gather across rank threads on one GPU; launches are serialized by
    a global lock that all_gather releases while it waits."""

    def __init__(self, world):
        self.world_size = world
        self.bar = threading.Barrier(world)
        self.slots = [None] * world
        self.lock = threading.Lock()

    def view(self, rank):
        g = self

        class _R:
            world_size = g.world_size
            rank_in_group = rank

            def all_gather(self, t, dim=0, use_custom=True):
                g.slots[rank] = t
                g.lock.release()
                g.bar.wait()
                g.lock.acquire()
                out = torch.cat(list(g.slots), dim=0)
                g.lock.release()
                g.bar.wait()
                g.lock.acquire()
                return out

        return _R()


def run_ranks(world, fn):
    g = _FakeGroup(world)
    outs = [None] * world
    errs = []

    def body(rk):
        g.lock.acquire()
        try:
            outs[rk] = fn(rk, g.view(rk))
            torch.cuda.synchronize()
        except Exception as e:  # noqa
            errs.append(e)
            g.bar.abort()
        finally:
            g.lock.release()

    th = [threading.Thread(target=body, args=(rk,)) for rk in range(world)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errs:
        raise errs[0]
    return outs


def run_verify(model, n, R, T, top_p, use_block, hasq_frac=1.0, greedy=False, seed0=0, tp=1):
    """Return list of completed (n+1)-token sequences and acceptance stats."""
    V = model.V
    gen = torch.Generator().manual_seed(1234 + seed0)
    # drafts sampled from q given the drafted prefix (host)
    drafts = np.zeros((R, n), dtype=np.int64)
    qlog = torch.zeros(R, n, V, dtype=torch.float32)
    tlog = torch.zeros(R * n, V, dtype=torch.float32)
    bonus = np.zeros(R, dtype=np.int64)
    for i in range(R):
        prefix = ()
        for k in range(n):
            ql = model.draft_logits(prefix)
            qlog[i, k] = ql.float()
            if greedy:
                x = int(torch.argmax(ql))
            else:
                x = int(torch.multinomial(torch.softmax(ql / T, -1), 1, generator=gen))
            tlog[i * n + k] = model.target_logits(prefix).float()
            drafts[i, k] = x
            prefix = prefix + (x,)
        if greedy:
            bonus[i] = int(torch.argmax(model.target_logits(prefix)))
        else:
            bonus[i] = int(torch.multinomial(
                nucleus(model.target_logits(prefix), T, top_p), 1, generator=gen))
    # input ids layout: per request [anchor, d0..d_{n-1}], target rows at 0..n-1
    tokens_per = n + 1
    input_ids = torch.zeros(R * tokens_per, dtype=torch.int32)
    tli = torch.zeros(R * n, dtype=torch.int32)
    bli = torch.zeros(R, dtype=torch.int32)
    for i in range(R):
        input_ids[i * tokens_per] = 0
        input_ids[i * tokens_per + 1: (i + 1) * tokens_per] = torch.from_numpy(drafts[i])
        for k in range(n):
            tli[i * n + k] = i * tokens_per + k
        bli[i] = i * tokens_per + n
    cu = torch.arange(1, R + 1, dtype=torch.int32) * n
    src = torch.arange(R, dtype=torch.int32)
    if hasq_frac < 1.0:
        drop = torch.rand(R, generator=gen) >= hasq_frac
        src[drop] = -1
    temps = torch.full((R,), 1e-10 if greedy else T, dtype=torch.float32, device=DEV)
    args = (input_ids.to(DEV), tli.to(DEV), n, cu.to(DEV), tlog.to(DEV),
            torch.from_numpy(bonus).to(torch.int32).to(DEV), bli.to(DEV), temps,
            top_p if top_p < 1.0 else None, src.to(DEV))

    def make_state(vs, cols, world):
        st = ProbDraftState.allocate(R, n, V, DEV)
        st.q[:R, :, :cols] = qlog[:, :, vs:vs + cols].to(DEV)
        st.ids[:R] = torch.from_numpy(drafts).to(torch.int32).to(DEV)
        st.temps[:R] = 1e-10 if greedy else T
        st.set_layout(vs, cols, world)
        st.verify_step = seed0 * 7919
        return st

    if tp == 1:
        out, verdict = prob_rejection_sample(*args, make_state(0, V, 1), use_block=use_block)
    else:
        cols = V // tp
        states = [make_state(rk * cols, cols, tp) for rk in range(tp)]
        res = run_ranks(tp, lambda rk, grp: prob_rejection_sample(
            *args, states[rk], use_block=use_block, tp_group=grp))
        for o, v in res[1:]:
            assert torch.equal(o, res[0][0]) and torch.equal(v, res[0][1]), "ranks disagree"
        out, verdict = res[0]
    out = out.cpu().numpy()
    verdict = verdict.cpu().numpy()
    seqs = []
    for i in range(R):
        na = int(verdict[0, i])
        emitted = [int(t) for t in out[i, : na + 1]]
        assert all(t >= 0 for t in emitted), (i, out[i], verdict[:, i])
        assert emitted[:na] == list(drafts[i, :na]), "accepted prefix != drafts"
        assert verdict[1, i] == n - na
        assert verdict[2, i] == bli[i].item() - n + na, "anchor row"
        if na < n:
            assert all(t == -1 for t in out[i, na + 1:]), out[i]
        seqs.append(emitted)
    return seqs, verdict[0]


def complete(model, seqs, L, T, top_p, greedy, gen):
    full = []
    for s in seqs:
        s = list(s[:L])
        while len(s) < L:
            pl = model.target_logits(tuple(s))
            if greedy:
                s.append(int(torch.argmax(pl)))
            else:
                s.append(int(torch.multinomial(nucleus(pl, T, top_p), 1, generator=gen)))
        full.append(tuple(s))
    return full


def joint_probs(model, L, T, top_p):
    V = model.V
    probs = {}
    for seq in itertools.product(range(V), repeat=L):
        pr = 1.0
        for k in range(L):
            pk = nucleus(model.target_logits(tuple(seq[:k])), T, top_p)
            pr *= float(pk[seq[k]])
            if pr == 0.0:
                break
        probs[seq] = pr
    return probs


def test_verify(use_block, top_p, hasq_frac=1.0, V=6, n=3, R=4096, rounds=12, T=1.0, L=3, tp=1):
    model = ARModel(V, n, seed=11 if use_block else 7)
    gen = torch.Generator().manual_seed(99)
    allseq = []
    acc = []
    for rd in range(rounds):
        seqs, na = run_verify(model, n, R, T, top_p, use_block, hasq_frac, seed0=rd + 1, tp=tp)
        allseq += complete(model, seqs, L, T, top_p, False, gen)
        acc.append(na.mean())
    probs = joint_probs(model, L, T, top_p)
    keys = list(probs)
    idx = {k: i for i, k in enumerate(keys)}
    counts = np.zeros(len(keys))
    for s in allseq:
        counts[idx[s]] += 1
    stat, dof, pv, bad = chisq(counts, [probs[k] for k in keys])
    name = "block" if use_block else "standard"
    print(f"[verify-{name}] tp={tp} V={V} top_p={top_p} hasq={hasq_frac} n={len(allseq)} "
          f"mean_accepted={np.mean(acc):.3f} chi2={stat:.1f} dof={dof} p={pv:.3f} "
          f"zero-prob hits={int(bad)}")
    assert bad == 0 and pv > 1e-3
    return np.mean(acc)


def test_greedy(V=6, n=3, R=512):
    model = ARModel(V, n, seed=5)
    seqs, na = run_verify(model, n, R, 1.0, 1.0, False, greedy=True)
    for s in seqs:
        prefix = []
        for t in s:
            assert t == int(torch.argmax(model.target_logits(tuple(prefix)))), s
            prefix.append(t)
    print(f"[verify-greedy] R={R} all emitted tokens == target argmax chain, "
          f"mean_accepted={na.mean():.3f}")


def test_acceptance_gain(V=50, n=5, R=2048, T=1.0, top_p=0.95):
    """Expected accepted length: greedy draft + sample-match (current ATOM)
    vs probabilistic draft + standard / block verification."""
    model = ARModel(V, n, scale_p=2.5, scale_q=2.5, mix=0.75, seed=3)
    _, na_std = run_verify(model, n, R, T, top_p, False, seed0=50)
    _, na_blk = run_verify(model, n, R, T, top_p, True, seed0=51)
    # current path: argmax draft, accept iff == target sample (prob p(x))
    gen = torch.Generator().manual_seed(5)
    tot = 0
    for i in range(R):
        prefix = ()
        k = 0
        while k < n:
            x = int(torch.argmax(model.draft_logits(prefix)))
            y = int(torch.multinomial(nucleus(model.target_logits(prefix), T, top_p), 1, generator=gen))
            if x != y:
                break
            prefix += (x,)
            k += 1
        tot += k
    print(f"[accept] V={V} n={n}: greedy-draft/match={tot / R:.3f} "
          f"prob/standard={na_std.mean():.3f} prob/block={na_blk.mean():.3f}")


if __name__ == "__main__":
    test_draft()
    test_draft(T=1.0, iters=800)
    test_greedy()
    test_verify(False, 1.0)
    test_verify(False, 0.8)
    test_verify(False, 0.9, hasq_frac=0.5)
    test_verify(True, 1.0)
    test_verify(True, 0.8)
    test_verify(True, 0.9, hasq_frac=0.5)
    test_draft_tp()
    test_verify(False, 0.85, V=8, tp=4, rounds=8)
    test_verify(True, 0.85, V=8, tp=4, rounds=8)
    test_verify(False, 1.0, V=8, hasq_frac=0.5, tp=4, rounds=8)
    test_acceptance_gain()
    print("ALL PASSED")
    sys.exit(0)
