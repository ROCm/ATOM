# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LeapQuant's kernels against torch references of the same algorithm.

LeapQuant (`atom/model_ops/fla_ops/leapquant.py`) keeps a GDN head's state
between window flushes as an int8 residual, fp32 scales and fp16 compensator
tokens, re-quantised once per window. A kernel that gets any of that wrong
still returns a plausible state, and no CPU test can see it, so these pin it:

* the quantiser is a torch mirror of `_lq_quantize`, and beats plain INT8;
* decode across windows follows a shadow that runs exact fp32 steps from the
  kernel's own boundary state, and its error against the exact trajectory
  does not grow from window to window;
* one stacked fold over every layer is the per-layer fold, bit for bit;
* the states real heads produce -- exactly low-rank, zero, tiny -- store and
  fold to finite values;
* every kernel gives the same bytes run after run.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises the LeapQuant Triton kernels; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.fla_ops.leapquant import (
    COLD_ITERS,
    leapquant_buffer_shapes,
    leapquant_fold,
    leapquant_gated_delta_rule,
    leapquant_load_state,
    leapquant_store_state,
)
from atom.model_ops.fla_ops.replayssm import replayssm_commit

DEV = "cuda"
# K heads shared by V heads, as in the Qwen GDN layers.
H, HV, K, V = 2, 4, 128, 128
RANK = 4
# The cursor folds once it holds CAP - 1 records: the paper's 16-token window.
CAP = 17
WINDOW = CAP - 1
# Every test's pool has this many slots: the kernels specialise on the pool's
# strides, so one geometry compiles each of them once.
POOL = 64
SCALE = K**-0.5
TINY = 1e-30


class Source:
    """GDN-like decode inputs. Per-channel scales and biases give the state the
    outlier rows and columns and the low-rank energy a real one has, which is
    what the compensators and the smoothing exist for."""

    def __init__(self, n: int, seed: int):
        self.n = n
        self.gen = torch.Generator(device=DEV).manual_seed(seed)
        self.k_scale = self.randn(H, K).mul(0.8).exp()
        self.k_bias = self.randn(H, K) * 2.0
        self.v_scale = self.randn(HV, V).mul(0.8).exp()
        self.v_bias = self.randn(HV, V)
        self.g_bias = self.randn(HV) * 0.5 - 3.0

    def randn(self, *shape):
        return torch.randn(*shape, device=DEV, generator=self.gen)

    def step(self):
        n = self.n
        q = self.randn(n, H, K) * self.k_scale + self.k_bias
        k = self.randn(n, H, K) * self.k_scale + self.k_bias
        v = self.randn(n, HV, V) * self.v_scale + self.v_bias
        g = -torch.nn.functional.softplus(self.randn(n, HV) * 0.5 + self.g_bias)
        beta = torch.sigmoid(self.randn(n, HV))

        # Values the model would produce in bf16, carried in fp32 so the
        # kernel writes an fp32 output: a bf16 one alone is 2e-3 off.
        def bf16(x):
            return x.bfloat16().float()

        return bf16(q), bf16(k), bf16(v), g, bf16(beta)


def torch_step(S, q, k, v, g, beta):
    """One exact GDN decode step on fp32 states ``[N, HV, K, V]``."""
    rep = HV // H
    qf = q.repeat_interleave(rep, 1)
    kf = k.repeat_interleave(rep, 1)
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * SCALE
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    S = S * g.exp()[..., None, None]
    u = beta[..., None] * (v - torch.einsum("nhkv,nhk->nhv", S, kf))
    S = S + kf[..., :, None] * u[..., None, :]
    return S, torch.einsum("nhkv,nhk->nhv", S, qf)


def make_states(n: int, steps: int, seed: int):
    """The exact recurrence run from zero for ``steps`` tokens."""
    src = Source(n, seed)
    S = torch.zeros(n, HV, K, V, device=DEV)
    for _ in range(steps):
        S, _ = torch_step(S, *src.step())
    return S, src


def unit(x):
    """`_lq_unit`: scaled by its max first, so a negligible direction is zeros."""
    m = x.abs().amax(-1, keepdim=True)
    x = torch.where(m > TINY, x / m.clamp_min(TINY), 0.0)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1.0)


def ref_quantize(S, iters=COLD_ITERS):
    """`_lq_quantize` from a cold start, in torch: the dequantised state."""
    S, low = S.clone(), torch.zeros_like(S)
    for _ in range(RANK):
        kv = unit(S.square().sum(-1).sqrt())
        for _ in range(iters):
            uv = torch.einsum("...kv,...k->...v", S, kv)
            kv = unit(torch.einsum("...kv,...v->...k", S, uv))
        uv = torch.einsum("...kv,...k->...v", S, kv)
        kh = kv.half().float()
        uh = uv.clamp(-65504, 65504).half().float()
        outer = kh[..., :, None] * uh[..., None, :]
        S, low = S - outer, low + outer
    c = S.abs().mean(-1).sqrt().clamp_min(1e-6)
    Sr = S * (1.0 / c)[..., :, None]
    amax = Sr.abs().amax(-2)
    b = torch.where(amax > TINY, amax / 127, 1.0)
    zq = (Sr * (1.0 / b)[..., None, :]).round().clamp(-127, 127)
    return zq * c[..., :, None] * b[..., None, :] + low


def int8_per_channel(S):
    """Plain symmetric INT8 per value channel: no compensators, no smoothing."""
    b = S.abs().amax(-2, keepdim=True).clamp_min(TINY) / 127
    return (S / b).round().clamp(-127, 127) * b


def rel(a, b) -> float:
    return ((a - b).norm() / b.norm()).item()


def slot_buffers(layers: int | None = None) -> list[torch.Tensor]:
    """A zeroed slot pool: residual, scales, compensators, then the records."""
    lead = (POOL,) if layers is None else (layers, POOL)
    sc_shape, comp_shape = leapquant_buffer_shapes(HV, K, V, RANK)
    return [
        torch.zeros(*lead, HV, K, V, device=DEV, dtype=torch.int8),
        torch.zeros(*lead, *sc_shape, device=DEV),
        torch.zeros(*lead, *comp_shape, device=DEV, dtype=torch.float16),
        torch.zeros(*lead, HV, CAP, K, device=DEV, dtype=torch.float16),
        torch.zeros(*lead, HV, CAP, V, device=DEV, dtype=torch.float16),
        torch.zeros(*lead, HV, CAP, device=DEV),
    ]


def scattered_slots(n: int) -> torch.Tensor:
    """``n`` slots of the pool, neither adjacent nor in order."""
    assert 2 * n <= POOL
    return torch.arange(POOL - 1, POOL - 1 - 2 * n, -2, device=DEV, dtype=torch.int32)


def cursors(fill: int = 0) -> torch.Tensor:
    return torch.full((POOL,), fill, device=DEV, dtype=torch.int32)


def decode(x, bufs, write_pos, slots, **kwargs):
    n = slots.numel()
    cu = torch.arange(n + 1, device=DEV, dtype=torch.int32)
    return leapquant_gated_delta_rule(
        *(t[None] for t in x),
        *bufs,
        write_pos,
        slots,
        cu,
        1,
        use_qk_l2norm_in_kernel=True,
        **kwargs,
    )


def commit(write_pos, slots):
    accepted = torch.ones(slots.numel(), device=DEV, dtype=torch.int32)
    replayssm_commit(write_pos, slots, accepted, 1, CAP)


def test_the_quantiser_is_its_torch_mirror_and_beats_plain_int8():
    n = 8
    S, _ = make_states(n, 256, seed=1)
    slots = scattered_slots(n)
    z, sc, comp = slot_buffers()[:3]
    leapquant_store_state(S, z, sc, comp, slots)
    got = leapquant_load_state(z, sc, comp, slots)

    assert rel(got, ref_quantize(S)) < 1e-3
    assert rel(got, S) < 0.5 * rel(int8_per_channel(S), S)


def test_decode_follows_its_shadow_and_the_error_does_not_compound():
    """The shadow runs exact fp32 steps and takes the kernel's boundary state at
    every flush, so it differs from the kernel by fp32 rounding alone. Against
    the exact trajectory, each window starts from a fresh quantisation, so a
    later window is no worse than an earlier one."""
    n = 8
    S0, src = make_states(n, 256, seed=2)
    slots = scattered_slots(n)
    bufs = slot_buffers()
    z, sc, comp = bufs[:3]
    write_pos = cursors()
    leapquant_store_state(S0, z, sc, comp, slots)
    write_pos[slots.long()] = -1  # the prefill sentinel
    shadow = leapquant_load_state(z, sc, comp, slots)
    exact = S0

    worst, errs, folds = 0.0, [], 0
    for _ in range(4 * WINDOW):
        x = src.step()
        commit(write_pos, slots)
        flush = write_pos[slots.long()] + 2 > CAP
        o = decode(x, bufs, write_pos, slots)[0].float()
        if flush.any():
            folds += 1
            folded = leapquant_load_state(z, sc, comp, slots)
            shadow = torch.where(flush[:, None, None, None], folded, shadow)
        shadow, o_shadow = torch_step(shadow, *x)
        exact, o_exact = torch_step(exact, *x)
        worst = max(worst, rel(o, o_shadow))
        errs.append(rel(o, o_exact))

    assert folds == 3
    assert worst < 1e-3
    windows = [sum(errs[i : i + WINDOW]) / WINDOW for i in range(0, len(errs), WINDOW)]
    # The first window starts from the prefill's cold fit, every later one
    # from a warm refit, so only the later ones are like for like.
    assert windows[-1] < 1.25 * windows[1], windows


def test_one_stacked_fold_is_the_per_layer_fold():
    layers, n = 2, 4
    slots = scattered_slots(n)
    stacked = slot_buffers(layers)
    z, sc, comp = stacked[:3]
    srcs = []
    for layer in range(layers):
        S0, src = make_states(n, 256, seed=10 + layer)
        leapquant_store_state(S0, z[layer], sc[layer], comp[layer], slots)
        srcs.append(src)
    per_layer = [b.clone() for b in stacked]
    write_pos = cursors()
    write_pos[slots.long()] = -1

    for _ in range(2 * CAP):
        commit(write_pos, slots)
        leapquant_fold(*stacked, write_pos, slots)
        for layer, src in enumerate(srcs):
            x = src.step()
            own = [b[layer] for b in stacked]
            ref = [b[layer] for b in per_layer]
            o = decode(x, own, write_pos, slots, fold=False)
            assert torch.equal(o, decode(x, ref, write_pos, slots))

    for got, want in zip(stacked, per_layer):
        assert torch.equal(got, want)


def test_degenerate_states_store_and_fold_finite():
    """A head whose decay wiped all but its last token is exactly low-rank: the
    compensators drain it and the power iterate on the residual underflows.
    Zero and tiny states are the same at any rank."""
    gen = torch.Generator(device=DEV).manual_seed(5)

    def randn(*shape):
        return torch.randn(*shape, device=DEV, generator=gen)

    def unit_keys():
        return torch.nn.functional.normalize(randn(HV, K), dim=-1)

    def outer(a, b):
        return a[..., :, None] * b[..., None, :]

    k1, u1, k2, u2 = unit_keys(), randn(HV, V), unit_keys(), randn(HV, V)
    cases = {
        "rank 1": outer(k1, u1),
        "rank 2": outer(k1, u1) + 0.3 * outer(k2, u2),
        "zero": torch.zeros(HV, K, V, device=DEV),
        "tiny": randn(HV, K, V) * 1e-25,
        "tiny rank 1 (1e-12)": outer(k1, u1) * 1e-12,
        # The power iterate's squares flush to zero here while the iterate
        # does not, which is what overflowed the fp16 keys.
        "tiny rank 1 (1e-13)": outer(k1, u1) * 1e-13,
        "generic": make_states(1, 512, seed=3)[0][0],
    }
    S = torch.stack(list(cases.values()))
    slots = scattered_slots(len(cases))
    z, sc, comp, buf_k, buf_u, buf_g = slot_buffers()
    leapquant_store_state(S, z, sc, comp, slots)
    cold = leapquant_load_state(z, sc, comp, slots)

    # A full window of one rank-1 token on top, then the warm fold.
    buf_k[:, :, :WINDOW] = k1.half()[None, :, None, :]
    buf_u[:, :, :WINDOW] = (0.1 * u1).half()[None, :, None, :]
    want = cold + WINDOW * outer(k1.half().float(), (0.1 * u1).half().float())
    leapquant_fold(z, sc, comp, buf_k, buf_u, buf_g, cursors(WINDOW), slots)
    warm = leapquant_load_state(z, sc, comp, slots)

    for i, name in enumerate(cases):
        assert torch.isfinite(cold[i]).all(), name
        assert torch.isfinite(warm[i]).all(), name
        assert torch.isfinite(comp[slots[i]].float()).all(), name
        m = S[i].abs().max()
        cold_err = rel(cold[i] / m, S[i] / m) if m > 0 else cold[i].abs().max().item()
        assert cold_err < 2e-2, name
        # Below rank 4 the window and the state are all compensators, so the
        # refit is exact to fp16 rounding; only the generic state keeps an
        # int8 residual.
        tol = 2e-2 if name == "generic" else 1e-4
        assert rel(warm[i], want[i]) < tol, name


def test_every_kernel_returns_the_same_bytes_every_run():
    """Races show up far more often on a GPU another job keeps busy, so a pass
    on an idle one is weaker evidence than a failure."""
    n, runs = POOL // 2, 20
    S0, src = make_states(n, 256, seed=12)
    slots = scattered_slots(n)
    bufs = slot_buffers()
    write_pos = cursors(-1)
    leapquant_store_state(S0, *bufs[:3], slots)
    # Walk one window, keeping the inputs of a step mid-window and of the flush.
    for _ in range(CAP):
        x = src.step()
        commit(write_pos, slots)
        h = int(write_pos[slots[0]])
        if h == CAP // 2:
            mid = ([b.clone() for b in bufs], write_pos.clone(), x)
        if h == CAP - 1:
            break
        decode(x, bufs, write_pos, slots)
    flush = ([b.clone() for b in bufs], write_pos.clone(), x)

    def decode_from(state):
        _, pos, xs = state
        return lambda *st: decode(xs, st, pos, slots)

    cases = {
        "decode, mid-window": (decode_from(mid), mid[0]),
        "decode + fold, flush step": (decode_from(flush), flush[0]),
        "stacked fold": (
            lambda *st: leapquant_fold(*(b[None] for b in st), flush[1], slots),
            flush[0],
        ),
        "store": (
            lambda *st: leapquant_store_state(S0, *st, slots),
            [torch.zeros_like(b) for b in bufs[:3]],
        ),
        "load": (lambda *st: leapquant_load_state(*st, slots), bufs[:3]),
    }
    for name, (fn, state) in cases.items():
        first = None
        for _ in range(runs):
            st = [b.clone() for b in state]
            out = fn(*st)
            got = st + ([out] if isinstance(out, torch.Tensor) else [])
            if first is None:
                first = got
            else:
                assert all(torch.equal(a, b) for a, b in zip(first, got)), name
