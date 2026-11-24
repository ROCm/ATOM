# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``gumbel_partials_bias``: vLLM's ``_gumbel_sample_kernel`` (``v1/worker/gpu/sample/gumbel.py``) on the
drafting path of one DSpark Markov step (bias rows, one cache column, fp32 noise, temperature applied).

Per row r (n rows, 200064 columns, 196 blocks of 1024):
1. ``y = bf16(f32(base[r]) + f32(bias[bias_rows[r]]))`` (round to nearest even); stored as the bf16 draft-cache row
   ``cache[slot, step]`` (pre-temperature) when the row's slot >= 0.
2. ``T = temperature[slot]`` (0 for slot < 0). T 0: ``v = y``. Otherwise ``v = y / T`` (fp32 IEEE division, skipped
   at T 1) ``+ noise`` with noise = ``-log(-log1p_neg_stable(u))``, u = murmur3_uniform32(seed ^ (step << 32),
   pos + 2^30, vocab id) in fp32 (Triton's ``tl.log`` as LLVM expands it on gfx950: ``kernel._log_normal``).
3. Per block b the (value, vocab id) Triton's ``tl.max(v, return_indices=True)`` returns for the 256-thread layout
   (thread t holds columns 4t .. 4t + 3): in-thread (c0 vs c1, >=), (c2 vs c3, >=), combine of the two; across the
   wave lanes xor 32, 16, 8, 4, 2, 1 with combine(own, partner); the wave's result is lane 63's; the four wave
   results combine (w0, w2), (w1, w3), then the two. combine keeps the first iff v1 > v2 or (v1 == v2 and i1 < i2).
   Without NaN this is the maximum and its lowest vocab id; with NaN the tree decides.
   ``local_max[r, b]`` fp32, ``local_argmax[r, b]`` int64. Columns >= 200064 are -inf with their own ids.

The torch reference below follows these rules on the noised values it is given (``block_tree_reference``) and builds
the noised values with torch fp64 math (``noised_values_approx``): torch has no gfx950 ``v_log_f32``, so the tests
compare the noise within 2 fp32 ulps and the tree exactly on exact (T 0) values; bit equality with Triton is checked
outside the tests.
"""

import torch

V, NB, BLOCK = 200064, 196, 1024


def _comb(v1, i1, v2, i2):
    keep = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return torch.where(keep, v1, v2), torch.where(keep, i1, i2)


def block_tree_reference(v: torch.Tensor):
    """fp32 [n, V] noised values -> (local_max fp32 [n, NB], local_argmax int64 [n, NB]) by Triton's tree."""
    n = v.shape[0]
    full = torch.full((n, NB * BLOCK), float("-inf"), dtype=torch.float32, device=v.device)
    full[:, :V] = v
    ids = torch.arange(NB * BLOCK, device=v.device).expand(n, -1)
    x = full.view(n, NB, 4, 64, 4)  # [row, block, wave, lane, element]
    ix = ids.reshape(n, NB, 4, 64, 4)
    k01 = x[..., 0] >= x[..., 1]
    p01 = torch.where(k01, x[..., 0], x[..., 1]), torch.where(k01, ix[..., 0], ix[..., 1])
    k23 = x[..., 2] >= x[..., 3]
    p23 = torch.where(k23, x[..., 2], x[..., 3]), torch.where(k23, ix[..., 2], ix[..., 3])
    av, ai = _comb(*p01, *p23)  # [n, NB, 4, 64]
    lanes = torch.arange(64, device=v.device)
    for m in (32, 16, 8, 4, 2, 1):
        av, ai = _comb(av, ai, av[..., lanes ^ m], ai[..., lanes ^ m])
    wv, wi = av[..., 63], ai[..., 63]  # [n, NB, 4]
    a0 = _comb(wv[..., 0], wi[..., 0], wv[..., 2], wi[..., 2])
    a1 = _comb(wv[..., 1], wi[..., 1], wv[..., 3], wi[..., 3])
    return _comb(*a0, *a1)


_M32 = 0xFFFFFFFF


def _rotl(x, s):
    return ((x << s) | (x >> (32 - s))) & _M32


def _mix(h, key):
    key = (key * 0xCC9E2D51) & _M32
    key = _rotl(key, 15)
    key = (key * 0x1B873593) & _M32
    h = h ^ key
    h = _rotl(h, 13)
    return (h * 5 + 0xE6546B64) & _M32


def _fmix(h):
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & _M32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & _M32
    return h ^ (h >> 16)


def uniform_reference(seed: int, step: int, pos: int, cols: torch.Tensor) -> torch.Tensor:
    """murmur3_uniform32 of (seed ^ (step << 32), pos + 2^30, cols) as fp32 (exact: two exact halves, one rounding)."""
    s = (seed ^ (step << 32)) & ((1 << 64) - 1)
    h = _mix(_mix(_mix(0, s & _M32), (s >> 32) & _M32), (pos + (1 << 30)) & _M32)
    c = cols.to(torch.int64)
    r = _fmix(_mix(torch.full_like(c, h), c) ^ 16)
    hi = (r >> 16).to(torch.float64)
    lo = (r & 0xFFFF).to(torch.float64)
    return (hi * 2.0 ** -16 + (lo + 0.5) * 2.0 ** -32).to(torch.float32)


def noise_approx(u: torch.Tensor) -> torch.Tensor:
    """-log(-log1p_neg_stable(u)) in fp64 from the fp32 u (the kernel's is fp32: within ~2 ulps of this)."""
    u = u.to(torch.float64)
    p = torch.full_like(u, 1.0 / 8.0)
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = 1.0 / k + u * p
    series = -u * p
    direct = torch.log(torch.clamp(1.0 - u, min=5.960464477539063e-08))
    l1 = torch.where(u < 0.25, series, direct)
    return (-torch.log(-l1)).to(torch.float32)


def cache_rows_reference(base: torch.Tensor, bias: torch.Tensor, bias_rows: torch.Tensor) -> torch.Tensor:
    """bf16 [n, V]: bf16(f32(base) + f32(bias[bias_rows]))."""
    return (base[:, :V].float() + bias[bias_rows.long(), :V].float()).to(torch.bfloat16)
