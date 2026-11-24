# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of ``norm_rope_nvfp4_insert`` and
``splitk_reduce_norm_rope_nvfp4_insert`` (root README, rule 3). Torch only.

Inputs, per token (``n`` tokens):
- ``qkv [n, (nq + 2 nkv + niq + 1) * 128]`` bf16, heads of 128 in the order
  q (nq) | k (nkv) | v (nkv) | index-q (niq) | index-k (1). The split-K form is a
  fp32 slab ``[sk, n, row]`` of the QKV GEMM's partials.
- Norm weights ``[128]`` bf16 for q, k, index-q, index-k; ``cos_sin_cache
  [max_pos, rotary_dim]`` bf16 (cos | sin halves); positions, the main and the index
  slot mappings (int64, -1 skips the token); ``k_scale`` / ``v_scale`` fp32 scalar
  (global dequantization) scales.

Arithmetic, in fp32 per token and head (q, k, index-q, index-k):
1. Gemma RMS norm: ``y = (x * rsqrt(mean(x^2) + eps)) * (1 + w)``.
2. Partial NeoX RoPE on dims ``[0, rotary_dim)``: ``cos, sin = cos_sin_cache[pos]``,
   halves ``x1, x2`` -> ``(x1 cos - x2 sin, x2 cos + x1 sin)``; dims above pass.
3. One rounding of the fp32 result: q to bf16;
   index-q to bf16, clamped to +-448, then e4m3 (RNE); k and index-k to bf16, then
   NVFP4. V is not normed: its bf16 values go to NVFP4.
The split-K form first sums the slab in fp32 in split order from zero,
``((0 + s0) + s1) + ...``, and rounds the sum to bf16 (the reduce's store).

NVFP4 of a 128-value row with global scale ``gs`` (``k_scale`` for K, ``v_scale``
for V, 1 for index-K), per 16-value group (8 groups):
``scale = e4m3(min((amax * RN(1/6)) * RN(1/gs), 448))`` (RNE),
``code = e2m1(x * RN(1 / (gs * scale)))`` (the nearest of 0, 0.5, 1, 1.5, 2, 3, 4, 6,
ties to the even code, saturating at 6, the sign in bit 3), all codes 0 for a zero
scale. The value is ``e2m1 * scale * gs``.

Page layouts (bytes; a block's pages are contiguous, the block stride is free):
- Main page ``[2 nkv, 128, 72]`` u8 per block: the K side is bytes
  ``[0, SIDE)``, ``SIDE = nkv * 128 * 72``, the V side the next ``SIDE`` bytes. For KV
  head ``h``, token ``pos = slot % 128`` of block ``slot // 128``, dim ``d``, group
  ``g``:
  - K codes ``h * 8192 + pos * 64 + d // 2`` (the even dim in the low nibble);
  - K scales ``nkv * 8192 + h * 1024 + k_scale_offset(pos, g)``;
  - V codes ``SIDE + h * 8192 + ((pos // 8 * 16 + d // 8) * 8 + d % 8) * 4 + pos % 8 // 2``,
    nibble ``pos % 2`` (a dword holds 8 tokens of one dim, a byte one dim of 2);
  - V scales ``SIDE + nkv * 8192 + h * 1024 + g * 128 + pos``.
- Index page ``[128, 72]`` u8 per block: codes ``pos * 64 + d // 2``, scales
  ``8192 + pos * 8 + g``.
Bytes and nibbles of tokens that are not written keep their contents.

Tolerances of the tests (``tests/attention/norm_rope_nvfp4_insert_d128_rot64_p128``): the kernels' sum order, rsqrt
and RoPE fma differ from this reference in the last fp32 bits, which moves a bf16, e4m3 or e2m1 rounding now and then.
- ``q_out``, ``index_q_out``: cosine >= ``COS_MIN``, relative L2 <= ``REL_L2`` (measured over every test batch:
  1 - cosine <= 4e-9, relative L2 <= 8.9e-5).
- Pages, dequantized at the written slots: cosine >= ``COS_MIN``, relative L2 <= ``PAGE_REL_L2``, at most
  ``CODE_MISMATCH`` of the written codes differ (measured: K 3.3e-6 of the codes, relative L2 <= 5.9e-4; V and
  index-K equal). Every bit the reference does not write equals the stale page's.
"""

import torch

D = 128  # head dim
BLK = 128  # tokens per page
WIDTH = 72  # bytes per token row: 64 code bytes + 8 scale bytes
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

COS_MIN = 0.99999
REL_L2 = 1e-3
PAGE_REL_L2 = 5e-3
CODE_MISMATCH = 1e-4


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``[..., 128]`` bf16 -> fp32."""
    xf = x.float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * r) * (1.0 + weight.float())


def neox_rope(
    y: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """``y [n, heads, 128]`` fp32, partial NeoX RoPE on the first ``rotary_dim`` dims."""
    rot = cos_sin_cache.shape[1]
    half = rot // 2
    cs = cos_sin_cache[positions].float()[:, None, :]
    cos, sin = cs[..., :half], cs[..., half:]
    x1, x2 = y[..., :half], y[..., half:rot]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, y[..., rot:]], dim=-1)


def nvfp4_quantize(x: torch.Tensor, gs: torch.Tensor):
    """``x [..., 128]`` fp32 -> codes ``[..., 128]`` u8 (4-bit values) and e4m3 scale
    bytes ``[..., 8]``; ``gs`` is a fp32 scalar tensor."""
    gs = gs.float().reshape(())
    groups = x.float().reshape(*x.shape[:-1], 8, 16)
    one_sixth = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device)
    scale = ((groups.abs().amax(-1) * one_sixth) * (1.0 / gs)).clamp(max=448.0)
    scale_e4m3 = scale.to(torch.float8_e4m3fn)
    sv = scale_e4m3.float()
    quant = torch.where(sv != 0, 1.0 / (gs * sv), torch.zeros_like(sv))
    y = groups * quant[..., None]
    grid = torch.tensor(E2M1_GRID, device=x.device)
    dist = (y.abs()[..., None] - grid).abs()
    is_min = dist == dist.amin(-1, keepdim=True)
    ids = torch.arange(8, device=x.device)
    # neighbouring codes alternate parity, so every tie has an even candidate
    even = torch.where(is_min & (ids % 2 == 0), ids, 8).amin(-1)
    mag = torch.where(even < 8, even, dist.argmin(-1))
    codes = mag | (torch.signbit(y).long() << 3)
    return codes.reshape(x.shape).to(torch.uint8), scale_e4m3.view(torch.uint8)


def nvfp4_dequantize(codes: torch.Tensor, scales: torch.Tensor, gs) -> torch.Tensor:
    """Inverse of ``nvfp4_quantize``: ``[..., 128]`` fp32."""
    grid = torch.tensor(E2M1_GRID, device=codes.device)
    c = codes.long()
    mag = grid[c & 7] * torch.where(c >= 8, -1.0, 1.0)
    sv = scales.view(torch.float8_e4m3fn).float() * torch.as_tensor(gs).float().reshape(())
    return (mag.reshape(*codes.shape[:-1], 8, 16) * sv[..., None]).reshape(codes.shape)


def k_scale_offset(pos: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Byte of (token, group) in a KV head's K scale region."""
    w, r = pos // 32, pos % 32
    n = (r // 8) * 4 + r % 4
    return (((w * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + g % 2


def flat_pages(cache: torch.Tensor) -> torch.Tensor:
    """``[blocks, page bytes]`` view of a page tensor whose block stride is free."""
    flat = cache.flatten(1)
    assert flat.data_ptr() == cache.data_ptr() and flat.stride(1) == 1
    return flat


def main_page_offsets(pos: torch.Tensor, nkv: int):
    """Byte offsets in a main page for tokens at ``pos [T]``: K codes ``[T, nkv, 64]``,
    K scales ``[T, nkv, 8]``, V codes ``[T, nkv, 128]`` (one per dim; the nibble is
    ``pos % 2``), V scales ``[T, nkv, 8]``."""
    dev = pos.device
    side = nkv * BLK * WIDTH
    p = pos[:, None, None]
    h = torch.arange(nkv, device=dev)[None, :, None]
    j = torch.arange(64, device=dev)[None, None, :]
    g = torch.arange(8, device=dev)[None, None, :]
    d = torch.arange(D, device=dev)[None, None, :]
    k_codes = h * (BLK * 64) + p * 64 + j
    k_scales = nkv * BLK * 64 + h * (BLK * 8) + k_scale_offset(p, g)
    v_codes = side + h * (BLK * 64) + ((p // 8 * 16 + d // 8) * 8 + d % 8) * 4 + (p % 8) // 2
    v_scales = side + nkv * BLK * 64 + h * (BLK * 8) + g * BLK + p
    return k_codes, k_scales, v_codes, v_scales


def index_page_offsets(pos: torch.Tensor):
    """Byte offsets in an index page: codes ``[T, 64]``, scales ``[T, 8]``."""
    p = pos[:, None]
    return p * 64 + torch.arange(64, device=pos.device), BLK * 64 + p * 8 + torch.arange(
        8, device=pos.device
    )


def _pack(codes: torch.Tensor) -> torch.Tensor:
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def insert_main_pages(kv_cache, slots, k_codes, k_scales, v_codes, v_scales) -> None:
    """Write the NVFP4 K / V rows ``[n, nkv, 128]`` (scales ``[n, nkv, 8]``) of the
    tokens with ``slots >= 0`` into ``kv_cache [blocks, 2 nkv, 128, 72]``."""
    nkv = kv_cache.shape[1] // 2
    pages = flat_pages(kv_cache)
    tok = (slots >= 0).nonzero().squeeze(1)
    if tok.numel() == 0:
        return
    s = slots[tok]
    blk, pos = s // BLK, s % BLK
    kc, ks, vc, vs = main_page_offsets(pos, nkv)
    b = blk[:, None, None]
    pages[b.expand_as(kc), kc] = _pack(k_codes[tok])
    pages[b.expand_as(ks), ks] = k_scales[tok]
    pages[b.expand_as(vs), vs] = v_scales[tok]
    # a V byte holds two tokens: one parity at a time, so each pass writes each
    # byte once and keeps the other token's nibble
    for parity in (0, 1):
        sel = (pos % 2) == parity
        if not bool(sel.any()):
            continue
        bb, off = b[sel].expand_as(vc[sel]), vc[sel]
        shift = 4 * parity
        old = pages[bb, off]
        new = (old & (0xF0 >> shift)) | (v_codes[tok[sel]] << shift)
        pages[bb, off] = new.to(torch.uint8)


def insert_index_pages(index_cache, slots, codes, scales) -> None:
    """Write the NVFP4 index-K rows ``[n, 128]`` of tokens with ``slots >= 0``."""
    pages = flat_pages(index_cache)
    tok = (slots >= 0).nonzero().squeeze(1)
    if tok.numel() == 0:
        return
    s = slots[tok]
    blk, pos = s // BLK, s % BLK
    co, so = index_page_offsets(pos)
    pages[blk[:, None].expand_as(co), co] = _pack(codes[tok])
    pages[blk[:, None].expand_as(so), so] = scales[tok]


def read_main_pages(kv_cache, slots):
    """K / V codes ``[T, nkv, 128]`` and scales ``[T, nkv, 8]`` of the tokens with
    ``slots >= 0`` (in token order)."""
    nkv = kv_cache.shape[1] // 2
    pages = flat_pages(kv_cache)
    s = slots[slots >= 0]
    blk, pos = s // BLK, s % BLK
    kc, ks, vc, vs = main_page_offsets(pos, nkv)
    b = blk[:, None, None]
    kb = pages[b.expand_as(kc), kc]
    k_codes = torch.stack([kb & 15, kb >> 4], dim=-1).flatten(-2)
    vb = pages[b.expand_as(vc), vc]
    v_codes = (vb >> (4 * (pos % 2))[:, None, None]) & 15
    return k_codes, pages[b.expand_as(ks), ks], v_codes, pages[b.expand_as(vs), vs]


def read_index_pages(index_cache, slots):
    pages = flat_pages(index_cache)
    s = slots[slots >= 0]
    blk, pos = s // BLK, s % BLK
    co, so = index_page_offsets(pos)
    cb = pages[blk[:, None].expand_as(co), co]
    return torch.stack([cb & 15, cb >> 4], dim=-1).flatten(-2), pages[blk[:, None].expand_as(so), so]


def written_bits(kv_cache, index_cache, slots, index_slots):
    """u8 masks (the shapes of the two caches' flat views) of the bits an insert of
    these slots writes."""
    nkv = kv_cache.shape[1] // 2
    n = slots.shape[0]
    dev = slots.device
    kv_mask = torch.zeros(kv_cache.shape[0], 2 * nkv * BLK * WIDTH, dtype=torch.uint8, device=dev)
    idx_mask = torch.zeros(index_cache.shape[0], BLK * WIDTH, dtype=torch.uint8, device=dev)
    full_c = torch.full((n, nkv, D), 15, dtype=torch.uint8, device=dev)
    full_s = torch.full((n, nkv, 8), 255, dtype=torch.uint8, device=dev)
    insert_main_pages(kv_mask.view(-1, 2 * nkv, BLK, WIDTH), slots, full_c, full_s, full_c, full_s)
    insert_index_pages(idx_mask.view(-1, BLK, WIDTH), index_slots, full_c[:, 0], full_s[:, 0])
    return kv_mask, idx_mask


def _heads(qkv: torch.Tensor, nq: int, nkv: int, niq: int):
    n = qkv.shape[0]
    x = qkv.view(n, nq + 2 * nkv + niq + 1, D)
    return (
        x[:, :nq],
        x[:, nq : nq + nkv],
        x[:, nq + nkv : nq + 2 * nkv],
        x[:, nq + 2 * nkv : nq + 2 * nkv + niq],
        x[:, nq + 2 * nkv + niq :],
    )


def norm_rope_nvfp4_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    index_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q_out: torch.Tensor,
    index_q_out: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
    prefill: bool = False,  # the kernel variant; the contract is the same
) -> None:
    """The op's contract, same arguments as the kernel: writes ``q_out``,
    ``index_q_out`` and the pages in place (module docstring)."""
    del prefill
    n = qkv.shape[0]
    if n == 0:
        return
    nq, nkv, niq = num_heads, num_kv_heads, num_index_heads
    q, k, v, iq, ik = _heads(qkv, nq, nkv, niq)

    def norm_rope(x, w):
        return neox_rope(gemma_rms_norm(x, w, eps), positions, cos_sin_cache)

    q_out.copy_(norm_rope(q, q_norm_weight).reshape(n, -1).to(q_out.dtype))
    iq_f = norm_rope(iq, index_q_norm_weight).to(torch.bfloat16).float()
    index_q_out.copy_(iq_f.clamp(-448.0, 448.0).reshape(n, -1).to(torch.float8_e4m3fn))
    k_bf = norm_rope(k, k_norm_weight).to(torch.bfloat16).float()
    ik_bf = norm_rope(ik, index_k_norm_weight).to(torch.bfloat16).float()
    one = torch.ones((), dtype=torch.float32, device=qkv.device)
    insert_main_pages(
        kv_cache,
        slot_mapping,
        *nvfp4_quantize(k_bf, k_scale),
        *nvfp4_quantize(v.float(), v_scale),
    )
    insert_index_pages(index_cache, index_slot_mapping, *nvfp4_quantize(ik_bf[:, 0], one))


def splitk_reduce(slab: torch.Tensor) -> torch.Tensor:
    """``[sk, n, row]`` fp32 -> ``[n, row]`` bf16: ``((0 + s0) + s1) + ...`` in fp32."""
    acc = torch.zeros_like(slab[0])
    for part in slab:
        acc = acc + part
    return acc.to(torch.bfloat16)


def splitk_reduce_norm_rope_nvfp4_insert(slab, split_k, *args, **kwargs) -> None:
    """The split-K form: same arguments as the kernel."""
    assert slab.shape[0] == split_k
    norm_rope_nvfp4_insert(splitk_reduce(slab), *args, **kwargs)
