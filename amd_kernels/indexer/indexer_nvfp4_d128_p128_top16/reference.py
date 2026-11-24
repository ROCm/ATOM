# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference for the indexer kernels: the contract they meet.

Index pages (NVIDIA's NVFP4 index-K page, no global scale): page ``p`` of the
cache ``[num_pages, 128, 72]`` uint8 is ``[128 tokens x 64 B]`` packed e2m1 codes
(dim ``2i`` in the low nibble of byte ``i``, dim ``2i + 1`` in the high one), then
``[128 tokens x 8 B]`` e4m3 group scales (byte ``8 t + g`` scales dims ``16 g ..
16 g + 15`` of token ``t``). A key element is ``e2m1 * scale``.
``nvfp4_index_pages`` builds pages from keys (test input); ``dequant_index_pages``
reads them back exactly.

Block scores (``score_decode``, ``score_prefill``), per index head ``h``, query row
``r`` and 128-token block ``b`` (``block_scores_reference``):

- Keys: the fp8 route (``exact=False``, what serving runs) scores
  ``k = e4m3_rn_satfinite(e2m1 * scale)`` (clamped to +-448, rounded once to
  e4m3: AITER's / MSA's Q8KV4 keys); the prefill scorer's exact route
  ``k = e2m1 * scale`` (exact in f16). The query is e4m3 ``[rows, H, 128]``; every index head shares the one
  index-K head.
- ``s[h, r, b] = max(q[r, h] . k[t])`` over the tokens ``t`` of block ``b`` with
  ``t < kv_len[r]``; ``-inf`` where the row sees none. The kernels accumulate the
  128-dim dot in fp32 (MFMA); the reference sums in fp64 (decode) or fp32
  (prefill, where the products of e4m3 operands are exact).
- Sentinels on the row's visible blocks ``b < cdiv(kv_len, 128)``: ``b <
  init_blocks`` -> ``1e30``; ``b >= cdiv(kv_len, 128) - local_blocks`` -> ``1e29``
  (local wins over init), the scores AITER's top-k orders by.
- ``kv_len``: decode row ``t`` of request ``i`` (``query_len`` rows per request)
  ``max(seq_len[i] - query_len + t + 1, 0)``; a prefill row (each request's chunk
  ends its sequence) its position + 1.
- What is written: decode, every block below ``cdiv(seq_len, 128)`` of every row
  of the request and nothing past it; prefill, the blocks up to the row's own
  block are the contract (past it a scorer writes -inf up to its tile's window and
  leaves the rest; the top-k reads only the row's blocks).
- Tolerance: sentinels and -inf bit for bit; finite scores within ``SCORE_RTOL``
  relative + ``SCORE_ATOL`` (fp8 route, fp32 summation order: measured <= 1.13e-4
  of ``max(|ref|, 1)``, relative L2 <= 1.1e-5) or ``EXACT_RTOL`` (prefill exact
  route: measured equal to the fp64 reference). Repeated launches are bitwise identical.

Top-k (``topk``, ``topk_reference``): AITER's ``pa_sparse_block_topk`` order.

- A row's live blocks: ``b < min(nblk, width)``, ``nblk = cdiv(kv_len, 128)``
  (decode) or ``num_valid_pages[row]`` (ragged prefill rows), ``width = slots *
  64`` with ``slots = pow2_ceil(cdiv(cdiv(max_seq_len, 128), 64))``.
- Key = the fp32 score's order-preserving bits (``bits ^ ((bits >> 31) &
  0x7fffffff)`` as int32): -0 below +0, NaN (either sign) equal to -inf; equal keys
  go to the larger block index. The top ``min(16, live)`` blocks in descending
  order, -1 after.
- Exact: indices equal item by item.

Page table (``topk.index_topk_prefill``, ``page_table_reference``): per (row, KV
head ``kvh``; index head ``h`` serves the ``nkv / H`` KV heads ``h * nkv / H + g``)
table row ``row * nkv + kvh`` of 128 entries: the selected full blocks (``b < self
= (kv_len - 1) // 128``) in top-k order, then the row's own block if selected, 8
pages each, page ``e`` of block ``b`` = ``bt[req, b] * 8 * nkv + kvh + e * nkv``
(the page-16 numbering of the attend's interleaved cache), zeros after; the context
bound ``n_full * 128 + kv_len - self * 128`` with the own block, else
``min(n_valid * 128, kv_len)``. Exact, item by item.

Rank-owned index cache (``split4.decode_score``): rank ``r``
owns the blocks ``b % 4 == r``; the caller's block tables keep ``p % 4 == b % 4``
for table entry ``p`` of block ``b``, and a rank's cache holds only its pages,
page ``p`` at slot ``p // 4`` (``rank_owned_pages``; ``sharded=False`` reads a
replicated cache at slot ``p``). Decode: every rank scores its blocks of all four
heads, the single-head scorer's values bit for bit, and stores them into the
head's rank in the rank-owned order (``rank_owned_position``: block ``64 c + 4 j
+ r`` at position ``64 c + 16 r + j``; -inf past the request's block count inside
a written run, nothing outside the runs); the top-k with ``rank_owned_order``
reads that order and gives the block-order top-k item by item.
"""

import torch

BLK = 128  # tokens per block = one index page
D = 128  # index head dim
TOPK = 16  # blocks the top-k selects
PPB = 8  # pages (of 16 tokens) per block in the emitted table
SENTINEL_INIT = 1e30
SENTINEL_LOCAL = 1e29
SCORE_RTOL, SCORE_ATOL = 2e-4, 1e-3  # fp8 route vs the fp64 / fp32 reference
EXACT_RTOL = 1e-5  # exact (f16 key) route
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # magnitudes of codes 0..7
KEY_POS_INF = 0x7F800000
KEY_NEG_INF = (0xFF800000 ^ 0x7FFFFFFF) - (1 << 32)  # the key of -inf (and NaN)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def score_width(max_seq_len: int) -> int:
    """The score buffer's block axis the top-k needs: a power of two of whole 64-block strips."""
    n = cdiv(cdiv(max(max_seq_len, 1), BLK), 64)
    return (1 << (n - 1).bit_length()) * 64


# ---------------------------------------------------------------- pages


def nvfp4_index_pages(keys: torch.Tensor) -> torch.Tensor:
    """Quantize ``keys [num_pages * 128, 128]`` into NVFP4 index pages ``[num_pages, 128, 72]`` uint8 (test input):
    per 16-dim group the e4m3 scale ``amax / 6`` (round to nearest even), codes the nearest e2m1 of ``x / scale``
    (ties to the smaller magnitude)."""
    t = keys.shape[0]
    assert keys.shape[1] == D and t % BLK == 0
    pages = torch.empty(t // BLK, BLK * 72, dtype=torch.uint8, device=keys.device)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=keys.device)
    step = 1 << 16  # tokens per chunk
    for t0 in range(0, t, step):
        x = keys[t0 : t0 + step].float().reshape(-1, D // 16, 16)
        scale = (x.abs().amax(-1) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
        s = scale.float()[..., None]
        y = torch.where(s > 0, x / s.clamp(min=1e-30), 0.0)
        mag = torch.bucketize(y.abs().contiguous(), mids)  # 0..7
        codes = (mag | torch.where(y < 0, 8, 0)).to(torch.uint8).reshape(-1, D)
        data = codes[:, 0::2] | (codes[:, 1::2] << 4)  # [tokens, 64]
        p0, p1 = t0 // BLK, (t0 + codes.shape[0]) // BLK
        pages[p0:p1, : BLK * 64] = data.reshape(p1 - p0, BLK * 64)
        pages[p0:p1, BLK * 64 :] = scale.view(torch.uint8).reshape(p1 - p0, BLK * 8)
    return pages.view(t // BLK, BLK, 72)


def dequant_index_pages(cache: torch.Tensor) -> torch.Tensor:
    """Every page of ``cache [num_pages, 128, 72]`` uint8 (any page stride)
    dequantized exactly, ``e2m1 * scale``: ``[num_pages, 128, 128]`` fp32."""
    nb = cache.shape[0]
    flat = cache.reshape(nb, BLK * 72)
    data = flat[:, : BLK * 64].reshape(nb, BLK, 64)
    codes = torch.stack([data & 0xF, data >> 4], -1).reshape(nb, BLK, D).long()
    grid = torch.tensor(E2M1, device=cache.device)
    vals = grid[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    scale = flat[:, BLK * 64 :].reshape(nb, BLK, 8).view(torch.float8_e4m3fn).float()
    return (vals.view(nb, BLK, 8, 16) * scale[..., None]).reshape(nb, BLK, D)


def fp8_keys(keys: torch.Tensor) -> torch.Tensor:
    """The fp8 route's keys: ``e4m3_rn_satfinite`` of exact keys, back in fp32."""
    return keys.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def fp8_index_pages(cache: torch.Tensor) -> torch.Tensor:
    """Every NVFP4 page as an FP8 index page ``[num_pages, 128, 128]`` e4m3, the pages AITER's FP8 scorers read."""
    return fp8_keys(dequant_index_pages(cache)).to(torch.float8_e4m3fn)


# ---------------------------------------------------------------- scores


def block_scores_reference(
    q_rows: torch.Tensor,  # [R, H, D] e4m3
    keys: torch.Tensor,  # [nb * BLK, D] fp32 / fp64 keys of one request, in order
    kv_lens: torch.Tensor,  # [R] visible tokens of each row
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """``[H, R, nb]`` in ``keys``' dtype: per block the max over each row's visible
    tokens of q . k, -inf where none is visible, the init (1e30) / local (1e29)
    sentinels on the row's visible blocks (local wins)."""
    nb = keys.shape[0] // BLK
    dots = torch.einsum("rhd,td->hrt", q_rows.to(keys.dtype), keys)
    tok = torch.arange(nb * BLK, device=keys.device)
    dots = dots.masked_fill(tok[None, None, :] >= kv_lens[None, :, None], -torch.inf)
    mx = dots.view(dots.shape[0], dots.shape[1], nb, BLK).amax(-1)
    rb = (kv_lens.clamp(min=0) + BLK - 1) // BLK  # visible blocks per row
    blk = torch.arange(nb, device=keys.device)[None, :]
    vis = blk < rb[:, None]
    init = vis & (blk < init_blocks)
    local = vis & (blk >= (rb - local_blocks).clamp(min=0)[:, None])
    mx = torch.where(init[None], torch.full_like(mx, SENTINEL_INIT), mx)
    return torch.where(local[None], torch.full_like(mx, SENTINEL_LOCAL), mx)


def request_keys(
    cache: torch.Tensor, block_table: torch.Tensor, req: int, nb: int, exact: bool
) -> torch.Tensor:
    """Request ``req``'s first ``nb`` blocks of keys ``[nb * 128, D]`` fp32 (fp8
    route: e4m3-rounded) from the NVFP4 pages."""
    keys = dequant_index_pages(cache[block_table[req, :nb].long()])
    return (keys if exact else fp8_keys(keys)).reshape(-1, D)


def decode_kv_lens(seq_len: int, query_len: int, device) -> torch.Tensor:
    """``kv_len`` of a decode request's ``query_len`` rows."""
    t = torch.arange(query_len, device=device)
    return (seq_len - query_len + 1 + t).clamp(min=0)


def decode_scores_reference(
    q: torch.Tensor,  # [num_reqs * query_len, H, D] e4m3
    cache: torch.Tensor,  # [num_pages, 128, 72] uint8
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: list[int],
    init_blocks: int,
    local_blocks: int,
    query_len: int,
) -> list[torch.Tensor]:
    """Per request ``[H, query_len, cdiv(seq_len, 128)]`` (fp64 sums): the decode
    scores every row of the request gets (all blocks below the block count), fp8
    keys."""
    out = []
    for r, s in enumerate(seq_lens):
        nb = cdiv(s, BLK)
        sl = slice(r * query_len, (r + 1) * query_len)
        out.append(
            block_scores_reference(
                q[sl],
                request_keys(cache, block_table, r, nb, False).double(),
                decode_kv_lens(s, query_len, q.device),
                init_blocks,
                local_blocks,
            )
        )
    return out


def prefill_scores_reference(
    q_rows: torch.Tensor,  # [rows, H, D] e4m3 rows of one request
    keys: torch.Tensor,  # [nb * BLK, D] fp32 keys of the request
    positions: torch.Tensor,  # [rows] the rows' positions
    init_blocks: int,
    local_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``([H, rows, nb] fp32 scores, [rows, nb] mask of the contract's blocks)``
    for prefill rows (blocks up to each row's own block)."""
    want = block_scores_reference(
        q_rows, keys, positions + 1, init_blocks, local_blocks
    )
    nb = keys.shape[0] // BLK
    seen = torch.arange(nb, device=keys.device)[None, :] <= (positions // BLK)[:, None]
    return want, seen


# ---------------------------------------------------------------- top-k


def _sort_keys(score: torch.Tensor) -> torch.Tensor:
    """AITER's int64 order key per score ``(key << 32) | block``, NaN as -inf."""
    bits = score.contiguous().view(torch.int32).to(torch.int64)
    key = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    key = torch.where((key > KEY_POS_INF) | (key < KEY_NEG_INF), KEY_NEG_INF, key)
    idx = torch.arange(score.shape[-1], device=score.device)
    return key * (1 << 32) + idx


def topk_reference(
    score: torch.Tensor,  # [H, rows, >= width] fp32
    nblk: torch.Tensor,  # [rows] block count of each row
    max_seq_len: int,
) -> torch.Tensor:
    """``[H, rows, 16]`` int32: each row's top ``min(16, live)`` blocks in AITER's
    order, -1 after."""
    width = score_width(max_seq_len)
    s = score[..., :width]
    comp = _sort_keys(s)
    blocks = torch.arange(width, device=score.device)
    live = blocks[None, :] < nblk.clamp(max=width)[:, None]
    comp = torch.where(live[None], comp, torch.iinfo(torch.int64).min)
    top = comp.topk(TOPK, dim=-1).values  # width >= 64
    blk = (top & 0xFFFFFFFF).to(torch.int32)
    rank = torch.arange(TOPK, device=score.device)[None, None]
    return torch.where(rank < live.sum(-1)[None, :, None], blk, -1)


def page_table_reference(
    topk: torch.Tensor,  # [H, rows, 16] int32 (topk_reference)
    kv_lens: torch.Tensor,  # [rows] causal tokens of each row
    req: torch.Tensor,  # [rows] the request of each row
    block_table: torch.Tensor,  # [num_reqs, cols] int32
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(sparse_bt [rows * nkv, 128], sparse_ctx [rows * nkv])`` int32, the
    attend's page table the top-k emits (module docstring)."""
    heads, rows, _ = topk.shape
    hpg = max(num_kv_heads // heads, 1)
    dev = topk.device
    blk = topk.long()
    kv = kv_lens.long()[None, :, None]
    self_blk = torch.where(kv > 0, (kv - 1) // BLK, 0)
    valid = (kv > 0) & (blk >= 0) & (blk <= self_blk)
    full = valid & (blk < self_blk)
    n_full = full.sum(-1)
    n_valid = valid.sum(-1)
    slot = torch.where(full, full.long().cumsum(-1) - 1, n_full[..., None])
    cols = block_table.shape[1]
    page = block_table.long()[req.long()[None, :, None], blk.clamp(0, cols - 1)]
    sbt = torch.zeros(rows * num_kv_heads, TOPK * PPB, dtype=torch.int64, device=dev)
    ctx_rows = torch.zeros(rows * num_kv_heads, dtype=torch.int64, device=dev)
    kvl = kv_lens.long()[None, :]
    ctx = torch.where(
        n_valid > n_full,
        n_full * BLK + kvl - self_blk[..., 0] * BLK,
        torch.minimum(n_valid * BLK, kvl),
    )  # [H, rows]
    e = torch.arange(PPB, device=dev)
    n_idx = torch.arange(rows, device=dev)[None, :, None]
    for g in range(hpg):
        kvh = torch.arange(heads, device=dev)[:, None, None] * hpg + g  # [H, 1, 1]
        row = n_idx * num_kv_heads + kvh  # [H, rows, 1]
        sel = valid
        r = row.expand_as(blk)[sel]
        col = slot[sel][:, None] * PPB + e[None, :]
        base = page[sel] * (PPB * num_kv_heads) + kvh.expand_as(blk)[sel]
        val = base[:, None] + e[None, :] * num_kv_heads
        sbt[r[:, None].expand_as(col), col] = val
        ctx_rows[row[..., 0].reshape(-1)] = ctx.reshape(-1)
    return sbt.to(torch.int32), ctx_rows.to(torch.int32)


# ---------------------------------------------------------------- rank-owned index cache


def rank_owned_pages(cache: torch.Tensor, rank: int) -> torch.Tensor:
    """Rank ``rank``'s cache of the rank-owned index cache: page ``p`` (``p % 4 ==
    rank``) at slot ``p // 4`` of a replicated ``cache``."""
    return cache[rank::4].contiguous()


def rank_owned_position(b: torch.Tensor) -> torch.Tensor:
    """The rank-owned order's position of block ``b``: ``64 c + 16 (b % 4) + (b %
    64) // 4``."""
    return (b & ~63) | ((b & 3) << 4) | ((b >> 2) & 15)
