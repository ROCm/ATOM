# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``sparse_pa_nvfp4_q8`` (gfx950): the request-level NVFP4 sparse decode attention with the
top-16 selection inside.

* attention against an fp32 torch reference that decodes the NVFP4 pages from their byte
  layout (``nvfp4_pages``): every launch option's error must stay within the error aiter's
  ``sparse_pa_nvfp4_decode`` itself has against that reference (bf16 output), no NaN with
  NaN-poisoned unwritten tail keys, padding requests and selected blocks past a row's position;
  one row per workgroup without the trims is bit-exact with aiter's kernel;
* the in-kernel selection writes exactly the ``topk_idx`` of aiter's ``topk_per_row_small_k``
  (random, heavily tied, ascending and -inf scores; rows of 1..7813 blocks)."""

import random

import pytest
import torch

aiter = pytest.importorskip("aiter")
pytest.importorskip("flydsl")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="needs gfx950",
)

from aiter.ops.flydsl import sparse_pa_nvfp4_decode  # noqa: E402

from atom.model_ops.minimax_m3.index_topk import (  # noqa: E402
    _small_k_selector,
    build_n_valid_column_per_row,
)
from atom.model_ops.minimax_m3.sparse_pa_nvfp4_q8 import (  # noqa: E402
    parse_cfg,
    sparse_pa_nvfp4_decode_q8,
)

QL = 8
SM = 128**-0.5
PAGE = 2 * 9216  # one kv head: [K data | K scales | V data | V scales]
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _offsets(device):
    t = torch.arange(128, device=device)[:, None]
    d = torch.arange(128, device=device)[None, :]
    j, r = d // 16, t % 32
    nn = 4 * (r // 8) + r % 4
    k_data = t * 64 + d // 2
    k_shift = (d % 2) * 4
    k_scale = 8192 + ((((t // 32) * 4 + j // 2) * 16 + nn) * 2 + (r % 8) // 4) * 2 + j % 2
    v_data = 9216 + ((t // 8) * 128 + d) * 4 + (t % 8) // 2
    v_shift = (t % 2) * 4
    v_scale = 9216 + 8192 + j * 128 + t
    return [x.expand(128, 128) for x in (k_data, k_shift, k_scale, v_data, v_shift, v_scale)]


def _decode(pages):
    kd, ksh, ksc, vd, vsh, vsc = (x.reshape(-1) for x in _offsets(pages.device))
    tab, p = E2M1.to(pages.device), pages.long()
    k = tab[(p[:, kd] >> ksh) & 0xF] * pages[:, ksc].view(torch.float8_e4m3fn).float()
    v = tab[(p[:, vd] >> vsh) & 0xF] * pages[:, vsc].view(torch.float8_e4m3fn).float()
    return k.view(-1, 128, 128), v.view(-1, 128, 128)


def _reference(q, kv, topk, bt, seq, ks, vs):
    out = torch.zeros(q.shape[0], q.shape[1], 128, device=q.device)
    for i in range(q.shape[0]):
        b, r = divmod(i, QL)
        pos = int(seq[b]) - QL + r
        blks = sorted({int(x) for x in topk[0, i].tolist() if 0 <= x <= pos // 128}) if pos >= 0 else []
        if not blks:
            continue
        idx = torch.tensor(blks, device=q.device)
        k, v = _decode(kv[bt[b, idx].long()])
        keypos = (idx[:, None] * 128 + torch.arange(128, device=q.device)).reshape(-1)
        keep = keypos <= pos
        s = (q[i].float() @ (k.reshape(-1, 128)[keep] * ks).T) * SM
        out[i] = torch.softmax(s, -1) @ (v.reshape(-1, 128)[keep] * vs)
    return out


def _pool(pages, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    p = torch.randint(0, 256, (pages, PAGE), dtype=torch.uint8, device="cuda", generator=g)
    for off in (8192, 9216 + 8192):
        p[:, off : off + 1024] = torch.randint(
            0x30, 0x40, (pages, 1024), dtype=torch.uint8, device="cuda", generator=g
        )
    return p


def _case(lens, seed):
    rng = random.Random(seed)
    b, rows = len(lens), len(lens) * QL
    nblk = max(1, max((x + 127) // 128 for x in lens))
    pool = 4 * b * nblk + 64
    kv = _pool(pool, seed)
    bt = torch.randperm(pool, device="cuda")[: b * (nblk + 2)].view(b, nblk + 2).int().contiguous()
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    _, _, ksc, _, _, vsc = _offsets(kv.device)
    for i, x in enumerate(lens):  # unwritten keys of each tail block: NaN K / V scales
        if x > 0 and x % 128:
            pg = int(bt[i, (x - 1) // 128])
            for t in range(x % 128, 128):
                kv[pg, ksc[t, ::16]] = 0x7F
                kv[pg, vsc[t, ::16]] = 0x7F
    sel = []
    for x in lens:
        base = None
        for r in range(QL):
            pos = x - QL + r
            if pos < 0:
                sel.append([-1] * 16)
                continue
            sb, cand = pos // 128, list(range(pos // 128))
            if base is None or rng.random() < 0.3:
                base = rng.sample(cand, min(15, len(cand)))
            row = [sb] + [c for c in base if c < sb][:15]
            if rng.random() < 0.2 and len(row) < 16:
                row.append(sb + 1)  # a block past the row's position: must be ignored
            row += [-1] * (16 - len(row))
            rng.shuffle(row)
            sel.append(row)
    topk = torch.tensor(sel, dtype=torch.int32, device="cuda")[None]
    q = (torch.randn(rows, 16, 128, device="cuda") * 0.7).to(torch.bfloat16)
    return q, kv, topk, bt, seq


CFGS = ["1x1", "1x4", "2x1", "2x4", "4x2", "8x1", "1x1-lm-t12-rx", "1x2-lm-t12-rx", "1x4-lm-t12-rx"]
LENS = [
    [300, 1000, 4000, 9000],
    [129, 135, 256, 263, 5],
    [0, 2000, 0, 77777],
    [random.Random(40 + i).randrange(1, 200000) for i in range(16)],
]


@pytest.mark.parametrize("lens", LENS)
def test_attention_vs_reference(lens):
    q, kv, topk, bt, seq = _case(lens, 1)
    ks = torch.tensor([0.75], device="cuda")
    vs = torch.tensor([1.25], device="cuda")
    ref = _reference(q, kv, topk, bt, seq, 0.75, 1.25)
    base = sparse_pa_nvfp4_decode(q, kv, topk, bt, seq, SM, k_scale=ks, v_scale=vs, query_len=QL).float()
    eb = (base - ref).abs()
    for cfg in CFGS:
        out = sparse_pa_nvfp4_decode_q8(
            q, kv, topk, bt, seq, SM, k_scale=ks, v_scale=vs, query_len=QL, **parse_cfg(cfg)
        ).float()
        e = (out - ref).abs()
        assert not torch.isnan(out).any(), cfg
        assert e.max() <= 1.5 * eb.max() + 1e-3, (cfg, float(e.max()), float(eb.max()))
        assert e.mean() <= 1.2 * eb.mean() + 1e-6, (cfg, float(e.mean()), float(eb.mean()))
    exact = sparse_pa_nvfp4_decode_q8(
        q, kv, topk, bt, seq, SM, k_scale=ks, v_scale=vs, query_len=QL, rows_per_wg=1, splits=4
    )
    if len(lens) * QL <= 64:  # aiter splits 4 ways there too: same partition, same order
        assert torch.equal(exact, base.to(torch.bfloat16))


@pytest.mark.parametrize("mode", ["rand", "ties", "asc", "neg"])
@pytest.mark.parametrize(
    "lens,width", [([3000, 100, 129, 2047], 1024), ([175000, 60000, 1000000, 8, 500], 8192)]
)
def test_in_kernel_selection_matches_aiter(mode, lens, width):
    torch.manual_seed(2)
    b, rows = len(lens), len(lens) * QL
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    if mode == "rand":
        sc = torch.randn(1, rows, width, device="cuda")
    elif mode == "ties":
        sc = torch.randint(0, 6, (1, rows, width), device="cuda").float() * 0.25
    elif mode == "asc":
        sc = torch.arange(width, device="cuda").float().expand(1, rows, width).contiguous()
    else:
        sc = -torch.rand(1, rows, width, device="cuda") - 1.0
        sc[..., ::7] = float("-inf")
    nvalid = build_n_valid_column_per_row(seq, seq, batch=b, total_q=rows, num_idx_heads=1, decode_max_q=QL)
    select, _ = _small_k_selector()
    ref = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    select(sc.view(rows, width), nvalid, ref.view(rows, 16), 16, 0, 1)
    nblk = max((x + 127) // 128 for x in lens)
    kv = _pool(2 * b * nblk + 16, 3)
    bt = torch.randperm(kv.shape[0], device="cuda")[: b * nblk].view(b, nblk).int().contiguous()
    q = (torch.randn(rows, 16, 128, device="cuda") * 0.7).to(torch.bfloat16)
    for cfg in ("1x1-lm-t12-rx", "1x4-lm-t12-rx", "2x1"):
        got = torch.full_like(ref, -7)
        out = sparse_pa_nvfp4_decode_q8(
            q, kv, got, bt, seq, SM, query_len=QL, index_score=sc, write_topk=True, **parse_cfg(cfg)
        )
        assert torch.equal(got, ref), cfg
        fed = sparse_pa_nvfp4_decode_q8(q, kv, ref, bt, seq, SM, query_len=QL, **parse_cfg(cfg))
        assert torch.equal(out, fed), cfg
