# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The Triton Q quantization matches mla_nvfp4_quantize_q byte for byte."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
requires_gfx1250 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx1250" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="AITER's NVFP4 MLA producers are gfx1250-only",
)
# q_aux: the first 64 bytes of a row are RoPE and residual E2M1 codes, the
# rest E4M3 scales and padding.
Q_AUX_CODES = 64
# kv_aux: each 2048-byte token16 tile starts with 1024 bytes of codes.
KV_AUX_TILE, KV_AUX_CODES = 2048, 1024


def _query(batch, seed, strided=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ch = torch.exp(torch.randn(576, generator=gen, device="cuda") * 1.5)
    head = torch.exp(torch.randn(1, 128, 1, generator=gen, device="cuda"))
    q = (
        torch.randn(batch, 128, 576, generator=gen, device="cuda") * ch * head * 0.1
    ).to(torch.bfloat16)
    q[torch.rand(q.shape, generator=gen, device="cuda") < 0.02] = 0
    q[torch.rand(q.shape, generator=gen, device="cuda") < 0.005] = -0.0
    q[:, :3, 512:528] = 0  # whole zero blocks
    nope = q[..., :512].contiguous()
    if strided:
        # RoPE as it leaves q_proj: a [B, 128, 64] view of a wider row.
        wide = torch.zeros(batch, 128, 192, dtype=torch.bfloat16, device="cuda")
        wide[..., 128:] = q[..., 512:]
        rope = wide[..., 128:]
    else:
        rope = q[..., 512:].contiguous()
    return q, nope, rope


@requires_gfx1250
@pytest.mark.parametrize(
    "batch,seed,strided", [(1, 0, False), (3, 1, True), (77, 2, False), (512, 3, True)]
)
def test_quantize_q_matches_the_aiter_producer(batch, seed, strided):
    pytest.importorskip("aiter.ops.mla_nvfp4")
    from aiter.ops.mla_nvfp4 import mla_nvfp4_quantize_q

    from atom.model_ops.attentions.nvfp4_q_triton import quantize_q

    q, nope, rope = _query(batch, seed, strided)
    want_n, want_a = mla_nvfp4_quantize_q(q.contiguous())
    got_n, got_a = quantize_q(nope, rope)
    torch.cuda.synchronize()
    bad = (got_n != want_n).nonzero()
    assert bad.numel() == 0, f"q_nope differs at {bad[:8].tolist()}"
    bad = (got_a != want_a).nonzero()
    assert bad.numel() == 0, f"q_aux differs at {bad[:8].tolist()}"


def test_quantize_q_replays_in_a_graph():
    from atom.model_ops.attentions.nvfp4_q_triton import quantize_q

    _, nope, rope = _query(64, 5)
    eager = [t.clone() for t in quantize_q(nope, rope)]
    static_n, static_r = torch.zeros_like(nope), torch.zeros_like(rope)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = quantize_q(static_n, static_r)
    static_n.copy_(nope)
    static_r.copy_(rope)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[0], eager[0]) and torch.equal(out[1], eager[1])


@pytest.mark.parametrize("rows", [16, 32])
def test_decode_prep_zeroes_padding_rows(rows):
    """A padding row (slot -1) gets a zero Q and no cache write; live rows are
    quantized as `quantize_q` does, for either Q rows per program."""
    from atom.model_ops.attentions.nvfp4_hp_ring import Nvfp4HpRing
    from atom.model_ops.attentions.nvfp4_q_triton import decode_prep, quantize_q

    batch, pad = 40, [3, 17, 39]
    _, nope, rope = _query(batch, 9, strided=True)
    gen = torch.Generator(device="cuda").manual_seed(10)
    k = torch.randn(batch, 576, generator=gen, device="cuda").to(torch.bfloat16)
    pos = torch.randint(0, 4000, (batch,), generator=gen, device="cuda")
    slots = torch.arange(batch, device="cuda") * 64 + pos % 64
    slots[pad] = -1
    ring = Nvfp4HpRing(batch + 1, 1, 576, device="cuda")
    kv_nope = torch.zeros(batch, 16384, dtype=torch.uint8, device="cuda")
    kv_aux = torch.zeros(batch, 8192, dtype=torch.uint8, device="cuda")
    req = torch.arange(batch, dtype=torch.int32, device="cuda")
    qn, qa = decode_prep(
        nope,
        rope,
        ring.buf[:, 0],
        req,
        k[:, :512],
        k[:, 512:],
        slots,
        kv_nope,
        kv_aux,
        pos,
        rows=rows,
    )
    want_n, want_a = quantize_q(nope, rope)
    torch.cuda.synchronize()
    live = torch.ones(batch, dtype=torch.bool, device="cuda")
    live[pad] = False
    assert torch.equal(qn[live], want_n[live]) and torch.equal(qa[live], want_a[live])
    assert (qn[~live] == 0).all()
    assert (qa[~live][..., :104] == 0).all() and (qa[~live][..., 104:] == 127).all()
    assert (kv_nope[pad] == 0).all() and (kv_aux[pad] == 0).all()


def test_decode_prep_q_planes_writes_the_rope_part():
    """With the NoPE already written (the fused K-up), only the RoPE bytes of
    q_aux are written, the same as quantizing the whole Q, and the tail and
    both cache planes are those of the whole-Q launch."""
    from atom.model_ops.attentions.nvfp4_hp_ring import Nvfp4HpRing
    from atom.model_ops.attentions.nvfp4_q_triton import decode_prep

    batch = 48
    _, nope, rope = _query(batch, 11, strided=True)
    gen = torch.Generator(device="cuda").manual_seed(12)
    k = torch.randn(batch, 576, generator=gen, device="cuda").to(torch.bfloat16)
    pos = torch.randint(0, 4000, (batch,), generator=gen, device="cuda")
    slots = torch.arange(batch, device="cuda") * 64 + pos % 64
    slots[[2, 30]] = -1
    req = torch.arange(batch, dtype=torch.int32, device="cuda")
    ring0 = Nvfp4HpRing(batch + 1, 1, 576, device="cuda")
    ring0.buf.copy_(
        (torch.randn(ring0.buf.shape, generator=gen, device="cuda")).to(torch.bfloat16)
    )
    outs = []
    for planes in (False, True):
        ring = ring0.buf.clone()
        kv_nope = torch.zeros(batch, 16384, dtype=torch.uint8, device="cuda")
        kv_aux = torch.zeros(batch, 8192, dtype=torch.uint8, device="cuda")
        if planes:
            qn, qa = outs[0][0].clone(), outs[0][1].clone()
            qa[..., :64] = 0x55
            qa[..., 96:] = 0x55
            decode_prep(
                None,
                rope,
                ring[:, 0],
                req,
                k[:, :512],
                k[:, 512:],
                slots,
                kv_nope,
                kv_aux,
                pos,
                q_planes=(qn, qa),
                use_aiter=False,
            )
        else:
            qn, qa = decode_prep(
                nope,
                rope,
                ring[:, 0],
                req,
                k[:, :512],
                k[:, 512:],
                slots,
                kv_nope,
                kv_aux,
                pos,
            )
        torch.cuda.synchronize()
        outs.append((qn, qa, ring, kv_nope, kv_aux))
    assert all(torch.equal(a, b) for a, b in zip(outs[1], outs[0]))


@requires_gfx1250
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("batch,steps", [(24, 40), (300, 6)])
def test_aiter_decode_prep_writes_the_same_bytes(batch, steps, fused):
    """AITER's gfx1250 decode prep (Q and K in one launch) against the Triton
    one, step after step from the same stale bytes: q_aux, the tail and both
    cache planes, over tile starts and edges, padding rows, K outliers and
    signed zeros, with the fused RoPE and with pre-roped input."""
    from aiter.rotary_embedding import get_rope

    from atom.model_ops.attentions.nvfp4_q_triton import _aiter_decode_prep, decode_prep

    if _aiter_decode_prep() is None:
        pytest.skip("this AITER has no mla_nvfp4_decode_prep")

    yarn = {
        "type": "yarn",
        "factor": 40,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    rope = get_rope(
        64,
        rotary_dim=64,
        max_position=163840,
        base=10000,
        is_neox_style=False,
        rope_scaling=yarn,
        dtype=torch.bfloat16,
    ).to("cuda")
    cos, sin = (rope.cos_cache, rope.sin_cache) if fused else (None, None)
    pages_per = 8
    gen = torch.Generator(device="cuda").manual_seed(batch + steps)
    req = torch.arange(batch, dtype=torch.int32, device="cuda")
    pos = torch.randint(0, 300, (batch,), generator=gen, device="cuda")
    pos[:8] = torch.tensor([0, 15, 16, 31, 63, 64, 127, 255], device="cuda")
    table = torch.randperm(batch * pages_per, generator=gen, device="cuda").view(
        batch, pages_per
    )
    ring0 = torch.randn((batch + 1, 16, 576), generator=gen, device="cuda").to(
        torch.bfloat16
    )
    stale = lambda n: torch.randint(
        0, 256, (batch * pages_per, n), generator=gen, dtype=torch.uint8, device="cuda"
    )
    nope0, aux0 = stale(16384), stale(8192)
    tri = [ring0.clone(), nope0.clone(), aux0.clone()]
    hip = [ring0.clone(), nope0.clone(), aux0.clone()]
    pad = [5, 17, batch - 1]
    for step in range(steps):
        k = torch.randn((batch, 576), generator=gen, device="cuda") * 0.5
        k = torch.where(
            torch.rand(k.shape, generator=gen, device="cuda") < 0.02, k * 30, k
        ).to(torch.bfloat16)
        k[torch.rand(k.shape, generator=gen, device="cuda") < 0.01] = -0.0
        q = (torch.randn((batch, 128, 192), generator=gen, device="cuda") * 2).to(
            torch.bfloat16
        )
        q[torch.rand(q.shape, generator=gen, device="cuda") < 0.01] = 0
        qn = torch.zeros((batch, 128, 256), dtype=torch.uint8, device="cuda")
        qa0 = torch.randint(
            0, 256, (batch, 128, 128), generator=gen, dtype=torch.uint8, device="cuda"
        )
        slots = table.gather(1, (pos // 64)[:, None]).squeeze(1) * 64 + pos % 64
        slots[pad] = -1
        outs = []
        for (ring, kn, ka), use_aiter in ((tri, False), (hip, True)):
            qa = qa0.clone()
            decode_prep(
                None,
                q[..., 128:],
                ring,
                req,
                k[:, :512],
                k[:, 512:],
                slots,
                kn,
                ka,
                pos,
                cos,
                sin,
                q_planes=(qn, qa),
                use_aiter=use_aiter,
            )
            outs.append(qa)
        torch.cuda.synchronize()
        # The HIP RoPE is the RoPE kernel's bf16 instructions; the Triton one's
        # emulation can lose the sign of an exact zero (a code 0 for 8), so
        # codes may differ by a zero's sign and in nothing else.
        assert _same_but_zero_sign(outs[0], outs[1], Q_AUX_CODES), (step, "q_aux")
        assert torch.equal(tri[0][..., :512], hip[0][..., :512]), (step, "tail")
        assert _same_bf16_but_zero_sign(tri[0][..., 512:], hip[0][..., 512:]), (
            step,
            "tail rope",
        )
        assert torch.equal(tri[1], hip[1]), (step, "kv_nope")
        assert _same_but_zero_sign(
            tri[2].view(-1, KV_AUX_TILE), hip[2].view(-1, KV_AUX_TILE), KV_AUX_CODES
        ), (step, "kv_aux")
        pos = torch.where(slots >= 0, pos + 1, pos)


def _same_but_zero_sign(a, b, codes):
    """Aux planes equal byte for byte, except that in the first ``codes`` bytes
    of each row, the E2M1 codes, nibbles 0 / 8 (+0 / -0) may swap."""
    lo_a, hi_a, lo_b, hi_b = a & 15, a >> 4, b & 15, b >> 4
    ok = lambda x, y: (x == y) | ((x | y) == 8) & ((x & 7) == 0) & ((y & 7) == 0)
    same_codes = (ok(lo_a, lo_b) & ok(hi_a, hi_b))[..., :codes].all()
    return bool(same_codes) and torch.equal(a[..., codes:], b[..., codes:])


def _same_bf16_but_zero_sign(a, b):
    """BF16 tensors equal except +0 / -0."""
    return bool(
        ((a.view(torch.int16) == b.view(torch.int16)) | ((a == 0) & (b == 0))).all()
    )
