# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The fused K-up + NVFP4 Q quantization writes the bytes of the BF16 BMM
followed by `quantize_q`, and `decode_prep` completes the RoPE part in place."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

H, P, L = 128, 128, 512


def _gfx1250():
    return "gfx1250" in getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


def _inputs(batch, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # q_proj output [B, H, 192]: NoPE 128 then roped RoPE 64.
    q = (torch.randn(batch, H, 192, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    q[torch.rand(q.shape, generator=gen, device="cuda") < 0.01] = 0
    w_k = (torch.randn(H, L, P, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    return q[..., :P], q[..., P:], w_k


def _reference(q_nope, q_rope, w_k):
    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16

    from atom.model_ops.attentions.nvfp4_q_triton import quantize_q

    out = torch.empty(q_nope.shape[0], H, L, dtype=torch.bfloat16, device="cuda")
    batched_gemm_bf16(q_nope.transpose(0, 1), w_k, YQ=out.transpose(0, 1))
    return quantize_q(out, q_rope)


@pytest.mark.parametrize("backend", ["triton", "gluon", "full"])
@pytest.mark.parametrize("batch,seed", [(1, 0), (37, 1), (512, 2), (1100, 5)])
def test_kup_quantize_matches_bmm_then_quantize(batch, seed, backend):
    pytest.importorskip("aiter.ops.triton.gemm.batched.batched_gemm_bf16")
    from atom.model_ops.attentions.nvfp4_q_triton import _aiter_kup_nvfp4, kup_quantize_q_nope

    if backend == "gluon" and _aiter_kup_nvfp4() is None:
        pytest.skip("this aiter's batched_gemm_bf16 has no nvfp4_out")
    if backend == "full" and not _gfx1250():
        pytest.skip("the whole-row K-up is gfx1250 Gluon")
    q_nope, q_rope, w_k = _inputs(batch, seed)
    if seed == 5:
        q_nope[: batch // 3, :, :64] *= 50  # a wide amax range across blocks
    want_n, want_a = _reference(q_nope, q_rope, w_k)
    qa = torch.full((batch, H, 128), 0xA5, dtype=torch.uint8, device="cuda")
    qn, qa = kup_quantize_q_nope(q_nope, w_k, qa=qa, backend=backend)
    torch.cuda.synchronize()
    bad = (qn != want_n).nonzero()
    assert bad.numel() == 0, f"q_nope differs at {bad[:8].tolist()}"
    assert torch.equal(qa[..., 64:96], want_a[..., 64:96])
    # Only the NoPE scales are written.
    assert (qa[..., :64] == 0xA5).all() and (qa[..., 96:] == 0xA5).all()


@pytest.mark.parametrize("kup,hip_q", [("auto", False), ("full", False), ("full", True)])
def test_decode_prep_completes_the_rope_part(kup, hip_q):
    pytest.importorskip("aiter.ops.triton.gemm.batched.batched_gemm_bf16")
    from atom.model_ops.attentions.nvfp4_hp_ring import Nvfp4HpRing
    from atom.model_ops.attentions.nvfp4_q_triton import decode_prep, kup_quantize_q_nope

    if (kup == "full" or hip_q) and not _gfx1250():
        pytest.skip("the whole-row K-up and the HIP Q kernel are gfx1250")
    batch, pad = 40, [5, 39]
    q_nope, q_rope, w_k = _inputs(batch, 3)
    want_n, want_a = _reference(q_nope, q_rope, w_k)
    gen = torch.Generator(device="cuda").manual_seed(4)
    k = torch.randn(batch, 576, generator=gen, device="cuda").to(torch.bfloat16)
    pos = torch.randint(0, 4000, (batch,), generator=gen, device="cuda")
    slots = torch.arange(batch, device="cuda") * 64 + pos % 64
    slots[pad] = -1
    req = torch.arange(batch, dtype=torch.int32, device="cuda")
    kv_nope = torch.zeros(batch, 16384, dtype=torch.uint8, device="cuda")
    kv_aux = torch.zeros(batch, 8192, dtype=torch.uint8, device="cuda")
    ref_nope, ref_aux = kv_nope.clone(), kv_aux.clone()
    ring = Nvfp4HpRing(batch + 1, 1, 576, device="cuda")
    ref_ring = ring.buf.clone()

    qn, qa = kup_quantize_q_nope(q_nope, w_k, backend=kup)
    out = decode_prep(
        None, q_rope, ring.buf[:, 0], req, k[:, :512], k[:, 512:], slots, kv_nope, kv_aux,
        pos, q_planes=(qn, qa), hip_q=hip_q,
    )
    # The same step without the fused K-up: BF16 q_nope into decode_prep.
    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16

    bf16 = torch.empty(batch, H, L, dtype=torch.bfloat16, device="cuda")
    batched_gemm_bf16(q_nope.transpose(0, 1), w_k, YQ=bf16.transpose(0, 1))
    ref_q = decode_prep(
        bf16, q_rope, ref_ring[:, 0], req, k[:, :512], k[:, 512:], slots, ref_nope, ref_aux, pos,
        hip_q=False,
    )
    torch.cuda.synchronize()
    assert out[0] is qn and out[1] is qa
    live = torch.ones(batch, dtype=torch.bool, device="cuda")
    live[pad] = False
    assert torch.equal(qn[live], want_n[live]) and torch.equal(qa[live], want_a[live])
    assert torch.equal(qa[live], ref_q[1][live])
    # Padding rows: the NoPE scales stay as written, the rest is a zero Q.
    assert torch.equal(qa[~live][..., 64:96], want_a[~live][..., 64:96])
    assert (qa[~live][..., :64] == 0).all() and (qa[~live][..., 96:104] == 0).all()
    assert (qa[~live][..., 104:] == 127).all()
    assert torch.equal(kv_nope, ref_nope) and torch.equal(kv_aux, ref_aux)
    assert torch.equal(ring.buf[:batch], ref_ring[:batch])


@pytest.mark.parametrize("hip", [False, True])
def test_try_nvfp4_decode_with_the_fused_kup(hip, monkeypatch):
    """One decode step through `try_nvfp4_decode`, with the fused K-up and with
    the BF16 BMM in front of it: same attention output, cache and tail. With
    ``hip`` the fused step's decode prep is the HIP kernel."""
    pytest.importorskip("aiter.ops.mla_nvfp4")
    if hip and not _gfx1250():
        pytest.skip("the HIP decode prep is gfx1250")
    from types import SimpleNamespace

    from aiter.mla import mla_decode_prepare_page_csr
    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16

    from atom.model_ops.attentions.nvfp4_mla import Nvfp4MlaCache, try_nvfp4_decode

    rows, pages, dim = 6, 12, 576
    q_nope, q_rope, w_k = _inputs(rows, 7)
    gen = torch.Generator(device="cuda").manual_seed(8)
    kv_a = torch.randn(rows, 640, generator=gen, device="cuda").to(torch.bfloat16)
    k_nope, k_rope = kv_a[:, :512], kv_a[:, 512:576].view(rows, 1, 64)
    pos = torch.tensor([0, 5, 16, 31, 70, 3], device="cuda")
    tables = torch.arange(rows * 2, dtype=torch.int32, device="cuda").view(rows, 2)
    slots = (tables[:, 0].long() * 64 + pos).clone()
    slots[5] = -1  # a padding row
    lens = (pos + 1).int()
    lens[5] = 0

    def step(fused):
        cache = Nvfp4MlaCache(1, pages, rows, dim, device="cuda", triton_ring=2, triton_q=True)
        cache.kv_nope.zero_()
        cache.kv_aux.zero_()
        cache.ring.buf.zero_()
        cache.params(dim**-0.5)
        tail = torch.tensor(cache.bind_rows(list(range(rows)), pad_to=rows), dtype=torch.int32,
                            device="cuda")
        tail[5] = cache.pad_row
        csr = (
            torch.zeros(rows + 1, dtype=torch.int32, device="cuda"),
            torch.zeros(rows * 2, dtype=torch.int32, device="cuda"),
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
        )
        mla_decode_prepare_page_csr(lens, tables, *csr)
        attn = SimpleNamespace(
            nvfp4=cache, nvfp4_layer=0, num_heads=H, num_kv_heads=1, kv_lora_rank=512,
            qk_rope_head_dim=64, scale=dim**-0.5, _q_scale=None, W_K_scale=None,
            rotary_emb=lambda positions, q, k: None, _v_up_proj_and_o_proj=lambda o: o,
        )
        meta = SimpleNamespace(max_seqlen_q=1, nvfp4_tail_slots=tail, nvfp4_csr=csr,
                               slot_mapping=slots)
        if fused:
            if hip:
                monkeypatch.setenv("ATOM_MLA_NVFP4_HIP_Q", "1")
            out = try_nvfp4_decode(attn, q_nope, q_rope, k_nope, k_rope, pos, meta, w_k=w_k)
            monkeypatch.delenv("ATOM_MLA_NVFP4_HIP_Q", raising=False)
        else:
            absorbed = torch.empty(rows, H, L, dtype=torch.bfloat16, device="cuda")
            batched_gemm_bf16(q_nope.transpose(0, 1), w_k, YQ=absorbed.transpose(0, 1))
            out = try_nvfp4_decode(attn, absorbed, q_rope, k_nope, k_rope, pos, meta)
        torch.cuda.synchronize()
        return out, cache

    got, got_cache = step(True)
    want, want_cache = step(False)
    assert torch.equal(got, want)
    assert torch.equal(got_cache.kv_nope, want_cache.kv_nope)
    assert torch.equal(got_cache.kv_aux, want_cache.kv_aux)
    assert torch.equal(got_cache.ring.buf, want_cache.ring.buf)
    assert got[5].abs().sum() == 0 and got[:5].abs().sum() > 0
