# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""RoPE inside the Triton Q and cache kernels matches the separate path
(copies + the model's rotary embedding + concat). The reference is AITER's
Triton RoPE (``forward_triton``, the backend under
``AITER_ROPE_TRITON_BACKEND=1``, as the DeepSeek-R1 MLPerf config sets).

NoPE bytes and the layout match exactly. A roped element may land one BF16
ulp from AITER's, depending on how the Triton build lowers AITER's kernel (see
`_rope_gptj`), so the bytes holding RoPE codes and scales are compared by the
share that matches. With already-roped inputs every byte matches."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# DeepSeek-R1's YaRN settings.
YARN = {
    "type": "yarn",
    "factor": 40,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


def _rope():
    from aiter.rotary_embedding import get_rope

    rope = get_rope(
        64,
        rotary_dim=64,
        max_position=163840,
        base=10000,
        is_neox_style=False,
        rope_scaling=YARN,
        dtype=torch.bfloat16,
    )
    return rope.to("cuda")


# One BF16 ulp below 8, which the roped K values of these inputs stay under.
ROPE_ATOL = 2.0**-5
# One ulp moves some RoPE codes and scales. On these inputs about 96.8% of the
# q_aux bytes and 99.99% of the kv_aux bytes still match.
Q_AUX_MATCH = 0.96
KV_AUX_MATCH = 0.9998


def _fraction_equal(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a == b).float().mean().item()


def _assert_rope_close(got: torch.Tensor, want: torch.Tensor) -> None:
    err = (got.float() - want.float()).abs().max().item()
    assert err <= ROPE_ATOL, f"roped K is {err} from the reference"


@pytest.mark.parametrize("batch,seed", [(1, 0), (37, 1), (300, 2), (1024, 3)])
def test_fused_rope_matches_the_separate_path(batch, seed):
    pytest.importorskip("aiter.ops.mla_nvfp4")
    from atom.model_ops.attentions.nvfp4_hp_ring import Nvfp4HpRing
    from atom.model_ops.attentions.nvfp4_q_triton import quantize_q
    from atom.model_ops.attentions.nvfp4_ring_triton import ring_append_quantize

    rope = _rope()
    assert rope.cos_cache.dtype == torch.bfloat16 and rope.cos_cache.shape[-1] == 32
    cos, sin = rope.cos_cache, rope.sin_cache
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, s=0.3):
        return (torch.randn(shape, generator=gen, device="cuda") * s).to(torch.bfloat16)

    # Q as it leaves the projection: NoPE [B, 128, 512] and a RoPE view of a wider row.
    q_nope = randn(batch, 128, 512)
    q_wide = randn(batch, 128, 192, s=2.0)
    q_rope = q_wide[..., 128:]
    # K as it leaves kv_a: NoPE [B, 512] and RoPE [B, 1, 64] views of one row.
    kv_a = randn(batch, 576 + 64, s=1.0)
    k_nope = kv_a[:, :512]
    k_rope = kv_a[:, 512:576].view(batch, 1, 64)
    q_rope_in, k_rope_in = q_rope.clone(), k_rope.clone()
    positions = torch.randint(0, 160000, (batch,), generator=gen, device="cuda")
    pages = 4 * batch
    req = torch.arange(batch, dtype=torch.int32, device="cuda")
    slots = torch.arange(batch, device="cuda") * 4 * 64 + positions % 64

    ring = Nvfp4HpRing(batch + 1, 1, 576, device="cuda")
    ring.buf.copy_(randn(*ring.buf.shape))
    nope = torch.randint(
        0, 256, (pages, 16384), dtype=torch.uint8, device="cuda", generator=gen
    )
    aux = torch.randint(
        0, 256, (pages, 8192), dtype=torch.uint8, device="cuda", generator=gen
    )
    r_ring, r_nope, r_aux = ring.buf.clone(), nope.clone(), aux.clone()
    start_ring, start_nope, start_aux = ring.buf.clone(), nope.clone(), aux.clone()

    # Separate path, as try_nvfp4_decode runs it without fused RoPE.
    q_pe, k_pe = q_rope.clone(), k_rope.clone()
    rope.forward_triton(positions, q_pe, k_pe)
    want_q = quantize_q(q_nope, q_pe)
    kv = torch.cat(
        (k_nope.reshape(batch, 512), k_pe.reshape(batch, 64)), dim=-1
    ).contiguous()
    ring_append_quantize(r_ring[:, 0], req, positions, kv, slots, r_nope, r_aux)

    # Fused path.
    got_q = quantize_q(q_nope, q_rope, positions, cos, sin)
    ring_append_quantize(
        ring.buf[:, 0],
        req,
        positions,
        k_nope,
        slots,
        nope,
        aux,
        rope=k_rope.reshape(batch, 64),
        cos=cos,
        sin=sin,
    )
    torch.cuda.synchronize()

    new = ring.buf[req.long(), 0, positions % 16]
    ref_tail = r_ring[req.long(), 0, positions % 16]
    assert torch.equal(new[:, :512], ref_tail[:, :512]), "tail NoPE differs"
    _assert_rope_close(new[:, 512:], k_pe.reshape(batch, 64))
    assert torch.equal(got_q[0], want_q[0]), "q_nope bytes differ"
    assert _fraction_equal(got_q[1], want_q[1]) >= Q_AUX_MATCH, "q_aux differs"
    assert torch.equal(nope, r_nope), "kv_nope differs"
    assert _fraction_equal(aux, r_aux) >= KV_AUX_MATCH, "kv_aux differs"
    # The inputs were not rotated in place.
    assert torch.equal(q_rope, q_rope_in) and torch.equal(k_rope, k_rope_in)

    # One launch for Q and K, RoPE in the kernel, then with roped inputs.
    from atom.model_ops.attentions.nvfp4_q_triton import decode_prep

    for fused in (True, False):
        ring.buf.copy_(start_ring)
        nope.copy_(start_nope)
        aux.copy_(start_aux)
        if fused:
            one_q = decode_prep(
                q_nope,
                q_rope,
                ring.buf[:, 0],
                req,
                k_nope,
                k_rope.reshape(batch, 64),
                slots,
                nope,
                aux,
                positions,
                cos,
                sin,
            )
        else:
            one_q = decode_prep(
                q_nope,
                q_pe,
                ring.buf[:, 0],
                req,
                k_nope,
                k_pe.reshape(batch, 64),
                slots,
                nope,
                aux,
                positions,
            )
        torch.cuda.synchronize()
        new = ring.buf[req.long(), 0, positions % 16]
        assert torch.equal(one_q[0], want_q[0]), fused
        assert torch.equal(new[:, :512], ref_tail[:, :512]), fused
        assert torch.equal(nope, r_nope), fused
        if fused:
            _assert_rope_close(new[:, 512:], k_pe.reshape(batch, 64))
            assert _fraction_equal(one_q[1], want_q[1]) >= Q_AUX_MATCH, fused
            assert _fraction_equal(aux, r_aux) >= KV_AUX_MATCH, fused
            assert torch.equal(q_rope, q_rope_in) and torch.equal(k_rope, k_rope_in)
        else:
            assert torch.equal(one_q[1], want_q[1]), fused
            assert torch.equal(ring.buf[:batch], r_ring[:batch]), fused
            assert torch.equal(aux, r_aux), fused
