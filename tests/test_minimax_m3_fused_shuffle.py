# SPDX-License-Identifier: Apache-2.0
"""GPU test for minimax_m3_fused_qknorm_rope_kv_insert_shuffle.

Verifies the SHUFFLE-cache reference reproduces the aiter fused
qknorm/rope/kv-insert semantics while writing the MAIN K/V into a page-16
SHUFFLE cache (so AITER ASM paged-attention can read it during decode).
"""

import pytest
import torch


# The repo conftest stubs atom.config for CPU scheduler tests; evict so the real
# kernels import (mirrors tests/test_minimax_m3_sparse_prefill_shuffle.py).
def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.minimax_m3.sparse_attn import (  # noqa: E402
    minimax_m3_fused_qknorm_rope_kv_insert_shuffle,
    minimax_m3_fused_qknorm_rope_kv_insert_shuffle_ref,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused MiniMax-M3 SHUFFLE tests require CUDA/ROCm",
)

HEAD_DIM = 128
ROTARY_DIM = 64


# --- reference math (verbatim from the aiter op test oracle) ---------------
def gemma_rmsnorm(x, weight, eps):
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps) * (1.0 + weight.float())


def apply_rope_neox_partial(x, positions, cos_sin_cache, rotary_dim):
    half = rotary_dim // 2
    cos_sin = cos_sin_cache[positions].float()
    cos = cos_sin[..., :half].unsqueeze(1)
    sin = cos_sin[..., half:].unsqueeze(1)
    rot = x[..., :rotary_dim]
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    out = x.clone()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x2 * cos + x1 * sin
    return out


def norm_rope_ref(x, weight, positions, cos_sin_cache, eps, dtype):
    normed = gemma_rmsnorm(x.float(), weight, eps)
    return apply_rope_neox_partial(normed, positions, cos_sin_cache, ROTARY_DIM).to(
        dtype
    )


def make_cos_sin_cache(max_pos, rotary_dim, dtype):
    base = 5_000_000.0
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda")
            / rotary_dim
        )
    )
    positions = torch.arange(max_pos, dtype=torch.float32, device="cuda")
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def test_minimax_m3_fused_qknorm_rope_kv_insert_shuffle():

    torch.manual_seed(123)
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096
    num_tokens = 17
    block_size = 16  # ASM_PAGE
    num_heads, num_kv_heads, num_index_heads = 16, 4, 4
    x = 16 // dtype.itemsize  # bf16 -> 8

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, dtype)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device="cuda"
    )

    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    iq_size = num_index_heads * HEAD_DIM
    ik_size = HEAD_DIM
    qkv = torch.randn(
        num_tokens,
        q_size + 2 * kv_size + iq_size + ik_size,
        dtype=dtype,
        device="cuda",
    )
    qkv_orig = qkv.clone()

    # page-16 physical blocks
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    # slot = page16*16 + intra; cover enough physical 16-pages.
    num_phys_blocks = num_blocks + 1
    slot_mapping = torch.randperm(
        num_phys_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens]

    # SHUFFLE caches.
    kv_cache_k = torch.zeros(
        num_phys_blocks,
        num_kv_heads,
        HEAD_DIM // x,
        block_size,
        x,
        dtype=dtype,
        device="cuda",
    )
    kv_cache_v = torch.zeros(
        num_phys_blocks,
        num_kv_heads,
        block_size // x,
        HEAD_DIM,
        x,
        dtype=dtype,
        device="cuda",
    )
    index_cache = torch.zeros(
        num_phys_blocks, block_size, HEAD_DIM, dtype=dtype, device="cuda"
    )

    q_out = torch.empty(num_tokens, q_size, dtype=dtype, device="cuda")
    index_q_out = torch.empty(num_tokens, iq_size, dtype=dtype, device="cuda")

    minimax_m3_fused_qknorm_rope_kv_insert_shuffle(
        qkv,
        q_w,
        k_w,
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        eps,
        iq_w,
        ik_w,
        num_index_heads,
        slot_mapping,
        kv_cache_k,
        kv_cache_v,
        index_cache,
        q_out,
        index_q_out,
        HEAD_DIM,
    )

    # --- references -------------------------------------------------------
    q_in, k_in, v_in, iq_in, ik_in = qkv_orig.split(
        [q_size, kv_size, kv_size, iq_size, ik_size], dim=-1
    )
    q_ref = norm_rope_ref(
        q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps, dtype
    ).view(num_tokens, q_size)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        k_w,
        positions,
        cos_sin,
        eps,
        dtype,
    )
    iq_ref = norm_rope_ref(
        iq_in.view(num_tokens, num_index_heads, HEAD_DIM),
        iq_w,
        positions,
        cos_sin,
        eps,
        dtype,
    ).view(num_tokens, iq_size)
    ik_ref = norm_rope_ref(
        ik_in.view(num_tokens, 1, HEAD_DIM), ik_w, positions, cos_sin, eps, dtype
    ).view(num_tokens, HEAD_DIM)
    v_ref = v_in.view(num_tokens, num_kv_heads, HEAD_DIM)

    # (1) q_out / index_q_out match refs.
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_q_out, iq_ref, rtol=1e-2, atol=1e-2)

    # (2)-(4) read back from SHUFFLE caches at each token's slot.
    d = torch.arange(HEAD_DIM, device="cuda")
    for token in range(num_tokens):
        slot = slot_mapping[token].item()
        phys, intra = slot // block_size, slot % block_size
        for h in range(num_kv_heads):
            k_back = kv_cache_k[phys, h, d // x, intra, d % x]
            torch.testing.assert_close(k_back, k_ref[token, h], rtol=1e-2, atol=1e-2)
            v_back = kv_cache_v[phys, h, intra // x, d, intra % x]
            torch.testing.assert_close(v_back, v_ref[token, h], rtol=0, atol=0)

        torch.testing.assert_close(
            index_cache.view(-1, HEAD_DIM)[slot], ik_ref[token], rtol=1e-2, atol=1e-2
        )


def test_fused_shuffle_triton_matches_pytorch_ref():
    """The production Triton kernel must match the pure-PyTorch reference exactly
    for all five sub-ops (q_out, index_q_out, K cache, V cache, index cache)."""
    torch.manual_seed(0)
    dev = "cuda"
    dt = torch.bfloat16
    NH, NKV, NIQ = 16, 1, 4
    HD, IDX, x, rot, eps = 128, 128, 8, 64, 1e-6
    nt, nblk = 37, 6
    P16, G = 16, 8
    nphys = nblk * G

    inv = 1.0 / (5e6 ** (torch.arange(0, rot, 2, device=dev).float() / rot))
    t = torch.arange(4096, device=dev).float()
    f = torch.outer(t, inv)
    csc = torch.cat((f.cos(), f.sin()), dim=-1).to(dt)

    qw = torch.randn(HD, device=dev, dtype=dt) * 0.1
    kw = torch.randn(HD, device=dev, dtype=dt) * 0.1
    iqw = torch.randn(IDX, device=dev, dtype=dt) * 0.1
    ikw = torch.randn(IDX, device=dev, dtype=dt) * 0.1
    pos = torch.randint(0, 4096, (nt,), dtype=torch.int64, device=dev)
    qsz, kvsz, iqsz, iksz = NH * HD, NKV * HD, NIQ * IDX, IDX
    qkv = torch.randn(nt, qsz + 2 * kvsz + iqsz + iksz, dtype=dt, device=dev)
    slot = torch.randperm(nblk * 128, device=dev)[:nt].to(torch.int64)

    def run(fn):
        kc = torch.zeros(nphys, NKV, HD // x, P16, x, dtype=dt, device=dev)
        vc = torch.zeros(nphys, NKV, P16 // x, HD, x, dtype=dt, device=dev)
        idxc = torch.zeros(nblk * 128, IDX, dtype=dt, device=dev)
        qo = torch.empty(nt, qsz, dtype=dt, device=dev)
        iqo = torch.empty(nt, iqsz, dtype=dt, device=dev)
        fn(
            qkv,
            qw,
            kw,
            csc,
            pos,
            NH,
            NKV,
            rot,
            eps,
            iqw,
            ikw,
            NIQ,
            slot,
            kc,
            vc,
            idxc,
            qo,
            iqo,
            IDX,
        )
        return qo, iqo, kc, vc, idxc

    r = run(minimax_m3_fused_qknorm_rope_kv_insert_shuffle_ref)
    tr = run(minimax_m3_fused_qknorm_rope_kv_insert_shuffle)
    names = ["q_out", "index_q_out", "kv_cache_k", "kv_cache_v", "index_cache"]
    for name, a, b in zip(names, r, tr):
        torch.testing.assert_close(
            b.float(), a.float(), rtol=2e-2, atol=2e-2, msg=f"{name} mismatch"
        )
