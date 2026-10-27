# SPDX-License-Identifier: MIT
"""`SparseMHAPagedAttentionImpl.rope_cache` with the MXFP4 caches (gfx950).

The fused aiter kernel quantizes K, V and the index key into 8704 B pages and
emits an e4m3 main q. This pins the plumbing around it against the same kernel
writing bf16: what lands in the pool pages, at which slots, and the q handed
on. The kernel's own numerics are aiter's tests'.
"""

import types

import pytest
import torch

from atom.model_ops.minimax_m3.kv_mxfp4 import PAGE_BYTES, dequantize_pages

aiter = pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter, "fused_qknorm_idxrqknorm"),
    reason="needs gfx950 and aiter.fused_qknorm_idxrqknorm",
)

NQ, NKV, D, NIQ, PAGES = 16, 1, 128, 1, 6


def make_impl(kv_mxfp4: bool, index_mxfp4: bool):
    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl

    dev = "cuda"
    impl = object.__new__(SparseMHAPagedAttentionImpl)
    torch.nn.Module.__init__(impl)
    g = torch.Generator(device=dev).manual_seed(0)

    def norm(n):
        w = torch.rand(n, device=dev, generator=g, dtype=torch.float32) + 0.5
        return types.SimpleNamespace(weight=w.to(torch.bfloat16), variance_epsilon=1e-6)

    rope = torch.nn.Module()
    pos = torch.arange(4096, device=dev, dtype=torch.float32)[:, None]
    inv = 1e7 ** (-torch.arange(0, 64, 2, device=dev, dtype=torch.float32) / 64)
    ang = pos * inv[None]
    rope.register_buffer("cos_cache", ang.cos().to(torch.bfloat16)[:, None, None, :])
    rope.register_buffer("sin_cache", ang.sin().to(torch.bfloat16)[:, None, None, :])
    rope.rotary_dim = 64
    impl.rotary_emb = rope
    impl.q_norm, impl.k_norm = norm(D), norm(D)
    impl.index_q_norm, impl.index_k_norm = norm(D), norm(D)
    impl.num_heads, impl.num_kv_heads, impl.head_dim = NQ, NKV, D
    impl.num_idx_heads, impl.index_q_size, impl.index_head_dim = NIQ, NIQ * D, D
    impl.skip_index_topk = False
    impl.kv_cache_dtype = "fp8"
    impl.index_cache_dtype = "mxfp4" if index_mxfp4 else "auto"
    impl.kv_nvfp4, impl.kv_mxfp4, impl.index_mxfp4 = False, kv_mxfp4, index_mxfp4
    impl.index_scale = impl.index_stage = impl.index_stage_slots = None
    impl.kv_stage = impl.kv_stage_slots = None
    impl._out_dtype = torch.bfloat16
    impl._index_q = impl._index_q_cache_key_info = None
    return impl


def run(impl, qkv, position, slots, k_cache, v_cache, index_cache):
    impl.index_cache = index_cache
    ctx = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(slot_mapping=slots),
        kv_cache_data={
            "layer_0": types.SimpleNamespace(
                k_cache=k_cache, v_cache=v_cache, k_scale=None, v_scale=None
            )
        },
    )
    impl.layer_num = 0
    return impl.rope_cache(None, None, None, qkv, position, ctx)


def test_mxfp4_pages_match_the_bf16_write():
    T = 37
    g = torch.Generator(device="cuda").manual_seed(1)
    width = (NQ + 2 * NKV + NIQ + 1) * D
    qkv = torch.randn(T, width, device="cuda", generator=g, dtype=torch.bfloat16)
    position = torch.arange(T, device="cuda", dtype=torch.int64) + 5
    # tokens land on scattered slots of 3 of the 6 pages; slot -1 is padding
    slots = torch.randperm(PAGES * 128, device="cuda", generator=g)[:T].to(torch.int64)
    slots[4] = -1

    # reference: the same kernel writing bf16 (plain page-128 layout); `ref` only
    # carries the norm weights and rotary tables
    ref = make_impl(False, False)
    rk = torch.zeros(PAGES, 128, NKV, D, device="cuda", dtype=torch.bfloat16)
    rv = torch.zeros_like(rk)
    ri = torch.zeros(PAGES, 128, D, device="cuda", dtype=torch.bfloat16)
    from atom.models.minimax_m3.model import _minimax_m3_cos_sin_cache

    cs = _minimax_m3_cos_sin_cache(ref.rotary_emb, qkv)
    q_bf16 = torch.empty(T, NQ * D, device="cuda", dtype=torch.bfloat16)
    iq_bf16 = torch.empty(T, NIQ * D, device="cuda", dtype=torch.bfloat16)
    aiter.fused_qknorm_idxrqknorm(
        qkv, ref.q_norm.weight, ref.k_norm.weight, cs, position, NQ, NKV, 64, 1e-6,
        ref.index_q_norm.weight, ref.index_k_norm.weight, NIQ, slots, rk, rv, ri,
        128, q_bf16, iq_bf16, slots, kv_cache_dtype="auto", asm_layout=False,
    )  # fmt: skip

    impl = make_impl(True, True)
    kp = torch.zeros(PAGES, NKV, PAGE_BYTES, device="cuda", dtype=torch.uint8)
    vp = torch.zeros_like(kp)
    ip = torch.zeros(PAGES, PAGE_BYTES, device="cuda", dtype=torch.uint8)
    q, _, _, kc, vc, ks, vs = run(impl, qkv, position, slots, kp, vp, ip)

    assert kc is kp and vc is vp and ks is None and vs is None
    assert q.dtype == torch.float8_e4m3fn and q.shape == (T, NQ, D)
    assert impl._index_q.dtype == torch.bfloat16
    torch.testing.assert_close(
        impl._index_q.float(), iq_bf16.view(T, NIQ, D).float(), rtol=0, atol=0
    )
    qd = q.float().view(T, NQ * D)
    assert (qd - q_bf16.float()).norm() / q_bf16.float().norm() < 0.04  # e4m3 cast

    valid = slots >= 0
    page, off = slots[valid] // 128, slots[valid] % 128

    def rel(got, want):
        return ((got - want).norm() / want.norm()).item()

    for pages_u8, want in ((kp, rk), (vp, rv)):
        got = dequantize_pages(pages_u8)[page, 0, off]  # [n, 128]
        assert rel(got, want[page, off, 0].float()) < 0.15
    got = dequantize_pages(ip)[page, off]
    assert rel(got, ri[page, off].float()) < 0.15
    # padding and untouched slots stay zero-bytes
    written = torch.zeros(PAGES * 128, dtype=torch.bool, device="cuda")
    written[slots[valid]] = True
    untouched = dequantize_pages(kp)[:, 0].reshape(-1, D)[~written]
    assert (untouched == 0).all()
