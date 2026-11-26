# SPDX-License-Identifier: MIT
"""`SparseMHAPagedAttentionImpl.rope_cache` with the MXFP4 caches (gfx950), against
the same aiter kernel writing bf16: what lands in the pages, at which slots, and the
q handed on. The kernel's own numerics are aiter's tests'."""

import types

import pytest
import torch

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, page_views
from atom.mono.weights.formats import dequantize_mxfp4

aiter = pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter, "fused_qknorm_idxrqknorm"),
    reason="needs gfx950 and aiter.fused_qknorm_idxrqknorm",
)

NQ, NKV, D, NIQ, PAGES = 16, 1, 128, 1, 6


DEV = "cuda"
WIDTH = (NQ + 2 * NKV + NIQ + 1) * D
BF16 = {"device": DEV, "dtype": torch.bfloat16}


def make_impl(kv_mxfp4: bool, index_cache_dtype: str):
    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl

    impl = object.__new__(SparseMHAPagedAttentionImpl)
    torch.nn.Module.__init__(impl)
    g = torch.Generator(device=DEV).manual_seed(0)

    def norm(n):
        w = torch.rand(n, device=DEV, generator=g) + 0.5
        return types.SimpleNamespace(weight=w.to(torch.bfloat16), variance_epsilon=1e-6)

    inv = 1e7 ** (-torch.arange(0, 64, 2, device=DEV, dtype=torch.float32) / 64)
    ang = torch.arange(4096, device=DEV, dtype=torch.float32)[:, None] * inv[None]
    rope = torch.nn.Module()
    rope.register_buffer("cos_cache", ang.cos().to(torch.bfloat16)[:, None, None, :])
    rope.register_buffer("sin_cache", ang.sin().to(torch.bfloat16)[:, None, None, :])
    rope.rotary_dim = 64
    impl.rotary_emb = rope
    impl.q_norm, impl.k_norm = norm(D), norm(D)
    impl.index_q_norm, impl.index_k_norm = norm(D), norm(D)
    impl.num_heads, impl.num_kv_heads, impl.head_dim = NQ, NKV, D
    impl.num_idx_heads, impl.index_q_size, impl.index_head_dim = NIQ, NIQ * D, D
    impl.skip_index_topk = False
    impl.kv_cache_dtype = "fp4" if kv_mxfp4 else "fp8"
    impl.index_cache_dtype = index_cache_dtype
    impl.index_scale = impl.index_stage = impl.index_stage_slots = None
    impl.kv_stage = impl.kv_stage_slots = None
    impl._index_q = impl._index_q_cache_key_info = None
    return impl


def run(impl, qkv, position, slots, k, v, index):
    impl.index_cache, impl.layer_num = index, 0
    layer = types.SimpleNamespace(k_cache=k, v_cache=v, k_scale=None, v_scale=None)
    ctx = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(slot_mapping=slots),
        kv_cache_data={"layer_0": layer},
    )
    return impl.rope_cache(None, None, None, qkv, position, ctx)


def qkv_of(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(T, WIDTH, generator=g, **BF16)


def pages(*lead, dtype=torch.uint8):
    return torch.zeros(PAGES, *lead, PAGE_BYTES, device=DEV, dtype=dtype)


def reference(qkv, position, slots):
    """The same kernel writing bf16 (plain page-128 layout): rk, rv, ri, index q."""
    from atom.models.minimax_m3.model import _minimax_m3_cos_sin_cache

    ref = make_impl(False, "auto")  # only carries the norm weights and rotary tables
    rk = torch.zeros(PAGES, 128, NKV, D, **BF16)
    rv, ri = torch.zeros_like(rk), torch.zeros(PAGES, 128, D, **BF16)
    cs = _minimax_m3_cos_sin_cache(ref.rotary_emb, qkv)
    q = torch.empty(len(qkv), NQ * D, **BF16)
    iq = torch.empty(len(qkv), NIQ * D, **BF16)
    aiter.fused_qknorm_idxrqknorm(
        qkv, ref.q_norm.weight, ref.k_norm.weight, cs, position, NQ, NKV, 64, 1e-6,
        ref.index_q_norm.weight, ref.index_k_norm.weight, NIQ, slots, rk, rv, ri,
        128, q, iq, slots, kv_cache_dtype="auto", asm_layout=False,
    )  # fmt: skip
    return rk, rv, ri, iq


def rel(got, want):
    return ((got - want).norm() / want.norm()).item()


@pytest.mark.parametrize("continued", [False, True])  # a cached prefix fills page 0
@pytest.mark.parametrize("index_dtype", ["fp4", "fp8"])  # KV + index, KV alone
def test_mxfp4_kv_pages_match_the_bf16_write(index_dtype, continued):
    T = 37
    qkv, position = qkv_of(T, 1), torch.arange(T, device=DEV) + 5
    slots = torch.randperm(PAGES * 128, device=DEV)[:T]  # scattered over the pages
    if continued:  # the cached prefix fills tokens 0..99; this chunk starts at 100
        slots = torch.arange(100, 100 + T, device=DEV)
    slots[4] = -1  # padding
    rk, rv, ri, iq_ref = reference(qkv, position, slots)

    impl = make_impl(True, index_dtype)
    kp, vp = pages(NKV), pages(NKV)
    ip = pages() if index_dtype == "fp4" else torch.zeros(PAGES, 128, D, device=DEV, dtype=aiter.dtypes.fp8)  # fmt: skip
    prefix = torch.arange(100, device=DEV)
    if continued:
        run(
            make_impl(True, index_dtype), qkv_of(100, 5), prefix + 5, prefix, kp, vp, ip
        )
    data0, scale0 = (t.clone() for t in page_views(kp))
    q, _, _, kc, vc, _, _ = run(impl, qkv, position, slots, kp, vp, ip)

    assert kc is kp and vc is vp
    assert q.dtype == torch.float8_e4m3fn and q.shape == (T, NQ, D)
    want_dtype = torch.float8_e4m3fn if index_dtype == "fp4" else torch.bfloat16
    assert impl._index_q.dtype == want_dtype  # only the MXFP4 index takes an fp8 q
    tol = 0.04 if index_dtype == "fp4" else 0.02
    assert rel(impl._index_q.float(), iq_ref.view(T, NIQ, D).float()) < tol
    ok = slots >= 0
    page, off = slots[ok] // 128, slots[ok] % 128
    for got_pages, want in ((kp, rk), (vp, rv)):
        got = dequantize_mxfp4(*page_views(got_pages))[page, 0, off]
        assert rel(got, want[page, off, 0].float()) < 0.15
    index_rows = ip.float() if index_dtype == "fp8" else dequantize_mxfp4(*page_views(ip))  # fmt: skip
    assert rel(index_rows[page, off], ri[page, off].float()) < 0.15
    # padding and untouched slots stay zero bytes; the prefix survives the write
    written = torch.zeros(PAGES * 128, dtype=torch.bool, device=DEV)
    written[slots[ok]] = True
    if continued:
        written[prefix] = True
    assert (dequantize_mxfp4(*page_views(kp))[:, 0].reshape(-1, D)[~written] == 0).all()
    if continued:  # the cached prefix's bytes survive the second write
        data1, scale1 = page_views(kp)
        assert torch.equal(data1[0, 0, :100], data0[0, 0, :100])
        assert torch.equal(scale1[0, 0, :100], scale0[0, 0, :100])


def test_a_top_k_reusing_layer_writes_kv_and_skips_the_index():
    T = 21
    qkv, position = qkv_of(T, 2), torch.arange(T, device=DEV)
    slots = position + 128  # page 1
    full, skip = make_impl(True, "fp4"), make_impl(True, "fp4")
    skip.skip_index_topk = True
    kp, vp, ip = pages(NKV), pages(NKV), pages()
    kp2, vp2, ip2 = (torch.zeros_like(t) for t in (kp, vp, ip))
    q_full = run(full, qkv, position, slots, kp, vp, ip)[0]
    q_skip = run(skip, qkv, position, slots, kp2, vp2, ip2)[0]

    assert skip._index_q is None and skip._index_q_cache_key_info is not None
    assert torch.equal(kp, kp2) and torch.equal(vp, vp2)  # the same K/V pages
    assert torch.equal(q_full.view(torch.uint8), q_skip.view(torch.uint8))
    assert (ip2 == 0).all() and (ip != 0).any()  # no index key written
    # the key a reusing layer looks the producer's selection up by
    bt = torch.zeros(2, 4, dtype=torch.int32, device=DEV)
    for impl in (full, skip):
        impl.topk, impl.init_blocks, impl.local_blocks = 16, 1, 1
        impl.indexer_cp_group = None
    args = (bt, bt[:, 0], 1, 512)
    assert skip._topk_cache_key("decode", None, *args) == full._topk_cache_key(
        "decode", full._index_q, *args
    )
