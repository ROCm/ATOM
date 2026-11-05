# SPDX-License-Identifier: MIT
"""`SparseMHAPagedAttentionImpl.rope_cache` with MXFP4 caches, and what the sparse
dispatch hands the selector and the reader.

GPU (gfx950): the fused aiter kernel quantizes the index key into 8704 B pages and
emits an e4m3 index q (MXFP4 index), or writes K/V as 8704 B pages per (page, kv head)
and an e4m3 main q (``--kv-cache-dtype fp4``). Those cases pin the plumbing around it
against the same kernel writing bf16: what lands in the pages, at which slots (a
chunk continuing a cached prefix included), and the q handed on. The kernel's own
numerics are aiter's tests'.

Without a GPU: the kernel is replaced by a recorder, which pins what the layer asks
of it -- the call is the fp4-KV one (page-128, plain layout, e4m3 q, no scale planes),
independent of the index dtype, a top-k reusing layer is refused, and prefill and
decode run the MXFP4 reader on a one-page-a-block table, not the page-16 ASM path.
These need `atom.model_ops.attention_mha` importable (aiter, triton), as the module
does; the CPU CI job skips them.
"""

import sys
import types

import pytest
import torch

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, page_views
from atom.mono.weights.formats import dequantize_mxfp4

try:
    import aiter
except ModuleNotFoundError:  # the CPU CI job: nothing here runs
    aiter = None


def _has_fp4_index_ops():
    """The MXFP4 index scorers and writer mode ship together; an older aiter has
    neither, and its writer aborts the process on an `mxfp4` index cache."""
    try:
        from aiter.ops.flydsl import minimax_m3_index_score_fp4_flydsl  # noqa: F401
    except ImportError:
        return False
    return True


def _mha():
    """`attention_mha`, or a skip where aiter / triton are absent."""
    pytest.importorskip("aiter")
    pytest.importorskip("triton")
    from atom.model_ops import attention_mha

    return attention_mha


gpu_gfx950 = pytest.mark.skipif(
    aiter is None
    or not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter, "fused_qknorm_idxrqknorm")
    or not _has_fp4_index_ops(),
    reason="needs gfx950 and aiter.fused_qknorm_idxrqknorm with the MXFP4 index",
)

NQ, NKV, D, NIQ, PAGES = 16, 1, 128, 1, 6


def make_impl(index_cache_dtype: str, kv_cache_dtype: str = "fp8", dev: str = "cuda"):
    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl

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
    impl.kv_cache_dtype = kv_cache_dtype
    impl.index_cache_dtype = index_cache_dtype
    impl.index_scale = impl.index_stage = impl.index_stage_slots = None
    impl.kv_stage = impl.kv_stage_slots = None
    impl._index_q = impl._index_q_cache_key_info = None
    return impl


def run(
    impl,
    qkv,
    position,
    slots,
    k_cache,
    v_cache,
    index_cache,
    k_scale=None,
    v_scale=None,
):
    impl.index_cache = index_cache
    ctx = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(slot_mapping=slots),
        kv_cache_data={
            "layer_0": types.SimpleNamespace(
                k_cache=k_cache, v_cache=v_cache, k_scale=k_scale, v_scale=v_scale
            )
        },
    )
    impl.layer_num = 0
    return impl.rope_cache(None, None, None, qkv, position, ctx)


def inputs(T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    width = (NQ + 2 * NKV + NIQ + 1) * D
    qkv = torch.randn(T, width, device="cuda", generator=g, dtype=torch.bfloat16)
    position = torch.arange(T, device="cuda", dtype=torch.int64) + 5
    # tokens land on scattered slots of 3 of the 6 pages; slot -1 is padding
    slots = torch.randperm(PAGES * 128, device="cuda", generator=g)[:T].to(torch.int64)
    slots[4] = -1
    return qkv, position, slots


def reference(qkv, position, slots):
    """The same kernel writing bf16 (plain page-128 layout): rk, rv, ri, q, index q."""
    T = qkv.shape[0]
    ref = make_impl("auto")  # only carries the norm weights and rotary tables
    rk = torch.zeros(PAGES, 128, NKV, D, device="cuda", dtype=torch.bfloat16)
    rv = torch.zeros_like(rk)
    ri = torch.zeros(PAGES, 128, D, device="cuda", dtype=torch.bfloat16)
    from atom.models.minimax_m3.model import _minimax_m3_cos_sin_cache

    cs = _minimax_m3_cos_sin_cache(ref.rotary_emb, qkv)
    q = torch.empty(T, NQ * D, device="cuda", dtype=torch.bfloat16)
    iq = torch.empty(T, NIQ * D, device="cuda", dtype=torch.bfloat16)
    aiter.fused_qknorm_idxrqknorm(
        qkv, ref.q_norm.weight, ref.k_norm.weight, cs, position, NQ, NKV, 64, 1e-6,
        ref.index_q_norm.weight, ref.index_k_norm.weight, NIQ, slots, rk, rv, ri,
        128, q, iq, slots, kv_cache_dtype="auto", asm_layout=False,
    )  # fmt: skip
    return rk, rv, ri, q, iq


def rel(got, want):
    return ((got - want).norm() / want.norm()).item()


def shuffle_cache():
    # the page-128 SHUFFLE views (x = 16 fp8 elements) and per-token scales
    k = torch.zeros(PAGES, NKV, D // 16, 128, 16, device="cuda", dtype=aiter.dtypes.fp8)
    v = torch.zeros(PAGES, NKV, 128 // 16, D, 16, device="cuda", dtype=aiter.dtypes.fp8)
    scale = torch.zeros(PAGES, NKV, 128, device="cuda", dtype=torch.float32)
    return k, v, scale, torch.zeros_like(scale)


@gpu_gfx950
def test_index_only_mxfp4_leaves_the_fp8_shuffle_kv_write_alone():
    T = 37
    qkv, position, slots = inputs(T, 4)
    _, _, ri, _, iq_bf16 = reference(qkv, position, slots)

    runs = {}
    for name, index_dtype in (("mxfp4", "fp4"), ("none", "auto")):
        impl = make_impl(index_dtype)
        k, v, ksc, vsc = shuffle_cache()
        index = (
            torch.zeros(PAGES, PAGE_BYTES, device="cuda", dtype=torch.uint8)
            if index_dtype == "fp4"
            else torch.zeros(PAGES, 128, D, device="cuda", dtype=torch.bfloat16)
        )
        q = run(impl, qkv, position, slots, k, v, index, ksc, vsc)[0]
        runs[name] = (impl, q, k, v, ksc, vsc, index)

    impl, q, k, v, ksc, vsc, index = runs["mxfp4"]
    _, q0, k0, v0, ksc0, vsc0, _ = runs["none"]
    assert q.dtype == torch.bfloat16  # the main q stays as it was
    for got, want in ((q, q0), (k, k0), (v, v0), (ksc, ksc0), (vsc, vsc0)):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    assert (ksc != 0).any() and (k.view(torch.uint8) != 0).any()
    assert impl._index_q.dtype == torch.float8_e4m3fn
    assert rel(impl._index_q.float(), iq_bf16.view(T, NIQ, D).float()) < 0.04
    valid = slots >= 0
    page, off = slots[valid] // 128, slots[valid] % 128
    assert (
        rel(dequantize_mxfp4(*page_views(index))[page, off], ri[page, off].float())
        < 0.15
    )


@gpu_gfx950
def test_a_top_k_reusing_layer_keys_its_reuse_on_the_e4m3_index_q():
    T = 21
    qkv, position, _ = inputs(T, 2)
    slots = torch.arange(T, device="cuda", dtype=torch.int64) + 128  # page 1

    full = make_impl("fp4")
    k, v, ksc, vsc = shuffle_cache()
    ip = torch.zeros(PAGES, PAGE_BYTES, device="cuda", dtype=torch.uint8)
    run(full, qkv, position, slots, k, v, ip, ksc, vsc)

    skip = make_impl("fp4")
    skip.skip_index_topk = True
    k2, v2, ksc2, vsc2 = shuffle_cache()
    ip2 = torch.zeros_like(ip)
    run(skip, qkv, position, slots, k2, v2, ip2, ksc2, vsc2)

    assert skip._index_q is None and skip._index_q_cache_key_info is not None
    assert (ip2 == 0).all() and (ip != 0).any()  # no index key written
    # the key a reusing layer looks the producer's selection up by
    bt = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    for impl in (full, skip):
        impl.topk, impl.init_blocks, impl.local_blocks = 16, 1, 1
        impl.indexer_cp_group = None
    args = (bt, bt[:, 0], 1, 512)
    assert full._index_q.dtype == torch.float8_e4m3fn
    assert skip._topk_cache_key("decode", None, *args) == full._topk_cache_key(
        "decode", full._index_q, *args
    )


# --- fp4 KV: the writer ------------------------------------------------------------


def fp4_pool():
    return torch.zeros(PAGES, NKV, PAGE_BYTES, device="cuda", dtype=torch.uint8)


def plain_index():
    return torch.zeros(PAGES, 128, D, device="cuda", dtype=torch.bfloat16)


def deq(pool):
    return dequantize_mxfp4(*page_views(pool))  # [PAGES, NKV, 128, D]


@gpu_gfx950
def test_fp4_kv_pages_hold_the_tokens_at_their_slots():
    """Not run by the CPU gate. K/V of tokens on scattered slots (one a padding -1)
    land in their (page, token) of the 8704 B pages; main q is e4m3, no scale planes."""
    T = 37
    qkv, position, slots = inputs(T, 6)
    rk, rv, _, rq, _ = reference(qkv, position, slots)
    impl = make_impl("auto", "fp4")
    k, v = fp4_pool(), fp4_pool()
    q, _, _, kc, vc, ks, vs = run(impl, qkv, position, slots, k, v, plain_index())
    assert q.dtype == torch.float8_e4m3fn and ks is None and vs is None
    assert kc is k and vc is v
    assert rel(q.float().view(T, -1), rq.float()) < 0.1
    valid = slots >= 0
    page, off = slots[valid] // 128, slots[valid] % 128
    for pool, ref in ((k, rk), (v, rv)):
        got = deq(pool)[page, 0, off]
        assert rel(got, ref[page, off, 0].float()) < 0.15
    # nothing outside the written slots changed: the other tokens' bytes stay zero
    written = torch.zeros(PAGES, 128, dtype=torch.bool, device="cuda")
    written[page, off] = True
    assert (page_views(k)[0][~written].view(-1) == 0).all()


@gpu_gfx950
def test_fp4_kv_second_chunk_continues_a_page_without_touching_the_prefix():
    """Not run by the CPU gate. Chunked prefill / a cached prefix: the first chunk
    fills page 0 up to token 99; the next chunk starts in page 0 at token 100, so
    its page is a partly written one. The earlier tokens' data and scale bytes must
    survive the second write, and the page then reads as one."""
    qkv, position, _ = inputs(71, 9)
    first = torch.arange(100, device="cuda", dtype=torch.int64)
    second = torch.arange(100, 171, device="cuda", dtype=torch.int64)  # page 0, then 1
    rk, _, _, _, _ = reference(qkv, position, second)
    impl = make_impl("auto", "fp4")
    k, v = fp4_pool(), fp4_pool()
    qkv1, pos1 = inputs(100, 5)[:2]
    run(impl, qkv1, pos1, first, k, v, plain_index())
    data0, scale0 = (t.clone() for t in page_views(k))
    run(impl, qkv, position, second, k, v, plain_index())
    data1, scale1 = page_views(k)
    assert torch.equal(data1[0, 0, :100], data0[0, 0, :100])
    assert torch.equal(scale1[0, 0, :100], scale0[0, 0, :100])
    page, off = second // 128, second % 128
    assert rel(deq(k)[page, 0, off], rk[page, off, 0].float()) < 0.15


# --- no GPU: what the layer asks of the kernel and of the reader --------------------


class Recorder:
    """Stands in for aiter.fused_qknorm_idxrqknorm."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@pytest.fixture
def cpu_layer(monkeypatch):
    mha = _mha()
    rec = Recorder()
    monkeypatch.setattr(mha.aiter, "fused_qknorm_idxrqknorm", rec)
    stub = types.ModuleType("atom.models.minimax_m3.model")
    stub._minimax_m3_cos_sin_cache = lambda rope, qkv: torch.zeros(1)
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.model", stub)
    return mha, rec


def cpu_inputs(T=9):
    width = (NQ + 2 * NKV + NIQ + 1) * D
    qkv = torch.zeros(T, width, dtype=torch.bfloat16)
    return qkv, torch.arange(T), torch.arange(T, dtype=torch.int64)


def cpu_run(impl, k, v, index):
    qkv, position, slots = cpu_inputs()
    return run(impl, qkv, position, slots, k, v, index)


def cpu_pool():
    return torch.zeros(PAGES, NKV, PAGE_BYTES, dtype=torch.uint8)


@pytest.mark.parametrize("index_dtype", ["fp8", "fp4", "auto"])
def test_fp4_kv_asks_the_kernel_for_page_128_mxfp4_pages(cpu_layer, index_dtype):
    """Whatever the index dtype: K/V are the pool's own pages (not re-viewed as
    page-16 SHUFFLE), the kernel's block size is 128, the layout plain, the main q
    e4m3 and there are no scale planes. The index dtype only picks the index branch."""
    _, rec = cpu_layer
    impl = make_impl(index_dtype, "fp4", dev="cpu")
    k, v = cpu_pool(), cpu_pool()
    index = cpu_pool()[:, 0] if index_dtype == "fp4" else torch.zeros(PAGES, 128, D)
    q, _, _, kc, vc, ks, vs = cpu_run(impl, k, v, index)
    ((args, kwargs),) = rec.calls
    assert args[13] is k and args[14] is v and args[15] is index
    assert args[16] == 128 and args[17].dtype == torch.float8_e4m3fn
    assert kwargs["kv_cache_dtype"] == "mxfp4" and kwargs["asm_layout"] is False
    assert kwargs["k_scale"] is None and kwargs["v_scale"] is None
    assert (
        kwargs["index_cache_dtype"]
        == {"fp8": "fp8", "fp4": "mxfp4", "auto": "auto"}[index_dtype]
    )
    assert not kwargs.get("skip_index_branch")
    assert q.dtype == torch.float8_e4m3fn and q.shape == (9, NQ, D)
    assert kc is k and vc is v and ks is None and vs is None
    want = torch.float8_e4m3fn if index_dtype == "fp4" else torch.bfloat16
    assert impl._index_q.dtype == want and impl._index_q.shape == (9, NIQ, D)


def test_a_top_k_reusing_layer_is_refused_under_fp4_kv(cpu_layer):
    """The startup gate refuses such a model; a layer that gets here anyway must not
    write wrong pages or reach the page-16 Triton writer."""
    _, rec = cpu_layer
    impl = make_impl("fp8", "fp4", dev="cpu")
    impl.skip_index_topk = True
    with pytest.raises(RuntimeError, match="skip-index"):
        cpu_run(impl, cpu_pool(), cpu_pool(), torch.zeros(PAGES, 128, D))
    assert rec.calls == []


@pytest.mark.parametrize(
    "kv,index,decode,want",
    [
        ("fp4", "fp8", False, {"pages_per_block": 1, "block_page_stride": 1}),
        ("fp4", "fp8", True, {"pages_per_block": 1, "block_page_stride": 1}),
        ("fp4", "fp4", False, {"pages_per_block": 1, "block_page_stride": 1, "index_mxfp4": True}),
        ("fp4", "fp4", True, {"pages_per_block": 1, "block_page_stride": 1, "index_mxfp4": True, "index_score": "S"}),
        ("fp8", "fp4", False, {"index_mxfp4": True}),
        ("fp8", "fp8", True, {}),
    ],
)  # fmt: skip
def test_the_selector_emits_one_page_a_block_exactly_for_fp4_kv(
    cpu_layer, kv, index, decode, want
):
    impl = make_impl(index, kv, dev="cpu")
    md = types.SimpleNamespace(index_score="S")
    assert impl._mxfp4_select_kwargs(md, decode=decode) == want


class Spy:
    def __init__(self, ret=None):
        self.calls, self.ret = [], ret

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        return self.ret


def _dispatch_impl(index_dtype, kv):
    impl = make_impl(index_dtype, kv, dev="cpu")
    impl.topk, impl.init_blocks, impl.local_blocks = 16, 1, 1
    impl.indexer_cp_group = None
    impl.index_topk_cache_state = None
    impl.scale = D**-0.5
    impl.layer_num = 0
    impl.index_cache = None
    return impl


@pytest.mark.parametrize("qlen", [1, 4])
@pytest.mark.parametrize("index_dtype", ["fp8", "fp4"])
def test_decode_runs_the_mxfp4_reader_on_the_emitted_rows(
    cpu_layer, monkeypatch, index_dtype, qlen
):
    """Decode / speculative verify (qlen 4): the selector emits one row per (token,
    kv head) at one page a block, the reader is given all of them -- the verify
    tokens are rows -- planned, with the sparse metadata owning the plan refresh, and
    the page-16 ASM decode is not called."""
    from atom.model_ops.minimax_m3 import index_topk, sparse_attn, sparse_attn_mxfp4

    reqs = 3
    rows = reqs * qlen
    emit = (
        torch.zeros(NKV, rows, 16),
        torch.zeros(rows * NKV, 16),
        torch.zeros(rows * NKV),
    )
    select, asm, reader = Spy(emit), Spy(), Spy(torch.zeros(rows, NQ, D))
    monkeypatch.setattr(index_topk, "minimax_m3_index_topk_decode", select)
    monkeypatch.setattr(sparse_attn, "minimax_m3_sparse_attn_decode_asm", asm)
    monkeypatch.setattr(sparse_attn_mxfp4, "sparse_pa_mxfp4", reader)
    impl = _dispatch_impl(index_dtype, "fp4")
    impl._index_q = torch.zeros(rows, NIQ, D)
    impl._index_q_cache_key_info = (tuple(impl._index_q.shape), torch.bfloat16, "cpu")
    decode = types.SimpleNamespace(
        block_table=torch.zeros(reqs, 4, dtype=torch.int32),
        seq_lens=torch.zeros(reqs, dtype=torch.int32),
        max_query_len=qlen,
    )
    sm = types.SimpleNamespace(
        decode=decode, max_seq_len=512, n_valid_column_per_row=None, index_score="S"
    )
    ctx = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(sparse_attention_metadata=sm)
    )
    q = torch.zeros(rows, NQ, D, dtype=torch.float8_e4m3fn)
    v = torch.zeros(rows, NKV, D, dtype=torch.bfloat16)
    out = impl._sparse_decode(q, None, v, cpu_pool(), cpu_pool(), None, None, ctx)
    assert out.shape == q.shape
    ((_, skw),) = select.calls
    assert skw["max_query_len"] == qlen and skw["emit_sparse_block_table"]
    assert skw["pages_per_block"] == 1 and skw["block_page_stride"] == 1
    assert skw.get("index_mxfp4", False) == (index_dtype == "fp4")
    ((args, rkw),) = reader.calls
    assert args[0] is q and args[3] is emit[1] and args[4] is emit[2]
    assert args[5] == NKV and args[7] is torch.bfloat16
    assert rkw == {"planned": True, "plan_step_owner": sm}
    assert asm.calls == []


def test_prefill_runs_the_mxfp4_reader_plan_free(cpu_layer, monkeypatch):
    from atom.model_ops.minimax_m3 import index_topk, sparse_attn, sparse_attn_mxfp4

    T = 40
    emit = (torch.zeros(NKV, T, 16), torch.zeros(T * NKV, 16), torch.zeros(T * NKV))
    select, asm, reader = Spy(emit), Spy(), Spy(torch.zeros(T, NQ, D))
    monkeypatch.setattr(index_topk, "minimax_m3_index_topk", select)
    monkeypatch.setattr(sparse_attn, "minimax_m3_sparse_attn_prefill_asm", asm)
    monkeypatch.setattr(sparse_attn_mxfp4, "sparse_pa_mxfp4", reader)
    impl = _dispatch_impl("fp8", "fp4")
    impl._index_q = torch.zeros(T, NIQ, D)
    impl._index_q_cache_key_info = (tuple(impl._index_q.shape), torch.bfloat16, "cpu")
    prefill = types.SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, T], dtype=torch.int32),
        seq_lens=torch.tensor([300 + T], dtype=torch.int32),
        context_lens=torch.tensor([300], dtype=torch.int32),  # a cached prefix
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        max_query_len=T,
        max_seq_len=300 + T,
        qo_indptr=None,
    )
    sm = types.SimpleNamespace(prefill=prefill, n_valid_column_per_row=None)
    ctx = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(sparse_attention_metadata=sm)
    )
    q = torch.zeros(T, NQ, D, dtype=torch.float8_e4m3fn)
    v = torch.zeros(T, NKV, D, dtype=torch.bfloat16)
    impl._sparse_prefill(q, None, v, cpu_pool(), cpu_pool(), None, None, ctx)
    ((sargs, skw),) = select.calls
    assert skw["pages_per_block"] == 1 and skw["block_page_stride"] == 1
    assert sargs[5] is prefill.context_lens  # the cached prefix reaches the selector
    ((args, rkw),) = reader.calls
    assert rkw == {}  # plan-free: a chunk's row count is new almost every forward
    assert args[3] is emit[1] and args[4] is emit[2]
    assert asm.calls == []
