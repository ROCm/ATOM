# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 sparse PA writing o_proj's per-token FP8 input (aiter #7401's
``pa_decode(quantized_output=, output_scale=, ...)``): the gate, the op, the
routing and, on gfx1250, the bytes against BF16 PA + the standalone quant. Also
the plan cap of the BF16 sparse prefill."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def m():
    """The modules under test; on a host without a usable AITER (the CPU job)
    against its stand-ins, kept for the module's tests."""
    with atom_without_aiter() as load:
        base_attention = load("atom.model_ops.base_attention")
        yield SimpleNamespace(
            dtypes=base_attention.dtypes,  # the aiter it was imported against
            attention_mha=load("atom.model_ops.attention_mha"),
            base_attention=base_attention,
            sparse_attn=load("atom.model_ops.minimax_m3.sparse_attn"),
        )


@pytest.mark.parametrize(
    "kv, kv_heads, gluon, want",
    [
        ("fp8", 1, False, True),
        ("bf16", 1, False, False),  # measured on FP8 KV only
        ("fp4", 1, False, False),  # the MXFP4 KV's PA writes no FP8
        ("fp8", 2, False, False),  # a token's row spans two PA sequences
        ("fp8", 1, True, False),  # gluon GPUs: not every call is FlyDSL's
    ],
)
def test_the_sparse_impl_writes_fp8_only_where_it_was_measured(
    monkeypatch, m, kv, kv_heads, gluon, want
):
    attention_mha = m.attention_mha
    impl_cls = attention_mha.SparseMHAPagedAttentionImpl

    def base_init(self, **kw):
        torch.nn.Module.__init__(self)
        self.kv_cache_dtype = kw["kv_cache_dtype"]

    monkeypatch.setattr(attention_mha.PagedAttentionImpl, "__init__", base_init)
    monkeypatch.setattr(attention_mha, "pa_decode_gluon_supports_arch", lambda: gluon)
    monkeypatch.setattr("atom.distributed.indexer_cp.indexer_cp_enabled", lambda: False)
    monkeypatch.setattr(
        "atom.model_ops.minimax_m3.mxfp4_pages.Mxfp4KvLayout", object, raising=False
    )
    impl = impl_cls(16, 128, 1.0, kv_heads, kv_cache_dtype=kv)
    assert impl.supports_output_quant is want
    assert attention_mha.PagedAttentionImpl.supports_output_quant is False


@pytest.mark.parametrize("dummy", [True, False])
def test_the_fp8_op_returns_what_the_impl_wrote(monkeypatch, m, dummy):
    """`unified_attention_fp8` -> impl.forward_impl: a dummy run writes inert rows;
    otherwise the backend gets the FP8 output and its descales."""
    attention_mha, base_attention, dtypes = m.attention_mha, m.base_attention, m.dtypes
    seen = {}

    def backend(q, k, v, k_cache, v_cache, k_scale, v_scale, ctx, **kw):
        seen.clear()
        seen.update(kw)
        if not kw:
            return q
        kw["output_scale"].fill_(2.0)
        return kw["quantized_output"].fill_(1.0)

    impl = object.__new__(attention_mha.PagedAttentionImpl)
    torch.nn.Module.__init__(impl)
    impl.num_heads, impl.num_kv_heads, impl.head_dim = 2, 1, 4
    impl.rope_cache = lambda q, k, v, *a: (q, k, v, None, None, None, None)
    impl.dispatch_backend = lambda *a: backend
    layers = {"L": SimpleNamespace(impl=impl)}
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers)
    )
    monkeypatch.setattr(base_attention, "get_current_atom_config", lambda: config)
    ctx = SimpleNamespace(context=SimpleNamespace(is_dummy_run=dummy))
    monkeypatch.setattr(attention_mha, "get_forward_context", lambda: ctx)
    q, kv = torch.randn(3, 8), torch.randn(3, 4)
    scale = torch.full((3, 1), float("nan"))
    # the op's FP8 buffer starts as a sentinel, so its zeros are the impl's writes
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **kw).fill_(3))
    out = base_attention.unified_attention_fp8(q, kv, kv, None, "L", None, scale)
    assert out.dtype == dtypes.fp8 and out.shape == q.shape
    want = 0.0 if dummy else 1.0
    assert (out.float() == want).all() and (scale == want + 1).all()
    assert dummy or seen["quantized_output"].shape == (3, 2, 4)
    # without the pair (every other layer) the backend gets the plain call
    impl.forward(query=q, key=kv, value=kv)
    assert dummy or seen == {}


@pytest.mark.parametrize("sp", [False, True])
def test_attention_hands_o_proj_the_fp8_pair(monkeypatch, m, sp):
    from atom.model_ops.paged_attention import Attention

    dtypes = m.dtypes

    calls = []

    def op(name):
        def run(q, k, v, positions, layer_name, qkv, scale):
            calls.append(name)
            return q.to(dtypes.fp8)

        return run

    for name in ("unified_attention_fp8", "sp_attention_fp8"):
        monkeypatch.setattr(torch.ops.aiter, name, op(name))
    attn = object.__new__(Attention)
    torch.nn.Module.__init__(attn)
    attn.per_token_fp8_out, attn.layer_name = True, "L"
    attn.sp_columns = object() if sp else None
    out, scale = attn.forward(torch.ones(3, 8), None, None)
    assert calls == ["sp_attention_fp8" if sp else "unified_attention_fp8"]
    assert out.dtype == dtypes.fp8 and scale.shape == (3, 1)


@pytest.mark.parametrize(
    "sp, asked, impl_writes, want",
    [
        (False, True, True, True),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, False),
        # under SP the head exchange decides, whatever the impl writes
        (True, True, False, True),
        (True, True, True, False),
    ],
)
def test_attention_turns_the_fp8_output_on(
    monkeypatch, m, sp, asked, impl_writes, want
):
    from atom.model_ops import paged_attention as pa

    class Impl(torch.nn.Module):
        supports_output_quant = impl_writes

        def __init__(self, **kw):
            super().__init__()

        @staticmethod
        def qkv_fields(heads, kv_heads, dim, **kw):
            return [(heads, dim), (kv_heads, dim), (kv_heads, dim)]

    config = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(pa, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(pa, "get_sp_world_size", lambda: 2 if sp else 1)
    monkeypatch.setattr(pa, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(pa, "get_attn_backend", lambda family: None)
    monkeypatch.setattr(
        pa, "sp_fp8_attention_output_supported", lambda: not impl_writes
    )
    monkeypatch.setattr(pa, "get_pcp_group", lambda: None)
    monkeypatch.setattr(pa, "head_exchange_comm", lambda group: None)
    monkeypatch.setattr(pa, "send_columns", lambda fields, device: object())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    attn = pa.Attention(
        16, 128, 1.0, 2, rotary_emb=torch.nn.Identity(), prefix="L",
        impl_cls=Impl, per_token_fp8_out=asked,
    )  # fmt: skip
    assert (attn.sp_columns is not None) is sp
    assert attn.per_token_fp8_out is want


def test_the_fp8_sparse_pa_hands_flydsl_the_fp8_output_alone(monkeypatch, m):
    """FlyDSL's PA gets the FP8 output, its pad-row count and the BF16 store off,
    on a plan with the FP8 budget. aiter quantizes and writes the pad rows in
    the PA launch where the plan allows it, else in launches after it."""
    import aiter.ops.flydsl.pa_decode as fly
    import aiter.ops.triton.gluon.pa_decode_gluon as gluon

    plans, seen = [], {}
    ba, sparse_attn = m.base_attention, m.sparse_attn
    monkeypatch.setattr(
        ba, "_flydsl_prepare_explicit_plan", lambda **kw: plans.append(kw)
    )
    monkeypatch.setattr(ba, "_flydsl_plan_scratch", lambda *a, **kw: (None,) * 3)
    monkeypatch.setattr(sparse_attn, "_fp8_output_plan_budget", lambda _: 1536)
    monkeypatch.setattr(gluon, "get_recommended_splits", lambda n, h: (n, h))
    monkeypatch.setattr(fly, "pa_decode", lambda *a, **kw: seen.update(kw, args=a))
    rows, bucket = 72, 75
    q = torch.empty((rows, 16, 128), dtype=torch.bfloat16)
    bt = torch.zeros((bucket, 8), dtype=torch.int32)
    fp8, scale = torch.empty((bucket, 16, 128), dtype=torch.uint8), torch.empty(
        bucket, 1
    )
    sparse_attn.sparse_pa_fp8_per_row(
        q, torch.empty((8, 1, 8, 16, 16)), torch.empty((8, 1, 1, 128, 16)),
        bt, torch.ones(bucket, dtype=torch.int32), 1.0, torch.empty_like(q),
        *(torch.ones(8, 1, 16),) * 2, fp8, scale,
    )  # fmt: skip
    (plan,) = plans
    assert plan["plan_workgroup_budget"] == 1536 and plan["num_seqs"] == rows
    assert plan["plan_partition_cap"] == (rows, 1)  # the static split count
    assert seen["args"][4].shape == (rows,) and seen["args"][5].shape == (rows, 8)
    assert seen["quantized_output"].data_ptr() == fp8.data_ptr()
    assert seen["output_scale"] is scale and seen["key_scale"].shape == (8, 1, 16, 1)
    assert seen["store_output"] is False and seen["quantized_pad_rows"] == 3


@pytest.mark.parametrize(
    "gfx, min_prefix_len, rows, default_cap",
    [
        ("gfx1250", 4096, 64, True),  # every row's prefix covers the window
        ("gfx1250", 4095, 64, False),  # a row still on its causal ramp
        ("gfx950", 4096, 64, False),
        ("gfx1250", 4096, 512, False),  # rows reach the default plan's budget
    ],
)
def test_saturated_cached_sparse_prefill_plans_at_the_default_cap(
    monkeypatch, m, gfx, min_prefix_len, rows, default_cap
):
    """Sparse prefill keeps the static split cap unless, on gfx1250, every row's
    prefix covers the selected window (256 page-16 columns = 4096 keys here) and
    the rows fit aiter's default plan budget (512 here)."""
    import aiter.ops.flydsl.pa_decode as fly
    import aiter.ops.triton.gluon.pa_decode_gluon as gluon

    ba, sparse_attn = m.base_attention, m.sparse_attn
    caps, budget_shapes = [], []
    monkeypatch.setattr(gluon, "get_recommended_splits", lambda n, h: 8)
    monkeypatch.setattr(sparse_attn, "_sparse_pa_ran_on_asm", lambda *a: False)
    monkeypatch.setattr(sparse_attn, "get_gfx_runtime", lambda: gfx)
    monkeypatch.setattr(
        fly,
        "pa_decode_default_workgroup_budget",
        lambda device, **kw: budget_shapes.append(kw) or 512,
    )
    monkeypatch.setattr(
        ba, "run_pa_decode", lambda **kw: caps.append(kw["plan_partition_cap"])
    )

    def meta(*shape):
        return torch.empty(shape, device="meta")

    sparse_attn.minimax_m3_sparse_attn_prefill_asm(
        q=meta(rows, 16, 128), k_cache=meta(8, 1, 8, 16, 16),
        v_cache=meta(8, 1, 1, 128, 16), topk_idx=None, block_table=None,
        query_req_id=None, query_abs_pos=None, qo_indptr=meta(rows + 1),
        num_kv_heads=1, sm_scale=1.0, output=meta(rows, 16, 128),
        sparse_bt=meta(rows, 256), sparse_ctx=meta(rows),
        min_prefix_len=min_prefix_len,
    )  # fmt: skip
    assert caps == [None if default_cap else 8]
    # asked for this PA's shape: head 128, page 16, 16 query heads a KV head
    assert all(
        (kw["head_dim"], kw["block_size"], kw["query_group_size"]) == (128, 16, 16)
        for kw in budget_shapes
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "decode, rows",
    [(True, 4), (True, 16), (True, 72), (True, 128), (True, 320)] + [(False, 300)],
)
def test_fp8_output_is_bf16_pa_then_the_per_token_quant(monkeypatch, m, decode, rows):
    """Bit for bit, on the plan the FP8 output runs; two pad rows past q's (filled
    with a sentinel) come back zero with a unit descale, for decode and prefill."""
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.quant import dynamic_per_token_scaled_quant

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("aiter's FlyDSL PA writes FP8 on gfx1250 only")
    base_attention, sparse_attn, dtypes = m.base_attention, m.sparse_attn, m.dtypes
    monkeypatch.setattr(base_attention.envs, "ATOM_PA_FLYDSL", True)
    torch.manual_seed(0)
    dev, pages, window, pad = "cuda", 4096, 2048, 2
    q = torch.randn(rows, 16, 128, device=dev, dtype=torch.bfloat16)
    # SHUFFLE pages of 16 tokens: K [page, 1, d // 16, token, 16], V [page, 1, 1, d, token]
    k_cache = (torch.randn(pages, 1, 8, 16, 16, device=dev) / 4).to(dtypes.fp8)
    v_cache = (torch.randn(pages, 1, 1, 128, 16, device=dev) / 4).to(dtypes.fp8)
    kv_scale = torch.rand(pages, 1, 16, device=dev) + 0.5
    # A decode row attends a whole top-k window, a prefill row a causal 1..2048.
    lens = torch.randint(1, window + 1, (rows,), dtype=torch.int32, device=dev)
    if decode:
        lens.fill_(window)
    bt = torch.randint(0, pages, (rows, window // 16), dtype=torch.int32, device=dev)
    fp8 = torch.full((rows + pad, 16, 128), 0x7F, device=dev, dtype=torch.uint8)
    fp8 = fp8.view(dtypes.fp8)
    scale = torch.full((rows + pad, 1), float("nan"), device=dev)
    # prefill keeps its plans on its step metadata, as the impl passes it
    owner = None if decode else SimpleNamespace()
    sparse_attn.sparse_pa_fp8_per_row(
        q, k_cache, v_cache, bt, lens, 128**-0.5, torch.empty_like(q),
        kv_scale, kv_scale, fp8, scale, plan_cache_owner=owner,
    )  # fmt: skip

    # the BF16 output of the same plan, then the quant o_proj would run
    prepare = base_attention._flydsl_prepare_explicit_plan
    budget = sparse_attn._fp8_output_plan_budget(q.device)
    monkeypatch.setattr(
        base_attention,
        "_flydsl_prepare_explicit_plan",
        lambda **kw: prepare(**kw, plan_workgroup_budget=budget),
    )
    bf16 = torch.empty_like(q)
    sparse_attn.sparse_pa_per_row(
        q, k_cache, v_cache, bt, lens, 1, 128**-0.5, bf16, kv_scale, kv_scale,
        plan_cache_owner=owner,
    )  # fmt: skip
    ref, ref_scale = torch.empty_like(fp8[:rows]), torch.empty_like(scale[:rows])
    dynamic_per_token_scaled_quant(ref.view(rows, -1), bf16.view(rows, -1), ref_scale)
    assert torch.equal(fp8[:rows].view(torch.uint8), ref.view(torch.uint8))
    assert torch.equal(scale[:rows], ref_scale)
    assert (fp8[rows:].float() == 0).all() and (scale[rows:] == 1).all()
