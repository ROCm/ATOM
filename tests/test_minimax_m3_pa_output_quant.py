# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 sparse PA writing o_proj's per-token FP8 input (aiter #7401's
``pa_decode(quantized_output=, output_scale=, ...)``): the gate, the op, the
routing and, on gfx1250, the bytes against BF16 PA + the standalone quant."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="base_attention defines @triton.jit kernels")
pytest.importorskip("aiter", reason="base_attention imports the AITER runtime")

from aiter import dtypes

from atom.model_ops import attention_mha, base_attention
from atom.model_ops.minimax_m3 import sparse_attn


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
    monkeypatch, kv, kv_heads, gluon, want
):
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
def test_the_fp8_op_returns_what_the_impl_wrote(monkeypatch, dummy):
    """`unified_attention_fp8` -> impl.forward: a dummy run writes inert rows;
    otherwise the backend gets the FP8 output and its descales."""
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
    out = base_attention.unified_attention_fp8(q, kv, kv, None, "L", None, scale)
    assert out.dtype == dtypes.fp8 and out.shape == q.shape
    want = 0.0 if dummy else 1.0
    assert (out.float() == want).all() and (scale == want + 1).all()
    assert dummy or seen["quantized_output"].shape == (3, 2, 4)
    # without the pair (every other layer) the backend gets the plain call
    impl.forward(query=q, key=kv, value=kv)
    assert dummy or seen == {}


@pytest.mark.parametrize("sp", [False, True])
def test_attention_hands_o_proj_the_fp8_pair(monkeypatch, sp):
    from atom.model_ops.paged_attention import Attention

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
def test_attention_turns_the_fp8_output_on(monkeypatch, sp, asked, impl_writes, want):
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


@pytest.mark.parametrize("fp8", [False, True])
def test_sparse_rows_hand_the_pa_the_fp8_output(monkeypatch, fp8):
    seen = {}
    monkeypatch.setattr(base_attention, "run_pa_decode", lambda **kw: seen.update(kw))
    monkeypatch.setattr(sparse_attn, "_sparse_pa_ran_on_asm", lambda *a: False)
    monkeypatch.setattr(sparse_attn, "_fp8_output_plan_budget", lambda _: 1536)
    rows, bucket = 72, 75  # a padded batch: the PA writes the last three inert
    q = torch.randn(rows, 16, 128, dtype=torch.bfloat16)
    out = torch.empty((bucket, 16, 128), dtype=torch.uint8)
    quant = {"quantized_output": out, "output_scale": torch.empty(bucket, 1)}
    sparse_attn.sparse_pa_per_row(
        *(q, torch.empty((1, 1, 8, 16, 16), dtype=torch.uint8)),
        torch.empty((1, 1, 1, 128, 16), dtype=torch.uint8),
        *(torch.zeros((rows, 8), dtype=torch.int32), torch.ones(rows)),
        *(1, 1.0, torch.empty_like(q), None, None),
        **(quant if fp8 else {}),
    )
    assert seen["plan_workgroup_budget"] == (1536 if fp8 else None)
    assert seen["plan_partition_cap"] == seen["max_context_partition_num"]
    if fp8:
        assert seen["quantized_output"].shape == (bucket, 16, 128)
        assert seen["quantized_output"].data_ptr() == out.data_ptr()
        with pytest.raises(ValueError, match="one KV head"):
            sparse_attn.sparse_pa_per_row(
                q, *(None,) * 4, 2, 1.0, q, None, None, **quant
            )


def test_run_pa_decode_hands_flydsl_the_fp8_output_alone(monkeypatch):
    """No BF16 store, and the rows past q's written inert by the same launch."""
    import aiter.ops.flydsl.pa_decode as fly

    seen = {}
    ba = base_attention
    monkeypatch.setattr(ba, "flydsl_decode_seqs", lambda **kw: 5)
    monkeypatch.setattr(ba, "_flydsl_context_bound", lambda *a: 2048)
    monkeypatch.setattr(ba, "_flydsl_prepare_explicit_plan", lambda **kw: None)
    monkeypatch.setattr(ba, "_flydsl_plan_scratch", lambda *a, **kw: (None,) * 3)
    monkeypatch.setattr(fly, "pa_decode", lambda *a, **kw: seen.update(kw))

    def m(*shape):
        return torch.empty(shape, device="meta")

    def call(**kw):
        ba.run_pa_decode(
            *(m(5, 16, 128), m(5, 16, 128), m(8, 1, 8, 16, 16), m(8, 1, 1, 128, 16)),
            *(m(5), m(5, 8), 1.0, 1, 4, 256, torch.bfloat16, None, None, None),
            **kw,
        )

    fp8, scale = m(8, 16, 128), m(8, 1)
    call(quantized_output=fp8, output_scale=scale)
    assert seen["quantized_output"] is fp8 and seen["output_scale"] is scale
    assert seen["store_output"] is False and seen["quantized_pad_rows"] == 3
    seen.clear()
    call()  # without the pair, FlyDSL gets the plain call
    assert seen and "quantized_output" not in seen and "store_output" not in seen
    with pytest.raises(ValueError, match="go together"):
        call(quantized_output=fp8)
    monkeypatch.setattr(ba, "flydsl_decode_seqs", lambda **kw: None)
    with pytest.raises(RuntimeError, match="only FlyDSL"):
        call(quantized_output=fp8, output_scale=scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "decode, rows",
    [(True, 4), (True, 16), (True, 72), (True, 128), (True, 320)] + [(False, 300)],
)
def test_fp8_output_is_bf16_pa_then_the_per_token_quant(monkeypatch, decode, rows):
    """Bit for bit, on the plan the FP8 output runs; two pad rows past q's (filled
    with a sentinel) come back zero with a unit descale, for decode and prefill."""
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.quant import dynamic_per_token_scaled_quant

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("aiter's FlyDSL PA writes FP8 on gfx1250 only")
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
    args = (q, k_cache, v_cache, bt, lens, 1, 128**-0.5)
    fp8 = torch.full((rows + pad, 16, 128), 0x7F, device=dev, dtype=torch.uint8)
    fp8 = fp8.view(dtypes.fp8)
    scale = torch.full((rows + pad, 1), float("nan"), device=dev)
    # the FP8 output through the phase's entry point, as the impl calls it
    phase = "decode" if decode else "prefill"
    getattr(sparse_attn, f"minimax_m3_sparse_attn_{phase}_asm")(
        q, k_cache, v_cache, *(None,) * (3 if decode else 5), 1, 128**-0.5,
        torch.empty_like(q), fp8, scale, kv_scale, kv_scale,
        sparse_bt=bt, sparse_ctx=lens,
    )  # fmt: skip

    # the BF16 output of the same plan, then the quant o_proj would run
    run = base_attention.run_pa_decode
    budget = sparse_attn._fp8_output_plan_budget(q.device)
    monkeypatch.setattr(
        base_attention,
        "run_pa_decode",
        lambda **kw: run(**{**kw, "plan_workgroup_budget": budget}),
    )
    bf16 = torch.empty_like(q)
    sparse_attn.sparse_pa_per_row(*args, bf16, kv_scale, kv_scale)
    ref, ref_scale = torch.empty_like(fp8[:rows]), torch.empty_like(scale[:rows])
    dynamic_per_token_scaled_quant(ref.view(rows, -1), bf16.view(rows, -1), ref_scale)
    assert torch.equal(fp8[:rows].view(torch.uint8), ref.view(torch.uint8))
    assert torch.equal(scale[:rows], ref_scale)
    assert (fp8[rows:].float() == 0).all() and (scale[rows:] == 1).all()
