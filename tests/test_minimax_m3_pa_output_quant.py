# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for MiniMax-M3 sparse PA -> prequantized o_proj."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops import base_attention
from atom.model_ops.minimax_m3 import sparse_attn
from atom.models.minimax_m3 import model as m3_model
from atom.models.minimax_m3.model import MiniMaxM3SparseAttention


class _QKV:
    def __call__(self, hidden_states, x_scale=None):
        del x_scale
        return hidden_states


class _OProj:
    def __init__(self, per_token_fp8: bool = True):
        self.calls = []
        # Online quantization sets these after load; the fused path depends on them.
        self.quant_type = (
            m3_model.QuantType.per_Token if per_token_fp8 else m3_model.QuantType.No
        )
        self.params_dtype = m3_model.dtypes.fp8 if per_token_fp8 else torch.bfloat16

    def __call__(self, values, x_scale=None):
        self.calls.append((values, x_scale))
        return values


class _QuantAttention:
    def __init__(self):
        self.quant_calls = []
        self.plain_calls = []

    def forward_quantized(self, q, k, v, positions, q_scale=None, qkv=None):
        self.quant_calls.append((q, k, v, positions, qkv))
        return torch.zeros_like(q, dtype=torch.uint8), torch.ones(
            (q.shape[0], 1), dtype=torch.float32
        )

    def __call__(self, *args, **kwargs):
        self.plain_calls.append((args, kwargs))
        return args[0]


def _sparse_layer(*, fused: bool, o_proj_per_token_fp8: bool = True):
    return SimpleNamespace(
        qkv_proj=_QKV(),
        o_proj=_OProj(o_proj_per_token_fp8),
        attn=_QuantAttention(),
        q_size=4,
        kv_size=2,
        index_q_size=2,
        idx_head_dim=2,
        sparse_pa_output_quant_supported=fused,
        sparse_pa_mxfp8_out=False,
        fused_qkv_chain=False,
        fast_oproj=False,
        o_proj_name="model.layers.0.self_attn.o_proj",
    )


def test_sparse_model_hands_fused_fp8_and_token_scale_to_o_proj():
    layer = _sparse_layer(fused=True)
    packed = torch.randn(3, 12)
    result = MiniMaxM3SparseAttention.forward(
        layer, torch.arange(3), packed, hidden_states_scale=None
    )

    assert not layer.attn.plain_calls
    assert len(layer.attn.quant_calls) == 1
    values, scale = layer.o_proj.calls[0]
    assert result is values
    assert values.dtype == torch.uint8
    assert scale.shape == (3, 1)
    assert scale.dtype == torch.float32


def test_sparse_model_default_path_keeps_o_proj_api_unchanged():
    layer = _sparse_layer(fused=False)
    packed = torch.randn(3, 12)
    MiniMaxM3SparseAttention.forward(
        layer, torch.arange(3), packed, hidden_states_scale=None
    )

    assert not layer.attn.quant_calls
    assert len(layer.attn.plain_calls) == 1
    _, scale = layer.o_proj.calls[0]
    assert scale is None


def test_sparse_runner_forces_flydsl_and_preserves_token_scale_layout(monkeypatch):
    captured = {}

    def run_pa_decode(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(base_attention, "run_pa_decode", run_pa_decode)
    # An aiter without the store_output / quantized_pad_rows options.
    monkeypatch.setattr(base_attention, "_flydsl_output_quant_option", lambda _: False)
    monkeypatch.setattr(
        "aiter.ops.triton.gluon.pa_decode_gluon.get_recommended_splits",
        lambda *_: 4,
    )

    def unexpected_asm(*_args, **_kwargs):
        raise AssertionError("prequantized output must not route to ASM")

    monkeypatch.setattr(sparse_attn, "_sparse_pa_ran_on_asm", unexpected_asm)

    q = torch.randn(2, 16, 128, dtype=torch.bfloat16)
    output = torch.empty_like(q)
    output_fp8 = torch.empty_like(q, dtype=torch.uint8)
    scale = torch.empty((2, 1), dtype=torch.float32)
    k_cache = torch.empty((1, 1, 8, 16, 16), dtype=torch.uint8)
    v_cache = torch.empty((1, 1, 1, 128, 16), dtype=torch.uint8)

    sparse_attn.sparse_pa_per_row(
        q,
        k_cache,
        v_cache,
        torch.zeros((2, 8), dtype=torch.int32),
        torch.ones(2, dtype=torch.int32),
        1,
        128**-0.5,
        output,
        None,
        None,
        quantized_output=output_fp8,
        output_scale=scale,
    )

    assert captured["q"].shape == (2, 16, 128)
    assert captured["output"].data_ptr() == output.data_ptr()
    assert captured["quantized_output"].data_ptr() == output_fp8.data_ptr()
    assert captured["output_scale"].data_ptr() == scale.data_ptr()
    assert captured["output_scale"].shape == (2, 1)
    assert captured["max_seqlen_q"] == 1
    assert captured["max_context_partition_num"] == 1
    assert captured["store_output"] is True
    assert captured["quantized_pad_rows"] == 0


def test_sparse_model_checks_o_proj_quant_at_forward_time():
    # At construction o_proj is not yet per-token FP8 (online quantization runs at
    # load time); the fused path must follow o_proj's state when forward runs.
    layer = _sparse_layer(fused=True, o_proj_per_token_fp8=False)
    packed = torch.randn(3, 12)
    MiniMaxM3SparseAttention.forward(layer, torch.arange(3), packed)
    assert not layer.attn.quant_calls
    assert len(layer.attn.plain_calls) == 1

    layer.o_proj.quant_type = m3_model.QuantType.per_Token
    layer.o_proj.params_dtype = m3_model.dtypes.fp8
    MiniMaxM3SparseAttention.forward(layer, torch.arange(3), packed)
    assert len(layer.attn.quant_calls) == 1


def _run_sparse_rows(monkeypatch, rows, bucket):
    captured = {}
    monkeypatch.setattr(
        base_attention, "run_pa_decode", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(base_attention, "_flydsl_output_quant_option", lambda _: True)
    monkeypatch.setattr(
        "aiter.ops.triton.gluon.pa_decode_gluon.get_recommended_splits",
        lambda *_: 1,
    )
    q = torch.randn(rows, 16, 128, dtype=torch.bfloat16)
    output_fp8 = torch.empty((bucket, 16, 128), dtype=torch.uint8)
    scale = torch.empty((bucket, 1), dtype=torch.float32)
    sparse_attn.sparse_pa_per_row(
        q,
        torch.empty((1, 1, 8, 16, 16), dtype=torch.uint8),
        torch.empty((1, 1, 1, 128, 16), dtype=torch.uint8),
        torch.zeros((rows, 8), dtype=torch.int32),
        torch.ones(rows, dtype=torch.int32),
        1,
        128**-0.5,
        torch.empty_like(q),
        None,
        None,
        quantized_output=output_fp8,
        output_scale=scale,
    )
    return captured, output_fp8, scale


def test_sparse_runner_skips_bf16_store_and_hands_the_padded_bucket(monkeypatch):
    captured, output_fp8, scale = _run_sparse_rows(monkeypatch, rows=5, bucket=8)

    assert captured["q"].shape == (5, 16, 128)
    assert captured["output"].shape == (5, 16, 128)
    assert captured["store_output"] is False
    assert captured["quantized_pad_rows"] == 3
    assert captured["quantized_output"].shape == (8, 16, 128)
    assert captured["quantized_output"].data_ptr() == output_fp8.data_ptr()
    assert captured["output_scale"].data_ptr() == scale.data_ptr()


def test_sparse_runner_unpadded_rows_pass_no_pad(monkeypatch):
    captured, _, _ = _run_sparse_rows(monkeypatch, rows=6, bucket=6)

    assert captured["store_output"] is False
    assert captured["quantized_pad_rows"] == 0


_BUCKET, _ROWS = 8, 5


class _QuantImpl:
    """Writes the scheduled FP8 rows (and, given the bucket, its pad rows) like
    the sparse PA does; records what the op body handed it."""

    supports_output_quant = True

    def __init__(self, writes_pad):
        self.writes_quantized_pad_rows = writes_pad
        self.calls = []

    def forward(self, query, key, value, position, q_scale, qkv, **kwargs):
        quantized_output = kwargs["quantized_output"]
        output_scale = kwargs["output_scale"]
        self.calls.append(SimpleNamespace(query=query, **kwargs))
        rows = query.shape[0]
        quantized_output[:rows].fill_(7)
        output_scale[:rows].fill_(0.5)
        if quantized_output.shape[0] > rows:
            quantized_output[rows:].zero_()
            output_scale[rows:].fill_(1.0)


def _quant_op(monkeypatch, writes_pad):
    from atom.utils import forward_context

    impl = _QuantImpl(writes_pad)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"layer": SimpleNamespace(impl=impl)}
        )
    )
    monkeypatch.setattr(base_attention, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(
            context=SimpleNamespace(is_prefill=True, scheduled_tokens=_ROWS)
        ),
    )
    q = torch.randn(_BUCKET, 16)
    fp8 = torch.full((_BUCKET, 16), 0x7F, dtype=torch.uint8)
    scale = torch.full((_BUCKET, 1), float("nan"))
    base_attention.unified_attention_with_output_quant_base.__wrapped__(
        q, None, q, q, torch.arange(_BUCKET), "layer", False, q, fp8, scale
    )
    return impl, fp8, scale


def test_padded_prefill_hands_the_pa_the_whole_bucket(monkeypatch):
    impl, fp8, scale = _quant_op(monkeypatch, writes_pad=True)

    (call,) = impl.calls
    assert call.query.shape[0] == _ROWS
    assert call.quantized_output.shape[0] == _BUCKET
    assert call.quantized_output.data_ptr() == fp8.data_ptr()
    assert call.output_scale.shape[0] == _BUCKET
    assert torch.equal(fp8[:_ROWS], torch.full((_ROWS, 16), 7, dtype=torch.uint8))
    assert torch.equal(
        fp8[_ROWS:], torch.zeros((_BUCKET - _ROWS, 16), dtype=torch.uint8)
    )
    assert torch.equal(scale[_ROWS:], torch.ones((_BUCKET - _ROWS, 1)))


def test_padded_prefill_fills_pad_rows_when_the_pa_cannot(monkeypatch):
    impl, fp8, scale = _quant_op(monkeypatch, writes_pad=False)

    (call,) = impl.calls
    assert call.quantized_output.shape[0] == _ROWS
    assert call.output_scale.shape[0] == _ROWS
    assert torch.equal(fp8[:_ROWS], torch.full((_ROWS, 16), 7, dtype=torch.uint8))
    assert torch.equal(
        fp8[_ROWS:], torch.zeros((_BUCKET - _ROWS, 16), dtype=torch.uint8)
    )
    assert torch.equal(scale[:_ROWS], torch.full((_ROWS, 1), 0.5))
    assert torch.equal(scale[_ROWS:], torch.ones((_BUCKET - _ROWS, 1)))


def _decode_rows(monkeypatch, rows, splits=4, **kwargs):
    captured = {}
    monkeypatch.setattr(
        base_attention, "run_pa_decode", lambda **kw: captured.update(kw)
    )
    monkeypatch.setattr(base_attention, "_flydsl_output_quant_option", lambda _: True)
    monkeypatch.setattr(
        "aiter.ops.triton.gluon.pa_decode_gluon.get_recommended_splits",
        lambda *_: splits,
    )
    monkeypatch.setattr(sparse_attn, "_cluster_resident_rows", lambda _: 240)
    q = torch.randn(rows, 16, 128, dtype=torch.bfloat16)
    sparse_attn.sparse_pa_per_row(
        q,
        torch.empty((1, 1, 8, 16, 16), dtype=torch.uint8),
        torch.empty((1, 1, 1, 128, 16), dtype=torch.uint8),
        torch.zeros((rows, 8), dtype=torch.int32),
        torch.ones(rows, dtype=torch.int32),
        1,
        128**-0.5,
        torch.empty_like(q),
        None,
        None,
        quantized_output=torch.empty((rows, 16, 128), dtype=torch.uint8),
        output_scale=torch.empty((rows, 1), dtype=torch.float32),
        **kwargs,
    )
    return captured


def test_fp8_decode_rows_get_one_cluster_of_eight_each(monkeypatch):
    # 128 decode rows: eight splits with a budget of eight tasks per row, so
    # the fused cluster8 epilogue applies instead of a split it cannot quantize.
    captured = _decode_rows(monkeypatch, 128, quantized_direct=False)
    assert captured["plan_partition_cap"] == 8
    assert captured["plan_workgroup_budget"] == 8 * 128


def test_fp8_decode_rows_with_eight_splits_get_the_cluster_budget(monkeypatch):
    # 72 rows: eight splits, but the default budget cannot hold 72 clusters.
    captured = _decode_rows(monkeypatch, 72, quantized_direct=False, splits=8)
    assert captured["plan_partition_cap"] == 8
    assert captured["plan_workgroup_budget"] == 8 * 72


def test_fp8_decode_rows_past_resident_clusters_keep_the_split(monkeypatch):
    captured = _decode_rows(monkeypatch, 241, quantized_direct=False)
    assert captured["plan_partition_cap"] == 4
    assert captured["plan_workgroup_budget"] is None


def test_fp8_prefill_rows_stay_direct(monkeypatch):
    captured = _decode_rows(monkeypatch, 128)
    assert captured["plan_partition_cap"] == 1
    assert captured["plan_workgroup_budget"] is None


# --- ATOM_MINIMAX_M3_PA_MXFP8_OUT: the NVFP4 attention writes o_proj's MXFP8 input


class _MxAttention(_QuantAttention):
    def __init__(self, kv_format: str = "nvfp4", num_kv_heads: int = 1):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        # Before the KV cache is bound -- when torch.compile traces the forward
        # in the warmup run -- `kv_nvfp4` is still False.
        self.impl = SimpleNamespace(kv_format=kv_format, kv_nvfp4=False)
        self.quant_kwargs = []

    def forward_quantized(self, q, k, v, positions, q_scale=None, qkv=None, mxfp8=False):
        self.quant_kwargs.append({"mxfp8": mxfp8})
        return super().forward_quantized(q, k, v, positions, q_scale, qkv)


def _mx_layer(*, enabled=True, fast_oproj=False):
    layer = _sparse_layer(fused=False)
    layer.attn = _MxAttention()
    layer.o_proj.native_a8_group_rows = 1
    layer.sparse_pa_mxfp8_out = enabled
    layer.fast_oproj = fast_oproj
    return layer


def test_mxfp8_out_hands_the_attention_fp8_and_scales_to_o_proj():
    layer = _mx_layer()
    MiniMaxM3SparseAttention.forward(layer, torch.arange(3), torch.randn(3, 12))
    assert layer.attn.quant_kwargs == [{"mxfp8": True}]
    assert not layer.attn.plain_calls
    values, scale = layer.o_proj.calls[0]
    assert values.dtype == torch.uint8 and scale is not None


def test_fast_oproj_routes_the_mxfp8_input_through_the_op(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops.aiter,
        "minimax_m3_o_proj_mxfp8",
        lambda x, s, name: calls.append((x.dtype, s is not None, name)) or x,
        raising=False,
    )
    layer = _mx_layer(fast_oproj=True)
    MiniMaxM3SparseAttention.forward(layer, torch.arange(3), torch.randn(3, 12))
    assert calls == [(torch.uint8, True, "model.layers.0.self_attn.o_proj")]
    assert not layer.o_proj.calls


def test_mxfp8_out_keeps_the_bf16_path_otherwise():
    layer = _mx_layer(enabled=False)
    MiniMaxM3SparseAttention.forward(layer, torch.arange(3), torch.randn(3, 12))
    assert not layer.attn.quant_kwargs
    assert len(layer.attn.plain_calls) == 1
    assert layer.o_proj.calls[0][1] is None


def _mx_conditions(
    monkeypatch, *, flag=True, kv_format="nvfp4", native_a8=1, kv_heads=1
):
    monkeypatch.setattr(m3_model.envs, "ATOM_MINIMAX_M3_PA_MXFP8_OUT", flag)
    monkeypatch.setattr(m3_model, "nvfp4_reads_page_table", lambda: True)
    attn = _MxAttention(kv_format, kv_heads)
    o_proj = SimpleNamespace(native_a8_group_rows=native_a8)
    return m3_model._sparse_pa_mxfp8_conditions(attn, o_proj)


def test_mxfp8_out_follows_the_declared_kv_format_not_the_binding(monkeypatch):
    """Decided at construction: torch.compile traces the forward in the warmup
    run, before the KV cache is bound (``impl.kv_nvfp4`` still False), and the
    compiled graph keeps that branch for every later step."""
    assert all(_mx_conditions(monkeypatch).values())


@pytest.mark.parametrize(
    "kwargs",
    [{"flag": False}, {"kv_format": ""}, {"native_a8": None}, {"kv_heads": 4}],
    ids=["flag-off", "fp8-kv", "o_proj-not-native-a8", "four-kv-heads"],
)
def test_mxfp8_out_conditions_refuse(monkeypatch, kwargs):
    assert not all(_mx_conditions(monkeypatch, **kwargs).values())


def test_sparse_impl_declares_the_kv_format_its_pool_takes(monkeypatch):
    from atom.model_ops.attentions.aiter_attention import AiterAttentionMetadataBuilder

    for flag, expected in ((True, "nvfp4"), (False, "")):
        monkeypatch.setattr(m3_model.envs, "ATOM_M3_SPARSE_KV_NVFP4", flag)
        module = SimpleNamespace(
            impl=SimpleNamespace(is_indexed_sparse_attention=True, kv_format=expected),
            base_attention=True,
            use_mla=False,
            num_kv_heads=1,
            head_dim=128,
        )
        kinds = AiterAttentionMetadataBuilder._module_kinds(None, module)
        assert [k.kv_format for k in kinds if hasattr(k, "kv_format")] == [expected]


# --- ATOM_M3_FAST_OPROJ: o_proj on the MXFP8 input through aiter's decode GEMM


class _FastOProj:
    """The attributes `_fast_oproj_conditions` reads; calling it is the linear."""

    def __init__(self, shuffled: bool):
        self.weight = torch.nn.Parameter(
            torch.zeros(6144, 2048, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        if shuffled:
            self.weight.is_shuffled = True
        self.weight_scale = torch.zeros(6144, 64, dtype=torch.uint8)
        self.bias = None
        self.native_a8_group_rows = 1
        self.reduce_results = False
        self.prefix = "model.layers.0.self_attn.o_proj"
        self.calls = []

    def __call__(self, x, x_scale=None):
        self.calls.append(x.shape[0])
        return torch.zeros(x.shape[0], 6144, dtype=torch.bfloat16)


@pytest.fixture
def decode_gemm(monkeypatch):
    """aiter's gfx1250 decode GEMM, served up to 128 rows; records its rows."""
    mod = importlib.import_module("aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250")
    calls = []
    monkeypatch.setattr(
        mod,
        "mxfp8_decode_gemm_supported",
        lambda m, n, k: 0 < m <= 128 and (n, k) == (6144, 2048),
    )

    def run(x, x_scale, w, w_scale):
        calls.append(x.shape[0])
        assert x_scale.dtype == torch.float8_e8m0fnu and x_scale.shape[1] == 64
        return torch.ones(x.shape[0], w.shape[0], dtype=torch.bfloat16)

    monkeypatch.setattr(mod, "mxfp8_decode_gemm", run)
    return calls


def _mx_input(rows):
    x = torch.zeros(rows, 2048, dtype=torch.float8_e4m3fn)
    return x, torch.zeros(rows, 64, dtype=torch.float8_e8m0fnu)


def test_fast_oproj_serves_decode_rows_and_keeps_the_linear_past_them(decode_gemm):
    o_proj = _FastOProj(shuffled=True)
    for rows in (8, 80, 128, 129, 14336):
        y = m3_model._o_proj_mxfp8(o_proj, *_mx_input(rows))
        assert y.shape == (rows, 6144)
    assert decode_gemm == [8, 80, 128]
    assert o_proj.calls == [129, 14336]


def test_fast_oproj_weight_is_judged_at_the_first_call(decode_gemm):
    """The weight is shuffled by `process_weights_after_loading`, after the
    attention's construction: the verdict waits for the first call."""
    o_proj = _FastOProj(shuffled=False)
    o_proj.weight.is_shuffled = True  # what the load does before any forward
    m3_model._o_proj_mxfp8(o_proj, *_mx_input(80))
    assert decode_gemm == [80] and not o_proj.calls


def test_fast_oproj_falls_back_to_the_linear_without_a_shuffled_weight(decode_gemm):
    o_proj = _FastOProj(shuffled=False)
    m3_model._o_proj_mxfp8(o_proj, *_mx_input(80))
    assert not decode_gemm and o_proj.calls == [80]
    assert m3_model._fast_oproj_conditions(o_proj)["preshuffled weight"] is False


def test_fast_oproj_op_looks_the_linear_up_by_name(decode_gemm, monkeypatch):
    o_proj = _FastOProj(shuffled=True)
    context = {"model.layers.0.self_attn.o_proj": o_proj}
    monkeypatch.setattr(
        m3_model,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=context)
        ),
    )
    x, scale = _mx_input(80)
    name = "model.layers.0.self_attn.o_proj"
    assert m3_model.minimax_m3_o_proj_mxfp8(x, scale, name).shape == (80, 6144)
    fake = m3_model._minimax_m3_o_proj_mxfp8_fake(x, scale, name)
    assert fake.shape == (80, 6144) and fake.dtype == torch.bfloat16
    assert decode_gemm == [80]


def _nvfp4_flydsl_call(monkeypatch, rows, bucket, scale_dtype, direct):
    from atom.model_ops.minimax_m3 import sparse_attn_nvfp4

    captured = {}
    monkeypatch.setattr(
        importlib.import_module("aiter.ops.flydsl.pa_decode"),
        "pa_decode",
        lambda *a, **kw: captured.update(kw, args=a),
    )
    monkeypatch.setattr(
        "aiter.ops.triton.gluon.pa_decode_gluon.get_recommended_splits",
        lambda *_: 3,
    )
    monkeypatch.setattr(
        base_attention,
        "_flydsl_prepare_explicit_plan",
        lambda **kw: captured.setdefault("plan", kw),
    )
    monkeypatch.setattr(
        base_attention, "_flydsl_plan_scratch", lambda *a, **kw: (None, None, None)
    )
    monkeypatch.setattr(sparse_attn, "_cluster_resident_rows", lambda _: 240)
    q = torch.randn(rows, 16, 128, dtype=torch.bfloat16)
    pages = 4
    payload = torch.zeros(pages, 1, 128, 64, dtype=torch.uint8)
    scales = torch.zeros(pages, 1, 128, 8, dtype=torch.uint8)
    width = 16 * 128 // 32 if scale_dtype == torch.float8_e8m0fnu else 1
    sparse_attn_nvfp4._flydsl_attention(
        q, payload, scales, payload, scales,
        torch.zeros(rows, 1, dtype=torch.int32), torch.ones(rows, dtype=torch.int32),
        128**-0.5, None, None,
        quantized_output=torch.empty(bucket, 16 * 128, dtype=torch.uint8),
        output_scale=torch.empty(bucket, width, dtype=scale_dtype),
        quantized_direct=direct,
    )  # fmt: skip
    return captured


@pytest.mark.parametrize("direct", [True, False], ids=["prefill", "decode"])
def test_mxfp8_out_keeps_the_bf16_split(monkeypatch, direct):
    """The MXFP8 output takes any output mode, so the attention keeps the BF16
    call's plan (and hence its BF16 values, which the split path quantized); the
    per-token output re-splits (direct prefill, cluster-of-8 decode)."""
    mx = _nvfp4_flydsl_call(monkeypatch, 64, 70, torch.float8_e8m0fnu, direct)
    assert mx["plan"]["plan_partition_cap"] == 3
    assert mx["plan"]["plan_workgroup_budget"] is None
    assert mx["store_output"] is False
    assert mx["quantized_pad_rows"] == 6
    assert mx["output_scale"].shape == (70, 64)
    token = _nvfp4_flydsl_call(monkeypatch, 64, 70, torch.float32, direct)
    assert token["plan"]["plan_partition_cap"] == (1 if direct else 8)


def test_fill_unit_scale_writes_e8m0_one_and_fp32_one():
    e8m0 = torch.zeros(2, 4, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    base_attention.fill_unit_scale(e8m0)
    assert (e8m0.view(torch.uint8) == 127).all()
    fp32 = torch.zeros(2, 1)
    base_attention.fill_unit_scale(fp32)
    assert (fp32 == 1).all()
