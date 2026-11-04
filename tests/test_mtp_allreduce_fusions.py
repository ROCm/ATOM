# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.5 MTP drafter's TP fusions, around their kernels:

- ATOM_MTP_FUSE_ALLREDUCE_NORM: the drafter layer gets the target's GEMM +
  all-reduce + RMSNorm and fused all-reduce + RMSNorm, but its input norm (of
  the replicated fc output) does not all-reduce, and the final norm does.
- ATOM_MTP_SELECT_ROWS_BEFORE_O_PROJ: step 0's rows are kept right after the
  attention; the gate gathers them, the GEMM + all-reduce gathers the residual.
- ATOM_MTP_ATTN_GATE_IN_REDUCE: the output gate (and the rows) handed to the
  unified-attention decode as a request, applied there or by the layer.
- ATOM_FUSED_MOE_TAIL_AR_NORM: a TP-partial MoEGatedSum goes to an all-reducing
  norm.

The kernels themselves are checked on 4 GPUs in
tests/model_ops/test_qwen35_mtp_tp4_fusions.py and on one in
tests/model_ops/test_qwen35_mtp_attn_gate_epilogue.py.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip("the model modules import aiter", allow_module_level=True)

DEV = "cuda"


def _module(name):
    import importlib

    try:
        return importlib.import_module(name)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"{name} does not import here: {exc}")


def _gate_math(x, gate):
    """`_gated_pointwise_kernel` in torch: the sigmoid rounded to the gate
    dtype, the product to x's."""
    g = torch.sigmoid(gate.float()).to(gate.dtype).float()
    return (x.float() * g).to(x.dtype)


# ---------------------------------------------------------------------------
# ATOM_MTP_FUSE_ALLREDUCE_NORM
# ---------------------------------------------------------------------------


def test_input_norm_all_reduces_except_where_the_input_is_complete():
    q35 = _module("atom.models.qwen3_5")
    rule = q35.input_norm_all_reduces
    assert not rule(True, 0, None)  # the embedding
    assert rule(True, 7, None)
    # The MTP layer (index num_hidden_layers) reads the replicated fc output.
    assert not rule(True, 40, True)
    assert not rule(False, 7, None)


@pytest.mark.parametrize("fuse", [False, True])
def test_mtp_predictor_builds_the_fused_layer_and_final_norm(monkeypatch, fuse):
    mtp = _module("atom.models.qwen3_5_mtp")
    monkeypatch.setenv("ATOM_MTP_FUSE_ALLREDUCE_NORM", "1" if fuse else "0")
    monkeypatch.setenv("ATOM_FUSED_MOE_TAIL_AR_NORM", "1")
    monkeypatch.setenv("ATOM_MTP_SELECT_ROWS_BEFORE_O_PROJ", "1")
    monkeypatch.setenv("ATOM_MTP_ATTN_GATE_IN_REDUCE", "1")
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=40,
        mtp_num_hidden_layers=1,
        rms_norm_eps=1e-6,
    )
    seen = {"layers": [], "norms": [], "defer": []}

    class _Layer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            seen["layers"].append(kwargs)
            self.self_attn = SimpleNamespace()

    class _Norm(nn.Module):
        def __init__(self, hidden, eps=1e-6, **kwargs):
            super().__init__()
            seen["norms"].append(kwargs)

    monkeypatch.setattr(mtp, "get_qwen3_5_text_config", lambda c: config)
    monkeypatch.setattr(mtp, "VocabParallelEmbedding", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(mtp, "ColumnParallelLinear", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(mtp, "ReplicatedLinear", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(mtp, "get_tp_group", lambda: SimpleNamespace(world_size=4))
    monkeypatch.setattr(mtp, "allreduce_norm_fusion_enabled", lambda: True)
    monkeypatch.setattr(mtp, "Qwen3_5DecoderLayer", _Layer)
    monkeypatch.setattr(mtp, "Qwen3_5RMSNorm", _Norm)
    monkeypatch.setattr(
        mtp,
        "defer_moe_gated_sums",
        lambda layers, norm, combine_tail=True, tp_partial=False: seen["defer"].append(
            (norm, combine_tail, tp_partial)
        ),
    )
    pred = mtp.Qwen3_5MultiTokenPredictor(SimpleNamespace(quant_config=None))

    (layer_kwargs,) = seen["layers"]
    assert layer_kwargs["layer_num"] == 40
    assert layer_kwargs["fuse_allreduce_norm"] is fuse
    assert layer_kwargs["input_is_reduced"] is True
    # norm, pre_fc_norm_hidden, pre_fc_norm_embedding: only the first reduces.
    assert seen["norms"][0] == {"fused_allreduce": fuse}
    assert seen["norms"][1:] == [{}, {}]
    assert seen["defer"] == [(pred.norm, False, True)]
    layer = pred.layers[0]
    assert layer.select_rows_before_o_proj is True
    assert layer.self_attn.gate_in_attention is True


# ---------------------------------------------------------------------------
# ATOM_MTP_SELECT_ROWS_BEFORE_O_PROJ
# ---------------------------------------------------------------------------


def _decoder_layer(qn, attn, fused, seen):
    layer = object.__new__(qn.Qwen3NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_type = "full_attention"
    layer.layer_scale = False
    layer.self_attn = attn
    layer.input_layernorm = type(
        "Norm", (), {"use_fused_quant": False, "__call__": lambda self, h, r=None: h}
    )()
    layer.post_attention_layernorm = SimpleNamespace(weight=None, variance_epsilon=0)
    if not fused:
        layer.post_attention_layernorm = lambda h, r: (h + r, h + r)
    layer.fuse_gemm_ar = fused
    object.__setattr__(layer, "gemm_ar_proj", SimpleNamespace(weight=2.0))
    layer.mlp = lambda h: (seen.setdefault("mlp_rows", h.shape[0]), h + 1)[1]
    return layer


@pytest.mark.parametrize("fused", [False, True])
def test_decoder_layer_keeps_the_rows_before_o_proj(monkeypatch, fused):
    qn = _module("atom.models.qwen3_next")
    calls = []

    def gemm_ar(x, w, residual, weight, eps, residual_rows=None):
        calls.append(residual_rows)
        if residual_rows is not None:
            residual = residual[residual_rows]
        return x * w + residual, x * w + residual

    monkeypatch.setattr(qn, "gluon_gemm_allreduce_gemma_rmsnorm", gemm_ar)

    def attention(hidden_states, positions, x_scale=None, select_rows=None):
        out = hidden_states * 3  # o_proj's input (skipped: the layer runs it)
        seen["attn_rows"] = select_rows
        return out if select_rows is None else out[select_rows]

    x = torch.randn(10, 4)
    rows = torch.tensor([9, 4, 0])
    seen = {}
    early = _decoder_layer(qn, attention, fused, seen)
    early.select_rows_before_o_proj = True
    h, r = early.forward(None, x, None, select_rows=rows)
    assert seen["attn_rows"] is rows and seen["mlp_rows"] == 3
    seen.clear()
    late = _decoder_layer(qn, attention, fused, seen)
    h_ref, r_ref = late.forward(None, x, None, select_rows=rows)
    assert seen["attn_rows"] is None and seen["mlp_rows"] == 3
    assert torch.equal(h, h_ref) and torch.equal(r, r_ref)
    if fused:
        # The kernel gathers the residual; the late path did not hand it rows.
        assert calls[0] is rows and calls[1] is None


def _attention(qn, qkv, attn_out, q_size, kv_size, attn):
    layer = object.__new__(qn.Qwen3NextAttention)
    nn.Module.__init__(layer)
    layer.attn_output_gate = True
    layer.q_size, layer.kv_size = q_size, kv_size
    layer.num_heads = layer.num_kv_heads = layer.head_dim = 1
    layer.qkv_proj = lambda h, x_scale=None: qkv
    layer.qk_norm = lambda q, k: (q, k)
    layer.rotary_emb = None
    layer.fuse_qk_norm_rope_cache = False
    layer.attn = attn
    layer.use_fused_sigmoid_mul_quant = False
    layer.skip_out_proj = True
    return layer


@pytest.mark.parametrize("gate_in_attention", [False, True])
@pytest.mark.parametrize("taken", [False, True])
@pytest.mark.parametrize("select", [False, True])
def test_attention_gate_and_rows_wherever_they_run(
    monkeypatch, gate_in_attention, taken, select
):
    """The output is `(attn * sigmoid(gate))[rows]` whether the attention took
    the request (ATOM_MTP_ATTN_GATE_IN_REDUCE) or the layer applied it."""
    qn = _module("atom.models.qwen3_next")
    epi = _module("atom.model_ops.attn_output_epilogue")
    monkeypatch.setattr(qn, "try_fused_qk_norm_mrope", lambda *a: None)
    monkeypatch.setattr(qn, "try_mrope_qk_fused", lambda *a: (a[2], a[3]))
    rows_n, q_size, kv_size = 7, 1024, 256
    gen = torch.Generator(device=DEV).manual_seed(3)
    qkv = torch.randn(rows_n, 2 * q_size + 2 * kv_size, generator=gen, device=DEV).to(
        torch.bfloat16
    )
    attn_out = torch.randn(rows_n, q_size, generator=gen, device=DEV).to(torch.bfloat16)
    gate = qkv[:, :q_size]
    rows = torch.tensor([6, 2, 3], device=DEV) if select else None
    seen = {}

    def attn(q, k, v):
        req = epi.current_attention_output_epilogue()
        seen["req"] = req
        if req is None or not taken:
            return attn_out
        req.applied = True
        assert req.gate.data_ptr() == qkv.data_ptr()  # the strided view
        out = _gate_math(attn_out, req.gate)
        return out if req.rows is None else out[req.rows]

    layer = _attention(qn, qkv, attn_out, q_size, kv_size, attn)
    layer.gate_in_attention = gate_in_attention
    kwargs = {"select_rows": rows} if select else {}
    out = layer.forward(torch.zeros(rows_n), torch.zeros(rows_n, 8), **kwargs)
    ref = _gate_math(attn_out, gate)
    if select:
        ref = ref[rows]
    assert torch.equal(out, ref)
    assert (seen["req"] is not None) == gate_in_attention
    assert epi.current_attention_output_epilogue() is None  # scope closed


@pytest.mark.parametrize("rows", [[0], [5, 1, 4], list(range(64))])
def test_output_gate_rows_is_the_gate_then_the_gather(rows):
    sg = _module("atom.model_ops.sigmoid_gate")
    gen = torch.Generator(device=DEV).manual_seed(len(rows))
    n = max(rows) + 2
    qkvg = (3 * torch.randn(n, 2560, generator=gen, device=DEV)).to(torch.bfloat16)
    x = torch.randn(n, 1024, generator=gen, device=DEV).to(torch.bfloat16)
    idx = torch.tensor(rows, device=DEV)
    ref = sg.output_gate_mul(x, qkvg[:, :1024], True)[idx]
    assert torch.equal(sg.output_gate_rows(x, qkvg[:, :1024], idx), ref)


# ---------------------------------------------------------------------------
# ATOM_MTP_ATTN_GATE_IN_REDUCE: the unified-attention decode takes the request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("select", [False, True])
def test_unified_decode_hands_the_request_to_aiter(monkeypatch, select):
    mha = _module("atom.model_ops.attention_mha")
    epi = _module("atom.model_ops.attn_output_epilogue")
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "1")
    seen = {}

    def fake_unified(q, k, v, out, **kw):
        seen.update(kw, out=out)

    monkeypatch.setattr(mha, "unified_attention", fake_unified)
    monkeypatch.setattr(mha, "unified_attention_takes_output_epilogue", lambda: True)
    impl = object.__new__(mha.PagedAttentionImpl)
    nn.Module.__init__(impl)
    impl.kv_cache_dtype = "fp8_k4v8"
    impl.sliding_window = -1
    impl.scale, impl.k4v8, impl.kv_scale, impl.sinks = 1.0, True, None, None
    impl.use_flash_layout = False
    md = SimpleNamespace(
        cu_seqlens_q=None,
        context_lens=None,
        max_seqlen_q=5,
        max_seqlen_k=10,
        block_tables=None,
        equal_seqlen_q=True,
    )
    q = torch.zeros(10, 4, 256, dtype=torch.float8_e4m3fn, device=DEV)
    gate = torch.zeros(10, 1024, dtype=torch.bfloat16, device=DEV)
    rows = torch.tensor([4, 9], device=DEV) if select else None
    ctx = SimpleNamespace(attn_metadata=md, context=SimpleNamespace(is_draft=False))
    with epi.AttentionOutputEpilogue(gate, rows) as req:
        o = impl.paged_attention_unified(q, None, None, None, None, None, None, ctx)
        # A second attention in the same scope does not take it again.
        assert epi.current_attention_output_epilogue() is None
    assert req.applied
    assert seen["output_gate"] is gate and seen["output_rows"] is rows
    assert o is seen["out"] and o.shape == (2 if select else 10, 4, 256)
    # Without a request: no epilogue arguments.
    seen.clear()
    impl.paged_attention_unified(q, None, None, None, None, None, None, ctx)
    assert "output_gate" not in seen and seen["out"].shape == (10, 4, 256)


# ---------------------------------------------------------------------------
# ATOM_FUSED_MOE_TAIL_AR_NORM
# ---------------------------------------------------------------------------


def _ar_norm(ln, monkeypatch, calls):
    monkeypatch.setattr(ln, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(ln, "fused_allreduce_moe_tail_supported", lambda: True)

    def gated(routed, shared, gate, residual, weight, eps):
        calls.append((routed, shared, gate))
        s = _gate_math(shared, gate).float() + routed.float()
        s = s.to(routed.dtype) + residual
        return s * 2, s

    monkeypatch.setattr(ln, "fused_allreduce_moe_tail_gemma_rmsnorm", gated)
    norm = ln.GemmaRMSNorm(8, fused_allreduce=True)
    return norm


@pytest.mark.parametrize("write_residual", [False, True])
@pytest.mark.parametrize("select", [False, True])
def test_all_reducing_norm_takes_a_partial_gated_sum(
    monkeypatch, write_residual, select
):
    ln = _module("atom.model_ops.layernorm")
    mt = _module("atom.model_ops.moe_tail_norm")
    calls = []
    norm = _ar_norm(ln, monkeypatch, calls)
    assert norm.takes_tp_partial_moe_gated_sum and not norm.takes_moe_gated_sum
    t = [torch.randn(6, 8, device=DEV).to(torch.bfloat16) for _ in range(3)]
    gate = torch.randn(6, 3, device=DEV).to(torch.bfloat16)[:, 2:]
    moe = mt.MoEGatedSum(t[0], t[1], gate)
    rows = torch.tensor([5, 0], device=DEV) if select and not write_residual else None
    out = norm.forward_moe_tail(moe, t[2], rows=rows, write_residual=write_residual)
    s = (_gate_math(t[1], gate).float() + t[0].float()).to(torch.bfloat16) + t[2]
    if write_residual:
        assert torch.equal(out[0], s * 2) and torch.equal(out[1], s)
        # forward() routes a MoEGatedSum the same way.
        assert all(torch.equal(a, b) for a, b in zip(norm(moe, t[2]), out))
    else:
        ref = s * 2
        assert torch.equal(out, ref if rows is None else ref[rows])
    assert calls[0][0] is t[0] and calls[0][2] is gate


def _sparse_moe_block(qn, tp_size, reduce_results=False):
    mlp = object.__new__(qn.Qwen3NextSparseMoeBlock)
    nn.Module.__init__(mlp)
    mlp.shared_expert = nn.Identity()
    mlp._use_dual_stream = False
    mlp.tp_size = tp_size
    mlp.reduce_results = reduce_results
    return mlp


@pytest.mark.parametrize("tp_partial", [False, True])
@pytest.mark.parametrize("reduce_results", [False, True])
def test_defer_hands_a_partial_sum_only_to_an_all_reducing_norm(
    monkeypatch, tp_partial, reduce_results
):
    qn = _module("atom.models.qwen3_next")
    ln = _module("atom.model_ops.layernorm")
    monkeypatch.setenv("ATOM_FUSED_MOE_TAIL_NORM", "1")
    monkeypatch.setenv("ATOM_MEGA_COMBINE_TAIL", "1")
    norm = _ar_norm(ln, monkeypatch, [])
    layer = SimpleNamespace(mlp=_sparse_moe_block(qn, 4, reduce_results))
    deferred = qn.defer_moe_gated_sums([layer], norm, tp_partial=tp_partial)
    expect = tp_partial and not reduce_results
    assert deferred == int(expect) and layer.mlp.defer_gated_sum is expect
    # The MoE combine's tail is the TP1 path's only.
    assert layer.mlp.combine_tail_norm is None
    # A plain norm still refuses a TP-partial sum.
    plain = ln.GemmaRMSNorm(8)
    other = SimpleNamespace(mlp=_sparse_moe_block(qn, 4))
    assert qn.defer_moe_gated_sums([other], plain, tp_partial=True) == 0


def test_gemm_ar_falls_back_to_a_gathered_residual_past_its_rows(monkeypatch):
    ln = _module("atom.model_ops.layernorm")
    seen = {}

    def fused_ar(y, residual, weight, eps, gemma_norm=False):
        seen["residual"] = residual
        return y + residual, y + residual

    monkeypatch.setattr(ln, "tensor_model_parallel_fused_allreduce_rmsnorm", fused_ar)
    rows = ln._GLUON_GEMM_AR_MAX_TOKENS + 1
    x = torch.randn(rows, 512, device=DEV).to(torch.bfloat16)
    w = torch.randn(256, 512, device=DEV).to(torch.bfloat16)
    residual = torch.randn(3 * rows, 256, device=DEV).to(torch.bfloat16)
    pick = torch.arange(rows, device=DEV) * 3
    out, _ = ln.gluon_gemm_allreduce_gemma_rmsnorm(x, w, residual, None, 1e-6, pick)
    assert torch.equal(seen["residual"], residual[pick])
    assert out.shape == (rows, 256)
