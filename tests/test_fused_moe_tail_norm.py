# SPDX-License-Identifier: MIT
"""ATOM_FUSED_MOE_TAIL_NORM: the MoE block's gated shared-expert add folded into
the next Gemma RMSNorm (atom/model_ops/moe_tail_norm.py).

CPU: the kernel's arithmetic, written out step for step in torch, against the
eager ops it replaces; `can_fuse_moe_tail`; and the wiring -- which MoE blocks
defer (`defer_moe_gated_sums`), what they return, how GemmaRMSNorm takes a
MoEGatedSum, the MTP drafter's row selection and the proposer's hand-off.

GPU (skipped without one): the Triton kernel against the production kernels
(qwen4_exp `sigmoid_mul` + aiter `fused_qk_rmsnorm_group_quant`): the new
residual bitwise, the normed output within one bf16 ulp, the row subset equal
to the full output's rows, and a CUDA-graph replay equal to the eager call.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

mt = pytest.importorskip(
    "atom.model_ops.moe_tail_norm",
    reason="moe_tail_norm imports aiter and triton",
    exc_type=ImportError,
)


def ln():
    """`atom.model_ops.layernorm`, for the tests that need it: it probes the GPU
    arch through aiter at import, which a CPU runner may not do -- the
    arithmetic tests then still run."""
    try:
        import atom.model_ops.layernorm as module
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"atom.model_ops.layernorm does not import here: {exc}")
    return module


HIDDEN, EXPERTS, EPS = 2048, 256, 1e-6


def _qwen3_next():
    # Needs a Triton driver at import (aiter probes the arch).
    try:
        import atom.models.qwen3_next as module
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"atom.models.qwen3_next does not import here: {exc}")
    return module


def kernel_math(routed, shared, gate, residual, weight, eps, rows=None):
    """`_moe_tail_rmsnorm_kernel` in torch, rounding for rounding."""
    if gate is not None:
        g = torch.sigmoid(gate.float()).to(gate.dtype).float()
        shared = (shared.float() * g).to(shared.dtype)
    moe = (shared.float() + routed.float()).to(routed.dtype)
    x = moe.float() + residual.float()
    rstd = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / x.shape[-1] + eps)
    out = ((x * rstd) * (weight.float() + 1.0)).to(routed.dtype)
    res = x.to(routed.dtype)
    if rows is not None:
        out = out.index_select(0, rows)
    return out, res


def gemma_forward_static(weight, eps, x, residual):
    """GemmaRMSNorm.forward_static (bf16 / fp16 inputs): the reference norm."""
    x = x + residual
    residual = x
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * (1.0 + weight.float())).to(residual.dtype), residual


def _inputs(rows, dtype=torch.bfloat16, seed=0, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)

    def rn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=device) * scale).to(dtype)

    logits = rn(rows, EXPERTS + 1, scale=3.0)
    return SimpleNamespace(
        routed=rn(rows, HIDDEN, scale=0.5),
        shared=rn(rows, HIDDEN),
        logits=logits,
        gate=logits[:, EXPERTS:],
        residual=rn(rows, HIDDEN, scale=4.0),
        weight=rn(HIDDEN, scale=0.1),
    )


def _norm(weight):
    norm = ln().GemmaRMSNorm(weight.shape[0], eps=EPS)
    norm.weight.data = weight.clone()
    return norm


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("rows", [1, 9, 64])
def test_kernel_math_is_the_unfused_ops(dtype, rows):
    """The new residual is bitwise what `shared_expert_gate_add` then the
    norm's residual add store; the normed output is the Gemma RMSNorm of it."""
    t = _inputs(rows, dtype)
    out, res = kernel_math(t.routed, t.shared, t.gate, t.residual, t.weight, EPS)
    moe = F.sigmoid(t.gate) * t.shared + t.routed  # the eager MoE block output
    assert torch.equal(res, moe + t.residual)
    ref, _ = gemma_forward_static(t.weight, EPS, moe, t.residual)
    # forward_static normalizes the bf16-rounded sum, the kernels the fp32 one.
    torch.testing.assert_close(out.float(), ref.float(), rtol=1.6e-2, atol=1e-2)


def test_kernel_math_rows_are_the_full_outputs_rows():
    t = _inputs(12)
    rows = torch.tensor([3, 7, 11, 0])
    full, _ = kernel_math(t.routed, t.shared, t.gate, t.residual, t.weight, EPS)
    sub, _ = kernel_math(t.routed, t.shared, t.gate, t.residual, t.weight, EPS, rows)
    assert torch.equal(sub, full[rows])


def test_can_fuse_refuses_what_the_kernel_does_not_take():
    t = _inputs(4)
    can = mt.can_fuse_moe_tail
    moe = mt.MoEGatedSum(t.routed, t.shared, t.gate)
    assert can(moe, t.residual, t.weight)
    assert can(mt.MoEGatedSum(t.routed, t.shared, None), t.residual, t.weight)
    assert can(moe, t.residual, t.weight, torch.tensor([1, 2]))
    # One 16-bit dtype throughout (eager would promote a wider one).
    assert not can(
        mt.MoEGatedSum(t.routed, t.shared, t.gate.float()), t.residual, t.weight
    )
    assert not can(moe, t.residual.float(), t.weight)
    # Dense rows; a per-row gate; a hidden-wide weight; 1-D integer rows.
    assert not can(
        mt.MoEGatedSum(t.routed.t().contiguous().t(), t.shared, t.gate),
        t.residual,
        t.weight,
    )
    assert not can(
        mt.MoEGatedSum(t.routed, t.shared, t.logits[:, :2]), t.residual, t.weight
    )
    assert not can(moe, t.residual, t.weight[:16])
    assert not can(moe, t.residual, t.weight, torch.tensor([1.0]))
    assert not can(moe, t.residual, t.weight, torch.tensor([[1]]))


def test_fakes_match_the_outputs():
    t = _inputs(5)
    out, res = mt._moe_tail_rmsnorm_fake(
        t.routed, t.shared, t.gate, t.residual, t.weight, EPS
    )
    assert out.shape == res.shape == t.routed.shape and out.dtype == t.routed.dtype
    rows = torch.tensor([0, 4])
    sub = mt._moe_tail_rmsnorm_rows_fake(
        t.routed, t.shared, t.gate, t.residual, t.weight, EPS, rows
    )
    assert sub.shape == (2, HIDDEN)
    full = mt._moe_tail_rmsnorm_rows_fake(
        t.routed, t.shared, t.gate, t.residual, t.weight, EPS, None
    )
    assert full.shape == t.routed.shape


def test_flag_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_FUSED_MOE_TAIL_NORM", raising=False)
    assert envs.ATOM_FUSED_MOE_TAIL_NORM is False
    monkeypatch.setenv("ATOM_FUSED_MOE_TAIL_NORM", "1")
    assert envs.ATOM_FUSED_MOE_TAIL_NORM is True


# ---------------------------------------------------------------------------
# GemmaRMSNorm taking a MoEGatedSum
# ---------------------------------------------------------------------------


def _fake_kernels(monkeypatch, calls):
    def full(routed, shared, gate, residual, weight, eps):
        calls.append("full")
        return kernel_math(routed, shared, gate, residual, weight, eps)

    def rows_only(routed, shared, gate, residual, weight, eps, rows):
        calls.append("rows")
        return kernel_math(routed, shared, gate, residual, weight, eps, rows)[0]

    monkeypatch.setattr(ln(), "moe_tail_rmsnorm", full)
    monkeypatch.setattr(ln(), "moe_tail_rmsnorm_rows", rows_only)


def test_norm_folds_the_gated_sum(monkeypatch):
    calls = []
    _fake_kernels(monkeypatch, calls)
    t = _inputs(6)
    norm = _norm(t.weight)
    moe = mt.MoEGatedSum(t.routed, t.shared, t.gate)
    out, res = norm(moe, t.residual)
    ref_out, ref_res = kernel_math(
        t.routed, t.shared, t.gate, t.residual, t.weight, EPS
    )
    assert calls == ["full"] and torch.equal(out, ref_out) and torch.equal(res, ref_res)
    rows = torch.tensor([5, 1])
    sub = norm.forward_moe_tail(moe, t.residual, rows=rows, write_residual=False)
    assert calls[-1] == "rows" and torch.equal(sub, ref_out[rows])
    whole = norm.forward_moe_tail(moe, t.residual, write_residual=False)
    assert torch.equal(whole, ref_out)


def test_norm_falls_back_to_the_unfused_ops(monkeypatch):
    """Inputs the kernel does not take: the gated add, then the norm's own path."""
    calls = []
    _fake_kernels(monkeypatch, calls)
    t = _inputs(4)
    norm = _norm(t.weight)
    norm.forward_cuda = lambda x, r: ln().GemmaRMSNorm.forward_static(
        norm.weight.data, EPS, x, r
    )
    monkeypatch.setattr(
        ln(), "gated_sum", lambda m: F.sigmoid(m.gate_logits) * m.shared + m.routed
    )
    gate32 = t.gate.float()  # promotes: the kernel would round differently
    moe = mt.MoEGatedSum(t.routed, t.shared, gate32)
    out, res = norm(moe, t.residual)
    ref = ln().GemmaRMSNorm.forward_static(
        norm.weight.data, EPS, F.sigmoid(gate32) * t.shared + t.routed, t.residual
    )
    assert calls == [] and torch.equal(out, ref[0]) and torch.equal(res, ref[1])
    rows = torch.tensor([2])
    sub = norm.forward_moe_tail(moe, t.residual, rows=rows, write_residual=False)
    assert torch.equal(sub, ref[0][rows])


def test_quantizing_or_allreduce_norms_refuse_a_gated_sum():
    norm = ln().GemmaRMSNorm(16)
    assert norm.takes_moe_gated_sum
    norm.use_fused_quant = True
    assert not norm.takes_moe_gated_sum
    norm.use_fused_quant, norm.fused_allreduce = False, True
    assert not norm.takes_moe_gated_sum


# ---------------------------------------------------------------------------
# Which MoE blocks defer, and what they return
# ---------------------------------------------------------------------------


def _moe_block(qwen3_next, t=None, *, tp=1, dual=False, shared=True):
    block = object.__new__(qwen3_next.Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block.n_routed_experts = EXPERTS
    block.is_rocm_aiter_fusion_shared_expert_enabled = False
    block.tp_size = tp
    block.reduce_results = True
    block._use_dual_stream = dual
    if t is not None:
        block.gate = lambda h: t.logits
        block.experts = lambda hidden_states, router_logits: t.routed
    block.shared_expert = (lambda h: t.shared) if shared else None
    return block


def _layer(mlp, norm=None, layer_scale=False):
    layer = nn.Module()
    layer.mlp = mlp
    layer.input_layernorm = norm if norm is not None else ln().GemmaRMSNorm(16)
    layer.layer_scale = layer_scale
    return layer


def test_defer_links_each_block_to_the_norm_that_reads_it(monkeypatch):
    qn = _qwen3_next()
    monkeypatch.setattr(qn, "is_plugin_mode", lambda: False)
    quant_norm = ln().GemmaRMSNorm(16)
    quant_norm.use_fused_quant = True
    final = ln().GemmaRMSNorm(16)
    layers = [
        _layer(_moe_block(qn)),  # -> layers[1].input_layernorm: plain
        _layer(_moe_block(qn)),  # -> layers[2].input_layernorm: quantizing
        _layer(_moe_block(qn, tp=4), norm=quant_norm),  # TP-partial
        _layer(_moe_block(qn, dual=True)),  # side-stream shared expert
        _layer(_moe_block(qn, shared=False)),  # shared expert fused in routed
        _layer(_moe_block(qn), layer_scale=True),
        _layer(nn.Identity()),  # a dense MLP
        _layer(_moe_block(qn)),  # -> the final norm
    ]
    monkeypatch.delenv("ATOM_FUSED_MOE_TAIL_NORM", raising=False)
    assert qn.defer_moe_gated_sums(layers, final) == 0
    monkeypatch.setenv("ATOM_FUSED_MOE_TAIL_NORM", "1")
    assert qn.defer_moe_gated_sums(layers, final) == 2
    deferred = [getattr(ly.mlp, "defer_gated_sum", None) for ly in layers]
    assert deferred == [True, False, False, False, False, False, None, True]
    # A final norm that cannot take it keeps the last block summing.
    final.use_fused_quant = True
    fresh = [_layer(_moe_block(qn))]
    assert qn.defer_moe_gated_sums(fresh, final) == 0
    # Plugin frontends patch the decoder forward: never.
    monkeypatch.setattr(qn, "is_plugin_mode", lambda: True)
    assert qn.defer_moe_gated_sums([_layer(_moe_block(qn))], ln().GemmaRMSNorm(16)) == 0


@pytest.mark.parametrize("defer", [False, True])
def test_moe_block_returns_the_unsummed_parts_when_deferring(monkeypatch, defer):
    qn = _qwen3_next()
    t = _inputs(5)
    monkeypatch.setattr(qn, "ATOM_FUSED_SIGMOID_GATE", False)
    block = _moe_block(qn, t)
    block.defer_gated_sum = defer
    out = block.forward(torch.zeros(5, HIDDEN, dtype=torch.bfloat16))
    if not defer:
        assert torch.equal(out, F.sigmoid(t.gate) * t.shared + t.routed)
        return
    assert isinstance(out, mt.MoEGatedSum)
    assert out.routed is t.routed and out.shared is t.shared
    assert out.gate_logits.shape == (5, 1) and torch.equal(out.gate_logits, t.gate)
    # The router GEMM's last column as is (strided), not a copy.
    assert out.gate_logits.data_ptr() == t.logits[:, EXPERTS:].data_ptr()


# ---------------------------------------------------------------------------
# MTP drafter: final norm over the selected rows; the proposer's hand-off
# ---------------------------------------------------------------------------


def _mtp():
    try:
        import atom.models.qwen3_5_mtp as module
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"atom.models.qwen3_5_mtp does not import here: {exc}")
    return module


class _Identity(nn.Module):
    def forward(self, x, residual=None):
        return x


def _predictor(mtp, t, layer_out):
    pred = object.__new__(mtp.Qwen3_5MultiTokenPredictor)
    nn.Module.__init__(pred)
    pred.dual_pre_fc_norm = False
    pred.pre_fc_norm_embedding = _Identity()
    pred.pre_fc_norm_hidden = _Identity()
    pred._fc = lambda x: x[:, :HIDDEN]
    pred.num_mtp_layers = 1

    class _Layer(nn.Module):
        def forward(self, positions, hidden_states, residual):
            return layer_out, t.residual

    pred.layers = nn.ModuleList([_Layer()])
    pred.norm = _norm(t.weight)
    return pred


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("select", [False, True])
def test_mtp_final_norm_covers_the_selected_rows(monkeypatch, deferred, select):
    mtp = _mtp()
    calls = []
    _fake_kernels(monkeypatch, calls)
    t = _inputs(8)
    ref_out, _ = kernel_math(t.routed, t.shared, t.gate, t.residual, t.weight, EPS)
    if deferred:
        layer_out = mt.MoEGatedSum(t.routed, t.shared, t.gate)
    else:
        layer_out = F.sigmoid(t.gate) * t.shared + t.routed
        pred_norm_static = ln().GemmaRMSNorm.forward_static
        monkeypatch.setattr(
            ln().GemmaRMSNorm,
            "forward",
            lambda self, x, r=None: pred_norm_static(self.weight.data, EPS, x, r),
        )
        ref_out = pred_norm_static(t.weight, EPS, layer_out, t.residual)[0]
    pred = _predictor(mtp, t, layer_out)
    rows = torch.tensor([3, 7, 1]) if select else None
    zeros = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    out = pred.forward(
        None, torch.zeros(8), zeros, inputs_embeds=zeros, select_rows=rows
    )
    assert torch.equal(out, ref_out if rows is None else ref_out[rows])
    assert calls == (["rows"] if deferred else [])


def test_mtp_forward_passes_select_rows_through(monkeypatch):
    mtp = _mtp()
    seen = {}

    class _Model(nn.Module):
        def forward(self, *args, select_rows=None):
            seen["rows"] = select_rows
            return "out"

    draft = object.__new__(mtp.Qwen3_5MTP)
    nn.Module.__init__(draft)
    draft.model = _Model()
    rows = torch.tensor([1])
    assert draft.forward(None, None, None, select_rows=rows) == "out"
    assert seen["rows"] is rows
    draft.forward(None, None, None)
    assert seen["rows"] is None


def test_proposer_hands_over_rows_to_a_selecting_model():
    try:
        from atom.spec_decode import eagle_proposer as ep
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"atom.spec_decode.eagle_proposer does not import here: {exc}")

    prop = object.__new__(ep.EagleProposer)
    hidden = torch.randn(6, 4)
    rows = torch.tensor([5, 2])

    def ids(h, out):
        return out.fill_(h.shape[0])

    prop.model = SimpleNamespace(forward_selects_rows=False, compute_draft_ids=ids)
    assert prop._select_rows_kwargs(rows) == {}
    sel, _ = prop._step0_head(hidden, 2, last_token_indices=rows)
    assert torch.equal(sel, hidden[rows])
    prop.model.forward_selects_rows = True
    assert prop._select_rows_kwargs(rows)["select_rows"] is rows
    picked = hidden[rows]
    sel, _ = prop._step0_head(picked, 2, last_token_indices=rows)
    assert sel is picked


# ---------------------------------------------------------------------------
# GPU: the Triton kernel against the production kernels
# ---------------------------------------------------------------------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _production(t):
    """shared_expert_gate_add (sigmoid_mul) then GemmaRMSNorm.forward_cuda (aiter)."""
    from atom.model_ops.sigmoid_gate import shared_expert_gate_add

    moe = shared_expert_gate_add(t.shared, t.gate, t.routed, True)
    return _norm(t.weight).cuda().forward_cuda(moe, t.residual)


def _ulps(a, b):
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs().max().item()


@gpu
@pytest.mark.parametrize("rows", [1, 33, 512, 4099])
def test_kernel_is_the_production_tail(rows):
    t = _inputs(rows, seed=rows, device="cuda")
    ref_out, ref_res = _production(t)
    out, res = mt.moe_tail_rmsnorm(
        t.routed, t.shared, t.gate, t.residual, t.weight, EPS
    )
    assert torch.equal(res, ref_res)
    assert _ulps(out, ref_out) <= 1
    # And the torch statement of the kernel, likewise.
    km_out, km_res = kernel_math(t.routed, t.shared, t.gate, t.residual, t.weight, EPS)
    assert torch.equal(res, km_res) and _ulps(out, km_out) <= 1


@gpu
def test_kernel_rows_and_pre_gated_shared():
    t = _inputs(64, seed=3, device="cuda")
    out, _ = mt.moe_tail_rmsnorm(t.routed, t.shared, t.gate, t.residual, t.weight, EPS)
    for dtype in (torch.int64, torch.int32):
        rows = torch.tensor([63, 0, 17, 17, 5], device="cuda", dtype=dtype)
        sub = mt.moe_tail_rmsnorm_rows(
            t.routed, t.shared, t.gate, t.residual, t.weight, EPS, rows
        )
        assert torch.equal(sub, out[rows.long()])
    whole = mt.moe_tail_rmsnorm_rows(
        t.routed, t.shared, t.gate, t.residual, t.weight, EPS, None
    )
    assert torch.equal(whole, out)
    # gate=None: `shared` already gated, as sigmoid_mul rounds it.
    g = torch.sigmoid(t.gate.float()).to(t.gate.dtype).float()
    gated = (t.shared.float() * g).to(t.shared.dtype)
    out2, _ = mt.moe_tail_rmsnorm(t.routed, gated, None, t.residual, t.weight, EPS)
    assert torch.equal(out2, out)


@gpu
def test_kernel_in_a_cuda_graph_and_on_strided_rows():
    t = _inputs(96, seed=5, device="cuda")
    eager = mt.moe_tail_rmsnorm(t.routed, t.shared, t.gate, t.residual, t.weight, EPS)
    rows = torch.arange(3, 96, 4, device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = mt.moe_tail_rmsnorm(
            t.routed, t.shared, t.gate, t.residual, t.weight, EPS
        )
        captured_rows = mt.moe_tail_rmsnorm_rows(
            t.routed, t.shared, t.gate, t.residual, t.weight, EPS, rows
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], eager[0]) and torch.equal(captured[1], eager[1])
    assert torch.equal(captured_rows, eager[0][rows])
    # Row-strided inputs (views into wider buffers) and a misaligned row view.
    wide = torch.randn(96, HIDDEN + 64, device="cuda").bfloat16()
    wide[:, :HIDDEN] = t.routed
    routed_view = wide[:, :HIDDEN]
    out, res = mt.moe_tail_rmsnorm(
        routed_view, t.shared, t.gate, t.residual, t.weight, EPS
    )
    assert torch.equal(out, eager[0]) and torch.equal(res, eager[1])
    off = torch.empty(96 * HIDDEN + 1, device="cuda", dtype=torch.bfloat16)
    shifted = off[1:].view(96, HIDDEN)  # 2-byte aligned rows
    shifted.copy_(t.shared)
    out, res = mt.moe_tail_rmsnorm(t.routed, shifted, t.gate, t.residual, t.weight, EPS)
    assert torch.equal(out, eager[0]) and torch.equal(res, eager[1])


# ---------------------------------------------------------------------------
# ATOM_MEGA_COMBINE_TAIL: the tail handed to the MoE combine
# ---------------------------------------------------------------------------


def test_combine_tail_request_scope_nests_and_restores():
    assert mt.current_combine_tail_request() is None
    a = mt.CombineTailRequest(None, None, None, None, 1e-6, True)
    b = mt.CombineTailRequest(None, None, None, None, 1e-6, False)
    with mt.combine_tail_request(a):
        assert mt.current_combine_tail_request() is a
        with mt.combine_tail_request(b):
            assert mt.current_combine_tail_request() is b
        assert mt.current_combine_tail_request() is a
    assert mt.current_combine_tail_request() is None


def test_norm_passes_a_combine_normed_result_through():
    t = _inputs(3)
    norm = _norm(t.weight)
    done = mt.MoENormed(t.routed, t.residual)
    out, res = norm(done, t.shared)
    assert out is t.routed and res is t.residual


@pytest.mark.parametrize("answered", [False, True])
def test_moe_block_hands_its_tail_to_the_combine(monkeypatch, answered):
    """The shared expert runs first; the experts call sees the request (the
    norm's weight / eps, the residual, the gate column); an answered request
    comes back as MoENormed, an unanswered one as the MoEGatedSum fallback."""
    qn = _qwen3_next()
    t = _inputs(5)
    calls = []
    block = _moe_block(qn, t)
    block.shared_expert = lambda h: (calls.append("shared"), t.shared)[1]
    nrm = _norm(t.weight)
    object.__setattr__(block, "combine_tail_norm", nrm)
    block.combine_tail_writes_residual = True
    block.defer_gated_sum = True
    monkeypatch.setenv("ATOM_MEGA_COMBINE_TAIL_MAX_TOKENS", "1024")

    def experts(hidden_states, router_logits):
        calls.append("experts")
        req = mt.current_combine_tail_request()
        assert req.shared is t.shared and req.residual is t.residual
        assert req.weight is nrm.weight and req.eps == nrm.variance_epsilon
        assert torch.equal(req.gate, t.gate) and req.write_residual
        if answered:
            req.result = ("normed", "res")
        return t.routed

    block.experts = experts
    out = block.forward(torch.zeros(5, HIDDEN, dtype=torch.bfloat16), t.residual)
    assert calls == ["shared", "experts"]
    assert mt.current_combine_tail_request() is None
    if answered:
        assert out == mt.MoENormed("normed", "res")
    else:
        assert isinstance(out, mt.MoEGatedSum) and out.routed is t.routed
    # Above the row threshold (and without a residual) the block sums as before.
    monkeypatch.setenv("ATOM_MEGA_COMBINE_TAIL_MAX_TOKENS", "4")
    calls.clear()
    block.experts = lambda hidden_states, router_logits: (
        (
            calls.append("experts:no-request")
            if mt.current_combine_tail_request() is None
            else None
        ),
        t.routed,
    )[1]
    out = block.forward(torch.zeros(5, HIDDEN, dtype=torch.bfloat16), t.residual)
    assert isinstance(out, mt.MoEGatedSum)
    assert calls == ["experts:no-request", "shared"]  # the usual order


def test_defer_links_the_combine_tail_only_when_asked(monkeypatch):
    qn = _qwen3_next()
    monkeypatch.setattr(qn, "is_plugin_mode", lambda: False)
    monkeypatch.setenv("ATOM_FUSED_MOE_TAIL_NORM", "1")
    final = ln().GemmaRMSNorm(16)
    layers = [_layer(_moe_block(qn)), _layer(_moe_block(qn))]
    monkeypatch.delenv("ATOM_MEGA_COMBINE_TAIL", raising=False)
    qn.defer_moe_gated_sums(layers, final)
    assert all(ly.mlp.combine_tail_norm is None for ly in layers)
    monkeypatch.setenv("ATOM_MEGA_COMBINE_TAIL", "1")
    qn.defer_moe_gated_sums(layers, final)
    assert layers[0].mlp.combine_tail_norm is layers[1].input_layernorm
    assert layers[0].mlp.combine_tail_writes_residual
    assert layers[1].mlp.combine_tail_norm is final
    assert not layers[1].mlp.combine_tail_writes_residual
    # Not a registered submodule (no second copy of the next layer's weight).
    assert "combine_tail_norm" not in dict(layers[0].mlp.named_children())
    fresh = [_layer(_moe_block(qn))]
    qn.defer_moe_gated_sums(fresh, final, combine_tail=False)
    assert fresh[0].mlp.defer_gated_sum and fresh[0].mlp.combine_tail_norm is None


def test_mori_call_site_asks_for_the_tail_only_when_it_can(monkeypatch):
    try:
        from atom.model_ops.fused_moe import mori_v2_prepare_finalize as mpf
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"mori_v2_prepare_finalize does not import here: {exc}")
    pytest.importorskip("aiter.ops.flydsl.kernels.mega_moe_gfx1250.mega_moe")
    t = _inputs(4)
    hs = torch.zeros(4, HIDDEN, dtype=torch.bfloat16)
    assert mpf._combine_tail_kwargs(hs, "none") == {}  # no request
    req = mt.CombineTailRequest(t.shared, t.gate, t.residual, t.weight, 1e-6, False)
    with mt.combine_tail_request(req):
        kw = mpf._combine_tail_kwargs(hs, "none")
        tail = kw["combine_tail"]
        assert (
            tail.shared is t.shared and tail.gate is t.gate and not tail.write_residual
        )
        assert mpf._combine_tail_kwargs(hs, "mxfp4") == {}  # quantized combine wire
        assert mpf._combine_tail_kwargs(hs[:3], "none") == {}  # not the block's rows


# ---------------------------------------------------------------------------
# ATOM_SHARED_EXPERT_GATED_GEMM
# ---------------------------------------------------------------------------


def test_interleave_gate_up_pairs_rows():
    w = torch.arange(12.0).view(6, 2)  # gate rows 0-2, up rows 3-5
    out = mt.interleave_gate_up(w)
    assert torch.equal(out[0::2], w[:3]) and torch.equal(out[1::2], w[3:])


@pytest.mark.parametrize("gated", [False, True])
def test_shared_expert_mlp_uses_the_gated_gemm(monkeypatch, gated):
    qn = _qwen3_next()
    mlp = object.__new__(qn.Qwen3NextMLP)
    nn.Module.__init__(mlp)
    mlp.expert_gate = None
    mlp.gated_gate_up = gated
    w = torch.randn(8, 4)
    mlp.gate_up_proj = SimpleNamespace(weight=w)
    mlp.gate_up_proj = lambda x: x @ w.T
    mlp.gate_up_proj.weight = w  # type: ignore[attr-defined]
    mlp.act_fn = lambda gu: F.silu(gu[:, :4]) * gu[:, 4:]
    mlp.down_proj = lambda x: x * 2
    seen = []
    monkeypatch.setattr(
        qn, "silu_gate_up_gemm", lambda x, wi: (seen.append(wi), x[:, :4] + 1)[1]
    )
    x = torch.randn(3, 4)
    out = mlp.forward(x)
    if gated:
        assert seen == [w] and torch.equal(out, (x[:, :4] + 1) * 2)
    else:
        assert seen == [] and torch.allclose(out, mlp.act_fn(x @ w.T) * 2)
