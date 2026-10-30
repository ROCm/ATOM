# SPDX-License-Identifier: MIT
"""ATOM_FUSED_SIGMOID_GATE: the one-launch sigmoid gate against the eager ops.

The Triton kernel needs a GPU (tests_gpu_pending/ runs it). What is checked
here is everything around it: that its arithmetic -- written out below step
for step from `_gated_pointwise_kernel` -- is bit-identical to the eager
expressions Qwen3-Next runs, for the two shapes it is given; that
`can_fuse_sigmoid_gate` turns away inputs whose eager result it would not
reproduce; and that the helpers the model calls hand it the right tensors and
keep the eager path when the flag is off. The model-level tests need
`atom.models.qwen3_next` to import, which takes a Triton driver.
"""

import pytest
import torch
import torch.nn.functional as F

sigmoid_gate = pytest.importorskip(
    "atom.model_ops.sigmoid_gate",
    reason="sigmoid_gate imports aiter and triton",
    exc_type=ImportError,
)


def _qwen3_next():
    # Needs a Triton driver at import (aiter probes the arch); the helper tests
    # above the model-level ones do not.
    try:
        import atom.models.qwen3_next as module
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"atom.models.qwen3_next does not import here: {exc}")
    return module


def kernel_math(x, gate, residual=None):
    """`_gated_pointwise_kernel` (SILU=False) in torch, rounding for rounding:
    sigmoid in fp32 rounded to the gate dtype, the product rounded to x's,
    the residual added in fp32, the store rounding to x's dtype."""
    g = torch.sigmoid(gate.float()).to(gate.dtype).float()
    result = (x.float() * g).to(x.dtype).float()
    if residual is not None:
        result = result + residual.float()
    return result.to(x.dtype)


def _moe_inputs(rows, hidden, experts, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    logits = (torch.randn(rows, experts + 1, generator=gen) * 4).to(dtype)
    shared = torch.randn(rows, hidden, generator=gen).to(dtype)
    routed = torch.randn(rows, hidden, generator=gen).to(dtype)
    return logits, shared, routed


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("rows", [1, 7, 64])
def test_kernel_math_is_the_eager_shared_expert_combine(dtype, rows):
    logits, shared, routed = _moe_inputs(rows, 2048, 256, dtype)
    gate = logits[:, 256:]
    eager = F.sigmoid(gate) * shared + routed
    assert sigmoid_gate.can_fuse_sigmoid_gate(shared, gate, routed)
    assert torch.equal(kernel_math(shared, gate, routed), eager)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("rows", [1, 13])
def test_kernel_math_is_the_eager_attention_output_gate(dtype, rows):
    q_size, kv_size = 1024, 256
    gen = torch.Generator().manual_seed(1)
    qkv = (torch.randn(rows, 2 * q_size + 2 * kv_size, generator=gen) * 3).to(dtype)
    gate, _ = torch.split(qkv, [q_size, q_size + 2 * kv_size], dim=-1)
    attn = torch.randn(rows, q_size, generator=gen).to(dtype)
    eager = attn * torch.sigmoid(gate)
    assert sigmoid_gate.can_fuse_sigmoid_gate(attn, gate, None)
    assert torch.equal(kernel_math(attn, gate), eager)


def test_can_fuse_refuses_what_the_kernel_would_round_differently():
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    gate_row = torch.zeros(4, 1, dtype=torch.bfloat16)
    can = sigmoid_gate.can_fuse_sigmoid_gate
    assert can(x, gate_row, x)
    # Eager promotes: an fp32 gate or residual keeps fp32 intermediates.
    assert not can(x, gate_row.float(), x)
    assert not can(x, gate_row, x.float())
    assert not can(x.float(), gate_row.float(), x.float())
    # A gate that is neither per-row nor per-element.
    assert not can(x, torch.zeros(4, 2, dtype=torch.bfloat16), None)
    # Rows must be dense (the kernel strides rows, not columns).
    assert not can(x.t().contiguous().t(), gate_row, None)
    assert not can(x, torch.zeros(8, 4, dtype=torch.bfloat16).t(), None)
    assert not can(x, gate_row, torch.zeros(8, 4, dtype=torch.bfloat16).t())
    assert not can(x.view(2, 2, 8), gate_row, None)


def _moe_block(qwen3_next, logits, shared, routed, n_routed):
    block = object.__new__(qwen3_next.Qwen3NextSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.n_routed_experts = n_routed
    block.is_rocm_aiter_fusion_shared_expert_enabled = False
    block.tp_size = 1
    block.reduce_results = False
    block.gate = lambda h: logits
    block.experts = lambda hidden_states, router_logits: routed
    block.shared_expert = lambda h: shared
    return block


def _recording_fused(calls):
    def fused(x, gate, residual=None):
        calls.append((x, gate, residual))
        return kernel_math(x, gate, residual)

    return fused


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_expert_gate_add(monkeypatch, enabled):
    logits, shared, routed = _moe_inputs(9, 2048, 256, torch.bfloat16, 7)
    calls = []
    monkeypatch.setattr(sigmoid_gate, "fused_sigmoid_gate", _recording_fused(calls))
    gate = logits[:, 256:]
    out = sigmoid_gate.shared_expert_gate_add(shared, gate, routed, enabled)
    assert torch.equal(out, F.sigmoid(gate) * shared + routed)
    assert len(calls) == (1 if enabled else 0)
    # Inputs the kernel would round differently stay eager with the flag on.
    calls.clear()
    out = sigmoid_gate.shared_expert_gate_add(shared, gate.float(), routed, enabled)
    assert out.dtype == torch.float32 and calls == []


@pytest.mark.parametrize("enabled", [False, True])
def test_output_gate_mul(monkeypatch, enabled):
    gen = torch.Generator().manual_seed(9)
    qkv = torch.randn(4, 2560, generator=gen).to(torch.bfloat16)
    gate = qkv[:, :1024]
    attn = torch.randn(4, 1024, generator=gen).to(torch.bfloat16)
    calls = []
    monkeypatch.setattr(sigmoid_gate, "fused_sigmoid_gate", _recording_fused(calls))
    out = sigmoid_gate.output_gate_mul(attn, gate, enabled)
    assert torch.equal(out, attn * torch.sigmoid(gate))
    assert len(calls) == (1 if enabled else 0)


@pytest.mark.parametrize("enabled", [False, True])
def test_moe_block_combine_call_site(monkeypatch, enabled):
    qwen3_next = _qwen3_next()
    rows, hidden, n_routed = 5, 2048, 256
    logits, shared, routed = _moe_inputs(rows, hidden, n_routed, torch.bfloat16, 3)
    calls = []
    monkeypatch.setattr(qwen3_next, "ATOM_FUSED_SIGMOID_GATE", enabled)
    monkeypatch.setattr(sigmoid_gate, "fused_sigmoid_gate", _recording_fused(calls))
    hidden_states = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    out = _moe_block(qwen3_next, logits, shared, routed, n_routed).forward(
        hidden_states
    )

    expected = F.sigmoid(logits[:, n_routed:]) * shared + routed
    assert torch.equal(out, expected)
    if not enabled:
        assert calls == []
        return
    x, gate, residual = calls[0]
    assert x is shared and residual is routed
    assert gate.shape == (rows, 1) and torch.equal(gate, logits[:, n_routed:])


def _attention(qwen3_next, qkv, attn_out, q_size, kv_size):
    layer = object.__new__(qwen3_next.Qwen3NextAttention)
    torch.nn.Module.__init__(layer)
    layer.attn_output_gate = True
    layer.q_size, layer.kv_size = q_size, kv_size
    layer.num_heads = layer.num_kv_heads = layer.head_dim = 1
    layer.qkv_proj = lambda h, x_scale=None: qkv
    layer.qk_norm = lambda q, k: (q, k)
    layer.rotary_emb = None
    layer.fuse_qk_norm_rope_cache = False
    layer.attn = lambda q, k, v: attn_out
    layer.use_fused_sigmoid_mul_quant = False
    layer.skip_out_proj = False
    layer.o_proj = lambda x: x
    return layer


@pytest.mark.parametrize("enabled", [False, True])
def test_attention_output_gate_call_site(monkeypatch, enabled):
    qwen3_next = _qwen3_next()
    rows, q_size, kv_size = 3, 1024, 256
    gen = torch.Generator().manual_seed(5)
    qkv = torch.randn(rows, 2 * q_size + 2 * kv_size, generator=gen).to(torch.bfloat16)
    attn_out = torch.randn(rows, q_size, generator=gen).to(torch.bfloat16)
    calls = []
    monkeypatch.setattr(qwen3_next, "ATOM_FUSED_SIGMOID_GATE", enabled)
    monkeypatch.setattr(sigmoid_gate, "fused_sigmoid_gate", _recording_fused(calls))
    monkeypatch.setattr(qwen3_next, "try_fused_qk_norm_mrope", lambda *a: None)
    monkeypatch.setattr(qwen3_next, "try_mrope_qk_fused", lambda *a: (a[2], a[3]))
    layer = _attention(qwen3_next, qkv, attn_out, q_size, kv_size)
    out = layer.forward(torch.zeros(rows), torch.zeros(rows, 8))

    assert torch.equal(out, attn_out * torch.sigmoid(qkv[:, :q_size]))
    if not enabled:
        assert calls == []
        return
    x, gate, residual = calls[0]
    assert x is attn_out and residual is None
    # The gate is the strided view of the packed projection, not a copy.
    assert gate.data_ptr() == qkv.data_ptr() and gate.stride(0) == qkv.shape[1]


def test_flag_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_FUSED_SIGMOID_GATE", raising=False)
    assert envs.ATOM_FUSED_SIGMOID_GATE is False
    monkeypatch.setenv("ATOM_FUSED_SIGMOID_GATE", "1")
    assert envs.ATOM_FUSED_SIGMOID_GATE is True


def test_fake_matches_kernel_output_layout():
    x = torch.zeros(6, 16, dtype=torch.bfloat16)[:, :8]
    out = sigmoid_gate._fused_sigmoid_gate_fake(x, x[:, :1])
    assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous()


def test_module_dtype_semantics_guard():
    """The eager ops' result dtype is the 16-bit one for the inputs the kernel
    accepts; were it wider, `can_fuse_sigmoid_gate` must have said no."""
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    g = torch.zeros(2, 1, dtype=torch.bfloat16)
    assert (F.sigmoid(g) * x + x).dtype == torch.bfloat16
    assert (F.sigmoid(g.float()) * x + x).dtype == torch.float32
    assert not sigmoid_gate.can_fuse_sigmoid_gate(x, g.float(), x)
