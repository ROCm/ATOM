# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for MiniMax-M3 sparse PA -> prequantized o_proj."""

from types import SimpleNamespace

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

    def forward_quantized(self, q, k, v, positions, qkv=None):
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


def test_sparse_decode_keeps_a_non_cluster8_split_for_the_fp8_output(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        base_attention, "run_pa_decode", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(base_attention, "_flydsl_output_quant_option", lambda _: True)
    monkeypatch.setattr(
        "aiter.ops.triton.gluon.pa_decode_gluon.get_recommended_splits",
        lambda *_: 4,
    )
    rows = 128
    q = torch.randn(rows, 16, 128, dtype=torch.bfloat16)
    args = (
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
    )
    quant = {
        "quantized_output": torch.empty((rows, 16, 128), dtype=torch.uint8),
        "output_scale": torch.empty((rows, 1), dtype=torch.float32),
    }

    sparse_attn.sparse_pa_per_row(*args, **quant, quantized_direct=False)
    assert captured["max_context_partition_num"] == 4
    assert captured["plan_partition_cap"] == 4

    sparse_attn.sparse_pa_per_row(*args, **quant)
    assert captured["max_context_partition_num"] == 1
