# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from conftest import attention_ref, compute_snr


class _MockAttnMaskType:
    causal = 1
    no_mask = 0


def _sbhd_to_bshd(t):
    return t.permute(1, 0, 2, 3).contiguous()


def _make_config(num_heads=8, num_kv_heads=8, kv_channels=64, qk_rope_head_dim=0, tp=1):
    return SimpleNamespace(
        num_attention_heads=num_heads,
        num_query_groups=num_kv_heads,
        kv_channels=kv_channels,
        qk_rope_head_dim=qk_rope_head_dim,
        tensor_model_parallel_size=tp,
        apply_query_key_layer_scaling=False,
        attention_dropout=0.0,
        context_parallel_size=1,
    )


def _make_args(backend="aiter_triton"):
    return SimpleNamespace(
        lumen_attn_backend=backend,
        lumen_fp8_quant_type="blockwise",
        lumen_fp8_attn="none",
        mxfp8_block_m_fwd=128,
        mxfp8_block_n_fwd=128,
        mxfp8_block_m_dq_bwd=128,
        mxfp8_block_n_dq_bwd=128,
        mxfp8_block_m_dkv_bwd=128,
        mxfp8_block_n_dkv_bwd=128,
        mxfp8_quant_block_size=128,
        grad_quant_type=None,
    )


def _patched_megatron_init(self, config):
    torch.nn.Module.__init__(self)
    self.config = config


@mock.patch("atomrl.lumen.modules.attention_mla.AttnMaskType", _MockAttnMaskType)
@mock.patch(
    "atomrl.lumen.modules.attention_mla.divide", side_effect=lambda a, b: a // b
)
@mock.patch(
    "atomrl.lumen.modules.attention_mla.MegatronModule.__init__", _patched_megatron_init
)
@mock.patch(
    "atomrl.lumen.modules.attention_mla.get_args", side_effect=lambda: _make_args()
)
class TestLumenDotProductAttentionMLAConstruction:
    def test_construction_equal_kv_dims(self, *_):
        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        config = _make_config()
        attn = LumenDotProductAttentionMLA(
            config=config,
            layer_number=1,
            attn_mask_type=_MockAttnMaskType.no_mask,
            attention_type="self",
            k_channels=64,
            v_channels=64,
        )
        assert attn.k_head_dim == 64
        assert attn.v_head_dim == 64
        assert attn.k_head_dim == attn.v_head_dim

    def test_construction_different_kv_dims(self, *_):
        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        config = _make_config()
        attn = LumenDotProductAttentionMLA(
            config=config,
            layer_number=1,
            attn_mask_type=_MockAttnMaskType.no_mask,
            attention_type="self",
            k_channels=96,
            v_channels=64,
        )
        assert attn.k_head_dim == 96
        assert attn.v_head_dim == 64

    def test_construction_from_config(self, *_):
        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        config = _make_config(kv_channels=64, qk_rope_head_dim=32)
        attn = LumenDotProductAttentionMLA(
            config=config,
            layer_number=1,
            attn_mask_type=_MockAttnMaskType.no_mask,
            attention_type="self",
        )
        assert attn.k_head_dim == 96
        assert attn.v_head_dim == 64


@mock.patch("atomrl.lumen.modules.attention_mla.AttnMaskType", _MockAttnMaskType)
@mock.patch(
    "atomrl.lumen.modules.attention_mla.divide", side_effect=lambda a, b: a // b
)
@mock.patch(
    "atomrl.lumen.modules.attention_mla.MegatronModule.__init__", _patched_megatron_init
)
@mock.patch(
    "atomrl.lumen.modules.attention_mla.get_args", side_effect=lambda: _make_args()
)
class TestLumenDotProductAttentionMLAForward:
    def test_forward_equal_dims_shape(self, *_):
        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        config = _make_config()
        attn = (
            LumenDotProductAttentionMLA(
                config=config,
                layer_number=1,
                attn_mask_type=_MockAttnMaskType.no_mask,
                attention_type="self",
                k_channels=64,
                v_channels=64,
            )
            .cuda()
            .eval()
        )
        sq, b, h, d = 64, 2, 8, 64
        torch.cuda.manual_seed(42)
        q = torch.randn(sq, b, h, d, device="cuda", dtype=torch.bfloat16) * 0.02
        k = torch.randn(sq, b, h, d, device="cuda", dtype=torch.bfloat16) * 0.02
        v = torch.randn(sq, b, h, d, device="cuda", dtype=torch.bfloat16) * 0.02
        out = attn(q, k, v, attention_mask=None)
        assert out.shape == (64, 2, 512)

    def test_forward_different_dims_shape(self, *_):
        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        config = _make_config()
        attn = (
            LumenDotProductAttentionMLA(
                config=config,
                layer_number=1,
                attn_mask_type=_MockAttnMaskType.no_mask,
                attention_type="self",
                k_channels=96,
                v_channels=64,
            )
            .cuda()
            .eval()
        )
        sq, b, h, h_kv, d_k, d_v = 64, 2, 8, 8, 96, 64
        torch.cuda.manual_seed(42)
        q = torch.randn(sq, b, h, d_k, device="cuda", dtype=torch.bfloat16) * 0.02
        k = torch.randn(sq, b, h_kv, d_k, device="cuda", dtype=torch.bfloat16) * 0.02
        v = torch.randn(sq, b, h_kv, d_v, device="cuda", dtype=torch.bfloat16) * 0.02
        out = attn(q, k, v, attention_mask=None)
        assert out.shape == (64, 2, 512)

    def test_forward_equal_dims_snr(self, *_):
        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        config = _make_config()
        attn = (
            LumenDotProductAttentionMLA(
                config=config,
                layer_number=1,
                attn_mask_type=_MockAttnMaskType.no_mask,
                attention_type="self",
                k_channels=64,
                v_channels=64,
            )
            .cuda()
            .eval()
        )
        sq, b, h, d = 64, 2, 8, 64
        sm_scale = 1.0 / (d**0.5)
        torch.cuda.manual_seed(42)
        q = torch.randn(sq, b, h, d, device="cuda", dtype=torch.bfloat16) * 0.02
        k = torch.randn(sq, b, h, d, device="cuda", dtype=torch.bfloat16) * 0.02
        v = torch.randn(sq, b, h, d, device="cuda", dtype=torch.bfloat16) * 0.02
        out = attn(q, k, v, attention_mask=None)
        q_bshd = _sbhd_to_bshd(q)
        k_bshd = _sbhd_to_bshd(k)
        v_bshd = _sbhd_to_bshd(v)
        out_ref = attention_ref(q_bshd, k_bshd, v_bshd, sm_scale)
        out_reshaped = out.reshape(sq, b, h, d).permute(1, 0, 2, 3)
        snr = compute_snr(out_ref, out_reshaped)
        assert snr > 15, f"equal dims SNR: {snr:.1f} dB"

    def test_v_padding_applied(self, *_):
        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        config = _make_config()
        attn = (
            LumenDotProductAttentionMLA(
                config=config,
                layer_number=1,
                attn_mask_type=_MockAttnMaskType.no_mask,
                attention_type="self",
                k_channels=96,
                v_channels=64,
            )
            .cuda()
            .eval()
        )
        sq, b, h, h_kv, d_k, d_v = 64, 2, 8, 8, 96, 64
        sm_scale = 1.0 / (d_k**0.5)
        torch.cuda.manual_seed(42)
        q = torch.randn(sq, b, h, d_k, device="cuda", dtype=torch.bfloat16) * 0.02
        k = torch.randn(sq, b, h_kv, d_k, device="cuda", dtype=torch.bfloat16) * 0.02
        v = torch.randn(sq, b, h_kv, d_v, device="cuda", dtype=torch.bfloat16) * 0.02
        out = attn(q, k, v, attention_mask=None)
        q_bshd = _sbhd_to_bshd(q)
        k_bshd = _sbhd_to_bshd(k)
        v_bshd = _sbhd_to_bshd(v)
        out_ref = attention_ref(q_bshd, k_bshd, v_bshd, sm_scale)
        out_reshaped = out.reshape(sq, b, h, d_v).permute(1, 0, 2, 3)
        snr = compute_snr(out_ref, out_reshaped)
        assert snr > 15, f"V padding SNR: {snr:.1f} dB"


@mock.patch("atomrl.lumen.modules.attention_mla.AttnMaskType", _MockAttnMaskType)
@mock.patch(
    "atomrl.lumen.modules.attention_mla.divide", side_effect=lambda a, b: a // b
)
@mock.patch(
    "atomrl.lumen.modules.attention_mla.MegatronModule.__init__", _patched_megatron_init
)
@mock.patch(
    "atomrl.lumen.modules.attention_mla.get_args", side_effect=lambda: _make_args()
)
class TestLumenDotProductAttentionMLABenchmark:
    def test_forward_throughput(self, *_):
        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        config = _make_config()
        attn = (
            LumenDotProductAttentionMLA(
                config=config,
                layer_number=1,
                attn_mask_type=_MockAttnMaskType.no_mask,
                attention_type="self",
                k_channels=64,
                v_channels=64,
            )
            .cuda()
            .eval()
        )
        sq, b, h, d = 64, 2, 8, 64
        torch.cuda.manual_seed(42)
        q = torch.randn(sq, b, h, d, device="cuda", dtype=torch.bfloat16) * 0.02
        k = torch.randn(sq, b, h, d, device="cuda", dtype=torch.bfloat16) * 0.02
        v = torch.randn(sq, b, h, d, device="cuda", dtype=torch.bfloat16) * 0.02
        for _ in range(3):
            attn(q, k, v, attention_mask=None)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 10
        start.record()
        for _ in range(iters):
            attn(q, k, v, attention_mask=None)
        end.record()
        torch.cuda.synchronize()
        avg_ms = start.elapsed_time(end) / iters
        assert avg_ms >= 0


class TestCpParamBundleUseSdma:
    @mock.patch("atomrl.lumen.modules.attention_mla.AttnMaskType", _MockAttnMaskType)
    @mock.patch(
        "atomrl.lumen.modules.attention_mla.divide", side_effect=lambda a, b: a // b
    )
    @mock.patch(
        "atomrl.lumen.modules.attention_mla.MegatronModule.__init__",
        _patched_megatron_init,
    )
    @mock.patch(
        "atomrl.lumen.modules.attention_mla.get_args", return_value=_make_args()
    )
    @mock.patch("atomrl.lumen.modules.attention_mla.parallel_state")
    def test_cp_param_bundle_includes_use_sdma(self, mock_ps, *_):
        mock_ps.get_context_parallel_group.return_value = "fake_cp_group"

        from atomrl.lumen.modules.attention_mla import LumenDotProductAttentionMLA

        config = _make_config()
        config.context_parallel_size = 2
        attn = LumenDotProductAttentionMLA(
            config=config,
            layer_number=1,
            attn_mask_type=_MockAttnMaskType.causal,
            attention_type="self",
            k_channels=64,
            v_channels=64,
        )

        with (
            mock.patch(
                "atomrl.lumen.modules.attention_mla._use_sdma_from_args",
                return_value=True,
            ),
            mock.patch("atomrl.lumen.modules.attention_mla.attention") as mock_attn,
        ):
            mock_attn.return_value = torch.randn(2, 4, 8, 64, device="cuda")
            sq, b, np_, hn = 4, 2, 8, 64
            q = torch.randn(sq, b, np_, hn, device="cuda")
            k = torch.randn(sq, b, 8, hn, device="cuda")
            v = torch.randn(sq, b, 8, hn, device="cuda")
            attn(q, k, v, attention_mask=None)

            _, kwargs = mock_attn.call_args
            bundle = kwargs["cp_param_bundle"]
            assert bundle is not None
            assert bundle["use_sdma"] is True
            assert bundle["cp_group"] == "fake_cp_group"
