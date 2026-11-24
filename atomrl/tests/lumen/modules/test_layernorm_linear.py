# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear


def compute_snr(ref, test):
    ref_f, test_f = ref.float(), test.float()
    noise = ref_f - test_f
    signal_power = (ref_f**2).mean()
    noise_power = (noise**2).mean()
    if noise_power == 0:
        return float("inf")
    return 10 * torch.log10(signal_power / noise_power).item()


def _make_config(normalization="RMSNorm"):
    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        normalization=normalization,
        layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=False,
    )


@mock.patch("atomrl.lumen.modules.layernorm_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.layernorm_linear._pg_size", return_value=1)
@mock.patch("atomrl.lumen.modules.layernorm_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear._use_sdma_from_args", return_value=False
)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.divide", side_effect=lambda a, b: a // b
)
@mock.patch("atomrl.lumen.modules.layernorm_linear._initialize_affine_weight_gpu")
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.set_tensor_model_parallel_attributes"
)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.gather_from_sequence_parallel_region",
    side_effect=lambda x, **kw: x,
)
class TestLumenLayerNormLinearConstruction:
    def test_rmsnorm_mode(self, *_):
        config = _make_config("RMSNorm")
        m = LumenLayerNormLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.use_rmsnorm is True
        assert m.ln_bias is None

    def test_layernorm_mode(self, *_):
        config = _make_config("LayerNorm")
        m = LumenLayerNormLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.use_rmsnorm is False
        assert m.ln_bias is not None

    def test_repr(self, *_):
        config = _make_config("RMSNorm")
        m = LumenLayerNormLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        r = repr(m)
        assert "RMSNorm" in r
        assert "64" in r
        assert "128" in r


@mock.patch("atomrl.lumen.modules.layernorm_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.layernorm_linear._pg_size", return_value=1)
@mock.patch("atomrl.lumen.modules.layernorm_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear._use_sdma_from_args", return_value=False
)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.divide", side_effect=lambda a, b: a // b
)
@mock.patch("atomrl.lumen.modules.layernorm_linear._initialize_affine_weight_gpu")
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.set_tensor_model_parallel_attributes"
)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.gather_from_sequence_parallel_region",
    side_effect=lambda x, **kw: x,
)
class TestLumenLayerNormLinearForward:
    def test_forward_shape(self, *_):
        config = _make_config("RMSNorm")
        m = LumenLayerNormLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out, _ = m(x)
        assert out.shape == (4, 128)

    def test_forward_rmsnorm_correctness(self, *_):
        from atomrl.lumen.ops.normalization.rmsnorm import rmsnorm

        config = _make_config("RMSNorm")
        m = LumenLayerNormLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.cuda.manual_seed(42)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out, _ = m(x)
        ln_out = rmsnorm(x, m.ln_weight, m.ln_eps)
        ref = F.linear(ln_out, m.weight, m.bias)
        snr = compute_snr(ref, out)
        assert snr > 10, f"SNR {snr:.1f} dB"

    def test_forward_layernorm_correctness(self, *_):
        from atomrl.lumen.ops.normalization.layernorm import layernorm

        config = _make_config("LayerNorm")
        m = LumenLayerNormLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.cuda.manual_seed(42)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out, _ = m(x)
        ln_out = layernorm(x, m.ln_weight, m.ln_bias, m.ln_eps)
        ref = F.linear(ln_out, m.weight, m.bias)
        snr = compute_snr(ref, out)
        assert snr > 10, f"SNR {snr:.1f} dB"


@mock.patch("atomrl.lumen.modules.layernorm_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.layernorm_linear._pg_size", return_value=1)
@mock.patch("atomrl.lumen.modules.layernorm_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear._use_sdma_from_args", return_value=False
)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.divide", side_effect=lambda a, b: a // b
)
@mock.patch("atomrl.lumen.modules.layernorm_linear._initialize_affine_weight_gpu")
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.set_tensor_model_parallel_attributes"
)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
@mock.patch(
    "atomrl.lumen.modules.layernorm_linear.gather_from_sequence_parallel_region",
    side_effect=lambda x, **kw: x,
)
class TestLumenLayerNormLinearFP8:
    def test_enable_fp8(self, *_):
        config = _make_config("RMSNorm")
        m = LumenLayerNormLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.scaling_type == "none"
        m.enable_fp8()
        assert m.scaling_type == "dynamic"
        assert m.scaling_manager is not None
