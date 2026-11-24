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
from atomrl.lumen.modules.parallel_linear import (
    LumenColumnParallelLinear,
    LumenRowParallelLinear,
)


def compute_snr(ref, test):
    ref_f, test_f = ref.float(), test.float()
    noise = ref_f - test_f
    signal_power = (ref_f**2).mean()
    noise_power = (noise**2).mean()
    if noise_power == 0:
        return float("inf")
    return 10 * torch.log10(signal_power / noise_power).item()


def _make_config():
    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
    )


@mock.patch("atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=2)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._use_sdma_from_args", return_value=False
)
@mock.patch("atomrl.lumen.modules.parallel_linear._initialize_affine_weight_cpu")
def test_column_parallel_cpu_init_preserves_partition_stride(*_):
    config = _make_config()
    config.perform_initialization = True
    config.use_cpu_initialization = True
    module = LumenColumnParallelLinear(
        64,
        128,
        config=config,
        init_method=lambda weight: weight.zero_(),
        bias=False,
        stride=2,
    )

    assert module.weight.tensor_model_parallel is True
    assert module.weight.partition_dim == 0
    assert module.weight.partition_stride == 2


@mock.patch("atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=1)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._use_sdma_from_args", return_value=False
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.divide", side_effect=lambda a, b: a // b
)
@mock.patch("atomrl.lumen.modules.parallel_linear._initialize_affine_weight_gpu")
@mock.patch("atomrl.lumen.modules.parallel_linear.set_tensor_model_parallel_attributes")
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.copy_to_tensor_model_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.gather_from_sequence_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.gather_from_tensor_model_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.reduce_from_tensor_model_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.reduce_scatter_to_sequence_parallel_region",
    side_effect=lambda x, **kw: x,
)
class TestLumenColumnParallelLinear:
    def test_construction(self, *_):
        config = _make_config()
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.weight.shape == (128, 64)
        assert m.bias is not None
        assert m.bias.shape == (128,)
        assert m.output_size_per_partition == 128

    def test_forward_shape(self, *_):
        config = _make_config()
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out, _ = m(x)
        assert out.shape == (4, 128)

    def test_forward_correctness(self, *_):
        config = _make_config()
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.cuda.manual_seed(42)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out, _ = m(x)
        ref = F.linear(x, m.weight, m.bias)
        snr = compute_snr(ref, out)
        assert snr > 20, f"SNR {snr:.1f} dB"

    def test_forward_no_bias(self, *_):
        config = _make_config()
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            bias=False,
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out, _ = m(x)
        assert out.shape == (4, 128)
        assert m.bias is None

    def test_enable_fp8(self, *_):
        config = _make_config()
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.scaling_type == "none"
        m.enable_fp8()
        assert m.scaling_type == "dynamic"

    def test_repr(self, *_):
        config = _make_config()
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        r = repr(m)
        assert "64" in r
        assert "128" in r


@mock.patch("atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=1)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._use_sdma_from_args", return_value=False
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.divide", side_effect=lambda a, b: a // b
)
@mock.patch("atomrl.lumen.modules.parallel_linear._initialize_affine_weight_gpu")
@mock.patch("atomrl.lumen.modules.parallel_linear.set_tensor_model_parallel_attributes")
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.copy_to_tensor_model_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.gather_from_sequence_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.gather_from_tensor_model_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.reduce_from_tensor_model_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.reduce_scatter_to_sequence_parallel_region",
    side_effect=lambda x, **kw: x,
)
class TestLumenRowParallelLinear:
    def test_construction(self, *_):
        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.weight.shape == (128, 64)
        assert m.input_size_per_partition == 64

    def test_forward_shape(self, *_):
        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out, _ = m(x)
        assert out.shape == (4, 128)

    def test_forward_correctness(self, *_):
        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.cuda.manual_seed(42)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out, _ = m(x)
        ref = F.linear(x, m.weight, m.bias)
        snr = compute_snr(ref, out)
        assert snr > 20, f"SNR {snr:.1f} dB"

    def test_enable_fp8(self, *_):
        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.scaling_type == "none"
        m.enable_fp8()
        assert m.scaling_type == "dynamic"

    def test_repr(self, *_):
        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        r = repr(m)
        assert "64" in r
        assert "128" in r


@mock.patch("atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=1)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._use_sdma_from_args", return_value=False
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.divide", side_effect=lambda a, b: a // b
)
@mock.patch("atomrl.lumen.modules.parallel_linear._initialize_affine_weight_gpu")
@mock.patch("atomrl.lumen.modules.parallel_linear.set_tensor_model_parallel_attributes")
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.copy_to_tensor_model_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.gather_from_sequence_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.gather_from_tensor_model_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.reduce_from_tensor_model_parallel_region",
    side_effect=lambda x, **kw: x,
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear.reduce_scatter_to_sequence_parallel_region",
    side_effect=lambda x, **kw: x,
)
class TestParallelLinearBenchmark:
    def test_column_parallel_throughput(self, *_):
        config = _make_config()
        m = LumenColumnParallelLinear(
            256,
            512,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        x = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 10
        start.record()
        for _ in range(iters):
            m(x)
        end.record()
        torch.cuda.synchronize()
        avg_ms = start.elapsed_time(end) / iters
        assert avg_ms > 0

    def test_row_parallel_throughput(self, *_):
        config = _make_config()
        m = LumenRowParallelLinear(
            256,
            512,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        x = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 10
        start.record()
        for _ in range(iters):
            m(x)
        end.record()
        torch.cuda.synchronize()
        avg_ms = start.elapsed_time(end) / iters
        assert avg_ms > 0
