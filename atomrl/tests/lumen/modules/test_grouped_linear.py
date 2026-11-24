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
from atomrl.lumen.modules.grouped_linear import (
    LumenColumnParallelGroupedLinear,
    LumenGroupedLinear,
    LumenRowParallelGroupedLinear,
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
        expert_model_parallel_size=1,
    )


@mock.patch("atomrl.lumen.modules.grouped_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.grouped_linear._pg_size", return_value=1)
@mock.patch(
    "atomrl.lumen.modules.grouped_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
class TestLumenGroupedLinearConstruction:
    def test_construction_basic(self, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            4,
            32,
            64,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert len(m.weights) == 4
        for w in m.weights:
            assert w.shape == (64, 32)
        assert m.biases is not None
        assert len(m.biases) == 4
        for b in m.biases:
            assert b.shape == (64,)

    def test_construction_column_parallel_grouped(self, *_):
        config = _make_config()
        m = LumenColumnParallelGroupedLinear(
            4,
            32,
            64,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert len(m.weights) == 4
        for w in m.weights:
            assert w.shape == (64, 32)

    def test_construction_row_parallel_grouped(self, *_):
        config = _make_config()
        m = LumenRowParallelGroupedLinear(
            4,
            32,
            64,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert len(m.weights) == 4
        for w in m.weights:
            assert w.shape == (64, 32)

    def test_construction_no_bias(self, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            4,
            32,
            64,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            bias=False,
        )
        assert m.biases is None

    def test_skip_bias_add(self, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            2,
            16,
            32,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            skip_bias_add=True,
        )
        for w in m.weights:
            torch.nn.init.kaiming_uniform_(w)
        x = torch.randn(10, 16, device="cuda", dtype=torch.bfloat16)
        m_splits = [10, 0]
        out, out_bias = m(x, m_splits)
        assert out.shape == (10, 32)
        assert out_bias is not None
        assert out_bias.shape == (2, 32)


@mock.patch("atomrl.lumen.modules.grouped_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.grouped_linear._pg_size", return_value=1)
@mock.patch(
    "atomrl.lumen.modules.grouped_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
class TestLumenGroupedLinearForward:
    def test_forward_single_expert(self, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            4,
            32,
            64,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        for w in m.weights:
            torch.nn.init.kaiming_uniform_(w)
        x = torch.randn(10, 32, device="cuda", dtype=torch.bfloat16)
        m_splits = [10, 0, 0, 0]
        out, _ = m(x, m_splits)
        assert out.shape == (10, 64)

    def test_forward_all_experts(self, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            4,
            32,
            64,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        for w in m.weights:
            torch.nn.init.kaiming_uniform_(w)
        x = torch.randn(14, 32, device="cuda", dtype=torch.bfloat16)
        m_splits = [5, 3, 4, 2]
        out, _ = m(x, m_splits)
        assert out.shape == (14, 64)

    def test_forward_correctness_vs_manual(self, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            4,
            32,
            64,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        for w in m.weights:
            torch.nn.init.kaiming_uniform_(w)
        torch.cuda.manual_seed(42)
        x = torch.randn(14, 32, device="cuda", dtype=torch.bfloat16)
        m_splits = [5, 3, 4, 2]
        out, _ = m(x, m_splits)
        ref_parts = []
        offset = 0
        for i, count in enumerate(m_splits):
            if count == 0:
                continue
            xi = x[offset : offset + count]
            yi = F.linear(xi, m.weights[i], m.biases[i])
            ref_parts.append(yi)
            offset += count
        ref = torch.cat(ref_parts, dim=0)
        snr = compute_snr(ref, out)
        assert snr > 20, f"SNR {snr:.1f} dB"

    def test_forward_empty_input(self, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            4,
            32,
            64,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        x = torch.randn(0, 32, device="cuda", dtype=torch.bfloat16)
        m_splits = [0, 0, 0, 0]
        out, _ = m(x, m_splits)
        assert out.shape == (0, 64)


def _echo_sharded_state(state_dict, *args, **kwargs):
    return dict(state_dict)


@mock.patch("atomrl.lumen.modules.grouped_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.grouped_linear._pg_size", return_value=1)
@mock.patch("atomrl.lumen.modules.grouped_linear.replace_prefix_for_sharding")
@mock.patch(
    "atomrl.lumen.modules.grouped_linear.make_sharded_tensors_for_checkpoint",
    side_effect=_echo_sharded_state,
)
class TestLumenGroupedLinearShardedStateDict:
    def test_sharded_state_dict_returns_dict(self, mock_make_sharded, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            2,
            16,
            32,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        sd = m.sharded_state_dict()
        assert isinstance(sd, dict)
        assert mock_make_sharded.call_count == m.num_gemms


@mock.patch("atomrl.lumen.modules.grouped_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.grouped_linear._pg_size", return_value=1)
@mock.patch(
    "atomrl.lumen.modules.grouped_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
class TestLumenGroupedLinearFP8:
    def test_enable_fp8(self, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            2,
            16,
            32,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.scaling_type == "none"
        m.enable_fp8()
        assert m.scaling_type == "dynamic"
        assert m.scaling_manager is not None


@mock.patch("atomrl.lumen.modules.grouped_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.grouped_linear._pg_size", return_value=1)
@mock.patch(
    "atomrl.lumen.modules.grouped_linear.make_sharded_tensors_for_checkpoint",
    return_value={},
)
class TestLumenGroupedLinearBenchmark:
    def test_forward_throughput(self, *_):
        config = _make_config()
        m = LumenGroupedLinear(
            4,
            256,
            512,
            parallel_mode=None,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        for w in m.weights:
            torch.nn.init.kaiming_uniform_(w)
        x = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
        m_splits = [256, 256, 256, 256]
        for _ in range(3):
            m(x, m_splits)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 10
        start.record()
        for _ in range(iters):
            m(x, m_splits)
        end.record()
        torch.cuda.synchronize()
        avg_ms = start.elapsed_time(end) / iters
        assert avg_ms > 0
