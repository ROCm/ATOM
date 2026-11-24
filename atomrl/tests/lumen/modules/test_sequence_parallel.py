# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
Tests for sequence parallelism (SP) integration in parallel linear modules.

Covers:
  - SP flag propagation (disabled when tp_size <= 1)
  - ColumnParallelLinear uses gather_from_sequence_parallel_region in SP mode
  - RowParallelLinear uses reduce_scatter_to_sequence_parallel_region in SP mode
  - SP vs non-SP produce equivalent outputs (mocked single-rank)
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from atomrl.lumen.modules.parallel_linear import (
    LumenColumnParallelLinear,
    LumenRowParallelLinear,
)

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_config(sequence_parallel=False, tp_size=1):
    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=1,
        lumen_tp_comm_overlap=False,
    )


_COMMON_PATCHES = [
    mock.patch("atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=None),
    mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=1),
    mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0),
    mock.patch(
        "atomrl.lumen.modules.parallel_linear._use_sdma_from_args", return_value=False
    ),
    mock.patch(
        "atomrl.lumen.modules.parallel_linear.divide", side_effect=lambda a, b: a // b
    ),
    mock.patch("atomrl.lumen.modules.parallel_linear._initialize_affine_weight_gpu"),
    mock.patch(
        "atomrl.lumen.modules.parallel_linear.set_tensor_model_parallel_attributes"
    ),
    mock.patch(
        "atomrl.lumen.modules.parallel_linear.make_sharded_tensors_for_checkpoint",
        return_value={},
    ),
]


def _apply_common_patches(func):
    for p in reversed(_COMMON_PATCHES):
        func = p(func)
    return func


class TestSequenceParallelFlagPropagation:

    @_apply_common_patches
    def test_sp_disabled_when_tp_size_1(self, *_):
        config = _make_config(sequence_parallel=True, tp_size=1)
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert m.sequence_parallel is False, "SP should be disabled when tp_size=1"

    @_apply_common_patches
    def test_sp_enabled_when_tp_size_gt_1(self, *_):
        with mock.patch(
            "atomrl.lumen.modules.parallel_linear._pg_size", return_value=2
        ):
            config = _make_config(sequence_parallel=True, tp_size=2)
            m = LumenColumnParallelLinear(
                64,
                128,
                config=config,
                init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            )
        assert m.sequence_parallel is True


@_CUDA
class TestColumnParallelSP:

    @_apply_common_patches
    @mock.patch(
        "atomrl.lumen.modules.parallel_linear.copy_to_tensor_model_parallel_region",
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
    def test_sp_calls_gather_from_sp_region(self, *_):
        """In SP mode, ColumnParallel should call gather_from_sequence_parallel_region."""
        gather_mock = mock.MagicMock(side_effect=lambda x, **kw: x)
        with mock.patch(
            "atomrl.lumen.modules.parallel_linear.gather_from_sequence_parallel_region",
            gather_mock,
        ):
            config = _make_config(sequence_parallel=True, tp_size=2)
            with mock.patch(
                "atomrl.lumen.modules.parallel_linear._pg_size", return_value=2
            ):
                m = LumenColumnParallelLinear(
                    64,
                    128,
                    config=config,
                    init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
                )
            m.sequence_parallel = True
            torch.nn.init.kaiming_uniform_(m.weight)
            x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
            m(x)
            assert (
                gather_mock.called
            ), "gather_from_sequence_parallel_region should be called in SP mode"

    @_apply_common_patches
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
    def test_non_sp_does_not_call_gather_from_sp(self, *_):
        gather_mock = mock.MagicMock(side_effect=lambda x, **kw: x)
        with mock.patch(
            "atomrl.lumen.modules.parallel_linear.gather_from_sequence_parallel_region",
            gather_mock,
        ):
            config = _make_config(sequence_parallel=False)
            m = LumenColumnParallelLinear(
                64,
                128,
                config=config,
                init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            )
            torch.nn.init.kaiming_uniform_(m.weight)
            x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
            m(x)
            assert (
                not gather_mock.called
            ), "Should not call gather_from_sp_region when SP is off"


@_CUDA
class TestRowParallelSP:

    @_apply_common_patches
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
    def test_sp_calls_reduce_scatter(self, *_):
        """In SP mode, RowParallel should call reduce_scatter_to_sequence_parallel_region."""
        rs_mock = mock.MagicMock(side_effect=lambda x, **kw: x)
        with mock.patch(
            "atomrl.lumen.modules.parallel_linear.reduce_scatter_to_sequence_parallel_region",
            rs_mock,
        ):
            config = _make_config(sequence_parallel=True, tp_size=2)
            with mock.patch(
                "atomrl.lumen.modules.parallel_linear._pg_size", return_value=2
            ):
                m = LumenRowParallelLinear(
                    128,
                    64,
                    config=config,
                    init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
                    input_is_parallel=True,
                )
            m.sequence_parallel = True
            torch.nn.init.kaiming_uniform_(m.weight)
            # input_is_parallel=True with tp_size=2: input is pre-sharded to 128//2=64
            x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
            m(x)
            assert rs_mock.called, "reduce_scatter should be called in SP mode"

    @_apply_common_patches
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
        "atomrl.lumen.modules.parallel_linear.reduce_scatter_to_sequence_parallel_region",
        side_effect=lambda x, **kw: x,
    )
    def test_non_sp_calls_reduce(self, *_):
        reduce_mock = mock.MagicMock(side_effect=lambda x, **kw: x)
        with mock.patch(
            "atomrl.lumen.modules.parallel_linear.reduce_from_tensor_model_parallel_region",
            reduce_mock,
        ):
            config = _make_config(sequence_parallel=False)
            m = LumenRowParallelLinear(
                128,
                64,
                config=config,
                init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
                input_is_parallel=True,
            )
            torch.nn.init.kaiming_uniform_(m.weight)
            x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
            m(x)
            assert (
                reduce_mock.called
            ), "reduce_from_tp_region should be called when SP is off"


@_CUDA
class TestSPForwardEquivalence:
    """SP and non-SP produce equivalent output on single rank (mocked)."""

    @_apply_common_patches
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
    def test_column_sp_matches_non_sp(self, *_):
        """On single rank with mocked comms, SP and non-SP produce same output."""
        torch.manual_seed(42)
        config_nosp = _make_config(sequence_parallel=False)

        m_sp = LumenColumnParallelLinear(
            64,
            128,
            config=config_nosp,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        m_nosp = LumenColumnParallelLinear(
            64,
            128,
            config=config_nosp,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )

        # Force SP on one model (comms are mocked as identity, so output should match)
        m_sp.sequence_parallel = True

        with torch.no_grad():
            m_nosp.weight.copy_(m_sp.weight)
            if m_sp.bias is not None:
                m_nosp.bias.copy_(m_sp.bias)

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out_sp, _ = m_sp(x)
        out_nosp, _ = m_nosp(x)
        torch.testing.assert_close(out_sp, out_nosp)
