# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for TP comm-GEMM overlap and async SDMA APIs.

Covers:
  - SdmaTpComm async APIs (allgather, allreduce, reduce_scatter) with mocks
  - ColumnParallelLinear overlap path selection when tp_comm_overlap=True
  - RowParallelLinear overlap path selection when tp_comm_overlap=True
  - ColumnParallelLinear non-SDMA forward correctness (use_sdma=False)
  - RowParallelLinear non-SDMA forward correctness (use_sdma=False)
  - Mock all mori imports since this is unit testing
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from atomrl.lumen.modules.parallel_linear import (
    LumenColumnParallelLinear,
    LumenRowParallelLinear,
)


def _make_config(
    sequence_parallel=False,
    lumen_tp_comm_overlap=False,
):
    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=1,
        lumen_tp_comm_overlap=lumen_tp_comm_overlap,
    )


# ===================================================================
# TestSdmaTpCommAsync — async APIs with mocks
# ===================================================================


@mock.patch("atomrl.lumen.modules.sdma_comm.SdmaTpContext")
class TestSdmaTpCommAsync:
    """Test SdmaTpComm async APIs using mocked mori."""

    def test_allgather_dim0_async_wait_returns_shape(self, mock_ctx):
        mock_ctx.get.return_value = SimpleNamespace(my_pe=0, npes=2)
        with mock.patch("atomrl.lumen.modules.sdma_comm._TpSdmaAllgather") as mock_ag:
            mock_ag_instance = mock.MagicMock()
            mock_ag.return_value = mock_ag_instance
            mock_ag_instance.start_async.return_value = True
            mock_ag_instance.wait_async.return_value = torch.randn(2, 16 * 64)

            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            SdmaTpComm.reset()
            comm = SdmaTpComm(mock.MagicMock())
            comm._ag = mock_ag_instance

            t = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
            comm.allgather_dim0_async(t)
            out = comm.wait_allgather_dim0()
            assert out.shape[0] == 32  # 2 * 16

    def test_allreduce_sum_async_wait_inplace(self, mock_ctx):
        mock_ctx.get.return_value = SimpleNamespace(my_pe=0, npes=2)
        with mock.patch("atomrl.lumen.modules.sdma_comm._TpSdmaAllreduce") as mock_ar:
            mock_ar_instance = mock.MagicMock()
            mock_ar.return_value = mock_ar_instance
            mock_ar_instance.start_async_inplace.return_value = True

            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            SdmaTpComm.reset()
            comm = SdmaTpComm(mock.MagicMock())
            comm._ar_handles = {torch.float32: mock_ar_instance}

            t = torch.randn(64, dtype=torch.float32, device="cuda")
            comm.allreduce_sum_async(t)
            comm.wait_allreduce_sum()
            mock_ar_instance.wait_async.assert_called_once()

    def test_allreduce_sum_async_raises_when_start_fails(self, mock_ctx):
        mock_ctx.get.return_value = SimpleNamespace(my_pe=0, npes=2)
        with mock.patch("atomrl.lumen.modules.sdma_comm._TpSdmaAllreduce") as mock_ar:
            mock_ar_instance = mock.MagicMock()
            mock_ar.return_value = mock_ar_instance
            mock_ar_instance.start_async_inplace.return_value = False

            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            SdmaTpComm.reset()
            comm = SdmaTpComm(mock.MagicMock())
            comm._ar_handles = {torch.float32: mock_ar_instance}

            t = torch.randn(64, dtype=torch.float32, device="cuda")
            with pytest.raises(RuntimeError, match="async start failed"):
                comm.allreduce_sum_async(t)
            assert not hasattr(comm, "_ar_async_dtype")

    def test_wait_allreduce_sum_clears_dtype_on_wait_failure(self, mock_ctx):
        mock_ctx.get.return_value = SimpleNamespace(my_pe=0, npes=2)
        with mock.patch("atomrl.lumen.modules.sdma_comm._TpSdmaAllreduce") as mock_ar:
            mock_ar_instance = mock.MagicMock()
            mock_ar.return_value = mock_ar_instance
            mock_ar_instance.wait_async.side_effect = RuntimeError("boom")

            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            SdmaTpComm.reset()
            comm = SdmaTpComm(mock.MagicMock())
            comm._ar_handles = {torch.float32: mock_ar_instance}
            comm._ar_async_dtype = torch.float32

            with pytest.raises(RuntimeError, match="boom"):
                comm.wait_allreduce_sum()
            assert not hasattr(comm, "_ar_async_dtype")

    def test_tp_sdma_allreduce_wait_async_raises_on_negative_status(self, mock_ctx):
        from atomrl.lumen.modules.sdma_comm import _TpSdmaAllreduce

        ar = _TpSdmaAllreduce(SimpleNamespace(my_pe=0, npes=2), dtype=torch.bfloat16)
        ar._handle = mock.MagicMock()
        ar._handle.wait_async.return_value = -1.0
        ar._async_output_buf = torch.zeros(4, dtype=torch.bfloat16)
        ar._async_tensor = torch.zeros(4, dtype=torch.bfloat16)
        ar._async_flat = torch.zeros(4, dtype=torch.bfloat16)
        ar._async_orig_shape = torch.Size([4])
        ar._async_stream = None

        with pytest.raises(RuntimeError, match="wait_async failed"):
            ar.wait_async()
        assert not hasattr(ar, "_async_tensor")
        assert not hasattr(ar, "_async_flat")
        assert not hasattr(ar, "_async_orig_shape")

    def test_tp_sdma_allreduce_wait_async_clears_state_when_handle_raises(
        self, mock_ctx
    ):
        from atomrl.lumen.modules.sdma_comm import _TpSdmaAllreduce

        ar = _TpSdmaAllreduce(SimpleNamespace(my_pe=0, npes=2), dtype=torch.bfloat16)
        ar._handle = mock.MagicMock()
        ar._handle.wait_async.side_effect = RuntimeError("native boom")
        ar._async_output_buf = torch.zeros(4, dtype=torch.bfloat16)
        ar._async_tensor = torch.zeros(4, dtype=torch.bfloat16)
        ar._async_flat = torch.zeros(4, dtype=torch.bfloat16)
        ar._async_orig_shape = torch.Size([4])
        ar._async_stream = None

        with pytest.raises(RuntimeError, match="native boom"):
            ar.wait_async()
        assert not hasattr(ar, "_async_tensor")
        assert not hasattr(ar, "_async_flat")
        assert not hasattr(ar, "_async_orig_shape")

    def test_reduce_scatter_dim0_async_wait_returns_chunk(self, mock_ctx):
        mock_ctx.get.return_value = SimpleNamespace(my_pe=0, npes=2)
        with mock.patch("atomrl.lumen.modules.sdma_comm._TpSdmaAllreduce"):
            mock_ar_instance = mock.MagicMock()
            mock_ar_instance._handle = mock.MagicMock()
            mock_ar_instance._handle.start_async.return_value = True
            mock_ar_instance._handle.wait_async = mock.MagicMock()
            mock_ar_instance._wire_dtype = torch.float32
            mock_ar_instance._needs_cast = False

            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            SdmaTpComm.reset()
            comm = SdmaTpComm(mock.MagicMock())
            comm._ar_handles = {torch.float32: mock_ar_instance}

            t = torch.randn(32, device="cuda", dtype=torch.float32)
            comm.reduce_scatter_dim0_async(t)
            out = comm.wait_reduce_scatter_dim0()
            assert out.shape[0] == 16


# ===================================================================
# TestColumnParallelOverlap — overlap path selection
# ===================================================================


@mock.patch(
    "atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=mock.MagicMock()
)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=2)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._use_sdma_from_args", return_value=True
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._tp_comm_overlap_from_args",
    return_value=False,
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
class TestColumnParallelOverlap:
    def test_overlap_path_selected_when_tp_comm_overlap_and_seq_parallel(self, *_):
        config = _make_config(sequence_parallel=True, lumen_tp_comm_overlap=True)
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        m.tp_comm_overlap = True
        m.sequence_parallel = True
        m.explicit_expert_comm = False
        m.use_sdma = True
        m.tp_size = 2
        m.gather_output = False

        with mock.patch.object(
            m,
            "_forward_sdma_overlap_column",
            return_value=(torch.randn(4, 8, 64, device="cuda"), None),
        ) as mock_overlap:
            x = torch.randn(4, 4, 64, device="cuda", dtype=torch.bfloat16)
            m.forward(x)
            mock_overlap.assert_called_once()

    def test_non_overlap_path_when_tp_comm_overlap_false(self, *_):
        config = _make_config(sequence_parallel=True, lumen_tp_comm_overlap=False)
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        m.tp_comm_overlap = False
        m.sequence_parallel = True
        m.use_sdma = True
        m.tp_size = 2

        with mock.patch.object(m, "_forward_sdma_pre_gemm") as mock_pre:
            mock_pre.return_value = torch.randn(
                4, 64, device="cuda", dtype=torch.bfloat16
            )
            x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
            m.forward(x)
            mock_pre.assert_called_once()


# ===================================================================
# TestRowParallelOverlap — overlap path selection
# ===================================================================


@mock.patch(
    "atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=mock.MagicMock()
)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=2)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._use_sdma_from_args", return_value=True
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._tp_comm_overlap_from_args", return_value=True
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
class TestRowParallelOverlap:
    def test_overlap_path_selected_when_tp_comm_overlap_and_sdma(self, *_):
        config = _make_config(lumen_tp_comm_overlap=True)
        m = LumenRowParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        m.tp_comm_overlap = True
        m.use_sdma = True
        m.tp_size = 2
        m.sequence_parallel = False

        with mock.patch.object(
            m,
            "_forward_sdma_overlap_row",
            return_value=torch.randn(4, 128, device="cuda"),
        ) as mock_overlap:
            x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
            m.forward(x)
            mock_overlap.assert_called_once()

    def test_non_overlap_path_when_tp_comm_overlap_false(self, *_):
        config = _make_config(lumen_tp_comm_overlap=False)
        with mock.patch(
            "atomrl.lumen.modules.parallel_linear._tp_comm_overlap_from_args",
            return_value=False,
        ):
            m = LumenRowParallelLinear(
                64,
                128,
                config=config,
                init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            )
        m.tp_comm_overlap = False
        m.use_sdma = True
        m.tp_size = 2

        with mock.patch.object(m, "_forward_sdma_post_gemm") as mock_post:
            mock_post.return_value = torch.randn(4, 128, device="cuda")
            x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
            m.forward(x)
            mock_post.assert_called_once()


# ===================================================================
# TestColumnParallelNonSdma — non-SDMA forward correctness
# ===================================================================


@mock.patch(
    "atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=mock.MagicMock()
)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=2)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._use_sdma_from_args", return_value=False
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._tp_comm_overlap_from_args",
    return_value=False,
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
class TestColumnParallelNonSdma:
    """Full forward correctness when use_sdma=False (non-SDMA comm paths)."""

    def _make_column(self, sequence_parallel=False, gather_output=False):
        config = _make_config(
            sequence_parallel=sequence_parallel, lumen_tp_comm_overlap=False
        )
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        )
        torch.nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            with torch.no_grad():
                m.bias.zero_()
        m.use_sdma = False
        m.tp_comm_overlap = False
        m.tp_size = 2
        m.sequence_parallel = sequence_parallel
        m.gather_output = gather_output
        m.explicit_expert_comm = False
        return m

    def test_forward_seq_parallel_no_gather(
        self,
        mock_rs,
        mock_reduce,
        mock_gather_tp,
        mock_gather_sp,
        mock_copy,
        *_,
    ):
        m = self._make_column(sequence_parallel=True, gather_output=False)
        x = torch.randn(4, 4, 64, device="cuda", dtype=torch.bfloat16)
        out, _bias = m.forward(x)

        mock_gather_sp.assert_called_once()
        assert out.shape == (4, 4, 64)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_no_seq_parallel_gather_output(
        self,
        mock_rs,
        mock_reduce,
        mock_gather_tp,
        mock_gather_sp,
        mock_copy,
        *_,
    ):
        m = self._make_column(sequence_parallel=False, gather_output=True)
        x = torch.randn(4, 4, 64, device="cuda", dtype=torch.bfloat16)
        out, _bias = m.forward(x)

        mock_gather_tp.assert_called_once()
        mock_gather_sp.assert_not_called()
        assert out.shape == (4, 4, 64)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_no_seq_parallel_no_gather(
        self,
        mock_rs,
        mock_reduce,
        mock_gather_tp,
        mock_gather_sp,
        mock_copy,
        *_,
    ):
        m = self._make_column(sequence_parallel=False, gather_output=False)
        x = torch.randn(4, 4, 64, device="cuda", dtype=torch.bfloat16)
        out, _bias = m.forward(x)

        mock_gather_sp.assert_not_called()
        mock_gather_tp.assert_not_called()
        assert out.shape == (4, 4, 64)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ===================================================================
# TestRowParallelNonSdma — non-SDMA forward correctness
# ===================================================================


@mock.patch(
    "atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=mock.MagicMock()
)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=2)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._use_sdma_from_args", return_value=False
)
@mock.patch(
    "atomrl.lumen.modules.parallel_linear._tp_comm_overlap_from_args",
    return_value=False,
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
class TestRowParallelNonSdma:
    """Full forward correctness when use_sdma=False (non-SDMA comm paths)."""

    def _make_row(self, sequence_parallel=False):
        config = _make_config(
            sequence_parallel=sequence_parallel, lumen_tp_comm_overlap=False
        )
        m = LumenRowParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        )
        torch.nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            with torch.no_grad():
                m.bias.zero_()
        m.use_sdma = False
        m.tp_comm_overlap = False
        m.tp_size = 2
        m.sequence_parallel = sequence_parallel
        m.explicit_expert_comm = False
        return m

    def test_forward_seq_parallel(
        self,
        mock_rs,
        mock_reduce,
        mock_gather_tp,
        mock_gather_sp,
        mock_copy,
        *_,
    ):
        m = self._make_row(sequence_parallel=True)
        x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
        out, _bias = m.forward(x)

        mock_rs.assert_called_once()
        assert out.shape == (4, 128)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_no_seq_parallel(
        self,
        mock_rs,
        mock_reduce,
        mock_gather_tp,
        mock_gather_sp,
        mock_copy,
        *_,
    ):
        m = self._make_row(sequence_parallel=False)
        x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
        out, _bias = m.forward(x)

        mock_reduce.assert_called_once()
        assert out.shape == (4, 128)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
