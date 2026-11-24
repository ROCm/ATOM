# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
Tests for deferred weight gradient computation.

Covers:
  - _DeferredWgrad unit tests (defer, execute, closure-based accumulation)
  - Autograd integration: QuantizedLinearFunction with delay_wgrad=True
  - Integration with LumenRowParallelLinear / LumenColumnParallelLinear
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

# =========================================================================
# Unit tests for _DeferredWgrad
# =========================================================================


class TestDeferredWgrad:

    def test_defer_and_execute(self):
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.grad = torch.zeros(4, 4)
        grad_val = torch.ones(4, 4)

        def _fn():
            weight.grad.add_(grad_val)

        dw.defer(_fn)
        assert dw.has_pending

        dw.execute()
        assert not dw.has_pending
        torch.testing.assert_close(weight.grad, grad_val)

    def test_defer_into_main_grad(self):
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.main_grad = torch.zeros(4, 4)
        grad_val = torch.ones(4, 4) * 2

        def _fn():
            weight.main_grad.add_(grad_val)

        dw.defer(_fn)
        dw.execute()
        torch.testing.assert_close(weight.main_grad, grad_val)

    def test_no_pending_execute_is_noop(self):
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        dw.execute()  # should not raise
        assert not dw.has_pending

    def test_accumulates_on_existing_grad(self):
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.grad = torch.ones(4, 4)

        def _fn():
            weight.grad.add_(torch.ones(4, 4) * 3)

        dw.defer(_fn)
        dw.execute()
        torch.testing.assert_close(weight.grad, torch.ones(4, 4) * 4)

    def test_double_defer_queues_both(self):
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.grad = torch.zeros(4, 4)

        dw.defer(lambda: weight.grad.add_(torch.ones(4, 4)))
        dw.defer(lambda: weight.grad.add_(torch.ones(4, 4) * 5))
        dw.execute()
        # Both closures run: 0 + 1 + 5 = 6
        torch.testing.assert_close(weight.grad, torch.ones(4, 4) * 6)


# =========================================================================
# Autograd integration: QuantizedLinearFunction with delay_wgrad
# =========================================================================


class TestQuantizedLinearDeferredWgrad:
    """Verify that QuantizedLinearFunction correctly defers wgrad."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.in_features = 32
        self.out_features = 16
        self.batch = 4

    def _run_deferred_vs_eager(self, scaling_type="none"):
        """Compare deferred wgrad against eager wgrad for numerical correctness."""
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad
        from atomrl.lumen.ops.quantize.linear import QuantizedLinearFunction

        torch.manual_seed(42)
        x = torch.randn(self.batch, self.in_features, requires_grad=True)
        w_eager = torch.nn.Parameter(torch.randn(self.out_features, self.in_features))
        w_defer = torch.nn.Parameter(w_eager.data.clone())

        dwg = _DeferredWgrad()

        # --- eager path ---
        y_eager = QuantizedLinearFunction.apply(
            x,
            w_eager,
            None,
            None,
            scaling_type,
            torch.float8_e4m3fn,
            128,
            "weight",
            True,
            True,
            False,  # gradient_accumulation_fusion
            False,  # delay_wgrad
            None,  # deferred_wgrad
            False,  # fp8_activation_store
        )
        loss_eager = y_eager.sum()
        loss_eager.backward()

        # --- deferred path ---
        x2 = x.detach().clone().requires_grad_(True)
        y_defer = QuantizedLinearFunction.apply(
            x2,
            w_defer,
            None,
            None,
            scaling_type,
            torch.float8_e4m3fn,
            128,
            "weight",
            True,
            True,
            False,  # gradient_accumulation_fusion
            True,  # delay_wgrad
            dwg,  # deferred_wgrad
            False,  # fp8_activation_store
        )
        loss_defer = y_defer.sum()
        loss_defer.backward()

        # dgrad should be computed immediately
        assert x2.grad is not None
        torch.testing.assert_close(x2.grad, x.grad, atol=1e-5, rtol=1e-3)

        # wgrad should NOT be computed yet
        assert w_defer.grad is None
        assert dwg.has_pending

        # execute deferred wgrad
        dwg.execute()
        assert not dwg.has_pending
        assert w_defer.grad is not None
        torch.testing.assert_close(w_defer.grad, w_eager.grad, atol=1e-5, rtol=1e-3)

    def test_bf16_deferred_wgrad(self):
        self._run_deferred_vs_eager(scaling_type="none")

    def test_deferred_wgrad_with_gradient_accumulation_fusion(self):
        """When gradient_accumulation_fusion=True, wgrad accumulates into main_grad."""
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad
        from atomrl.lumen.ops.quantize.linear import QuantizedLinearFunction

        torch.manual_seed(42)
        x = torch.randn(self.batch, self.in_features, requires_grad=True)
        w = torch.nn.Parameter(torch.randn(self.out_features, self.in_features))
        w.main_grad = torch.zeros(self.out_features, self.in_features)

        dwg = _DeferredWgrad()

        y = QuantizedLinearFunction.apply(
            x,
            w,
            None,
            None,
            "none",
            torch.float8_e4m3fn,
            128,
            "weight",
            True,
            True,
            True,  # gradient_accumulation_fusion
            True,  # delay_wgrad
            dwg,
            False,  # fp8_activation_store
        )
        y.sum().backward()

        assert w.grad is None
        assert dwg.has_pending
        assert torch.all(w.main_grad == 0)

        dwg.execute()
        assert not dwg.has_pending
        assert w.main_grad.abs().sum() > 0

    def test_deferred_wgrad_creates_grad_when_none(self):
        """When weight.grad is None and GAF is off, deferred execution creates it."""
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad
        from atomrl.lumen.ops.quantize.linear import QuantizedLinearFunction

        torch.manual_seed(42)
        x = torch.randn(self.batch, self.in_features, requires_grad=True)
        w = torch.nn.Parameter(torch.randn(self.out_features, self.in_features))
        assert w.grad is None

        dwg = _DeferredWgrad()

        y = QuantizedLinearFunction.apply(
            x,
            w,
            None,
            None,
            "none",
            torch.float8_e4m3fn,
            128,
            "weight",
            True,
            True,
            False,  # gradient_accumulation_fusion
            True,  # delay_wgrad
            dwg,
            False,  # fp8_activation_store
        )
        y.sum().backward()

        assert w.grad is None
        dwg.execute()
        assert w.grad is not None
        assert w.grad.abs().sum() > 0


# =========================================================================
# Integration with parallel linear modules
# =========================================================================


def _make_config(sequence_parallel=False):
    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        lumen_tp_comm_overlap=False,
    )


_PARALLEL_LINEAR_PATCHES = [
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
    mock.patch(
        "atomrl.lumen.modules.parallel_linear.copy_to_tensor_model_parallel_region",
        side_effect=lambda x, **kw: x,
    ),
    mock.patch(
        "atomrl.lumen.modules.parallel_linear.gather_from_sequence_parallel_region",
        side_effect=lambda x, **kw: x,
    ),
    mock.patch(
        "atomrl.lumen.modules.parallel_linear.gather_from_tensor_model_parallel_region",
        side_effect=lambda x, **kw: x,
    ),
    mock.patch(
        "atomrl.lumen.modules.parallel_linear.reduce_from_tensor_model_parallel_region",
        side_effect=lambda x, **kw: x,
    ),
    mock.patch(
        "atomrl.lumen.modules.parallel_linear.reduce_scatter_to_sequence_parallel_region",
        side_effect=lambda x, **kw: x,
    ),
]


def _apply_patches(func):
    """Apply all parallel linear patches to a function."""
    for p in reversed(_PARALLEL_LINEAR_PATCHES):
        func = p(func)
    return func


class TestDeferredWgradModuleIntegration:
    """Test that parallel linear modules expose the execute_deferred_wgrad API."""

    @_apply_patches
    def test_row_parallel_has_deferred_wgrad(self, *_):
        from atomrl.lumen.modules.parallel_linear import LumenRowParallelLinear

        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            64,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            input_is_parallel=True,
        )
        assert hasattr(m, "_deferred_wgrad")
        assert hasattr(m, "execute_deferred_wgrad")
        assert hasattr(m, "backward_dw")
        assert callable(m.execute_deferred_wgrad)
        assert callable(m.backward_dw)
        assert not m._deferred_wgrad.has_pending

    @_apply_patches
    def test_column_parallel_has_deferred_wgrad(self, *_):
        from atomrl.lumen.modules.parallel_linear import LumenColumnParallelLinear

        config = _make_config()
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert hasattr(m, "_deferred_wgrad")
        assert hasattr(m, "execute_deferred_wgrad")
        assert hasattr(m, "backward_dw")

    @_apply_patches
    def test_execute_deferred_wgrad_noop_when_no_pending(self, *_):
        from atomrl.lumen.modules.parallel_linear import LumenRowParallelLinear

        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            64,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            input_is_parallel=True,
        )
        m.execute_deferred_wgrad()  # should not raise
        m.backward_dw()  # should not raise

    @_apply_patches
    def test_manual_defer_and_execute_via_module(self, *_):
        """Manually defer a wgrad and execute it via the module API."""
        from atomrl.lumen.modules.parallel_linear import LumenRowParallelLinear

        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            64,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            input_is_parallel=True,
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        m.weight.grad = torch.zeros_like(m.weight)
        grad_val = torch.ones_like(m.weight)

        m._deferred_wgrad.defer(lambda: m.weight.grad.add_(grad_val))
        assert m._deferred_wgrad.has_pending

        m.execute_deferred_wgrad()
        assert not m._deferred_wgrad.has_pending
        torch.testing.assert_close(m.weight.grad, grad_val)

    @_apply_patches
    def test_backward_dw_alias(self, *_):
        """backward_dw() should behave identically to execute_deferred_wgrad()."""
        from atomrl.lumen.modules.parallel_linear import LumenRowParallelLinear

        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            64,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            input_is_parallel=True,
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        m.weight.grad = torch.zeros_like(m.weight)
        grad_val = torch.ones_like(m.weight) * 7

        m._deferred_wgrad.defer(lambda: m.weight.grad.add_(grad_val))
        m.backward_dw()
        assert not m._deferred_wgrad.has_pending
        torch.testing.assert_close(m.weight.grad, grad_val)
