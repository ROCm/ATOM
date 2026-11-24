# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MXFP4 weight cache invalidation across optimizer shapes.

The cache holds each layer's quantized weight so that gradient-accumulation
micro-batches reuse it. Nothing else invalidates it, so a hook that fails to
fire leaves the run training against the step-0 weights without raising.
"""

import torch
from atomrl.lumen.quantize import register_mxfp4_weight_optimizer_hooks
from torch import nn


class _MegatronStyleOptimizer:
    """Stands in for Megatron's ChainedOptimizer / DistributedOptimizer.

    Those wrap the torch optimizers rather than subclassing them, so they
    expose ``step()`` but not ``register_step_post_hook``.
    """

    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1
        return "step-result"


def _model_with_cache():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    for layer in model:
        layer._mxfp4_w_cache = ((False, False), "fp4", "scale")
    return model


def _cached(model):
    return [hasattr(layer, "_mxfp4_w_cache") for layer in model]


class TestMXFP4WeightCacheHook:
    def test_weight_version_invalidates_cache_without_optimizer_hook(self, monkeypatch):
        """Generic nn.Linear training must not depend on Megatron's setup hook."""
        from atomrl.lumen.ops.quantize import linear as linear_mod
        from atomrl.lumen.ops.quantize import ops as ops_mod
        from atomrl.lumen.quantize import _mxfp4_cached_weight

        builds = 0

        def _quantize(*_args, **_kwargs):
            nonlocal builds
            builds += 1
            desc = type("_Desc", (), {})()
            desc.data = torch.full((32, 16), builds, dtype=torch.uint8)
            desc.scale = torch.ones((1, 1), dtype=torch.uint8)
            return desc

        monkeypatch.setattr(linear_mod, "quantize_input", _quantize)
        monkeypatch.setattr(
            ops_mod,
            "transpose_packed_fp4",
            lambda data, **_kwargs: data.t().contiguous(),
        )

        module = nn.Linear(32, 32, bias=False)
        first, _ = _mxfp4_cached_weight(
            module,
            module.weight,
            None,
            None,
            "mxfp4",
            None,
            32,
            gemm_rows=32,
        )
        reused, _ = _mxfp4_cached_weight(
            module,
            module.weight,
            None,
            None,
            "mxfp4",
            None,
            32,
            gemm_rows=32,
        )
        assert reused is first and builds == 1

        with torch.no_grad():
            module.weight.add_(1)

        rebuilt, _ = _mxfp4_cached_weight(
            module,
            module.weight,
            None,
            None,
            "mxfp4",
            None,
            32,
            gemm_rows=32,
        )
        assert rebuilt is not first and builds == 2

    def test_torch_optimizer_post_step_hook_clears_cache(self):
        model = _model_with_cache()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        register_mxfp4_weight_optimizer_hooks(model, optimizer)

        assert all(_cached(model))
        optimizer.step()
        assert not any(_cached(model))

    def test_optimizer_without_post_step_hook_clears_cache(self):
        model = _model_with_cache()
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(model, optimizer)

        optimizer.step()
        assert not any(_cached(model))
        assert optimizer.steps == 1, "wrapping must still run the original step()"

    def test_native_parameter_cache_is_cleared(self):
        """Native parallel linears keep the cache on weight, not the module."""
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        for layer in model:
            layer.weight._mxfp4_w_cache = ((False, False), "fp4", "scale")
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(model, optimizer)

        optimizer.step()

        assert not any(hasattr(layer.weight, "_mxfp4_w_cache") for layer in model)

    def test_grouped_expert_weights_are_cleared(self):
        """A grouped MoE layer holds its experts as weight0..weightN.

        The linear's forward takes the weight as an argument, so the cache
        lands on a Parameter the module does not expose as ``.weight``. Sweeping
        only that name left every expert quantized from the step-0 masters for
        the whole run while the dense layers updated.
        """
        experts = nn.Module()
        for i in range(3):
            experts.register_parameter(f"weight{i}", nn.Parameter(torch.zeros(8, 8)))
            getattr(experts, f"weight{i}")._mxfp4_w_cache = (
                (False, False),
                "fp4",
                "scale",
            )
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(experts, optimizer)

        optimizer.step()

        assert not any(
            hasattr(getattr(experts, f"weight{i}"), "_mxfp4_w_cache") for i in range(3)
        )

    def test_wrapped_step_returns_original_result(self):
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(_model_with_cache(), optimizer)

        assert optimizer.step() == "step-result"

    def test_model_chunk_list_is_walked(self):
        """Megatron hands out a list of chunks under virtual pipeline parallelism."""
        chunks = [_model_with_cache(), _model_with_cache()]
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(chunks, optimizer)

        optimizer.step()
        assert not any(flag for chunk in chunks for flag in _cached(chunk))

    def test_cache_is_recreated_and_cleared_each_step(self):
        model = _model_with_cache()
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(model, optimizer)

        for _ in range(3):
            for layer in model:
                layer._mxfp4_w_cache = ((False, False), "fp4", "scale")
            optimizer.step()
            assert not any(_cached(model))
