# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Partial checkpoint loads (`only_params`) for asymmetric rapidserve decode.

Under an asymmetric topology the decode process aliases most of its parameters
into the prefill process via CUDA IPC, but its TP=1 attention weights have no
counterpart among prefill's TP=N shards and must come from the checkpoint. Those
are loaded with `only_params` scoping the load to exactly the un-aliased names.

Two properties matter, and the second is the dangerous one:

1. Only the named parameters are written. Writing an aliased parameter would
   push checkpoint bytes through the alias into the OTHER process's memory.
2. Post-load processing is scoped the same way. `process_weights_after_loading`
   shuffles/quantizes IN PLACE, so running it on an aliased module would corrupt
   prefill's weights with no crash and no error — both processes would then
   produce garbage.

Plain CPU torch; `loading_core` imports no AITER.
"""

import pytest
import torch
from torch import nn

from atom.model_loader.loading_core import load_weights_into_model


class _HFConfig:
    """Minimal stand-in; loading_core reads only these."""

    num_hidden_layers = 1
    n_routed_experts = 0
    num_experts = 0


def _default_weight_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)


class _Tiny(nn.Module):
    """Two independent leaves, standing in for an aliased and a local module."""

    def __init__(self):
        super().__init__()
        self.aliased = nn.Linear(4, 4, bias=False)
        self.local = nn.Linear(4, 4, bias=False)


def _load(model, only_params):
    weights = {
        "aliased.weight": torch.full((4, 4), 7.0),
        "local.weight": torch.full((4, 4), 9.0),
    }
    return load_weights_into_model(
        model=model,
        model_name_or_path="<synthetic>",
        hf_config=_HFConfig(),
        only_params=only_params,
        default_weight_loader=_default_weight_loader,
        fuse_shared_expert=lambda *a, **k: False,
        is_rank0=lambda: True,
        weights_iterator=lambda *a, **k: iter(weights.items()),
    )


class TestOnlyParamsScopesTheLoad:
    def test_none_loads_everything(self):
        m = _Tiny()
        nn.init.zeros_(m.aliased.weight)
        nn.init.zeros_(m.local.weight)
        _load(m, only_params=None)
        assert m.aliased.weight[0, 0] == 7.0
        assert m.local.weight[0, 0] == 9.0

    def test_unnamed_param_is_left_untouched(self):
        """The aliased parameter must keep its pre-load value."""
        m = _Tiny()
        nn.init.constant_(m.aliased.weight, -1.0)
        nn.init.zeros_(m.local.weight)
        _load(m, only_params={"local.weight"})
        assert m.aliased.weight[0, 0] == -1.0, "alias was overwritten by the load"
        assert m.local.weight[0, 0] == 9.0

    def test_empty_set_loads_nothing(self):
        m = _Tiny()
        nn.init.constant_(m.aliased.weight, -1.0)
        nn.init.constant_(m.local.weight, -2.0)
        _load(m, only_params=set())
        assert m.aliased.weight[0, 0] == -1.0
        assert m.local.weight[0, 0] == -2.0


class TestPostProcessScoping:
    """`process_weights_after_loading` must run only on locally-loaded modules.

    Exercises the derivation in loader.load_model:
        post_process_modules = {n.rpartition(".")[0] for n in only_params}
    """

    @staticmethod
    def _modules_for(only_params):
        return {n.rpartition(".")[0] for n in only_params}

    def test_maps_param_names_to_owning_modules(self):
        got = self._modules_for(
            {
                "layers.0.attn.wq_b.weight",
                "layers.0.attn.wq_b.weight_scale",
                "layers.1.attn.wo_b.weight",
            }
        )
        assert got == {"layers.0.attn.wq_b", "layers.1.attn.wo_b"}

    def test_excludes_modules_with_no_loaded_param(self):
        """An aliased MoE module must not appear, or it gets re-shuffled."""
        got = self._modules_for({"layers.0.attn.wq_b.weight"})
        assert "layers.0.ffn.experts" not in got

    def test_top_level_param_maps_to_root_module(self):
        assert self._modules_for({"lm_head_bias"}) == {""}


class TestMaterializedParamKeepsStampedAttributes:
    """Meta -> real materialization must preserve the Parameter's attributes.

    Layers stamp bound methods onto the Parameter OBJECT at construction —
    `self.weight.weight_loader = self.weight_loader` (linear.py:501,
    embed_head.py:158) — and the dispatcher reaches for them by attribute
    (weight_dispatch.py:137). `p.data = <real tensor>` raises for a meta
    parameter, so a NEW Parameter is unavoidable; without copying `__dict__`
    across, the load dies with "'Parameter' object has no attribute
    'weight_loader'".
    """

    @staticmethod
    def _stamped_meta_param():
        p = nn.Parameter(torch.empty(4, 4, device="meta"), requires_grad=False)
        p.weight_loader = lambda *a, **k: None
        p.weight_loader_process = lambda *a, **k: None
        p.is_shuffled = True
        return p

    @staticmethod
    def _materialize(p):
        """Mirrors ModelRunner._load_unaliased_weights."""
        new = nn.Parameter(
            torch.empty(p.shape, dtype=p.dtype), requires_grad=p.requires_grad
        )
        new.__dict__.update(p.__dict__)
        return new

    def test_data_assignment_is_not_possible_from_meta(self):
        """Documents WHY a new object is needed rather than in-place mutation."""
        p = self._stamped_meta_param()
        with pytest.raises(RuntimeError, match="incompatible tensor type"):
            p.data = torch.empty(4, 4)

    def test_bare_replacement_loses_the_loader(self):
        p = self._stamped_meta_param()
        bare = nn.Parameter(torch.empty(p.shape), requires_grad=p.requires_grad)
        assert not hasattr(bare, "weight_loader")

    def test_carry_over_preserves_callables_and_markers(self):
        p = self._stamped_meta_param()
        new = self._materialize(p)
        assert callable(new.weight_loader)
        assert callable(new.weight_loader_process)
        assert new.is_shuffled is True

    def test_materialized_param_is_real_and_shaped(self):
        new = self._materialize(self._stamped_meta_param())
        assert not new.is_meta
        assert tuple(new.shape) == (4, 4)


class TestPostProcessSkipsMixedModules:
    """A module mixing loaded and aliased params must NOT be post-processed.

    The case that broke V4-Pro: prefill's post-load hook converts the MoE scales
    from Parameters to plain attributes (moe.py:1103-1108), so they arrive as
    `__attr__` and decode's meta Parameter slot survives. Decode then reloads
    `w13_weight_scale` locally, which put `...ffn.experts` into the
    post-processing set — and that module's `w13_weight` is a 90GB IPC ALIAS.
    Re-shuffling it allocates a second copy and writes through the alias into
    prefill's weights. Measured: load_cost 121GB against ~35GB of real tensors.

    Owning one loaded param is not sufficient; every param must be local.
    """

    @staticmethod
    def _modules_for(model, only_params):
        """Mirrors the selection in loader.load_model."""
        loaded = set(only_params)
        out = set()
        for module_name, module in model.named_modules():
            owned = [
                f"{module_name}.{n}" if module_name else n
                for n, _ in module.named_parameters(recurse=False)
            ]
            if owned and all(n in loaded for n in owned):
                out.add(module_name)
        return out

    @staticmethod
    def _moe_like():
        m = nn.Module()
        m.experts = nn.Module()
        m.experts.w13_weight = nn.Parameter(torch.zeros(4, 4))  # aliased
        m.experts.w13_weight_scale = nn.Parameter(torch.zeros(4))  # loaded
        m.attn = nn.Module()
        m.attn.weight = nn.Parameter(torch.zeros(4, 4))  # loaded
        return m

    def test_mixed_module_is_excluded(self):
        got = self._modules_for(
            self._moe_like(),
            {"experts.w13_weight_scale", "attn.weight"},
        )
        assert "experts" not in got, "aliased w13_weight would be re-shuffled"

    def test_fully_local_module_is_included(self):
        got = self._modules_for(
            self._moe_like(),
            {"experts.w13_weight_scale", "attn.weight"},
        )
        assert "attn" in got

    def test_all_params_loaded_includes_the_module(self):
        got = self._modules_for(
            self._moe_like(),
            {"experts.w13_weight", "experts.w13_weight_scale", "attn.weight"},
        )
        assert "experts" in got

    def test_paramless_container_modules_are_excluded(self):
        """Modules with no direct params must not vacuously qualify."""
        got = self._modules_for(self._moe_like(), {"attn.weight"})
        assert "" not in got  # the root owns no params of its own
