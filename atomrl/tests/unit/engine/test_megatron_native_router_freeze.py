# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""megatron_cfg.freeze_moe_router: NeMo-RL keeps a GRPO MoE's router frozen."""

from __future__ import annotations

import pytest
import torch
from atomrl.engine.training import megatron_native_engine as eng
from atomrl.workers.actor_worker import AtomActorWorker


def _layer(with_router: bool) -> torch.nn.Module:
    mlp = torch.nn.Module()
    if with_router:
        mlp.router = torch.nn.Linear(4, 2, bias=False)
    mlp.experts = torch.nn.Linear(4, 4)
    layer = torch.nn.Module()
    layer.mlp = mlp
    layers = torch.nn.ModuleList([layer, layer])
    decoder = torch.nn.Module()
    decoder.layers = layers
    model = torch.nn.Module()
    model.decoder = decoder
    return model


def test_freezes_only_the_router() -> None:
    model = _layer(with_router=True)
    eng._freeze_moe_router(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        "decoder.layers.0.mlp.experts.weight",
        "decoder.layers.0.mlp.experts.bias",
    }
    assert not model.decoder.layers[0].mlp.router.weight.requires_grad


def test_refuses_a_model_without_a_router() -> None:
    with pytest.raises(ValueError, match="no MoE router"):
        eng._freeze_moe_router(_layer(with_router=False))


def test_actor_passes_the_flag_to_the_engine() -> None:
    """actor_worker copies megatron_cfg into engine_config key by key; a key it
    does not copy silently does nothing."""
    import inspect

    assert '"freeze_moe_router": meg_cfg.get("freeze_moe_router", False)' in (
        inspect.getsource(AtomActorWorker)
    )
