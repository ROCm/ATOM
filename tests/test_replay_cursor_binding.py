# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""The ReplaySSM cursor comes from the builder, once; each layer carries its records."""

import ast
import types
from pathlib import Path

import pytest
import torch

ATTENTIONS = Path(__file__).resolve().parents[1] / "atom/model_ops/attentions"


def lift(path, cls, name):
    """Method `name` of class `cls` in `path`, run on its own (CPU lane)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    scope = {"torch": torch}
                    code = compile(ast.Module([item], []), str(path), "exec")
                    exec(code, scope)  # noqa: S102
                    return scope[name]
    raise AssertionError(f"{cls}.{name} not found in {path}")


def _runner(slots: int = 4):
    return types.SimpleNamespace(
        mamba_k_cache=torch.zeros((2, slots, 3)),
        mamba_v_cache=torch.zeros((2, slots, 2, 2)),
        replayssm_buf_k=torch.zeros((2, slots, 1, 4, 2)),
        replayssm_buf_u=torch.zeros((2, slots, 1, 4, 2)),
        replayssm_buf_g=torch.zeros((2, slots, 1, 4, 2)),
        replayssm_write_pos=torch.full((slots,), -1, dtype=torch.int32),
    )


@pytest.mark.parametrize("replayssm", [True, False])
def test_the_builder_hands_over_the_runners_cursor(replayssm):
    cursor = lift(ATTENTIONS / "gdn_attn.py", "GDNStateMixin", "replay_cursor")
    builder = types.SimpleNamespace(model_runner=_runner(), replayssm=replayssm)
    expected = builder.model_runner.replayssm_write_pos if replayssm else None
    assert cursor(builder) is expected


def test_a_backend_without_records_has_no_cursor():
    cursor = lift(
        ATTENTIONS / "backends.py", "AttentionMetadataBuilder", "replay_cursor"
    )
    assert cursor(types.SimpleNamespace()) is None


class _Layer:
    base_linear_attention = True
    layer_num = 5


@pytest.mark.parametrize("model", ["gdn", "kimi_k3"])
@pytest.mark.parametrize("replayssm", [True, False])
def test_each_layer_carries_its_records(model, replayssm):
    gdn_attn = pytest.importorskip("atom.model_ops.attentions.gdn_attn")
    if model == "gdn":
        build = gdn_attn.GDNAttentionMetadataBuilder.build_kv_cache_tensor
    else:
        kimi = pytest.importorskip("atom.model_ops.attentions.kimi_mla_gdn_attn")
        build = kimi._KimiMLAGDNCommon.build_kv_cache_tensor
    layer = _Layer()
    builder = types.SimpleNamespace(
        model_runner=_runner(),
        replayssm=replayssm,
        pool_rows={gdn_attn.LINEAR_STATE_ROWS: {layer: 1}},
    )
    tensor = build(builder, layer)
    assert tensor.per_request_state
    assert (tensor.replay_buf_k is not None) == replayssm
