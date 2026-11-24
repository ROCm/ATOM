# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""The ReplaySSM cursor comes from the builder, once; each layer carries its records."""

import ast
import types
from pathlib import Path

import pytest
import torch

ATTENTIONS = Path(__file__).resolve().parents[1] / "atom/model_ops/attentions"


def lift(path, cls, name, **scope):
    """Method `name` of class `cls` in `path`, run on its own (CPU lane)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    scope = {"torch": torch, **scope}
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


def _module_constant(path, name):
    """A literal module-level constant, read from the source."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


class _Layer:
    base_linear_attention = True
    layer_num = 5


@pytest.mark.parametrize(
    "path, cls",
    [
        (ATTENTIONS / "gdn_attn.py", "GDNAttentionMetadataBuilder"),
        (ATTENTIONS / "kimi_mla_gdn_attn.py", "_KimiMLAGDNCommon"),
    ],
    ids=["gdn", "kimi_k3"],
)
@pytest.mark.parametrize("replayssm", [True, False])
def test_each_layer_binds_its_own_pool_rows(path, cls, replayssm):
    rows = _module_constant(ATTENTIONS / "gdn_attn.py", "LINEAR_STATE_ROWS")
    build = lift(path, cls, "build_kv_cache_tensor", LINEAR_STATE_ROWS=rows)
    runner = _runner()
    layer = _Layer()
    builder = types.SimpleNamespace(
        model_runner=runner, replayssm=replayssm, pool_rows={rows: {layer: 1}}
    )
    tensor = build(builder, layer)
    assert tensor.per_request_state
    # Row 1 of every pool, not row 0: the layer's own slice.
    assert tensor.k_cache.data_ptr() == runner.mamba_k_cache[1].data_ptr()
    assert tensor.v_cache.data_ptr() == runner.mamba_v_cache[1].data_ptr()
    for part in ("k", "u", "g"):
        records = getattr(tensor, f"replay_buf_{part}")
        if replayssm:
            pool = getattr(runner, f"replayssm_buf_{part}")
            assert records.data_ptr() == pool[1].data_ptr()
        else:
            assert records is None
