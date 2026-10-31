# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Switch wiring of the Qwen3-Next / Qwen3.5 decode fusions (no kernel runs).

  F1 ATOM_GDN_REPLAYSSM_FUSED_GATING       gating folded into ReplaySSM

Each is off by default and, when on, must only apply where the fused output
is exactly what the unfused path consumes. Numerics are in
tests/test_replayssm_fused_gating.py (GPU).
"""

import pytest
import torch

try:
    from atom.model_ops import attention_gdn
    from atom.model_ops.fla_ops import replayssm
except Exception as e:  # noqa: BLE001 -- aiter / triton need a ROCm build to import
    pytest.skip(f"needs an importable aiter: {e}", allow_module_level=True)

SWITCHES = ("ATOM_GDN_REPLAYSSM_FUSED_GATING",)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# defaults                                                                     #
# --------------------------------------------------------------------------- #


def test_all_switches_default_off():
    from atom.utils import envs

    assert envs.ATOM_GDN_REPLAYSSM_FUSED_GATING is False


def test_switches_are_documented():
    import pathlib

    doc = (
        pathlib.Path(__file__).parents[1] / "docs" / "environment_variables.md"
    ).read_text()
    assert [name for name in SWITCHES if f"**{name}**" not in doc] == []


# --------------------------------------------------------------------------- #
# F1                                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("prefills", "decodes", "has_spec", "expected"),
    [
        (0, 0, True, True),  # pure MTP verify (the graph-captured case)
        (0, 3, False, True),  # pure decode
        (0, 2, True, False),  # spec + plain decodes: g/beta are index_selected
        (1, 0, True, False),  # prefill in the batch
        (2, 0, False, False),  # pure prefill
        (0, 0, False, False),  # nothing to do
    ],
)
def test_f1_applies_only_to_uniform_replayssm_batches(
    monkeypatch, prefills, decodes, has_spec, expected
):
    monkeypatch.setenv("ATOM_GDN_REPLAYSSM_FUSED_GATING", "1")
    fn = attention_gdn.replayssm_gate_in_kernel
    assert fn(True, prefills, decodes, has_spec) is expected
    # never without ReplaySSM (FlyDSL / lossy / fused_recurrent keep theirs)
    assert fn(False, prefills, decodes, has_spec) is False


def test_f1_off_by_default():
    assert attention_gdn.replayssm_gate_in_kernel(True, 0, 0, True) is False


def _replay_args(T_tot=4, H=4, HV=16, K=128, V=128, cap=16):
    z = torch.zeros
    return {
        "q": z(1, T_tot, H, K, dtype=torch.bfloat16),
        "k": z(1, T_tot, H, K, dtype=torch.bfloat16),
        "v": z(1, T_tot, HV, V, dtype=torch.bfloat16),
        "ckpt": z(2, HV, K, V),
        "buf_k": z(2, HV, cap, K, dtype=torch.bfloat16),
        "buf_u": z(2, HV, cap, V, dtype=torch.bfloat16),
        "buf_g": z(2, HV, cap),
        "write_pos": z(2, dtype=torch.int32),
        "slot_idx": z(1, dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, T_tot], dtype=torch.int32),
        "max_query_len": T_tot,
    }


def _gate_args(T_tot=4, HV=16):
    row = torch.zeros(T_tot, 64 + 2 * HV, dtype=torch.bfloat16)
    return {
        "a": row[:, 64 + HV :],
        "b": row[:, 64 : 64 + HV],
        "A_log": torch.zeros(HV),
        "dt_bias": torch.zeros(HV, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"beta": torch.zeros(1, 4, 16)}, "both g and beta"),
        ({"is_kda": True}, "scalar GDN gate"),
        ({"a": None}, "needs a"),
        ({"dt_bias": None}, "needs dt_bias"),
    ],
)
def test_f1_entry_refuses_incomplete_gate_inputs(change, match):
    kw = dict(_replay_args(), g=None, beta=None, **_gate_args())
    kw.update(change)
    with pytest.raises(AssertionError, match=match):
        replayssm.replayssm_gated_delta_rule(**kw)


def test_f1_entry_takes_strided_gate_columns_without_copying(monkeypatch):
    """a/b are column slices of the in_proj output. `.contiguous()` on them
    would launch two copy kernels and cancel the saving, so they must reach
    the kernel as the same views."""
    seen = {}

    class _Kernel:
        def __getitem__(self, grid):
            def launch(**kw):
                seen.update(kw)

            return launch

    monkeypatch.setattr(replayssm, "_replayssm_fwd_kernel", _Kernel())
    monkeypatch.setattr(replayssm, "get_gfx", lambda: "gfx950")
    gate = _gate_args()
    replayssm.replayssm_gated_delta_rule(
        **_replay_args(), g=None, beta=None, route="serial", **gate
    )
    assert seen["GATE_IN_KERNEL"] is True
    assert seen["a"] is gate["a"] and seen["b"] is gate["b"]
    assert seen["stride_a_tok"] == gate["a"].stride(0) == 64 + 2 * 16
    assert seen["stride_b_tok"] == gate["b"].stride(0)
    assert seen["IS_BETA_HEADWISE"] is False


def test_f1_entry_unfused_call_is_unchanged(monkeypatch):
    seen = {}

    class _Kernel:
        def __getitem__(self, grid):
            return lambda **kw: seen.update(kw)

    monkeypatch.setattr(replayssm, "_replayssm_fwd_kernel", _Kernel())
    monkeypatch.setattr(replayssm, "get_gfx", lambda: "gfx950")
    g = torch.zeros(1, 4, 16)
    beta = torch.zeros(1, 4, 16, dtype=torch.bfloat16)
    replayssm.replayssm_gated_delta_rule(
        **_replay_args(), g=g, beta=beta, route="serial"
    )
    assert seen["GATE_IN_KERNEL"] is False
    assert seen["g"] is g and seen["beta"] is beta
    assert seen["stride_beta_tok"] == beta.stride(1)
