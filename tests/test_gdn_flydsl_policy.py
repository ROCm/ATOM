# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN backend policy, without AITER or Triton.

`select_policy` reads only the backend settings, the device arch and whether
HIP is present. `atom.model_ops.fla_ops` imports Triton kernels at
package import, so this file loads `gdn_flydsl.py` on its own with Triton and
its two sibling kernel modules stubbed, and the arch test stubs AITER's
`get_gfx_runtime`. That keeps the policy matrix in the non-GPU unit tests,
which run without Triton and AITER.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "atom/model_ops/fla_ops/gdn_flydsl.py"
PACKAGE = "_gdn_flydsl_policy_probe"


@pytest.fixture
def fly(monkeypatch):
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    triton.cdiv = lambda a, b: -(-a // b)
    language = types.ModuleType("triton.language")
    language.constexpr = object
    triton.language = language
    package = types.ModuleType(PACKAGE)
    package.__path__ = []
    chunk_o = types.ModuleType(f"{PACKAGE}.chunk_o")
    chunk_o.chunk_fwd_o = None
    l2norm = types.ModuleType(f"{PACKAGE}.l2norm")
    l2norm.l2norm_fwd = None
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.gdn_flydsl", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as m:
        for name, stub in {
            "triton": triton,
            "triton.language": language,
            PACKAGE: package,
            chunk_o.__name__: chunk_o,
            l2norm.__name__: l2norm,
            module.__name__: module,
        }.items():
            m.setitem(sys.modules, name, stub)
        spec.loader.exec_module(module)
    return module


def _state():
    return torch.empty(1, 24, 128, 128, dtype=torch.bfloat16)


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("no_hip", [False, True])
@pytest.mark.parametrize("prefill_backend", ["auto", "flydsl", "triton"])
@pytest.mark.parametrize("replayssm", [False, True])
@pytest.mark.parametrize("gfx942", [False, True])
def test_policy_keeps_prefill_and_drops_decode_under_replayssm(
    fly, monkeypatch, allowed, no_hip, prefill_backend, replayssm, gfx942
):
    backends = {"prefill": prefill_backend, "decode": "auto"}
    monkeypatch.setattr(fly, "backend", backends.__getitem__)
    monkeypatch.setattr(fly, "ops", lambda: None if no_hip else object())
    monkeypatch.setattr(fly, "_validated_on_gfx942", lambda: gfx942)
    policy = fly.select_policy(
        allowed=allowed,
        replayssm=replayssm,
        lossy_decode=False,
        state=_state(),
        activation_dtype=torch.bfloat16,
    )
    available = allowed and not no_hip
    assert policy.prefill == (
        available and prefill_backend != "triton" and (gfx942 or not replayssm)
    )
    assert policy.decode == (available and gfx942 and not replayssm)


@pytest.mark.parametrize("prefill_backend", ["auto", "flydsl"])
def test_replayssm_prefill_needs_gfx942_even_when_forced(
    fly, monkeypatch, prefill_backend
):
    monkeypatch.setattr(fly, "backend", lambda stage: prefill_backend)
    monkeypatch.setattr(fly, "ops", lambda: object())
    monkeypatch.setattr(fly, "_validated_on_gfx942", lambda: False)
    kwargs = {
        "allowed": True,
        "lossy_decode": False,
        "state": _state(),
        "activation_dtype": torch.bfloat16,
    }
    assert fly.select_policy(replayssm=True, **kwargs) == fly.GDNFlyDSLPolicy()
    # Without the ring, the arch gate for prefill does not apply.
    assert fly.select_policy(replayssm=False, **kwargs).prefill


@pytest.mark.parametrize("prefill_backend", ["auto", "flydsl"])
@pytest.mark.parametrize("replayssm", [False, True])
def test_policy_is_empty_without_hip_before_probing_the_arch(
    fly, monkeypatch, prefill_backend, replayssm
):
    def no_arch():
        raise RuntimeError("get_gfx_runtime needs a ROCm device")

    chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx_runtime = no_arch
    monkeypatch.setitem(sys.modules, chip_info.__name__, chip_info)
    monkeypatch.setattr(fly, "backend", lambda stage: prefill_backend)
    monkeypatch.setattr(fly.torch.version, "hip", None)
    policy = fly.select_policy(
        allowed=True,
        replayssm=replayssm,
        lossy_decode=False,
        state=_state(),
        activation_dtype=torch.bfloat16,
    )
    assert policy == fly.GDNFlyDSLPolicy()


AITER_OPS = {
    "aiter.ops.flydsl.linear_attention_kernels": ["flydsl_gdr_decode"],
    "aiter.ops.flydsl.linear_attention_prefill_kernels": [
        "chunk_gated_delta_rule_fwd_h_flydsl_opt",
        "gdn_prepare_flydsl_supported",
        "gdn_prepare_fwd_flydsl",
    ],
    "aiter.ops.prefill_batch_metadata": ["build_gated_delta_rule_prefill_metadata"],
}


@pytest.mark.parametrize(
    "missing", [*AITER_OPS, *(s for symbols in AITER_OPS.values() for s in symbols)]
)
def test_ops_raises_on_hip_when_an_aiter_symbol_is_missing(fly, monkeypatch, missing):
    monkeypatch.setattr(fly.torch.version, "hip", "6.4")
    for name, symbols in AITER_OPS.items():
        stub = None
        if name != missing:
            stub = types.ModuleType(name)
            for symbol in symbols:
                if symbol != missing:
                    setattr(stub, symbol, object())
        monkeypatch.setitem(sys.modules, name, stub)
    with pytest.raises(ImportError):
        fly.ops()


@pytest.mark.parametrize("stage", ["prefill", "decode"])
@pytest.mark.parametrize("replayssm", [False, True])
@pytest.mark.parametrize("hip", [True, False])
@pytest.mark.parametrize("lossy_decode", [False, True])
def test_invalid_backend_raises_whatever_the_gates(
    fly, monkeypatch, stage, replayssm, hip, lossy_decode
):
    monkeypatch.setenv(f"ATOM_GDN_{stage.upper()}_BACKEND", "bogus")
    if hip:
        monkeypatch.setattr(fly, "ops", lambda: object())
    else:
        monkeypatch.setattr(fly.torch.version, "hip", None)
    monkeypatch.setattr(fly, "_validated_on_gfx942", lambda: True)
    with pytest.raises(ValueError, match=f"Invalid ATOM GDN {stage} backend"):
        fly.select_policy(
            allowed=True,
            replayssm=replayssm,
            lossy_decode=lossy_decode,
            state=_state(),
            activation_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize(
    ("arch", "validated"), [("gfx942", True), ("gfx950", False), ("gfx1250", False)]
)
def test_flydsl_is_validated_on_gfx942_only(fly, monkeypatch, arch, validated):
    chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx_runtime = lambda: arch
    monkeypatch.setitem(sys.modules, chip_info.__name__, chip_info)
    assert fly._validated_on_gfx942() is validated
