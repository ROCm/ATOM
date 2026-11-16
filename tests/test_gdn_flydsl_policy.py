# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN backend policy, without AITER or Triton.

`select_policy` reads only the backend settings, the device arch and whether
AITER's FlyDSL ops import. `atom.model_ops.fla_ops` imports Triton kernels at
package import, so this file loads `gdn_flydsl.py` on its own with Triton and
its two sibling kernel modules stubbed. That keeps the policy matrix in the
non-GPU unit tests, which run without Triton and AITER.
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


def _state(device="cpu"):
    return torch.empty(1, 24, 128, 128, dtype=torch.bfloat16, device=device)


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("missing_ops", [False, True])
@pytest.mark.parametrize("prefill_backend", ["auto", "flydsl", "triton"])
@pytest.mark.parametrize("replayssm", [False, True])
@pytest.mark.parametrize("gfx942", [False, True])
def test_policy_keeps_prefill_and_drops_decode_under_replayssm(
    fly, monkeypatch, allowed, missing_ops, prefill_backend, replayssm, gfx942
):
    backends = {"prefill": prefill_backend, "decode": "auto"}
    monkeypatch.setattr(fly, "backend", backends.__getitem__)
    monkeypatch.setattr(fly, "ops", lambda: None if missing_ops else object())
    # Decode would be eligible on this device; only the ring rules it out.
    monkeypatch.setattr(fly, "decode_device_supported", lambda device: True)
    monkeypatch.setattr(fly, "replayssm_prefill_device_supported", lambda d: gfx942)
    policy = fly.select_policy(
        allowed=allowed,
        replayssm=replayssm,
        lossy_decode=False,
        state=_state(),
        activation_dtype=torch.bfloat16,
    )
    available = allowed and not missing_ops
    assert policy.prefill == (
        available and prefill_backend != "triton" and (gfx942 or not replayssm)
    )
    assert policy.decode == (available and not replayssm)


@pytest.mark.parametrize("prefill_backend", ["auto", "flydsl"])
def test_replayssm_prefill_needs_gfx942_even_when_forced(
    fly, monkeypatch, prefill_backend
):
    monkeypatch.setattr(fly, "backend", lambda stage: prefill_backend)
    monkeypatch.setattr(fly, "ops", lambda: object())
    monkeypatch.setattr(fly, "decode_device_supported", lambda device: True)
    monkeypatch.setattr(fly, "replayssm_prefill_device_supported", lambda d: False)
    kwargs = {
        "allowed": True,
        "lossy_decode": False,
        "state": _state(),
        "activation_dtype": torch.bfloat16,
    }
    assert fly.select_policy(replayssm=True, **kwargs) == fly.GDNFlyDSLPolicy()
    # Without the ring, the arch gate for prefill does not apply.
    assert fly.select_policy(replayssm=False, **kwargs).prefill


@pytest.mark.parametrize(
    ("hip", "device", "arch", "supported"),
    [
        (None, "cuda:0", "gfx942:sramecc+:xnack-", False),
        ("6.4", "cpu", "gfx942:sramecc+:xnack-", False),
        ("6.4", "cuda:0", "gfx942:sramecc+:xnack-", True),
        ("6.4", "cuda:0", "gfx950:sramecc+:xnack-", False),
        ("6.4", "cuda:0", "gfx1250", False),
    ],
)
def test_replayssm_prefill_device_is_gfx942_only(
    fly, monkeypatch, hip, device, arch, supported
):
    monkeypatch.setattr(fly.torch.version, "hip", hip)
    monkeypatch.setattr(
        fly.torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(gcnArchName=arch),
    )
    device = torch.device(device)
    assert fly.replayssm_prefill_device_supported(device) is supported
