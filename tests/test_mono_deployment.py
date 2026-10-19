# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What a persistent mono kernel needs of the machine, refused by name."""

from types import SimpleNamespace

import pytest
import torch

from atom.mono.runtime.deployment import first_refusal, persistent_kernel_refusals


def _refusal(kv_transfer_config=None):
    atom_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    return first_refusal(persistent_kernel_refusals(atom_config, 256))


def test_a_gfx950_with_every_cta_resident_is_served(fake_gpu):
    fake_gpu()
    assert _refusal() is None
    assert _refusal(kv_transfer_config={}) is None


def test_no_device_is_not_a_refusal(monkeypatch):
    # a contract check builds compile-only and never binds a device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _refusal() is None


@pytest.mark.parametrize(
    ("device", "kv_transfer_config", "reason"),
    [
        ({"arch": "gfx942"}, None, "not gfx950"),
        ({"cus": 240}, None, "240 CUs, the kernels need 256"),
        ({}, {"kv_connector": "lmcache"}, "KV connector"),
    ],
)
def test_each_missing_precondition_is_named(
    fake_gpu, device, kv_transfer_config, reason
):
    fake_gpu(**device)
    assert _refusal(kv_transfer_config) == reason
