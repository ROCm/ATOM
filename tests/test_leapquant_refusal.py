# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_ENABLE_LEAPQUANT refuses, at startup, every setup it would get wrong.

Each of these would otherwise start and serve a wrong state rather than fail:
KDA's per-channel gates are not what the GDN kernels decay by, a verify
window is more than the one token the decode kernel takes, a prefix-cache
image copies the int8 residual without its scales, compensators and window
records, and the int8 state is a ReplaySSM checkpoint, so ReplaySSM cannot
be off.
"""

from types import SimpleNamespace

import pytest

# The module imports aiter at load, which a non-GPU runner cannot satisfy;
# `exc_type` because the module is found and its import is what fails.
GDNStateMixin = pytest.importorskip(
    "atom.model_ops.attentions.gdn_attn",
    reason="the GDN builder's module imports aiter at load",
    exc_type=ImportError,
).GDNStateMixin


def _builder(*, kda=False, num_spec=0, prefix_caching=False):
    runner = SimpleNamespace(
        config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_text"),
            enable_prefix_caching=prefix_caching,
        ),
        num_spec_tokens=num_spec,
    )
    return SimpleNamespace(model_runner=runner, _is_kda=lambda: kda)


@pytest.mark.parametrize(
    ("setup", "replayssm", "error", "match"),
    [
        ({"kda": True}, None, NotImplementedError, "KDA"),
        ({"num_spec": 1}, None, NotImplementedError, "speculative decoding"),
        ({"prefix_caching": True}, None, NotImplementedError, "prefix caching"),
        ({}, "0", ValueError, "ATOM_ENABLE_REPLAYSSM=0"),
    ],
)
def test_leapquant_refuses_at_startup(monkeypatch, setup, replayssm, error, match):
    monkeypatch.setenv("ATOM_ENABLE_LEAPQUANT", "1")
    if replayssm is None:
        monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM", raising=False)
    else:
        monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", replayssm)
    builder = _builder(**setup)
    with pytest.raises(error, match=match):
        GDNStateMixin._init_gdn_state(builder, builder.model_runner)
