# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What ATOM_ENABLE_LEAPQUANT accepts and refuses at startup.

Refused, because each would otherwise start and serve a wrong state: KDA's
per-channel gates are not what the GDN kernels decay by, and the int8 state
is a ReplaySSM checkpoint, so ReplaySSM cannot be off. Speculative decoding
and prefix caching start, with the record buffer sized for the verify window
and generation keeping no checkpoints.
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
        forward_vars={},
    )
    return SimpleNamespace(
        model_runner=runner,
        _is_kda=lambda: kda,
        max_bs=4,
        device="cpu",
        _state_index_buffers=GDNStateMixin._state_index_buffers,
    )


def _leapquant_env(monkeypatch, replayssm=None, window=None):
    monkeypatch.setenv("ATOM_ENABLE_LEAPQUANT", "1")
    if replayssm is None:
        monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM", raising=False)
    else:
        monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", replayssm)
    if window is None:
        monkeypatch.delenv("ATOM_LEAPQUANT_WINDOW", raising=False)
    else:
        monkeypatch.setenv("ATOM_LEAPQUANT_WINDOW", str(window))


@pytest.mark.parametrize(
    ("setup", "replayssm", "error", "match"),
    [
        ({"kda": True}, None, NotImplementedError, "KDA"),
        ({}, "0", ValueError, "ATOM_ENABLE_REPLAYSSM=0"),
    ],
)
def test_leapquant_refuses_at_startup(monkeypatch, setup, replayssm, error, match):
    _leapquant_env(monkeypatch, replayssm)
    builder = _builder(**setup)
    with pytest.raises(error, match=match):
        GDNStateMixin._init_gdn_state(builder, builder.model_runner)


@pytest.mark.parametrize(
    ("num_spec", "prefix_caching", "window", "cache_len"),
    [
        # p + 1 records: a fold once the cursor reaches p.
        (0, False, 16, 17),
        (0, True, 16, 17),
        # A verify window of T takes p + 2T - 1, so h still folds at p.
        (1, True, 16, 19),
        (3, False, 16, 23),
        # Never below the two windows the flush invariant needs.
        (3, False, 0, 8),
    ],
)
def test_leapquant_sizes_the_record_buffer_for_the_verify_window(
    monkeypatch, num_spec, prefix_caching, window, cache_len
):
    _leapquant_env(monkeypatch, window=window)
    builder = _builder(num_spec=num_spec, prefix_caching=prefix_caching)
    GDNStateMixin._init_gdn_state(builder, builder.model_runner)
    assert builder.leapquant and builder.replayssm
    assert builder.replayssm_max_query_len == num_spec + 1
    assert builder.replayssm_cache_len == cache_len
