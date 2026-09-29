# SPDX-License-Identifier: MIT
"""Dispatch must distinguish one MTP request from several decode requests."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "dsv4_mono_policy",
    Path(__file__).parents[1] / "atom/model_ops/dsv4_monokernel.py",
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def context(**changes):
    values = {
        "is_prefill": False,
        "running_bs": 1,
        "running_tokens": 4,
        "ubatch_token_offset": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_decode_and_mtp3():
    for qlen in (1, 2, 3, 4):
        assert policy.shape_supported(context(running_tokens=qlen), qlen)


def test_multiple_requests_and_padded_graph_are_not_one_mtp_request():
    assert not policy.shape_supported(context(running_bs=4), 4)
    assert not policy.shape_supported(context(running_bs=2), 4)
    assert not policy.shape_supported(context(running_tokens=8), 4)
    assert not policy.shape_supported(context(running_tokens=8), 8)


def test_prefill_and_ubatch_are_rejected():
    assert not policy.shape_supported(context(is_prefill=True), 4)
    assert not policy.shape_supported(context(ubatch_token_offset=4), 4)
    assert not policy.shape_supported(None, 4)


def test_stable_compressor_requires_live_layer_bucket_and_real_metadata():
    op = SimpleNamespace(_closed=False)
    adapter = SimpleNamespace(ops={4: op})
    fwd = SimpleNamespace(
        context=context(is_dummy_run=False), attn_metadata=object(), ubatch_slices=None
    )
    assert policy.stable_compressor_projection_supported(adapter, fwd, 4)
    assert not policy.stable_compressor_projection_supported(None, fwd, 4)
    assert not policy.stable_compressor_projection_supported(adapter, fwd, 1)
    op._closed = True
    assert not policy.stable_compressor_projection_supported(adapter, fwd, 4)
    op._closed = False
    for attr, value in (("attn_metadata", None), ("ubatch_slices", [object()])):
        original = getattr(fwd, attr)
        setattr(fwd, attr, value)
        assert not policy.stable_compressor_projection_supported(adapter, fwd, 4)
        setattr(fwd, attr, original)
    for changes in (
        {"is_prefill": True},
        {"running_bs": 2},
        {"running_tokens": 3},
        {"is_dummy_run": True},
        {"ubatch_token_offset": 1},
    ):
        fwd.context = context(is_dummy_run=False)
        for attr, value in changes.items():
            setattr(fwd.context, attr, value)
        assert not policy.stable_compressor_projection_supported(adapter, fwd, 4)
