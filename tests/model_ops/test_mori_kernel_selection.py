# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which async protocol may be paired with which MoRI kernel.

aiter owns the (low_latency, internode) -> kernel mapping; what stays here is
how MoriPrepareAndFinalize talks to the op it was handed. mori raises on
dispatch_recv/combine_recv for everything but AsyncLL, so getting this wrong
is an exception on the first TBO step, not a slow path. Keyed on the kernel
name so these run without mori -- the inter-node kernels cannot be exercised
on a single-node box at all.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

import torch

from atom.model_ops.fused_moe import mori_prepare_finalize as mpf

# Every kernel aiter can emit. A new one must be classified below too, so it
# cannot silently inherit a protocol.
_KNOWN_KERNELS = ["IntraNode", "InterNodeV1", "AsyncLL", "InterNodeV1LL"]

# Ground truth from mori/python/mori/ops/dispatch_combine.py.
_SPLIT_SEND_RECV_KERNELS = {"AsyncLL"}


@pytest.mark.parametrize("kernel_name", _KNOWN_KERNELS)
def test_protocol_matches_what_the_kernel_implements(kernel_name):
    assert mpf.MoriPrepareAndFinalize.uses_split_send_recv(kernel_name) == (
        kernel_name in _SPLIT_SEND_RECV_KERNELS
    )


def test_only_one_kernel_has_the_split_api():
    """Guards the inverse too: nothing else may opt into send/recv."""
    classified = [
        k for k in _KNOWN_KERNELS if mpf.MoriPrepareAndFinalize.uses_split_send_recv(k)
    ]
    assert classified == ["AsyncLL"]


def _fake_op(kernel_name):
    return SimpleNamespace(
        config=SimpleNamespace(kernel_type=SimpleNamespace(name=kernel_name))
    )


def _prepare_finalize(kernel_name):
    """A MoriPrepareAndFinalize with every mori object stubbed out."""
    with patch.object(mpf, "MORI_AVAILABLE", True):
        return mpf.MoriPrepareAndFinalize(
            _fake_op(kernel_name),
            max_tokens_per_rank=8,
            num_dispatchers=2,
            dispatch_format=mpf.MoriDispatchFormat(
                dtype=torch.bfloat16,  # passthrough: no quantizer runs
                quant_type=None,
                scale_dim=0,
                scale_type_size=4,
            ),
            is_async=True,
            tbo_mori_ops=[_fake_op(kernel_name), _fake_op(kernel_name)],
        )


@pytest.mark.parametrize(
    "kernel_name,expected",
    [
        ("IntraNode", "comm_stream"),
        ("InterNodeV1", "comm_stream"),
        ("InterNodeV1LL", "comm_stream"),
        ("AsyncLL", "ll"),
    ],
)
def test_async_path_follows_the_kernel(kernel_name, expected, monkeypatch):
    """A non-AsyncLL op reaching _prepare_async_ll would hit a dispatch_recv
    that mori raises on; this is what keeps InterNodeV1LL off that path."""
    pf = _prepare_finalize(kernel_name)
    taken = []
    for name in ("_prepare_async_ll", "_prepare_async_comm_stream"):
        monkeypatch.setattr(
            pf, name, lambda *a, _n=name, **k: taken.append(_n) or "receiver"
        )
    for name in ("_finalize_async_ll", "_finalize_async_comm_stream"):
        monkeypatch.setattr(
            pf, name, lambda *a, _n=name, **k: taken.append(_n) or "receiver"
        )

    tok = torch.zeros(2, 4, dtype=torch.bfloat16)
    ids = torch.zeros(2, 2, dtype=torch.int32)
    wts = torch.zeros(2, 2, dtype=torch.float32)
    pf.prepare_async(tok, wts, ids, 4, None, False)
    pf.finalize_async(tok, tok, wts, ids, False)

    assert taken == [f"_prepare_async_{expected}", f"_finalize_async_{expected}"]


def test_supports_async_is_false_outside_tbo():
    """A constant True would drive prepare_async on a thread with no
    TBOContext; the layer is built once and used on both kinds of step."""
    pf = _prepare_finalize("IntraNode")
    assert pf.supports_async() is False


def test_protocol_is_derived_not_passed_in():
    """The layer reads the kernel off the op it was handed rather than being
    told, so it cannot be configured into a protocol mori would raise on."""
    assert _prepare_finalize("InterNodeV1LL")._low_latency is False
    assert _prepare_finalize("AsyncLL")._low_latency is True
