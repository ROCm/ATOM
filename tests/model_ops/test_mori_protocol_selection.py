# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Select the MoRI async protocol from latency mode and topology.

aiter owns the (low_latency, internode) -> kernel mapping; what stays here is
how MoriPrepareAndFinalize talks to the selected op. MoRI exposes split
dispatch/receive only for intra-node low latency; the inter-node low-latency
kernel uses the comm-stream protocol.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

import torch

from atom.model_ops.fused_moe import mori_prepare_finalize as mpf


def _prepare_finalize(num_ops=2, *, low_latency=False, internode=False):
    """A MoriPrepareAndFinalize with every mori object stubbed out."""
    mori_ops = [object() for _ in range(num_ops)]
    with patch.object(mpf, "MORI_AVAILABLE", True):
        return mpf.MoriPrepareAndFinalize(
            mori_ops,
            max_tokens_per_rank=8,
            num_dispatchers=2,
            dispatch_format=mpf.MoriDispatchFormat(
                dtype=torch.bfloat16,  # passthrough: no quantizer runs
                quant_type=None,
                scale_dim=0,
                scale_type_size=4,
            ),
            low_latency=low_latency,
            internode=internode,
        )


@pytest.mark.parametrize(
    "low_latency,internode,expected",
    [
        (False, False, "comm_stream"),
        (False, True, "comm_stream"),
        (True, True, "comm_stream"),
        (True, False, "ll"),
    ],
)
def test_async_path_follows_latency_mode_and_topology(
    low_latency, internode, expected, monkeypatch
):
    """Only intra-node low latency may use the split send/recv protocol."""
    pf = _prepare_finalize(low_latency=low_latency, internode=internode)
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


def test_supports_async_requires_active_tbo_and_a_second_slot():
    with patch("atom.utils.tbo.ubatching.tbo_active", return_value=False):
        assert _prepare_finalize(num_ops=2).supports_async() is False

    with patch("atom.utils.tbo.ubatching.tbo_active", return_value=True):
        assert _prepare_finalize(num_ops=1).supports_async() is False
        assert _prepare_finalize(num_ops=2).supports_async() is True
