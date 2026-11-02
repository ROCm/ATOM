# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3's runner shares one peer buffer between K2 and the DSA indexer. A
rank that finishes a layer's indexer runs MLA and then the next K2, whose AR
pushes land on every peer from offset 0; a peer still polling its indexer's
last-pass hist must not have those pairs there. So the indexer's regions start
past every K2 width's (``runner.ix_peer_offset``), at every width either runs."""

import pytest

pytest.importorskip("flydsl", reason="the K3 kernels' layouts import FlyDSL")
pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")

from atom.models.kimi_k3.mono.config import MAX_ROWS
from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.runner import ix_peer_offset
from atom.mono.dsa_indexer import kernel as kix
from atom.mono.dsa_indexer.plan import IndexerShape

SHAPE = IndexerShape(32, 128, 2048, 1536, "bf16", "fp8", 128, 0)
MAX_CTX = 131072  # max_model_len 128K (over DCP8)


def _end(layout):
    return max(o + n for o, n in layout.values())


@pytest.mark.parametrize("qlen", [1, 8])
def test_the_indexer_s_regions_start_past_every_k2_width_s(qlen):
    off = ix_peer_offset(MAX_ROWS)
    k2_end = max(_end(k2.peer_layout(s)) for s in range(1, MAX_ROWS + 1))
    assert off >= k2_end
    for rows in range(qlen, MAX_ROWS + 1, qlen):
        key = kix.IxBuild(SHAPE, rows, 8, MAX_CTX, qlen=qlen)
        if kix.unfit(key):
            continue
        # the indexer's layout is from its own start, ``off`` into the buffer
        assert off + min(o for o, _ in kix.peer_layout(key).values()) >= k2_end, rows


def test_a_shared_offset_of_zero_overlaps():
    """The layout this guards against: at offset 0 the indexer's hist sits in
    K2's AR region (the review's case: 16 rows)."""
    key = kix.IxBuild(SHAPE, 16, 8, MAX_CTX)
    hist_at, hist_n = kix.peer_layout(key)["hist"]
    ar_at, ar_n = k2.peer_layout(16)["ar"]
    assert hist_at < ar_at + ar_n and ar_at < hist_at + hist_n
