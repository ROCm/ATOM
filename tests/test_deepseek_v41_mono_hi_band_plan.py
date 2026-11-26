# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""atom/models/deepseek_v41/mono/hi_bands.py: the width plan (static widths unchanged without bands; a band's widths
share its programs) and the live words staged each step."""

import pytest
import torch

from atom.models.deepseek_v41.mono import hi_bands as B


def test_no_bands_is_the_static_split():
    """V4.1's widths (6 rows a request, static up to 4, one request up to 16 rows): as before, and 5+ refused."""
    got = B.plan_widths([1, 2, 6, 12, 18, 24], 6, 4, one_request_max=16)
    assert {w: (x.requests, x.req_rows, x.band) for w, x in got.items()} == {
        1: (1, 1, None), 2: (1, 2, None), 6: (1, 6, None), 12: (2, 6, None), 18: (3, 6, None), 24: (4, 6, None)
    }  # fmt: skip
    assert all(x.prog_rows == w for w, x in got.items())
    with pytest.raises(ValueError):
        B.plan_widths([6, 30], 6, 4, one_request_max=16)
    # one request past 6 rows, no batch width beside it
    assert B.plan_widths([8, 16], 6, 4, one_request_max=16)[16].requests == 1
    # a one-request width past 6 rows beside a batch width
    with pytest.raises(ValueError):
        B.plan_widths([12, 16], 6, 4, one_request_max=16)


def test_bands_start_past_the_static_widths():
    """V4.1's bands 6, 8 (widths 30-48: the smallest band equals static_max + 1 .. 2)."""
    got = B.plan_widths([6, 24, 30, 36, 42, 48], 6, 4, (6, 8))
    assert got[24].band is None and got[24].prog_rows == 24
    for w, band in ((30, 6), (36, 6), (42, 8), (48, 8)):
        assert (got[w].band, got[w].prog_rows) == (band, band * 6)


def test_flash_bands_to_16_requests():
    """Flash (4 rows a request, static B1-4): bands 6 / 8 / 12 / 16 serve 5-16 requests."""
    widths = [4 * b for b in range(1, 17)]
    got = B.plan_widths(widths, 4, 4, (6, 8, 12, 16))
    band = {w // 4: got[w].band for w in widths}
    assert band == {1: None, 2: None, 3: None, 4: None, 5: 6, 6: 6, 7: 8, 8: 8, 9: 12, 10: 12, 11: 12, 12: 12,
                    13: 16, 14: 16, 15: 16, 16: 16}  # fmt: skip
    with pytest.raises(ValueError):
        B.plan_widths([68], 4, 4, (6, 8, 12, 16))  # 17 requests: no band


def test_live_words_and_dead_slots():
    widths = B.plan_widths([16, 20, 24, 28, 32], 4, 4, (6, 8))
    step = torch.zeros(4, dtype=torch.int32)
    lw = B.LiveWords(step, widths)
    lw.stage(20)
    assert step.tolist() == [0, 20, 4, 5]
    lw.stage(32)
    assert step.tolist() == [0, 32, 4, 8]
    step[0] = 7
    lw.stage(16)  # a static width: the words stay
    assert step.tolist() == [7, 32, 4, 8]
    assert lw.dead(20) == range(5, 6) and lw.dead(28) == range(7, 8)
    assert lw.dead(24) is None and lw.dead(32) is None and lw.dead(16) is None
    # the counter alone: no room for the words
    with pytest.raises(ValueError):
        B.LiveWords(torch.zeros(1, dtype=torch.int32), widths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA graph")
def test_live_words_staged_in_a_graph():
    widths = B.plan_widths([16, 20, 24], 4, 4, (6,))
    step = torch.zeros(4, dtype=torch.int32, device="cuda")
    lw = B.LiveWords(step, widths)
    lw.stage(24)  # warm
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        lw.stage(20)
    step.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert step.tolist() == [0, 20, 4, 5]
