# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ATOM_DSV41_MONO_HI_MAX_REQUESTS``: the widths the hi hook serves."""

import pytest

pytest.importorskip("triton")

from atom.models.deepseek_v41.mono.hi_compose import ROWS  # noqa: E402
from atom.models.deepseek_v41.mono.hi_runner import served_rows  # noqa: E402


def test_default_serves_every_width():
    assert served_rows(8) == list(ROWS)


def test_capped_widths():
    assert served_rows(5) == [6, 12, 18, 24, 30]
    assert served_rows(1) == [6]


@pytest.mark.parametrize("n", [0, 9])
def test_out_of_range(n):
    with pytest.raises(ValueError):
        served_rows(n)
