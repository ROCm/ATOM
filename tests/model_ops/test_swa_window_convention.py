# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every MHA path must give a sliding-window layer the same keys.

`sliding_window` is the HF span: a query sees itself and the `W - 1` keys
before it. aiter's varlen and unified kernels both take an *inclusive* left
offset (`[i - left, i]`), so the left window they need is `W - 1`. The CK
prefill path once passed `W`, which let prefill see one key more than decode.
"""

import pytest

pytest.importorskip("aiter")

from atom.model_ops.base_attention import window_left


@pytest.mark.parametrize("span", [1, 128, 2048])
def test_left_offset_is_span_minus_one(span):
    assert window_left(span) == span - 1


@pytest.mark.parametrize("span", [-1, 0, None])
def test_no_window_is_unbounded(span):
    assert window_left(span) == -1
