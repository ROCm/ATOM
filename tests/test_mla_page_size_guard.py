# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Sparse MLA (DSA) runs only at ATOM_MLA_PAGE_SIZE=1: the indexer's top-k
indices are token slots, which a paged MLA kernel would read as page ids.

Builds real MLAAttention layers against the stand-in AITER, so Pre Checkin
(no AITER) runs this file."""

import pytest
from test_mla_decode_min_heads import _layer, _set, aiter_shape, native  # noqa: F401


@pytest.mark.parametrize(
    "sparse, changes, rejected",
    [
        (True, ["page size 64"], True),
        # No segmented layout here, but sparse prefill still pages by the env.
        (True, ["page size 64", "triton mla"], True),
        (False, ["page size 64"], False),
        (False, ["page size 64", "triton mla"], False),
        (True, [], False),
    ],
    ids=["sparse-seg", "sparse-triton", "dense-seg", "dense-triton", "sparse-page1"],
)
def test_sparse_layer_requires_page_size_one(
    native, monkeypatch, sparse, changes, rejected  # noqa: F811
):
    _set(monkeypatch, native, "indexer")
    for change in changes:
        _set(monkeypatch, native, change)
    if rejected:
        with pytest.raises(
            RuntimeError, match="Sparse MLA requires ATOM_MLA_PAGE_SIZE=1"
        ):
            _layer(native, sparse=sparse)
    else:
        _layer(native, sparse=sparse)
