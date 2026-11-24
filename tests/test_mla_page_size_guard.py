# SPDX-License-Identifier: MIT
"""Sparse MLA (DSA) runs only at ATOM_MLA_PAGE_SIZE=1: the indexer's top-k
indices are token slots, which a paged MLA kernel would read as page ids."""

import pytest


@pytest.mark.parametrize(
    "page_size,is_sparse_mla,rejected",
    [(1, True, False), (64, True, True), (64, False, False), (1, False, False)],
)
def test_sparse_mla_requires_page_size_one(page_size, is_sparse_mla, rejected):
    pytest.importorskip("triton", reason="attention_mla defines @triton.jit kernels")
    pytest.importorskip("aiter", reason="attention_mla imports the AITER runtime")
    from atom.model_ops.attention_mla import check_sparse_mla_page_size

    kwargs = {"mla_page_size": page_size, "is_sparse_mla": is_sparse_mla}
    if rejected:
        with pytest.raises(RuntimeError, match="ATOM_MLA_PAGE_SIZE=1"):
            check_sparse_mla_page_size(**kwargs)
    else:
        check_sparse_mla_page_size(**kwargs)
