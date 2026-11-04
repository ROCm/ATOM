# SPDX-License-Identifier: MIT
"""The sampled DSpark Markov op's input contract (`dspark_markov_sample`).
Its draws and store are checked on the GPU (`test_spec_lossless_kernels`)."""

import pytest
import torch

pytest.importorskip("aiter", reason="the Markov sampler is a Triton/AITER op")

from atom.model_ops.spec_sampling.dspark_markov_sample import (
    dspark_markov_sample,
    dspark_markov_sample_scratch,
)
from atom.model_ops.spec_sampling.sparse_q_store import SparseQStore
from tests.test_dspark_markov_argmax import _ids, _tables


def test_the_sampled_op_checks_its_tables_as_the_greedy_one_does():
    """Both ops index W2 by the logits' column: a table of another vocab must
    raise in the sampled op too, not read the wrong rows."""
    rows, vocab = 5, 64
    base, prev, w1, w2 = _tables(rows, vocab, 16, "cpu")
    store = SparseQStore.allocate(rows, 8, "cpu")
    scratch = dspark_markov_sample_scratch(rows, vocab, "cpu")
    i32 = {"dtype": torch.int32}
    rest = (
        *store,
        torch.empty(rows, vocab),
        torch.zeros(rows, **i32),
        0,
        torch.ones(rows),
        torch.zeros(rows, **i32),
        torch.ones(rows),
        torch.zeros(1, **i32),
        0,
        scratch.buf,
    )
    with pytest.raises(ValueError, match="markov_w2"):
        dspark_markov_sample(base, prev, w1, w2[:-1], _ids(rows, "cpu"), *rest)
    with pytest.raises(ValueError, match="markov_w1"):
        dspark_markov_sample(base, prev, w1[:-1], w2, _ids(rows, "cpu"), *rest)
