# SPDX-License-Identifier: MIT

import runpy
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
direct_register_custom_op = runpy.run_path(
    ROOT / "atom" / "utils" / "custom_register.py"
)["direct_register_custom_op"]
TEST_LIB = torch.library.Library("atom_test_sparse_indexer", "FRAGMENT")


def _write_index(
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_indptr: torch.Tensor,
    owned_counts: torch.Tensor,
) -> None:
    kv_cache.copy_(weights.flatten()[: kv_cache.numel()])
    sparse_indices.copy_(torch.arange(sparse_indices.numel(), dtype=torch.int32))
    sparse_indptr.add_(1)
    owned_counts.add_(2)


direct_register_custom_op(
    op_name="write_index",
    op_func=_write_index,
    mutates_args=["kv_cache", "sparse_indices", "sparse_indptr", "owned_counts"],
    fake_impl=lambda *_args, **_kwargs: None,
    target_lib=TEST_LIB,
    dispatch_key="CPU",
)


def test_side_effect_op_stays_ordered_without_a_synthetic_output():
    op = torch.ops.atom_test_sparse_indexer.write_index.default
    assert len(op._schema.returns) == 0
    assert {
        argument.name
        for argument in op._schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    } == {
        "kv_cache",
        "sparse_indices",
        "sparse_indptr",
        "owned_counts",
    }

    def fn(weights, kv_cache, sparse_indices, sparse_indptr, owned_counts):
        torch.ops.atom_test_sparse_indexer.write_index(
            weights, kv_cache, sparse_indices, sparse_indptr, owned_counts
        )
        return torch.cat((kv_cache.float(), sparse_indices.float()))

    weights = torch.arange(7 * 13, dtype=torch.bfloat16).reshape(7, 13)
    cache_storage = torch.zeros(5, dtype=torch.bfloat16)
    kv_cache = cache_storage[1:4]
    sparse_indices = torch.full((3,), -1, dtype=torch.int32)
    sparse_indptr = torch.zeros(2, dtype=torch.int32)
    owned_counts = torch.zeros(2, dtype=torch.int32)
    output = torch.compile(fn, fullgraph=True)(
        weights,
        kv_cache,
        sparse_indices,
        sparse_indptr,
        owned_counts,
    )
    torch.testing.assert_close(output, torch.tensor([0, 1, 2, 0, 1, 2.0]))
