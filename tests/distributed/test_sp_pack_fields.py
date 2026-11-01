# SPDX-License-Identifier: MIT
"""Both Ulysses QKV transports must hand each rank the heads its queries use."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("drives a Triton kernel; needs a GPU", allow_module_level=True)

from atom.distributed import ulysses_sp
from atom.distributed.sp_kernels import pack_fields


@pytest.mark.parametrize("world", [2, 4, 8])
def test_rank_receives_its_query_heads_kv_groups_and_indexer(monkeypatch, world):
    # Fused q | k | v | indexer q | indexer k, inside a padded row stride.
    q_heads, kv_heads, index_heads, dim = 8, 2, 4, 16
    widths = [q_heads * dim, kv_heads * dim, kv_heads * dim, index_heads * dim, dim]
    offsets = torch.tensor([0, *widths]).cumsum(0).tolist()
    source = torch.randn(12, offsets[-1] + 32, device="cuda").bfloat16()[:, :-32]
    kv_shards, index_shards = min(world, kv_heads), min(world, index_heads)
    spec = (
        (offsets[0], widths[0], ulysses_sp._SCATTER),
        (offsets[1], widths[1], kv_shards),
        (offsets[2], widths[2], kv_shards),
        (offsets[3], widths[3], index_shards),
        (offsets[4], widths[4], ulysses_sp._REPLICATE),
    )
    packed = pack_fields(source, spec, world)

    def heads(field, indices):
        return [
            source[:, offsets[field] + h * dim : offsets[field] + (h + 1) * dim]
            for h in indices
        ]

    monkeypatch.setattr(ulysses_sp, "_SP_WORLD_SIZE", world)
    monkeypatch.setattr(ulysses_sp, "prefer_custom_gather", lambda x, group: True)
    for rank in range(world):
        queries = range(rank * q_heads // world, (rank + 1) * q_heads // world)
        kv = sorted({h // (q_heads // kv_heads) for h in queries})
        index = sorted({h // (q_heads // index_heads) for h in queries})
        expected = torch.cat(
            heads(0, queries)
            + heads(1, kv)
            + heads(2, kv)
            + heads(3, index)
            + heads(4, [0]),
            dim=1,
        )
        assert torch.equal(packed[rank], expected)
        group = SimpleNamespace(rank_in_group=rank, custom_all_gather=lambda x: x)
        monkeypatch.setattr(ulysses_sp, "get_sp_group", lambda group=group: group)
        gathered = ulysses_sp._exchange_columns(source, spec, SimpleNamespace())
        assert torch.equal(gathered, expected)
