# SPDX-License-Identifier: MIT
"""Both Ulysses QKV transports must hand each rank the heads its queries use."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("drives a Triton kernel; needs a GPU", allow_module_level=True)

from atom.distributed import ulysses_sp
from atom.distributed.sp_kernels import pack_columns


@pytest.mark.parametrize("world", [2, 4, 8])
def test_rank_receives_its_query_heads_kv_groups_and_indexer(monkeypatch, world):
    # Fused q | k | v | indexer q | indexer k, inside a padded row stride.
    q_heads, kv_heads, index_heads, dim = 8, 2, 4, 16
    fields = [(q_heads, dim), (kv_heads, dim), (kv_heads, dim), (index_heads, dim)]
    fields.append((1, dim))
    offsets = torch.tensor([0] + [h * d for h, d in fields]).cumsum(0).tolist()
    source = torch.randn(12, offsets[-1] + 32, device="cuda").bfloat16()[:, :-32]

    def heads(field, indices):
        return [
            source[:, offsets[field] + h * dim : offsets[field] + (h + 1) * dim]
            for h in indices
        ]

    settings = SimpleNamespace(sequence_parallel_size=world)
    monkeypatch.setattr(ulysses_sp, "get_current_atom_config", lambda: settings)
    columns = ulysses_sp.send_columns(fields, source.device)
    packed = pack_columns(source, columns)
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
        comm = SimpleNamespace(all_gather=lambda x, dim: x)
        group = SimpleNamespace(rank_in_group=rank, device_communicator=comm)
        gathered = ulysses_sp.exchange_columns(source, group, columns)
        assert torch.equal(gathered, expected)
