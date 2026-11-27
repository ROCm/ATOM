import types

import pytest
import torch

embed_head = pytest.importorskip(
    "atom.model_ops.embed_head",
    reason="embed_head requires the Triton/AITER runtime",
    exc_type=ImportError,
)

VOCAB, DIM, TP = 16, 8, 4


def _embedding(monkeypatch, rank: int, full_table: bool = True):
    group = types.SimpleNamespace(rank_in_group=rank, world_size=TP)
    monkeypatch.setattr(embed_head, "get_tp_group", lambda: group)
    return embed_head.VocabParallelEmbedding(VOCAB, DIM, full_table=full_table)


def _table(seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    # Nonzero, normal entries: the tables the full-table lookup serves. In the
    # default dtype for the loader tests (the module allocates in it).
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(VOCAB, DIM, generator=g) + 0.5
    sign = torch.randint(0, 2, (VOCAB, DIM), generator=g) * 2 - 1
    return (t * sign).to(dtype)


@pytest.mark.parametrize("rank", range(TP))
def test_shard_is_a_view_of_the_loaded_table(monkeypatch, rank):
    emb = _embedding(monkeypatch, rank)
    table = _table()
    emb.weight.weight_loader(emb.weight, table)
    assert emb.replicated_lookup
    assert torch.equal(emb.table, table)
    # A drafter sharing the embedding addresses `weight` by `vocab_start_idx`.
    shard = VOCAB // TP
    assert torch.equal(emb.weight, table[rank * shard : (rank + 1) * shard])
    assert emb.weight.data_ptr() == emb.table[rank * shard].data_ptr()


def test_sharded_embedding_is_unchanged(monkeypatch):
    emb = _embedding(monkeypatch, 1, full_table=False)
    assert emb.table is None and not emb.replicated_lookup
    table = _table()
    emb.weight.weight_loader(emb.weight, table)
    assert torch.equal(emb.weight, table[4:8])
    assert not emb.replicated_lookup


@pytest.mark.parametrize("bad", [0.0, -0.0, float("nan"), 1e-40])
def test_tables_the_all_reduce_would_change_keep_it(monkeypatch, bad):
    emb = _embedding(monkeypatch, 0)
    table = _table()
    table[5, 3] = bad
    emb.weight.weight_loader(emb.weight, table)
    assert not emb.replicated_lookup


def test_row_equals_the_sum_of_the_masked_shards():
    """The identity the full-table lookup rests on: the all-reduce of the ranks'
    masked lookups (zeros outside a rank's shard) is the row itself, bit for
    bit, in either accumulation order."""
    table = _table(1, torch.bfloat16)
    shard = VOCAB // TP
    for token in range(VOCAB):
        parts = [
            table[token] if r * shard <= token < (r + 1) * shard
            else torch.zeros(DIM, dtype=torch.bfloat16)
            for r in range(TP)
        ]
        fp32 = sum(p.float() for p in parts).to(torch.bfloat16)
        bf16 = parts[0]
        for p in parts[1:]:
            bf16 = bf16 + p
        row = table[token]
        assert torch.equal(fp32.view(torch.int16), row.view(torch.int16))
        assert torch.equal(bf16.view(torch.int16), row.view(torch.int16))
