# SPDX-License-Identifier: MIT
"""Real UVA lookup under warmup, capture, and changing graph replay inputs.

Run pytest on one GPU, or torchrun --nproc-per-node=4 -m
tests.model_ops.engram.test_overlap to include AITER collectives.
"""

import mmap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    # Before the imports, which reach Triton through the hash kernels. As a
    # `pytestmark` this ran after them, so a CPU runner failed collection
    # instead of skipping.
    pytest.skip("requires GPU", allow_module_level=True)

from atom.model_ops.engram.device.hashing import (
    EngramHashTables,
    engram_row_indices_reference,
    engram_snapshot,
    engram_snapshot_indices,
)
from atom.model_ops.engram.device.runtime import EngramBatch
from atom.model_ops.engram.device.staging import EngramStaging
from atom.model_ops.engram.mapping import EngramConfig
from atom.model_ops.engram.tables import HostEmbeddingTable
from tests.model_ops.engram.test_hash_bounds import V41_FLASH, build, tiny_config
from tests.model_ops.engram.test_hashing import case, compress

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def make_batch(mapping, lengths, seed):
    tokens, histories, masks = case(
        mapping,
        lengths,
        seed=seed,
        dead=[(0, -1)],
        masked=[(len(lengths) - 1, lengths[-1] - 1)],
    )
    device = torch.device("cuda")
    tables = EngramHashTables.from_mapping(mapping, device)
    flat = torch.as_tensor(np.concatenate(tokens), dtype=torch.int32, device=device)
    # Deliberately permuted slots and strided cursor history.
    slots = torch.arange(len(lengths) - 1, -1, -1, dtype=torch.int32, device=device)
    cursor = torch.zeros(len(lengths), tables.ngram, dtype=torch.int64, device=device)
    cursor[slots.long(), 1:] = torch.as_tensor(np.stack(histories), device=device)
    batch = EngramBatch(
        compress(tables, flat, masks, device),
        torch.as_tensor(
            np.repeat(np.arange(len(lengths)), lengths),
            dtype=torch.int32,
            device=device,
        ),
        torch.as_tensor(
            np.concatenate(([0], np.cumsum(lengths))), dtype=torch.int32, device=device
        ),
        cursor[:, 1:],
        slots,
    )
    return batch, tokens, histories, masks


@pytest.mark.parametrize("config", [tiny_config(), EngramConfig.from_hf(V41_FLASH)])
@pytest.mark.parametrize("lengths", [[1, 1], [1, 6, 3], [64]])
def test_snapshot_keeps_old_history_and_padding(config, lengths):
    mapping = build(config)
    tables = EngramHashTables.from_mapping(mapping, torch.device("cuda"))
    batch, tokens, histories, masks = make_batch(mapping, lengths, 7)
    count = sum(lengths)
    snapshot = torch.empty(count + 5, tables.ngram, dtype=torch.int64, device="cuda")
    engram_snapshot(tables, batch, snapshot)
    # Emulate prepare advancing the cursor, and release all temporary inputs.
    batch.history.fill_(31)
    batch.compressed.fill_(17)
    for layer in config.layer_ids:
        out = torch.empty(count + 5, tables.heads, dtype=torch.int64, device="cuda")
        engram_snapshot_indices(tables, layer, snapshot, out)
        expected = engram_row_indices_reference(
            mapping, layer, tokens, histories, masks
        )
        np.testing.assert_array_equal(out[:count].cpu().numpy(), expected)
        assert torch.all(out[count:] == -1)


def make_staging(mapping, group=None):
    config = mapping.config
    device = torch.device("cuda", torch.cuda.current_device())
    size = 1 if group is None else group.world_size
    rank = 0 if group is None else group.rank_in_group
    local_heads = (config.num_hash_heads + size - 1) // size
    tables, backing = {}, []
    for layer, rows in zip(config.layer_ids, config.num_embeddings):
        memory = mmap.mmap(-1, rows * config.head_dim * 2)
        backing.append(memory)
        tensor = torch.frombuffer(memory, dtype=torch.bfloat16).view(
            rows, config.head_dim
        )
        tensor.copy_(
            (torch.arange(rows * config.head_dim).view_as(tensor) % 127).float()
        )
        table = HostEmbeddingTable(tensor, rows, config.head_dim)
        start_head = rank * local_heads
        end_head = min(config.num_hash_heads, start_head + local_heads)
        start = int(mapping.head_offsets[layer][start_head])
        end = int(
            mapping.head_offsets[layer][end_head - 1]
            + mapping.head_vocab_sizes[layer][end_head - 1]
        )
        assert table.enable_uva(start, end)
        tables[layer] = table
    host = SimpleNamespace(
        device=device,
        max_num_tokens=32,
        layer_ids=config.layer_ids,
        local_heads=local_heads,
        head_start=rank * local_heads,
        total_heads=config.num_hash_heads,
        embed_width=config.num_hash_heads * config.head_dim,
        prefetcher=SimpleNamespace(_tables=tables),
        _tp_group=group,
        buffers={
            layer: SimpleNamespace(
                gpu=torch.empty(
                    32,
                    config.num_hash_heads * config.head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
            )
            for layer in config.layer_ids
        },
    )
    # The staging hangs off the device lookup, which owns the hash tables and
    # the row-index buffer; the host half of it is what holds the rest.
    uva = SimpleNamespace(
        host=host,
        hash_tables=EngramHashTables.from_mapping(mapping, device),
        row_ids=torch.empty(
            32, config.num_hash_heads, dtype=torch.int64, device=device
        ),
    )
    return EngramStaging(uva), backing


def exercise_replay(group=None):
    from contextlib import nullcontext

    from aiter.dist.parallel_state import graph_capture

    mapping = build(tiny_config())
    staging, _ = make_staging(mapping, group)
    if group is not None:
        assert staging.collective is not None
        assert staging.collective is not group.device_communicator.ca_comm
    output = {
        layer: torch.empty_like(buffer.gpu)
        for layer, buffer in staging.host.buffers.items()
    }
    x = torch.ones(128, 128, device="cuda")
    y = torch.empty_like(x)
    reduced = torch.empty_like(x)
    rank = 0 if group is None else group.rank_in_group

    def forward(rows):
        rows.stage()
        # Stand in for layer 0, including a main-stream collective for TP.
        # Unequal work across ranks makes overlap with the independent
        # side-stream communicator sensitive to accidental shared state.
        for _ in range(rank + 1):
            torch.mm(x, x, out=y)
        if group is not None:
            reduced.copy_(group.all_reduce(y))
        for layer in mapping.config.layer_ids:
            output[layer][: rows.width].copy_(rows.get(layer)[0])
        rows.join()

    try:
        graphs = {}
        # One prepare followed by warmup AND capture, just like ModelRunner.
        with graph_capture() if group is not None else nullcontext() as ctx:
            for width in (8, 16):
                rows = staging.prepare(None, width)
                forward(rows)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, stream=None if ctx is None else ctx.stream
                ):
                    forward(rows)
                graphs[width] = graph
        for iteration in range(120):
            width = (8, 16)[iteration % 2]
            lengths = ([1, 2], [1, 6, 3], [1, 1])[iteration % 3]
            if sum(lengths) > width:
                lengths = [2, 3]
            batch, tokens, histories, masks = make_batch(
                mapping, lengths, iteration + 3
            )
            staging.prepare(batch, width)
            value = iteration % 5 + 1
            x.fill_(value + rank)
            # Must not hash the cursor after prepare updates it.
            batch.history.fill_(91)
            batch.compressed.fill_(23)
            graphs[width].replay()
            if group is not None:
                expected_sum = 128 * sum(
                    (value + peer) ** 2 for peer in range(group.world_size)
                )
                torch.testing.assert_close(
                    reduced, torch.full_like(reduced, expected_sum), rtol=0, atol=0
                )
            for layer, table in staging.host.prefetcher._tables.items():
                indices = engram_row_indices_reference(
                    mapping, layer, tokens, histories, masks
                )
                expected = table._tensor[torch.as_tensor(indices)].reshape(
                    sum(lengths), staging.host.embed_width
                )
                torch.testing.assert_close(
                    output[layer][: sum(lengths)].cpu(), expected, rtol=0, atol=0
                )
                assert torch.all(output[layer][sum(lengths) : width] == 0)
        # The eager path also reads fresh inputs after graphs have been used.
        batch, tokens, histories, masks = make_batch(mapping, [4, 5], 99)
        rows = staging.prepare(batch, 12)
        forward(rows)
        for layer, table in staging.host.prefetcher._tables.items():
            indices = engram_row_indices_reference(
                mapping, layer, tokens, histories, masks
            )
            expected = table._tensor[torch.as_tensor(indices)].reshape(9, -1)
            torch.testing.assert_close(
                output[layer][:9].cpu(), expected, rtol=0, atol=0
            )
        torch.cuda.synchronize()
    finally:
        torch.cuda.synchronize()
        if staging.collective is not None:
            staging.collective.close()
        for table in staging.host.prefetcher._tables.values():
            table.disable_uva()


def test_uva_graph_replay_uses_each_steps_inputs():
    exercise_replay()


if __name__ == "__main__":
    import os

    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    initialize_model_parallel(tensor_model_parallel_size=size)
    try:
        exercise_replay(get_tp_group())
        if rank == 0:
            print(
                "PASS: private TP gather + main all-reduce, 2 graph buckets, "
                "120 changing replays with rank skew"
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
