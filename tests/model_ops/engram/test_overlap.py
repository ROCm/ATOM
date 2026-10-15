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
    EngramBatch,
    EngramHashTables,
    engram_cursor_rows_reference,
    engram_row_indices_reference,
    engram_snapshot,
    engram_snapshot_indices,
)
from atom.model_ops.engram.device.staging import EngramStaging, EngramStep
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


@pytest.mark.parametrize("config", [tiny_config(), EngramConfig.from_hf(V41_FLASH)])
@pytest.mark.parametrize("lengths", [[1, 1], [1, 6, 3], [6] * 8, [129, 3, 1]])
@pytest.mark.parametrize("replay", [False, True])
def test_snapshot_stages_all_cursor_prefixes_without_mutating_history(
    config, lengths, replay
):
    mapping = build(config)
    tables = EngramHashTables.from_mapping(mapping, torch.device("cuda"))
    batch, tokens, histories, masks = make_batch(mapping, lengths, 7)
    count = sum(lengths)
    starts = np.arange(len(lengths)) * 31 + 2
    positions = torch.tensor(
        np.concatenate(
            [np.arange(start, start + n) for start, n in zip(starts, lengths)]
        ),
        dtype=torch.int64,
        device="cuda",
    )
    # Unscheduled prefixes and requests must retain their sentinels. Snapshot
    # padding is still written as -2, including across a CTA boundary.
    snapshot = torch.empty(count + 5, tables.ngram, dtype=torch.int64, device="cuda")
    candidates = torch.full(
        (len(lengths) + 1, max(lengths) + 2, tables.ngram),
        -99,
        dtype=torch.int64,
        device="cuda",
    )
    original = batch.history.clone()

    def prepare():
        engram_snapshot(
            tables, batch, snapshot, cursor_positions=positions, cursor_out=candidates
        )

    prepare()
    if replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepare()
        # Fixed addresses, changed data: replay must not reuse a cached result.
        batch.compressed.fill_(-1)
        candidates.fill_(-99)
        graph.replay()
        masks = [np.zeros(n, dtype=bool) for n in lengths]
    expected = engram_cursor_rows_reference(mapping, tokens, histories, starts, masks)
    for i, (length, rows) in enumerate(zip(lengths, expected)):
        np.testing.assert_array_equal(candidates[i, :length].cpu().numpy(), rows)
        assert torch.all(candidates[i, length:] == -99)
    assert torch.all(candidates[len(lengths) :] == -99)
    assert torch.equal(batch.history, original)
    assert torch.all(snapshot[count:] == -2)
    for layer in config.layer_ids:
        out = torch.empty(count + 5, tables.heads, dtype=torch.int64, device="cuda")
        engram_snapshot_indices(tables, layer, snapshot, out)
        np.testing.assert_array_equal(
            out[:count].cpu().numpy(),
            engram_row_indices_reference(mapping, layer, tokens, histories, masks),
        )
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


MAX_BS = 8


def make_step(tables, max_tokens, verify):
    """``EngramStep`` over fixed buffers, as the V4.1 metadata publishes them."""
    device = torch.device("cuda")

    def zeros(*shape, dtype=torch.int32):
        return torch.zeros(*shape, dtype=dtype, device=device)

    rows = zeros(max_tokens + 1)
    candidates = (
        torch.full((MAX_BS, max_tokens, tables.ngram), -99, dtype=torch.int64,
                   device=device)
        if verify
        else None
    )  # fmt: skip
    return EngramStep(
        input_ids=zeros(max_tokens),
        live=rows[:1],
        dead=rows[1:],
        batch_ids=zeros(max_tokens),
        cu_seqlens=zeros(MAX_BS + 1),
        positions=zeros(max_tokens),
        cursor=zeros(MAX_BS, tables.ngram, dtype=torch.int64),
        history_index=zeros(MAX_BS),
        candidates=candidates,
    )


def fill_step(step, tokens, histories, masks, starts, live=True):
    """One step's metadata into the fixed buffers (what a replay reads)."""
    lengths = [len(t) for t in tokens]
    count, n = sum(lengths), len(lengths)
    keep = np.concatenate(masks) if masks is not None else np.ones(count, bool)
    # Deliberately permuted slots; padding requests get no tokens.
    slots = np.arange(n - 1, -1, -1)
    step.input_ids.zero_()
    step.input_ids[:count] = torch.as_tensor(np.concatenate(tokens))
    step.live.fill_(count if live else 0)
    step.dead.zero_()
    step.dead[:count] = torch.as_tensor((~keep).astype(np.int32))
    step.batch_ids.fill_(-1)
    step.batch_ids[:count] = torch.as_tensor(np.repeat(np.arange(n), lengths))
    cu = np.concatenate(([0], np.cumsum(lengths)))
    step.cu_seqlens.fill_(count)
    step.cu_seqlens[: n + 1] = torch.as_tensor(cu)
    step.positions[:count] = torch.as_tensor(
        np.concatenate([np.arange(a, a + k) for a, k in zip(starts, lengths)])
    )
    step.cursor.fill_(-7)
    step.cursor[torch.as_tensor(slots, device="cuda"), 1:] = torch.as_tensor(
        np.stack(histories), device="cuda"
    )
    step.history_index.zero_()
    step.history_index[:n] = torch.as_tensor(slots)
    if step.candidates is not None:
        step.candidates.fill_(-99)
    return slots


def exercise_replay(group=None, verify=False):
    from contextlib import nullcontext

    from aiter.dist.parallel_state import graph_capture

    mapping = build(tiny_config())
    staging, _ = make_staging(mapping, group)
    tables = staging.uva.hash_tables
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
    step = make_step(tables, staging.host.max_num_tokens, verify)

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

    def check(tokens, histories, masks, starts, slots, width):
        lengths = [len(t) for t in tokens]
        count = sum(lengths)
        for layer, table in staging.host.prefetcher._tables.items():
            indices = engram_row_indices_reference(
                mapping, layer, tokens, histories, masks
            )
            expected = table._tensor[torch.as_tensor(indices)].reshape(
                count, staging.host.embed_width
            )
            torch.testing.assert_close(
                output[layer][:count].cpu(), expected, rtol=0, atol=0
            )
            assert torch.all(output[layer][count:width] == 0)
        cursors = engram_cursor_rows_reference(
            mapping, tokens, histories, starts, masks
        )
        if verify:
            # every accepted prefix staged, the committed history untouched
            for i, (length, rows) in enumerate(zip(lengths, cursors)):
                got = step.candidates[i, :length].cpu().numpy()
                np.testing.assert_array_equal(got, rows)
                assert torch.all(step.candidates[i, length:] == -99)
            assert torch.all(step.candidates[len(lengths) :] == -99)
            for i, slot in enumerate(slots):
                np.testing.assert_array_equal(
                    step.cursor[slot, 1:].cpu().numpy(), histories[i]
                )
        else:
            # the cursor advanced in the forward, after the snapshot read it
            for i, slot in enumerate(slots):
                np.testing.assert_array_equal(
                    step.cursor[slot].cpu().numpy(), cursors[i][lengths[i] - 1]
                )

    try:
        graphs = {}
        # One prepare followed by warmup AND capture, just like ModelRunner;
        # the capture's step has no live token, so it writes nothing.
        with graph_capture() if group is not None else nullcontext() as ctx:
            for width in (8, 16):
                fill_step(step, [[1, 2]], [[3] * tables.width], None, [5], live=False)
                before = step.cursor.clone()
                rows = staging.prepare(step, width)
                forward(rows)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, stream=None if ctx is None else ctx.stream
                ):
                    forward(rows)
                graphs[width] = graph
                assert torch.equal(step.cursor, before)
        for iteration in range(120):
            width = (8, 16)[iteration % 2]
            lengths = ([1, 2], [1, 6, 3], [1, 1])[iteration % 3]
            if sum(lengths) > width:
                lengths = [2, 3]
            tokens, histories, masks = case(
                mapping, lengths, seed=iteration + 3, dead=[(0, -1)],
                masked=[(len(lengths) - 1, lengths[-1] - 1)],
            )  # fmt: skip
            starts = list(np.arange(len(lengths)) * 31 + 2)
            slots = fill_step(step, tokens, histories, masks, starts)
            value = iteration % 5 + 1
            x.fill_(value + rank)
            graphs[width].replay()
            if group is not None:
                expected_sum = 128 * sum(
                    (value + peer) ** 2 for peer in range(group.world_size)
                )
                torch.testing.assert_close(
                    reduced, torch.full_like(reduced, expected_sum), rtol=0, atol=0
                )
            check(tokens, histories, masks, starts, slots, width)
        # A DP dummy replays a graph with no live token: nothing moves.
        tokens, histories, masks = case(mapping, [2, 3], seed=7)
        fill_step(step, tokens, histories, masks, [4, 9], live=False)
        cursor, staged = step.cursor.clone(), (
            None if step.candidates is None else step.candidates.clone()
        )
        graphs[8].replay()
        torch.cuda.synchronize()
        assert torch.equal(step.cursor, cursor)
        if staged is not None:
            assert torch.equal(step.candidates, staged)
        # The eager path also reads fresh inputs after graphs have been used.
        tokens, histories, masks = case(mapping, [4, 5], seed=99)
        slots = fill_step(step, tokens, histories, masks, [6, 40])
        rows = staging.prepare(step, 12)
        forward(rows)
        check(tokens, histories, masks, [6, 40], slots, 12)
        torch.cuda.synchronize()
    finally:
        torch.cuda.synchronize()
        if staging.collective is not None:
            staging.collective.close()
        for table in staging.host.prefetcher._tables.values():
            table.disable_uva()


@pytest.mark.parametrize("verify", [False, True])
def test_uva_graph_replay_uses_each_steps_inputs(verify):
    exercise_replay(verify=verify)


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
