# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8 PLE n-gram table in page-locked host memory, read over UVA.

The CPU half checks `PleHostTable`'s shard addresses by following them the way
the kernel does; the GPU half checks the kernel against the HBM embedding it
replaces, bit for bit.
"""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.engram import host_memory
from atom.model_ops.qwen4_exp import ple_host_table
from atom.model_ops.qwen4_exp.ple_host_table import PleHostTable

DIM = 160
SHARD_ROWS = 37
NUM_SHARDS = 6
# The last shard is short, like a checkpoint whose padded table is not a
# multiple of the shard width.
TABLE_ROWS = SHARD_ROWS * NUM_SHARDS - 5


def _table(dtype=torch.bfloat16, rows=TABLE_ROWS):
    gen = torch.Generator().manual_seed(0)
    return torch.randn(rows, DIM, generator=gen).to(dtype)


def _shards(table):
    return list(table.split(SHARD_ROWS))


def _probe_ids(rows, count=512):
    """Random ids plus every edge: first/last row, each side of every shard
    boundary, and the padded tail."""
    gen = torch.Generator().manual_seed(1)
    edges = {0, rows - 1, rows - 2}
    for shard in range(1, -(-rows // SHARD_ROWS)):
        edges |= {shard * SHARD_ROWS - 1, shard * SHARD_ROWS}
    ids = torch.cat(
        [torch.tensor(sorted(edges)), torch.randint(0, rows, (count,), generator=gen)]
    )
    return ids.to(torch.int64)


def _follow(bases, ids, start, end, dtype, shard_rows=SHARD_ROWS):
    """The kernel's addressing on the host: row `i` is at
    `bases[i // shard_rows] + (i % shard_rows) * row_bytes`; another rank's
    rows are zeros."""
    row_bytes = DIM * dtype.itemsize
    out = torch.zeros(ids.numel(), DIM, dtype=dtype)
    for n, i in enumerate(ids.tolist()):
        if start <= i < end:
            address = int(bases[i // shard_rows]) + (i % shard_rows) * row_bytes
            raw = bytearray(ctypes.string_at(address, row_bytes))
            out[n] = torch.frombuffer(raw, dtype=dtype)
    return out


def _bitwise_equal(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.fixture
def fake_pin(monkeypatch):
    """`pin_rows` as copies into fresh host memory; `fail_at` refuses one
    shard (by its first row), as a full page-locked pool would."""
    state = SimpleNamespace(calls=0, fail_at=None, copy=[])

    def pin_rows(sources, copy=None):
        state.calls += 1
        state.copy.append(copy)
        if state.fail_at is not None and state.calls == state.fail_at:
            return None
        return tuple(s.clone() for s in sources)

    monkeypatch.setattr(host_memory, "pin_rows", pin_rows)
    # Copies in plain memory: the device would see them where the host does.
    monkeypatch.setattr(ple_host_table, "device_address", lambda address: address)
    return state


def _load(table, host, start=0, end=TABLE_ROWS, shards=None):
    """What the shard loader hands over: each shard's overlap with the rank."""
    for index, shard in enumerate(_shards(table) if shards is None else shards):
        first = index * SHARD_ROWS
        lo, hi = max(first, start), min(first + shard.shape[0], end)
        if lo < hi:
            host.load_rows(index, lo, shard[lo - first : hi - first])


def _host(start=0, end=TABLE_ROWS, dtype=torch.bfloat16, device="cpu"):
    return PleHostTable(start, end, DIM, dtype, SHARD_ROWS, torch.device(device), "ple")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_shard_addresses_find_every_row(fake_pin, dtype):
    table = _table(dtype)
    host = _host(dtype=dtype)
    _load(table, host)
    bases = host.shard_bases(NUM_SHARDS)
    assert host.on_host
    assert host.pinned_bytes == table.nbytes
    ids = _probe_ids(TABLE_ROWS)
    assert _bitwise_equal(_follow(bases, ids, 0, TABLE_ROWS, dtype), table[ids])


@pytest.mark.parametrize("rank", [0, 1])
def test_tp_rank_holds_and_addresses_only_its_rows(fake_pin, rank):
    # Two ranks split the table mid-shard.
    table = _table(rows=SHARD_ROWS * NUM_SHARDS)
    rows = table.shape[0] // 2
    start, end = rank * rows, (rank + 1) * rows
    host = _host(start, end)
    _load(table, host, start, end)
    bases = host.shard_bases(NUM_SHARDS)
    assert host.pinned_bytes == table[start:end].nbytes
    for shard in range(NUM_SHARDS):
        assert (bases[shard] != 0) == (shard in host.shards)
    ids = _probe_ids(table.shape[0])
    expected = torch.where(
        ((ids >= start) & (ids < end))[:, None], table[ids], torch.zeros(())
    ).to(table.dtype)
    got = _follow(bases, ids, start, end, table.dtype)
    assert _bitwise_equal(got, expected)


@pytest.mark.parametrize("fail_at", [1, 4])
def test_refused_page_lock_moves_the_table_to_device_memory(fake_pin, fail_at):
    fake_pin.fail_at = fail_at
    table = _table()
    host = _host()
    _load(table, host)
    bases = host.shard_bases(NUM_SHARDS)
    assert not host.on_host
    assert host.pinned_bytes == 0
    # Shards after the refusal are not offered for page-locking at all.
    assert fake_pin.calls == fail_at
    ids = _probe_ids(TABLE_ROWS)
    assert _bitwise_equal(_follow(bases, ids, 0, TABLE_ROWS, table.dtype), table[ids])


def test_missing_shard_is_an_error_and_an_empty_load_is_not(fake_pin):
    table = _table()
    host = _host()
    for index, shard in enumerate(_shards(table)):
        if index != 2:
            host.load_rows(index, index * SHARD_ROWS, shard)
    with pytest.raises(ValueError, match=r"shards \[2\] were never loaded"):
        host.shard_bases(NUM_SHARDS)

    dummy = _host()
    bases = dummy.shard_bases(NUM_SHARDS)
    assert not dummy.on_host and fake_pin.calls == NUM_SHARDS - 1
    assert int(bases[0]) == dummy._hbm.data_ptr()


def test_shards_are_copied_even_with_uva_copy_off(fake_pin, monkeypatch):
    monkeypatch.setenv("ATOM_ENGRAM_UVA_COPY", "0")
    _load(_table(), _host())
    assert fake_pin.copy == [True] * NUM_SHARDS


def test_rows_outside_the_rank_are_refused(fake_pin):
    host = _host(0, SHARD_ROWS * 2)
    with pytest.raises(ValueError, match="outside"):
        host.load_rows(2, SHARD_ROWS * 2, torch.zeros(1, DIM, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="outside"):
        host.load_rows(1, SHARD_ROWS, torch.zeros(SHARD_ROWS + 1, DIM))
    assert fake_pin.calls == 0


# ---- GPU: the kernel against the HBM embedding it replaces -----------------

_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="needs a ROCm GPU (UVA reads of page-locked host memory)",
)


def _ops():
    pytest.importorskip("triton")
    from atom.model_ops.qwen4_exp.ops import ple

    return ple


def _checkpoint_shards(table, directory):
    """The shards as the loader gets them: views of a safetensors mapping,
    adjacent and not page-aligned."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    path = directory / "ple.safetensors"
    names = [f"shard_{i}" for i in range(-(-table.shape[0] // SHARD_ROWS))]
    save_file({n: s.contiguous() for n, s in zip(names, _shards(table))}, path)
    with safe_open(path, framework="pt", device="cpu") as f:
        return [f.get_tensor(n) for n in names]


def _device_host(table, mode, monkeypatch, tmp_path, start=0, end=None):
    """`PleHostTable` loaded with the real page-locking in `mode`."""
    end = table.shape[0] if end is None else end
    if mode == "hbm":
        monkeypatch.setattr(host_memory, "pin_rows", lambda sources, copy=None: None)
    host = _host(start, end, table.dtype, "cuda")
    _load(table, host, start, end, _checkpoint_shards(table, tmp_path))
    bases = host.shard_bases(NUM_SHARDS).cuda()
    assert host.on_host == (mode != "hbm")
    return host, bases


@_gpu
@pytest.mark.parametrize("mode", ["copy", "hbm"])
def test_uva_lookup_matches_hbm_embedding_bitwise(monkeypatch, tmp_path, mode):
    ple = _ops()
    table = _table()
    host, bases = _device_host(table, mode, monkeypatch, tmp_path)
    ids = _probe_ids(TABLE_ROWS, 4096).cuda()
    got = ple.sharded_table_lookup(
        ids, bases, None, 0, TABLE_ROWS, SHARD_ROWS, DIM, torch.bfloat16
    )
    expected = torch.nn.functional.embedding(ids, table.cuda())
    assert _bitwise_equal(got, expected)
    del host


@_gpu
def test_fp8_uva_lookup_matches_fp8_embedding_bitwise(monkeypatch, tmp_path):
    ple = _ops()
    table = _table(torch.float8_e4m3fn)
    host, bases = _device_host(table, "copy", monkeypatch, tmp_path)
    scale = torch.tensor([0.1875], device="cuda")
    ids = _probe_ids(TABLE_ROWS, 4096).cuda()
    got = ple.sharded_table_lookup(
        ids, bases, scale, 0, TABLE_ROWS, SHARD_ROWS, DIM, torch.bfloat16
    )
    expected = ple.fp8_embedding_lookup(
        ids, table.cuda(), scale, 0, TABLE_ROWS, torch.bfloat16
    )
    assert _bitwise_equal(got, expected)
    del host


@_gpu
def test_uva_lookup_zeroes_rows_of_other_ranks_and_placeholders(monkeypatch, tmp_path):
    ple = _ops()
    table = _table(rows=SHARD_ROWS * NUM_SHARDS)
    start, end = SHARD_ROWS + 5, 4 * SHARD_ROWS - 3
    host, bases = _device_host(table, "copy", monkeypatch, tmp_path, start, end)
    ids = torch.cat([_probe_ids(table.shape[0], 2048), torch.tensor([-1])]).cuda()
    got = ple.sharded_table_lookup(
        ids, bases, None, start, end, SHARD_ROWS, DIM, torch.bfloat16
    )
    owned = (ids >= start) & (ids < end)
    expected = torch.nn.functional.embedding(ids.clamp(min=0), table.cuda())
    expected = torch.where(owned[:, None], expected, torch.zeros_like(expected))
    assert _bitwise_equal(got, expected)
    del host


@_gpu
def test_uva_lookup_replays_in_a_cuda_graph(monkeypatch, tmp_path):
    ple = _ops()
    table = _table()
    host, bases = _device_host(table, "copy", monkeypatch, tmp_path)
    ids = torch.zeros(256, dtype=torch.int64, device="cuda")

    def lookup():
        return ple.sharded_table_lookup(
            ids, bases, None, 0, TABLE_ROWS, SHARD_ROWS, DIM, torch.bfloat16
        )

    lookup()  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = lookup()
    reference = table.cuda()
    for seed in range(3):
        gen = torch.Generator().manual_seed(seed)
        ids.copy_(torch.randint(0, TABLE_ROWS, (256,), generator=gen))
        graph.replay()
        torch.cuda.synchronize()
        assert _bitwise_equal(out, torch.nn.functional.embedding(ids, reference))
    del graph, host


@_gpu
def test_ngram_embedding_on_host_matches_hbm_embedding(monkeypatch):
    """The model's n-gram embedding: built on the device it goes to the host,
    with ATOM_ENGRAM_UVA=0 it is the HBM `VocabParallelEmbedding`; both
    load the same shards and return the same bytes."""
    pytest.importorskip("aiter.dist.parallel_state", exc_type=ImportError)
    from atom.model_ops import embed_head
    from atom.model_ops.qwen4_exp import ple_layer

    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(embed_head, "get_tp_group", lambda: group)
    monkeypatch.setattr(ple_layer, "get_tp_group", lambda: group)
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        eos_token_id=0,
        vocab_size=4096,
        split_ngram_parts=16,
        ngram_vocab_size_base=997,
        make_ngram_vocab_size_divisible_by=128,
    )

    def build(uva):
        monkeypatch.setenv("ATOM_ENGRAM_UVA", "1" if uva else "0")
        with torch.device("cuda"):
            layer = ple_layer.Qwen4ExpNGramEmbedding(
                config, 16 * 32, 0, 1024, 8, prefix="ple"
            )
        return layer

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        host_layer, hbm_layer = build(True), build(False)
        assert isinstance(host_layer.ngram_embedding, ple_layer._Qwen4ExpHostEmbedding)
        assert isinstance(hbm_layer.ngram_embedding, embed_head.VocabParallelEmbedding)
        rows = host_layer.table_rows
        gen = torch.Generator().manual_seed(2)
        table = torch.randn(rows, host_layer.head_dim, generator=gen).bfloat16()
        for layer in (host_layer, hbm_layer):
            weight = layer.ngram_embedding.weight
            for index, shard in enumerate(table.split(layer.checkpoint_shard_rows)):
                weight.weight_loader(weight, shard, index)
            if hasattr(layer.ngram_embedding, "process_weights_after_loading"):
                layer.ngram_embedding.process_weights_after_loading()
        assert host_layer.ngram_embedding.weight.numel() == 0
        assert host_layer.ngram_embedding.table.on_host
        tokens = torch.randint(0, config.vocab_size, (300,), generator=gen).cuda()
        starts = torch.tensor([0, 7, 120, 300], dtype=torch.int32, device="cuda")
        context = torch.randint(0, config.vocab_size, (3, 2), generator=gen).cuda()
        got = host_layer(tokens, starts, context)
        expected = hbm_layer(tokens, starts, context)
        assert _bitwise_equal(got, expected)
        # Padding rows of the table are reachable by id, and as checkpointed.
        tail = torch.arange(rows - 130, rows, device="cuda")
        assert _bitwise_equal(
            host_layer.ngram_embedding(tail), hbm_layer.ngram_embedding(tail)
        )
    finally:
        torch.set_default_dtype(previous)
