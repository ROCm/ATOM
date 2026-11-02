# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8 PLE n-gram table in page-locked host memory, read over UVA.

The CPU half checks `PleHostTable`'s shard addresses by following them the way
the kernel does; the GPU half checks the kernel against the HBM embedding it
replaces, bit for bit.
"""

import ctypes
import gc
import logging
import threading
import weakref
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


def _weight(dtype=torch.bfloat16):
    """The module's placeholder parameter: `[0, dim]`, on the CPU."""
    return torch.nn.Parameter(torch.empty(0, DIM, dtype=dtype), requires_grad=False)


def _host(start=0, end=TABLE_ROWS, dtype=torch.bfloat16, device="cpu"):
    return PleHostTable(
        start, end, _weight(dtype), SHARD_ROWS, torch.device(device), "ple"
    )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
)
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
    # The table is the module's parameter now, as an HBM embedding's is.
    assert _bitwise_equal(host.weight.data, table)
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
    assert dummy.weight.shape == (TABLE_ROWS, DIM)
    assert int(bases[0]) == dummy.weight.data_ptr()


@pytest.mark.parametrize("fail_at", [None, 2])
def test_only_an_empty_table_is_handed_to_initialize(fake_pin, fail_at):
    """A dummy load's table is filled (`--load_dummy`'s mode) once it exists,
    before its addresses are taken; a loaded one, host or HBM, is the
    checkpoint's and is not."""
    fake_pin.fail_at = fail_at
    filled = []

    def initialize(host):
        assert host.weight.shape == (TABLE_ROWS, DIM)
        host.weight.data.fill_(0.5)
        filled.append(host)

    dummy = _host()
    bases = dummy.shard_bases(NUM_SHARDS, lambda: initialize(dummy))
    assert filled == [dummy]
    ids = _probe_ids(TABLE_ROWS)
    got = _follow(bases, ids, 0, TABLE_ROWS, torch.bfloat16)
    assert torch.equal(got, torch.full_like(got, 0.5))

    table = _table()
    loaded = _host()
    _load(table, loaded)
    bases = loaded.shard_bases(NUM_SHARDS, lambda: initialize(loaded))
    assert filled == [dummy]
    assert _bitwise_equal(_follow(bases, ids, 0, TABLE_ROWS, table.dtype), table[ids])


def test_only_a_refused_page_lock_is_a_warning(fake_pin, caplog):
    """A dummy load has no rows to page-lock: its table belongs in HBM, as
    expected as any other. A refused page-lock is a fallback to flag."""
    caplog.set_level(logging.INFO, logger="atom")
    _host().shard_bases(NUM_SHARDS)
    fake_pin.fail_at = 2
    refused = _host()
    _load(_table(), refused)
    refused.shard_bases(NUM_SHARDS)
    dummy = "ple: no n-gram rows loaded (dummy load); the table is allocated in HBM"
    failed = (
        "ple: page-locking the n-gram table on the host failed; falling back to HBM"
    )
    logged = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if "n-gram" in record.getMessage()
    ]
    assert logged == [(logging.INFO, dummy), (logging.WARNING, failed)]


class _Embedding(torch.nn.Module):
    """`_Qwen4ExpHostEmbedding` as the weight lifecycle sees it: the
    parameter, the table and the addresses the kernel reads."""

    def __init__(self, device="cpu", dtype=torch.bfloat16):
        super().__init__()
        self.weight = _weight(dtype)
        if dtype.itemsize == 1:
            self.weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.register_buffer(
            "shard_bases",
            torch.zeros(NUM_SHARDS, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.table = PleHostTable(
            0, TABLE_ROWS, self.weight, SHARD_ROWS, torch.device(device), "ple"
        )

    def bind(self):
        """What `process_weights_after_loading` does with the addresses."""
        self.shard_bases.copy_(self.table.shard_bases(NUM_SHARDS))

    def release_weight_memory(self):
        self.table.release()

    def resume_weight_memory(self):
        self.table.resume()
        self.bind()

    def sync_weight(self, param_name, tensor, shard_id=None):
        """`_Qwen4ExpHostEmbedding.sync_weight` at TP1, less its shape check."""
        if param_name != "weight":
            return False
        if shard_id is None and not self.table.on_host:
            self.table.check_resumed()
            self.table.check_rows(tensor)
            return False
        self.table.sync_rows(0 if shard_id is None else shard_id * SHARD_ROWS, tensor)
        return True


def _runner(model, device="cpu"):
    from atom.rollout.memory_manager import MemoryManagerMixin

    runner = MemoryManagerMixin()
    runner.model, runner.device, runner.label = model, torch.device(device), "ple"
    runner.enforce_eager = True
    return runner


@pytest.mark.parametrize("fail_at", [None, 1, 4])
def test_releasing_the_weights_releases_the_table(fake_pin, monkeypatch, fail_at):
    """`MemoryManagerMixin` releases `named_parameters()`, and asks the
    module to free the rest: a table that fell back to HBM goes with the
    parameters, page-locked rows are freed by the table. Either comes back
    at a new address on resume, and the kernel's addresses follow it; the
    weight sync that follows a resume fills it."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    fake_pin.fail_at = fail_at
    table = _table()
    model = _Model()
    emb = model.ple.ngram_embedding
    _load(table, emb.table)
    emb.bind()
    runner = _runner(model)
    before = emb.shard_bases.clone()
    pinned = [weakref.ref(p) for _, p in emb.table._rows.values() if p is not None]
    assert len(pinned) == (NUM_SHARDS if fail_at is None else 0)
    pins = fake_pin.calls
    # The released HBM table, one block the heap could hand back as it was,
    # stays held, so the resumed one cannot come back where it was.
    held = emb.weight.data

    runner.release_memory(tags=["weights"])
    assert emb.weight.numel() == 0
    assert emb.table.pinned_bytes == 0
    gc.collect()
    assert all(ref() is None for ref in pinned)
    # Someone else takes the freed memory, so the resumed table cannot come
    # back where it was.
    squatters = [torch.empty_like(table)]
    squatters += [torch.empty_like(shard) for shard in _shards(table)]

    runner.resume_memory(tags=["weights"])
    if fail_at is None:
        assert emb.table.on_host
        assert fake_pin.calls == pins + NUM_SHARDS
        assert emb.table.pinned_bytes == table.nbytes
    else:
        assert fake_pin.calls == pins
        assert emb.weight.shape == table.shape
        assert int(emb.shard_bases[0]) == emb.weight.data_ptr()
    assert not torch.equal(emb.shard_bases, before)
    new = table.neg()
    assert _updater(model).update_weights(_sync_tensors(new, torch.ones(4, 4), True))
    del squatters, held
    ids = _probe_ids(TABLE_ROWS)
    got = _follow(emb.shard_bases, ids, 0, TABLE_ROWS, table.dtype)
    assert _bitwise_equal(got, new[ids])


@pytest.mark.parametrize("shards", [False, True])
@pytest.mark.parametrize("fail_at", [None, 1])
def test_a_sync_between_release_and_resume_is_out_of_order(
    fake_pin, monkeypatch, fail_at, shards
):
    """The rows are not missing from the checkpoint, they are released: the
    error says to sync after resuming, which then works. By the table's own
    name too, which on the HBM fallback is the updater's to sync."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    fake_pin.fail_at = fail_at
    table = _table()
    model = _Model()
    emb = model.ple.ngram_embedding
    _load(table, emb.table)
    emb.bind()
    runner = _runner(model)
    runner.release_memory(tags=["weights"])
    tensors = _sync_tensors(table.neg(), torch.ones(4, 4), shards)
    tensors = tensors[:NUM_SHARDS] if shards else tensors[:1]
    with pytest.raises(ValueError, match="released; sync its rows after the weights"):
        _updater(model).update_weights(tensors)
    runner.resume_memory(tags=["weights"])
    assert _updater(model).update_weights(tensors) == len(tensors)
    ids = _probe_ids(TABLE_ROWS)
    got = _follow(emb.shard_bases, ids, 0, TABLE_ROWS, table.dtype)
    assert _bitwise_equal(got, table.neg()[ids])


@pytest.mark.parametrize("fail_at", [None, 1])
def test_a_released_table_has_no_addresses(fake_pin, monkeypatch, fail_at):
    """Between release and resume there are no rows to address -- on the HBM
    fallback `weight` is released and empty -- so asking raises rather than
    handing the kernel an address that is not the table's."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    fake_pin.fail_at = fail_at
    model = _Model()
    emb = model.ple.ngram_embedding
    _load(_table(), emb.table)
    emb.bind()
    runner = _runner(model)
    runner.release_memory(tags=["weights"])
    with pytest.raises(ValueError, match="released; take its addresses after"):
        emb.table.shard_bases(NUM_SHARDS)
    runner.resume_memory(tags=["weights"])
    emb.table.shard_bases(NUM_SHARDS)


def test_resumed_rows_are_host_memory_whatever_the_default_device(fake_pin):
    """The rows a resume page-locks and the addresses it hands out are host
    tensors even under a device default (`ModelRunner` sets one while it
    builds the model): the page-locking copies with `memmove`."""
    table = _table()
    host = _host()
    _load(table, host)
    host.release()
    torch.set_default_device("meta")
    try:
        host.resume()
        bases = host.shard_bases(NUM_SHARDS)
    finally:
        torch.set_default_device(None)
    assert bases.device.type == "cpu"
    assert all(pinned.device.type == "cpu" for _, pinned in host._rows.values())
    assert host.pinned_bytes == table.nbytes


class _HbmEmbedding(torch.nn.Module):
    """`_Qwen4ExpEmbedding` as the updater sees it: the `weight_loader` it
    inherits from `VocabParallelEmbedding` takes the whole table, and
    `sync_weight` (`_ShardsIntoWeight`) writes one checkpoint shard."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(_table(), requires_grad=False)

    def weight_loader(self, param, loaded_weight):
        param.data.copy_(loaded_weight)

    def sync_weight(self, param_name, tensor, shard_id=None):
        if param_name != "weight" or shard_id is None:
            return False
        self.weight.data.narrow(0, shard_id * SHARD_ROWS, tensor.shape[0]).copy_(tensor)
        return True


def test_shards_reach_a_module_that_also_has_a_whole_table_loader(fake_pin):
    """A packed name is offered to the module's `sync_weight` before its
    `weight_loader`, which would be handed a shard and its index and raise."""
    model = _Model()
    emb = model.ple.ngram_embedding = _HbmEmbedding()
    new = _table().neg()
    proj = torch.randn(4, 4)
    updated = _updater(model).update_weights(_sync_tensors(new, proj, True))
    assert updated == NUM_SHARDS + 1
    assert torch.equal(emb.weight, new)
    assert torch.equal(model.proj.weight, proj)


def test_the_modules_holding_weight_memory_are_found_once(fake_pin, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = _Model()
    _load(_table(), model.ple.ngram_embedding.table)
    model.ple.ngram_embedding.bind()
    walks = []
    modules = model.modules
    model.modules = lambda: walks.append(1) or modules()
    runner = _runner(model)
    for _ in range(2):
        runner.release_memory(tags=["weights"])
        runner.resume_memory(tags=["weights"])
    assert len(walks) == 1
    assert runner._weight_memory_modules() == [model.ple.ngram_embedding]


class _Graphs:
    """A runner's captured decode graphs, as `release_cudagraphs` sees them."""

    def __init__(self):
        self.sizes = [1, 2, 4]

    def keys(self):
        return list(self.sizes)

    def release_graphs(self):
        self.sizes = []


def test_a_resumed_table_at_new_addresses_waits_for_a_recapture(fake_pin, monkeypatch):
    """Releasing the weights drops every captured graph and marks the
    recapture pending before the rows go: no graph holding the old addresses
    survives to be replayed. `resume_memory` recaptures once the weights are
    on the device and the KV pool is back -- neither is, here -- so after the
    table has moved the recapture is still pending. The GPU half recaptures
    for real."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = _Model()
    emb = model.ple.ngram_embedding
    _load(_table(), emb.table)
    emb.bind()
    runner = _runner(model)
    runner.enforce_eager = False
    runner.config = SimpleNamespace(sleep_keeps_memory_resident=False)
    runner.req_graph = _Graphs()
    runner.kv_cache = None
    runner.capture_cudagraph = lambda: pytest.fail("recaptured without a KV pool")
    before = emb.shard_bases.clone()

    runner.release_memory(tags=["weights"])
    assert runner.req_graph.keys() == []
    assert runner._graphs_released_for_sleep
    squatters = [torch.empty_like(shard) for shard in _shards(_table())]
    runner.resume_memory(tags=["weights"])
    assert not torch.equal(emb.shard_bases, before)
    assert runner._graphs_released_for_sleep
    del squatters


class _Model(torch.nn.Module):
    """The PLE table and one ordinary parameter, named as the trainer sends
    them: the table whole, or as the checkpoint's shards."""

    def __init__(self):
        super().__init__()
        self.packed_modules_mapping = {
            f".ngram_embedding.shard_{shard}.": (".ngram_embedding.", shard)
            for shard in range(NUM_SHARDS)
        }
        self.ple = torch.nn.Module()
        self.ple.ngram_embedding = _Embedding()
        self.proj = torch.nn.Linear(4, 4, bias=False)


def _updater(model, device="cpu"):
    from atom.rollout.weight_updater import WeightUpdaterMixin

    class _Updater(WeightUpdaterMixin):
        label, rank, world_size = "ple", 0, 1

        def clear_kv_cache(self):
            pass

    updater = _Updater()
    updater.model, updater.device = model, torch.device(device)
    return updater


def _sync_tensors(table, proj, shards):
    if not shards:
        return [("ple.ngram_embedding.weight", table), ("proj.weight", proj)]
    return [
        (f"ple.ngram_embedding.shard_{index}.weight", shard)
        for index, shard in enumerate(_shards(table))
    ] + [("proj.weight", proj)]


@pytest.mark.parametrize("shards", [False, True])
@pytest.mark.parametrize("fail_at", [None, 1])
def test_a_weight_sync_reaches_the_rows_the_lookup_reads(fake_pin, fail_at, shards):
    """`update_weights` with the table and an ordinary parameter updates
    both. The rows are written in place: nothing is page-locked again and
    the kernel's addresses do not move, so a captured graph reads the new
    rows. A whole HBM fallback table is `weight` and is synced as any other
    parameter; its checkpoint shards, which have no parameter of their own,
    are written into it."""
    fake_pin.fail_at = fail_at
    old, new = _table(), _table().neg()
    model = _Model()
    emb = model.ple.ngram_embedding
    _load(old, emb.table)
    emb.bind()
    before = emb.shard_bases.clone()
    pinned = [p.data_ptr() for _, p in emb.table._rows.values() if p is not None]
    pins = fake_pin.calls
    proj = torch.randn(4, 4)

    hbm = emb.weight.data_ptr()

    updated = _updater(model).update_weights(_sync_tensors(new, proj, shards))

    assert updated == (NUM_SHARDS + 1 if shards else 2)
    assert torch.equal(model.proj.weight, proj)
    assert emb.weight.data_ptr() == hbm
    assert torch.equal(emb.shard_bases, before)
    assert fake_pin.calls == pins
    assert [p.data_ptr() for _, p in emb.table._rows.values() if p is not None] == (
        pinned
    )
    ids = _probe_ids(TABLE_ROWS)
    got = _follow(emb.shard_bases, ids, 0, TABLE_ROWS, new.dtype)
    assert _bitwise_equal(got, new[ids])


@pytest.mark.parametrize("fail_at", [None, 1])
def test_an_update_waits_for_the_lookups_once_before_any_row(
    fake_pin, monkeypatch, fail_at
):
    """The rows are written from the host, after one device-wide wait per
    update for the lookups of the step that ended -- not one per shard."""
    fake_pin.fail_at = fail_at
    model = _Model()
    emb = model.ple.ngram_embedding
    _load(_table(), emb.table)
    emb.bind()
    events = []
    sync_rows = emb.table.sync_rows
    monkeypatch.setattr(
        emb.table, "sync_rows", lambda *a: events.append("row") or sync_rows(*a)
    )
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device=None: events.append(device)
    )
    updater = _updater(model, device="cuda")
    shards = _sync_tensors(_table().neg(), torch.ones(4, 4), True)[:NUM_SHARDS]
    for _ in range(2):
        events.clear()
        assert updater.update_weights(shards) == NUM_SHARDS
        assert events == [torch.device("cuda")] + ["row"] * NUM_SHARDS


@pytest.mark.parametrize("rank", [0, 1])
def test_a_synced_table_writes_only_the_ranks_rows(fake_pin, rank):
    table = _table(rows=SHARD_ROWS * NUM_SHARDS)
    rows = table.shape[0] // 2
    start, end = rank * rows, (rank + 1) * rows
    host = _host(start, end)
    _load(table, host, start, end)
    bases = host.shard_bases(NUM_SHARDS)
    new = table.neg()
    host.sync_rows(0, new)
    # One checkpoint shard straddling the ranks' boundary.
    straddling = rows // SHARD_ROWS
    first = straddling * SHARD_ROWS
    new[first : first + SHARD_ROWS] = 2.0
    host.sync_rows(first, new[first : first + SHARD_ROWS])
    ids = _probe_ids(table.shape[0])
    expected = torch.where(
        ((ids >= start) & (ids < end))[:, None], new[ids], torch.zeros(())
    ).to(table.dtype)
    assert _bitwise_equal(_follow(bases, ids, start, end, table.dtype), expected)


def test_a_sync_the_table_cannot_take_is_an_error(fake_pin):
    table = _table()
    host = _host()
    _load(table, host)
    with pytest.raises(ValueError, match="expected \\[rows, 160\\]"):
        host.sync_rows(0, table[:, :8])
    fp8 = _host(dtype=torch.float8_e4m3fn)
    _load(table.to(torch.float8_e4m3fn), fp8)
    with pytest.raises(ValueError, match="expected torch.float8_e4m3fn"):
        fp8.sync_rows(0, table)
    partial = _host()
    _load(table, partial, shards=_shards(table)[:2])
    with pytest.raises(ValueError, match="never loaded"):
        partial.sync_rows(0, table)
    # Through the updater too: raised, not counted as skipped.
    model = _Model()
    _load(table, model.ple.ngram_embedding.table)
    with pytest.raises(ValueError, match="expected \\[rows, 160\\]"):
        _updater(model).update_weights([("ple.ngram_embedding.weight", table[:, :8])])


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


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
def test_fp8_rows_are_read_in_the_encoding_they_are_stored_in(fake_pin, dtype):
    """The kernel reads an FP8 table's rows as the table's own dtype: OCP
    `e4m3fn` or `e4m3fnuz`, which give the same byte values a factor of two
    apart. The rows the addresses lead to are the table's bytes either way."""
    table = _table(dtype)
    host = _host(dtype=dtype)
    _load(table, host)
    ids = _probe_ids(TABLE_ROWS)
    got = _follow(host.shard_bases(NUM_SHARDS), ids, 0, TABLE_ROWS, dtype)
    assert _bitwise_equal(got, table[ids])
    assert ple_host_table.fp8_rows_fnuz(host.dtype) == (dtype == torch.float8_e4m3fnuz)
    other = (
        torch.float8_e4m3fnuz if dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
    )
    finite = torch.isfinite(got.view(other).float()) & torch.isfinite(got.float())
    assert not torch.equal(got.view(other).float()[finite], got.float()[finite])
    with pytest.raises(ValueError, match="not an FP8 E4M3"):
        ple_host_table.fp8_rows_fnuz(torch.bfloat16)


def test_the_host_table_is_on_by_default_only_where_validated():
    for arch in ("gfx950", "gfx942"):
        assert ple_host_table.validated_arch(arch)
    for arch in ("gfx90a", "gfx1250", "unknown"):
        assert not ple_host_table.validated_arch(arch)


def test_addresses_and_syncs_wait_for_a_load_in_flight(fake_pin):
    """`load_rows` page-locks outside the lock and records the rows under
    it; reading the addresses or syncing rows takes it too."""
    table = _table()
    host = _host()
    shards = _shards(table)
    threads = [
        threading.Thread(target=host.load_rows, args=(i, i * SHARD_ROWS, shard))
        for i, shard in enumerate(shards)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    done = {}
    with host._lock:
        for work in (
            lambda: done.update(bases=host.shard_bases(NUM_SHARDS)),
            lambda: done.update(synced=host.sync_rows(0, table.neg())),
        ):
            thread = threading.Thread(target=work)
            thread.start()
            thread.join(0.2)
            assert thread.is_alive() and not done
            threads.append(thread)
    for thread in threads:
        thread.join()
    ids = _probe_ids(TABLE_ROWS)
    assert set(done) == {"bases", "synced"}
    got = _follow(done["bases"], ids, 0, TABLE_ROWS, table.dtype)
    assert _bitwise_equal(got, table.neg()[ids])


@pytest.mark.parametrize("fail_at", [None, 1])
def test_an_fp8_table_syncs_where_its_rows_are(fake_pin, monkeypatch, fail_at):
    """A same-dtype FP8 sync by the table's own name. On the host `weight`
    is a placeholder: the module writes the rows, ahead of the updater's FP8
    branches, which would copy into the placeholder. An HBM fallback table
    is `weight` and goes through the FP8 branch as any FP8 parameter does.
    BF16 rows raise on either, rather than reach the FP8 requantization."""
    fake_pin.fail_at = fail_at
    dtype = torch.float8_e4m3fn
    model = _Model()
    emb = model.ple.ngram_embedding = _Embedding(dtype=dtype)
    _load(_table(dtype), emb.table)
    emb.bind()
    asked = []
    sync = emb.sync_weight
    monkeypatch.setattr(
        emb, "sync_weight", lambda *a: asked.append(sync(*a)) or asked[-1]
    )
    new = _table().neg().to(dtype)
    updater = _updater(model)
    assert updater.update_weights([("ple.ngram_embedding.weight", new)]) == 1
    assert asked == [fail_at is None]
    with pytest.raises(ValueError, match="expected torch.float8_e4m3fn n-gram rows"):
        updater.update_weights([("ple.ngram_embedding.weight", new.bfloat16())])
    ids = _probe_ids(TABLE_ROWS)
    got = _follow(emb.shard_bases, ids, 0, TABLE_ROWS, dtype)
    assert _bitwise_equal(got, new[ids])


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
        ids, bases, None, 0, TABLE_ROWS, SHARD_ROWS, DIM, torch.bfloat16, table.dtype
    )
    expected = torch.nn.functional.embedding(ids, table.cuda())
    assert _bitwise_equal(got, expected)
    del host


@_gpu
@pytest.mark.parametrize("count", [1, 5, 16, 64, 255, 257, 1024, 5000, 20000])
def test_uva_lookup_of_any_size_matches_hbm_embedding_bitwise(
    monkeypatch, tmp_path, count
):
    """Every rows-per-program the lookup picks from the batch size."""
    ple = _ops()
    table = _table()
    host, bases = _device_host(table, "copy", monkeypatch, tmp_path)
    gen = torch.Generator().manual_seed(count)
    ids = torch.randint(0, TABLE_ROWS, (count,), generator=gen).cuda()
    got = ple.sharded_table_lookup(
        ids, bases, None, 0, TABLE_ROWS, SHARD_ROWS, DIM, torch.bfloat16, table.dtype
    )
    assert _bitwise_equal(got, torch.nn.functional.embedding(ids, table.cuda()))
    del host


@_gpu
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
def test_fp8_uva_lookup_matches_fp8_embedding_bitwise(monkeypatch, tmp_path, dtype):
    """`fp8_embedding_lookup` reads the HBM table as its tensor's dtype; the
    host one is read as the table's, so the two agree in either encoding."""
    ple = _ops()
    table = _table(dtype)
    host, bases = _device_host(table, "copy", monkeypatch, tmp_path)
    scale = torch.tensor([0.1875], device="cuda")
    ids = _probe_ids(TABLE_ROWS, 4096).cuda()
    got = ple.sharded_table_lookup(
        ids, bases, scale, 0, TABLE_ROWS, SHARD_ROWS, DIM, torch.bfloat16, table.dtype
    )
    expected = ple.fp8_embedding_lookup(
        ids, table.cuda(), scale, 0, TABLE_ROWS, torch.bfloat16
    )
    assert _bitwise_equal(got, expected)
    values = table.cuda()[ids].to(torch.bfloat16).float() * scale
    assert _bitwise_equal(got, values.to(torch.bfloat16))
    del host


@_gpu
def test_uva_lookup_zeroes_rows_of_other_ranks_and_placeholders(monkeypatch, tmp_path):
    ple = _ops()
    table = _table(rows=SHARD_ROWS * NUM_SHARDS)
    start, end = SHARD_ROWS + 5, 4 * SHARD_ROWS - 3
    host, bases = _device_host(table, "copy", monkeypatch, tmp_path, start, end)
    ids = torch.cat([_probe_ids(table.shape[0], 2048), torch.tensor([-1])]).cuda()
    got = ple.sharded_table_lookup(
        ids, bases, None, start, end, SHARD_ROWS, DIM, torch.bfloat16, table.dtype
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
            ids,
            bases,
            None,
            0,
            TABLE_ROWS,
            SHARD_ROWS,
            DIM,
            torch.bfloat16,
            table.dtype,
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


_NGRAM_CONFIG = SimpleNamespace(
    ngram_size=3,
    heads_per_ngram=8,
    eos_token_id=0,
    vocab_size=4096,
    split_ngram_parts=16,
    ngram_vocab_size_base=997,
    make_ngram_vocab_size_divisible_by=128,
)


@pytest.fixture
def ngram_layer(monkeypatch):
    """Builds the model's n-gram embedding on the device, BF16, at TP 1: on
    the host path (`uva`) or the HBM `VocabParallelEmbedding`."""
    pytest.importorskip("aiter.dist.parallel_state", exc_type=ImportError)
    from atom.model_ops import embed_head
    from atom.model_ops.qwen4_exp import ple_layer

    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(embed_head, "get_tp_group", lambda: group)
    monkeypatch.setattr(ple_layer, "get_tp_group", lambda: group)
    # The host path whatever GPU runs the test; the arch guard has its own.
    monkeypatch.setattr(ple_layer, "get_gfx_runtime", lambda: "gfx950")
    config = SimpleNamespace(
        load_dummy=None, parallel_config=SimpleNamespace(data_parallel_size_local=1)
    )
    monkeypatch.setattr(ple_layer, "get_current_atom_config", lambda: config)

    def build(uva, config=_NGRAM_CONFIG):
        monkeypatch.setenv("ATOM_ENGRAM_UVA", "1" if uva else "0")
        with torch.device("cuda"):
            return ple_layer.Qwen4ExpNGramEmbedding(
                config, 16 * 32, 0, 1024, 8, prefix="ple"
            )

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield build
    finally:
        torch.set_default_dtype(previous)


def _load_layer(layer, table):
    weight = layer.ngram_embedding.weight
    for index, shard in enumerate(table.split(layer.checkpoint_shard_rows)):
        weight.weight_loader(weight, shard, index)
    if hasattr(layer.ngram_embedding, "process_weights_after_loading"):
        layer.ngram_embedding.process_weights_after_loading()


def _ngram_model(layer):
    """The layer as the updater sees it in the model, shard names mapped."""
    model = torch.nn.Module()
    model.ple = layer
    model.packed_modules_mapping = {
        f".ngram_embedding.shard_{shard}.": (".ngram_embedding.", shard)
        for shard in range(layer.split_ngram_parts)
    }
    return model


def _layer_sync_tensors(layer, table, by_shard):
    if not by_shard:
        return [("ple.ngram_embedding.weight", table)]
    return [
        (f"ple.ngram_embedding.shard_{index}.weight", part)
        for index, part in enumerate(table.split(layer.checkpoint_shard_rows))
    ]


def _layer_table(layer):
    gen = torch.Generator().manual_seed(2)
    return torch.randn(layer.table_rows, layer.head_dim, generator=gen).bfloat16()


def _layer_inputs():
    gen = torch.Generator().manual_seed(3)
    vocab = _NGRAM_CONFIG.vocab_size
    tokens = torch.randint(0, vocab, (300,), generator=gen).cuda()
    starts = torch.tensor([0, 7, 120, 300], dtype=torch.int32, device="cuda")
    context = torch.randint(0, vocab, (3, 2), generator=gen).cuda()
    return tokens, starts, context


@_gpu
def test_the_load_log_counts_a_copy_per_data_parallel_rank(
    ngram_layer, monkeypatch, caplog
):
    """Each DP rank page-locks a whole table of its own: the log says what
    this rank holds and what the node's DP ranks hold together."""
    from atom.model_ops.qwen4_exp import ple_layer

    parallel = SimpleNamespace(data_parallel_size_local=4)
    monkeypatch.setattr(
        ple_layer,
        "get_current_atom_config",
        lambda: SimpleNamespace(load_dummy=None, parallel_config=parallel),
    )
    caplog.set_level(logging.INFO, logger="atom")
    layer = ngram_layer(True)
    table = _layer_table(layer)
    _load_layer(layer, table)
    gib = table.nbytes / 1024**3
    assert (
        f"ple.ngram_embedding: n-gram table in page-locked host memory, "
        f"{gib:.1f} GiB on this "
        f"rank read over UVA, none in HBM; every data-parallel rank page-locks "
        f"its own {gib:.1f} GiB copy, {4 * gib:.1f} GiB for the 4 on this node"
    ) in caplog.messages


@_gpu
def test_ngram_embedding_on_host_matches_hbm_embedding(ngram_layer):
    """The model's n-gram embedding: built on the device it goes to the host,
    with ATOM_ENGRAM_UVA=0 it is the HBM `VocabParallelEmbedding`; both
    load the same shards and return the same bytes."""
    from atom.model_ops import embed_head
    from atom.model_ops.qwen4_exp import ple_layer

    host_layer, hbm_layer = ngram_layer(True), ngram_layer(False)
    assert isinstance(host_layer.ngram_embedding, ple_layer._Qwen4ExpHostEmbedding)
    assert isinstance(hbm_layer.ngram_embedding, embed_head.VocabParallelEmbedding)
    rows = host_layer.table_rows
    table = _layer_table(host_layer)
    for layer in (host_layer, hbm_layer):
        _load_layer(layer, table)
    assert host_layer.ngram_embedding.weight.numel() == 0
    assert host_layer.ngram_embedding.table.on_host
    inputs = _layer_inputs()
    assert _bitwise_equal(host_layer(*inputs), hbm_layer(*inputs))
    # Padding rows of the table are reachable by id, and as checkpointed.
    tail = torch.arange(rows - 130, rows, device="cuda")
    assert _bitwise_equal(
        host_layer.ngram_embedding(tail), hbm_layer.ngram_embedding(tail)
    )


@_gpu
def test_released_weights_free_the_hbm_fallback(ngram_layer, monkeypatch):
    """A refused page-lock leaves the table in HBM as the module's `weight`:
    releasing the weights frees it, and after resuming and syncing the
    weights the lookups read the new allocation."""
    monkeypatch.setattr(host_memory, "pin_rows", lambda sources, copy=None: None)
    host_layer, hbm_layer = ngram_layer(True), ngram_layer(False)
    table = _layer_table(host_layer)
    for layer in (host_layer, hbm_layer):
        _load_layer(layer, table)
    embedding = host_layer.ngram_embedding
    assert not embedding.table.on_host
    table_bytes = embedding.weight.nbytes
    assert table_bytes == table.nbytes
    synced = {n: p.detach().cpu() for n, p in host_layer.named_parameters()}
    inputs = _layer_inputs()
    expected = hbm_layer(*inputs)
    runner = _runner(host_layer, "cuda")
    old_base = int(embedding.shard_bases[0])

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    runner.release_memory(tags=["weights"])
    assert allocated - torch.cuda.memory_allocated() >= table_bytes
    squatter = torch.empty(table_bytes, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="released; take its addresses after"):
        embedding.table.shard_bases(embedding.shard_bases.numel())
    # The whole table is `weight`, which the updater would otherwise find
    # empty and skip as a shape mismatch.
    updater = _updater(host_layer, "cuda")
    with pytest.raises(ValueError, match="released; sync its rows after the weights"):
        updater.update_weights([("ngram_embedding.weight", table)])

    runner.resume_memory(tags=["weights"])
    assert embedding.weight.is_cuda and embedding.weight.shape == table.shape
    for name, param in host_layer.named_parameters():
        param.data.copy_(synced[name])
    assert int(embedding.shard_bases[0]) == embedding.weight.data_ptr() != old_base
    assert _bitwise_equal(host_layer(*inputs), expected)
    del squatter


@_gpu
@pytest.mark.parametrize("shards", [False, True])
def test_weight_sync_reaches_the_rows_a_captured_lookup_reads(ngram_layer, shards):
    """`update_weights` on the host table, whole or as the checkpoint's
    shards: the rows are written in place, the shard addresses do not move,
    and a lookup captured in a CUDA graph before the sync replays onto the
    new rows -- the bytes the HBM embedding returns after the same sync."""
    host_layer, hbm_layer = ngram_layer(True), ngram_layer(False)
    old = _layer_table(host_layer)
    new = torch.randn(old.shape, generator=torch.Generator().manual_seed(4))
    new = new.bfloat16()
    for layer in (host_layer, hbm_layer):
        _load_layer(layer, old)
    embedding = host_layer.ngram_embedding
    assert embedding.table.on_host
    bases = embedding.shard_bases.clone()
    pinned = [p.data_ptr() for _, p in embedding.table._rows.values()]
    ids = torch.zeros(256, dtype=torch.int64, device="cuda")
    embedding(ids)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = embedding(ids)

    def sync(layer):
        tensors = _layer_sync_tensors(layer, new, shards)
        assert _updater(_ngram_model(layer), "cuda").update_weights(tensors) == len(
            tensors
        )

    sync(host_layer)
    sync(hbm_layer)
    assert torch.equal(embedding.shard_bases, bases)
    assert [p.data_ptr() for _, p in embedding.table._rows.values()] == pinned
    reference = hbm_layer.ngram_embedding
    for seed in range(3):
        gen = torch.Generator().manual_seed(seed)
        ids.copy_(torch.randint(0, host_layer.table_rows, (256,), generator=gen))
        graph.replay()
        torch.cuda.synchronize()
        assert _bitwise_equal(out, reference(ids))
        assert not _bitwise_equal(out, torch.nn.functional.embedding(ids, old.cuda()))
    inputs = _layer_inputs()
    assert _bitwise_equal(host_layer(*inputs), hbm_layer(*inputs))
    del graph


@_gpu
@pytest.mark.parametrize("mode", ["zero", "xavier", "empty"])
def test_dummy_load_fills_the_table_as_an_hbm_embedding(ngram_layer, monkeypatch, mode):
    """`--load_dummy`: `load_model` runs `initialize_dummy_weights` before the
    post-load hook, while `weight` is still the `[0, dim]` placeholder. The
    table the hook allocates gets what the HBM `VocabParallelEmbedding.weight`
    gets from the same RNG state; `empty` leaves it uninitialized."""
    from atom.model_loader import loader
    from atom.model_ops.qwen4_exp import ple_layer

    monkeypatch.setattr(
        ple_layer, "get_current_atom_config", lambda: SimpleNamespace(load_dummy=mode)
    )
    calls = []
    initialize = loader.initialize_dummy_weights

    def spy(model, init_mode):
        calls.append(init_mode)
        initialize(model, init_mode)

    monkeypatch.setattr(loader, "initialize_dummy_weights", spy)
    host_layer, hbm_layer = ngram_layer(True), ngram_layer(False)
    embedding = host_layer.ngram_embedding
    if mode != "empty":
        loader.initialize_dummy_weights(host_layer, mode)
    torch.manual_seed(7)
    embedding.process_weights_after_loading()
    assert not embedding.table.on_host
    assert int(embedding.shard_bases[0]) == embedding.weight.data_ptr()
    if mode == "empty":
        assert calls == []
        return
    assert calls == [mode, mode]
    torch.manual_seed(7)
    loader.initialize_dummy_weights(hbm_layer, mode)
    reference = hbm_layer.ngram_embedding.weight
    assert reference.shape == embedding.weight.shape
    assert _bitwise_equal(embedding.weight, reference)
    assert torch.isfinite(embedding.weight).all()
    assert (embedding.weight == 0).all() == (mode == "zero")
    ids = torch.arange(0, host_layer.table_rows, 997, device="cuda")
    assert _bitwise_equal(embedding(ids), reference[ids])


@_gpu
def test_the_host_table_stays_off_where_not_validated(ngram_layer, monkeypatch):
    """gfx90a, for one: the table stays the HBM embedding it was."""
    from atom.model_ops.qwen4_exp import ple_layer

    monkeypatch.setattr(ple_layer, "get_gfx_runtime", lambda: "gfx90a")
    layer = ngram_layer(True)
    assert isinstance(layer.ngram_embedding, ple_layer._Qwen4ExpEmbedding)
    for arch in ple_host_table.VALIDATED_ARCHS:
        monkeypatch.setattr(ple_layer, "get_gfx_runtime", lambda arch=arch: arch)
        layer = ngram_layer(True)
        assert isinstance(layer.ngram_embedding, ple_layer._Qwen4ExpHostEmbedding)


@_gpu
def test_fp8_ngram_embedding_on_host_matches_hbm_embedding(ngram_layer):
    """The FP8 table, loaded through each embedding's `load_shard`."""
    from atom.model_ops.qwen4_exp import ple_layer

    config = SimpleNamespace(**vars(_NGRAM_CONFIG), ple_embedding_dtype="float8_e4m3fn")
    host_layer, hbm_layer = ngram_layer(True, config), ngram_layer(False, config)
    assert isinstance(host_layer.ngram_embedding, ple_layer._Qwen4ExpHostEmbedding)
    assert isinstance(hbm_layer.ngram_embedding, ple_layer._Qwen4ExpFP8Embedding)
    table = _layer_table(host_layer).to(torch.float8_e4m3fn)
    for layer in (host_layer, hbm_layer):
        scale = layer.ngram_embedding.weight_scale
        scale.weight_loader(scale, torch.tensor([0.0625]))
        _load_layer(layer, table)
    assert host_layer.ngram_embedding.table.on_host
    inputs = _layer_inputs()
    assert _bitwise_equal(host_layer(*inputs), hbm_layer(*inputs))


@_gpu
@pytest.mark.parametrize("uva", [True, False])
def test_an_fp8_table_in_hbm_syncs_its_shards_and_refuses_bf16_rows(
    ngram_layer, monkeypatch, uva
):
    """An FP8 table in HBM -- the host table after a refused page-lock, or the
    HBM embedding: shards synced by name land in the table, and BF16 rows for
    the whole table raise as on the host, rather than reach the updater's FP8
    requantization, which knows no PLE scale and leaves the rows as they were
    while counting them updated."""
    monkeypatch.setattr(host_memory, "pin_rows", lambda sources, copy=None: None)
    config = SimpleNamespace(**vars(_NGRAM_CONFIG), ple_embedding_dtype="float8_e4m3fn")
    layer = ngram_layer(uva, config)
    embedding = layer.ngram_embedding
    scale = embedding.weight_scale
    scale.weight_loader(scale, torch.tensor([0.0625]))
    _load_layer(layer, _layer_table(layer).to(torch.float8_e4m3fn))
    assert embedding.weight.is_cuda
    new = _layer_table(layer).neg().to(torch.float8_e4m3fn)
    updater = _updater(_ngram_model(layer), "cuda")
    shards = _layer_sync_tensors(layer, new, True)
    assert updater.update_weights(shards) == len(shards)
    assert _bitwise_equal(embedding.weight.data.cpu(), new)
    with pytest.raises(ValueError, match="expected torch.float8_e4m3fn n-gram rows"):
        updater.update_weights(_layer_sync_tensors(layer, new.bfloat16(), False))
    assert _bitwise_equal(embedding.weight.data.cpu(), new)


@_gpu
def test_released_host_rows_come_back_where_the_recaptured_graph_reads(ngram_layer):
    """Releasing the weights frees the page-locked rows and the graphs that
    captured their addresses. Resuming page-locks them again and refreshes
    the addresses before the recapture; after the weight sync the replay
    reads the synced rows, as the HBM embedding returns them."""
    host_layer, hbm_layer = ngram_layer(True), ngram_layer(False)
    table = _layer_table(host_layer)
    for layer in (host_layer, hbm_layer):
        _load_layer(layer, table)
    embedding = host_layer.ngram_embedding
    pinned_bytes = embedding.table.pinned_bytes
    assert pinned_bytes == table.nbytes
    ids = torch.zeros(256, dtype=torch.int64, device="cuda")
    embedding(ids)  # compile outside the capture
    torch.cuda.synchronize()
    events = []

    class Graphs:
        def __init__(self):
            self.capture()

        def capture(self):
            events.append(("capture", embedding.shard_bases.clone()))
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = embedding(ids)

        def keys(self):
            return [256] if self.graph is not None else []

        def release_graphs(self):
            self.graph = None

    graphs = Graphs()
    runner = _runner(host_layer, "cuda")
    runner.enforce_eager = False
    runner.config = SimpleNamespace(sleep_keeps_memory_resident=False)
    runner.req_graph = graphs
    runner.capture_cudagraph = graphs.capture
    runner.kv_cache = torch.empty(1, device="cuda")
    resume = embedding.resume_weight_memory

    def resumed():
        resume()
        events.append(("resume", embedding.shard_bases.clone()))

    embedding.resume_weight_memory = resumed
    synced = {n: p.detach().clone() for n, p in host_layer.named_parameters()}
    old_bases = embedding.shard_bases.clone()
    sizes = [p.nbytes for _, p in embedding.table._rows.values()]

    runner.release_memory(tags=["weights"])
    assert embedding.table.pinned_bytes == 0 and graphs.graph is None
    assert runner._graphs_released_for_sleep
    squatter = host_memory.pin_rows([torch.empty(n, dtype=torch.uint8) for n in sizes])
    runner.resume_memory(tags=["weights"])
    assert [event for event, _ in events] == ["capture", "resume", "capture"]
    assert torch.equal(events[2][1], embedding.shard_bases)
    assert not torch.equal(embedding.shard_bases, old_bases)
    assert embedding.table.pinned_bytes == pinned_bytes
    for name, param in host_layer.named_parameters():
        param.data.copy_(synced[name])
    new = torch.randn(table.shape, generator=torch.Generator().manual_seed(5))
    parts = new.bfloat16().split(host_layer.checkpoint_shard_rows)
    for index, part in enumerate(parts):
        assert embedding.sync_weight("weight", part, index)
        hbm_layer.ngram_embedding.weight.weight_loader(
            hbm_layer.ngram_embedding.weight, part, index
        )
    reference = hbm_layer.ngram_embedding
    for seed in range(3):
        gen = torch.Generator().manual_seed(seed)
        ids.copy_(torch.randint(0, host_layer.table_rows, (256,), generator=gen))
        graphs.graph.replay()
        torch.cuda.synchronize()
        assert _bitwise_equal(graphs.out, reference(ids))
    del squatter
