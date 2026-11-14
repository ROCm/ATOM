# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8 PLE n-gram table in page-locked host memory, read over UVA.

The CPU half checks `PleHostTable`'s shard addresses by following them the way
the kernel does; the GPU half checks the kernel against the HBM embedding it
replaces, bit for bit.
"""

import ctypes
import gc
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from atom.model_ops.engram import host_memory
from atom.model_ops.qwen4_exp.ple_host_table import HostTableHooks, PleHostTable
from atom.rollout.memory_manager import MemoryManagerMixin
from atom.rollout.weight_updater import WeightUpdaterMixin

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
    """`pin_rows` as copies into fresh host memory, weakly kept in `pinned`;
    `fail_at` refuses one shard (by its first row), as a full page-locked pool
    would."""
    state = SimpleNamespace(calls=0, fail_at=None, copy=[], pinned=[])

    def pin_rows(sources, copy=False):
        state.calls += 1
        state.copy.append(copy)
        if state.fail_at is not None and state.calls == state.fail_at:
            return None
        pinned = tuple(s.clone() for s in sources)
        state.pinned += [weakref.ref(p) for p in pinned]
        return pinned

    monkeypatch.setattr(host_memory, "pin_rows", pin_rows)
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
    assert host.on_host and fake_pin.copy == [True] * NUM_SHARDS
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


def test_a_missing_shard_is_an_error(fake_pin):
    table = _table()
    host = _host()
    for index, shard in enumerate(_shards(table)):
        if index != 2:
            host.load_rows(index, index * SHARD_ROWS, shard)
    with pytest.raises(ValueError, match=r"shards \[2\] were never loaded"):
        host.shard_bases(NUM_SHARDS)


class _Embedding(HostTableHooks, torch.nn.Module):
    """`_Qwen4ExpHostEmbedding` less the kernel, holding rows `[start, end)`;
    its shard loader hands over each shard's overlap, as
    `_embedding_shard_loader` does."""

    def __init__(self, dtype=torch.bfloat16, start=0, end=TABLE_ROWS):
        super().__init__()
        self.prefix, self.num_embeddings = "ple", TABLE_ROWS
        self.weight = _weight(dtype)
        self.weight.weight_loader = self._shard_loader
        if dtype.itemsize == 1:
            self.weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.register_buffer(
            "shard_bases", torch.zeros(NUM_SHARDS, dtype=torch.int64), persistent=False
        )
        self.table = PleHostTable(
            start, end, self.weight, SHARD_ROWS, torch.device("cpu"), "ple"
        )

    def _shard_loader(self, param, shard, index):
        first = index * SHARD_ROWS
        lo = max(first, self.table.vocab_start)
        hi = min(first + shard.shape[0], self.table.vocab_end)
        if lo < hi:
            self.load_shard(index, lo, shard[lo - first : hi - first])

    def bind(self):
        """What `process_weights_after_loading` does with the addresses."""
        self.shard_bases.copy_(self.table.shard_bases(NUM_SHARDS))


def _runner(model, device="cpu"):
    runner = MemoryManagerMixin()
    runner.model, runner.device, runner.label = model, torch.device(device), "ple"
    runner.enforce_eager = True
    return runner


@pytest.mark.parametrize("fail_at", [None, 1, 4])
def test_releasing_the_weights_releases_the_table(fake_pin, monkeypatch, fail_at):
    """A table that fell back to HBM goes with the parameters, page-locked rows
    with the table; either comes back at a new address that the kernel's
    addresses follow."""
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
    pinned = list(fake_pin.pinned)
    assert sum(ref() is not None for ref in pinned) == (
        NUM_SHARDS if fail_at is None else 0
    )
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


@pytest.mark.parametrize("sleep_again", [False, True])
@pytest.mark.parametrize("shard", [0, 1])
@pytest.mark.parametrize("failure", ["rows", "fallback"])
def test_a_resume_that_runs_out_of_memory_can_be_retried(
    fake_pin, monkeypatch, failure, shard, sleep_again
):
    """A resume that fails at a shard -- its rows cannot be allocated, or
    page-locking is refused and the table does not fit on the device either
    -- keeps the shards it has not restored: the table stays released, a
    release frees the shards already back, and once memory is back a retry
    restores them and the sync after it is read where the addresses point."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    table = _table()
    model = _Model()
    emb = model.ple.ngram_embedding
    _load(table, emb.table)
    emb.bind()
    runner = _runner(model)
    runner.release_memory(tags=["weights"])

    empty, allocations = torch.empty, 0

    def scarce(*size, **kwargs):
        nonlocal allocations
        if len(size) == 2 and size[1] == DIM:
            allocations += 1
            if allocations == shard + 1 + (failure == "fallback"):
                raise RuntimeError("out of memory")
        return empty(*size, **kwargs)

    if failure == "fallback":
        fake_pin.fail_at = fake_pin.calls + shard + 1
    monkeypatch.setattr(torch, "empty", scarce)
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.resume_memory(tags=["weights"])
    monkeypatch.setattr(torch, "empty", empty)
    fake_pin.fail_at = None
    with pytest.raises(ValueError, match="released; take its addresses after"):
        emb.table.shard_bases(NUM_SHARDS)
    if sleep_again:
        runner.release_memory(tags=["weights"])
        assert emb.table.pinned_bytes == 0

    runner.resume_memory(tags=["weights"])
    assert emb.table.on_host
    assert emb.table.pinned_bytes == table.nbytes
    new = table.neg()
    assert _updater(model).update_weights(_sync_tensors(new, torch.ones(4, 4), True))
    ids = _probe_ids(TABLE_ROWS)
    got = _follow(emb.shard_bases, ids, 0, TABLE_ROWS, table.dtype)
    assert _bitwise_equal(got, new[ids])


@pytest.mark.parametrize("shards", [False, True])
@pytest.mark.parametrize("fail_at", [None, 1])
def test_a_sync_between_release_and_resume_is_out_of_order(
    fake_pin, monkeypatch, fail_at, shards
):
    """Between release and resume the rows are released, not missing: a sync,
    or taking the addresses, says to wait for the resume."""
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
    with pytest.raises(ValueError, match="released; take its addresses after"):
        emb.table.shard_bases(NUM_SHARDS)
    runner.resume_memory(tags=["weights"])
    assert _updater(model).update_weights(tensors) == len(tensors)
    ids = _probe_ids(TABLE_ROWS)
    got = _follow(emb.shard_bases, ids, 0, TABLE_ROWS, table.dtype)
    assert _bitwise_equal(got, table.neg()[ids])


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
    resumed = [ref() for ref in fake_pin.pinned[NUM_SHARDS:]]
    assert len(resumed) == NUM_SHARDS
    assert all(rows.device.type == "cpu" for rows in resumed)
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
    class _Updater(WeightUpdaterMixin):
        label, rank, world_size = "ple", 0, 1

        def clear_kv_cache(self):
            pass

    updater = _Updater()
    updater.model, updater.device = model, torch.device(device)
    updater._init_weight_updater()
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
    """The rows are written in place: nothing is page-locked again and the
    addresses do not move. A whole HBM fallback table is synced as `weight`."""
    fake_pin.fail_at = fail_at
    old, new = _table(), _table().neg()
    model = _Model()
    emb = model.ple.ngram_embedding
    _load(old, emb.table)
    emb.bind()
    before = emb.shard_bases.clone()
    pins = fake_pin.calls
    proj = torch.randn(4, 4)

    updated = _updater(model).update_weights(_sync_tensors(new, proj, shards))

    assert updated == (NUM_SHARDS + 1 if shards else 2)
    assert torch.equal(model.proj.weight, proj)
    assert torch.equal(emb.shard_bases, before)
    assert torch.equal(emb.table.shard_bases(NUM_SHARDS), before)
    assert fake_pin.calls == pins
    ids = _probe_ids(TABLE_ROWS)
    got = _follow(emb.shard_bases, ids, 0, TABLE_ROWS, new.dtype)
    assert _bitwise_equal(got, new[ids])


@pytest.mark.parametrize("fail_at", [None, 1])
def test_an_update_waits_for_the_lookups_once_before_any_row(
    fake_pin, monkeypatch, fail_at
):
    """The rows are written from the host, after one device-wide wait per
    update for the lookups of the step that ended -- not one per shard. A
    sync's completion may wait again, for its writes, once the rows are in."""
    fake_pin.fail_at = fail_at
    old, new = _table(), _table().neg()
    model = _Model()
    emb = model.ple.ngram_embedding
    _load(old, emb.table)
    emb.bind()
    ids = _probe_ids(TABLE_ROWS)

    def rows():
        return _follow(emb.shard_bases, ids, 0, TABLE_ROWS, old.dtype)

    waits = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device=None: waits.append((device, rows()))
    )
    updater = _updater(model, device="cuda")
    for before, after in ((old, new), (new, old)):
        waits.clear()
        shards = _sync_tensors(after, torch.ones(4, 4), True)[:NUM_SHARDS]
        assert updater.update_weights(shards) == NUM_SHARDS
        assert waits and {device for device, _ in waits} == {torch.device("cuda")}
        assert _bitwise_equal(waits[0][1], before[ids])
        assert all(_bitwise_equal(seen, after[ids]) for _, seen in waits[1:])
        assert _bitwise_equal(rows(), after[ids])


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
    partial = _host()
    _load(table, partial, shards=_shards(table)[:2])
    with pytest.raises(ValueError, match="never loaded"):
        partial.sync_rows(0, table)
    # Through the updater too: raised, not counted as skipped.
    model = _Model()
    _load(table, model.ple.ngram_embedding.table)
    updater = _updater(model)
    with pytest.raises(ValueError, match="expected \\[rows, 160\\]"):
        updater.update_weights([("ple.ngram_embedding.weight", table[:, :8])])
    with pytest.raises(ValueError, match=f"expected {TABLE_ROWS} n-gram rows"):
        updater.update_weights([("ple.ngram_embedding.weight", table[:8])])


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


@pytest.mark.parametrize("start", [0, 100])
@pytest.mark.parametrize("fail_at", [None, 1])
def test_an_fp8_table_syncs_where_its_rows_are(fake_pin, fail_at, start):
    """The whole FP8 table by its own name, on the host or in HBM, at TP1 or
    as the second TP rank: the module writes this rank's rows, ahead of the
    updater's FP8 branch (which would copy the whole table into them). BF16
    rows raise."""
    fake_pin.fail_at = fail_at
    dtype = torch.float8_e4m3fn
    model = _Model()
    emb = model.ple.ngram_embedding = _Embedding(dtype, start)
    _load(_table(dtype), emb.table, start)
    emb.bind()
    new = _table().neg().to(dtype)
    updater = _updater(model)
    assert updater.update_weights([("ple.ngram_embedding.weight", new)]) == 1
    with pytest.raises(ValueError, match="expected torch.float8_e4m3fn n-gram rows"):
        updater.update_weights([("ple.ngram_embedding.weight", new.bfloat16())])
    ids = torch.arange(start, TABLE_ROWS)
    got = _follow(emb.shard_bases, ids, start, TABLE_ROWS, dtype)
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
@pytest.mark.parametrize("count", [1, 5, 16, 64, 255, 257, 1024, 5000, 20000])
@pytest.mark.parametrize("mode", ["copy", "hbm"])
def test_uva_lookup_of_any_size_matches_hbm_embedding_bitwise(
    monkeypatch, tmp_path, mode, count
):
    """Every rows-per-program the lookup picks from the batch size."""
    ple = _ops()
    table = _table()
    host, bases = _device_host(table, mode, monkeypatch, tmp_path)
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
    the host path (`uva`) or the HBM `_Qwen4ExpEmbedding`."""
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
@pytest.mark.parametrize("fp8", [False, True])
def test_ngram_embedding_on_host_matches_hbm_embedding(ngram_layer, fp8):
    """Built on the device the table goes to the host; with ATOM_ENGRAM_UVA=0
    it is the HBM embedding. Both load the same shards and return the same
    bytes, padding rows included."""
    from atom.model_ops.qwen4_exp import ple_layer

    config, hbm_cls = _NGRAM_CONFIG, ple_layer._Qwen4ExpEmbedding
    if fp8:
        config = SimpleNamespace(**vars(config), ple_embedding_dtype="float8_e4m3fn")
        hbm_cls = ple_layer._Qwen4ExpFP8Embedding
    host_layer, hbm_layer = ngram_layer(True, config), ngram_layer(False, config)
    assert isinstance(host_layer.ngram_embedding, ple_layer._Qwen4ExpHostEmbedding)
    assert isinstance(hbm_layer.ngram_embedding, hbm_cls)
    table = _layer_table(host_layer)
    if fp8:
        table = table.to(torch.float8_e4m3fn)
    for layer in (host_layer, hbm_layer):
        if fp8:
            scale = layer.ngram_embedding.weight_scale
            scale.weight_loader(scale, torch.tensor([0.0625]))
        _load_layer(layer, table)
    embedding = host_layer.ngram_embedding
    assert embedding.weight.numel() == 0 and embedding.table.on_host
    inputs = _layer_inputs()
    assert _bitwise_equal(host_layer(*inputs), hbm_layer(*inputs))
    tail = torch.arange(
        host_layer.table_rows - 130, host_layer.table_rows, device="cuda"
    )
    assert _bitwise_equal(embedding(tail), hbm_layer.ngram_embedding(tail))


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
    """A lookup captured before the sync replays onto the new rows."""
    host_layer, hbm_layer = ngram_layer(True), ngram_layer(False)
    old = _layer_table(host_layer)
    new = torch.randn(old.shape, generator=torch.Generator().manual_seed(4))
    new = new.bfloat16()
    for layer in (host_layer, hbm_layer):
        _load_layer(layer, old)
    embedding = host_layer.ngram_embedding
    assert embedding.table.on_host
    bases = embedding.shard_bases.clone()
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
    assert torch.equal(embedding.table.shard_bases(bases.numel()), bases.cpu())
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
def test_the_host_table_stays_off_where_not_validated_under_a_plugin_or_for_a_dummy_load(
    ngram_layer, monkeypatch
):
    """gfx90a, for one, a framework plugin and `--load_dummy`: the table stays
    the HBM embedding."""
    from atom.model_ops.qwen4_exp import ple_layer

    monkeypatch.setattr(ple_layer, "get_gfx_runtime", lambda: "gfx90a")
    layer = ngram_layer(True)
    assert isinstance(layer.ngram_embedding, ple_layer._Qwen4ExpEmbedding)
    for arch in ple_layer.VALIDATED_ARCHS:
        monkeypatch.setattr(ple_layer, "get_gfx_runtime", lambda arch=arch: arch)
        layer = ngram_layer(True)
        assert isinstance(layer.ngram_embedding, ple_layer._Qwen4ExpHostEmbedding)
    monkeypatch.setattr(ple_layer, "is_plugin_mode", lambda: True)
    layer = ngram_layer(True)
    assert isinstance(layer.ngram_embedding, ple_layer._Qwen4ExpEmbedding)
    monkeypatch.setattr(ple_layer, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(
        ple_layer, "get_current_atom_config", lambda: SimpleNamespace(load_dummy="zero")
    )
    layer = ngram_layer(True)
    assert isinstance(layer.ngram_embedding, ple_layer._Qwen4ExpEmbedding)


@_gpu
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("uva", [True, False])
def test_an_fp8_table_in_hbm_syncs_the_ranks_rows_and_refuses_bf16_rows(
    ngram_layer, monkeypatch, uva, tp
):
    """Shards and the whole table land in the rank's rows; BF16 rows for the
    whole table raise rather than reach the updater's FP8 requantization,
    which knows no PLE scale."""
    from atom.model_ops.qwen4_exp import ple_layer

    group = SimpleNamespace(world_size=tp, rank_in_group=tp - 1)
    monkeypatch.setattr(ple_layer, "get_tp_group", lambda: group)
    monkeypatch.setattr(host_memory, "pin_rows", lambda sources, copy=None: None)
    config = SimpleNamespace(**vars(_NGRAM_CONFIG), ple_embedding_dtype="float8_e4m3fn")
    layer = ngram_layer(uva, config)
    embedding = layer.ngram_embedding
    scale = embedding.weight_scale
    scale.weight_loader(scale, torch.tensor([0.0625]))
    old = _layer_table(layer).to(torch.float8_e4m3fn)
    _load_layer(layer, old)
    assert embedding.weight.is_cuda
    rows = slice(embedding.vocab_start_idx, embedding.vocab_end_idx)
    new = _layer_table(layer).neg().to(torch.float8_e4m3fn)
    updater = _updater(_ngram_model(layer), "cuda")
    for by_shard, table in ((True, new), (False, old)):
        tensors = _layer_sync_tensors(layer, table, by_shard)
        assert updater.update_weights(tensors) == len(tensors)
        assert _bitwise_equal(embedding.weight.data.cpu(), table[rows])
    with pytest.raises(ValueError, match="expected torch.float8_e4m3fn n-gram rows"):
        updater.update_weights(_layer_sync_tensors(layer, new.bfloat16(), False))
    assert _bitwise_equal(embedding.weight.data.cpu(), old[rows])


@_gpu
def test_released_host_rows_come_back_where_the_recaptured_graph_reads(ngram_layer):
    """Resuming refreshes the addresses before the recapture, and the replay
    reads the synced rows."""
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
    sizes = [part.nbytes for part in table.split(host_layer.checkpoint_shard_rows)]

    runner.release_memory(tags=["weights"])
    assert embedding.table.pinned_bytes == 0 and graphs.graph is None
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
