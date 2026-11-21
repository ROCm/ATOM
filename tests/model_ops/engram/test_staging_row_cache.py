# SPDX-License-Identifier: MIT
"""`EngramStaging.start` over quantized tables with row caches: under graph
capture a decode-sized width takes both layers' lookups in one go, eagerly (or
a width too large for that) they go one by one, and either way each layer's
buffer ends up holding exactly what the plain gather returns -- eagerly and
under CUDA-graph replay with changing row ids. The hashing is replaced by the
ids themselves: this pins the staging, not the hash (`test_overlap`)."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    # Before the imports, which reach Triton through the gather kernels.
    pytest.skip("needs a GPU for the UVA gather", allow_module_level=True)

import atom.model_ops.engram.device.staging as staging_mod
from atom.model_ops.engram.device.uva import (
    UvaMissStaging,
    UvaRowCache,
    uva_gather_into,
)
from tests.model_ops.engram.test_uva_row_cache import (
    BLOCK,
    DIM,
    HEADS,
    _ids,
    _same_set_rows,
    _Table,
)
from tests.model_ops.engram.test_uva_row_cache_pair import _OtherTable

LAYERS = (1, 14)
MAX_TOKENS = 512


class _Staging(staging_mod.EngramStaging):
    """No compression, snapshot or cursor: the row ids are given."""

    def _snapshot(self, width):
        pass

    def _advance_cursor(self, width):
        pass


def _build(max_tokens=MAX_TOKENS):
    tables = {LAYERS[0]: _Table(), LAYERS[1]: _OtherTable()}
    device = torch.device("cuda")
    host = SimpleNamespace(
        device=device,
        layer_ids=list(LAYERS),
        max_num_tokens=max_tokens,
        local_heads=HEADS,
        head_start=0,
        total_heads=HEADS,
        embed_width=HEADS * DIM,
        _tp_group=None,
        prefetcher=SimpleNamespace(_tables=tables),
        buffers={
            layer: SimpleNamespace(
                gpu=torch.zeros(
                    max_tokens, HEADS * DIM, dtype=torch.bfloat16, device=device
                )
            )
            for layer in LAYERS
        },
    )
    miss = UvaMissStaging(max_tokens * HEADS, DIM, BLOCK, device)
    uva = SimpleNamespace(
        host=host,
        hash_tables=SimpleNamespace(ngram=3),
        row_ids=torch.empty(max_tokens, HEADS, dtype=torch.int64, device=device),
        row_cache={
            layer: UvaRowCache(DIM, BLOCK, miss, device, slots_log2=9)
            for layer in LAYERS
        },
    )
    return host, uva


@pytest.fixture
def given_ids(monkeypatch):
    ids = {
        layer: torch.full((MAX_TOKENS, HEADS), -1, dtype=torch.int64, device="cuda")
        for layer in LAYERS
    }
    hashed = []

    def fake_indices(tables, layer, snapshot, out):
        hashed.append(layer)
        return out.copy_(ids[layer][: out.shape[0]])

    monkeypatch.setattr(staging_mod, "engram_snapshot_indices", fake_indices)
    pairs = []
    real_pair = staging_mod.uva_gather_pair_into

    def recorded_pair(*args, **kwargs):
        pairs.append(real_pair(*args, **kwargs))
        return pairs[-1]

    monkeypatch.setattr(staging_mod, "uva_gather_pair_into", recorded_pair)
    return ids, hashed, pairs


def _reference(host, ids, width):
    refs = {}
    for layer in LAYERS:
        out = torch.empty(width, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        uva_gather_into(
            host.prefetcher._tables[layer],
            ids[layer][:width],
            out,
            head_start=0,
            local_heads=HEADS,
            total_heads=HEADS,
        )
        refs[layer] = out.view(width, HEADS * DIM)
    return refs


def _check(host, ids, width):
    torch.cuda.synchronize()
    for layer, ref in _reference(host, ids, width).items():
        got = host.buffers[layer].gpu[:width]
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), layer


@pytest.mark.parametrize("width,paired", [(5, False), (160, False), (300, False)])
def test_start_eager(given_ids, width, paired):
    ids, hashed, pairs = given_ids
    host, uva = _build()
    staging = _Staging(uva)
    assert not staging.flat  # every layer's gather writes its own buffer
    gen = torch.Generator().manual_seed(width)
    hot = _same_set_rows(uva.row_cache[LAYERS[0]].sets_log2)
    staging.step = SimpleNamespace(served_by_round=False)
    for step in range(8):
        for layer in LAYERS:
            ids[layer][:width].copy_(_ids(gen, width, hot if step % 2 else None))
        hashed.clear()
        staging.start(width)
        staging.join()
        assert hashed == list(LAYERS)
        _check(host, ids, width)
        clocks = [int(uva.row_cache[layer].clock) for layer in LAYERS]
        assert clocks == [2 + step] * 2
    assert pairs == ([True] * 8 if paired else [])


def test_start_graph_replay(given_ids):
    ids, _, pairs = given_ids
    width = 96
    host, uva = _build()
    staging = _Staging(uva)
    staging.step = SimpleNamespace(served_by_round=False)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        staging.start(width)
        for layer in LAYERS:
            torch.cuda.current_stream().wait_event(staging.done[layer])
        staging.join()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        staging.start(width)
        for layer in LAYERS:
            torch.cuda.current_stream().wait_event(staging.done[layer])
        staging.join()
    gen = torch.Generator().manual_seed(7)
    hot = _same_set_rows(uva.row_cache[LAYERS[0]].sets_log2)
    for step in range(30):
        for layer in LAYERS:
            ids[layer][:width].copy_(_ids(gen, width, hot if step % 3 else None))
        graph.replay()
        _check(host, ids, width)
    miss = uva.row_cache[LAYERS[0]].staging
    assert not bool(miss.count.any())
    assert pairs and all(pairs)
