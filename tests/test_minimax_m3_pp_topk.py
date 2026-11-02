# SPDX-License-Identifier: MIT
"""Exercise real memo/relay methods; replace only GPU scoring and page emit."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from atom.model_ops.minimax_m3.pp_topk import (
    PP_TOPK_INDICES,
    PP_TOPK_METADATA,
    MiniMaxM3PPTopK,
    bind_received_topk,
    transfer_signature,
)
from atom.models.utils import IntermediateTensors
from tests.test_minimax_m3_index_cache_rows import definitions


def schedule(n=60, *, sparse=None, pattern=None, offset=0, freq=4):
    cfg = SimpleNamespace(
        use_index_cache=True,
        index_topk_freq=freq,
        index_topk_pattern=pattern,
        index_skip_topk_offset=offset,
        sparse_attention_config={"sparse_attention_freq": sparse or [1] * n},
    )
    ns = {"PretrainedConfig": object}
    definitions(
        "atom/models/minimax_m3/model.py",
        {
            "_sparse_attention_layer_ids",
            "_sparse_attention_layer_ordinals",
            "_should_skip_minimax_m3_index_topk",
        },
        ns,
    )
    return [
        (i, ordinal, skip)
        for i in sorted(ns["_sparse_attention_layer_ids"](cfg))
        for skip, ordinal in [ns["_should_skip_minimax_m3_index_topk"](cfg, i)]
    ]


def key(mode="prefill", rows=6, *, width=32, max_q=3):
    return (
        mode,
        (rows, 2, 128),
        torch.bfloat16,
        torch.device("cpu"),
        (2, width),
        (width, 1),
        (2,),
        3,
        0,
        1,
        2,
        max_q,
        4096,
        False,
    )


def memo(layer, ordinal, reuse):
    ns = {"torch": torch}
    methods = {"_topk_cache_state", "_load_cached_topk", "_store_cached_topk"}
    definitions(
        "atom/model_ops/attention_mha.py", methods, ns, "SparseMHAPagedAttentionImpl"
    )
    cls = type("Memo", (), {n: ns[n] for n in methods})
    obj = cls()
    obj.index_topk_cache_state = {}
    obj.skip_index_topk = reuse
    obj.layer_num, obj.sparse_layer_ordinal = layer, ordinal
    obj.kv_nvfp4, obj.kv_mxfp4 = True, False
    return obj


def run_stage(start, end, sched, md, incoming, *, mode="prefill", seed=0):
    relay = MiniMaxM3PPTopK(start, end, 60, sched)
    relay.begin(md, incoming)
    traces = {}
    for layer, ordinal, reuse in sched:
        if not start <= layer < end:
            continue
        impl = memo(layer, ordinal, reuse)
        local_key = key(mode, width=32 + start)
        value = impl._load_cached_topk(md, local_key)
        if value is None:
            assert not reuse, f"stage {start} missed reuse at layer {layer}"
            ids = torch.arange(36, dtype=torch.int32).reshape(2, 6, 3) + layer + seed
            # Physical pages intentionally differ by stage and are never sent.
            value = (ids, torch.full((12, 24), start), torch.full((12,), start))
            impl._store_cached_topk(md, local_key, value)
        traces[layer] = value[0].clone()
    out = IntermediateTensors(
        {"hidden_states": torch.zeros(6, 8), "residual": torch.zeros(6, 8)}
    )
    relay.attach(md, out)
    return out, traces


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize(
    "partition", [(15, 15, 15, 15), (16, 16, 16, 12), (13, 1, 1, 45)]
)
def test_pp_partition_preserves_single_stage_selections(mode, partition):
    sched = schedule()
    _, expected = run_stage(0, 60, sched, SimpleNamespace(), None, mode=mode)
    pos, incoming, observed = 0, None, {}
    for size in partition:
        incoming, trace = run_stage(
            pos, pos + size, sched, SimpleNamespace(), incoming, mode=mode
        )
        observed.update(trace)
        pos += size
    assert PP_TOPK_INDICES not in incoming.tensors
    for layer in expected:
        torch.testing.assert_close(observed[layer], expected[layer], rtol=0, atol=0)


@pytest.mark.parametrize(
    "policy", [{"offset": 2}, {"pattern": "FSSSS" * 12}, {"freq": 1}]
)
def test_boundaries_use_model_schedule(policy):
    sched = schedule(**policy)
    _, expected = run_stage(0, 60, sched, SimpleNamespace(), None)
    incoming = None
    for start in (0, 15, 30, 45):
        incoming, trace = run_stage(
            start, start + 15, sched, SimpleNamespace(), incoming
        )
        for layer, ids in trace.items():
            torch.testing.assert_close(ids, expected[layer], rtol=0, atol=0)


def test_dense_only_stage_relays_unchanged_source():
    sched = schedule(sparse=[1, 0, 0, 0, 1] + [0] * 55)
    first, _ = run_stage(0, 1, sched, SimpleNamespace(), None)
    middle, _ = run_stage(1, 4, sched, SimpleNamespace(), first)
    assert middle.tensors[PP_TOPK_METADATA]["source"] == (0, 0)
    last, trace = run_stage(4, 60, sched, SimpleNamespace(), middle)
    assert 4 in trace and PP_TOPK_INDICES not in last.tensors


def test_missing_or_wrong_source_cannot_reuse_previous_forward():
    sched = schedule()
    received, _ = run_stage(0, 15, sched, SimpleNamespace(), None)
    relay = MiniMaxM3PPTopK(15, 30, 60, sched)
    md = SimpleNamespace()
    relay.begin(md, received)
    assert md._index_topk_cache_state
    with pytest.raises(RuntimeError, match="requires top-k from layer 12"):
        relay.begin(md, IntermediateTensors({}))
    assert md._index_topk_cache_state == {}
    received, _ = run_stage(0, 15, sched, SimpleNamespace(), None)
    received.tensors[PP_TOPK_METADATA]["source"] = (8, 8)
    with pytest.raises(RuntimeError, match="computing layer mismatch"):
        relay.begin(md, received)


def test_interleaved_microbatches_keep_their_own_memo():
    sched = schedule()
    first, _ = run_stage(0, 15, sched, SimpleNamespace(), None, seed=10)
    second, _ = run_stage(0, 15, sched, SimpleNamespace(), None, seed=100)
    relay = MiniMaxM3PPTopK(15, 30, 60, sched)
    md1, md2 = SimpleNamespace(), SimpleNamespace()
    relay.begin(md1, first)
    relay.begin(md2, second)
    impl = memo(15, 15, True)
    torch.testing.assert_close(
        impl._load_cached_topk(md2, key())[0] - impl._load_cached_topk(md1, key())[0],
        torch.full((2, 6, 3), 90, dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("mxfp4", [False, True])
def test_received_ids_rebuild_local_pages_and_memoize(monkeypatch, mode, mxfp4):
    emitted = []
    mod = ModuleType("atom.model_ops.minimax_m3.index_topk")

    def emit(ids, table, starts, prefix, **kwargs):
        emitted.append((table, kwargs))
        return table.clone(), prefix.clone()

    mod.emit_sparse_block_table = emit
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    md = SimpleNamespace()
    local_table = torch.tensor([[17, 23], [61, 37]], dtype=torch.int32)
    fields = SimpleNamespace(
        block_table=local_table,
        cu_seqlens_q=torch.tensor([0, 3, 6]),
        context_lens=torch.tensor([128, 256]),
        seq_lens=torch.tensor([131, 259]),
    )
    md.prefill = md.decode = fields
    received, _ = run_stage(0, 15, schedule(), SimpleNamespace(), None, mode=mode)
    relay = MiniMaxM3PPTopK(15, 30, 60, schedule())
    relay.begin(md, received)
    impl = memo(15, 15, True)
    impl.kv_nvfp4, impl.kv_mxfp4 = False, mxfp4
    value = impl._load_cached_topk(md, key(mode, width=2))
    assert emitted[0][0] is local_table
    assert emitted[0][1]["pages_per_block"] == (1 if mxfp4 else 8)
    assert emitted[0][1]["decode"] == (mode == "decode")
    torch.testing.assert_close(value[1], local_table)
    assert impl._load_cached_topk(md, key(mode, width=2)) is value
    assert len(emitted) == 1


def test_wrong_phase_and_shape_rejected():
    entry = {
        "pp_signature": transfer_signature(key()),
        "value": (torch.zeros(2, 6, 3, dtype=torch.int32), None, None),
    }
    with pytest.raises(RuntimeError, match="geometry/policy mismatch"):
        bind_received_topk(entry, key("decode"), None, nvfp4=True, mxfp4=False)
    entry["value"] = (torch.zeros(2, 5, 3, dtype=torch.int32), None, None)
    with pytest.raises(RuntimeError, match="shape"):
        bind_received_topk(entry, key(), None, nvfp4=True, mxfp4=False)


def test_model_forward_imports_before_layers_and_exports_after(monkeypatch):
    ns = {"torch": torch, "IntermediateTensors": IntermediateTensors}
    definitions(
        "atom/models/minimax_m3/model.py", {"forward"}, ns, "MiniMaxM3SparseForCausalLM"
    )
    md = SimpleNamespace()
    ctx = ModuleType("atom.utils.forward_context")
    ctx.get_forward_context = lambda: SimpleNamespace(
        attn_metadata=SimpleNamespace(sparse_attention_metadata=md)
    )
    monkeypatch.setitem(sys.modules, ctx.__name__, ctx)
    first, _ = run_stage(0, 15, schedule(), SimpleNamespace(), None)
    seen = []

    def inner(ids, pos, intermediate, embeds):
        assert PP_TOPK_INDICES not in intermediate.tensors
        shared = memo(15, 15, True)._load_cached_topk(md, key())
        seen.append(shared[0].clone())
        impl = memo(28, 28, False)
        impl._store_cached_topk(
            md, key(), (torch.full((2, 6, 3), 28, dtype=torch.int32), None, None)
        )
        return IntermediateTensors({"hidden_states": torch.zeros(6, 8)})

    model = SimpleNamespace(
        _mono=SimpleNamespace(supports=lambda *a: False),
        _pp_topk=MiniMaxM3PPTopK(15, 30, 60, schedule()),
        embed_before_model=False,
        model=inner,
    )
    out = ns["forward"](model, torch.zeros(6), torch.zeros(6), first)
    assert len(seen) == 1
    assert out.tensors[PP_TOPK_METADATA]["source"] == (28, 28)
    assert torch.all(out.tensors[PP_TOPK_INDICES] == 28)
