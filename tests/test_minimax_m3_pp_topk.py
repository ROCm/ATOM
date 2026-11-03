# SPDX-License-Identifier: MIT
"""Exercise the shared runner hooks and M3 memo, replacing GPU scoring and page emission."""

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from atom.config import _is_minimax_m3_config
from atom.models.utils import IntermediateTensors
from tests.test_minimax_m3_index_cache_rows import definitions


def reference_pages(
    indices,
    table,
    starts,
    prefix,
    max_q,
    *,
    decode=False,
    pages_per_block=8,
    block_page_stride=8,
):
    heads, rows, topk = indices.shape
    bt = torch.zeros(rows * heads, topk * pages_per_block, dtype=torch.int32)
    ctx = torch.zeros(rows * heads, dtype=torch.int32)
    for req in range(table.shape[0]):
        start = req * max_q if decode else int(starts[req])
        end = start + max_q if decode else int(starts[req + 1])
        base = int(prefix[req]) - max_q if decode else int(prefix[req])
        for row in range(start, end):
            length = base + row - start + 1
            tail = (length - 1) // 128
            for head in range(heads):
                selected = indices[head, row].tolist()
                full = [b for b in selected if 0 <= b < tail]
                chosen = full + ([tail] if tail in selected else [])
                pages = [
                    (int(table[req, b]) * block_page_stride + j) * heads + head
                    for b in chosen
                    for j in range(pages_per_block)
                ]
                out_row = row * heads + head
                bt[out_row, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
                ctx[out_row] = len(full) * 128 + (
                    length - tail * 128 if tail in selected else 0
                )
    return bt, ctx


@pytest.fixture(autouse=True)
def emitter(monkeypatch):
    # Exercise the production allocation/launch wrapper with a CPU kernel oracle.
    mod = ModuleType("atom.model_ops.minimax_m3.index_topk")
    mod.torch = torch
    mod.PAGES_PER_SPARSE_BLOCK = 8
    mod.SPARSE_BLOCK_SIZE = 128
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(
                indices,
                starts,
                prefix,
                topk,
                sh,
                sr,
                sk,
                table,
                bt,
                ctx,
                stride_b,
                stride_out,
                interval,
                block_size,
                **kw,
            ):
                assert (sh, sr, sk) == indices.stride()
                assert stride_b == table.stride(0)
                assert stride_out == bt.stride(0)
                assert interval == 1 and block_size == 128
                assert grid[1:] == (table.shape[0], indices.shape[0])
                assert topk == indices.shape[-1]
                assert kw["NUM_KV_HEADS"] == indices.shape[0]
                assert kw["DECODE_MAX_Q"] in (0, grid[0])
                calls.append((grid, kw))
                expected = reference_pages(
                    indices,
                    table,
                    starts,
                    prefix,
                    grid[0],
                    decode=kw["DECODE_MAX_Q"] > 0,
                    pages_per_block=kw["pages_per_block"],
                    block_page_stride=kw["block_page_stride"],
                )
                bt.copy_(expected[0])
                ctx.copy_(expected[1])

            return launch

    mod._emit_sparse_block_table_kernel = Kernel()
    definitions(
        "atom/model_ops/minimax_m3/index_topk.py",
        {"_alloc_emit", "minimax_m3_emit_sparse_block_table"},
        mod.__dict__,
    )
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return calls


def attention_metadata():
    return SimpleNamespace(
        prefill=SimpleNamespace(
            block_table=torch.arange(32, dtype=torch.int32).reshape(2, 16),
            cu_seqlens_q=torch.tensor([0, 2, 6], dtype=torch.int32),
            context_lens=torch.tensor([641, 896], dtype=torch.int32),
            max_query_len=4,
        ),
        decode=SimpleNamespace(
            block_table=torch.arange(32, dtype=torch.int32).reshape(2, 16),
            seq_lens=torch.tensor([643, 900], dtype=torch.int32),
            max_query_len=3,
        ),
    )


@pytest.fixture
def model_schedule(monkeypatch):
    mod = ModuleType("atom.models.minimax_m3.model")
    mod.PretrainedConfig = object
    definitions(
        "atom/models/minimax_m3/model.py",
        {
            "_sparse_attention_layer_ids",
            "_sparse_attention_layer_ordinals",
            "_should_skip_minimax_m3_index_topk",
        },
        mod.__dict__,
    )
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return mod


def config(**overrides):
    return SimpleNamespace(
        **(
            {
                "model_type": "minimax_m3",
                "num_hidden_layers": 60,
                "use_index_cache": True,
                "index_topk_freq": 4,
                "sparse_attention_config": {"sparse_attention_freq": [1] * 60},
            }
            | overrides
        )
    )


def runner(start, end, cfg, *, mla=False, wrappers=(), dummy=False):
    md = attention_metadata()
    context = SimpleNamespace(is_dummy_run=dummy)
    ns = {
        "get_pp_group": lambda: SimpleNamespace(
            world_size=4, is_first_rank=start == 0, is_last_rank=end == 60
        ),
        "get_forward_context": lambda: SimpleNamespace(
            attn_metadata=SimpleNamespace(sparse_attention_metadata=md),
            context=context,
        ),
        "_is_minimax_m3_config": _is_minimax_m3_config,
        "logger": logging.getLogger(__name__),
    }
    names = {
        "_setup_pp_shared_indexer",
        "_load_pp_shared_indexer",
        "_store_pp_shared_indexer",
    }
    definitions("atom/model_engine/model_runner.py", names, ns, "ModelRunner")
    cls = type("Runner", (), {n: ns[n] for n in names})
    obj = cls()
    obj.config = SimpleNamespace(hf_config=cfg)
    obj.has_mla_indexer = mla
    obj._test_context = context
    obj._test_layer_range = range(start, end)
    obj.model = torch.nn.Module()
    obj.model.start_layer, obj.model.end_layer = start, end
    for name in reversed(wrappers):
        wrapper = torch.nn.Module()
        wrapper.add_module(name, obj.model)
        obj.model = wrapper
    obj.attn_metadata_builder = SimpleNamespace(
        _sparse_kv_indices_gpu=torch.zeros(64, dtype=torch.int32),
        index_topk_out=4,
    )
    obj.rank_name = "test"
    obj._setup_pp_shared_indexer()
    return obj, md


def memo(layer, reuse, layout="nvfp4"):
    ns = {"torch": torch}
    names = {"_topk_cache_state", "_load_cached_topk", "_store_cached_topk"}
    definitions(
        "atom/model_ops/attention_mha.py", names, ns, "SparseMHAPagedAttentionImpl"
    )
    obj = type("Memo", (), {n: ns[n] for n in names})()
    obj.index_topk_cache_state = {}
    obj.skip_index_topk = reuse
    obj.kv_nvfp4 = layout == "nvfp4"
    obj.layer_num = obj.sparse_layer_ordinal = layer
    return obj


def stage(obj, md, cfg, schedule, incoming, layout, mode, seed=0):
    obj._load_pp_shared_indexer(incoming)
    traces = {}
    for layer in sorted(schedule._sparse_attention_layer_ids(cfg)):
        if layer not in obj._test_layer_range:
            continue
        reuse, _ = schedule._should_skip_minimax_m3_index_topk(cfg, layer)
        impl = memo(layer, reuse, layout)
        key = (mode, obj._test_layer_range.start)  # Key belongs to this stage.
        value = impl._load_cached_topk(md, key)
        if value is None:
            assert not reuse, f"layer {layer} missed its incoming top-k"
            ids = torch.arange(36, dtype=torch.int32).reshape(2, 6, 3) + layer + seed
            if layout == "nvfp4":
                value = (ids, None, None)
            else:
                ids = ids % 8
                meta = getattr(md, mode)
                decode = mode == "decode"
                pages = 8
                value = (
                    ids,
                    *reference_pages(
                        ids,
                        meta.block_table,
                        meta.seq_lens if decode else meta.cu_seqlens_q,
                        meta.seq_lens if decode else meta.context_lens,
                        meta.max_query_len,
                        decode=decode,
                        pages_per_block=pages,
                        block_page_stride=pages,
                    ),
                )
            impl._store_cached_topk(md, key, value)
        assert impl._load_cached_topk(md, key) is value if reuse else True
        traces[layer] = value
    out = IntermediateTensors({"hidden_states": torch.zeros(6, 8)})
    obj._store_pp_shared_indexer(out)
    assert set(out.tensors) == (
        {"hidden_states", "sparse_kv_indices"}
        if obj._pp_send_needs_sparse
        else {"hidden_states"}
    )
    return out, traces


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("layout", ["fp8", "nvfp4"])
@pytest.mark.parametrize(
    "partition", [(15, 15, 15, 15), (16, 16, 16, 12), (13, 1, 1, 45)]
)
def test_partition_preserves_all_cached_values(model_schedule, mode, layout, partition):
    cfg = config()
    whole, md = runner(0, 60, cfg)
    _, expected = stage(whole, md, cfg, model_schedule, None, layout, mode)
    start, incoming = 0, None
    for size in partition:
        obj, md = runner(start, start + size, cfg)
        incoming, traces = stage(obj, md, cfg, model_schedule, incoming, layout, mode)
        for layer, values in traces.items():
            for actual, want in zip(values, expected[layer]):
                torch.testing.assert_close(actual, want, rtol=0, atol=0)
        start += size
    assert set(incoming.tensors) == {"hidden_states"}


@pytest.mark.parametrize(
    "wrappers",
    [("model",), ("language_model", "model"), ("model", "language_model", "model")],
)
def test_wrapped_model_transfers_topk_at_each_equal_boundary(model_schedule, wrappers):
    # CausalLM, the M3.1 conditional-generation wrapper, and UBatchWrapper.
    cfg = config()
    incoming = None
    for start in range(0, 60, 15):
        obj, md = runner(start, start + 15, cfg, wrappers=wrappers)
        assert obj._pp_recv_needs_sparse == (start != 0)
        assert obj._pp_send_needs_sparse == (start != 45)
        incoming, _ = stage(obj, md, cfg, model_schedule, incoming, "fp8", "prefill")


def test_dummy_forward_relays_empty_values_then_real_forward_refreshes(model_schedule):
    cfg = config()
    stages = [runner(i, i + 15, cfg, dummy=True) for i in range(0, 60, 15)]
    incoming = None
    for obj, md in stages:
        obj._load_pp_shared_indexer(incoming)
        # Attention deliberately skips dummy forwards, leaving the memo empty.
        incoming = IntermediateTensors({"hidden_states": torch.zeros(6, 8)})
        obj._store_pp_shared_indexer(incoming)
        if obj._pp_send_needs_sparse:
            assert obj._pp_rank_local_keys == ("sparse_kv_indices",)
            assert incoming.tensors["sparse_kv_indices"] is None
    incoming = None
    for obj, md in stages:
        obj._test_context.is_dummy_run = False
        incoming, traces = stage(
            obj, md, cfg, model_schedule, incoming, "fp8", "prefill"
        )
        assert all(value[0] is not None for value in traces.values())


@pytest.mark.parametrize(
    "policy",
    [
        {"index_skip_topk_offset": 2},
        {"index_topk_pattern": "FSSSS" * 12},
        {"index_topk_freq": 1},
        {
            "sparse_attention_config": {
                "sparse_attention_freq": [1, 0, 0, 0, 1] + [0] * 55
            }
        },
    ],
)
def test_existing_schedule_and_dense_only_relay(model_schedule, policy):
    cfg = config(**policy)
    whole, md = runner(0, 60, cfg)
    _, expected = stage(whole, md, cfg, model_schedule, None, "fp8", "prefill")
    incoming = None
    for start, end in [(0, 1), (1, 4), (4, 15), (15, 60)]:
        obj, md = runner(start, end, cfg)
        incoming, traces = stage(
            obj, md, cfg, model_schedule, incoming, "fp8", "prefill"
        )
        for layer, values in traces.items():
            for actual, want in zip(values, expected[layer]):
                torch.testing.assert_close(actual, want)


def test_forward_reset_and_independent_microbatch_memos(model_schedule):
    cfg = config()
    first, first_md = runner(0, 15, cfg)
    left, lmd = runner(15, 30, cfg)
    right, rmd = runner(15, 30, cfg)
    one, _ = stage(
        first, first_md, cfg, model_schedule, None, "nvfp4", "decode", seed=10
    )
    two, _ = stage(
        first, first_md, cfg, model_schedule, None, "nvfp4", "decode", seed=100
    )
    left._load_pp_shared_indexer(one)
    right._load_pp_shared_indexer(two)
    impl = memo(15, True)
    a = impl._load_cached_topk(lmd, ("decode",))[0]
    b = impl._load_cached_topk(rmd, ("decode",))[0]
    torch.testing.assert_close(b - a, torch.full_like(a, 90))
    with pytest.raises(KeyError):
        left._load_pp_shared_indexer(IntermediateTensors({}))
    assert lmd._index_topk_cache_state == {}


def test_glm_existing_buffer_copy_and_physical_row_width():
    cfg = config(
        model_type="glm", indexer_types=["full"] + ["shared"] * 59, index_topk=3
    )
    source, _ = runner(0, 15, cfg, mla=True)
    target, _ = runner(15, 30, cfg, mla=True)
    assert source._pp_rank_local_keys == target._pp_rank_local_keys == ()
    source.attn_metadata_builder._sparse_kv_indices_gpu.copy_(torch.arange(64))
    out = IntermediateTensors({"hidden_states": torch.zeros(6, 8)})
    source._store_pp_shared_indexer(out)
    assert out["sparse_kv_indices"].numel() == 6 * 4  # Not logical topk=3.
    target._load_pp_shared_indexer(out)
    torch.testing.assert_close(
        target.attn_metadata_builder._sparse_kv_indices_gpu[:24],
        torch.arange(24, dtype=torch.int32),
    )
    assert set(out.tensors) == {"hidden_states"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_index_cache": False},
        {"sparse_attention_config": {"sparse_attention_freq": [0] * 60}},
    ],
)
def test_no_index_reuse_needs_no_sparse_metadata(model_schedule, overrides):
    obj, md = runner(15, 30, config(**overrides))
    assert not obj._pp_rank_local_keys
    obj._load_pp_shared_indexer(
        IntermediateTensors({"hidden_states": torch.zeros(6, 8)})
    )
    assert not hasattr(md, "_index_topk_cache_state")


@pytest.mark.parametrize("layout", ["fp8", "nvfp4"])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_import_rebuilds_once_with_local_pages(emitter, layout, mode):
    md = attention_metadata()
    meta = getattr(md, mode)
    meta.block_table.add_(100)
    indices = torch.arange(36, dtype=torch.int32).reshape(2, 6, 3) % 8
    md._index_topk_cache_state = {"pp_topk": indices}
    impl = memo(15, True, layout)
    value = impl._load_cached_topk(md, (mode,))
    assert value[0] is indices
    if layout == "nvfp4":
        assert value[1:] == (None, None)
        assert not emitter
    else:
        decode = mode == "decode"
        pages = 8
        expected = reference_pages(
            indices,
            meta.block_table,
            meta.seq_lens if decode else meta.cu_seqlens_q,
            meta.seq_lens if decode else meta.context_lens,
            meta.max_query_len,
            decode=decode,
            pages_per_block=pages,
            block_page_stride=pages,
        )
        torch.testing.assert_close(value[1:], expected)
        assert len(emitter) == 1
    assert "pp_topk" not in md._index_topk_cache_state
    count = len(emitter)
    assert impl._load_cached_topk(md, (mode,)) is value
    assert memo(16, True, layout)._load_cached_topk(md, (mode,)) is value
    assert len(emitter) == count
