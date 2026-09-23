# SPDX-License-Identifier: MIT
"""Admission gates, empty ranks and exact production owner accounting."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.models.deepseek_v41.config import normalize_hf_config, validate_runtime_config


def _runtime_pieces():
    """The cache and the runtime model, which reach AITER through their ops.

    Everything else in this module is admission and geometry arithmetic, which
    a CPU-only runner can and should still check -- so these two come in here
    rather than at module scope.
    """
    pytest.importorskip("aiter", reason="the paged cache and runtime reach AITER")
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.models.deepseek_v41.runtime import DeepseekV41RuntimeModel

    return PagedAttentionCache, DeepseekV41RuntimeModel


def test_production_geometry_has_only_four_global_owners():
    fixture = Path(__file__).parents[2] / "models/deepseek_v41/fixtures/config.json"
    config = normalize_hf_config(json.loads(fixture.read_text()))
    geo = V41PoolGeometry(
        config.num_hidden_layers,
        tuple(
            (owner, config.compress_ratios[owner])
            for owner in config.kv_source_layer_ids
        ),
        32,
        config.sliding_window,
        config.head_dim,
        config.index_head_dim,
    )
    assert geo.owners == ((2, 2), (8, 2), (14, 2), (20, 1))
    # Every owner has a compressor ring, ratio-1 included: the width is the
    # widest owner's pool window plus speculative slack, so one field serves
    # both ratios rather than one per ratio.
    assert geo.compress_owners == (2, 8, 14, 20)
    assert geo.compress_ring_slots == 2
    # One field per owner: the index rows are a region of their own, bought
    # with the page and addressed by the same block id.
    assert len(geo.page_fields) == 4
    assert geo.page_bytes == 32 * (3 / 2 + 1) * 512 * 2
    # 132 B per index row: 128 of data and one FP32 scale, the preshuffled
    # block divided by the 16 rows it names.
    assert geo.paged_bytes == 32 * (3 / 2 + 1) * (512 * 2 + 132)
    assert sum(f.bytes_per_entry for f in geo.state_fields) <= geo.state_bytes
    assert geo.state_fields[0].layers == 40
    assert all(field.in_checkpoint for field in geo.state_fields)


def runtime_config(**overrides):
    fields = {
        "enforce_eager": True,
        "compilation_config": SimpleNamespace(level=0),
        "speculative_config": None,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "enable_dp_attention": False,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "kv_transfer_config": None,
        "enable_rapidserve": False,
        "plugin_config": None,
        "online_quant_config": None,
        "eplb_enable": False,
        "kv_cache_dtype": "bf16",
        "index_cache_dtype": "fp8",
        "kv_cache_block_size": 16,
        "tensor_parallel_size": 4,
        "enable_expert_parallel": True,
        "hf_config": SimpleNamespace(),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "override",
    [
        {"enforce_eager": False},
        {"compilation_config": SimpleNamespace(level=2)},
        {"speculative_config": SimpleNamespace(method="mtp", num_speculative_tokens=5)},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
        {"parallel_config": SimpleNamespace(data_parallel_size=2)},
        {"enable_dp_attention": True},
        {"enable_tbo": True},
        {"enable_tbo_decode": True},
        {"kv_transfer_config": {"connector": "moriio"}},
        {"enable_rapidserve": True},
        {"plugin_config": object()},
        {"online_quant_config": {}},
        {"eplb_enable": True},
        {"kv_cache_dtype": "fp8"},
        {"index_cache_dtype": "bf16"},
        {"kv_cache_block_size": 3},
    ],
)
def test_unimplemented_modes_fail_before_loading(override):
    validate_runtime_config(runtime_config())
    with pytest.raises(ValueError):
        validate_runtime_config(runtime_config(**override))


def test_empty_rank_padding_has_no_cache_writes(monkeypatch):
    PagedAttentionCache, DeepseekV41RuntimeModel = _runtime_pieces()
    from atom.models.deepseek_v41 import runtime

    geo = V41PoolGeometry(2, ((1, 2),), 32, 4, 512, 32)
    cache = PagedAttentionCache(geo, 4, 2, "cpu")
    cache.backing.fill_(57)
    before = cache.backing.clone()
    step = cache.begin_step([])
    metadata = SimpleNamespace(
        step=step,
        cache=cache,
        next_histories=np.empty((0, 3), dtype=np.int64),
        image_mask=None,
    )
    monkeypatch.setattr(
        runtime, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    model = DeepseekV41RuntimeModel.__new__(DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(model)
    model.do_not_compile = True
    model.config = SimpleNamespace(hidden_size=64, hc_mult=4)
    model.topology = []
    model.layers = torch.nn.ModuleList()
    model.embed = torch.nn.Embedding(16, 64)
    # No layers are constructed: a step with no requests must not reach one.
    output = model(torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32))
    assert output.shape == (8, 64) and output.count_nonzero() == 0
    torch.testing.assert_close(cache.backing, before, rtol=0, atol=0)


def test_a_forward_reads_nothing_the_forward_before_it_selected(monkeypatch):
    """Graph capture runs the model twice over one step.

    Anything a layer fills on a miss is a kernel the recorded pass skips, so
    the graph does not contain it; its replay then reads the capture batch's
    answer while every kernel that did get recorded reads the live step. The
    two disagree by exactly the padding a `has_invalid=False` attention kernel
    dereferences, so this is a fault, not a drift. `tiles` and `indptrs` are
    not in here because no layer fills them -- `begin_step` does, once.
    """
    PagedAttentionCache, DeepseekV41RuntimeModel = _runtime_pieces()
    from atom.models.deepseek_v41 import runtime

    geo = V41PoolGeometry(2, ((1, 2),), 32, 4, 512, 32)
    cache = PagedAttentionCache(geo, 4, 2, "cpu")
    step = cache.begin_step([RequestSpan(0, 0, 0, 1, 0, (0,))], plans={})
    memos = {name: getattr(step, name) for name in ("selected", "candidates")}
    for name, memo in memos.items():
        memo["what the last forward worked out"] = name
    metadata = SimpleNamespace(
        step=step,
        cache=cache,
        engram_embeddings=None,
        image_mask=None,
    )
    monkeypatch.setattr(
        runtime, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    seen = {}
    original_begin = runtime.v41_begin_forward

    def observe_begin(hidden):
        original_begin(hidden)
        seen.update({name: dict(memo) for name, memo in memos.items()})

    monkeypatch.setattr(runtime, "v41_begin_forward", observe_begin)
    model = DeepseekV41RuntimeModel.__new__(DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(model)
    model.do_not_compile = True
    model.config = SimpleNamespace(hidden_size=64, hc_mult=4)
    model.topology = []
    model.layers = torch.nn.ModuleList()
    model.embed = torch.nn.Embedding(16, 64)
    model(torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32))
    assert seen == {name: {} for name in memos}


@pytest.mark.parametrize("cache_dtype", ["bf16", "fp4"])
@pytest.mark.parametrize("graph", [False, True])
def test_supported_cache_and_piecewise_graph_modes(cache_dtype, graph):
    """The main pool takes either format against the one index plane."""
    from atom.config import CUDAGraphMode

    validate_runtime_config(
        runtime_config(
            kv_cache_dtype=cache_dtype,
            index_cache_dtype="fp8",
            enforce_eager=not graph,
            compilation_config=SimpleNamespace(
                level=0, cudagraph_mode=CUDAGraphMode.PIECEWISE
            ),
        )
    )


@pytest.mark.parametrize("index_dtype", ["bf16", "fp4"])
def test_an_index_plane_other_than_fp8_is_refused(index_dtype):
    """The paged scorer is the only one a cache has, and it reads FP8.

    A plane stored otherwise has no reader, which is a load-time refusal and
    not a slower path: the format is the runtime's, not a tuning knob.
    """
    with pytest.raises(ValueError, match="index plane other than fp8"):
        validate_runtime_config(runtime_config(index_cache_dtype=index_dtype))


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_dspark_passes_production_admission(graph, dynamic):
    from atom.config import CUDAGraphMode, DSparkConfig

    value = runtime_config(
        model="/model",
        enforce_eager=not graph,
        compilation_config=SimpleNamespace(
            level=0, cudagraph_mode=CUDAGraphMode.PIECEWISE
        ),
        hf_config=SimpleNamespace(),
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=5,
            model="/model",
            synthetic_acceptance_rates=None,
        ),
        dspark=DSparkConfig(
            confidence_schedule=dynamic,
            ragged=dynamic,
            calibration_profile="profile.json" if dynamic else None,
        ),
    )
    validate_runtime_config(value)
    value.kv_cache_dtype = "fp4"
    with pytest.raises(ValueError, match="BF16"):
        validate_runtime_config(value)


def test_level3_requires_full_graph_or_eager_runtime():
    from atom.config import CUDAGraphMode

    validate_runtime_config(runtime_config(compilation_config=SimpleNamespace(level=3)))
    validate_runtime_config(
        runtime_config(
            enforce_eager=False,
            compilation_config=SimpleNamespace(
                level=3, cudagraph_mode=CUDAGraphMode.FULL
            ),
        )
    )
    with pytest.raises(ValueError, match="level 3 CUDA Graph mode"):
        validate_runtime_config(
            runtime_config(
                enforce_eager=False,
                compilation_config=SimpleNamespace(
                    level=3, cudagraph_mode=CUDAGraphMode.PIECEWISE
                ),
            )
        )
