# SPDX-License-Identifier: MIT
"""Admission gates, empty ranks and exact production owner accounting."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.models.deepseek_v41.config import normalize_hf_config, validate_runtime_config
from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step


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


def test_lmcache_mp_is_the_only_kv_transfer_admitted():
    validate_runtime_config(
        runtime_config(kv_transfer_config={"kv_connector": "lmcache_mp"})
    )
    for connector in ("lmcache_offload", "mooncake", "moriio", "multi"):
        with pytest.raises(ValueError, match="KV transfer other than lmcache_mp"):
            validate_runtime_config(
                runtime_config(kv_transfer_config={"kv_connector": connector})
            )
    with pytest.raises(ValueError, match="RapidServe"):
        validate_runtime_config(
            runtime_config(
                kv_transfer_config={"kv_connector": "lmcache_mp"},
                enable_rapidserve=True,
            )
        )


def vllm_plugin_config(**overrides):
    """A config the way the vLLM plugin builds one for V4.1.

    `plugin_config.is_vllm` is what distinguishes it from the other plugin
    backends, which still have no V4.1 bridge; `kv_cache_block_size` is 256
    because that is the PAGE size `atom.config` forces for this model and the
    proxy layer's block size on the vLLM side.
    """
    fields = {
        "plugin_config": SimpleNamespace(is_vllm=True),
        "enable_prefix_caching": False,
        "kv_cache_block_size": 256,
    }
    fields.update(overrides)
    return runtime_config(**fields)


def test_vllm_plugin_text_path_is_admitted():
    validate_runtime_config(vllm_plugin_config())


@pytest.mark.parametrize(
    "override",
    [
        # DSpark is admitted (see the test below); every other method still
        # has no driver on this path.
        {
            "speculative_config": SimpleNamespace(
                method="eagle3", num_speculative_tokens=5
            )
        },
        # CSA2 blocks are reusable only at whole-PAGE boundaries after the
        # compressor has run, so vLLM's hash-based reuse would hand back
        # blocks whose STATE side was never replayed.
        {"enable_prefix_caching": True},
    ],
)
def test_vllm_plugin_refuses_what_the_bridge_cannot_drive(override):
    with pytest.raises(ValueError):
        validate_runtime_config(vllm_plugin_config(**override))


def test_vllm_plugin_admits_dspark_speculation():
    """vLLM owns the DSpark draft; the bridge owes only the CSA2 state a
    verification step leaves behind, which it now stages and commits."""
    validate_runtime_config(
        vllm_plugin_config(
            speculative_config=SimpleNamespace(
                method="dspark", num_speculative_tokens=5, model=None
            ),
            # What `--kv-cache-dtype auto` resolves to for this model; the
            # draft's own `fp8_ds_mla` lives in vLLM's speculative config, not
            # here, because the two pools no longer share a CacheConfig.
            kv_cache_dtype="bf16",
            # The native DSpark knobs this gate reads; their defaults are off,
            # and the dynamic-schedule branch below them is a native concern.
            dspark=SimpleNamespace(
                confidence_schedule=None, ragged=False, calibration_profile=None
            ),
        )
    )


def test_other_plugin_backends_are_still_refused_outright():
    with pytest.raises(ValueError, match="plugin mode outside vLLM"):
        validate_runtime_config(
            runtime_config(plugin_config=SimpleNamespace(is_vllm=False))
        )


def test_vllm_plugin_kv_transfer_is_gated_on_its_own_allow_list():
    """The plugin's transport is vLLM's `kv_connector`, not an ATOM name.

    Resolving it through `KVConnectorFactory` is a category error, so the gate
    forks on the mode. This also pins that the gate is *reachable* at all: it
    reads `Config.kv_transfer_config`, which the plugin path populates in
    `atom.plugin.config`. Before that plumbing existed the dict was always
    empty here and every connector was admitted without the question being
    asked -- a check that passes because it was never run.
    """
    validate_runtime_config(vllm_plugin_config(kv_transfer_config=None))
    # The allow-list is empty on this path, so every connector is refused --
    # including `lmcache_mp`, which is admitted natively and means nothing
    # here. A PAGE prefix restored without its CSA2 STATE kills the engine, so
    # there is no partial transport worth admitting.
    for connector in (
        "AtomLMCacheOffloadConnector",
        "LMCacheConnectorV1",
        "lmcache_mp",
        "NixlConnector",
    ):
        with pytest.raises(ValueError, match="KV transfer other than lmcache_mp"):
            validate_runtime_config(
                vllm_plugin_config(kv_transfer_config={"kv_connector": connector})
            )
def _attach_uncompiled_backbone(model):
    """The runtime model's one compiled graph, built bare and run uncompiled,
    for a model assembled without its constructor."""
    from atom.models.deepseek_v41 import runtime

    model.replay = False
    backbone = runtime._Backbone.__new__(runtime._Backbone)
    torch.nn.Module.__init__(backbone)
    backbone.do_not_compile = True
    backbone.__dict__["owner"] = model
    model.backbone = backbone


def test_empty_rank_padding_has_no_cache_writes(monkeypatch):
    PagedAttentionCache, DeepseekV41RuntimeModel = _runtime_pieces()
    from atom.models.deepseek_v41 import runtime

    geo = V41PoolGeometry(2, ((1, 2),), 32, 4, 512, 32)
    cache = PagedAttentionCache(geo, 4, 2, "cpu")
    cache.backing.fill_(57)
    before = cache.backing.clone()
    step = begin_step(cache, [])
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
    _attach_uncompiled_backbone(model)
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
    step = begin_step(cache, [PagedRequest(0, 0, 0, 1, 0, (0,))], plans={})
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
    _attach_uncompiled_backbone(model)
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


@pytest.mark.parametrize("index_dtype", ["bf16", "fp8_e4m3"])
def test_an_index_plane_other_than_fp8_or_fp4_is_refused(index_dtype):
    """The paged scorer reads FP8 or FP4 and nothing else.

    A plane stored otherwise has no reader, which is a load-time refusal and
    not a slower path.
    """
    with pytest.raises(ValueError, match="index plane other than fp8 or fp4"):
        validate_runtime_config(runtime_config(index_cache_dtype=index_dtype))


@pytest.mark.parametrize("index_dtype", ["fp8", "fp4"])
def test_an_fp8_or_fp4_index_plane_is_admitted(index_dtype):
    validate_runtime_config(runtime_config(index_cache_dtype=index_dtype))


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("level", [0, 3])
def test_native_dspark_passes_production_admission(graph, dynamic, tp_size, level):
    from atom.config import CUDAGraphMode, DSparkConfig

    value = runtime_config(
        model="/model",
        tensor_parallel_size=tp_size,
        enable_expert_parallel=False,
        enforce_eager=not graph,
        compilation_config=SimpleNamespace(
            level=level,
            cudagraph_mode=(
                CUDAGraphMode.FULL if level == 3 else CUDAGraphMode.PIECEWISE
            ),
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


@pytest.mark.parametrize("rows", [8, 16, 32])
def test_shortening_the_index_block_moves_bytes_without_adding_any(rows):
    """A block's length is a layout choice, not a capacity one.

    The data and the scales are packed with nothing between them, so the block
    grows and shrinks in step with the rows it holds and a row's share is the
    same number at every length. Choosing 8 to make a candidate list a block
    table therefore costs no pool capacity -- only the granularity at which
    the two regions interleave changes.
    """
    geo = V41PoolGeometry(
        40, ((2, 2), (20, 1)), 64, 128, 512, 128, packed=True, index_block_rows=rows
    )
    assert geo.index_row_bytes == 132
    assert (
        geo.paged_bytes
        == V41PoolGeometry(
            40, ((2, 2), (20, 1)), 64, 128, 512, 128, packed=True
        ).paged_bytes
    )


@pytest.mark.parametrize("rows", [4, 12, 24, 20])
def test_an_index_block_the_scorer_cannot_page_over_is_refused(rows):
    """`pa_mqa_logits` takes whole 16-row MFMA tiles, or exactly 8.

    Nothing downstream would report a block of 12: the writer would interleave
    at one length and the scorer read at another, which is a wrong score and
    not a fault. So the geometry is where it has to be caught.
    """
    with pytest.raises(ValueError, match="whole 16-row MFMA tiles"):
        V41PoolGeometry(40, ((2, 2), (20, 1)), 64, 128, 512, 128, index_block_rows=rows)


def test_a_page_that_does_not_hold_whole_index_blocks_is_refused():
    """The second gate: legal block length, but a PAGE that cannot hold it.

    Ratio 2 halves the PAGE before the count is taken, so the floor is twice
    the block -- which is why this fires on 16 rows at a 16-token PAGE.
    """
    with pytest.raises(ValueError, match="needs whole 16-row blocks"):
        V41PoolGeometry(40, ((2, 2), (20, 1)), 16, 128, 512, 128)
    V41PoolGeometry(40, ((2, 2), (20, 1)), 16, 128, 512, 128, index_block_rows=8)


def test_the_scorer_plane_fits_the_candidate_list_at_short_contexts():
    """The candidate plane is wider than the context one below ~16k tokens.

    `candidate_topk_blocks` comes from the checkpoint and `max_model_len` does
    not bound it, so which of the two planes is widest flips with the context
    length. Sizing from the context alone fits every long-context
    configuration and none of the short ones -- and the failure surfaces as
    "a [8192, 16384] scorer temporary exceeds its workspace", naming the
    scorer rather than the budget it was sized from. This is checkable from a
    geometry alone: no model, no GPU, no vLLM.
    """
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace

    geo = V41PoolGeometry(
        26, ((20, 1),), 256, 128, 512, 128, index_block_rows=8, index_topk=512
    )
    # 8192 tokens of context: 32 PAGEs of 256, so the context plane is
    # 32 * 256 = 8192 wide. The candidate list is 2048 blocks of 8 = 16384.
    columns, candidate_blocks = 32, 2048
    context_width = columns * geo.rows_per_page(1)
    candidate_width = candidate_blocks * geo.index_block_rows
    assert candidate_width > context_width, "fixture no longer exercises the case"

    ws = ScoreWorkspace(geo, 8192, columns, "cpu", candidate_blocks=candidate_blocks)
    # The call the old sizing refused, at the exact shape it refused it in.
    ws.logits(8192, candidate_width)

    # And the context plane still fits, so this widened rather than traded.
    ws.logits(8192, context_width)

    # Negative control: without the candidate width the same call must fail,
    # or this test would pass against the defect it exists to catch.
    narrow = ScoreWorkspace(geo, 8192, columns, "cpu")
    with pytest.raises(ValueError, match="exceeds its workspace"):
        narrow.logits(8192, candidate_width)


def test_the_attention_output_buffer_is_one_row_per_token_not_per_sequence():
    """Sized from every leading dim of `hidden`, not from `shape[0]`.

    ATOM hands this layer `[batch, tokens, dim]`; upstream's native V4.1 gets
    `[tokens, dim]`, and its `_alloc_attn_out(hidden_states.shape[0], ...)`
    is correct only for that shape. Copying the form without the premise
    allocates one row per sequence, and the 518 tests here all passed on it --
    only a kernel's own `out=` check caught it, at server startup.
    """
    import torch

    pytest.importorskip("aiter", reason="the attention module reaches AITER")
    from atom.models.deepseek_v41.attention import Attention

    heads, head_dim = 16, 512
    stub = SimpleNamespace(heads=heads, head_dim=head_dim)

    for shape, tokens in (((8192, 7168), 8192), ((1, 8192, 7168), 8192)):
        hidden = torch.empty(shape, dtype=torch.bfloat16, device="cpu")
        out = Attention._alloc_attn_out(stub, hidden)
        assert out.shape[-2:] == (heads, head_dim)
        # The leading dims are whatever `hidden` had, so the flatten this
        # path performs next yields one row per token under either layout.
        assert (
            out.shape[:-2].numel() == tokens
        ), f"{shape} gave {out.shape[:-2].numel()} rows, expected {tokens}"
