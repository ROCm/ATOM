# SPDX-License-Identifier: MIT
"""MiniMax-M3.1's NVFP4 KV pool and index pages, and their startup refusals.

An NVFP4 KV page is one buffer [K side | V side] of ``2 * Hkv * 9216`` B (e2m1 data and
e4m3 scales per 16, no per-tensor scale in the page); an index page is
[128 x 64 B data][128 x 8 B scales] = 9216 B. Pinned here: the bytes per token the
capacity arithmetic rests on, the views the kernels are handed, the combinations that
must fail at startup and not in a forward.
"""

import types

import pytest
import torch

from atom.config import DCPConfig, _check_fp4_kv, resolve_draft_kv_dtype
from atom.kv_transfer.disaggregation.sharded_transfer import consumer_head_slices
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.minimax_m3.nvfp4_pages import (
    HEAD_DATA_BYTES,
    HEAD_SCALE_BYTES,
    INDEX_PAGE_BYTES,
    Nvfp4KvLayout,
    head_regions,
    index_page_views,
    kv_page_bytes,
)
from tests.test_minimax_m3_mxfp4_cache import (
    OFFLOAD,
    _dspark_spec,
    _eagle3_spec,
    _m3,
    check,
    no_synthetic_acceptance_import,  # noqa: F401 -- autouse: keeps triton out
    sparse_hf,
)


def pool(layers=3, heads=1, **kw):
    return MhaKvPool(
        layers=layers,
        block_size=128,
        num_kv_heads=heads,
        head_dim=128,
        kv_dtype=torch.uint8,
        layout=Nvfp4KvLayout(),
        **kw,
    )


def test_page_geometry():
    assert INDEX_PAGE_BYTES == 9216 == 128 * (64 + 8)
    assert kv_page_bytes(1) == 18432 and kv_page_bytes(4) == 4 * 18432
    # side regions of 2 heads: K data, K scales, V data, V scales, in this order
    assert head_regions(0, 0, 2) == (0, 2 * 8192)
    assert head_regions(0, 1, 2) == (8192, 2 * 8192 + 1024)
    assert head_regions(1, 1, 2) == (2 * 9216 + 8192, 2 * 9216 + 2 * 8192 + 1024)


def test_a_token_costs_144_bytes_of_k_and_v():
    p = pool(layers=60, heads=1)
    assert p.entry_bytes == 60 * kv_page_bytes(1)
    assert p.entry_bytes / (60 * 128) == 144
    assert p.scale_fields == []


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_the_view_is_one_page_of_both_sides_and_aliases_the_pool(heads):
    p = pool(layers=2, heads=heads)
    p.allocate(5, "cpu")
    k, v = p.kv_views(1)
    assert k.shape == (5, kv_page_bytes(heads)) and k.dtype == torch.uint8
    assert k.stride() == (kv_page_bytes(heads), 1) and v.data_ptr() == k.data_ptr()
    k.fill_(7)
    assert not (p.kv_views(0)[0] == 7).any()


def test_index_page_views_split_data_from_scale():
    pages = (torch.arange(6 * INDEX_PAGE_BYTES) % 251).to(torch.uint8)
    pages = pages.view(6, INDEX_PAGE_BYTES)
    data, scale = index_page_views(pages[1:5])
    assert data.shape == (4, 128, 64) and scale.shape == (4, 128, 8)
    assert data.stride() == (INDEX_PAGE_BYTES, 64, 1)
    assert scale.stride() == (INDEX_PAGE_BYTES, 8, 1)
    assert torch.equal(data[0].flatten(), pages[1, :8192])
    assert torch.equal(scale[3].flatten(), pages[4, 8192:])
    with pytest.raises(ValueError):
        index_page_views(pages[:, :8704])


@pytest.mark.parametrize(
    "kw", [{"blocks_per_entry": 2}, {"num_kv_heads": 1, "head_dim": 64}]
)
def test_only_the_128_page_of_head_dim_128_is_a_pool(kw):
    base = {
        "layers": 1,
        "block_size": 128,
        "num_kv_heads": 1,
        "head_dim": 128,
        "kv_dtype": torch.uint8,
        "layout": Nvfp4KvLayout(),
    }
    with pytest.raises(ValueError):
        MhaKvPool(**{**base, **kw})


NV = {"kv_cache_dtype": "nvfp4", "index_cache_dtype": "nvfp4"}


def test_nvfp4_kv_and_index_pass():
    check(**NV)


@pytest.mark.parametrize("kv", ["fp8", "bf16"])
def test_nvfp4_index_needs_nvfp4_kv(kv):
    with pytest.raises(ValueError, match="needs --kv-cache-dtype nvfp4"):
        check(kv_cache_dtype=kv, index_cache_dtype="nvfp4")


@pytest.mark.parametrize(
    "kw, msg",
    [
        (
            {"kv_cache_dtype": "nvfp4", "index_cache_dtype": "bf16"},
            "needs --index-cache-dtype nvfp4, got bf16",
        ),
        (
            {"kv_cache_dtype": "nvfp4", "index_cache_dtype": "fp4"},
            "needs --index-cache-dtype nvfp4, got fp4",
        ),
        (
            {"kv_cache_dtype": "fp4", "index_cache_dtype": "nvfp4"},
            "needs --index-cache-dtype fp4, got nvfp4",
        ),
    ],
)
def test_nvfp4_and_mxfp4_do_not_mix(kw, msg):
    with pytest.raises(ValueError, match=msg):
        check(**kw)


def test_the_shared_refusals_hold_for_nvfp4():
    with pytest.raises(ValueError, match="bridges"):
        check(plugin_mode=True, **NV)
    with pytest.raises(ValueError, match="128 tokens"):
        check(kv_cache_block_size=16, **NV)
    with pytest.raises(ValueError, match="indexed sparse"):
        check(hf=sparse_hf(freq=[1, 1, 0, 1]), **NV)
    with pytest.raises(ValueError, match="draft would have FP4 KV"):
        check(
            speculative_config=types.SimpleNamespace(
                method="eagle3", draft_kv_dtype="nvfp4"
            ),
            **NV,
        )
    with pytest.raises(ValueError, match="gfx950, got gfx942"):
        check(arch="gfx942", **NV)


def test_nvfp4_kv_is_for_minimax_m3_only():
    _check_fp4_kv("nvfp4", is_m3=True, is_v41=False)
    with pytest.raises(ValueError, match="MiniMax-M3 only"):
        _check_fp4_kv("nvfp4", is_m3=False, is_v41=True)


def test_a_config_with_nvfp4_kv_defaults_the_index_to_nvfp4(monkeypatch):
    cfg = _m3(monkeypatch, kv_cache_dtype="nvfp4", hf=sparse_hf())
    assert cfg.index_cache_dtype == "nvfp4"


@pytest.mark.parametrize("kv,index", [("nvfp4", "nvfp4"), ("nvfp4", None)])
def test_nvfp4_caches_keep_the_indexer_cp_flag(monkeypatch, caplog, kv, index):
    """The NVFP4 index scores its CP shard with aiter's NVFP4 scorer, as the MXFP4
    index does with its own, so the flag stays on."""
    with caplog.at_level("WARNING", logger="atom"):
        cfg = _m3(
            monkeypatch,
            kv_cache_dtype=kv,
            index_cache_dtype=index,
            tensor_parallel_size=4,
            hf={**sparse_hf(), "num_key_value_heads": 4},
            dcp_config=DCPConfig(indexer_dcp_only=True),
        )
    assert cfg.dcp_config.indexer_dcp_only is True
    assert "indexer_dcp_only disabled" not in caplog.text


def test_nvfp4_index_is_refused_off_minimax_m3(monkeypatch):
    with pytest.raises(ValueError, match="MiniMax-M3 only"):
        _m3(
            monkeypatch,
            architectures=["LlamaForCausalLM"],
            kv_cache_dtype="bf16",
            index_cache_dtype="nvfp4",
        )


def test_the_own_pool_draft_of_an_nvfp4_target_is_bf16_a_shared_one_keeps_nvfp4():
    assert resolve_draft_kv_dtype("nvfp4", own_pool=True) == "bf16"
    assert resolve_draft_kv_dtype("nvfp4", own_pool=False) == "nvfp4"


def test_nvfp4_index_cache_is_one_9216_byte_uint8_page():
    mod = pytest.importorskip("atom.model_ops.attentions.aiter_attention")

    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    b.block_size = 128
    b._has_sparse_attention = True
    b.row_counts = lambda: {mod.INDEX_ROWS: 3}
    cfg = types.SimpleNamespace(
        index_cache_dtype="nvfp4",
        hf_config=types.SimpleNamespace(
            sparse_attention_config={"sparse_index_dim": 128}
        ),
    )
    b.model_runner = types.SimpleNamespace(config=cfg)
    assert b._index_cache_spec(128) == (
        3,
        (INDEX_PAGE_BYTES,),
        torch.uint8,
        (-1, INDEX_PAGE_BYTES),
    )
    with pytest.raises(ValueError, match="NVFP4 index cache is 128-wide"):
        b._index_cache_spec(256)


NV_WIDTHS = (8192, 1024, 8192, 1024)


@pytest.mark.parametrize("heads", [1, 4])
def test_transfer_pages_declare_the_four_runs_of_a_page(heads):
    p = pool(layers=2, heads=heads)
    p.allocate(3, "cpu")
    pages = p.transfer_pages("mha", "mha.x", [10, 11], [16, 16])
    assert [pg.head_major_runs for pg in pages] == [(heads, NV_WIDTHS)] * 2
    assert pages[0].region.unit_bytes == kv_page_bytes(heads)
    # each run's heads tile the page in order: K data, K scales, V data, V scales
    assert sum(heads * w for w in NV_WIDTHS) == kv_page_bytes(heads)
    assert head_regions(1, 0, heads)[0] == heads * (8192 + 1024)


def _kv_page(heads):
    p = pool(layers=1, heads=heads)
    p.allocate(1, "cpu")
    return p.transfer_pages("mha", "mha.x", [10])[0]


@pytest.mark.parametrize("rank", range(4))
def test_a_tp4_rank_cuts_its_head_of_each_run_of_a_tp1_page(rank):
    # each row is that head's K / V data or scales, where head_regions puts them
    src, dst = _kv_page(4), _kv_page(1)
    cuts = consumer_head_slices(
        src.region.unit_bytes,
        dst.region.unit_bytes,
        src.head_major_runs,
        dst.head_major_runs,
        rank,
        4,
        "kv",
    )
    sizes = (HEAD_DATA_BYTES, HEAD_SCALE_BYTES)
    assert cuts.tolist() == [
        [head_regions(side, rank, 4)[k], head_regions(side, 0, 1)[k], sizes[k]]
        for side in (0, 1)
        for k in (0, 1)
    ]


MORIIO = {"kv_connector": "moriio", "kv_role": "kv_both"}
PD = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
STORE = {"kv_connector": "mooncake_store", "kv_role": "offload"}
LMCACHE_BY_ANY_NAME = [
    OFFLOAD,
    {"kv_connector": "LMCacheConnectorV1", "kv_role": "offload"},
    {"kv_connector": "multi", "connectors": [OFFLOAD]},
]


@pytest.mark.parametrize("kvt", LMCACHE_BY_ANY_NAME)
def test_lmcache_offload_takes_nvfp4_by_any_name(kvt):
    check(kv_transfer_config=kvt, **NV)


@pytest.mark.parametrize(
    "kvt",
    [
        PD,
        {**PD, "kv_role": "kv_consumer"},
        STORE,
        {"kv_connector": "multi", "connectors": [PD, STORE]},
    ],
)
def test_nvfp4_pages_run_under_mooncake_pd_and_store(kvt):
    check(kv_transfer_config=kvt, **NV)


@pytest.mark.parametrize(
    "kvt",
    [
        MORIIO,
        {"kv_connector": "multi", "connectors": [MORIIO, OFFLOAD]},
        # every leaf, not any: P/D beside an offload tier that moves no NVFP4 page
        {
            "kv_connector": "multi",
            "connectors": [PD, {"kv_connector": "lmcache_mp", "kv_role": "offload"}],
        },
    ],
)
def test_one_transport_that_moves_no_nvfp4_pages_refuses_nvfp4(kvt):
    with pytest.raises(
        ValueError, match="only under lmcache_offload, mooncake, mooncake_store$"
    ):
        check(kv_transfer_config=kvt, **NV)


def test_an_unknown_connector_keeps_the_factory_error(monkeypatch):
    with pytest.raises(ValueError, match="unknown KV connector 'x'"):
        check(kv_transfer_config={"kv_connector": "x"}, **NV)
    # with a DSpark draft of its own dtype too: refused before the draft check
    spec = _dspark_spec()
    spec.kv_cache_dtype = "fp8"
    with pytest.raises(ValueError, match="unknown KV connector 'x'"):
        _m3(
            monkeypatch,
            hf=sparse_hf(),
            kv_transfer_config={"kv_connector": "x"},
            speculative_config=spec,
            **NV,
        )


@pytest.mark.parametrize("kvt", [OFFLOAD, PD, STORE])
def test_mxfp4_still_refuses_every_transport(kvt):
    with pytest.raises(ValueError, match="fp4: it does not run with KV offload"):
        check(kv_transfer_config=kvt)


DRAFT_DTYPE_TRANSPORTS = [
    *LMCACHE_BY_ANY_NAME,
    PD,
    {**PD, "kv_role": "kv_consumer"},
    STORE,
    {"kv_connector": "multi", "connectors": [PD, STORE]},
    {"kv_connector": "multi", "connectors": [PD, OFFLOAD]},
]


@pytest.mark.parametrize("kvt", DRAFT_DTYPE_TRANSPORTS)
@pytest.mark.parametrize("asked, draft", [("fp8", "fp8"), (None, "bf16")])
def test_nvfp4_takes_a_dspark_draft_of_its_own_dtype_under_each_declaring_transport(
    monkeypatch, kvt, asked, draft
):
    spec = _dspark_spec()
    spec.kv_cache_dtype = asked
    config = _m3(
        monkeypatch,
        kv_cache_dtype="nvfp4",
        index_cache_dtype="nvfp4",
        hf=sparse_hf(),
        kv_transfer_config=kvt,
        speculative_config=spec,
    )
    assert config.speculative_config.draft_kv_dtype == draft


@pytest.mark.parametrize("kvt", [OFFLOAD, PD, STORE])
@pytest.mark.parametrize("asked", ["fp8", None])
def test_every_transport_still_refuses_an_eagle3_draft_dtype_beside_nvfp4(
    monkeypatch, kvt, asked
):
    spec = _eagle3_spec()
    spec.kv_cache_dtype = asked
    with pytest.raises(ValueError, match="differs.*except a MiniMax-M3 DSpark draft"):
        _m3(
            monkeypatch,
            kv_cache_dtype="nvfp4",
            index_cache_dtype="nvfp4",
            hf=sparse_hf(),
            kv_transfer_config=kvt,
            speculative_config=spec,
        )


def test_moving_nvfp4_pages_does_not_imply_moving_a_draft_dtype(monkeypatch):
    # The two declarations are separate: one without the other refuses the draft.
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    monkeypatch.setitem(KVConnectorFactory._moves_own_dtype_draft, "mooncake", False)
    nvfp4 = {"kv_cache_dtype": "nvfp4", "index_cache_dtype": "nvfp4", "hf": sparse_hf()}
    for kvt in (PD, {"kv_connector": "multi", "connectors": [PD, STORE]}):
        # still declared for NVFP4 pages: only the draft flag was taken away
        topology = types.SimpleNamespace(kv_transfer_config=kvt)
        assert KVConnectorFactory.topology_moves_nvfp4_pages(topology)
        spec = _dspark_spec()
        spec.kv_cache_dtype = "fp8"
        with pytest.raises(
            ValueError,
            match="draft KV dtype fp8 differs.*under lmcache_offload, mooncake_store",
        ):
            _m3(monkeypatch, kv_transfer_config=kvt, speculative_config=spec, **nvfp4)


def test_the_store_namespace_hashes_the_draft_kv_dtype(monkeypatch):
    # What `moves_own_dtype_draft` on lmcache_offload / mooncake_store rests on.
    from atom.kv_transfer.offload.config import build_page_namespace

    chunk = types.SimpleNamespace(chunk_size=256)

    def built(asked):
        spec = _dspark_spec()
        spec.kv_cache_dtype = asked
        return _m3(
            monkeypatch,
            hf=sparse_hf(),
            kv_transfer_config=STORE,
            speculative_config=spec,
            **NV,
        )

    fp8, bf16 = built("fp8"), built(None)
    assert bf16.speculative_config.draft_kv_dtype == "bf16"
    assert build_page_namespace(fp8, chunk, 1) != build_page_namespace(bf16, chunk, 1)


LMCACHE_MP = {"kv_connector": "lmcache_mp", "kv_role": "offload"}


@pytest.mark.parametrize(
    "kvt",
    [
        MORIIO,
        # every leaf, not any: declaring transports beside one that does not declare
        {"kv_connector": "multi", "connectors": [PD, LMCACHE_MP]},
        {
            "kv_connector": "multi",
            "connectors": [{**MORIIO, "kv_role": "kv_producer"}, STORE],
        },
    ],
)
def test_one_transport_that_moves_no_own_dtype_draft_refuses_it(monkeypatch, kvt):
    spec = _dspark_spec()
    spec.kv_cache_dtype = "fp8"
    with pytest.raises(
        ValueError,
        match="draft KV dtype fp8 differs.*under lmcache_offload, mooncake, mooncake_store",
    ):
        _m3(
            monkeypatch,
            kv_cache_dtype="nvfp4",
            index_cache_dtype="nvfp4",
            hf=sparse_hf(),
            kv_transfer_config=kvt,
            speculative_config=spec,
        )
