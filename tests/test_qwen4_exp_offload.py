# SPDX-License-Identifier: MIT

"""Qwen3.8 offload: QSA pages are the checkpoint destination, PLE rides along."""

from types import SimpleNamespace

import pytest
import torch

from atom.config import KVCacheTensor
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.model_ops.attentions.pool_layout.entry_arena import (
    EntryField,
    LayerMajorArena,
    entry_bytes_for,
    published_bytes_for,
)
from atom.model_ops.attentions.pool_layout.layer_major_page_geometry import (
    layer_major_page_unit_regions,
    layer_major_page_unit_views,
)


def _fields():
    block = 4
    return [
        EntryField("k", 2, (block, 2, 8), torch.bfloat16),
        EntryField("v", 2, (block, 2, 8), torch.bfloat16),
        EntryField("index_raw", 2, (block, 1, 4), torch.bfloat16),
        EntryField("index_compressed", 2, (block // 4, 1, 4), torch.bfloat16),
    ]


def test_qsa_regions_are_one_contiguous_block_each():
    arena = LayerMajorArena(_fields(), entries=3, device="cpu")
    published = published_bytes_for(_fields())
    assert published <= entry_bytes_for(_fields())
    bases, sizes = layer_major_page_unit_regions([arena])
    assert int(sizes.sum()) == published
    assert len(bases) == 8  # 4 fields * 2 layers
    for _, tensor in arena.region_tensors():
        assert tensor.is_contiguous()
        assert tensor.shape[0] == 3


def test_one_qsa_block_round_trips_through_page_views():
    arena = LayerMajorArena(_fields(), entries=2, device="cpu")
    blob = torch.arange(published_bytes_for(_fields()), dtype=torch.uint8)
    at = 0
    for _, tensor in arena.region_tensors():
        row = tensor[1].view(torch.uint8).reshape(-1)
        row.copy_(blob[at : at + row.numel()])
        at += row.numel()
    views = layer_major_page_unit_views([arena], [1])
    packed = torch.cat([view.view(torch.uint8).reshape(-1) for view in views])
    assert torch.equal(packed, blob)


def test_dense_codec_keeps_qsa_side_caches_and_ignores_absent_ones():
    blocks = 4
    k = torch.zeros(blocks, 8, dtype=torch.bfloat16)
    v = torch.zeros(blocks, 8, dtype=torch.bfloat16)
    raw = torch.zeros(blocks, 4, dtype=torch.bfloat16)
    comp = torch.zeros(blocks, 2, dtype=torch.bfloat16)
    qsa = KVCacheTensor(
        layer_num=0,
        k_cache=k,
        v_cache=v,
        index_raw=raw,
        index_compressed=comp,
    )
    plain = KVCacheTensor(layer_num=1, k_cache=k, v_cache=v)
    qsa_codec = DenseKVByteCodec({"qsa": qsa}, num_blocks=blocks)
    plain_codec = DenseKVByteCodec({"mha": plain}, num_blocks=blocks)
    assert qsa_codec.bytes_per_block == (8 + 8 + 4 + 2) * 2
    assert plain_codec.bytes_per_block == (8 + 8) * 2


def _builder():
    pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    builder = object.__new__(Qwen4ExpMetadataBuilder)
    builder.num_spec = 1
    builder.ngram_context_len = 2
    builder.eos_token_id = 7
    builder._ssm_storage = lambda: "kv"
    builder._qsa_fields = lambda: [SimpleNamespace(name=n) for n in ("k", "v")]
    builder.num_state_layers = lambda: 2
    builder._state_shape_for_runner = lambda: ((3, 4), (2, 2))
    builder._state_dtypes = lambda: (torch.bfloat16, torch.float32)
    builder.model_runner = SimpleNamespace(
        config=SimpleNamespace(
            torch_dtype=torch.bfloat16,
            pipeline_parallel_size=1,
            enable_rapidserve=False,
            kv_cache_dtype="bf16",
            kv_cache_block_size=64,
            enable_dp_attention=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_tbo=False,
            enable_tbo_decode=False,
            speculative_config=SimpleNamespace(method="mtp"),
            kv_transfer_config={
                "kv_connector": "lmcache_offload",
                "kv_role": "offload",
            },
            hf_config=SimpleNamespace(
                model_type="qwen4_exp_text",
                num_attention_heads=2,
                ple_layer_ids=[2],
                ngram_size=3,
                ple_conv_kernel_size=4,
                hidden_size=8,
                hc_count=2,
                indexer_compress_ratio=4,
            ),
        )
    )
    return builder, Qwen4ExpMetadataBuilder


def test_ple_bytes_join_the_gdn_image():
    builder, cls = _builder()
    planes = cls._checkpoint_plane_shapes(builder)
    assert [layers for _, layers in planes] == [2, 2, 1, 1]
    # conv (3, 4) bf16 + ssm (2, 2) fp32, two layers; PLE conv 10*16 bf16, ngram 3 int64.
    assert sum(nbytes * layers for nbytes, layers in planes) == 80 + 320 + 24
    builder.model_runner.mamba_k_cache = torch.zeros(2, 4, 3, 4, dtype=torch.bfloat16)
    builder.model_runner.mamba_v_cache = torch.zeros(2, 4, 2, 2, dtype=torch.float32)
    builder.model_runner.ple_conv_state = torch.zeros(4, 16, 10, dtype=torch.bfloat16)
    builder.model_runner.ple_ngram_state = torch.zeros(4, 3, dtype=torch.int64)
    views = cls.state_entry_views(builder, 1)
    assert sum(view.numel() * view.element_size() for view in views) == 80 + 320 + 24


def test_layout_id_names_ple_and_forks_under_pipeline_parallel(monkeypatch):
    builder, cls = _builder()
    monkeypatch.setattr(
        "atom.model_ops.attentions.gdn_attn.get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    layout = cls.state_transfer(builder).paged_layout_id
    assert layout.startswith("gdn-paged-state-v1:")
    assert layout.endswith(":ple_order=conv,ngram")
    # `allocate_per_req_cache` lays a PLE conv slot out (channels, state_len):
    # hc_count * hidden_size = 16 channels, (4 - 1) * 3 + num_spec = 10 taps.
    assert ":ple_conv=(16,10),torch.bfloat16:" in layout
    assert ":ple_ngram=(3,),torch.int64:" in layout
    builder.model_runner.config.pipeline_parallel_size = 2
    assert not cls._uses_paged_checkpoints(builder)
    assert not cls.state_transfer(builder).copies


def _layout_id(builder, cls, monkeypatch):
    monkeypatch.setattr(
        "atom.model_ops.attentions.gdn_attn.get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    return cls.state_transfer(builder).paged_layout_id


def test_layout_id_names_the_page_fields_and_the_ssm_storage(monkeypatch):
    """BF16 draft rows add `*_draft` fields and FlyDSL decode stores the SSM
    transposed; neither shows in the parent's shapes, so both are spelled."""
    builder, cls = _builder()
    plain = _layout_id(builder, cls, monkeypatch)
    assert ":ssm_storage=kv:page=k,v:ple_conv=" in plain

    names = ("k", "v", "k_scale", "v_scale", "k_draft", "v_draft")
    builder._qsa_fields = lambda: [SimpleNamespace(name=n) for n in names]
    draft = _layout_id(builder, cls, monkeypatch)
    assert ":page=k,v,k_scale,v_scale,k_draft,v_draft:" in draft

    builder._ssm_storage = lambda: "vk"
    transposed = _layout_id(builder, cls, monkeypatch)
    assert ":ssm_storage=vk:" in transposed
    assert len({plain, draft, transposed}) == 3


def _flydsl_builder(monkeypatch, *, replayssm=False):
    """A builder with one GDN layer whose 128x128 SSM state FlyDSL can decode."""
    from atom.model_ops.attentions.gdn_attn import LINEAR_STATE_ROWS
    from atom.model_ops.fla_ops import gdn_flydsl

    monkeypatch.setattr(gdn_flydsl, "ops", lambda: object())
    monkeypatch.setattr(gdn_flydsl, "decode_device_supported", lambda device: True)
    gdn_flydsl.backend.cache_clear()
    builder, cls = _builder()
    del builder._ssm_storage
    builder.replayssm = replayssm
    builder.device = torch.device("cpu")
    builder._state_shape_for_runner = lambda: ((3, 4), (2, 128, 128))
    builder._state_dtypes = lambda: (torch.bfloat16, torch.float32)

    class _GdnLayer:
        base_linear_attention = True
        impl = SimpleNamespace(
            allow_aiter_flydsl=True, dt_bias=torch.zeros(1, dtype=torch.bfloat16)
        )

    layer = _GdnLayer()
    builder._pool_rows = {LINEAR_STATE_ROWS: {layer: 0}}
    return builder, cls, layer


@pytest.mark.parametrize(
    "env, replayssm, storage",
    [
        ({}, False, "vk"),
        ({"ATOM_ENABLE_GDN_DECODE_LOSSY_FAST": "1"}, False, "kv"),
        ({"ATOM_GDN_DECODE_BACKEND": "triton"}, False, "kv"),
        ({}, True, "kv"),
    ],
)
def test_ssm_storage_follows_the_decode_switches(monkeypatch, env, replayssm, storage):
    pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
    from atom.model_ops.fla_ops import gdn_flydsl

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    try:
        builder, cls, _ = _flydsl_builder(monkeypatch, replayssm=replayssm)
        assert cls._ssm_storage(builder) == storage
        layout = _layout_id(builder, cls, monkeypatch)
        assert f":ssm_storage={storage}:" in layout
    finally:
        gdn_flydsl.backend.cache_clear()


def test_binding_refuses_an_ssm_order_the_layout_id_does_not_say(monkeypatch):
    pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
    from atom.model_ops.fla_ops import gdn_flydsl

    try:
        builder, cls, layer = _flydsl_builder(monkeypatch)
        state = torch.zeros(2, 2, 128, 128)
        monkeypatch.setattr(
            GDNAttentionMetadataBuilder,
            "build_kv_cache_tensor",
            lambda self, module: SimpleNamespace(v_cache=state),
        )
        bound = cls._build_gdn_cache_tensor(builder, layer)
        assert bound.v_cache.stride() == state.transpose(-1, -2).stride()
        builder._ssm_storage_order = "kv"
        with pytest.raises(RuntimeError, match="ssm_storage=kv"):
            cls._build_gdn_cache_tensor(builder, layer)
    finally:
        gdn_flydsl.backend.cache_clear()


def test_ple_layout_id_and_checkpoint_planes_match_the_allocated_state(monkeypatch):
    """The layout id, checkpoint plane sizes and slot size all come from the
    tensors `allocate_per_req_cache` really creates, so none can drift."""
    pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
    import re

    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
    from atom.model_ops.attentions.qwen4_exp_attn import PLE_STATE_SLOT_CLASS

    if not torch.cuda.is_available():
        pytest.skip("allocates the PLE state on the GPU")
    builder, cls = _builder()
    builder.slots_per_req = lambda: 1
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder, "allocate_per_req_cache", lambda self, e: {}
    )
    caches = cls.allocate_per_req_cache(builder, {PLE_STATE_SLOT_CLASS: 4})
    states = [caches["ple_conv_state"], caches["ple_ngram_state"]]
    assert int(states[1][0, 0]) == builder.eos_token_id

    fragment = cls._ple_layout_fragment(builder)
    declared = re.findall(r":ple_(\w+)=\(([\d,]+)\),(torch\.\w+)", fragment)
    assert [tag for tag, _, _ in declared] == ["conv", "ngram"]
    assert fragment.endswith(":ple_order=conv,ngram")
    for (_, dims, dtype), state in zip(declared, states):
        assert tuple(int(d) for d in dims.split(",") if d) == tuple(state.shape[1:])
        assert dtype == str(state.dtype)
    slot_bytes = [state[0].numel() * state.element_size() for state in states]
    assert cls._ple_checkpoint_planes(builder) == [(n, 1) for n in slot_bytes]
    assert cls._ple_state_spec(builder).entry_bytes == sum(slot_bytes)


@pytest.mark.parametrize(
    "kv_transfer_config, paged",
    [
        (None, False),
        ({}, False),
        ({"kv_connector": "lmcache_offload", "kv_role": "offload"}, True),
        ({"kv_connector": "lmcache_mp", "kv_role": "offload"}, True),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"}
                ],
            },
            True,
        ),
    ],
)
def test_page_checkpoints_follow_an_offload_connector(
    monkeypatch, kv_transfer_config, paged
):
    """A server without offload keeps main's fork slots; only a tier that reads
    the PAGE image moves Qwen3.8 onto it."""
    builder, cls = _builder()
    monkeypatch.setattr(
        "atom.model_ops.attentions.gdn_attn.get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    builder.model_runner.config.kv_transfer_config = kv_transfer_config
    assert cls._uses_paged_checkpoints(builder) is paged
    assert bool(cls.state_transfer(builder).copies) is paged
    if not paged:
        assert cls.checkpoint_page_unit_bytes(builder) is None


def test_page_checkpoints_do_not_hang_on_the_model_type_spelling():
    """Only `Qwen4ExpBackend` builds this builder; a flattened or renamed text
    type must not turn the state tier off after validation accepted offload."""
    builder, cls = _builder()
    builder.model_runner.config.hf_config.model_type = "qwen4_exp_flattened"
    assert cls._uses_paged_checkpoints(builder)


def test_validate_config_allows_offload_and_rejects_rapidserve():
    builder, _ = _builder()
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpBackend

    Qwen4ExpBackend.validate_config(builder.model_runner.config)
    builder.model_runner.config.enable_rapidserve = True
    with pytest.raises(ValueError, match="fork slots"):
        Qwen4ExpBackend.validate_config(builder.model_runner.config)


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        {"kv_connector": "lmcache_offload", "kv_role": "offload"},
        {"kv_connector": "lmcache_mp", "kv_role": "offload"},
        {
            "kv_connector": "multi",
            "connectors": [{"kv_connector": "lmcache_offload", "kv_role": "offload"}],
        },
    ],
)
def test_validate_config_allows_offload_connectors(kv_transfer_config):
    builder, _ = _builder()
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpBackend

    builder.model_runner.config.kv_transfer_config = kv_transfer_config
    Qwen4ExpBackend.validate_config(builder.model_runner.config)


@pytest.mark.parametrize(
    "kv_transfer_config, backend",
    [
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, "mooncake"),
        ({"kv_connector": "moriio", "kv_role": "kv_consumer"}, "moriio"),
        # No connector named: the factory defaults to moriio.
        ({"kv_role": "kv_producer"}, "moriio"),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                ],
            },
            "mooncake",
        ),
        # The connector is built from the canonical name, so the check is too.
        ({"kv_connector": "Mooncake", "kv_role": "kv_producer"}, "mooncake"),
        ({"kv_connector": "MoRIIO", "kv_role": "kv_consumer"}, "moriio"),
        ({"kv_connector": " moriio", "kv_role": "kv_consumer"}, "moriio"),
        # An offload backend that does not read the Qwen3.8 PAGE image.
        ({"kv_connector": "mooncake_store", "kv_role": "offload"}, "mooncake_store"),
    ],
)
def test_validate_config_rejects_pd_connectors(kv_transfer_config, backend):
    """A P/D peer would get the QSA pages and continue from a zero GDN/PLE state."""
    builder, _ = _builder()
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpBackend

    builder.model_runner.config.kv_transfer_config = kv_transfer_config
    with pytest.raises(ValueError, match=f"kv_connector={backend!r}"):
        Qwen4ExpBackend.validate_config(builder.model_runner.config)


def _transfer_config(kv_transfer_config):
    return SimpleNamespace(kv_transfer_config=kv_transfer_config)


@pytest.mark.parametrize(
    "kv_transfer_config, backend",
    [
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, "mooncake"),
        ({"kv_connector": "Mooncake", "kv_role": "kv_producer"}, "mooncake"),
        ({"kv_connector": "MoRIIO", "kv_role": "kv_consumer"}, "moriio"),
        ({"kv_connector": " moriio", "kv_role": "kv_consumer"}, "moriio"),
        ({"kv_role": "kv_producer"}, "moriio"),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                    {"kv_connector": " MOONCAKE ", "kv_role": "kv_producer"},
                ],
            },
            "mooncake",
        ),
    ],
)
def test_qwen38_refuses_every_pd_spelling(kv_transfer_config, backend):
    """A raw-string check let `"Mooncake"` past while the factory built the
    mooncake connector from its canonical name; the GDN/PLE state then never
    reached decode."""
    from atom.model_ops.attentions.qwen4_exp_transfer import check_qwen38_kv_transfer

    with pytest.raises(ValueError, match=f"disaggregated.*kv_connector={backend!r}"):
        check_qwen38_kv_transfer(_transfer_config(kv_transfer_config))


def test_qwen38_refuses_an_offload_backend_that_skips_its_image():
    """mooncake_store is an offload tier, but its layout check is Qwen3-Next's:
    refused at validation, not after PAGE checkpoints resized the KV budget."""
    from atom.model_ops.attentions.qwen4_exp_transfer import (
        check_qwen38_kv_transfer,
        qwen38_offload_configured,
    )

    config = _transfer_config({"kv_connector": "Mooncake_Store", "kv_role": "offload"})
    with pytest.raises(ValueError, match="kv_connector='mooncake_store' does not read"):
        check_qwen38_kv_transfer(config)
    assert not qwen38_offload_configured(config)


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        None,
        {},
        {"kv_connector": "lmcache_offload", "kv_role": "offload"},
        {"kv_connector": " LMCache_MP", "kv_role": "offload"},
        {"kv_connector": "LMCacheMPConnector", "kv_role": "offload"},
        {
            "kv_connector": "multi",
            "connectors": [{"kv_connector": "LMCacheOffloadConnector"}],
        },
    ],
)
def test_qwen38_accepts_the_lmcache_tiers(kv_transfer_config):
    from atom.model_ops.attentions.qwen4_exp_transfer import (
        check_qwen38_kv_transfer,
        qwen38_offload_configured,
    )

    config = _transfer_config(kv_transfer_config)
    check_qwen38_kv_transfer(config)
    assert qwen38_offload_configured(config) is bool(kv_transfer_config)


def _paged_builder(blocks=3, *, spec=True):
    """A builder with a CPU QSA arena allocated the way the runner does."""
    builder, cls = _builder()
    builder._qsa_fields = _fields
    builder.device = torch.device("cpu")
    builder.kv_pools = {}
    builder._pool_rows = {}
    cls.allocate_kv_cache_tensors(builder, blocks=blocks, buf=None)
    published = published_bytes_for(_fields())
    builder.model_runner.state_runtime = (
        SimpleNamespace(
            checkpoint_spec=SimpleNamespace(
                page_unit_bytes=published, image_bytes=published + published // 2
            )
        )
        if spec
        else None
    )
    return builder, cls


def test_qsa_builder_inherits_the_page_contract_through_hooks():
    """One PAGE contract: the QSA builder overrides only the pool hooks."""
    _, cls = _builder()
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder

    for name in (
        "_page_unit_regions",
        "page_unit_views",
        "page_unit_segments",
        "checkpoint_page_unit_bytes",
        "_with_paged_state_contract",
    ):
        assert name not in vars(cls), name
        assert name in vars(GDNAttentionMetadataBuilder), name
    for name in ("_page_unit_pools", "_published_page_unit_bytes"):
        assert name in vars(cls), name
    assert not hasattr(GDNAttentionMetadataBuilder, "_page_unit_arenas")


def test_qsa_regions_are_built_once_per_allocation(monkeypatch):
    """The PAGE copy and the store thread read the image through the arena's
    regions every step; building them is fields x layers `as_strided` views."""
    builder, cls = _paged_builder()
    arena = cls._page_unit_pools(builder)[0]
    assert arena is builder.qsa_arena
    arena.region_tensors()
    monkeypatch.setattr(
        LayerMajorArena,
        "view",
        lambda *_: pytest.fail("region views rebuilt after the first walk"),
    )
    for _ in range(3):
        cls.page_unit_views(builder, [0, 1])
        cls.page_unit_segments(builder, [0, 1])
        cls.get_kv_transfer_tensors(builder)
    assert cls._page_unit_pools(builder)[0] is arena

    cls.release_kv_pools(builder)
    with pytest.raises(RuntimeError, match="no QSA arena"):
        cls._page_unit_pools(builder)


def test_page_unit_regions_are_cached_per_pool_until_release(monkeypatch):
    """Called several times a step: the key is the pool objects, not a walk
    over every region tensor, and a release drops it before the addresses
    change."""
    from atom.model_ops.attentions.pool_layout import layer_major_page_geometry

    builder, cls = _paged_builder()
    first = cls._page_unit_regions(builder)
    real = layer_major_page_geometry.layer_major_page_unit_regions
    walks = []

    def counted(pools):
        walks.append(len(pools))
        return real(pools)

    monkeypatch.setattr(
        layer_major_page_geometry, "layer_major_page_unit_regions", counted
    )
    for _ in range(3):
        assert cls._page_unit_regions(builder) is first
    assert walks == []

    cls.release_kv_pools(builder)
    assert builder._page_unit_region_cache is None
    cls.allocate_kv_cache_tensors(builder, blocks=3, buf=None)
    bases, _ = cls._page_unit_regions(builder)
    assert walks == [1]
    assert bases.tolist() == [base for base, _ in builder.qsa_arena.region_layout()]


def test_page_unit_segments_refuse_without_a_spec_and_views_do_not():
    """The store's gather must trim to the image, so with no spec it fails at
    once rather than gather a whole unit (padding and whatever follows). Only
    `page_unit_views` keeps the untrimmed path, for callers ahead of the
    runtime."""
    for runtime in (None, SimpleNamespace(checkpoint_spec=None)):
        builder, cls = _paged_builder(spec=False)
        builder.model_runner.state_runtime = runtime
        with pytest.raises(RuntimeError, match="without a checkpoint spec"):
            cls.page_unit_segments(builder, [1, 2])
        views = cls.page_unit_views(builder, [1, 2])
        assert sum(view.numel() * view.element_size() for view in views) == (
            2 * published_bytes_for(_fields())
        )


def test_page_unit_segments_name_the_bytes_page_unit_views_name():
    builder, cls = _paged_builder()
    ptrs, sizes = cls.page_unit_segments(builder, [1, 2])
    views = cls.page_unit_views(builder, [1, 2])
    assert ptrs.tolist() == [view.data_ptr() for view in views]
    assert sizes.tolist() == [view.numel() * view.element_size() for view in views]


def test_page_unit_segments_trim_to_the_image():
    builder, cls = _paged_builder()
    image = builder.model_runner.state_runtime.checkpoint_spec.image_bytes
    _, sizes = cls.page_unit_segments(builder, [0, 2])
    assert int(sizes.sum()) == image


def test_transfer_contract_matches_the_arena_byte_layout():
    builder, cls = _paged_builder()
    transfer = cls.get_kv_transfer_tensors(builder)
    assert transfer.paged_state_region_count == 8  # 4 fields * 2 layers
    assert transfer.paged_state_checkpoint_spec is (
        builder.model_runner.state_runtime.checkpoint_spec
    )
    assert [
        (page.region.base_addr, page.region.unit_bytes) for page in transfer.pages
    ] == builder.qsa_arena.region_layout()


def test_transfer_contract_catches_published_regions_drifting(monkeypatch):
    """Published views and the copy's regions read the same pool, so the check
    compares them with the arena's own offsets: a reorder there is caught."""
    builder, cls = _paged_builder()
    builder.qsa_arena.region_tensors()
    rows = builder.qsa_arena._regions
    rows[0], rows[2] = rows[2], rows[0]  # k.0 <-> v.0
    builder._page_unit_region_cache = None
    with pytest.raises(RuntimeError, match="do not match the checkpoint PAGE unit"):
        cls.get_kv_transfer_tensors(builder)


def test_transfer_contract_catches_a_missing_field():
    """`rope_positions` (or any field) absent from the published set fails
    closed. The unit-size check sees it first; a reorder, which keeps the
    size, is what needs the arena comparison (the test above)."""
    builder, cls = _paged_builder()
    builder.qsa_arena._regions = [
        row
        for row in builder.qsa_arena.region_tensors()
        if not row[0].startswith("index_compressed.")
    ]
    builder._page_unit_region_cache = None
    with pytest.raises(RuntimeError, match="PAGE unit is 592 B"):
        cls.get_kv_transfer_tensors(builder)


def test_arena_region_layout_matches_its_views():
    arena = LayerMajorArena(_fields(), entries=5, device="cpu")
    expected = [
        (view[layer].data_ptr(), view[layer].stride(0) * view.element_size())
        for field in arena.fields
        for view in (arena.view(field.name),)
        for layer in range(field.layers)
    ]
    assert arena.region_layout() == expected


def test_arena_region_tensors_name_each_field_layer_and_match_the_layout():
    """The walk `MhaKvPool` / `MlaKvPool` / Qwen3.8 share: role names and order
    as the MHA pool always published them, and the views sit where the
    declared offsets say."""
    arena = LayerMajorArena(_fields(), entries=5, device="cpu")
    regions = arena.region_tensors()
    assert [role for role, _ in regions] == [
        f"{field.name}.layer_{layer}"
        for field in arena.fields
        for layer in range(field.layers)
    ]
    assert [
        (tensor.data_ptr(), tensor.stride(0) * tensor.element_size())
        for _, tensor in regions
    ] == arena.region_layout()
    again = arena.region_tensors()
    assert again is not regions
    assert all(a is b for (_, a), (_, b) in zip(regions, again, strict=True))
