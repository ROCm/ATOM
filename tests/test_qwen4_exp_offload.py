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
from atom.model_ops.attentions.pool_layout.mha_page_unit_geometry import (
    mha_page_unit_regions,
    mha_page_unit_views,
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
    bases, sizes = mha_page_unit_regions([arena])
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
    views = mha_page_unit_views([arena], [1])
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
    from atom.model_ops.attentions.pool_layout import mha_page_unit_geometry

    builder, cls = _paged_builder()
    first = cls._page_unit_regions(builder)
    real = mha_page_unit_geometry.mha_page_unit_regions
    walks = []

    def counted(pools):
        walks.append(len(pools))
        return real(pools)

    monkeypatch.setattr(mha_page_unit_geometry, "mha_page_unit_regions", counted)
    for _ in range(3):
        assert cls._page_unit_regions(builder) is first
    assert walks == []

    cls.release_kv_pools(builder)
    assert builder._page_unit_region_cache is None
    cls.allocate_kv_cache_tensors(builder, blocks=3, buf=None)
    bases, _ = cls._page_unit_regions(builder)
    assert walks == [1]
    assert bases.tolist() == [base for base, _ in builder.qsa_arena.region_layout()]


def test_page_unit_segments_take_the_untrimmed_path_before_the_spec():
    """The store thread can ask before the runtime exists: same safe path as
    `page_unit_views`, not an AttributeError."""
    for runtime in (None, SimpleNamespace(checkpoint_spec=None)):
        builder, cls = _paged_builder(spec=False)
        builder.model_runner.state_runtime = runtime
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
