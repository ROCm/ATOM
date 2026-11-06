# SPDX-License-Identifier: MIT

"""Qwen3.8 offload: QSA pages are the checkpoint destination, PLE rides along."""

import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.model_ops.attentions.pool_layout.entry_arena import (
    EntryField,
    published_bytes_for,
)
from atom.model_ops.attentions.qwen4_exp_attn import (
    PLE_STATE_SLOT_CLASS,
    Qwen4ExpBackend,
    Qwen4ExpMetadataBuilder,
)


def qsa_like_fields(block=4):
    """Shaped after Qwen3.8's QSA arena, scaled down."""
    return [
        EntryField("k", 2, (block, 2, 8), torch.bfloat16),
        EntryField("v", 2, (block, 2, 8), torch.bfloat16),
        EntryField("index_raw", 2, (block, 1, 4), torch.bfloat16),
        EntryField("index_compressed", 2, (block // 4, 1, 4), torch.bfloat16),
    ]


def _builder():
    builder = object.__new__(Qwen4ExpMetadataBuilder)
    builder.num_spec = 1
    builder.ngram_context_len = 2
    builder.eos_token_id = 7
    builder._ssm_storage = "kv"
    builder._qsa_fields = lambda: qsa_like_fields()[:2]
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


@pytest.fixture
def one_tp_rank(monkeypatch):
    monkeypatch.setattr(
        "atom.model_ops.attentions.gdn_attn.get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )


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


def test_layout_id_names_ple_and_forks_under_pipeline_parallel(one_tp_rank):
    builder, cls = _builder()
    layout = cls.state_transfer(builder).paged_layout_id
    assert layout.startswith("gdn-paged-state-v1:")
    # `allocate_per_req_cache` lays a PLE conv slot out (channels, state_len):
    # hc_count * hidden_size = 16 channels, (4 - 1) * 3 + num_spec = 10 taps.
    # The planes are spelled in their image order.
    assert layout.endswith(
        ":ple_conv=(16,10),torch.bfloat16:ple_ngram=(3,),torch.int64"
    )
    builder.model_runner.config.pipeline_parallel_size = 2
    assert not cls._uses_paged_checkpoints(builder)
    assert not cls.state_transfer(builder).copies


def test_ple_turns_a_midstep_readable_transfer_off_and_says_so(
    monkeypatch, one_tp_rank, caplog
):
    """The PLE windows are snapshotted only where a forward ends."""
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder

    builder, cls = _builder()
    parent = GDNAttentionMetadataBuilder.state_transfer(builder)
    readable = replace(
        parent, readable_midstep=True, midstep_grid=64, midstep_min_offset=64
    )
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder, "state_transfer", lambda self: readable
    )
    with caplog.at_level(logging.WARNING, logger="atom"):
        transfer = cls.state_transfer(builder)
    assert "ATOM_GDN_STATE_MIDSTEP=1 does not apply" in caplog.text
    assert not transfer.readable_midstep
    assert (transfer.midstep_grid, transfer.midstep_min_offset) == (0, 0)
    assert transfer.copies == readable.copies
    assert transfer.paged_layout_id.startswith(readable.paged_layout_id + ":")


def _layout_id(builder, cls):
    return cls.state_transfer(builder).paged_layout_id


def test_layout_id_names_the_page_fields_and_the_ssm_storage(one_tp_rank):
    builder, cls = _builder()
    plain = _layout_id(builder, cls)
    kv = "(4,2,8),torch.bfloat16"
    assert f":ssm_storage=kv:page=k={kv};v={kv}:ple_conv=" in plain

    names = ("k", "v", "k_scale", "v_scale", "k_draft", "v_draft")
    builder._qsa_fields = lambda: [
        EntryField(n, 2, (4, 2, 8), torch.bfloat16) for n in names
    ]
    draft = _layout_id(builder, cls)
    assert f":page={';'.join(f'{n}={kv}' for n in names)}:" in draft

    builder._ssm_storage = "vk"
    transposed = _layout_id(builder, cls)
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
def test_ssm_storage_follows_the_decode_switches(
    monkeypatch, one_tp_rank, env, replayssm, storage
):
    from atom.model_ops.fla_ops import gdn_flydsl

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    try:
        builder, cls, _ = _flydsl_builder(monkeypatch, replayssm=replayssm)
        assert builder._ssm_storage == storage
        layout = _layout_id(builder, cls)
        assert f":ssm_storage={storage}:" in layout
    finally:
        gdn_flydsl.backend.cache_clear()


def test_binding_refuses_an_ssm_order_the_layout_id_does_not_say(monkeypatch):
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
        builder._ssm_storage = "kv"
        with pytest.raises(RuntimeError, match="ssm_storage=kv"):
            cls._build_gdn_cache_tensor(builder, layer)
    finally:
        gdn_flydsl.backend.cache_clear()


def test_allocated_ple_state_is_what_the_layout_id_spells(monkeypatch):
    builder, cls = _builder()
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder

    if not torch.cuda.is_available():
        pytest.skip("allocates the PLE state on the GPU")
    builder.slots_per_req = lambda: 1
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder, "allocate_per_req_cache", lambda self, e: {}
    )
    caches = cls.allocate_per_req_cache(builder, {PLE_STATE_SLOT_CLASS: 4})
    conv, ngram = caches["ple_conv_state"], caches["ple_ngram_state"]
    # `:ple_conv=(16,10),torch.bfloat16:ple_ngram=(3,),torch.int64`.
    assert (tuple(conv.shape), conv.dtype) == ((4, 16, 10), torch.bfloat16)
    assert (tuple(ngram.shape), ngram.dtype) == ((4, 3), torch.int64)
    assert bool((ngram == builder.eos_token_id).all())
    assert cls._ple_state_spec(builder).entry_bytes == conv[0].nbytes + ngram[0].nbytes


@pytest.mark.parametrize(
    "kv_transfer_config, paged",
    [
        (None, False),
        ({}, False),
        ({"kv_connector": "lmcache_offload", "kv_role": "offload"}, True),
        ({"kv_connector": " LMCache_MP", "kv_role": "offload"}, True),
        ({"kv_connector": "LMCacheMPConnector", "kv_role": "offload"}, True),
        (
            {
                "kv_connector": "multi",
                "connectors": [{"kv_connector": "LMCacheOffloadConnector"}],
            },
            True,
        ),
    ],
)
def test_lmcache_tiers_are_accepted_and_turn_page_checkpoints_on(
    one_tp_rank, kv_transfer_config, paged
):
    """A server without offload keeps main's fork slots."""
    builder, cls = _builder()
    builder.model_runner.config.kv_transfer_config = kv_transfer_config
    Qwen4ExpBackend.validate_config(builder.model_runner.config)
    assert cls._uses_paged_checkpoints(builder) is paged
    assert bool(cls.state_transfer(builder).copies) is paged
    if not paged:
        assert cls.checkpoint_page_unit_bytes(builder) is None


def test_page_checkpoints_do_not_hang_on_the_model_type_spelling():
    builder, cls = _builder()
    builder.model_runner.config.hf_config.model_type = "qwen4_exp_flattened"
    assert cls._uses_paged_checkpoints(builder)


@pytest.mark.parametrize("switch", ["pipeline_parallel_size", "enable_rapidserve"])
@pytest.mark.parametrize(
    "kv_transfer_config, error",
    [
        ({"kv_connector": "lmcache_offload", "kv_role": "offload"}, "fork slots"),
        # A P/D connector is refused for being one, not for the deployment.
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, "disaggregated"),
    ],
)
def test_pipeline_parallel_and_rapidserve_refuse_offload(
    switch, kv_transfer_config, error
):
    builder, _ = _builder()
    config = builder.model_runner.config
    setattr(config, switch, 2 if switch == "pipeline_parallel_size" else True)
    config.kv_transfer_config = kv_transfer_config
    with pytest.raises(ValueError, match=error):
        Qwen4ExpBackend.validate_config(config)


def _paged_builder(blocks=3, *, spec=True):
    """A builder with a CPU QSA arena allocated the way the runner does."""
    builder, cls = _builder()
    builder._qsa_fields = qsa_like_fields
    builder.device = torch.device("cpu")
    builder.kv_pools = {}
    builder._pool_rows = {}
    cls.allocate_kv_cache_tensors(builder, blocks=blocks, buf=None)
    published = published_bytes_for(qsa_like_fields())
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


class _QSALayer:
    """A QSA module as the bind walk sees it: hashable, with a row space."""

    is_qsa_attention = True

    def __init__(self, layer_num, row_space="qsa"):
        self.layer_num = layer_num
        self.qsa_row_space = row_space
        self.bound = None

    def bind_caches(self, *caches, **scales):
        self.bound = (caches, scales)


def test_a_bf16_target_gives_the_mtp_draft_layer_its_own_rows():
    """With BF16 caches the MTP layer is one more row of the target's fields,
    and the in-process codec is handed exactly the rows that layer binds."""
    builder, cls = _builder()
    del builder._qsa_fields  # the real declaration, not `_builder`'s stand-in
    hf = builder.model_runner.config.hf_config
    hf.head_dim, hf.indexer_head_dim = 8, 4
    builder.model_runner._get_num_kv_heads = lambda: 2
    builder.block_size = 4
    builder.cache_rope_positions = False
    builder.device = torch.device("cpu")
    builder.kv_pools = {}
    target, draft = _QSALayer(0), _QSALayer(1)
    builder._pool_rows = {"qsa": {target: 0, draft: 1}}

    names = [field.name for field in cls._qsa_fields(builder)]
    assert names == ["k", "v", "index_raw", "index_compressed"]
    cls.allocate_kv_cache_tensors(builder, blocks=2, buf=None)
    arena = builder.qsa_arena
    for layer, row in ((target, 0), (draft, 1)):
        tensor = cls.build_kv_cache_tensor(builder, layer)
        caches, scales = layer.bound
        handed = (tensor.k_cache, tensor.v_cache, *tensor.extra_planes)
        for name, bound, view in zip(names, caches, handed, strict=False):
            assert view.data_ptr() == arena.view(name)[row].data_ptr(), name
            assert view.data_ptr() == bound.data_ptr(), name
        assert len(handed) == len(names)
        assert caches[4] is None  # no rope positions without image input
        assert set(scales.values()) == {None}
        assert tensor.k_scale is None and tensor.v_scale is None


# `DenseKVByteCodec` reads these in this order; a reorder misreads every stored
# chunk, so a new plane goes last.
_STORED_ORDER = (
    "k",
    "v",
    "k_scale",
    "v_scale",
    "index_raw",
    "index_compressed",
    "rope_positions",
    "index_compressed_scale",
)


@pytest.mark.parametrize(
    "kv_cache_dtype, index_cache_dtype, draft_space",
    [
        ("fp8", "bf16", "qsa_draft"),
        ("fp8", "fp8", "qsa_draft"),
        ("fp8", "fp4", "qsa_draft"),
        ("fp8", "fp8", "qsa"),  # ATOM_QSA_DRAFT_KV_BF16=0
    ],
)
def test_every_quantized_qsa_layer_hands_the_codec_the_rows_it_binds(
    kv_cache_dtype, index_cache_dtype, draft_space
):
    """The in-process codec moves exactly what a layer registers: every field
    of its row space, its own row, in the stored order."""
    builder, cls = _builder()
    del builder._qsa_fields  # the real declaration, not `_builder`'s stand-in
    config = builder.model_runner.config
    config.kv_cache_dtype, config.index_cache_dtype = kv_cache_dtype, index_cache_dtype
    config.hf_config.head_dim, config.hf_config.indexer_head_dim = 8, 32
    builder.model_runner._get_num_kv_heads = lambda: 2
    builder.block_size = 4
    builder.cache_rope_positions = True
    builder.device = torch.device("cpu")
    builder.kv_pools = {}
    targets, draft = [_QSALayer(0), _QSALayer(1)], _QSALayer(2, draft_space)
    rows = {"qsa": {layer: i for i, layer in enumerate(targets)}}
    rows.setdefault(draft_space, {})[draft] = len(rows.get(draft_space, ()))
    builder._pool_rows = rows
    cls.allocate_kv_cache_tensors(builder, blocks=2, buf=None)
    arena = builder.qsa_arena
    names = {field.name for field in arena.fields}
    for layer in (*targets, draft):
        sfx = "_draft" if layer.qsa_row_space == "qsa_draft" else ""
        row = builder.pool_rows[layer.qsa_row_space][layer]
        tensor = cls.build_kv_cache_tensor(builder, layer)
        handed = [
            t.data_ptr()
            for t in (tensor.k_cache, tensor.v_cache, tensor.k_scale, tensor.v_scale)
            + tensor.extra_planes
            if t is not None
        ]
        assert handed == [
            arena.view(name + sfx)[row].data_ptr()
            for name in _STORED_ORDER
            if name + sfx in names
        ]
        caches, scales = layer.bound
        bound = (*caches, *scales.values())
        assert {t.data_ptr() for t in bound if t is not None} == set(handed)


def test_a_released_arena_leaves_no_page_pools():
    builder, cls = _paged_builder()
    assert cls._page_unit_pools(builder) == [builder.qsa_arena]
    cls.release_kv_pools(builder)
    with pytest.raises(RuntimeError, match="no QSA arena"):
        cls._page_unit_pools(builder)


def test_page_unit_regions_follow_a_reallocated_arena():
    """A new arena of another size puts its fields at other offsets; the PAGE
    copy must address the new one."""
    builder, cls = _paged_builder(blocks=3)
    cls._page_unit_regions(builder)
    cls.release_kv_pools(builder)
    cls.allocate_kv_cache_tensors(builder, blocks=5, buf=None)
    bases, sizes = cls._page_unit_regions(builder)
    base = builder.qsa_arena.buf.data_ptr()
    # k and v are 128 B a layer, the index fields 32 B and 8 B. A field starts
    # at its 256 B-aligned entry offset (0, 256, 512, 768) times 5 entries,
    # and its second layer one layer's 5 entries later.
    assert sizes.tolist() == [128, 128, 128, 128, 32, 32, 8, 8]
    assert (bases - base).tolist() == [0, 640, 1280, 1920, 2560, 2720, 3840, 3880]


def test_page_unit_segments_name_the_view_bytes_trimmed_to_the_image():
    builder, cls = _paged_builder()
    ptrs, sizes = cls.page_unit_segments(builder, [1, 2])
    views = cls.page_unit_views(builder, [1, 2])
    assert ptrs.tolist() == [view.data_ptr() for view in views]
    assert sizes.tolist() == [view.numel() * view.element_size() for view in views]
    image = builder.model_runner.state_runtime.checkpoint_spec.image_bytes
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


def _transfer_of(regions):
    return KVTransferTensors(
        pages=[page_region(tensor, semantic_role=role) for role, tensor in regions]
    )


def test_transfer_contract_catches_published_regions_drifting():
    builder, cls = _paged_builder()
    rows = builder.qsa_arena.region_tensors()
    rows[0], rows[2] = rows[2], rows[0]  # k.0 <-> v.0
    with pytest.raises(RuntimeError, match="do not match its checkpoint PAGE unit"):
        cls._with_paged_state_contract(builder, _transfer_of(rows))


def test_transfer_contract_catches_a_missing_field():
    builder, cls = _paged_builder()
    rows = [
        row
        for row in builder.qsa_arena.region_tensors()
        if not row[0].startswith("index_compressed.")
    ]
    with pytest.raises(RuntimeError, match="do not match its checkpoint PAGE unit"):
        cls._with_paged_state_contract(builder, _transfer_of(rows))


def test_a_page_unit_the_arena_does_not_cover_is_refused():
    builder, cls = _paged_builder()
    spec = builder.model_runner.state_runtime.checkpoint_spec
    spec.page_unit_bytes += 8
    with pytest.raises(RuntimeError, match="but the published regions cover 592 B"):
        cls.get_kv_transfer_tensors(builder)
