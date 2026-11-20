# SPDX-License-Identifier: MIT

"""Qwen3.8 offload: QSA pages are the checkpoint destination, PLE rides along."""

import logging
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.model_ops.attentions import gdn_attn, qwen4_exp_attn
from atom.model_ops.attentions.gdn_attn import (
    LINEAR_STATE_ROWS,
    GDNAttentionMetadataBuilder,
)
from atom.model_ops.attentions.pool_layout.entry_arena import (
    EntryField,
    published_bytes_for,
)
from atom.model_ops.attentions.pool_layout.sub_pool_spec import plan_pools
from atom.model_ops.attentions.qwen4_exp_attn import (
    PLE_STATE_SLOT_CLASS,
    Qwen4ExpBackend,
    Qwen4ExpMetadataBuilder,
)
from atom.model_ops.fla_ops import gdn_flydsl
from atom.plugin.vllm.kv_transfer.mp_page_layout import build_mp_transfer_tensors


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
    builder.replayssm = False
    builder.ngram_context_len = 2
    builder.eos_token_id = 7
    builder._ssm_storage = "kv"
    builder._qsa_fields = lambda: qsa_like_fields()[:2]
    builder.num_state_layers = lambda: 2
    builder._state_shape_for_runner = lambda: ((3, 4), (2, 2))
    builder._state_dtypes = lambda: (torch.bfloat16, torch.float32)
    builder._replayssm_bytes_per_slot = lambda: 0
    builder.model_runner = SimpleNamespace(
        config=SimpleNamespace(
            torch_dtype=torch.bfloat16,
            pipeline_parallel_size=1,
            enable_rapidserve=False,
            kv_cache_dtype="bf16",
            index_cache_dtype=None,
            kv_cache_block_size=64,
            enable_dp_attention=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_tbo=False,
            enable_tbo_decode=False,
            enable_prefix_caching=True,
            state_checkpoint_interval_tokens=8192,
            speculative_config=SimpleNamespace(method="mtp"),
            kv_transfer_config={
                "kv_connector": "lmcache_offload",
                "kv_role": "offload",
            },
            hf_config=SimpleNamespace(
                model_type="qwen4_exp_text",
                num_attention_heads=2,
                ple_layer_ids=[2],
                linear_conv_kernel_dim=4,
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


def test_decode_image_steps_the_ple_windows_back_to_the_verify_step():
    """A verify leaves each PLE window `history + num_spec` wide; the image of
    step `s` is the history-wide window starting `s` columns in. Like the GDN
    conv, it comes from the live slot, also without ReplaySSM, where the SSM
    comes from the step's slot."""
    builder, cls = _builder()
    builder.num_spec = num_spec = 2
    # (ple_conv_kernel_size - 1) * ngram_size taps; ngram_size - 1 context ids.
    conv_history, ngram_context = 3 * 3, 2
    runner = builder.model_runner
    runner.mamba_k_cache = torch.zeros(2, 4, 3, 4, dtype=torch.bfloat16)
    runner.mamba_v_cache = torch.zeros(2, 4, 2, 2, dtype=torch.float32)
    conv = runner.ple_conv_state = torch.arange(
        4 * 16 * (conv_history + num_spec), dtype=torch.float32
    ).reshape(4, 16, conv_history + num_spec)
    ngram = runner.ple_ngram_state = torch.arange(
        4 * (ngram_context + num_spec), dtype=torch.int64
    ).reshape(4, ngram_context + num_spec)
    src = (conv[1].clone(), ngram[1].clone())
    for step in range(num_spec + 1):
        cls._build_decode_images(
            builder,
            [SimpleNamespace(src_slot=1, scratch_slot=3, step=step, step_slot=2)],
        )
        assert torch.equal(
            conv[3, :, :conv_history], src[0][:, step : step + conv_history]
        )
        assert torch.equal(
            ngram[3, :ngram_context], src[1][step : step + ngram_context]
        )
    assert torch.equal(conv[1], src[0]) and torch.equal(ngram[1], src[1])


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


@pytest.mark.parametrize("ple", [True, False])
@pytest.mark.parametrize("pages", [True, False])
def test_ple_keeps_midstep_off_on_the_page_and_the_fork_path(
    monkeypatch, one_tp_rank, caplog, pages, ple
):
    """The PLE windows are snapshotted only where a forward ends, so under
    `ATOM_GDN_STATE_MIDSTEP=1` a PLE build reads no state midstep, whether it
    copies PAGE images (an offload tier) or forks slots (none, the default).
    A build without PLE keeps the parent's answer on both."""
    monkeypatch.setenv("ATOM_GDN_STATE_MIDSTEP", "1")
    qwen4_exp_attn._log_ple_midstep_off.cache_clear()
    builder, cls = _builder()
    config = builder.model_runner.config
    if not pages:
        config.kv_transfer_config = None
    if not ple:
        config.hf_config.ple_layer_ids = []
    with caplog.at_level(logging.WARNING, logger="atom"):
        transfer = cls.state_transfer(builder)
    assert transfer.copies == pages
    assert transfer.readable_midstep == (not ple)
    assert (transfer.midstep_grid, transfer.midstep_min_offset) == (
        (0, 0) if ple else (64, 3)
    )
    assert ("ATOM_GDN_STATE_MIDSTEP=1 does not apply" in caplog.text) == ple
    if pages:
        assert (":ple_conv=" in transfer.paged_layout_id) == ple


def test_the_ple_midstep_warning_is_logged_once(monkeypatch, one_tp_rank, caplog):
    monkeypatch.setenv("ATOM_GDN_STATE_MIDSTEP", "1")
    qwen4_exp_attn._log_ple_midstep_off.cache_clear()
    with caplog.at_level(logging.WARNING, logger="atom"):
        for _ in range(3):
            builder, cls = _builder()
            cls.state_transfer(builder)
    assert caplog.text.count("ATOM_GDN_STATE_MIDSTEP=1 does not apply") == 1


@pytest.mark.parametrize(
    "replayssm, num_spec, keep, state_entries",
    [
        (True, 1, "snapshot", 8 * 1 + 4),
        (False, 1, "snapshot", 8 * 2 + 4),
        (True, 0, "snapshot", 8 * 1 + 4),
        (False, 0, "settled", 8 * 1),
    ],
)
def test_decode_keep_builds_the_step_image_with_its_ple_windows(
    one_tp_rank, monkeypatch, replayssm, num_spec, keep, state_entries
):
    """Under MTP or ReplaySSM a decode boundary is behind the slot, and the
    step's image covers the PLE planes too: they share GDN's slots, scratch
    included. Nothing claims decode checkpoints are off."""
    logged = []
    monkeypatch.setattr(
        gdn_attn, "_log_no_decode_checkpoints", lambda: logged.append(True)
    )
    builder, cls = _builder()
    builder.replayssm, builder.num_spec = replayssm, num_spec
    assert cls.state_transfer(builder).decode_keep == keep
    plan = plan_pools(cls.sub_pool_specs(builder), 1 << 30, max_num_seqs=8)
    assert plan.entries[PLE_STATE_SLOT_CLASS] == state_entries
    assert not logged


@pytest.mark.parametrize("gdn_layers", [2, 0])
def test_every_pp_stage_keeps_the_same_decode_image(one_tp_rank, gdn_layers):
    """A stage with no GDN or PLE layer still has their planes, the PLE ones
    from the model's config, so it keeps and sizes as the others do."""
    builder, cls = _builder()
    builder.num_state_layers = lambda: gdn_layers
    assert cls.state_transfer(builder).decode_keep == "snapshot"
    plan = plan_pools(cls.sub_pool_specs(builder), 1 << 30, max_num_seqs=8)
    assert plan.entries[PLE_STATE_SLOT_CLASS] == 8 * 2 + 4


def test_a_plane_besides_ple_keeps_no_decode_image(one_tp_rank, monkeypatch):
    """The step's image rebuilds conv, SSM and PLE; any other plane would be
    restored as the slot is."""
    logged = []
    monkeypatch.setattr(
        gdn_attn, "_log_no_decode_checkpoints", lambda: logged.append(True)
    )
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder,
        "_checkpoint_plane_shapes",
        lambda self: [(8, 1)] * 3,
    )
    builder, cls = _builder()
    assert cls.state_transfer(builder).decode_keep == "none"
    assert logged


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
    monkeypatch.setattr(gdn_flydsl, "ops", lambda: object())
    monkeypatch.setattr(gdn_flydsl, "_validated_on_gfx942", lambda: True)
    gdn_flydsl.backend.cache_clear()
    builder, cls = _builder()
    del builder._ssm_storage
    builder.replayssm = replayssm
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
    """The layout id and the binding read one decision."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    try:
        builder, cls, layer = _flydsl_builder(monkeypatch, replayssm=replayssm)
        assert f":ssm_storage={storage}:" in _layout_id(builder, cls)
        state = torch.zeros(2, 2, 128, 128)
        monkeypatch.setattr(
            GDNAttentionMetadataBuilder,
            "build_kv_cache_tensor",
            lambda self, module: SimpleNamespace(v_cache=state),
        )
        bound = cls._build_gdn_cache_tensor(builder, layer)
        assert bound.v_cache.is_contiguous() == (storage == "kv")
    finally:
        gdn_flydsl.backend.cache_clear()


def test_allocated_ple_state_is_what_the_layout_id_spells(monkeypatch, one_tp_rank):
    builder, cls = _builder()
    builder.device = torch.device("cpu")
    builder.slots_per_req = lambda: 1
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder, "allocate_per_req_cache", lambda self, e: {}
    )
    caches = cls.allocate_per_req_cache(builder, {PLE_STATE_SLOT_CLASS: 4})
    conv, ngram = caches["ple_conv_state"], caches["ple_ngram_state"]
    assert conv.shape[0] == ngram.shape[0] == 4
    spelled = "".join(
        f":ple_{tag}={tuple(state.shape[1:])},{state.dtype}".replace(" ", "")
        for tag, state in (("conv", conv), ("ngram", ngram))
    )
    assert _layout_id(builder, cls).endswith(spelled)
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


@pytest.mark.parametrize(
    "kv_transfer_config, error",
    [
        ({"kv_connector": "lmcache_offload", "kv_role": "offload"}, "fork slots"),
        # A P/D connector is refused for being one, not for the deployment.
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, "disaggregated"),
    ],
)
def test_rapidserve_refuses_offload(kv_transfer_config, error):
    builder, _ = _builder()
    config = builder.model_runner.config
    config.enable_rapidserve = True
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
        assert [role for role, _ in tensor.extra_planes] == names[2:]
        handed = (tensor.k_cache, tensor.v_cache, *dict(tensor.extra_planes).values())
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
    of its row space, its own row, in the stored order. The vLLM MP publisher
    names the same planes after those fields."""
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
        stored = [name for name in _STORED_ORDER if name + sfx in names]
        handed = [
            t.data_ptr()
            for t in (tensor.k_cache, tensor.v_cache, tensor.k_scale, tensor.v_scale)
            + tuple(dict(tensor.extra_planes).values())
            if t is not None
        ]
        assert handed == [arena.view(name + sfx)[row].data_ptr() for name in stored]
        caches, scales = layer.bound
        bound = (*caches, *scales.values())
        assert {t.data_ptr() for t in bound if t is not None} == set(handed)
        published = build_mp_transfer_tensors([tensor], num_blocks=2).pages
        roles = {"k": "k_cache", "v": "v_cache"}
        assert [page.region.semantic_role for page in published] == [
            f"L{layer.layer_num:04d}.{roles.get(name, name)}" for name in stored
        ]


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
