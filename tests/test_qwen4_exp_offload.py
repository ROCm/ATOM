# SPDX-License-Identifier: MIT

"""Qwen3.8 offload: QSA pages are the checkpoint destination, PLE rides along."""

import functools
from types import SimpleNamespace

import pytest
import torch

from atom.config import KVCacheTensor
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.model_ops.attentions.pool_layout.entry_arena import (
    EntryField,
    LayerMajorArena,
    entry_bytes_for,
)
from atom.model_ops.attentions.pool_layout.mha_page_unit_geometry import (
    mha_page_unit_regions,
    mha_page_unit_views,
)
from atom.model_ops.attentions.pool_layout.qsa_page_unit_geometry import (
    QsaRegionPool,
    qsa_published_page_bytes,
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
    pool = QsaRegionPool(arena)
    published = qsa_published_page_bytes(_fields())
    assert published <= entry_bytes_for(_fields())
    bases, sizes = mha_page_unit_regions([pool])
    assert int(sizes.sum()) == published
    assert len(bases) == 8  # 4 fields * 2 layers
    for _, tensor in pool.region_tensors():
        assert tensor.is_contiguous()
        assert tensor.shape[0] == 3


def test_one_qsa_block_round_trips_through_page_views():
    arena = LayerMajorArena(_fields(), entries=2, device="cpu")
    pool = QsaRegionPool(arena)
    blob = torch.arange(qsa_published_page_bytes(_fields()), dtype=torch.uint8)
    at = 0
    for _, tensor in pool.region_tensors():
        row = tensor[1].view(torch.uint8).reshape(-1)
        row.copy_(blob[at : at + row.numel()])
        at += row.numel()
    views = mha_page_unit_views([pool], [1])
    packed = torch.cat([view.view(torch.uint8).reshape(-1) for view in views])
    assert torch.equal(packed, blob)


def _mixed_fields(fills=(0.0, 0.0, 0.0, 0.0, 0.0)):
    """fp8 K/V, fp32 scales, bf16 and fp4-in-uint8 index rows, the element
    types a QSA block holds, plus a signed 1-byte type."""
    block = 4
    dtypes = (
        (torch.float8_e4m3fn, (block, 2, 8)),
        (torch.float32, (block, 2)),
        (torch.bfloat16, (block, 1, 4)),
        (torch.uint8, (block, 1, 2)),
        (torch.int8, (block, 1, 4)),
    )
    return [
        EntryField(f"f{i}", 2, shape, dtype, fill=fill)
        for i, ((dtype, shape), fill) in enumerate(zip(dtypes, fills))
    ]


def _block_bytes(arena, name, block):
    return arena.view(name)[:, block].contiguous().view(torch.uint8)


def test_reset_entries_zeroes_only_those_blocks_of_a_runner_buffer():
    """The runner's QSA arena sits on a shared buffer, so every fill is zero;
    the fp8 and 1-byte fields take the byte path `index_fill_` needs."""
    fields = _mixed_fields()
    blocks = 6
    buf = torch.full((entry_bytes_for(fields) * blocks,), 0x7F, dtype=torch.uint8)
    arena = LayerMajorArena(fields, blocks, "cpu", buf=buf)
    arena.reset_entries(torch.tensor([1, 4]))
    arena.reset_entries(torch.tensor([2]))
    for field in fields:
        for block in range(blocks):
            raw = _block_bytes(arena, field.name, block)
            want = 0 if block in (1, 2, 4) else 0x7F
            assert raw.eq(want).all(), (field.name, field.dtype, block)


def test_reset_entries_restores_a_non_zero_fill():
    fills = (1.0, float("-inf"), -2.0, 3.0, -1.0)
    fields = _mixed_fields(fills)
    arena = LayerMajorArena(fields, 3, "cpu")
    arena.buf.fill_(0x7F)
    arena.reset_entries(torch.tensor([1]))
    for field, fill in zip(fields, fills):
        assert arena.view(field.name)[:, 1].float().eq(fill).all(), field.name
        for block in (0, 2):
            assert _block_bytes(arena, field.name, block).eq(0x7F).all()


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
    ],
)
def test_validate_config_rejects_pd_connectors(kv_transfer_config, backend):
    """A P/D peer would get the QSA pages and continue from a zero GDN/PLE state."""
    builder, _ = _builder()
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpBackend

    builder.model_runner.config.kv_transfer_config = kv_transfer_config
    with pytest.raises(ValueError, match=f"kv_connector={backend!r}"):
        Qwen4ExpBackend.validate_config(builder.model_runner.config)


class _HostEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def test_clear_page_units_resets_the_qsa_arena_with_no_kv_pools(monkeypatch):
    """Qwen3.8's PAGE images live in the QSA arena and it has no MHA pool, so a
    released unit is cleared there; `kv_pools` would size nothing."""
    builder, cls = _builder()
    from atom.model_ops.attentions import qwen4_exp_attn
    from atom.utils import CpuGpuBuffer

    monkeypatch.setattr(
        qwen4_exp_attn,
        "CpuGpuBuffer",
        functools.partial(CpuGpuBuffer, pin_memory=False),
    )
    monkeypatch.setattr(torch.cuda, "Event", _HostEvent)
    fields = _mixed_fields()
    blocks = 6
    buf = torch.full((entry_bytes_for(fields) * blocks,), 0x7F, dtype=torch.uint8)
    builder.qsa_arena = LayerMajorArena(fields, blocks, "cpu", buf=buf)
    builder.kv_pools = {}
    builder.device = torch.device("cpu")
    cls.clear_page_units(builder, [1, 4])
    cls.clear_page_units(builder, iter([2]))
    cls.clear_page_units(builder, ())
    for field in fields:
        for block in range(blocks):
            raw = _block_bytes(builder.qsa_arena, field.name, block)
            want = 0 if block in (1, 2, 4) else 0x7F
            assert raw.eq(want).all(), (field.name, block)


def _paged_builder(blocks=3, *, spec=True):
    """A builder with a CPU QSA arena allocated the way the runner does."""
    builder, cls = _builder()
    builder._qsa_fields = _fields
    builder.device = torch.device("cpu")
    builder.kv_pools = {}
    builder.pool_rows = {}
    cls.allocate_kv_cache_tensors(builder, blocks=blocks, buf=None)
    published = qsa_published_page_bytes(_fields())
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


def test_qsa_region_pool_is_built_once_per_allocation(monkeypatch):
    builder, cls = _paged_builder()
    from atom.model_ops.attentions.pool_layout import qsa_page_unit_geometry

    built = []
    monkeypatch.setattr(
        qsa_page_unit_geometry.QsaRegionPool,
        "__init__",
        lambda self, arena: built.append(arena),
    )
    pool = cls._page_unit_pools(builder)[0]
    for _ in range(3):
        cls.page_unit_views(builder, [0, 1])
        cls.page_unit_segments(builder, [0, 1])
        cls.get_kv_transfer_tensors(builder)
    assert cls._page_unit_pools(builder)[0] is pool
    assert built == []

    cls.release_kv_pools(builder)
    assert builder.qsa_region_pool is None
    with pytest.raises(RuntimeError, match="no QSA arena"):
        cls._page_unit_pools(builder)


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
