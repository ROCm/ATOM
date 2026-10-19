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
    builder.model_runner.config.pipeline_parallel_size = 2
    assert not cls._uses_paged_checkpoints(builder)
    assert not cls.state_transfer(builder).copies


def test_validate_config_allows_offload_and_rejects_rapidserve():
    builder, cls = _builder()
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpBackend

    Qwen4ExpBackend.validate_config(builder.model_runner.config)
    builder.model_runner.config.enable_rapidserve = True
    with pytest.raises(ValueError, match="fork slots"):
        Qwen4ExpBackend.validate_config(builder.model_runner.config)
