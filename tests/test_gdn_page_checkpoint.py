# SPDX-License-Identifier: MIT

"""GDN PAGE-image geometry for the Qwen hybrid family.

The layout id is what keeps two workers from reassembling one image at the
wrong offsets. The region bytes are what ``lmcache_mp`` checks against the
checkpoint unit: alignment padding in ``MhaKvPool.entry_bytes`` is not a
published region, so the unit is the published sum.
"""

from types import SimpleNamespace

import torch

from atom.kv_transfer.disaggregation.page_region import page_region
from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
from atom.model_ops.attentions.pool_layout.mha_page_unit_geometry import (
    mha_page_unit_bases,
    mha_page_unit_regions,
    mha_page_unit_stream_sizes,
    mha_page_unit_views,
    mha_published_page_bytes,
)
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool


def _uses_paged_checkpoints(config):
    builder = SimpleNamespace(model_runner=SimpleNamespace(config=config))
    return GDNAttentionMetadataBuilder._uses_paged_checkpoints(builder)


def _config(model_type, *, pp=1, rapidserve=False, wrapped=False):
    hf = SimpleNamespace(model_type=model_type, num_attention_heads=32)
    if wrapped:
        hf = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=SimpleNamespace(
                model_type=model_type, num_attention_heads=32
            ),
        )
    return SimpleNamespace(
        hf_config=hf,
        pipeline_parallel_size=pp,
        enable_rapidserve=rapidserve,
    )


def _layout(monkeypatch, **kwargs):
    params = dict(
        layers=45,
        shape_k=(3, 1536),
        shape_v=(8, 128, 128),
        dtype_k=torch.bfloat16,
        dtype_v=torch.bfloat16,
        tp=8,
        num_spec=0,
    )
    params.update(kwargs)
    monkeypatch.setattr(
        "atom.model_ops.attentions.gdn_attn.get_tp_group",
        lambda: SimpleNamespace(world_size=params["tp"]),
    )
    builder = SimpleNamespace(
        num_spec=params["num_spec"],
        _uses_paged_checkpoints=lambda: True,
        num_state_layers=lambda: params["layers"],
        _state_shape_for_runner=lambda: (params["shape_k"], params["shape_v"]),
        _state_dtypes=lambda: (params["dtype_k"], params["dtype_v"]),
    )
    return GDNAttentionMetadataBuilder.state_transfer(builder).paged_layout_id


class TestWhoCopies:
    def test_qwen35_text_config_copies(self):
        assert _uses_paged_checkpoints(_config("qwen3_5_moe_text"))
        assert _uses_paged_checkpoints(_config("qwen3_5_text"))
        assert _uses_paged_checkpoints(_config("qwen3_next"))

    def test_wrapper_model_type_resolves_through_text_config(self):
        assert _uses_paged_checkpoints(
            _config("qwen3_5_moe_text", wrapped=True)
        )

    def test_unrelated_models_stay_on_fork(self):
        assert not _uses_paged_checkpoints(_config("qwen4_exp"))
        assert not _uses_paged_checkpoints(_config("kimi_linear"))

    def test_pipeline_parallel_and_rapidserve_stay_on_fork(self):
        assert not _uses_paged_checkpoints(_config("qwen3_5_moe_text", pp=2))
        assert not _uses_paged_checkpoints(
            _config("qwen3_5_moe_text", rapidserve=True)
        )


class TestTheLayoutId:
    def test_it_names_the_plane_order_and_the_carry_rule(self, monkeypatch):
        layout = _layout(monkeypatch)
        assert layout.startswith("gdn-paged-state-v1:")
        assert ":order=conv-all-layers,ssm-all-layers" in layout
        assert layout.endswith(":carry=all")

    def test_tp_and_spec_move_the_id(self, monkeypatch):
        assert _layout(monkeypatch) != _layout(monkeypatch, tp=4)
        assert _layout(monkeypatch) != _layout(monkeypatch, num_spec=3)

    def test_shape_dtype_and_layers_move_the_id(self, monkeypatch):
        assert _layout(monkeypatch) != _layout(monkeypatch, shape_k=(4, 1536))
        assert _layout(monkeypatch) != _layout(monkeypatch, dtype_v=torch.float32)
        assert _layout(monkeypatch) != _layout(monkeypatch, layers=44)


def _pool(*, kv_dtype, layers=15, block_size=16):
    pool = MhaKvPool(
        layers=layers,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=256,
        kv_dtype=kv_dtype,
    )
    pool.allocate(4, "cpu")
    return pool


class TestMhaRegionsMatchThePublishedPages:
    def test_bf16_published_bytes_are_the_whole_entry(self):
        pool = _pool(kv_dtype=torch.bfloat16)
        published = mha_published_page_bytes([pool])
        _, sizes = mha_page_unit_regions([pool])
        assert published == int(sizes.sum()) == pool.entry_bytes

    def test_narrow_dtype_drops_alignment_padding(self):
        """fp8-width fields declare scales. Their group is padded to 256 B,
        and that padding is not a transfer region."""
        pool = _pool(kv_dtype=torch.int8)
        published = mha_published_page_bytes([pool])
        _, sizes = mha_page_unit_regions([pool])
        assert published == int(sizes.sum())
        assert published < pool.entry_bytes

    def test_regions_are_the_page_region_contract(self):
        pool = _pool(kv_dtype=torch.int8)
        bases, sizes = mha_page_unit_regions([pool])
        published = [
            (
                page_region(tensor, semantic_role=role).region.base_addr,
                page_region(tensor, semantic_role=role).region.unit_bytes,
            )
            for role, tensor in pool.region_tensors()
        ]
        assert published == list(zip(bases.tolist(), sizes.tolist(), strict=True))

    def test_a_scheduler_entry_is_one_unit_even_when_it_holds_many_blocks(self):
        """Kernel blocks are packed into the entry ``page_pool`` charges.
        A checkpoint unit is that entry, so block id 2 is two entries down."""
        pool = MhaKvPool(
            layers=2,
            block_size=16,
            blocks_per_entry=16,
            num_kv_heads=1,
            head_dim=256,
            kv_dtype=torch.int8,
        )
        pool.allocate(4, "cpu")
        regions = mha_page_unit_regions([pool])
        bases, sizes = regions
        assert int(sizes.sum()) == mha_published_page_bytes([pool])
        got = mha_page_unit_bases(regions, [[0, 2], [1, 3]])
        want = []
        for image in ([0, 2], [1, 3]):
            for unit in image:
                want.extend(
                    int(base) + unit * int(size) for base, size in zip(bases, sizes)
                )
        assert got.shape == (2, len(sizes) * 2)
        assert got.reshape(-1).tolist() == want
        assert mha_page_unit_stream_sizes(regions, 3).tolist() == sizes.tolist() * 3

    def test_views_follow_the_page_stream_and_drop_the_tail(self):
        pool = _pool(kv_dtype=torch.bfloat16, layers=2, block_size=16)
        regions = list(pool.region_tensors())
        for index, (_, tensor) in enumerate(regions):
            tensor.view(torch.uint8).fill_(index + 1)
        unit = int(mha_page_unit_regions([pool])[1][0])
        # One full region plus three bytes of the next. The rest is padding
        # the packer must not see.
        image = unit + 3
        views = mha_page_unit_views([pool], [1, 0], image_bytes=image)
        blob = b"".join(bytes(view.reshape(-1).view(torch.uint8)) for view in views)
        assert len(blob) == image
        assert blob[:unit] == bytes([1]) * unit
        assert blob[unit:] == bytes([2, 2, 2])

    def test_a_unit_past_the_pool_is_refused(self):
        pool = _pool(kv_dtype=torch.bfloat16, layers=1, block_size=16)
        try:
            mha_page_unit_views([pool], [4])
        except IndexError as exc:
            assert "outside" in str(exc)
        else:
            raise AssertionError("expected an out-of-range PAGE unit to raise")
