# SPDX-License-Identifier: MIT

"""GDN PAGE-image geometry for the Qwen hybrid family.

The layout id is what keeps two workers from reassembling one image at the
wrong offsets. The region bytes are what ``lmcache_mp`` checks against the
checkpoint unit: alignment padding in ``MhaKvPool.entry_bytes`` is not a
published region, so the unit is the published sum.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.page_region import page_region
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import (
    EntryField,
    LayerMajorArena,
)
from atom.model_ops.attentions.pool_layout.layer_major_page_unit_geometry import (
    layer_major_page_unit_bases,
    layer_major_page_unit_bytes,
    layer_major_page_unit_regions,
    layer_major_page_unit_segments,
    layer_major_page_unit_stream_sizes,
    layer_major_page_unit_views,
)


def _published_bytes(*, layers, entry_tokens, kv_dtype, num_kv_heads=1, head_dim=256):
    """One entry's K and V for every layer, head and token, plus an fp32 scale
    per head and token when the cache is narrower than its values."""
    scale = torch.float32.itemsize if kv_dtype.itemsize == 1 else 0
    per_token = num_kv_heads * (head_dim * kv_dtype.itemsize + scale)
    return layers * 2 * entry_tokens * per_token


def _builder_cls():
    # gdn_attn imports aiter at module load; CI's CPU runner has none, so only
    # the tests that need the builder skip there.
    pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder

    return GDNAttentionMetadataBuilder


def _uses_paged_checkpoints(config):
    cls = _builder_cls()
    builder = SimpleNamespace(model_runner=SimpleNamespace(config=config))
    builder._wants_paged_checkpoints = cls._wants_paged_checkpoints.__get__(builder)
    return cls._uses_paged_checkpoints(builder)


def _config(model_type, *, pp=1, rapidserve=False, wrapped=False):
    hf = SimpleNamespace(model_type=model_type, num_attention_heads=32)
    if wrapped:
        hf = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=SimpleNamespace(model_type=model_type, num_attention_heads=32),
        )
    return SimpleNamespace(
        hf_config=hf,
        pipeline_parallel_size=pp,
        enable_rapidserve=rapidserve,
    )


def _transfer(monkeypatch, **kwargs):
    params = {
        "layers": 45,
        "shape_k": (3, 1536),
        "shape_v": (8, 128, 128),
        "dtype_k": torch.bfloat16,
        "dtype_v": torch.bfloat16,
        "tp": 8,
        "num_spec": 0,
        "replayssm": False,
        "midstep": False,
    }
    params.update(kwargs)
    builder_cls = _builder_cls()
    monkeypatch.setattr(
        "atom.model_ops.attentions.gdn_attn.get_tp_group",
        lambda: SimpleNamespace(world_size=params["tp"]),
    )
    monkeypatch.setenv("ATOM_GDN_STATE_MIDSTEP", "1" if params["midstep"] else "0")
    builder = SimpleNamespace(
        num_spec=params["num_spec"],
        replayssm=params["replayssm"],
        _conv_state_len=lambda: params["shape_k"][0],
        _uses_paged_checkpoints=lambda: True,
        num_state_layers=lambda: params["layers"],
        _state_shape_for_runner=lambda: (params["shape_k"], params["shape_v"]),
        _state_dtypes=lambda: (params["dtype_k"], params["dtype_v"]),
    )
    for name in (
        "_decode_keep",
        "_decode_image_is_conv_and_ssm",
        "_checkpoint_plane_shapes",
    ):
        setattr(builder, name, getattr(builder_cls, name).__get__(builder))
    return builder_cls.state_transfer(builder)


def _layout(monkeypatch, **kwargs):
    return _transfer(monkeypatch, **kwargs).paged_layout_id


@pytest.mark.parametrize("midstep", [False, True])
@pytest.mark.parametrize(
    "replayssm, num_spec, keep",
    [
        (True, 3, "snapshot"),
        (True, 0, "snapshot"),
        (False, 3, "snapshot"),
        (False, 0, "settled"),
    ],
)
def test_a_decode_boundary_keeps_what_the_slot_can_give(
    monkeypatch, replayssm, num_spec, keep, midstep
):
    """Under MTP or ReplaySSM the slot is ahead of a decode boundary."""
    transfer = _transfer(
        monkeypatch, replayssm=replayssm, num_spec=num_spec, midstep=midstep
    )
    assert transfer.readable_midstep is midstep
    assert transfer.decode_keep == keep


@pytest.mark.parametrize(
    ("replayssm", "prefix_caching", "interval", "spare"),
    [
        (True, True, 8192, True),
        (False, True, 8192, False),
        (True, False, 8192, False),
        (True, True, 0, False),
    ],
)
def test_decode_images_get_spare_slots_only_where_they_are_built(
    monkeypatch, replayssm, prefix_caching, interval, spare
):
    """A full batch leaves no free slot to build a decode image in."""
    transfer = _transfer(monkeypatch, replayssm=replayssm)  # skips without aiter
    from atom.model_ops.attentions.gdn_attn import DECODE_SNAPSHOT_SCRATCH_SLOTS

    builder = SimpleNamespace(
        model_runner=SimpleNamespace(
            config=SimpleNamespace(
                enable_prefix_caching=prefix_caching,
                state_checkpoint_interval_tokens=interval,
            )
        ),
        state_transfer=lambda: transfer,
        _state_shape_for_runner=lambda: ((3, 16), (2, 4, 4)),
        _state_dtypes=lambda: (torch.bfloat16, torch.bfloat16),
        num_state_layers=lambda: 2,
        _replayssm_bytes_per_slot=lambda: 0,
        slots_per_req=lambda: 1,
        state_transfer_slots=lambda: 0,
    )
    spec = _builder_cls().state_spec(builder)
    assert spec.extra_entries == (DECODE_SNAPSHOT_SCRATCH_SLOTS if spare else 0)


class TestWhoCopies:
    def test_qwen35_text_config_copies(self):
        assert _uses_paged_checkpoints(_config("qwen3_5_moe_text"))
        assert _uses_paged_checkpoints(_config("qwen3_5_text"))
        assert _uses_paged_checkpoints(_config("qwen3_next"))

    def test_wrapper_model_type_resolves_through_text_config(self):
        assert _uses_paged_checkpoints(_config("qwen3_5_moe_text", wrapped=True))

    def test_unrelated_models_stay_on_fork(self):
        assert not _uses_paged_checkpoints(_config("qwen4_exp"))
        assert not _uses_paged_checkpoints(_config("kimi_linear"))

    def test_pipeline_parallel_and_rapidserve_stay_on_fork(self):
        assert not _uses_paged_checkpoints(_config("qwen3_5_moe_text", pp=2))
        assert not _uses_paged_checkpoints(_config("qwen3_5_moe_text", rapidserve=True))


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


def _fork_transfer(monkeypatch, *, midstep):
    _builder_cls()
    from atom.model_ops.attentions.gdn_attn import GDNStateMixin

    monkeypatch.setenv("ATOM_GDN_STATE_MIDSTEP", "1" if midstep else "0")
    builder = SimpleNamespace(_conv_state_len=lambda: 3)
    return GDNStateMixin.state_transfer(builder)


class TestForkMidstep:
    def test_off_unless_asked_for(self, monkeypatch):
        transfer = _fork_transfer(monkeypatch, midstep=False)
        assert transfer.forks and not transfer.readable_midstep

    def test_on_reads_the_chunk_grid_and_the_conv_window(self, monkeypatch):
        transfer = _fork_transfer(monkeypatch, midstep=True)
        from atom.model_ops.fla_ops.chunk import CHUNK_SIZE

        assert transfer.forks and transfer.readable_midstep
        assert transfer.midstep_grid == CHUNK_SIZE
        assert transfer.midstep_min_offset == 3


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
        published = _published_bytes(
            layers=15, entry_tokens=16, kv_dtype=torch.bfloat16
        )
        _, sizes = layer_major_page_unit_regions([pool])
        assert (
            layer_major_page_unit_bytes([pool])
            == published
            == int(sizes.sum())
            == pool.entry_bytes
        )

    def test_narrow_dtype_drops_alignment_padding(self):
        """fp8-width fields declare scales. Their group is padded to 256 B,
        and that padding is not a transfer region."""
        pool = _pool(kv_dtype=torch.int8)
        published = _published_bytes(layers=15, entry_tokens=16, kv_dtype=torch.int8)
        _, sizes = layer_major_page_unit_regions([pool])
        assert layer_major_page_unit_bytes([pool]) == published == int(sizes.sum())
        assert published < pool.entry_bytes

    def test_a_qsa_arena_block_round_trips_through_its_regions(self):
        """Qwen3.8's QSA arena is a PAGE pool of its own. The padding its
        `entry_bytes` bills between an int8 plane and its scales is no region."""
        arena = LayerMajorArena(
            [
                EntryField("k", 2, (16, 3), torch.int8),
                EntryField("k_scale", 2, (16,), torch.float32),
            ],
            2,
            "cpu",
        )
        bases, sizes = layer_major_page_unit_regions([arena])
        published = 2 * 16 * 3 + 2 * 16 * 4  # int8 k rows + fp32 k_scale, two layers
        assert len(bases) == 4 and int(sizes.sum()) == published < arena.entry_bytes
        blob = (torch.arange(published) % 251).to(torch.uint8)
        at = 0
        for _, tensor in arena.region_tensors():
            row = tensor[1].view(torch.uint8).reshape(-1)
            row.copy_(blob[at : at + row.numel()])
            at += row.numel()
        views = layer_major_page_unit_views([arena], [1])
        packed = torch.cat([view.view(torch.uint8).reshape(-1) for view in views])
        assert torch.equal(packed, blob)

    def test_regions_are_the_page_region_contract(self):
        pool = _pool(kv_dtype=torch.int8)
        bases, sizes = layer_major_page_unit_regions([pool])
        published = [
            (
                page_region(tensor, semantic_role=role).region.base_addr,
                page_region(tensor, semantic_role=role).region.unit_bytes,
            )
            for role, tensor in pool.region_tensors()
        ]
        assert published == list(zip(bases.tolist(), sizes.tolist(), strict=True))

    @pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.int8])
    def test_region_layout_names_the_published_regions(self, kv_dtype):
        """The GDN transfer check compares the published regions with this."""
        pool = _pool(kv_dtype=kv_dtype)
        published = [
            (region.base_addr, region.unit_bytes)
            for region in (
                page_region(tensor, semantic_role=role).region
                for role, tensor in pool.region_tensors()
            )
        ]
        assert published == pool.region_layout()

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
        regions = layer_major_page_unit_regions([pool])
        bases, sizes = regions
        assert (
            layer_major_page_unit_bytes([pool])
            == int(sizes.sum())
            == _published_bytes(layers=2, entry_tokens=256, kv_dtype=torch.int8)
        )
        got = layer_major_page_unit_bases(regions, [[0, 2], [1, 3]])
        want = []
        for image in ([0, 2], [1, 3]):
            for unit in image:
                want.extend(
                    int(base) + unit * int(size) for base, size in zip(bases, sizes)
                )
        assert got.shape == (2, len(sizes) * 2)
        assert got.reshape(-1).tolist() == want
        assert (
            layer_major_page_unit_stream_sizes(regions, 3).tolist()
            == sizes.tolist() * 3
        )

    def test_views_follow_the_page_stream_and_drop_the_tail(self):
        pool = _pool(kv_dtype=torch.bfloat16, layers=2, block_size=16)
        regions = list(pool.region_tensors())
        for index, (_, tensor) in enumerate(regions):
            tensor.view(torch.uint8).fill_(index + 1)
        unit = int(layer_major_page_unit_regions([pool])[1][0])
        # One full region plus three bytes of the next. The rest is padding
        # the packer must not see.
        image = unit + 3
        views = layer_major_page_unit_views([pool], [1, 0], image_bytes=image)
        blob = b"".join(bytes(view.reshape(-1).view(torch.uint8)) for view in views)
        assert len(blob) == image
        assert blob[:unit] == bytes([1]) * unit
        assert blob[unit:] == bytes([2, 2, 2])

    def test_segments_are_the_views_as_addresses(self):
        """The store gathers from addresses rather than views, so the two have
        to name the same bytes in the same order, tail trim included -- at an
        exact segment boundary, mid-segment, and with no trim at all."""
        pool = _pool(kv_dtype=torch.int8, layers=3, block_size=16)
        regions = layer_major_page_unit_regions([pool])
        unit = int(regions[1].sum())
        first = int(regions[1][0])
        for image in (0, first, first + 5, 2 * unit - 7, 3 * unit):
            views = layer_major_page_unit_views([pool], [2, 0, 3], image_bytes=image)
            ptrs, sizes = layer_major_page_unit_segments(
                regions, [2, 0, 3], entries=4, image_bytes=image
            )
            want = [
                (view.data_ptr(), view.numel() * view.element_size()) for view in views
            ]
            assert list(zip(ptrs.tolist(), sizes.tolist(), strict=True)) == want

    def test_a_unit_past_the_pool_is_refused(self):
        pool = _pool(kv_dtype=torch.bfloat16, layers=1, block_size=16)
        try:
            layer_major_page_unit_views([pool], [4])
        except IndexError as exc:
            assert "outside" in str(exc)
        else:
            raise AssertionError("expected an out-of-range PAGE unit to raise")
