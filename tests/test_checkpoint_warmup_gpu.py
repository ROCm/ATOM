# SPDX-License-Identifier: MIT
"""Startup checkpoint warmup must preserve pages without a later capture reset."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_ops.attentions.pool_layout.entry_arena import (
    EntryMajorArena,
    SplitEntryMajorArena,
    plan_field_planes,
)
from atom.model_ops.attentions.pool_layout.paged_state_copy import DescriptorStaging
from atom.model_ops.attentions.pool_layout.v4_pool_geometry import UnifiedPoolGeometry

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


def test_v4_warmup_preserves_padding_and_state_without_capture(monkeypatch):
    from test_v4_checkpoint_slot_copy import TestTheBuilderDeclaresWhatItDrops

    import atom.model_ops.attentions.backends as base_mod
    import atom.model_ops.attentions.deepseek_v4_attn as mod

    b = TestTheBuilderDeclaresWhatItDrops.builder_stub()
    b._field_window_layers = ()
    b._field_window_dtype = None
    b.compress_ratios = (0, 0, 4, 128, 4, 128, 4)
    b._arena_planes, arena_rows = plan_field_planes(
        b._state_fields(), b._plane_row_widths()
    )
    b.pool_geometry = geo = UnifiedPoolGeometry(
        b._geometry_ratios(),
        num_blocks=8,
        num_slots=1,
        ring_slots=b.win_with_spec,
        block_size=b.block_size,
        arena_rows=arena_rows,
        slot_align_rows=2,
    )
    width = b._plane_row_widths()[0]
    plane = torch.zeros(geo.plane_rows, width, dtype=torch.uint8, device="cuda")
    index_bytes = (b.block_size // 4) * (b.index_head_dim + 4)
    index = torch.zeros(
        len(b.csa_layers), 8, index_bytes, dtype=torch.uint8, device="cuda"
    )
    arena = SplitEntryMajorArena(
        [
            EntryMajorArena(
                b._arena_planes[0],
                geo.slot_positions,
                "cuda",
                buf=plane.flatten(),
                slot_stride=geo.slot_rows * width,
                live_entries=1,
            )
        ]
    )
    b.model_runner = SimpleNamespace(
        v4_kv_plane=plane,
        v4_kv_plane_rope=None,
        v4_csa_idx_kv=index,
        v4_state_arena=arena,
        pool_plan=SimpleNamespace(entries={mod.STATE_SLOT_CLASS: 1}),
        config=SimpleNamespace(
            kv_cache_block_size=256,
            decode_context_parallel_size=1,
            enforce_eager=True,
        ),
    )
    b._checkpoint_range_cache = None
    b._checkpoint_plan_cache = None
    b._checkpoint_slot_base_cache = None
    b._slot_view_cache = None
    b._page_unit_region_cache = None
    b._page_unit_region_owners = ()
    b._indexer_block_bytes = index_bytes
    unit_bytes = geo.block_bytes(width) + len(b.csa_layers) * index_bytes
    spec = PagedStateCheckpointSpec(
        unit_bytes,
        geo.slot_bytes(width),
        "warmup-test",
        image_bytes=b.checkpoint_image_bytes(),
    )
    b.model_runner.state_runtime = SimpleNamespace(checkpoint_spec=spec)
    plan = b._checkpoint_copy_plan()
    b._checkpoint_staging_cache = DescriptorStaging(plan.num_spans, "cuda")
    null = plane[: geo.envelope_rows]
    assert not torch.count_nonzero(null)
    assert torch.isneginf(plane.view(torch.float32)).any()
    before_plane, before_index = plane.clone(), index.clone()

    launch = Mock(wraps=base_mod.launch_copy_descriptor)
    monkeypatch.setattr(base_mod, "launch_copy_descriptor", launch)
    b.warmup_per_req_cache()
    torch.cuda.synchronize()

    assert launch.call_count == 1
    assert not torch.count_nonzero(null)
    assert torch.equal(plane, before_plane)
    assert torch.equal(index, before_index)


def test_gdn_warmup_preserves_padding_and_state_without_capture(monkeypatch):
    from test_kda_checkpoint_slot_copy import Mixin, builder

    import atom.model_ops.attentions.backends as base_mod
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
    from atom.model_ops.attentions.pool_layout.page_unit_geometry import (
        PageUnitGeometryMixin,
    )

    data = builder(num_slots=2)
    b = object.__new__(GDNAttentionMetadataBuilder)
    b.replayssm = False
    b.model_runner = data.model_runner
    for name in ("num_state_layers", "_state_shape_for_runner", "_state_dtypes"):
        setattr(b, name, getattr(data, name))
    b.model_runner.mamba_k_cache = b.model_runner.mamba_k_cache.cuda().fill_(1.25)
    b.model_runner.mamba_v_cache = b.model_runner.mamba_v_cache.cuda().fill_(-0.5)
    pool = torch.zeros(8, 128, dtype=torch.uint8, device="cuda")
    image_bytes = Mixin.checkpoint_image_bytes(b)
    spec = PagedStateCheckpointSpec(
        128, image_bytes, "gdn-warmup", image_bytes=image_bytes
    )
    b.model_runner.state_runtime = SimpleNamespace(checkpoint_spec=spec)
    b._page_unit_regions = lambda: (
        np.array([pool.data_ptr()], dtype=np.int64),
        np.array([128], dtype=np.int64),
    )
    for name in ("_page_unit_bases", "_page_unit_stream_sizes"):
        setattr(b, name, getattr(PageUnitGeometryMixin, name).__get__(b))
    plan = b._checkpoint_copy_plan()
    b._checkpoint_staging_cache = DescriptorStaging(plan.num_spans, "cuda")
    before_k = b.model_runner.mamba_k_cache.clone()
    before_v = b.model_runner.mamba_v_cache.clone()

    launch = Mock(wraps=base_mod.launch_copy_descriptor)
    monkeypatch.setattr(base_mod, "launch_copy_descriptor", launch)
    b.warmup_per_req_cache()
    torch.cuda.synchronize()

    assert launch.call_count == 1
    assert not torch.count_nonzero(pool)
    assert torch.equal(b.model_runner.mamba_k_cache, before_k)
    assert torch.equal(b.model_runner.mamba_v_cache, before_v)
