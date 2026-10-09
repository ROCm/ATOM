"""Does every byte of V4.1's arena belong to exactly one leg of the offload?

The PAGE checks in `_v41_page_tensors` each compare the published planes
against another statement of the PAGE unit, so a plane missing from every
statement passes all of them. This file tests the one identity that spans both
legs, and tests it the only way that means anything: by breaking the thing it
is supposed to notice and requiring it to notice.
"""

import math

import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.plugin.vllm.deepseek_v41_bridge import v41_page_planes

_V41 = "/data/amd_int/models/DeepSeek-V4.1-Flash"


def _geometry(packed: bool):
    transformers = pytest.importorskip("transformers")
    from atom.model_ops.attentions.deepseek_v41.backend import build_v41_pool_geometry

    try:
        cfg = transformers.AutoConfig.from_pretrained(_V41, trust_remote_code=True)
    except Exception:  # noqa: BLE001 - the checkpoint is not on every box
        pytest.skip(f"no V4.1 checkpoint at {_V41}")
    return build_v41_pool_geometry(
        getattr(cfg, "text_config", cfg), 256, packed=packed, speculative_tokens=0
    )


def _cache(geometry, pages, slots):
    return PagedAttentionCache(geometry, pages, slots, torch.device("cpu"))


def _moved(cache, planes):
    pages = cache.num_pages
    unit = [p.numel() // pages * p.element_size() for _, p in planes]
    return sum(unit) * pages + cache.num_slots * int(cache.geometry.state_bytes)


# Both layouts: the fp4 one has a different per-plane width, and a check that
# happened to hold only for bf16 would be an accident of one dtype.
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("pages,slots", [(64, 8), (1000, 64)])
def test_the_arena_balances(packed, pages, slots):
    cache = _cache(_geometry(packed), pages, slots)
    assert _moved(cache, v41_page_planes(cache)) == cache.backing.numel()


@pytest.mark.parametrize("packed", [False, True])
def test_the_published_planes_are_the_cache_regions(packed):
    """Same planes, same order: the order IS the stored object's key space."""
    cache = _cache(_geometry(packed), 64, 8)
    published = [
        p.numel() // cache.num_pages * p.element_size()
        for _, p in v41_page_planes(cache)
    ]
    assert published == [size for _, size in cache.unit_regions()]
    assert sum(published) == int(cache.geometry.paged_bytes)


@pytest.mark.parametrize("drop", range(5))
def test_dropping_any_plane_breaks_the_balance(drop):
    """The mutation the identity exists for.

    Index planes are the ones that go missing quietly -- they carry the
    indexer's keys, so losing one restores a prefix whose candidate selection
    reads bytes nobody wrote. Nothing crashes; the answers just drift.
    """
    cache = _cache(_geometry(False), 64, 8)
    planes = v41_page_planes(cache)
    assert len(planes) == 5, "fixture assumes 1 main + 4 index planes"
    assert _moved(cache, planes) == cache.backing.numel()
    assert _moved(cache, planes[:drop] + planes[drop + 1 :]) != cache.backing.numel()


def test_the_state_leg_moves_the_entry_whole():
    """The half the balance cannot see.

    A field dropped from `state_fields` would move both sides of the identity
    together. What rules it out is that this path never selects fields: the
    views are `state_bytes[slot]`, the whole entry. If that ever becomes a
    per-field gather, this test is the one that should be made to fail.
    """
    cache = _cache(_geometry(False), 64, 8)
    entry = cache.state_bytes[0]
    assert entry.numel() == int(cache.geometry.state_bytes)
    assert entry.dtype == torch.uint8
    fields = sum(
        f.layers * math.prod(f.shape) * torch.empty(0, dtype=f.dtype).element_size()
        for f in cache.geometry.state_fields
    )
    # Equal but for the alignment padding the arena adds between fields.
    assert 0 <= int(cache.geometry.state_bytes) - fields < cache.geometry.alignment
