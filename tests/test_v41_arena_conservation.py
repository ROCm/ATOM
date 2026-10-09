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


def _big_pools_fit() -> bool:
    """A 251,111-page pool is ~186 GiB; only build it where that fits."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) > 200 * 1024 * 1024
    except OSError:
        pass
    return False


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


def _slack(cache, planes):
    """What the arena holds that neither leg moves: the inter-region padding."""
    return cache.backing.numel() - _moved(cache, planes)


# Both layouts: the fp4 one has a different per-plane width, and a check that
# happened to hold only for bf16 would be an accident of one dtype.
# 1,000 pages pads to nothing and 1,001 does not, on the bf16 geometry. Both
# are here on purpose: the exact-equality version of this check passed every
# CPU-sized pool it was written against and then fired on the first real one,
# which paged 251,111 blocks and left 512 bytes between the regions.
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("pages,slots", [(64, 8), (1000, 64), (1001, 64), (251111, 64)])
def test_the_arena_balances_within_the_padding(packed, pages, slots):
    geometry = _geometry(packed)
    if pages > 2000 and not _big_pools_fit():
        pytest.skip("not enough host memory for a full-size pool")
    cache = _cache(geometry, pages, slots)
    slack = _slack(cache, v41_page_planes(cache))
    assert 0 <= slack < geometry.alignment


def test_some_pool_size_actually_pads():
    """The check would be vacuous if the padding were always zero.

    Without this, a tolerance of `< alignment` could be hiding the fact that
    the identity is exact and the slack is dead code -- and the next person
    would tighten it back and reproduce the false positive.
    """
    geometry = _geometry(False)
    slacks = {
        _slack(_cache(geometry, pages, 8), v41_page_planes(_cache(geometry, pages, 8)))
        for pages in (64, 1000, 1001, 1023)
    }
    assert any(s > 0 for s in slacks), f"no pool size padded: {sorted(slacks)}"


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
    geometry = cache.geometry
    assert 0 <= _slack(cache, planes) < geometry.alignment
    short = planes[:drop] + planes[drop + 1 :]
    # Far outside the padding, not merely different: the smallest plane here
    # is 2,112 B/block, so the tolerance cannot absorb a missing one.
    assert _slack(cache, short) >= geometry.alignment


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
