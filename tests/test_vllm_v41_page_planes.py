# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's PAGE planes, as the offload connector publishes them.

The connector cannot infer these from vLLM's registration dict: V4.1 declares
one proxy layer whose tensor is `(2, blocks, 256, 1, head_size)` of uint8,
where the leading `2` is a byte-accounting axis rather than K and V. Every
assertion here is about a failure that is otherwise silent or arrives far from
its cause:

* **Plane order is the key space.** An object stored under one order is
  unreadable under another, and nothing checks it at read time -- the bytes
  come back and decode into the wrong planes.
* **The STATE tail must stay out.** It is addressed by request slot, not by
  page, so a positional mover would hand one request another's window rings.
* **`num_blocks` is the pool's page count**, never the proxy tensor's leading
  dimension, which also spans the withheld STATE tail.

No GPU and no vLLM: the planes are modelled as plain CPU tensors with the
strides `PagedAttentionCache` gives them.
"""

import itertools

import pytest

torch = pytest.importorskip("torch")

from atom.plugin.vllm.deepseek_v41_bridge import v41_page_planes

# The shipped DeepSeek-V4.1-Flash geometry, BF16 pool: 40 layers, owners
# (2,2) (8,2) (14,2) (20,1), head_dim 512, index_head_dim 128,
# candidate_block_size 8, PAGE = 256 tokens.
PAGE_BYTES = 655_360
INDEX_ROW_BYTES = 132
OWNER_ROWS = {2: 128, 8: 128, 14: 128, 20: 256}
PAGED_BYTES = PAGE_BYTES + sum(OWNER_ROWS.values()) * INDEX_ROW_BYTES


class FakeGeometry:
    def __init__(self, page_bytes, paged_bytes, state_bytes):
        self.page_bytes = page_bytes
        self.paged_bytes = paged_bytes
        self.state_bytes = state_bytes


class FakeCache:
    """A `PagedAttentionCache`-shaped object carved out of one flat arena.

    Built the way the real one is -- the main pages at the head, each index
    plane dense in its own rows at a fixed offset past them, the per-request
    state as a disjoint view of the tail -- so that a mover which strays out of
    a plane lands somewhere this test can see.
    """

    def __init__(self, num_pages=8, num_slots=4, state_bytes=5_276_672):
        self.num_pages = num_pages
        self.num_slots = num_slots
        main_bytes = num_pages * PAGE_BYTES
        index_bytes = num_pages * sum(OWNER_ROWS.values()) * INDEX_ROW_BYTES
        paged = main_bytes + index_bytes
        self.backing = torch.zeros(paged + num_slots * state_bytes, dtype=torch.uint8)
        self.page_bytes = self.backing[:main_bytes].view(num_pages, PAGE_BYTES)
        self.index_planes = {}
        offset = main_bytes
        for owner, rows in OWNER_ROWS.items():
            width = INDEX_ROW_BYTES
            self.index_planes[owner] = self.backing.as_strided(
                (num_pages, rows, width),
                (rows * width, width, 1),
                self.backing.storage_offset() + offset,
            )
            offset += num_pages * rows * width
        self.state_bytes = self.backing[paged:].view(num_slots, state_bytes)
        self.geometry = FakeGeometry(PAGE_BYTES, PAGED_BYTES, state_bytes)

    def unit_regions(self):
        regions = [(self.page_bytes.data_ptr(), PAGE_BYTES)]
        regions += [
            (p.data_ptr(), p.stride(0) * p.element_size())
            for p in self.index_planes.values()
        ]
        return regions


def test_plane_order_is_the_cache_s_own_unit_order():
    """The order is the key space, so it is derived, never restated."""
    cache = FakeCache()
    planes = v41_page_planes(cache)
    published = [
        plane.numel() // cache.num_pages * plane.element_size() for _, plane in planes
    ]
    assert published == [size for _, size in cache.unit_regions()]


def test_the_main_page_leads_and_index_planes_follow_owner_order():
    cache = FakeCache()
    roles = [role for role, _ in v41_page_planes(cache)]
    assert roles == ["dsv41.page"] + [
        f"dsv41.index_plane.{owner}" for owner in OWNER_ROWS
    ]


def test_published_bytes_close_against_the_declared_page_unit():
    """`Σ plane unit_bytes == geometry.paged_bytes`, the L1 self-check."""
    cache = FakeCache()
    published = sum(
        plane.numel() // cache.num_pages * plane.element_size()
        for _, plane in v41_page_planes(cache)
    )
    assert published == PAGED_BYTES == cache.geometry.paged_bytes


def test_no_plane_reaches_the_state_tail():
    """The per-request STATE is addressed by slot; a page mover must not see it.

    Checked on addresses rather than by writing and reading back, because the
    failure this guards against is a plane whose *stride* runs past its own
    region -- which a value check on page 0 would not notice.
    """
    cache = FakeCache()
    state_start = cache.state_bytes.data_ptr()
    state_end = state_start + cache.state_bytes.numel()
    for role, plane in v41_page_planes(cache):
        begin = plane.data_ptr()
        end = begin + plane.numel() * plane.element_size()
        assert end <= state_start or begin >= state_end, role


def test_every_plane_is_contiguous_and_whole_in_blocks():
    """What `DenseKVByteCodec` requires of each segment it slices."""
    cache = FakeCache()
    for role, plane in v41_page_planes(cache):
        assert plane.is_contiguous(), role
        assert plane.numel() % cache.num_pages == 0, role


def test_planes_tile_the_paged_region_without_gaps_or_overlap():
    """Together the planes are the paged region exactly -- once each.

    A carving error that double-counted a plane would still satisfy the sum
    check above if another plane were short by the same amount.
    """
    cache = FakeCache()
    spans = sorted(
        (plane.data_ptr(), plane.data_ptr() + plane.numel() * plane.element_size())
        for _, plane in v41_page_planes(cache)
    )
    assert spans[0][0] == cache.backing.data_ptr()
    for (_, prev_end), (begin, _) in itertools.pairwise(spans):
        assert begin == prev_end
    assert spans[-1][1] == cache.state_bytes.data_ptr()


def test_the_v41_proxy_layer_is_recognised_for_non_immediate_block_reuse():
    """V4.1 has V4's global-arena property and must get V4's reuse patch.

    The markers are matched as substrings, and `".atom_deepseek_v4_proxy"` is
    NOT a substring of `"...atom_deepseek_v41_proxy"` -- so V4.1 silently went
    without it. Its PAGE and STATE share one address space (a slot's ring is
    an offset past the absolute end of the paged region), which is exactly the
    layout the patch exists to stop vLLM from recycling out from under.
    """
    from atom.plugin.vllm.deepseek_v4_prefix_patch import _V4_PROXY_LAYER_MARKERS
    from atom.plugin.vllm.deepseek_v41_bridge import (
        ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
    )

    assert any(
        marker in ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME
        for marker in _V4_PROXY_LAYER_MARKERS
    )


def test_the_state_tier_derives_its_size_and_identity_from_the_geometry():
    """So speculative decoding, when it lands, cannot reuse the wrong bytes.

    `ring_slots` and `compress_ring_slots` both carry `speculative_tokens`, so
    turning spec on grows the STATE image. The tier must take both its entry
    size and its layout identity from the geometry rather than restating them:
    the first keeps the transfer sized correctly, and the second is what puts
    the larger images in their own key space instead of over the old ones.
    """
    pytest.importorskip("lmcache")
    from atom.plugin.vllm.kv_transfer.v41_state import V41StateViews

    class Geo(FakeGeometry):
        layout_id = "dsv41-bf16-...:spec=5"

    cache = FakeCache()
    cache.geometry = Geo(PAGE_BYTES, PAGED_BYTES, 9_999_991)
    views = V41StateViews(cache, stage_depth=1)
    assert views.entry_bytes == cache.geometry.state_bytes
    assert views.layout_id == cache.geometry.layout_id


def test_the_bridge_sizes_the_pool_for_no_speculation_and_refuses_it():
    """These two must be changed together or the pool is sized short.

    `v41_proxy_geometry` passes `speculative_tokens=0`, which is only correct
    while speculative decoding is refused -- the slack it leaves out is real
    pool bytes. Whoever lifts the refusal has to lift this too, and this test
    is what says so.
    """
    import re

    with open("atom/plugin/vllm/deepseek_v41_bridge.py") as handle:
        src = handle.read()
    assert re.search(r"speculative_tokens=0", src), (
        "v41_proxy_geometry no longer hardcodes speculative_tokens=0 -- if "
        "speculative decoding is now supported, drop this test; if not, the "
        "pool is being sized from an unverified source"
    )
    with open("atom/plugin/vllm/platform.py") as handle:
        platform = handle.read()
    assert "does not support speculative" in platform, (
        "the hardcoded speculative_tokens=0 is only safe while the platform "
        "refuses speculative decoding, and that refusal is gone"
    )
