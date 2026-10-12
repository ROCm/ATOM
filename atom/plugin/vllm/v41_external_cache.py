# SPDX-License-Identifier: MIT
"""A CSA2 cache that pages out of vLLM's own KV allocation.

ATOM's `PagedAttentionCache` allocates its pool itself. On this path the bytes
already exist: vLLM's `KVCacheManager` allocated a slab for ATOM's proxy
attention layer, and allocating a second pool beside it would double-count the
memory vLLM has already budgeted. So the cache is handed that slab instead,
through the one seam core leaves for it.

Why the whole slab and not the paged part: V4.1 buys its pool in two
currencies, and `V41PoolGeometry.window` states a slot's STATE ring as an
offset past the *absolute* end of the paged region, which the `_indices`
kernels dereference against a single base. PAGE and STATE therefore live in
one address space, and vLLM -- which only knows how to buy the first currency
-- has to be made to buy a slab wide enough for the second behind it.
`deepseek_v41_state_reserve_patch` is what does that: it withholds the tail's
worth of block ids from the scheduler while leaving the tensor's size alone.

Everything in this file is a statement about vLLM's allocation, which is why
it is here and not beside the cache it extends.
"""

import torch

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache


class VllmBackedPagedAttentionCache(PagedAttentionCache):
    """`PagedAttentionCache` over a slab somebody else owns."""

    def __init__(self, *args, backing: torch.Tensor, **kwargs):
        # Set before `super().__init__`, which calls `_allocate_backing` while
        # it runs. An attribute assigned after that call would not exist yet
        # when the override reads it.
        self._external_backing = backing
        super().__init__(*args, **kwargs)

    def _allocate_backing(self, size, device):
        """The slab, trimmed and zeroed to core's contract.

        The arena has to behave exactly like the `torch.zeros` it replaces:
        flat uint8, contiguous, on this cache's device, zeroed, and exactly
        `size` elements. The trim is not an optimization -- every region is a
        `view` with its own strides, and a longer tail does not divide. What
        it drops is the rounding slack of a STATE tail bought in whole pages,
        under one page of a pool that holds tens of thousands.
        """
        backing = self._external_backing
        if not isinstance(backing, torch.Tensor):
            raise TypeError("An external CSA2 backing must be a tensor")
        if backing.dtype != torch.uint8 or backing.dim() != 1:
            raise ValueError("An external CSA2 backing must be flat uint8")
        if not backing.is_contiguous():
            raise ValueError("An external CSA2 backing must be contiguous")
        # Where `torch.zeros(device=device)` would have put it -- an index-less
        # `cuda` means the current device, which is what vLLM hands down in
        # `device_config.device`, and comparing that to a tensor's
        # always-indexed `.device` would never match.
        resolved = torch.empty(0, device=device).device
        if backing.device != resolved:
            raise ValueError(
                f"An external CSA2 backing on {backing.device} cannot back a "
                f"cache on {resolved}"
            )
        if backing.numel() < size:
            raise ValueError(
                f"An external CSA2 backing of {backing.numel()} bytes is short "
                f"of the {size} this geometry needs"
            )
        arena = backing[:size]
        # Core's own allocation is zeroed; the cursor reset and every cold read
        # assume it, and a reused slab is not.
        arena.zero_()
        return arena
