# SPDX-License-Identifier: MIT
"""An LRU bounded by the bytes its entries actually retain."""

from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any

# Bytes an entry retains beyond its payload: the key object, the value object
# (a NumPy array header, a tuple), and the ordered-dict node. Measured on
# CPython 3.12 for a 16-byte key and an int32 array, rounded up.
ENTRY_OVERHEAD = 384


class ByteLRU:
    """`key -> value`, least recently used evicted first once the entries'
    costs exceed `budget` bytes. Not thread-safe: callers hold their lock
    around each call, all of which are O(1) apart from eviction."""

    def __init__(
        self,
        budget: int,
        on_evict: Callable[[Hashable, Any], None] | None = None,
    ):
        self.budget = budget
        self.bytes = 0
        self._items: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._on_evict = on_evict

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._items

    def peek(self, key: Hashable):
        """The value for `key`, or None, leaving its recency alone."""
        item = self._items.get(key)
        return None if item is None else item[0]

    def get(self, key: Hashable):
        """The value for `key` (now most recently used), or None."""
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item[0]

    def put(self, key: Hashable, value, payload_bytes: int) -> None:
        """Store `value`, charged `payload_bytes` plus `ENTRY_OVERHEAD`, and
        evict down to the budget. An entry larger than the budget is not kept."""
        cost = payload_bytes + ENTRY_OVERHEAD
        self.pop(key)
        if cost > self.budget:
            return
        self._items[key] = (value, cost)
        self.bytes += cost
        while self.bytes > self.budget:
            old_key, (old_value, old_cost) = self._items.popitem(last=False)
            self.bytes -= old_cost
            if self._on_evict is not None:
                self._on_evict(old_key, old_value)

    def pop(self, key: Hashable):
        """Remove `key` without calling `on_evict`; its value, or None."""
        item = self._items.pop(key, None)
        if item is None:
            return None
        self.bytes -= item[1]
        return item[0]

    def clear(self) -> None:
        self._items.clear()
        self.bytes = 0
