"""DeepSeek-V4 compressed/SWA/arena KV-cache manager."""

from atom.kv_cache.base_manager import BaseKvCacheManager
from atom.kv_cache.dsv4.compressed_pool import Dsv4CompressedPool


class Dsv4KvCacheManager(BaseKvCacheManager):
    """Coordinate compressed blocks, paged SWA, CSA sources, and arena lending."""

    @staticmethod
    def _make_primary_free_list(
        capacity: int, *, initially_backed: bool
    ) -> Dsv4CompressedPool:
        return Dsv4CompressedPool(capacity, initially_backed=initially_backed)

    def ids_conserved(self) -> bool:
        """Public invariant helper for repeated arena borrow/return tests."""
        return self._free_list.ids_conserved() and self.swa._free_list.ids_conserved()
