"""Scheduler-facing KV manager contract tests."""

from conftest import MockConfig

from atom.kv_cache.factory import make_kv_cache_manager
from atom.kv_cache.protocol import KvCacheManager
from atom.model_engine.sequence import Sequence


def test_factory_result_satisfies_runtime_protocol():
    manager = make_kv_cache_manager(MockConfig())
    assert isinstance(manager, KvCacheManager)
    assert manager.num_total_blocks == 10
    assert manager.kv_usage() == 0.0


def test_dense_window_hooks_and_tables_are_safe_noops():
    manager = make_kv_cache_manager(MockConfig())
    seq = Sequence([1, 2, 3, 4], block_size=4)
    manager.allocate(seq)
    manager.materialize_window(seq, len(seq))
    manager.ensure_window_for_tokens(seq, 0, len(seq))
    manager.finish_prefill_chunk(seq)
    tables = manager.build_batch_tables([seq])
    assert tables.is_empty
    assert tables.logical_csa_boundary_source_ids.tolist() == [-1]


def test_manager_metrics_and_block_access():
    manager = make_kv_cache_manager(MockConfig(num_kvcache_blocks=4))
    seq = Sequence([1, 2, 3, 4], block_size=4)
    manager.allocate(seq)
    assert manager.kv_usage() == 0.25
    block = manager.get_block(seq.block_table[0])
    assert block.block_id == seq.block_table[0]
    manager.deallocate(seq)
    assert manager.kv_usage() == 0.0


def test_per_request_slot_count_is_exposed_without_free_list_reachthrough():
    manager = make_kv_cache_manager(MockConfig(num_per_req_cache_groups=3))
    assert manager.num_free_per_req_cache_groups == 3
