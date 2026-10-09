"""Whether V4.1's PAGE layout may be stored once for the whole TP group.

Getting this wrong is expensive in one direction and wrong in the other.
Answering "sharded" when the ranks are identical costs a factor of TP in tier
capacity and nothing else. Answering "replicated" when they are not hands
rank 0's KV to rank 3, which no counter reports and no test downstream of here
would catch.
"""

import pytest


class _Config:
    def __init__(self, tp):
        self.tensor_parallel_size = tp


def _connector(tp, v41=True):
    from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

    c = object.__new__(AtomLMCacheOffloadConnector)
    c._is_deepseek_v41 = v41
    c._config = _Config(tp)
    c._kv_cache_config = None
    c._attn_group_id = 0
    return c


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_v41_declares_full_replication(tp):
    assert _connector(tp)._page_tp_replication_factor() == tp


def test_a_missing_tp_size_reads_as_one():
    c = _connector(4)
    del c._config.tensor_parallel_size
    assert c._page_tp_replication_factor() == 1


def test_a_non_v41_model_still_asks_the_spec():
    """The blindness is specific to the proxy, not a new default.

    With no kv_cache_groups to read, every other model keeps the conservative
    answer rather than inheriting V4.1's.
    """
    assert _connector(4, v41=False)._page_tp_replication_factor() == 1


def test_the_native_publisher_agrees():
    """The cross-check, since vLLM's spec cannot see through the proxy.

    V4.1's claim rests on two code paths in this tree reaching the same
    conclusion about one model. If the native publisher stops declaring full
    replication, this plugin-side claim is no longer backed by anything and
    has to be re-derived rather than kept.
    """
    import inspect

    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )

    src = inspect.getsource(DeepseekV41MetadataBuilder.get_kv_transfer_tensors)
    assert "tp_replication_factor=tp_size" in src
    assert "native_state_tp_replication_factor=tp_size" in src


def test_the_geometry_takes_no_tensor_parallelism():
    """The plugin side's own evidence, not borrowed from the native side.

    A cache with any dimension split across ranks cannot be sized without
    knowing how many ranks there are. `build_v41_pool_geometry` is never told,
    so it cannot be describing a sharded pool. If a TP parameter is ever added
    to it, that premise is gone and this test says so.
    """
    import inspect

    from atom.model_ops.attentions.deepseek_v41.backend import build_v41_pool_geometry

    params = set(inspect.signature(build_v41_pool_geometry).parameters)
    assert not {p for p in params if "tp" in p or "parallel" in p or "rank" in p}
    assert params == {
        "hf_config",
        "block_size",
        "packed",
        "speculative_tokens",
        "index_fp4",
    }
