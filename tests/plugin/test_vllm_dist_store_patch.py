import sys
from types import ModuleType

from atom.plugin.vllm.dist_store_patch import apply_vllm_tcp_store_patch


def test_tcp_store_patch_updates_network_helper_and_loaded_executors(monkeypatch):
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    utils = ModuleType("vllm.utils")
    utils.__path__ = []
    network_utils = ModuleType("vllm.utils.network_utils")
    network_utils.aiter_requires_tcp_store = lambda: False
    utils.network_utils = network_utils
    vllm.utils = utils

    multiproc = ModuleType("vllm.v1.executor.multiproc_executor")
    multiproc.aiter_requires_tcp_store = lambda: False
    uniproc = ModuleType("vllm.v1.executor.uniproc_executor")
    uniproc.aiter_requires_tcp_store = lambda: False

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.network_utils", network_utils)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.executor.multiproc_executor",
        multiproc,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.executor.uniproc_executor",
        uniproc,
    )

    apply_vllm_tcp_store_patch()

    assert network_utils.aiter_requires_tcp_store()
    assert multiproc.aiter_requires_tcp_store()
    assert uniproc.aiter_requires_tcp_store()

    patched_helper = network_utils.aiter_requires_tcp_store
    network_utils.aiter_requires_tcp_store = lambda: False
    multiproc.aiter_requires_tcp_store = lambda: False
    apply_vllm_tcp_store_patch()
    assert network_utils.aiter_requires_tcp_store is patched_helper
    assert multiproc.aiter_requires_tcp_store is patched_helper
