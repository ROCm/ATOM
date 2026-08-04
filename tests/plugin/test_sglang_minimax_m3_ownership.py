from types import SimpleNamespace

from atom.plugin.sglang.minimax_m3_bridge import (
    maybe_get_minimax_m3_pools_from_sglang_batch,
)
from atom.plugin.sglang.runtime import attention_backend_resolver


def test_graph_batch_uses_current_backend_pools(monkeypatch):
    token_pool = object()
    req_pool = object()
    backend = SimpleNamespace(
        token_to_kv_pool=token_pool,
        req_to_token_pool=req_pool,
    )
    monkeypatch.setattr(
        attention_backend_resolver,
        "_get_current_attention_backend",
        lambda: backend,
    )

    assert maybe_get_minimax_m3_pools_from_sglang_batch(SimpleNamespace()) == (
        token_pool,
        req_pool,
    )
