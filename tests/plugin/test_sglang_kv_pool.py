import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from atom.plugin.sglang.kv_pool import (
    get_sglang_token_to_kv_pool,
    maybe_get_sglang_kv_pools,
)


def _package(name):
    mod = ModuleType(name)
    mod.__path__ = []
    return mod


def test_sglang_kv_pool_prefers_forward_batch_attributes():
    token_pool = object()
    req_pool = object()
    forward_batch = SimpleNamespace(
        token_to_kv_pool=token_pool,
        req_to_token_pool=req_pool,
    )

    assert maybe_get_sglang_kv_pools(forward_batch) == (token_pool, req_pool)
    assert get_sglang_token_to_kv_pool(forward_batch) is token_pool


def test_sglang_kv_pool_falls_back_to_attention_backend_and_patches_batch():
    token_pool = object()
    req_pool = object()
    forward_batch = SimpleNamespace()
    forward_context_mod = ModuleType("sglang.srt.model_executor.forward_context")
    forward_context_mod.get_attn_backend = lambda: SimpleNamespace(
        token_to_kv_pool=token_pool,
        req_to_token_pool=req_pool,
    )

    fake_modules = {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.model_executor": _package("sglang.srt.model_executor"),
        "sglang.srt.model_executor.forward_context": forward_context_mod,
    }

    with patch.dict(sys.modules, fake_modules):
        assert maybe_get_sglang_kv_pools(forward_batch) == (token_pool, req_pool)

    assert forward_batch.token_to_kv_pool is token_pool
    assert forward_batch.req_to_token_pool is req_pool


def test_sglang_kv_pool_raises_clear_error_when_unavailable():
    with pytest.raises(RuntimeError, match="token_to_kv_pool"):
        get_sglang_token_to_kv_pool(SimpleNamespace(), caller="test caller")
