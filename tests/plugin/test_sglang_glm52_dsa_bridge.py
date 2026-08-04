import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _package(name):
    mod = ModuleType(name)
    mod.__path__ = []
    return mod


def test_glm52_pool_lookup_falls_back_to_sglang_attention_backend():
    aiter_mod = ModuleType("aiter")
    aiter_mod.dtypes = SimpleNamespace(fp8="fp8", d_dtypes={})
    aiter_mod.get_mla_metadata_info_v1 = lambda *args, **kwargs: None
    aiter_mod.get_mla_metadata_v1 = lambda *args, **kwargs: None

    attention_utils_mod = ModuleType("sglang.srt.layers.attention.utils")
    attention_utils_mod.create_flashinfer_kv_indices_triton = (
        lambda *args, **kwargs: None
    )

    token_pool = object()
    req_pool = object()
    forward_context_mod = ModuleType("sglang.srt.model_executor.forward_context")
    forward_context_mod.get_attn_backend = lambda: SimpleNamespace(
        token_to_kv_pool=token_pool,
        req_to_token_pool=req_pool,
    )
    runtime_mod = _package("atom.plugin.sglang.runtime")
    model_arch_mod = ModuleType("atom.plugin.sglang.runtime.model_arch")
    model_arch_mod.is_glm52_dsa_config = lambda config: True

    fake_modules = {
        "atom.plugin.sglang.runtime": runtime_mod,
        "atom.plugin.sglang.runtime.model_arch": model_arch_mod,
        "aiter": aiter_mod,
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.layers": _package("sglang.srt.layers"),
        "sglang.srt.layers.attention": _package("sglang.srt.layers.attention"),
        "sglang.srt.layers.attention.utils": attention_utils_mod,
        "sglang.srt.model_executor": _package("sglang.srt.model_executor"),
        "sglang.srt.model_executor.forward_context": forward_context_mod,
    }

    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("atom.plugin.sglang.glm52_dsa_bridge", None)
        bridge = importlib.import_module("atom.plugin.sglang.glm52_dsa_bridge")

        assert bridge.maybe_get_glm52_dsa_pools_from_sglang_backend(
            SimpleNamespace()
        ) == (token_pool, req_pool)
