from types import SimpleNamespace

import pytest

from atom.plugin.sglang.models.kv_cache_utils import is_fp8_kv_cache_dtype


@pytest.mark.parametrize(
    "cache_dtype",
    [
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "torch.float8_e4m3fn",
        SimpleNamespace(cache_dtype="fp8_e4m3"),
    ],
)
def test_fp8_kv_cache_dtype_aliases(cache_dtype):
    assert is_fp8_kv_cache_dtype(cache_dtype)


@pytest.mark.parametrize("cache_dtype", ["bf16", "bfloat16", "auto", None])
def test_non_fp8_kv_cache_dtypes(cache_dtype):
    assert not is_fp8_kv_cache_dtype(cache_dtype)
