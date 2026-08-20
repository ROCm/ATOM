# SPDX-License-Identifier: MIT

import pytest

from atom.model_ops.v4_indexer_config import (
    resolve_v4_index_cache_dtype,
    v4_fp4_indexer_supported,
)


@pytest.mark.parametrize("gfx", ["gfx950", "gfx1250", "future_gfx"])
def test_v4_indexer_defaults_to_fp4_off_gfx942(gfx):
    assert resolve_v4_index_cache_dtype(None, gfx=gfx) == "fp4"
    assert v4_fp4_indexer_supported(gfx)


def test_v4_indexer_defaults_to_fp8_on_gfx942():
    assert resolve_v4_index_cache_dtype(None, gfx="gfx942") == "fp8"
    assert not v4_fp4_indexer_supported("gfx942")


def test_v4_indexer_defaults_to_fp8_when_integration_lacks_fp4_layout():
    assert (
        resolve_v4_index_cache_dtype(None, gfx=None, allow_fp4_default=False) == "fp8"
    )


@pytest.mark.parametrize("requested", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("gfx", ["gfx942", "gfx950"])
def test_v4_indexer_preserves_explicit_override(requested, gfx):
    assert resolve_v4_index_cache_dtype(requested, gfx=gfx) == requested
