from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from atom.plugin.vllm.attention.backend import (
    AiterMhaBackendForVllm,
    AiterMhaFlexibleBlockBackendForVllm,
)
from atom.plugin.vllm.attention.layer_mha import (
    _mha_backend_for_layer,
)


def test_target_mha_keeps_physical_kernel_block_size_16():
    assert AiterMhaBackendForVllm.get_supported_kernel_block_sizes() == [16]
    assert (
        _mha_backend_for_layer(
            47,
            SimpleNamespace(num_hidden_layers=48),
        )
        is AiterMhaBackendForVllm
    )


def test_draft_mha_accepts_hybrid_logical_page_size():
    assert (
        _mha_backend_for_layer(
            48,
            SimpleNamespace(num_hidden_layers=48),
        )
        is AiterMhaFlexibleBlockBackendForVllm
    )
    assert AiterMhaFlexibleBlockBackendForVllm.get_supported_kernel_block_sizes() != [
        16
    ]
