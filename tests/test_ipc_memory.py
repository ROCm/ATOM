# SPDX-License-Identifier: MIT
"""Shared pool allocation restores settings even when allocation fails."""

import pytest
import torch

from atom.utils.ipc_memory import allocate_shared_kv_buffer


@pytest.mark.parametrize(
    "env_name",
    ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"],
)
@pytest.mark.parametrize("fails", [False, True])
def test_native_pool_restores_allocator(monkeypatch, env_name, fails):
    for name in (
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_HIP_ALLOC_CONF",
    ):
        monkeypatch.delenv(name, raising=False)
    conf = "max_split_size_mb:128,expandable_segments:True"
    monkeypatch.setenv(env_name, conf)
    settings = []
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", settings.append)
    result = object()

    def allocate(size, *, dtype, device):
        assert size == 64 and dtype == torch.uint8 and device == "cuda:0"
        assert settings == ["max_split_size_mb:128,expandable_segments:False"]
        if fails:
            raise RuntimeError("allocation failure")
        return result

    monkeypatch.setattr(torch, "zeros", allocate)
    if fails:
        with pytest.raises(RuntimeError, match="allocation failure"):
            allocate_shared_kv_buffer(64, "cuda:0")
    else:
        assert allocate_shared_kv_buffer(64, "cuda:0") is result
    assert settings[-1] == conf


def test_default_allocator_is_not_reconfigured(monkeypatch):
    for name in (
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_HIP_ALLOC_CONF",
    ):
        monkeypatch.delenv(name, raising=False)

    def unexpected(*args):
        pytest.fail("default allocator must not be reconfigured")

    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", unexpected)
    assert allocate_shared_kv_buffer(8, "cpu").tolist() == [0] * 8
