# SPDX-License-Identifier: MIT
"""CPU dispatch tests for the optional SP4 QuickReduce collective."""

import math
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest
import torch

from atom.distributed import quick_reduce_scatter as quick_rs

# The arch gate imports AITER's chip_info lazily.
chip_info = pytest.importorskip("aiter.jit.utils.chip_info")


def tensor_metadata(shape=(8192, 1536), dtype=torch.bfloat16, **attributes):
    tensor = Mock(spec=torch.Tensor)
    tensor.shape = shape
    tensor.ndim = len(shape)
    tensor.dtype = dtype
    tensor.device = torch.device("cuda:0")
    tensor.is_cuda = True
    tensor.is_contiguous.return_value = True
    tensor.numel.return_value = math.prod(shape)
    tensor.element_size.return_value = torch.empty((), dtype=dtype).element_size()
    tensor.new_empty.return_value = sentinel.shard
    for name, value in attributes.items():
        setattr(tensor, name, value)
    return tensor


@pytest.fixture
def runtime(monkeypatch):
    comm = SimpleNamespace(
        disabled=False,
        world_size=4,
        qr_quant_level=SimpleNamespace(name="INT4"),
        qr_max_size=64 * 1024**2,
        use_fp16_kernels=True,
        _ptr=123,
    )
    group = SimpleNamespace(device_communicator=SimpleNamespace(qr_comm=comm))
    kernel = Mock()
    monkeypatch.setattr(quick_rs, "_qr_reduce_scatter", kernel)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    return SimpleNamespace(comm=comm, group=group, kernel=kernel)


@pytest.mark.parametrize(
    "shape,contiguous,selected",
    [
        ((4, 3 * 1024**2 - 8), True, False),  # Just below 24 MiB.
        ((4, 3 * 1024**2), True, True),
        ((4, 8 * 1024**2), True, True),  # The communicator's maximum.
        ((4, 8 * 1024**2 + 8), True, False),
        ((4, 3 * 1024**2 + 1), True, False),  # Shard is not 16-byte aligned.
        ((3, 4 * 1024**2), True, False),  # Rows do not split across 4 ranks.
        ((4, 3 * 1024**2), False, False),
    ],
)
def test_only_supported_shards_use_the_int4_kernel(
    runtime, shape, contiguous, selected
):
    x = tensor_metadata(shape)
    x.is_contiguous.return_value = contiguous
    result = quick_rs.try_quick_reduce_scatter(x, runtime.group)
    assert result is (sentinel.shard if selected else None)
    assert runtime.kernel.call_count == int(selected)
    if selected:
        x.new_empty.assert_called_once_with((shape[0] // 4, shape[1]))
        runtime.kernel.assert_called_once_with(123, x, sentinel.shard, True)
