# SPDX-License-Identifier: MIT
"""CPU dispatch tests for the optional SP4 QuickReduce collective."""

import math
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest
import torch

from atom.distributed import quick_reduce_scatter as quick_rs


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
    loader = Mock(return_value=SimpleNamespace(qr_reduce_scatter=kernel))
    properties = Mock(
        return_value=SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
    )
    monkeypatch.setattr(quick_rs, "import_module", loader)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    return SimpleNamespace(
        comm=comm, group=group, kernel=kernel, loader=loader, properties=properties
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cast", [False, True])
def test_supported_tensor_uses_existing_int4_communicator(runtime, dtype, cast):
    runtime.comm.use_fp16_kernels = cast
    x = tensor_metadata(dtype=dtype)

    result = quick_rs.try_quick_reduce_scatter(x, runtime.group)

    assert result is sentinel.shard
    x.new_empty.assert_called_once_with((2048, 1536))
    runtime.kernel.assert_called_once_with(123, x, sentinel.shard, cast)


@pytest.mark.parametrize(
    "shape,selected",
    [
        ((4, 3 * 1024**2 - 8), False),  # Just below 24 MiB.
        ((4, 3 * 1024**2), True),
        ((4, 8 * 1024**2), True),  # The communicator's maximum.
        ((4, 8 * 1024**2 + 8), False),
        ((4, 3 * 1024**2 + 1), False),  # Shard is not 16-byte aligned.
    ],
)
def test_payload_limits_keep_small_or_unsupported_shards_on_fallback(
    runtime, shape, selected
):
    x = tensor_metadata(shape)
    result = quick_rs.try_quick_reduce_scatter(x, runtime.group)
    assert result is (sentinel.shard if selected else None)
    assert runtime.kernel.call_count == int(selected)
    assert runtime.loader.call_count == int(selected)


@pytest.mark.parametrize("case", ["missing", "disabled", "none", "fp", "two_ranks"])
def test_unavailable_or_non_int4_communicator_does_not_load_kernel(runtime, case):
    if case == "missing":
        runtime.group.device_communicator.qr_comm = None
    elif case == "disabled":
        runtime.comm.disabled = True
    elif case in {"none", "fp"}:
        runtime.comm.qr_quant_level.name = case.upper()
    else:
        runtime.comm.world_size = 2

    assert quick_rs.try_quick_reduce_scatter(tensor_metadata(), runtime.group) is None
    runtime.loader.assert_not_called()
    runtime.properties.assert_not_called()


@pytest.mark.parametrize("case", ["cpu", "float32", "scalar", "uneven", "strided"])
def test_incompatible_tensor_keeps_normal_collective(runtime, case):
    x = tensor_metadata()
    if case == "cpu":
        x.is_cuda = False
    elif case == "float32":
        x.dtype = torch.float32
    elif case == "scalar":
        x = tensor_metadata(())
    elif case == "uneven":
        x = tensor_metadata((3, 4 * 1024**2))
    else:
        x.is_contiguous.return_value = False

    assert quick_rs.try_quick_reduce_scatter(x, runtime.group) is None
    runtime.loader.assert_not_called()
    runtime.properties.assert_not_called()


@pytest.mark.parametrize(
    "properties", [SimpleNamespace(gcnArchName="gfx942"), SimpleNamespace()]
)
def test_other_architectures_keep_normal_collective(runtime, properties):
    runtime.properties.return_value = properties
    assert quick_rs.try_quick_reduce_scatter(tensor_metadata(), runtime.group) is None
    runtime.loader.assert_not_called()


@pytest.mark.parametrize(
    "name", ["aiter", "aiter.ops", "aiter.ops.quick_reduce_scatter"]
)
def test_missing_companion_module_keeps_normal_collective(runtime, name):
    runtime.loader.side_effect = ModuleNotFoundError(name=name)
    x = tensor_metadata()
    assert quick_rs.try_quick_reduce_scatter(x, runtime.group) is None
    runtime.kernel.assert_not_called()
    x.new_empty.assert_not_called()


def test_missing_companion_entrypoint_keeps_normal_collective(runtime):
    runtime.loader.return_value = SimpleNamespace()
    assert quick_rs.try_quick_reduce_scatter(tensor_metadata(), runtime.group) is None
    runtime.kernel.assert_not_called()


def test_broken_companion_dependency_is_reported(runtime):
    runtime.loader.side_effect = ModuleNotFoundError(
        "missing dependency", name="dependency"
    )
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        quick_rs.try_quick_reduce_scatter(tensor_metadata(), runtime.group)


def test_kernel_build_or_execution_failure_is_reported(runtime):
    runtime.kernel.side_effect = RuntimeError("kernel failure")
    with pytest.raises(RuntimeError, match="kernel failure"):
        quick_rs.try_quick_reduce_scatter(tensor_metadata(), runtime.group)
