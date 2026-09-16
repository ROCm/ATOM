# SPDX-License-Identifier: MIT
"""CPU contracts for Dense single-upload staging metadata."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class _CudaTensor(torch.Tensor):
    """CPU storage reporting CUDA properties for wrapper-only validation."""

    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True

    def to(self, *args, **kwargs):
        return self


def _tensor(values, *, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype).as_subclass(_CudaTensor)


@pytest.fixture
def staging(monkeypatch):
    fake_triton = ModuleType("triton")
    fake_language = ModuleType("triton.language")
    fake_triton.__path__ = []
    fake_triton.language = fake_language
    fake_triton.jit = lambda function: function
    fake_triton.cdiv = lambda value, divisor: (value + divisor - 1) // divisor
    source = (
        Path(__file__).parents[1]
        / "atom/kv_transfer/offload/dense/triton_kv_staging.py"
    )
    name = "_dense_bulk_ids_cpu_contract"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "triton", fake_triton)
        imports.setitem(sys.modules, "triton.language", fake_language)
        spec.loader.exec_module(module)

    uploads = []
    launches = []

    def upload(values, *, dtype):
        uploads.append(tuple(values))
        return _tensor(values, dtype=dtype)

    def upload_array(array):
        uploads.append(tuple(int(value) for value in array))
        return _tensor(array.tolist(), dtype=torch.int64)

    class Kernel:
        def __init__(self, direction):
            self.direction = direction

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.direction, grid, args, kwargs))

            return launch

    module.torch = SimpleNamespace(
        Tensor=torch.Tensor,
        device=torch.device,
        int64=torch.int64,
        uint8=torch.uint8,
        tensor=upload,
        from_numpy=upload_array,
    )
    module._pack_chunk_major_kernel = Kernel("pack")
    module._unpack_chunk_major_kernel = Kernel("unpack")
    return SimpleNamespace(module=module, uploads=uploads, launches=launches)


def test_dense_metadata_is_uploaded_once_and_sliced_per_pipeline_group(staging):
    module = staging.module
    segments = [
        _tensor(range(32), dtype=torch.uint8),
        _tensor(range(48), dtype=torch.uint8),
    ]
    prepared = module.prepare_chunk_major_groups(
        segments,
        [2, 3],
        [((2, 1), (2, 0, 1)), ((1,), (1,))],
        torch.device("cuda:0"),
    )

    assert prepared.group_count == 2
    assert prepared.upload_count == 1
    # Two tile tables (one per group geometry) then the single plan metadata.
    # Each table is one upload carrying both columns: jobs then tiles-within-job.
    assert staging.uploads[:2] == [(0, 1, 2, 3, 0, 0, 0, 0), (0, 1, 0, 0)]
    assert staging.uploads[2:] == [
        (
            segments[0].data_ptr(),
            segments[1].data_ptr(),
            2,
            3,
            0,
            2,
            2,
            1,
            0,
            2,
            0,
            10,
            2,
            0,
            1,
            1,
            0,
            0,
            1,
        )
    ]

    device_buf = _tensor(range(15), dtype=torch.uint8)
    module.fused_pack_chunk_major_prepared(prepared, 0, device_buf)
    module.fused_unpack_chunk_major_prepared(prepared, 1, device_buf)

    # One program per 1024-byte tile actually staged, not jobs x widest job.
    assert [call[:2] for call in staging.launches] == [
        ("pack", (4,)),
        ("unpack", (2,)),
    ]
    pack_args = staging.launches[0][2]
    unpack_args = staging.launches[1][2]
    assert pack_args[4].tolist() == [2, 1]
    assert pack_args[5].tolist() == [0, 2]
    assert pack_args[6].tolist() == [0, 10]
    assert pack_args[7].tolist() == [2, 0, 1]
    assert unpack_args[4].tolist() == [1]
    assert unpack_args[7].tolist() == [1]
    assert pack_args[8].tolist() == [0, 1, 2, 3]
    assert pack_args[9].tolist() == [0, 0, 0, 0]
    assert unpack_args[8].tolist() == [0, 1]
    assert unpack_args[9].tolist() == [0, 0]
    for _direction, _grid, args, kwargs in staging.launches:
        for metadata_view in args[1:8]:
            assert (
                metadata_view.untyped_storage().data_ptr()
                == prepared.metadata.untyped_storage().data_ptr()
            )
        assert kwargs == {
            "NUM_SEGMENTS": 2,
            "BLOCK_BYTES": 1024,
            "num_warps": 2,
        }


@pytest.mark.parametrize(
    ("counts", "block_ids"),
    [((2,), (0,)), ((-1,), ()), ((1,), (0, 1))],
)
def test_dense_metadata_rejects_invalid_group_shapes_before_upload(
    staging, counts, block_ids
):
    with pytest.raises(ValueError):
        staging.module.prepare_chunk_major_groups(
            [_tensor(range(16), dtype=torch.uint8)],
            [4],
            [(counts, block_ids)],
            torch.device("cuda:0"),
        )
    assert not staging.uploads


def test_dense_tile_table_sizes_each_job_by_its_own_bytes(staging):
    module = staging.module
    # One chunk of four blocks over a wide segment (4 KiB) and a narrow one
    # (16 B): the wide job wants four tiles, the narrow job one.  A rectangular
    # grid would have launched both at four.
    tile_job, tile_pos = module._tile_table((4,), (1024, 4), torch.device("cuda:0"))
    assert tile_job.tolist() == [0, 0, 0, 0, 1]
    assert tile_pos.tolist() == [0, 1, 2, 3, 0]
    assert len(staging.uploads) == 1

    # A repeat is served from the memo, so steady-state transfers upload nothing
    # for the table.
    again = module._tile_table((4,), (1024, 4), torch.device("cuda:0"))
    assert again[0] is tile_job and again[1] is tile_pos
    assert len(staging.uploads) == 1


def test_dense_tile_table_skips_jobs_with_no_bytes(staging):
    module = staging.module
    # A zero-block chunk owns no bytes in any segment, so it gets no programs at
    # all; the rectangular grid launched it and masked every lane off.
    tile_job, tile_pos = module._tile_table(
        [0, 3], [1024, 2048], torch.device("cuda:0")
    )
    assert tile_job.tolist() == [2, 2, 2, 3, 3, 3, 3, 3, 3]
    assert tile_pos.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5]


def test_dense_tile_table_cache_is_bounded(staging):
    module = staging.module
    for extra in range(module._TILE_TABLE_CACHE_SIZE + 4):
        module._tile_table((1,), (1024 * (extra + 1),), torch.device("cuda:0"))
    assert len(module._TILE_TABLE_CACHE) == module._TILE_TABLE_CACHE_SIZE
