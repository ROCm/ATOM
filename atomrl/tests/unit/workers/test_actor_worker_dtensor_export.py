# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FSDP2 yields DTensor shards to weight sync; the export must write whole tensors."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from atomrl.workers.actor_worker import AtomActorWorker

_WORLD = 2
_SHAPES = {
    "model.embed_tokens.weight": (6, 4),
    "model.norm.weight": (8,),
}


def _full(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(len(name))
    return torch.randn(shape, generator=generator)


class _ShardedEngine:
    def __init__(self, mesh) -> None:
        self._mesh = mesh

    def get_per_tensor_param(self):
        from torch.distributed.tensor import Shard, distribute_tensor

        def params():
            for name, shape in _SHAPES.items():
                yield name, distribute_tensor(
                    _full(name, shape), self._mesh, [Shard(0)]
                )

        return params(), None


def _export_rank(rank: int, store_path: str, sync_dir: str) -> None:
    # A file store, not a port picked up front: the port can be taken again
    # before rank 0 binds it, and the test then fails at rendezvous.
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=_WORLD,
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh

        worker = SimpleNamespace(
            _engine=_ShardedEngine(init_device_mesh("cpu", (_WORLD,))),
            rank=rank,
            _ensure_weight_http_server=lambda _output_dir: "",
        )
        AtomActorWorker.export_state_dict_safetensors(worker, sync_dir)
    finally:
        dist.destroy_process_group()


def test_export_writes_whole_tensors_from_fsdp2_shards(tmp_path) -> None:
    store = tmp_path / "store"
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    mp.spawn(_export_rank, args=(str(store), str(sync_dir)), nprocs=_WORLD, join=True)

    from safetensors.torch import load_file

    loaded: dict[str, torch.Tensor] = {}
    for path in sorted(sync_dir.glob("model-*.safetensors")):
        loaded.update(load_file(str(path)))
    assert set(loaded) == set(_SHAPES)
    for name, shape in _SHAPES.items():
        torch.testing.assert_close(loaded[name], _full(name, shape))
