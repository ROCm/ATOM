# SPDX-License-Identifier: MIT
"""Restartable harness responses with an identical cache snapshot on every TP rank."""

import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
from contextlib import contextmanager
from pathlib import Path

import aiter
import torch
from aiter.dist.parallel_state import get_tp_group


def evaluation_identity(model, implementation, max_length):
    root = Path(__file__).resolve().parents[3]
    sources = sorted((root / "atom").rglob("*.py")) + sorted(
        Path(__file__).parent.rglob("*.py")
    )
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    aiter_root = Path(aiter.__file__).resolve().parent
    for arguments in (("rev-parse", "HEAD"), ("diff", "HEAD")):
        digest.update(
            subprocess.check_output(["git", "-C", str(aiter_root), *arguments])
        )
    directory = Path(model).resolve()
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    ):
        path = directory / filename
        if path.exists():
            digest.update(filename.encode())
            digest.update(path.read_bytes())
    digest.update(
        (Path(__file__).parent / "fixtures/reference_manifest.json").read_bytes()
    )
    return {
        "model": str(directory),
        "implementation": implementation,
        "max_length": max_length,
        "source_sha256": digest.hexdigest(),
        "tp_size": get_tp_group().world_size,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "custom_all_reduce": False,
        "device": torch.cuda.get_device_name(),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("ATOM_", "AITER_", "HIP_", "ROCBLAS_", "TORCH_"))
        },
    }


def seed_worker_caches(directory, identity, ranks):
    """Only rank zero's committed responses are authoritative after a crash."""
    directory = Path(directory)
    manifest = directory / "identity.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Response cache belongs to a different evaluation build")
    else:
        if (directory / "tp0_rank0.db").exists():
            raise ValueError("Response cache has no evaluation identity")
        manifest.write_text(json.dumps(identity, indent=2))
    source = directory / "tp0_rank0.db"
    for rank in range(1, ranks):
        destination = directory / f"tp{rank}_rank0.db"
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(str(destination) + suffix).unlink(missing_ok=True)
        if source.exists():
            with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
                src.backup(dst)


@contextmanager
def response_cache(model, directory, identity):
    if directory is None:
        yield None
        return
    group = get_tp_group()
    rank = group.rank_in_group
    lock = None
    previous_hook = model.cache_hook
    error = [None]
    try:
        if rank == 0:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                lock = (directory / "active.lock").open("a")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seed_worker_caches(directory, identity, group.world_size)
            except (OSError, ValueError, sqlite3.Error) as exc:
                error[0] = str(exc)
        if group.world_size > 1:
            torch.distributed.broadcast_object_list(
                error, src=group.ranks[0], group=group.cpu_group
            )
        if error[0] is not None:
            raise RuntimeError(error[0])
        # Every TP worker evaluates the same requests; lm_eval sees rank zero
        # on each one. Its data-parallel cache suffix is appended to our prefix.
        yield str(directory / f"tp{rank}")
    finally:
        if model.cache_hook is not previous_hook:
            model.cache_hook.dbdict.close()
            model.set_cache_hook(previous_hook)
        if lock is not None:
            lock.close()
