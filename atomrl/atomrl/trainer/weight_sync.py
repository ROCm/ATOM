# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Parameter synchronization between Trainer and Rollouter.

Provides two backends:

1. **Filesystem (safetensors)** — simple, works across process boundaries
   and even across nodes via shared NFS.  Used as the default.
2. **NCCL broadcast** — lower latency for same-node setups where trainer
   and rollouter share an NCCL process group.

The filesystem path is preferred for AtomRL because the Rollouter runs
vLLM in a subprocess that does *not* share the trainer's NCCL group.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class FilesystemWeightSync:
    """Sync weights via safetensors files on a shared filesystem.

    The Trainer calls :meth:`push` after each sync point to write a new
    weight snapshot.  The Rollouter calls :meth:`pull` before waking the
    inference engine to get the latest snapshot path.

    A monotonically increasing ``version`` counter is written alongside
    the weights so the Rollouter can detect staleness.
    """

    def __init__(self, sync_dir: str, keep_versions: int = 2) -> None:
        self._sync_dir = Path(sync_dir)
        self._sync_dir.mkdir(parents=True, exist_ok=True)
        self._keep_versions = keep_versions
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def push(
        self,
        state_dict: dict[str, torch.Tensor],
        version: int | None = None,
    ) -> str:
        """Write a weight snapshot and return its directory path."""
        if version is not None:
            self._version = version
        else:
            self._version += 1

        ver_dir = self._sync_dir / f"v{self._version}"
        ver_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            from safetensors.torch import save_file

            cpu_tensors = {k: v.contiguous().cpu() for k, v in state_dict.items()}
            save_file(cpu_tensors, str(ver_dir / "model.safetensors"))
        except (OSError, RuntimeError):
            torch.save(
                {k: v.cpu() for k, v in state_dict.items()},
                str(ver_dir / "model_weights.pt"),
            )

        (ver_dir / "version.txt").write_text(str(self._version))
        elapsed = time.time() - t0
        logger.info(
            "WeightSync.push: v%d, %d params, %.1fs",
            self._version,
            len(state_dict),
            elapsed,
        )

        self._cleanup_old_versions()
        return str(ver_dir)

    def latest_path(self) -> str | None:
        """Return the path of the latest weight snapshot, or None."""
        ver_dir = self._sync_dir / f"v{self._version}"
        if ver_dir.exists():
            return str(ver_dir)
        return None

    def latest_version(self) -> int:
        """Read the latest version from disk."""
        max_v = 0
        for d in self._sync_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    v = int(d.name[1:])
                    max_v = max(max_v, v)
                except ValueError:
                    pass
        return max_v

    def _cleanup_old_versions(self) -> None:
        """Remove old snapshots beyond ``keep_versions``."""
        versions = sorted(
            (int(d.name[1:]), d)
            for d in self._sync_dir.iterdir()
            if d.is_dir() and d.name.startswith("v")
        )
        while len(versions) > self._keep_versions:
            _, old_dir = versions.pop(0)
            shutil.rmtree(old_dir, ignore_errors=True)
