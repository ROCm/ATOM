# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experiment only: dump the MLA latent seen by the NVFP4 path, per request.

Set ATOM_MLA_NVFP4_DUMP_DIR and run eager (a replayed CUDA graph runs no
Python). Records go to ``<dir>/<pid>_<n>.pt`` in chunks:

- ``("prefill", layer, ring_rows, cu_seqlens, kv)``: the roped prefill latent
- ``("decode", step, layer, ring_rows, positions, kv, q)``: one token per row;
  ``q`` (BF16 ``[rows, 128, 576]``) every ``ATOM_MLA_NVFP4_DUMP_Q_EVERY`` steps.

Ring rows identify requests across steps (batch rows do not).
"""

import os

import torch

DUMP_DIR = os.getenv("ATOM_MLA_NVFP4_DUMP_DIR")
Q_EVERY = int(os.getenv("ATOM_MLA_NVFP4_DUMP_Q_EVERY", "64"))
CHUNK_STEPS = 128

_records: list = []
_step = -1
_chunk = 0


def enabled() -> bool:
    return DUMP_DIR is not None and not torch.cuda.is_current_stream_capturing()


def _flush() -> None:
    global _records, _chunk
    if not _records:
        return
    os.makedirs(DUMP_DIR, exist_ok=True)
    torch.save(_records, os.path.join(DUMP_DIR, f"{os.getpid()}_{_chunk:05d}.pt"))
    _records, _chunk = [], _chunk + 1


def prefill(layer, ring_rows, cu_seqlens, kv) -> None:
    _records.append(
        ("prefill", layer, list(map(int, ring_rows)), cu_seqlens.cpu(), kv.cpu())
    )
    if layer == 0:
        _flush()


def decode(layer, ring_rows, positions, kv, q_nope, q_rope) -> None:
    global _step
    if layer == 0:
        _step += 1
        if _step % CHUNK_STEPS == 0:
            _flush()
    q = None
    if _step % Q_EVERY == 0:
        q = torch.cat((q_nope, q_rope), dim=-1).cpu()
    _records.append(
        ("decode", _step, layer, ring_rows.cpu(), positions.cpu(), kv.cpu(), q)
    )


def finish() -> None:
    _flush()
