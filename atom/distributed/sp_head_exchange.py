# SPDX-License-Identifier: MIT
"""M3 SP4 head exchange directly into the o-projection's final row layout."""

import hashlib
from pathlib import Path

import torch
from aiter.jit.core import AITER_CSRC_DIR, compile_ops

from atom.utils import envs

_CSRC = Path(AITER_CSRC_DIR)
_SOURCES = (_CSRC / "kernels/sp_head_exchange.cu", _CSRC / "pybind/sp_head_exchange_pybind.cu")
_HASH = hashlib.sha256(b"".join(p.read_bytes() for p in (
    *_SOURCES, _CSRC / "include/custom_all_reduce.cuh",
    _CSRC / "include/aiter_tensor.h",
))).hexdigest()[:12]


def _build_args(*args, **kwargs):
    return {"md_name": "module_sp_head_exchange_" + _HASH,
            "srcs": [str(p) for p in _SOURCES]}


@compile_ops("module_custom_all_reduce", fc_name="sp_head_exchange", gen_func=_build_args, develop=True)
def _exchange(handle: int, input: torch.Tensor, output: torch.Tensor,
              registered_buffer: int, registered_bytes: int, stage: bool, blocks: int) -> None: ...


def head_exchange_communicator(x):
    """Runtime gate inside the opaque attention op; leave decode on its AG path."""
    from atom.distributed.ulysses_sp import get_sp_group, get_sp_world_size

    if (not envs.ATOM_SP_HEAD_EXCHANGE or not envs.ATOM_USE_CUSTOM_ALL_GATHER
            or get_sp_world_size() != 4 or x.ndim != 2
            or x.shape[0] < 8192 or x.shape[0] % 4
            or x.shape[1] not in (1024, 2048) or x.dtype != torch.bfloat16
            or not x.is_contiguous()):
        return None
    ca = getattr(getattr(get_sp_group(), "device_communicator", None), "ca_comm", None)
    if ca is None or ca.disabled or getattr(ca, "_pool", None) is None:
        return None
    if not ca.should_custom_ag(x):
        return None
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return ca if get_gfx_runtime() == "gfx950" else None


def exchange_heads(x, ca, *, registered=False):
    """Read just this rank's token slice from each peer and concatenate heads.

    All operations use the communicator's current stream. Registered scratch
    must be produced immediately before this call; end_sync protects its reuse.
    """
    out = torch.empty((x.shape[0] // 4, x.shape[1] * 4), device=x.device, dtype=x.dtype)
    pool = ca._pool["input"]
    graph_registered = (ca._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
                        and ca.enable_register_for_capturing)
    _exchange(ca._ptr, x, out, pool.data_ptr, pool.max_size,
              not (registered or graph_registered), 80)
    return out
