# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode Context Parallel (DCP) distributed-access helpers (ATOM native mode).

Thin wrappers around the DCP world size (from the ATOM config) and the DCP
process group (from aiter's parallel state), mirroring ``pcp_utils.py``. This
keeps the ``getattr(config, "decode_context_parallel_size", 1)`` + ``get_dcp_group``
boilerplate out of the attention / metadata-builder ``__init__``s.

Scope: ATOM native (server) mode only. The vLLM plugin resolves its DCP world
size / group from ``vllm.distributed`` instead and does not use this module.

The DCP compute / communication primitives (``cp_lse_ag_out_rs``, ``reorg_kvcache``,
``dcp_gather_compressed_kv``, ...) live in ``atom.model_ops.dcp_ops``; this module is
only the distributed-access layer.
"""

from atom.config import get_current_atom_config


def get_dcp_world_size() -> int:
    """DCP world size from the current global ATOM config (1 = DCP disabled).

    For call sites that run before the global config context is established (dist-env
    init, ``BlockManager``/scheduler construction) — where ``get_current_atom_config()``
    asserts-not-None — read ``config.decode_context_parallel_size`` off the local
    config object directly instead of calling this.
    """
    return get_current_atom_config().decode_context_parallel_size


def dcp_is_enabled() -> bool:
    """True when Decode Context Parallel is active (world size > 1)."""
    return get_dcp_world_size() > 1


def get_dcp_group():
    """The DCP process group (aiter parallel state). Only valid when DCP is enabled."""
    from aiter.dist.parallel_state import get_dcp_group as _get_dcp_group

    return _get_dcp_group()


def get_dcp_rank() -> int:
    """This rank's position within the DCP group (0 when DCP is disabled)."""
    return get_dcp_group().rank_in_group if dcp_is_enabled() else 0
