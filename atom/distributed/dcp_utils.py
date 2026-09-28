# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode Context Parallel (DCP) distributed-access helpers (ATOM native mode).

Wraps the DCP world size (ATOM config) and process group (aiter parallel
state), mirroring ``pcp_utils.py``.

Scope: the world-size / group wrappers are ATOM native (server) mode only —
the vLLM plugin resolves those from ``vllm.distributed`` instead.
``dcp_persistent_supported()`` is the one exception (arch-based, shared by
both). DCP compute / communication primitives live in ``atom.model_ops.dcp_ops``;
this module is only the distributed-access layer.
"""

from atom.config import get_current_atom_config
from atom.utils import envs


def get_dcp_world_size() -> int:
    """DCP world size from the global ATOM config (1 = DCP disabled).

    Before the global config exists (dist-env init, ``BlockManager``/scheduler
    construction), read ``config.decode_context_parallel_size`` off the local
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


def dcp_persistent_supported() -> bool:
    """Whether DCP decode can run in *persistent* mode on this GPU.

    Needs an lse-emitting ASM decode kernel; only gfx950 ships one (gfx942
    falls back to the triton stage2 reduce instead). Cache the result once
    per ``__init__`` — calling this per-forward would graph-break on
    ``get_gfx()``.
    """
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx950"


def mla_dcp_decode_is_persistent(
    is_sparse: bool,
    dcp_world_size: int,
    dcp_persistent_supported: bool,
    *,
    sparse_metadata_rebuild: bool = False,
) -> bool:
    """Whether a DCP decode reaches ``mla_decode_fwd`` in persistent mode.

    Settled at construction time (mirrors the live per-step decision in
    ``_forward_decode``) because the gathered head width must be fixed before
    forward. Sparse MLA needs the caller to rebuild work/reduce metadata per
    full indexer layer first; only gfx950 supports persistent mode at all.

    Lives here, not in ``atom.model_ops.attention_mla``: dependency-free, so
    it stays importable without triton/aiter, and
    ``mla_dcp_sparse_prefill_is_persistent`` below can share it.
    """
    if dcp_world_size <= 1 or (is_sparse and not sparse_metadata_rebuild):
        return False
    return dcp_persistent_supported and envs.ATOM_MLA_PAGE_SIZE <= 1


def mla_dcp_sparse_prefill_is_persistent(
    dcp_world_size: int,
    dcp_persistent_supported: bool,
    *,
    sparse_metadata_rebuild: bool = False,
) -> bool:
    """Whether a DCP sparse prefill reaches ``mla_decode_fwd`` in persistent mode.

    A thin ``is_sparse=True`` call into ``mla_dcp_decode_is_persistent``, not
    a second copy: neither is gated on KV cache dtype, since the work-metadata
    buffers are allocated for the layer's real dtype either way.

    This is the single source `_forward_prefill_mla` derives its gathered pad
    width from -- its per-forward assert independently re-derives the same
    condition inline (on purpose, so drift is still catchable) rather than
    calling this function; keep the two in sync.
    """
    return mla_dcp_decode_is_persistent(
        True,
        dcp_world_size,
        dcp_persistent_supported,
        sparse_metadata_rebuild=sparse_metadata_rebuild,
    )


def dcp_prefill_merge_bf16_ok() -> bool:
    """Whether the DCP sparse-prefill partial merge may accumulate in bf16.

    ``cp_lse_ag_out_rs``'s ``reduce_scatter`` sums in the tensor dtype (bf16),
    and that rounding is measurable on *some* GPUs and not others:

        gfx942  dcp8 nshot=20 full gsm8k: bf16 0.9166-0.9174, fp32 0.9522-0.9598 (-3.5pp)
        gfx950  dcp8 nshot=200 full gsm8k, fp8 KV: bf16 0.9575, fp32 0.9575 (identical)

    So the fp32 merge is a platform-specific fix, not a universal one, and not
    free: it doubles this collective's bytes, costing ~18% end-to-end wall
    clock on the gfx950 200-shot run.
    """
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx950"
