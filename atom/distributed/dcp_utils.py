# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode Context Parallel (DCP) distributed-access helpers (ATOM native mode).

Thin wrappers around the DCP world size (from the ATOM config) and the DCP
process group (from aiter's parallel state), mirroring ``pcp_utils.py``. This
keeps the ``getattr(config, "decode_context_parallel_size", 1)`` + ``get_dcp_group``
boilerplate out of the attention / metadata-builder ``__init__``s.

Scope: the DCP world-size / group wrappers are ATOM native (server) mode only —
the vLLM plugin resolves those from ``vllm.distributed`` instead. The
platform-capability query ``dcp_persistent_supported()`` is the one exception:
it is arch-based (not distributed-access) and shared by both native and plugin.

The DCP compute / communication primitives (``cp_lse_ag_out_rs``, ``reorg_kvcache``,
``dcp_gather_compressed_kv``, ...) live in ``atom.model_ops.dcp_ops``; this module is
only the distributed-access layer.
"""

from atom.config import get_current_atom_config
from atom.utils import envs


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


def dcp_persistent_supported() -> bool:
    """Whether DCP decode can run in *persistent* mode on this GPU.

    Persistent DCP needs an lse-emitting ASM decode kernel (per-token
    ``return_lse`` for the cross-rank merge); only gfx950 ships those — gfx942
    has no bf16 lse persistent kernel — so DCP must stay non-persistent
    elsewhere (lse then comes from the triton stage2 reduce). Platform-capability
    query shared by native and the vLLM plugin; cache the result once per
    ``__init__`` to avoid a per-forward ``get_gfx()`` (graph-break).
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
    """Whether a DCP decode will reach ``mla_decode_fwd`` in persistent mode.

    The live decision is made per step in ``_forward_decode``; this mirrors the
    parts of it that are already settled at construction time, because the
    gathered head width has to be fixed there (it sizes the persistent work
    descriptors as well as the kernel's nhead). Sparse MLA under DCP is
    persistent only when the caller rebuilds work/reduce metadata after each
    full indexer layer compacts its rank-local top-k. Only gfx950 ships the
    lse-emitting persistent kernel DCP needs, and persistent mode wants page
    size 1. The one remaining runtime gate, ``dpa_persistent_supported``, is
    unconditionally true, so nothing here can claim persistent mode that the
    step then refuses.

    ``dcp_persistent_supported`` is taken as an argument rather than queried
    here, the way ``should_use_persistent_mode`` takes it: callers already cache
    it to keep ``get_gfx()`` off the per-forward path.

    Lives here, not in ``atom.model_ops.attention_mla``: dependency-free
    (only ``atom.utils.envs``), so it stays importable, and testable,
    without triton/aiter, and so ``mla_dcp_sparse_prefill_is_persistent``
    below can share this body instead of duplicating it.
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

    Mirrors the gate ``_forward_prefill_mla`` applies per forward and is the
    single source the gathered pad width is derived from -- the two must move
    together, or a path runs one way while its width was padded for the other.

    A thin ``is_sparse=True`` call into ``mla_dcp_decode_is_persistent``, not
    a second copy of its body: decode's ``is_sparse and not
    sparse_metadata_rebuild`` guard is exactly this call site's ``not
    sparse_metadata_rebuild`` once ``is_sparse`` is pinned true, and neither
    function has ever been gated on KV cache dtype -- the work-metadata
    buffers are allocated and filled for the layer's real dtype regardless
    (`get_mla_metadata_info_v1`/`get_mla_metadata_v1` take `dtype_q`/`dtype_kv`
    unconditionally), and persistent vs non-persistent agree to bf16 rounding.
    `_forward_prefill_mla`'s `use_work_meta` has the matching `dcp_world_size
    <= 1` arm for symmetry, but it is unreachable for bf16 today: that call
    site only reaches `use_work_meta` at all when `use_decode_kernel` is true,
    and `use_decode_kernel` is `kv_cache_dtype.startswith("fp8") or return_lse`
    with `return_lse` never passed at the non-DCP call site.

    This is the only producer `_forward_prefill_mla`'s per-forward assert
    checks against; that assert independently re-derives the same condition
    inline rather than calling this function, specifically so drift between
    the two spellings is still catchable -- collapsing this function into
    decode's does not touch that.
    """
    return mla_dcp_decode_is_persistent(
        True,
        dcp_world_size,
        dcp_persistent_supported,
        sparse_metadata_rebuild=sparse_metadata_rebuild,
    )


def dcp_prefill_merge_bf16_ok() -> bool:
    """Whether the DCP sparse-prefill partial merge may accumulate in bf16.

    ``cp_lse_ag_out_rs`` accumulates in the tensor dtype: the LSE math is fp32,
    but the per-rank corrected output is stored bf16 and the W-way
    ``reduce_scatter`` then sums in bf16. Over 78 layers x thousands of prefill
    rows that rounding is measurable on *some* GPUs and not others:

        gfx942  dcp8 nshot=20 full gsm8k  bf16 merge 0.9166-0.9174
                                          fp32 merge 0.9522-0.9598   (-3.5pp)
                (both KV dtypes drop by the same amount, so the amplifier is the
                platform, not fp8 KV)
        gfx950  dcp8 nshot=200 full gsm8k (ctx~32k, fp8 KV -- the harshest case
                available: ~10x the prefill rows of the gfx942 run)
                                          bf16 merge 0.9575
                                          fp32 merge 0.9575           (identical)

    So the fp32 merge is a platform-specific fix, not a universal one, and it is
    not free: it doubles this collective's bytes and cost ~18% end-to-end wall
    clock on the gfx950 200-shot run (1:48:35 -> 1:28:54).

    """
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx950"
