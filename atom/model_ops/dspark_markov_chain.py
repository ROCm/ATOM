# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The DSpark Markov chain of ONE request in one persistent launch (gfx1250).

``DSparkMarkovHead.sample_next`` runs the block's K sequential steps as K pairs
of ``dspark_markov_argmax`` launches, and every step re-reads the whole
``markov_w2`` table (V x 256 bf16, 102 MB for a 200K vocabulary). Here one
launch runs all K steps: aiter's ``markov_argmax_chain``, one workgroup per CU,
each holding its share of W2 in LDS and registers for the whole chain, so later
steps read no W2 from memory; after each step the workgroups exchange their
(max, id) through a workspace and every workgroup reduces them itself.

Numerics: bit-identical to ``dspark_markov_argmax`` -- per tile the same chain
of eight ``v_wmma_f32_16x16x32_bf16`` its Triton stage 1 emits on gfx1250 (the
same operands, the same K split across lanes), ``fp32(base) + acc``, lowest id
on ties.

Served when (``usable``): ONE request (the block's rows are its K positions),
K <= 8, bf16 logits / tables of rank 256, and aiter's
``markov_argmax_chain_supported`` (gfx1250, a vocabulary of whole 16-id tiles
that fits the CUs). Anything else, an aiter without the op, or
``ATOM_DSPARK_FUSED_MARKOV_CHAIN=0`` keeps the two-stage path.

The exchange waits for every workgroup, which holds when the launch has the GPU
to itself. Its waits are bounded so that a workgroup that never becomes
resident (another process or a long kernel on a side stream holding CUs) cannot
hang the step; such a wait sets a bit in an error word and that step's drafts
are not to be trusted. ``check_errors``, called by the draft on the host every
step, reads the word without waiting on the device and raises, rather than
drafting on.
"""

import importlib
import logging
from typing import Any, NamedTuple

import torch

from atom.utils import envs

logger = logging.getLogger("atom")

RANK = 256
MAX_STEPS = 8
_UNSET = object()
_op: Any = _UNSET  # aiter.ops.markov_argmax_chain, or None: off or no op
_failed = False
_calls = 0


class _DeviceState(NamedTuple):
    ws: Any  # aiter MarkovArgmaxChainWorkspace: exchange slots + error word
    err_host: torch.Tensor  # pinned copy of the error word
    err_copied: torch.cuda.Event  # recorded after each copy into `err_host`


_state: dict[int, _DeviceState] = {}


def _aiter_op():
    """aiter's ``markov_argmax_chain`` module, or None when
    ``ATOM_DSPARK_FUSED_MARKOV_CHAIN`` is off or aiter lacks the op; looked
    up once."""
    global _op
    if _op is _UNSET:
        if not envs.ATOM_DSPARK_FUSED_MARKOV_CHAIN:
            op = None
            logger.info(
                "ATOM_DSPARK_FUSED_MARKOV_CHAIN=0: the DSpark Markov chain keeps "
                "the two-stage path"
            )
        else:
            try:
                op = importlib.import_module("aiter.ops.markov_argmax_chain")
            except ImportError:
                op = None
                logger.info(
                    "aiter has no markov_argmax_chain: the DSpark Markov chain "
                    "keeps the two-stage path"
                )
        _op = op
    return _op


def usable(
    base: torch.Tensor, ids: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """``base`` [1, K, V] bf16 (one request's block), ``ids`` [1, K + 1]."""
    if base.dim() != 3 or base.shape[0] != 1 or ids.shape != (1, base.shape[1] + 1):
        return False
    K, V = base.shape[1], base.shape[2]
    if base.device.type != "cuda" or not 1 <= K <= MAX_STEPS:
        return False
    if base.dtype != torch.bfloat16 or base.stride(2) != 1:
        return False
    for w in (w1, w2):
        if (
            w.dtype != torch.bfloat16
            or w.shape != (V, RANK)
            or not w.is_contiguous()
            or w.data_ptr() % 16
        ):
            return False
    if _failed:
        return False
    op = _aiter_op()
    return op is not None and op.markov_argmax_chain_supported(
        V, RANK, K, id_dtype=ids.dtype, device=base.device
    )


def markov_chain_bs1(base, ids, w1, w2) -> "torch.Tensor | None":
    """Drafted ids into ``ids[0, 1:]`` (``ids[0, 0]`` is the anchor) and the
    ``[K, 256]`` W1 rows of the K steps returned; None when not served."""
    global _failed
    if not usable(base, ids, w1, w2):
        return None
    op = _aiter_op()
    dev = base.device
    st = _state.get(dev.index)
    if st is not None:
        return op.markov_argmax_chain(base[0], ids[0], w1, w2, st.ws)
    if torch.cuda.is_current_stream_capturing():
        # The workspace must outlive the graph pool: allocated on an eager call.
        return None
    st = _DeviceState(
        op.markov_argmax_chain_workspace(MAX_STEPS, dev),
        torch.zeros(1, dtype=torch.int32, pin_memory=True),
        torch.cuda.Event(),
    )
    try:
        # The first launch builds aiter's module (JIT) if it is not prebuilt.
        embeds = op.markov_argmax_chain(base[0], ids[0], w1, w2, st.ws)
    except RuntimeError as e:
        _failed = True
        logger.warning(
            "DSpark Markov chain kernel unavailable, keeping the two-stage path: %s", e
        )
        return None
    _state[dev.index] = st
    return embeds


def check_errors(every: int = 512) -> None:
    """Raise if a bounded wait of the one-launch chain gave up.

    Host side, outside any recording. Every ``every`` calls it reads the
    error word as copied the time before (through a pinned buffer and an
    event, so it never waits on the device), then copies it again.
    """
    global _calls
    _calls += 1
    if _calls % every:
        return
    for st in _state.values():
        if st.err_copied.query() and st.err_host.item():
            raise RuntimeError(
                "DSpark Markov chain: a workgroup of the one-launch chain was not "
                f"resident (steps 0x{st.err_host.item():x} waited it out), so those "
                "steps' drafts are not to be trusted. Something else holds CUs of "
                "this GPU; set ATOM_DSPARK_FUSED_MARKOV_CHAIN=0 to sample with "
                "the two-stage path."
            )
        st.err_host.copy_(st.ws.err, non_blocking=True)
        st.err_copied.record()
