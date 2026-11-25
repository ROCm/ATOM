# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A speculative verify's sampling and epilogue as one recorded graph
(ATOM_HOSTNEXT_POST_GRAPH, default off).

After the verify graph, a decode step's GPU work up to the draft passes is the
presence penalty on the verify logits (`PresencePenaltyStep.apply`, with its
anchor marks), the rejection sampler's verify (`SpecRejectionSampling.verify`:
the sampler's stats and select kernels, ``torch.rand``, the min(1, p/q)
decision kernel) and the fused verify epilogue (`verify_epilogue`). Eagerly
that is ~7 launches and ~330 us of host work a step. Here the same launches are
recorded once per step shape and replayed: ~30 us of host work.

Everything the recording reads is at a fixed address for a given shape (the
verify graph's logits, ``forward_vars``, the drafter's step-0 buffers, the
presence rows in a fixed device buffer, the hand-off rings); what changes from
step to step on the host goes through one small int64 upload before the
replay (`PostVerifyGraphs.scalars`):

* [0] the sampler's Philox counter. Eagerly the Gluon select draws row r's
  uniform at counter ``(r, calls)`` from launch scalars
  (`sampler.fused_topk_topp_sample`); recorded, `_verify_uniform_kernel` makes
  the same uniforms with aiter's own `_philox_uniform` at the device counter
  and the select reads them (``uniform=``). The host counter advances as it
  does eagerly, so eager and recorded calls interleave on one sequence.
* [1:7] the claimed hand-off slots' offsets and sequence numbers
  (`verify_epilogue`'s ``host_dev``).

``torch.rand`` inside a recording draws at the generator's offset at replay
time (PyTorch's graph-safe Philox), as the eager call would. So the tokens,
verdict, marks and staged buffers are the eager chain's, bit for bit; a step
whose inputs moved (a shape not seen, a buffer reallocated) runs the eager
chain. A shape is recorded the second time it is seen (the first runs eagerly,
which compiles the common kernels).
"""

import logging
import os
from collections import Counter

import numpy as np
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

logger = logging.getLogger("atom")

POST_GRAPH = os.environ.get("ATOM_HOSTNEXT_POST_GRAPH", "0") == "1"
# ATOM_HOSTNEXT_POST_GRAPH_CHECK=1 (tests): every replay is also run eagerly
# from the same state and both results compared (see `PostVerifyGraphs.check`).
POST_GRAPH_CHECK = os.environ.get("ATOM_HOSTNEXT_POST_GRAPH_CHECK", "0") == "1"
_RING = 8  # pinned rows for the per-step scalars (async H2D)
_NSCAL = 8

try:  # aiter's Gluon select draws with this function (gfx1250)
    from aiter.ops.triton._gluon_kernels.gfx1250.fusions.topk_topp_sample import (
        _philox_uniform,
    )
except Exception:  # pragma: no cover - other archs: the recorded path is off
    _philox_uniform = None


@gluon.jit
def _verify_uniform_kernel(u_ptr, state_ptr, seed_lo, seed_hi, NW: gl.constexpr):
    """``u[r]`` = the Gluon select's PHILOX draw of row ``r`` at offset
    ``state[0]`` (`_philox_uniform`, the select's own function)."""
    L1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    r = gl.program_id(0)
    offs = gl.arange(0, 32 * NW, layout=L1)
    off = gl.load(state_ptr)
    u = _philox_uniform(r, seed_lo, seed_hi, off & 0xFFFFFFFF, off >> 32, offs)
    gl.store(u_ptr + r + offs * 0, u, mask=offs == 0)


def rs_verify_recordable(rs, logits, metadata, temperatures, top_k, top_ps, gluon_seed, scal):
    """`SpecRejectionSampling.verify` with nothing taken from the host per call:
    with ``gluon_seed`` (the eager call's Gluon select seed), the select's
    uniforms come from `_verify_uniform_kernel` at the counter ``scal[0]``."""
    from aiter.ops.triton.fusions import topk_topp_sample as tts

    from atom.model_ops.sampler import SAMPLER_EPS
    from atom.model_ops.spec_rejection import _EPS, _kb, _rs_verify_kernel

    cu = metadata.cu_num_draft_tokens
    tli = metadata.target_logits_indices
    bli = metadata.bonus_logits_indices
    bs = cu.numel()
    n_v = tli.numel()
    S = int(metadata.num_spec_steps)
    assert bs == rs.active_bs, f"verify of {bs} requests, staged {rs.active_bs}"
    dev = logits.device
    kb_p = _kb(top_k)
    dbg = torch.empty(n_v + bs, 3 * kb_p + 2, dtype=torch.int32, device=dev)
    rows = dict(verify_rows=tli, bonus_rows=bli, cu_num_draft_tokens=cu)
    if gluon_seed is not None:
        n = n_v + bli.numel()
        u_sel = torch.empty(n, dtype=torch.float32, device=dev)
        _verify_uniform_kernel[(n,)](
            u_sel,
            scal,
            gluon_seed & 0xFFFFFFFF,
            (gluon_seed >> 32) & 0xFFFFFFFF,
            NW=1,
            num_warps=1,
        )
        tokens = tts.topk_topp_sample(
            logits,
            temperatures,
            int(top_k),
            top_ps,
            eps=SAMPLER_EPS,
            gluon_select=True,
            uniform=u_sel,
            debug=dbg,
            **rows,
        )
    else:
        tokens = tts.topk_topp_sample(
            logits, temperatures, int(top_k), top_ps, eps=SAMPLER_EPS, debug=dbg, **rows
        )
    lp = max(kb_p, rs.KB_Q)
    u = torch.rand(bs * S * (lp + 1), dtype=torch.float32, device=dev)
    out_v = torch.empty(n_v, dtype=torch.int32, device=dev)
    _rs_verify_kernel[(bs,)](
        dbg,
        rs.records,
        rs.record_tok,
        rs.step_slots,
        cu,
        tli,
        metadata.input_ids,
        temperatures[:bs].float().contiguous(),
        u,
        out_v,
        S,
        int(top_k),
        KB_P=kb_p,
        K_Q=rs.draft_k,
        KB_Q=rs.KB_Q,
        LP=lp,
        S_MAX=S,
        EPS=_EPS,
        num_warps=1,
    )
    return out_v, tokens[n_v:]


def _ptr(t):
    return 0 if t is None else t.data_ptr()


class PostVerifyGraphs:
    """The recordings, by step shape, and the per-step scalar upload."""

    def __init__(self, device):
        self.device = device
        self.scal = torch.zeros(_NSCAL, dtype=torch.int64, device=device)
        self._pin = torch.zeros(_RING, _NSCAL, dtype=torch.int64).pin_memory()
        self._pin_np = self._pin.numpy()
        self._pin_next = 0
        self.graphs: dict = {}
        self.seen: Counter = Counter()
        self.stats: Counter = Counter()
        self.pool = None
        self._logged_moved = False

    def upload(self, values) -> None:
        """This step's scalars into ``scal`` (stream-ordered before the replay)."""
        row = self._pin_next
        self._pin_next = (row + 1) % _RING
        self._pin_np[row, : len(values)] = values
        self.scal.copy_(self._pin[row], non_blocking=True)

    def lookup(self, key, signature):
        """The recording for ``key`` if its inputs are where they were, else None
        (and whether to record now)."""
        ent = self.graphs.get(key)
        if ent is not None:
            if ent["sig"] == signature:
                return ent, False
            self.stats["moved"] += 1
            if not self._logged_moved:
                self._logged_moved = True
                logger.warning(
                    "ATOM_HOSTNEXT_POST_GRAPH: the inputs of %s moved; eager chain", key
                )
            return None, False
        self.seen[key] += 1
        return None, self.seen[key] >= 2

    def record(self, key, signature, body):
        """Record ``body()`` (returns the outputs to keep) for ``key``; the caller
        replays it."""
        from atom.utils.h2d import capturing

        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        cur = torch.cuda.current_stream()
        cap_stream = None if cur == torch.cuda.default_stream() else cur
        with capturing(), torch.cuda.graph(
            graph, self.pool, stream=cap_stream, capture_error_mode="thread_local"
        ):
            outs = body()
        ent = dict(graph=graph, outs=outs, sig=signature)
        self.graphs[key] = ent
        self.stats["capture"] += 1
        logger.info("ATOM_HOSTNEXT_POST_GRAPH: recorded the post-verify chain at %s", key)
        return ent
