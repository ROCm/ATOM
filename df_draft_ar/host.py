# SPDX-License-Identifier: Apache-2.0
"""DraftArNormComm: Bf16ArNormComm (amd-kernels fd233ef) whose quantized output is ATOM's draft linear input."""
import functools

import torch

from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144 import common as C
from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144.host import Bf16ArNormComm, OWNER_MIN_ROWS, rows_per_group
from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144.ipc import _f32_bits, _run_compiled

from .kernel import compile_allreduce_gemma_bf16_mxfp8


@functools.cache
def _kernel(rpw, quantized, owner, rsq):
    # quantized -> row-major codes + row-major scales (ATOM rule); bf16 out -> row-major bf16
    return compile_allreduce_gemma_bf16_mxfp8(rpw, False, quantized, owner, srm=quantized, use_rsq=rsq)


class DraftArNormComm(Bf16ArNormComm):
    """``rsq``: per output kind, use v_rsq_f32 (Triton's tl.rsqrt) instead of 1 / sqrt (the upstream rule)."""

    rsq_q = True  # K1: q codes match the Triton kernel's more often with v_rsq (tl.rsqrt) than with 1 / sqrt
    rsq_bf16 = False
    # owned-residual form from OWNER_MIN_ROWS rows: each rank adds / norms / quantizes its quarter of the rows and the
    # peers copy the codes; the residual out holds the OWNED rows only (the next call of the same row count reads only
    # those), so every call of one chain must use the same form and row count.
    owner = True

    def _launch_atom(self, x, residual, weight, eps, quantized, owner=False):
        if not self.bufs:
            raise RuntimeError("communicator is closed")
        m = x.shape[0]
        if not (0 < m <= C.MAX_ROWS and tuple(x.shape) == (m, C.H)):
            raise ValueError("expected 1..2048 rows of hidden 6144")
        if x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("partial must be contiguous bf16")
        if residual.shape != x.shape or residual.dtype != x.dtype or residual.stride(1) != 1 or residual.stride(0) % 8:
            raise ValueError("residual must be bf16 with a row stride divisible by eight")
        if tuple(weight.shape) != (C.H,) or weight.dtype != x.dtype or not weight.is_contiguous():
            raise ValueError("weight must be contiguous bf16 [6144]")
        if self.capturing and torch.cuda.is_current_stream_capturing():
            xs = self._row(x)
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("graph calls require the communicator capture() context")
            slot = self._eager_call % 2
            self._eager_call += 1
            self.stage[slot, :m].copy_(x)
            xs = self.stage_table[slot].data_ptr()
        out = torch.empty((m, C.H), dtype=torch.float8_e4m3fn if quantized else torch.bfloat16, device=self.device)
        scales = torch.empty((m, C.H // 32), dtype=torch.uint8, device=self.device) if quantized else None
        res = torch.empty_like(x)
        owner = owner and m >= OWNER_MIN_ROWS
        rsq = self.rsq_q if quantized else self.rsq_bf16
        _run_compiled(_kernel(rows_per_group(m), quantized, owner, rsq),
                      m, self.rank, xs, *self.bufs, self.counts.data_ptr(), residual.stride(0),
                      residual.data_ptr(), weight.data_ptr(), out.data_ptr(), scales.data_ptr() if quantized else 0,
                      res.data_ptr(), out.numel() * out.element_size(), scales.numel() if quantized else 0,
                      _f32_bits(eps), torch.cuda.current_stream())
        return (out, scales, res) if quantized else (out, res)

    def ar_add_rmsnorm_atom(self, x, residual, weight, eps):
        """-> (bf16 normed [M, 6144], residual out)."""
        return self._launch_atom(x, residual, weight, eps, False, self.owner)

    def ar_add_rmsnorm_mxfp8_atom(self, x, residual, weight, eps):
        """-> (e4m3 codes [M, 6144] row-major, uint8 E8M0 scales [M, 192] row-major, residual out)."""
        return self._launch_atom(x, residual, weight, eps, True, self.owner)

    def warm_up_atom(self, weight, eps, rows=(8, 64, 448, 896, 1792)):
        """Collective: compile every row tier of both outputs (rows <= 256 share tier 1)."""
        for m in rows:
            x = torch.zeros((m, C.H), dtype=torch.bfloat16, device=self.device)
            self.ar_add_rmsnorm_atom(x, x, weight, eps)
            self.ar_add_rmsnorm_mxfp8_atom(x, x, weight, eps)
