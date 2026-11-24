# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared H6144 I768 expert: MXFP8 activations, weights and MFMA at every M."""
from dataclasses import dataclass
import torch
from .shared_quant import shared_quant


# the one-launch MXFP4 + MXFP8 input quant wins at these rows only in the reduced (summed) mode
SHARED_QUANT_REDUCED_ONLY = (4032, 4096)
# first row of the combined routed / shared launch (below it the small-tile host folds the shared tiles in)
SHARED_PERSISTENT_FROM = 449

# One side stream per device (``shared_parallel_for``), shared by every layer's SharedWeights: torch's caching
# allocator keeps one block pool per stream, and per-layer streams hold ~20 GiB of shared-expert outputs per rank
# under DBO until the allocator retries (sync + free every segment).
_SIDE_STREAMS = {}


def side_stream(device):
    """The shared expert's side stream of ``device`` (created on first use)."""
    device = torch.device(device)
    index = torch.cuda.current_device() if device.index is None else device.index
    stream = _SIDE_STREAMS.get(index)
    if stream is None:
        stream = _SIDE_STREAMS[index] = torch.cuda.Stream(device=torch.device("cuda", index))
    return stream


# Inside a DBO microbatch the shared expert stays on the caller's stream: DBO already overlaps one microbatch's
# compute with the other's collectives (serial, the 32768-token DBO steps took a stable 466-472 ms). A server passes
# its query with ``set_dbo_query`` (vLLM: ``vllm.v1.worker.ubatching.dbo_enabled``); without one ``DBO`` answers.
DBO = False
_dbo_query = None


def set_dbo_query(query) -> None:
    """``query()`` -> whether the current call runs inside a DBO microbatch; None returns to ``DBO``."""
    global _dbo_query
    _dbo_query = query


def _in_dbo() -> bool:
    return bool(_dbo_query()) if _dbo_query is not None else DBO


@dataclass
class SharedWeights:
    """The shared expert's MXFP8 weights and scales in the kernels' layout, and the device's side stream."""

    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    stream: object

    @classmethod
    def from_row_major(cls, w13, s13, w2, s2):
        """From row-major e4m3 [N, K] weights and e8m0 [N, K / 32] scales (gate rows, then up rows in ``w13``)."""
        assert w13.shape == (1536,6144) and w2.shape == (6144,768)
        assert s13.shape == (1536,192) and s2.shape == (6144,24)
        # A bit-exact permutation. The returned object holds only these layouts.
        v13 = w13.view(torch.uint8).reshape(96,16,96,4,16).permute(0,2,3,1,4).contiguous()
        b13 = s13.view(torch.uint8).reshape(48,2,16,24,2,4).permute(0,3,5,2,4,1).contiguous()
        v2 = w2.view(torch.uint8).reshape(384,16,12,4,16).permute(0,2,3,1,4).contiguous()
        b2 = s2.view(torch.uint8).reshape(192,2,16,3,2,4).permute(0,3,5,2,4,1).contiguous()
        return cls(v13,b13,v2,b2,side_stream(w13.device))


def shared_expert(x, w, *, x_quant=None, scales_tiled=False, stream=None):
    """The shared expert of bf16 ``x`` [n, 6144] (``shared_dense``, from 96 rows) -> bf16 [n, 6144]; below 96 rows the
    shared tiles run folded into the routed small-tile launches.

    Args:
        x_quant: ``shared_quant(x)``'s (codes, scales) if already computed.
        stream: launch there instead of on the current stream; the buffers still come from the current stream's
            allocator pool and are recorded on ``stream`` (``shared_expert_parallel``).
    """
    n = x.shape[0]
    if not shared_dense_for(n):
        raise ValueError(f"the standalone shared expert takes >= 96 rows, got {n}")
    q, s = shared_quant(x, stream=stream) if x_quant is None else x_quant
    from .shared_dense import shared_dense
    return shared_dense(q, s, w, config=shared_dense_config_for(n), scales_tiled=scales_tiled, stream=stream)


def _record(stream, *tensors):
    """Records the current stream's ``tensors`` on ``stream``, so their blocks wait for its work before reuse."""
    if stream is not None and stream != torch.cuda.current_stream():
        for t in tensors:
            t.record_stream(stream)


def shared_expert_parallel(x, w, current, **kwargs):
    """``shared_expert`` on ``w.stream`` (the device's side stream) after the work queued on ``current`` so far.

    Every buffer comes from ``current``'s allocator pool (recorded on the side stream), so the allocator caches nothing
    under the side stream. The caller joins with ``current.wait_stream(w.stream)`` before it reads the result."""
    side = w.stream
    side.wait_stream(current)
    _record(side, x, *(kwargs.get("x_quant") or ()))
    return shared_expert(x, w, stream=side, **kwargs)


def shared_folded_for(n):
    """Whether the shared tiles lead both routed 16 / 32-row GEMM launches (through 448 rows; from ~480 rows the
    folded tiles' repeated weight reads cost as much as the second stream)."""
    return n <= 448


# second stream vs serial: -2.1..-6.6% at 16385-28672 rows, -0.5..-1.5% to 30720, noise at 31744 / 32768
SHARED_PARALLEL_MAX = 30720


def shared_parallel_for(n):
    """Whether the shared expert runs on the side stream (96..``SHARED_PARALLEL_MAX`` rows, not inside DBO)."""
    return 96 <= n <= SHARED_PARALLEL_MAX and not _in_dbo()


def shared_dual_quant_for(n):
    # the MXFP4 + MXFP8 input quant in one launch on these rows (the same bits as two)
    return (n <= 1024 or SHARED_QUANT_REDUCED_ONLY[0] <= n <= SHARED_QUANT_REDUCED_ONLY[1]
            or 7425 <= n <= 8192
            or 10240 <= n <= 10880 or 13568 <= n <= 32768)


def shared_dense_for(n):
    return n >= 96


def shared_dense_config_for(n):
    if n <= 512:
        return (2,2),(2,2)
    if n <= 1024:
        return (2,4),(4,4)
    return (4,4),(4,4)


def shared_in_persistent_for(n):
    """Whether routed and shared GEMMs share one launch (the BM128 host) on BF16-entry rows: -2.7..-5.3% in graphs
    at 449-1024 rows vs the small-tile host with the shared expert on the second stream."""
    # the adaptive 3584-4096 tier did not pass the entry's noise gate
    return SHARED_PERSISTENT_FROM<=n<=3583
