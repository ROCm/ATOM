# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the lm_head shard GEMM: the load-time weight layout, the tile by M, the launch, the warm-up."""
import functools

import torch

from amd_kernels.warm_up import entry

from .kernel import MT_MAX, compile_bf16_gemm_n50016_k6144
from .indexed import compile_indexed_bf16_gemm_n50016_k6144
from .large import compile_large_bf16_gemm_n50016_k6144
from .large_indexed import compile_large_indexed_bf16_gemm_n50016_k6144
from .reference import K, N

MAX_M = 2048


def supports(w: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    """Whether ``w`` is the [50016, 6144] bf16 contiguous shard this GEMM (and ``shuffle_weight_``) takes; the GEMM
    has no bias."""
    return bias is None and tuple(w.shape) == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous()


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """The load-time layout [3126 groups, 96 K blocks, 16 rows, 64 K] of the [50016, 6144] shard, as [50016, 6144].

    Bit-exact (a permutation). The caller replaces the parameter with it once, before graph capture, and frees the
    row-major tensor (no second copy); every reader of the weight then calls bf16_gemm_n50016_k6144(shuffled=True).
    """
    assert w.shape == (N, K) and w.dtype == torch.bfloat16
    return w.view(N // 16, 16, K // 64, 64).permute(0, 2, 1, 3).contiguous().view(N, K)


def shuffle_weight_(w: torch.Tensor) -> torch.Tensor:
    """shuffle_weight in place: ``w`` keeps its storage and holds the load-time layout (the temporary is freed)."""
    ws = shuffle_weight(w)
    w.copy_(ws)
    del ws
    return w


def unshuffle_weight(ws: torch.Tensor) -> torch.Tensor:
    """The inverse of shuffle_weight (row-major [50016, 6144])."""
    assert ws.shape == (N, K) and ws.dtype == torch.bfloat16
    return ws.view(N // 16, K // 64, 16, 64).permute(0, 2, 1, 3).contiguous().view(N, K)


def m_tile_fragments(m: int) -> int:
    """16-row fragments per wave. Keep the fewest M tiles, then balance their sizes.

    Four waves cover 1-128 rows; eight waves cover two M halves above that range.
    """
    if m <= 16 * MT_MAX:
        return (m + 15) // 16
    # Keep the fewest M tiles, then balance their fragment count. This
    # avoids padding 448/896 draft rows to 512/1024 compute rows.
    tiles = (m + 255) // 256
    return (m + tiles * 32 - 1) // (tiles * 32)


def m_halves(m: int) -> int:
    return 1 if m <= 16 * MT_MAX else 2


@functools.cache
def _kernel(mt: int, shuf: bool, mh: int, large: bool = False, publish: bool = False):
    """The tile variant: contiguous rows, or (``publish``) the row-indexed GEMM publishing into the exchange."""
    if publish:
        if large:
            return compile_large_indexed_bf16_gemm_n50016_k6144(mt, int(shuf))
        return compile_indexed_bf16_gemm_n50016_k6144(mt, int(shuf), mh)
    if large:
        return compile_large_bf16_gemm_n50016_k6144(mt, int(shuf))
    return compile_bf16_gemm_n50016_k6144(mt, int(shuf), mh)


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _bf16_gemm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, *,
                           shuffled: bool = False, row_indices: torch.Tensor | None = None,
                           publish: tuple | None = None) -> torch.Tensor | None:
    """out [M, 50016] bf16 = x [M, 6144] bf16 @ W^T, fp32 accumulate; W = w (row-major) or shuffle_weight's layout.

    Any M (above 2048 rows: launches of 2048 rows each, the same bits as separate calls). Deterministic, no atomic
    output adds. CUDA-graph safe (no host sync, no allocation when out is given). The first call of each mode
    (layout, contiguous / publish) compiles the mode's variants (``amd_kernels.warm_up``).

    publish = RowAllToAll.slot() of all_to_all_bf16_n50016_ranks4, with row_indices (contiguous device int64 [M],
    each index in [0, len(x)), duplicates valid, re-read on every graph replay; 1 <= M <= the exchange's max rows,
    no out): the rows x[row_indices] go straight into the exchange's send slot in map order and the kernel
    publishes them (no hidden-row gather, no stage copy); the caller then runs RowAllToAll.pull(M, counts, out).
    The same bits as the contiguous call on x[row_indices]. Returns None.
    """
    assert (row_indices is None) == (publish is None), "row_indices and publish go together"
    if publish is not None:
        pool, slot_bytes, counter, flag = publish
        m = row_indices.numel()
        assert out is None and 1 <= m and m * N * 2 <= slot_bytes
        assert counter.dtype == torch.int32 and counter.device == x.device
        assert x.ndim == 2 and x.shape[1] == K and x.dtype == torch.bfloat16 and x.is_contiguous()
        assert w.shape == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous() and w.device == x.device
        assert row_indices.ndim == 1 and row_indices.dtype == torch.int64 and row_indices.is_contiguous()
        assert row_indices.device == x.device and x.numel() * x.element_size() <= 0xFFFFFFFF and m <= MAX_M
        mt, mh = m_tile_fragments(m), m_halves(m)
        rows = 16 * mt * mh
        exe = _kernel(mt, bool(shuffled), mh, m > 192, True)
        _run(exe, x.data_ptr(), w.data_ptr(), pool, row_indices, x.shape[0], m, (m + rows - 1) // rows, counter,
             slot_bytes, flag, torch.cuda.current_stream())
        return None
    m = x.shape[0]
    assert x.ndim == 2 and x.shape[1] == K and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert w.shape == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous()
    if out is None:
        out = torch.empty(m, N, dtype=torch.bfloat16, device=x.device)
    assert out.shape == (m, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    if m > MAX_M:
        for i in range(0, m, MAX_M):
            _bf16_gemm(x[i:i + MAX_M], w, out[i:i + MAX_M], shuffled=shuffled)
        return out
    assert x.device == w.device == out.device
    if m:
        mt, mh = m_tile_fragments(m), m_halves(m)
        rows = 16 * mt * mh
        exe = _kernel(mt, bool(shuffled), mh, m > 192)
        _run(exe, x.data_ptr(), w.data_ptr(), out.data_ptr(), m, (m + rows - 1) // rows, torch.cuda.current_stream())
    return out


def _mode(x, w, out=None, *, shuffled=False, row_indices=None, publish=None):
    """What a call fixes about its kernels besides M (``amd_kernels.warm_up``): the weight layout and the call form."""
    return bool(shuffled), "publish" if publish is not None else "contiguous"


_KEEP: list = []  # the private publish slot of the warm-up call being made


def _inputs(mode, n, x, w, out=None, *, shuffled=False, row_indices=None, publish=None):
    """One warm-up call at ``n`` rows (zeros, row 0). A variant compiled before the warm-up launches even in
    compile-only mode, so a publish call gets a private slot pair, counter and flag, never the caller's exchange."""
    if mode[1] == "contiguous":
        return (torch.zeros(n, K, dtype=torch.bfloat16, device=x.device), w), {"shuffled": shuffled}
    idx = torch.zeros(n, dtype=torch.int64, device=x.device)
    slot_bytes = n * N * 2
    _KEEP[:] = [torch.empty(2 * slot_bytes, dtype=torch.uint8, device=x.device),
                torch.zeros(1, dtype=torch.int32, device=x.device),
                torch.zeros(16, dtype=torch.int32, device=x.device)]
    publish = (_KEEP[0].data_ptr(), slot_bytes, _KEEP[1], _KEEP[2].data_ptr())
    return (x, w), {"shuffled": shuffled, "row_indices": idx, "publish": publish}


# one row count per tile variant: the small tiles, the narrow and wide eight-wave tiles (m_tile_fragments, m_halves,
# large above 192 rows); every other row count up to MAX_M reuses one of these 14
WARMUP_ROWS = [16 * mt for mt in range(1, MT_MAX + 1)] + [32 * mt for mt in range(5, MT_MAX + 1)] + [300, 513]
WARMUP_MODES = ("contiguous", "publish")

bf16_gemm_n50016_k6144 = entry("gemm/bf16_n50016_k6144", mode=_mode, sizes=lambda mode: WARMUP_ROWS,
                               inputs=_inputs)(_bf16_gemm)


def warmup_bf16_gemm_n50016_k6144(device: torch.device, shuffled: bool, mode: str = "contiguous") -> None:
    """Compile the tile variants of one call mode for the caller's weight layout, before graph capture.

    Each mode is its own kernel, so warm up only the modes the caller runs (once per mode):
    contiguous = bf16_gemm_n50016_k6144(x, w), publish = row_indices=..., publish=...
    """
    assert mode in WARMUP_MODES, mode
    w = torch.zeros(N, K, dtype=torch.bfloat16, device=device)
    x = torch.zeros(max(WARMUP_ROWS), K, dtype=torch.bfloat16, device=device)
    indices = torch.arange(x.shape[0], dtype=torch.int64, device=device)
    publish = None
    if mode == "publish":  # a private scratch slot pair, counter and flag (no peer is involved)
        slot_bytes = x.shape[0] * N * 2
        pool = torch.empty(2 * slot_bytes, dtype=torch.uint8, device=device)
        counter = torch.zeros(1, dtype=torch.int32, device=device)
        flag = torch.zeros(16, dtype=torch.int32, device=device)
        publish = (pool.data_ptr(), slot_bytes, counter, flag.data_ptr())
    for m in WARMUP_ROWS:
        if mode == "contiguous":
            bf16_gemm_n50016_k6144(x[:m], w, shuffled=shuffled)
        else:
            bf16_gemm_n50016_k6144(x, w, shuffled=shuffled, row_indices=indices[:m], publish=publish)
    torch.cuda.synchronize(device)
