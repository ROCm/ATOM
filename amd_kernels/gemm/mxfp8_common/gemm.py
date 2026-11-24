# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Native gfx950 MXFP8 GEMM with table-selected AITER and LDS-staged kernels: the dispatcher
of every MXFP8 linear (``mxfp8_linear``, ``mxfp8_gemm``, ``mxfp8_gemm_shuffled``,
``mxfp8_gemm_prequant``) and its warm-up.

Both implementations consume the same E4M3 weights and E8M0 per-32 scales.
The AITER implementation is adapted from PR #4254 at revision 1e2e3a5 (its MXFP8 case).

The tuned table is ``mxfp8_gemm_tuned.py``; the switches are constants at their
served default; ``DBO`` / ``set_dbo_query`` stand in for vLLM's ``dbo_enabled()``;
and ``mxfp8_linear`` / ``mxfp8_gemm`` take an optional explicit ``config`` (tests and
benchmarks). ``_select_config`` picks the table row. The kernels and rows of
one shape live in that shape's folder (``_SHAPE_HOSTS``: gemm/mxfp8_n2560_k6144,
mxfp8_n6144_k2048, mxfp8_n6144_k8192).
"""

import csv
import dataclasses
import functools
import importlib
import io
from dataclasses import dataclass

import torch

from .mxfp8_gemm_tuned import TUNED_CSV

MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8


@dataclass(frozen=True)
class _Config:
    tile_m: int
    tile_n: int
    tile_k: int
    waves_per_eu: int
    xcd_swizzle: int
    split_k: int
    kernel: str = "aiter"
    waves_m: int = 0
    waves_n: int = 0
    # decode kernel only: W steps / A blocks in flight ahead of the MFMAs
    la_w: int = 0
    la_a: int = 0
    min_m: int = 0
    w_cm: int | None = None


# The decode-tier kernel (mxfp8_gemm_decode.py) rows: tile_m = 16 * row tiles, tile_n =
# 16 * column tiles, waves_m = K-waves, split_k = K split over workgroups, waves_n =
# scheduling barriers on / off, la_w / la_a = the lookahead pair; the activation goes in
# the weight's (16, 16) layout (mxfp8_e4m3_quantize_shuffled).
_DECODE_KERNEL = "decode"
# the same kernel quantizing the bf16 activation itself (no quantizer launch); a
# pre-quantized activation (mxfp8_gemm) runs such a row as a plain decode row
_DECODE_FQ_KERNEL = "decode_fq"
_DECODE_KERNELS = (_DECODE_KERNEL, _DECODE_FQ_KERNEL)
# the LDS-staged kernel (mxfp8_gemm_staged.py)
_STAGED_KERNEL = "staged_ab"
# rows whose activation is the weight's (16, 16) code layout (the decode-tier rows): only a
# producer that writes that layout (``prequant_code_tile_rows``) or a caller that quantizes
# itself (allow_decode) may select them
_SHUFFLED_A_KERNELS = _DECODE_KERNELS
# the persistent 4-wave kernel (mxfp8_gemm_4wave.py): a 256x256 tile per workgroup of 4
# waves, min(tiles, 256) workgroups; `_nt` = non-temporal output stores (the output
# larger than the 256 MB infinity cache); row-major codes, preshuffled scales, N % 256 == 0
_FOURWAVE_KERNEL = "4wave"
_FOURWAVE_NT_KERNEL = "4wave_nt"
_FOURWAVE_KERNELS = (_FOURWAVE_KERNEL, _FOURWAVE_NT_KERNEL)
# Shapes with kernels of their own (gemm/mxfp8_n<N>_k<K>/host.py): ``select`` (their rows
# ahead of the table), ``LAUNCHERS`` (kernel name -> launch), and optionally
# ``select_prequant``, ``select_deferred``, ``decode_launcher``. The dispatcher imports a
# host on first use.
_SHAPE_HOSTS = {
    (2560, 6144): "amd_kernels.gemm.mxfp8_n2560_k6144.host",
    (6144, 2048): "amd_kernels.gemm.mxfp8_n6144_k2048.host",
    (6144, 8192): "amd_kernels.gemm.mxfp8_n6144_k8192.host",
}


@functools.cache
def _shape_host(n: int, k: int):
    name = _SHAPE_HOSTS.get((n, k))
    return importlib.import_module(name) if name is not None else None


@functools.cache
def _shape_launcher(kernel: str):
    """The launch of a shape folder's kernel, or None for the common kernels."""
    for name in _SHAPE_HOSTS.values():
        launch = importlib.import_module(name).LAUNCHERS.get(kernel)
        if launch is not None:
            return launch
    return None


# the AITER kernel's and the split-K reduce's arguments (pointers and i32, <= 48 bytes)
# preloaded into SGPRs instead of s_load'ed before the first loads
_PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


# Graph-replay tuned on one MI355X with pointer-distinct rotating weights above
# LLC. M is an inclusive upper bound; dtype-specific rows override the "any"
# AITER defaults only within their measured range.
@functools.cache
def _load_tuned_configs() -> dict[
    tuple[int, int, str], tuple[tuple[int, _Config], ...]
]:
    configs: dict[tuple[int, int, str], list[tuple[int, _Config]]] = {}
    # the CSV text of the tuned table (mxfp8_gemm_tuned.py)
    for row in csv.DictReader(io.StringIO(TUNED_CSV)):
        shape = (int(row["N"]), int(row["K"]), row["out_dtype"])
        config = _Config(
            tile_m=int(row["tile_m"]),
            tile_n=int(row["tile_n"]),
            tile_k=int(row["tile_k"]),
            waves_per_eu=int(row["waves_per_eu"]),
            xcd_swizzle=int(row["xcd_swizzle"]),
            split_k=int(row["split_k"]),
            kernel=row["kernel"],
            waves_m=int(row["waves_m"]),
            waves_n=int(row["waves_n"]),
            la_w=int(row.get("la_w") or 0),
            la_a=int(row.get("la_a") or 0),
            min_m=int(row.get("min_m") or 0),
            w_cm=int(row["w_cm"]) if row.get("w_cm") else None,
        )
        if config.w_cm is not None and (
            config.kernel not in _DECODE_KERNELS or config.w_cm not in (0, 1, 2)
        ):
            raise ValueError(f"Invalid decode weight cache modifier: {row}")
        if config.min_m and (
            row["out_dtype"] == "any" or not 0 < config.min_m <= int(row["M"])
        ):
            raise ValueError(f"Invalid dtype-specific MXFP8 GEMM range: {row}")
        if config.kernel not in (
            "aiter",
            _STAGED_KERNEL,
            *_DECODE_KERNELS,
            *_FOURWAVE_KERNELS,
        ):
            raise ValueError(f"Unknown MXFP8 GEMM kernel: {config.kernel}")
        if config.kernel in _FOURWAVE_KERNELS:
            # tile_m / tile_n = the tile (256 / 128 per side, sub-tiles of the 256x256
            # parent); split_k = 1 runs the remainder round of parent tiles as one
            # round of sub-tiles, 0 leaves it
            if not (
                row["out_dtype"] == "bf16"
                and config.tile_m in (128, 256)
                and config.tile_n in (128, 256)
                and config.tile_k == 128
                and config.waves_m == 2
                and config.waves_n == 2
                and config.split_k in (0, 1)
                and shape[0] % 256 == 0
                and shape[1] % 256 == 0
            ):
                raise ValueError(f"Unsupported 4-wave MXFP8 GEMM config: {row}")
        if config.kernel in _DECODE_KERNELS:
            steps = shape[1] // 128 // (config.split_k * config.waves_m)
            fold = _decode_fold(config)
            if not (
                row["out_dtype"] == "bf16"
                and (config.tile_m % 16 == 0 or fold > 1)
                and steps % fold == 0
                and config.tile_n % 16 == 0
                and shape[0] % config.tile_n == 0
                and 1 <= config.waves_m <= 8
                and config.waves_n in (0, 1)
                and config.split_k >= 1
                and shape[1] % (256 * config.split_k * config.waves_m) == 0
                and (
                    # la 0 / 0: the deepest lookahead that fits the registers
                    config.la_w == config.la_a == 0
                    or (
                        1 <= config.la_w <= steps
                        and 1
                        <= config.la_a
                        <= steps * max(1, config.tile_m // 16) // fold
                    )
                )
            ):
                raise ValueError(f"Unsupported decode MXFP8 GEMM config: {row}")
        if config.kernel == _STAGED_KERNEL:
            tiles = (64, 128) if config.waves_n == 2 else (128, 256)
            if not (
                row["out_dtype"] in ("bf16", "fp16")
                and config.waves_m == 2
                and config.waves_n in (2, 4)
                and config.tile_m in tiles
                and config.tile_n in tiles
                and config.tile_k == 128
                and config.split_k > 0
                and shape[0] % config.tile_n == 0
                and shape[0] >= config.tile_n
                and shape[1] % (256 * config.split_k) == 0
            ):
                raise ValueError(f"Unsupported staged MXFP8 GEMM config: {row}")
        configs.setdefault(shape, []).append((int(row["M"]), config))
    return {
        shape: tuple(sorted(shape_configs, key=lambda row: row[0]))
        for shape, shape_configs in configs.items()
    }


def tuned_shapes(out_dtype: torch.dtype = torch.bfloat16) -> frozenset[tuple[int, int]]:
    """The (N, K) weight shapes whose table has rows tuned for ``out_dtype`` (bf16 or
    fp16 output). A caller that switches its linears to this entry by shape switches
    these: any other shape runs only the AITER default rows or the generic fallback
    tiles, which were never measured against other kernels."""
    name = {torch.bfloat16: "bf16", torch.float16: "fp16"}[out_dtype]
    return frozenset((n, k) for n, k, d in _load_tuned_configs() if d == name)


def _select_config(
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype = torch.bfloat16,
    allow_decode: bool = True,
) -> _Config:
    """The decode-tier rows read the activation in the weight's (16, 16) layout. A caller
    that quantizes itself (mxfp8_linear) or a producer that writes that layout
    (``prequant_code_tile_rows``: the fused norm + quant) gets them; a caller that brings
    row-major codes would pay a host-side shuffle of ~5-7 us, more than the kernel saves,
    so it asks with ``allow_decode=False`` and gets the "any" (AITER) row where the
    dtype-specific row is a decode row."""
    host = _shape_host(n, k)
    if host is not None:
        config = host.select(m, out_dtype, allow_decode)
        if config is not None:
            return config
    configs = _load_tuned_configs()
    dtype_name = "bf16" if out_dtype == torch.bfloat16 else "fp16"
    tuned_configs = configs.get((n, k, dtype_name))
    if tuned_configs is not None and m <= tuned_configs[-1][0]:
        for tuned_m, config in tuned_configs:
            if m <= tuned_m:
                if m >= config.min_m and (
                    allow_decode or config.kernel not in _SHUFFLED_A_KERNELS
                ):
                    return config
                break
    # Preserve the AITER defaults outside the dtype-specific measured ranges.
    tuned_configs = configs.get((n, k, "any"))
    if tuned_configs is not None:
        for tuned_m, config in tuned_configs:
            if m <= tuned_m:
                return config
        return tuned_configs[-1][1]

    # Fallback configs for untuned (N, K) shapes.
    if m <= 16:
        config = _Config(32, 64, 128, 0, 0, 1)
    elif m <= 256:
        config = _Config(64, 64, 128, 0, 0, 1)
    elif m <= 1024:
        config = _Config(128, 128, 128, 2, 4, 1)
    else:
        config = _Config(128, 256, 128, 2, 4, 1)
    # The kernel requires N to be divisible by tile_n.
    tile_n = config.tile_n
    while n % tile_n:
        tile_n //= 2
    split_k = 1
    if m <= 256:
        # Use split-K to fill the 256 MI355X CUs at small M.
        workgroups = ((m + config.tile_m - 1) // config.tile_m) * (n // tile_n)
        needed = (256 + workgroups - 1) // workgroups
        for candidate in range(needed, min(32, k // 256) + 1):
            if k % (candidate * 256) == 0:
                split_k = candidate
                break
    return _Config(
        config.tile_m,
        tile_n,
        config.tile_k,
        config.waves_per_eu,
        config.xcd_swizzle,
        split_k,
    )


# The first _run_compiled compiles and executes; warmup must cover capture shapes.
@functools.cache
def _gemm_launcher(kernel: str, _constexprs):
    import flydsl.compiler as flyc

    if kernel == _STAGED_KERNEL:
        from .mxfp8_gemm_staged import launch_gemm
        hints = None
    elif kernel == "aiter":
        from .mxfp8_gemm_kernel import launch_gemm
        hints = _PRELOAD_HINTS
    else:
        raise ValueError(f"Unknown MXFP8 GEMM kernel: {kernel}")

    exe = flyc.jit(launch_gemm.func)
    if hints is not None:
        exe.compile_hints = hints
    return exe


@functools.cache
def _fourwave_launcher(
    k: int,
    xcd_swizzle: bool,
    nt: bool,
    tiles: tuple,
    grid_max: int = 256,
):
    from .mxfp8_gemm_4wave import compile_mxfp8_gemm_4w

    return compile_mxfp8_gemm_4w(
        K=k, use_xcd_remap=xcd_swizzle, store_nt=nt, tiles=tiles, grid_max=grid_max
    )


# Inside a DBO microbatch the other microbatch's compute and collectives share the GPU.
# The persistent 4-wave launch (one workgroup per CU, each with a fixed share of the
# tiles) then waits for its slowest CU: next to the sequence-parallel collectives a
# [32768 x 6144] x [6144 x 2560] GEMM slows +75%, as a plain grid (one workgroup per
# work item) +47%; alone the plain grid costs +3%.
_PLAIN_GRID_MAX = 1 << 20


# Every 4-wave launch asks vLLM's ``vllm.v1.worker.ubatching.dbo_enabled()`` (true
# while a step runs as two microbatches; the TP4 deployments split prefill steps of >= 8192
# tokens, so a microbatch has >= 4096 rows). A server passes that query with
# ``set_dbo_query``; without one, ``DBO`` answers (tests and benchmarks set it).
DBO = False
_dbo_query = None


def set_dbo_query(query) -> None:
    """``query()`` -> whether the current launch runs inside a DBO microbatch (vLLM:
    ``vllm.v1.worker.ubatching.dbo_enabled``); None returns to ``DBO``."""
    global _dbo_query
    _dbo_query = query


def _fourwave_grid_max() -> int:
    dbo_enabled = _dbo_query if _dbo_query is not None else lambda: DBO

    return _PLAIN_GRID_MAX if dbo_enabled() else 256


def _fourwave_plan(m: int, n: int, config: _Config) -> list[tuple[tuple, int, int]]:
    """The launches of a 4-wave row: (tiles, tile_off, n_work) over the 256x256 parent
    grid's index space. The row's tile runs everything; with split_k = 1 and the 256x256
    tile, a remainder of R <= 128 parent tiles (a round with idle CUs) is run instead as
    one round of sub-tiles: 128x128 for R <= 64, 128x256 for R <= 128 (measured: qkv 8192
    127 -> 109 us, 16384 205 -> 195; the 128x256 half beats 256x128 by 1-2%, a larger
    remainder is cheaper as the plain round; uniform smaller tiles for the whole GEMM
    lose, their loop pays 1.5-2x the L2 traffic per MFMA)."""
    parents = ((m + 255) // 256) * ((n + 255) // 256)
    tiles = (config.tile_m // 64, config.tile_n // 64)
    split = (256 // config.tile_m) * (256 // config.tile_n)
    r = parents % 256
    if tiles == (4, 4) and config.split_k == 1 and parents > 256 and 0 < r <= 128:
        tail = (2, 2) if r <= 64 else (2, 4)
        tail_split = (256 // (64 * tail[0])) * (256 // (64 * tail[1]))
        # the sub-tile launch deals its parents out by XCD, 8 at a time (mxfp8_gemm_4wave
        # tile_coords); the parents beyond the grid are dropped
        n_tail = (r + 7) // 8 * 8 * tail_split if config.xcd_swizzle else r * tail_split
        return [((4, 4), 0, parents - r), (tail, parents - r, n_tail)]
    if split > 1 and config.xcd_swizzle:
        parents = (parents + 7) // 8 * 8
    return [(tiles, 0, parents * split)]


def _launch_fourwave(config, activation, activation_scale, weight, weight_scale, out):
    """The persistent 4-wave kernel on the whole output (M any: rows beyond M read zero
    and their stores are dropped by the buffer bounds; N % 256 == 0 by the row check)."""
    from amd_kernels.common.tensor_shim import _run_compiled

    m, k = activation.shape
    n = out.shape[1]
    i8 = lambda t: t.view(torch.int8).view(-1)  # noqa: E731
    u8 = lambda t: t.view(torch.uint8).view(-1)  # noqa: E731
    stream = torch.cuda.current_stream()
    grid_max = _fourwave_grid_max()
    for tiles, tile_off, n_work in _fourwave_plan(m, n, config):
        _run_compiled(
            _fourwave_launcher(k, config.xcd_swizzle != 0, config.kernel == _FOURWAVE_NT_KERNEL, tiles, grid_max),
            i8(activation), i8(weight), out.view(-1), u8(activation_scale), u8(weight_scale),
            m, n, tile_off, n_work, stream,
        )  # fmt: skip


def _decode_fold(config: _Config) -> int:
    """decode_fq rows with a tile of 8 / 4 rows run the folded kernel (M <= 8 / 4)."""
    if config.kernel == _DECODE_FQ_KERNEL and config.tile_m in (4, 8):
        return 16 // config.tile_m
    return 1


@functools.cache
def _decode_launcher(n, k, config: _Config, a_bf16: bool):
    host = _shape_host(n, k)
    if host is not None and hasattr(host, "decode_launcher"):
        exe = host.decode_launcher(config, a_bf16)
        if exe is not None:
            return exe

    from .mxfp8_gemm_decode import compile_mxfp8_gemm_decode

    fold = _decode_fold(config) if a_bf16 else 1
    return compile_mxfp8_gemm_decode(
        N=n,
        K=k,
        RT=max(1, config.tile_m // 16),
        NI=config.tile_n // 16,
        KW=config.waves_m,
        KS=config.split_k,
        LA=(config.la_w, config.la_a)
        if config.la_w
        else None,  # 0: the deepest that fits
        sched_barriers=bool(config.waves_n),
        A_BF16=a_bf16,
        FOLD=fold,
        W_CM=config.w_cm,
    )


@functools.cache
def _dummy_i32(device):
    """Stands in for pointers a launch does not read (the staged kernel's work items and
    slabs, the activation scales of a row that quantizes the bf16 activation itself)."""
    return torch.zeros(4, dtype=torch.int32, device=device)


@functools.cache
def _reduce_launcher(_constexprs):
    import flydsl.compiler as flyc

    from .mxfp8_gemm_kernel import launch_splitk_reduce

    exe = flyc.jit(launch_splitk_reduce.func)
    exe.compile_hints = _PRELOAD_HINTS
    return exe


class SkPartials:
    """Split-K partials consumed by a fused epilogue or standalone reduce."""

    __slots__ = ("tmp", "split_k", "m", "n", "out_dtype")

    def __init__(self, tmp, split_k, m, n, out_dtype):
        self.tmp = tmp
        self.split_k = split_k
        self.m = m
        self.n = n
        self.out_dtype = out_dtype

    def materialize(self) -> torch.Tensor:
        from amd_kernels.common.tensor_shim import _run_compiled, ptr_arg

        name = "bf16" if self.out_dtype == torch.bfloat16 else "fp16"
        out = torch.empty(
            (self.m, self.n), dtype=self.out_dtype, device=self.tmp.device
        )
        _run_compiled(
            _reduce_launcher((self.split_k, name)),
            ptr_arg(self.tmp),
            ptr_arg(out),
            (self.m * self.n) // 2,
            self.m * self.n,
            torch.cuda.current_stream(),
            self.split_k,
            name,
        )
        return out


def preshuffle_mxfp8_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight

    return (
        shuffle_weight(weight, layout=(16, 16)),
        shuffle_scale_a16w4(weight_scale, 1, False),
    )


def preshuffle_mxfp8_weight_(weight: torch.Tensor, weight_scale: torch.Tensor) -> bool:
    """``preshuffle_mxfp8_weight`` in place (the tensors keep their storage). False, and
    nothing changed, when the weight is not an E4M3 [N, K] with E8M0 [N, K / 32] scales of
    a shape the table has rows for (bf16 out)."""
    n, k = weight.shape
    if (
        weight.dtype != MXFP8_VALUE_DTYPE
        or tuple(weight_scale.shape) != (n, k // 32)
        or (n, k) not in tuned_shapes(torch.bfloat16)
    ):
        return False
    w_sh, s_sh = preshuffle_mxfp8_weight(weight, weight_scale)
    weight.copy_(w_sh)
    weight_scale.copy_(s_sh)
    return True


def mxfp8_e4m3_quantize_aiter(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-major E4M3 codes and E8M0 scales in AITER's shuffled layout, from the FlyDSL
    streaming kernel (mxfp8_quant.py; the Triton quantizer's bytes)."""
    from . import mxfp8_quant

    m, k = x.shape
    if k % 256 != 0:
        raise ValueError(f"MXFP8 quantization requires K % 256 == 0, got {k}")
    x = x.contiguous()
    quant = torch.empty((m, k), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    padded_m = (m + 31) // 32 * 32
    scale = torch.empty((padded_m, k // 32), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    mxfp8_quant.quantize(x, quant, scale)
    return quant, scale


mxfp8_e4m3_quantize_flydsl = mxfp8_e4m3_quantize_aiter  # the name the benchmarks use


def mxfp8_e4m3_quantize_shuffled(
    x: torch.Tensor, block_m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize for the decode-tier GEMM: codes in the weight's (16, 16) preshuffled
    layout, rows padded to ``block_m`` (the padding rows get zero codes, the GEMM masks
    their outputs), scales in AITER's shuffled layout (the FlyDSL kernel)."""
    from . import mxfp8_quant

    m, k = x.shape
    if k % 256 != 0 or block_m % 16 != 0:
        raise ValueError(
            f"needs K % 256 == 0 and 16-row tiles, got K {k}, block {block_m}"
        )
    x = x.contiguous()
    rows = (m + block_m - 1) // block_m * block_m
    quant = torch.empty((rows, k), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scale = torch.empty(
        ((rows + 31) // 32 * 32, k // 32), dtype=MXFP8_SCALE_DTYPE, device=x.device
    )
    mxfp8_quant.quantize(x, quant, scale, shuffled=True)
    return quant, scale


def _preshuffle_activation_scale(scale: torch.Tensor) -> torch.Tensor:
    """Row-major e8m0 activation scales [M, K/32] -> AITER's shuffled layout, the rows padded
    to 32 with 127 (no served caller brings row-major scales)."""
    from aiter.ops.shuffle import shuffle_scale_a16w4

    m, scale_k = scale.shape
    padded_m = (m + 31) // 32 * 32
    if padded_m != m:
        padded = torch.full(
            (padded_m, scale_k), 127, dtype=scale.dtype, device=scale.device
        )
        padded[:m].copy_(scale)
        scale = padded
    return shuffle_scale_a16w4(scale, 1, False)


def _validate_oproj_output(out, activation, weight, weight_scale, dtype, m, n, k, activation_scale=None):
    """Validate a caller-owned contiguous BF16 o_proj arena before any kernel writes it."""
    if (n, k) != (6144, 2048) or dtype != torch.bfloat16:
        raise ValueError("out= supports BF16 o_proj N6144 K2048")
    if out.shape != (m, n) or out.dtype != dtype or out.device != activation.device or not out.is_contiguous():
        raise ValueError("out= must be contiguous [M, 6144] BF16 on the activation device")
    inputs = (activation, weight, weight_scale) + (() if activation_scale is None else (activation_scale,))
    if any(torch._C._overlaps(out, tensor) for tensor in inputs):
        raise ValueError("out= must not overlap the inputs")


def mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    config: _Config | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ W^T`` for a bf16 / fp16 activation and a preshuffled MXFP8 weight: the
    activation quantizer is the one the table's kernel wants (the decode kernel reads
    the codes in the weight's (16, 16) layout, the others row-major). ``config``: run
    this configuration instead of the table's (tests, benchmarks). ``out`` may supply a
    contiguous BF16 o_proj arena (N6144 K2048);
    the returned tensor is that same object. No output copy is made.
    """
    m, k = x.shape
    n = weight.shape[0]
    if m == 0:
        return torch.empty((0, n), dtype=out_dtype, device=x.device) if out is None else out
    # The GEMM validates the prepared activation below. Preserve the original-input
    # alias check here as well, without repeating all output checks on that path.
    if out is not None and torch._C._overlaps(out, x):
        raise ValueError("out= must not overlap the inputs")
    explicit = config
    if config is None:
        config = _select_config(m, n, k, out_dtype)
    if config.kernel == _DECODE_FQ_KERNEL:
        if out is not None:
            _validate_oproj_output(out, x, weight, weight_scale, out_dtype, m, n, k)
        out = torch.empty((m, n), dtype=out_dtype, device=x.device) if out is None else out
        _launch_decode(
            config,
            x.contiguous(),
            _dummy_i32(x.device),
            weight,
            weight_scale,
            out,
            True,
        )
        return out
    if config.kernel == _DECODE_KERNEL:
        if out is not None:
            _validate_oproj_output(out, x, weight, weight_scale, out_dtype, m, n, k)
        activation, activation_scale = mxfp8_e4m3_quantize_shuffled(x, config.tile_m)
        out = torch.empty((m, n), dtype=out_dtype, device=x.device) if out is None else out
        _launch_decode(config, activation, activation_scale, weight, weight_scale, out)
        return out
    activation, activation_scale = mxfp8_e4m3_quantize_aiter(x)
    return mxfp8_gemm(
        activation, activation_scale, weight, weight_scale, out_dtype,
        activation_scale_preshuffled=True, config=explicit, out=out,
    )  # fmt: skip


def mxfp8_gemm(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    activation_scale_preshuffled: bool = False,
    *,
    config: _Config | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``A @ W^T`` for a row-major MXFP8 activation (codes [M, K], scales [M, K/32] or
    preshuffled) and a preshuffled weight. Row-major codes never take a decode row (those
    read the (16, 16) layout: ``mxfp8_gemm_shuffled``, or ``mxfp8_linear`` from bf16).
    ``out`` may supply a contiguous BF16 o_proj arena (N6144 K2048); the returned tensor is
    that same object. ``config``: run this configuration instead of the table's (tests,
    benchmarks)."""
    m, k = activation.shape
    n = weight.shape[0]
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"Unsupported output dtype: {out_dtype}")
    if out is not None:
        _validate_oproj_output(out, activation, weight, weight_scale, out_dtype, m, n, k, activation_scale)
    if config is None:
        config = _select_config(m, n, k, out_dtype, allow_decode=False)
    if config.kernel in _SHUFFLED_A_KERNELS:
        raise ValueError(f"{config.kernel} rows read (16, 16) codes, not row-major ones")
    if not activation_scale_preshuffled:
        activation_scale = _preshuffle_activation_scale(activation_scale)
    out = torch.empty((m, n), dtype=out_dtype, device=activation.device) if out is None else out
    launch = _shape_launcher(config.kernel)
    if launch is not None:
        launch(config, activation, activation_scale, weight, weight_scale, out)
        return out
    if config.kernel in _FOURWAVE_KERNELS:
        _launch_fourwave(
            config, activation, activation_scale, weight, weight_scale, out
        )
        return out
    if n % config.tile_n:
        raise ValueError(f"MXFP8 GEMM tile_n {config.tile_n} does not divide N {n}")
    _launch_tiled(config, activation, activation_scale, weight, weight_scale, out)
    return out


def _select_prequant_config(m: int, n: int, k: int, out_dtype: torch.dtype) -> _Config:
    """A codes input can avoid the K-wave reduction paid by tiny bf16 rows.

    BF16 callers retain their rows, including the in-kernel quantizers. The code layout is
    still (16,16); at 17-32 rows both the 16-row tile and the 32-row tile
    allocate exactly 32 rows, so producers' buffer sizes and bytes agree.
    """
    host = _shape_host(n, k)
    if host is not None and hasattr(host, "select_prequant"):
        config = host.select_prequant(m, out_dtype)
        if config is not None:
            return config
    return _select_config(m, n, k, out_dtype)


def _deferred_qkv_config(m: int, n: int, k: int, config: _Config) -> _Config:
    """QKV partials consumed by the fused norm/RoPE/insert reduction: the shape folder's
    rows (``select_deferred``; TP4 QKV), else ``config``."""
    host = _shape_host(n, k)
    if host is not None and hasattr(host, "select_deferred"):
        return host.select_deferred(m, config)
    return config


def prequant_code_tile_rows(
    m: int, n: int, k: int, out_dtype: torch.dtype = torch.bfloat16
) -> int:
    """For a producer of pre-quantized activations (the fused norm + quant): 0 when the
    table's row for this call reads row-major codes, else the row tile (a multiple of 16)
    of its decode row. The producer then writes the codes in the weight's (16, 16)
    layout, rows padded to the tile, and calls ``mxfp8_gemm_shuffled``: the decode row
    without the host-side shuffle ``mxfp8_gemm`` would need."""
    if k % 256 or out_dtype != torch.bfloat16:
        return 0
    config = _select_prequant_config(m, n, k, out_dtype)
    if config.kernel not in _SHUFFLED_A_KERNELS:
        return 0
    return max(16, config.tile_m)  # a folded fused row: the plain kernel, 16-row tile


def prequant_buffers(
    m: int, n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Uninitialized (codes uint8, e8m0 scales, tile) for a producer that writes this GEMM's
    pre-quantized input of ``m`` rows: codes in ``prequant_code_tile_rows``' layout, rows
    padded to the tile (tile 0: row-major), scales in AITER's layout. The GEMM masks the
    padding rows' outputs."""
    tile = prequant_code_tile_rows(m, n, k, torch.bfloat16)
    pad = max(tile, 16)
    padded = -(-m // pad) * pad
    codes = torch.empty((padded, k), dtype=torch.uint8, device=device)
    scales = torch.empty((-(-padded // 32) * 32, k // 32), dtype=torch.uint8, device=device)
    return codes, scales, tile


def mxfp8_gemm_shuffled(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    m: int,
    defer_reduce: bool = False,
    *,
    config: _Config | None = None,
) -> torch.Tensor | SkPartials:
    """``A @ W^T`` on a decode row for codes already in the (16, 16) layout
    ([ceil(m / tile) * tile, K], the tile from ``prequant_code_tile_rows``) and scales in
    AITER's shuffled layout ([ceil(rows / 32) * 32, K / 32]). With ``defer_reduce``,
    TP4 QKV also tunes the split for a fused norm/RoPE/insert consumer: return
    ``SkPartials`` where measured faster, keeping the producer's code layout.
    An explicit ``config`` (tests/benchmarks) overrides both tables."""
    n, k = weight.shape[0], activation.shape[1]
    held = activation.shape[0]
    deferred = False
    if config is None:
        config = _select_prequant_config(m, n, k, out_dtype)
        if defer_reduce:
            table = config
            config = _deferred_qkv_config(m, n, k, config)
            deferred = config is not table
    if config.kernel not in _SHUFFLED_A_KERNELS:
        raise ValueError(f"no shuffled-A row for M {m}, N {n}, K {k}")
    if config.tile_m < 16:
        config = dataclasses.replace(config, tile_m=16)
    rows = (m + config.tile_m - 1) // config.tile_m * config.tile_m
    # a deferred row reads the first `rows` code rows of the producer's buffer: the (16, 16)
    # layout and the shuffled scales are 16- / 32-row-block major, so a prefix is its codes
    if (
        (held != rows and not (deferred and rows <= held and held % 16 == 0))
        or activation_scale.shape[0] < (rows + 31) // 32 * 32
    ):
        raise ValueError(
            f"shuffled activation {tuple(activation.shape)} / scale "
            f"{tuple(activation_scale.shape)} do not match M {m}, tile {config.tile_m}"
        )
    out = torch.empty((m, n), dtype=out_dtype, device=activation.device)
    result = _launch_decode(
        config,
        activation,
        activation_scale,
        weight,
        weight_scale,
        out,
        defer_reduce=defer_reduce and (m >= 17 or deferred) and config.split_k > 1,
    )
    if result is not None:
        return result
    return out


def mxfp8_gemm_prequant(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    m: int,
) -> torch.Tensor:
    """``A @ W^T`` [m, N] for codes a pre-quantizing producer wrote in the layout
    ``prequant_code_tile_rows`` picks for this call (norm.gemma_add_rmsnorm_h6144
    ``gemma_add_rmsnorm_mxfp8_for``, the all-reduce + norm communicators with
    ``gemm_weight``): the decode row on (16, 16) codes, else ``mxfp8_gemm`` on row-major
    codes; scales in AITER's shuffled layout either way."""
    n, k = weight.shape
    if m == 0:
        return torch.empty((0, n), dtype=out_dtype, device=weight.device)
    if prequant_code_tile_rows(m, n, k, out_dtype):
        return mxfp8_gemm_shuffled(activation, activation_scale, weight, weight_scale, out_dtype, m)
    return mxfp8_gemm(
        activation, activation_scale, weight, weight_scale, out_dtype, activation_scale_preshuffled=True
    )


def _launch_decode(
    config,
    activation,
    activation_scale,
    weight,
    weight_scale,
    out,
    a_bf16=False,
    *,
    defer_reduce=False,
):
    """One launch of the decode kernel over the whole [M, N] output (plus the split-K
    reduce when the row splits K over workgroups); ``a_bf16``: the activation is the bf16
    tensor itself and the kernel quantizes it."""
    from amd_kernels.common.tensor_shim import _run_compiled, ptr_arg

    m, n = out.shape
    k = weight.shape[1]
    exe = _decode_launcher(n, k, config, a_bf16)
    n_mb = (m + config.tile_m - 1) // config.tile_m
    grid = config.split_k * n_mb * (n // config.tile_n)
    stream = torch.cuda.current_stream()
    u8 = lambda t: t.view(torch.uint8).view(-1)  # noqa: E731
    if config.split_k == 1:
        _run_compiled(
            exe, u8(activation), u8(weight), u8(out), u8(activation_scale), u8(weight_scale),
            m, activation_scale.numel() * activation_scale.element_size(), n_mb, grid, stream,
        )  # fmt: skip
        return
    tmp = torch.empty((config.split_k, m, n), dtype=torch.float32, device=out.device)
    _run_compiled(
        exe, u8(activation), u8(weight), u8(tmp), u8(activation_scale), u8(weight_scale),
        m, activation_scale.numel() * activation_scale.element_size(), n_mb, grid, stream,
    )  # fmt: skip
    if defer_reduce:
        return SkPartials(tmp, config.split_k, m, n, out.dtype)
    _run_compiled(
        _reduce_launcher((config.split_k, "bf16")), ptr_arg(tmp), ptr_arg(out),
        (m * n) // 2, m * n, stream, config.split_k, "bf16",
    )  # fmt: skip


def _launch_tiled(config, activation, activation_scale, weight, weight_scale, out):
    """``out = A @ W^T`` on an AITER or LDS-staged row (``config.tile_n`` divides N), with
    the split-K reduce when the row splits K."""
    from amd_kernels.common.tensor_shim import _run_compiled, ptr_arg

    m, k = activation.shape
    n = out.shape[1]
    out_dtype_name = "bf16" if out.dtype == torch.bfloat16 else "fp16"
    compiled_config = (
        n,
        k,
        config.tile_m,
        config.tile_n,
        config.tile_k,
        out_dtype_name,
        config.waves_per_eu,
        config.xcd_swizzle,
        config.split_k,
    )
    stream = torch.cuda.current_stream()
    if config.kernel == _STAGED_KERNEL:
        compiled_config += (config.waves_m, config.waves_n)
        dummy = _dummy_i32(activation.device)
        extra = (ptr_arg(dummy), ptr_arg(dummy), m, n, 0, stream)
    else:
        extra = (m, n, stream)
    gemm_exe = _gemm_launcher(config.kernel, compiled_config)
    if config.split_k == 1:
        _run_compiled(
            gemm_exe,
            ptr_arg(out),
            ptr_arg(activation),
            ptr_arg(weight),
            ptr_arg(activation_scale),
            ptr_arg(weight_scale),
            *extra,
            *compiled_config,
        )
        return
    tmp = torch.empty(
        (config.split_k, m, n), dtype=torch.float32, device=activation.device
    )
    _run_compiled(
        gemm_exe,
        ptr_arg(tmp),
        ptr_arg(activation),
        ptr_arg(weight),
        ptr_arg(activation_scale),
        ptr_arg(weight_scale),
        *extra,
        *compiled_config,
    )
    _run_compiled(
        _reduce_launcher((config.split_k, out_dtype_name)),
        ptr_arg(tmp),
        ptr_arg(out),
        (m * n) // 2,
        m * n,
        stream,
        config.split_k,
        out_dtype_name,
    )


def warmup_mxfp8_gemm_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    max_num_tokens: int,
    out_dtype: torch.dtype,
) -> None:
    """Compile every variant a linear of this preshuffled weight's (N, K) launches for
    1..max_num_tokens rows (the table's rows through every entry, their activation
    quantizers; at K 6144 the fused norm + quant producer). The variants depend on the
    shape, not on the weight's values, so one call per shape is enough: later calls with
    the same key and no larger budget return at once. Frees the warm-up's buffers."""
    n, k = weight.shape
    key = (n, k, weight.device, out_dtype)
    if _warmed_weights.get(key, 0) >= max_num_tokens:
        return
    _warmup_weight(weight, weight_scale, max_num_tokens, out_dtype)
    if k == 6144:  # the fused add + norm + quant that writes this GEMM's codes
        from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import warmup_for_gemm

        warmup_for_gemm(weight.device)
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    _warmed_weights[key] = max_num_tokens


# warmup_mxfp8_gemm_weight's key -> the token budget it compiled for
_warmed_weights: dict[tuple, int] = {}


def _warmup_weight(weight, weight_scale, max_num_tokens, out_dtype) -> None:
    n, k = weight.shape
    representatives = {}
    for m in range(1, max_num_tokens + 1):
        for allow_decode in (True, False):
            config = _select_config(m, n, k, out_dtype, allow_decode)
            prequant = _select_prequant_config(m, n, k, out_dtype)
            key: tuple = (config, prequant, _deferred_qkv_config(m, n, k, prequant))
            if config.kernel in _FOURWAVE_KERNELS:
                # the remainder round's sub-tile kernels are launchers of their own
                key += tuple(tiles for tiles, _, _ in _fourwave_plan(m, n, config))
            representatives.setdefault(key, m)
    for m in representatives.values():
        activation = weight.new_zeros((m, k))
        scales = torch.full(
            ((m + 31) // 32 * 32, k // 32),
            127,
            dtype=weight_scale.dtype,
            device=weight.device,
        )
        mxfp8_gemm(
            activation,
            scales,
            weight,
            weight_scale,
            out_dtype,
            activation_scale_preshuffled=True,
        )
        # the layer path: the row's activation quantizer + the same kernel
        mxfp8_linear(
            weight.new_zeros((m, k), dtype=out_dtype),
            weight,
            weight_scale,
            out_dtype,
        )
        # a producer of (16, 16) codes: the plain decode kernel (also for a fused row)
        tile = prequant_code_tile_rows(m, n, k, out_dtype)
        if tile:
            rows = (m + tile - 1) // tile * tile
            mxfp8_gemm_shuffled(
                weight.new_zeros((rows, k)),
                torch.full(
                    ((rows + 31) // 32 * 32, k // 32), 127,
                    dtype=weight_scale.dtype, device=weight.device,
                ),
                weight, weight_scale, out_dtype, m,
            )  # fmt: skip
            prequant = _select_prequant_config(m, n, k, out_dtype)
            if _deferred_qkv_config(m, n, k, prequant) != prequant:
                mxfp8_gemm_shuffled(
                    weight.new_zeros((rows, k)),
                    torch.full(((rows + 31) // 32 * 32, k // 32), 127,
                               dtype=weight_scale.dtype, device=weight.device),
                    weight, weight_scale, out_dtype, m, defer_reduce=True,
                )
    # the activation quantizer's variants: (K, layout, rows per wave), the row-major
    # codes switch to PREFILL_RPW from PREFILL_ROWS rows
    from .mxfp8_quant import PREFILL_ROWS

    for rows in (1, PREFILL_ROWS):
        if rows > max_num_tokens:
            continue
        x = weight.new_zeros((rows, k), dtype=out_dtype)
        mxfp8_e4m3_quantize_aiter(x)
        mxfp8_e4m3_quantize_shuffled(x, 16)
