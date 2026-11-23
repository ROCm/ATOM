# SPDX-License-Identifier: MIT
"""The shared expert's activation and w2's input quant as one launch.

    out = aiter.silu_and_mul(combined, limit)   # BF16 [M, K], HIP
    q, s = quantize_fp8(out)                    # E4M3 [M, K], E8M0 [M, K/32], Triton

becomes `shared_act_quant(combined, limit) -> (q, s)`: the same bytes from one
FlyDSL kernel, without the BF16 buffer in between. A decode step launches a
thousand kernels and one that does almost nothing still costs microseconds, so
this is a launch-count fusion: at 30-78 rows it takes 2.5-3 us off a layer.

Bit-exact, which aiter's `fused_clamp_act_mul` (ATOM_V4_USE_TRITON_FUSION) is
not: the activation is rounded to BF16 before it is quantized, as the buffer
`silu_and_mul` writes is, the amax floor is 1e-10 and the scale is
ceil-pow2(amax / 448), `quantize_fp8`'s rule. The SiLU is the HIP kernel's own
sequence (min / med3 clamp, OCML exp as exp2 of a compensated product, rcp),
under the FP32 flush-to-zero mode HIP kernels run in.

The device code is plugin_poc's `ops/silu_medium` (written for TP4 prefill
rows [1025..4352, 1152]) with the contract widened: BF16 [M, 2K] contiguous,
K a multiple of 32, any M. gfx950 only (v_cvt_pk_bf16_f32, v_cvt_pk_fp8_f32).
"""

import logging

import torch
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.jit.utils.torch_guard import torch_compile_guard

logger = logging.getLogger("atom")

_LOGGED: set[str] = set()


def log_once(key: str, message: str) -> None:
    """One line a process for each launch-count toggle, so a server log shows
    which path its layers took."""
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(message)


def shared_act_quant_enabled(w2, limit: float, wanted: bool) -> bool:
    """Whether a shared expert takes `shared_act_quant` (decided once, at build):
    ``wanted`` (a recipe's ``model_options.shared_act_quant``), the kernel
    package importable, gfx950, a clamp (the kernel bakes one in), and a `w2`
    that reads the row-major group-32 E8M0 pair `quantize_fp8` would hand it."""
    if not wanted:
        return False
    try:
        from ops.silu_medium import dep8_kernel  # noqa: F401
    except ImportError:
        return False
    enabled = (
        w2.native_a8_group_rows is not None
        and float(limit) > 0.0
        and w2.input_size % 32 == 0
        and torch.cuda.is_available()
        and get_gfx_runtime() == "gfx950"
    )
    log_once(
        f"shared_act_quant:{enabled}",
        "dsv41 shared expert act + quant: "
        + (
            "fused, one launch (model_options.shared_act_quant)"
            if enabled
            else "unfused (needs gfx950, a clamp and a group-32 MXFP8 w2)"
        ),
    )
    return enabled


def _kernel_takes(combined: torch.Tensor) -> bool:
    """The tensors the kernel reads; anything else runs the unfused pair."""
    return (
        combined.is_cuda
        and combined.dtype == torch.bfloat16
        and combined.ndim == 2
        and combined.shape[0] > 0
        and combined.shape[1] % 64 == 0
        and combined.is_contiguous()
        and combined.data_ptr() % 16 == 0
        and combined.numel() * 2 < 2**31
    )


def _shared_act_quant_fake(
    combined: torch.Tensor, limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = combined.shape
    device = combined.device
    return (
        torch.empty((rows, width // 2), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((rows, width // 64), dtype=torch.float8_e8m0fnu, device=device),
    )


@torch_compile_guard(gen_fake=_shared_act_quant_fake, mutates_args=[])
def shared_act_quant(
    combined: torch.Tensor, limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`quantize_fp8(silu_and_mul(combined, limit))` of BF16 `combined` =
    [M, gate K | up K]: (E4M3 [M, K], E8M0 [M, K / 32]), one launch. A tensor
    the kernel does not read (`_kernel_takes`) runs that pair as it is, so the
    op answers for every input `shared_act_quant_enabled` lets through."""
    rows, width = combined.shape
    k = width // 2
    if not _kernel_takes(combined):
        from aiter import silu_and_mul

        from atom.model_ops.blockscale import quantize_fp8

        mid = torch.empty((rows, k), dtype=combined.dtype, device=combined.device)
        silu_and_mul(mid, combined, limit)
        return quantize_fp8(mid)
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from ops.silu_medium.dep8_kernel import _build

    out = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=combined.device)
    scale = torch.empty(
        (rows, k // 32), dtype=torch.float8_e8m0fnu, device=combined.device
    )
    _run_compiled(
        _build(k, float(limit), combined.device.index),
        combined.flatten(),
        out.view(torch.int32).flatten(),
        scale.view(torch.uint8).flatten(),
        out.numel(),
        fx.Stream(torch.cuda.current_stream(combined.device)),
    )
    return out, scale
