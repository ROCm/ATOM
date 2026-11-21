# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 decode router in one launch on gfx950 (``m31_router.hip``).

For 1..16 rows it runs the router GEMV, the sigmoid gate (correction bias,
top-4, renormalize) and aiter's block_m 16 MoE sort, which is handed to the
next ``fused_moe`` through ``aiter.register_presorted_moe``: the router
hgemm, ``topk_gating`` and the sort become one kernel. The sort is written in
both aiter layouts (the aux sort's, and ``moe_sorting``'s under
``sorted_token_ids_moe_sorting``, which differ only in the padding ids), and
the launch also zeroes the MoE output an accumulating GEMM2 adds into. An aiter
whose pipeline cannot take the presort logs that once and sorts again. The logits are fp32
split-K sums of bf16 products (the separate path rounds them to bf16), so a
near tie may pick a different expert; everything after the selection is the
separate path's arithmetic.

The code object is built with hipcc once per source and ROCm under
``~/.cache/atom/hipcc`` and launched through the HIP module API.
``ATOM_M3_FUSED_ROUTER=0`` disables it; a build or load failure disables it
with a warning.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import os
import pathlib
import subprocess
import tempfile

import torch

try:  # aiter with the caller-presort hook of fused_moe
    from aiter.presorted_moe import register_presorted_moe
except ImportError:  # older aiter: the fused router stays off
    register_presorted_moe = None

logger = logging.getLogger("atom")

_SRC = pathlib.Path(__file__).with_name("m31_router.hip")
_HIPCC_FLAGS = ("--genco", "--offload-arch=gfx950", "-O3")
MAX_TOKENS = 16
NUM_EXPERTS, TOPK, HIDDEN, BLOCK_M = 128, 4, 6144, 16
# kernel and grid per row bucket: (experts per CTA, K step, split) -> CTAs
_KERNELS = {8: ("m31_router_t8", 48), 16: ("m31_router_t16", 64)}
_MAX_SPLIT = 12


class _Args(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        (n, ctypes.c_void_p)
        for n in (
            "x",
            "w",
            "bias",
            "part",
            "cnt",
            "topk_w",
            "topk_ids",
            "sorted_ids",
            "sorted_w",
            "sorted_eids",
            "num_valid",
            "m_indices",
            "reverse_sorted",
            "sorted_ids_ms",
            "n_local",
            "moe_buf",
        )
    ] + [
        ("tok", ctypes.c_int32),
        ("x_stride", ctypes.c_int32),
        ("scale", ctypes.c_float),
    ]


assert (
    ctypes.sizeof(_Args) == 140
), "kernarg ABI drift: keep in step with m31_router.hip"


class _Hip:
    """The HIP runtime torch already mapped, and the module's kernels."""

    def __init__(self, code_object: str):
        lib = ctypes.CDLL(_loaded_library("libamdhip64") or "libamdhip64.so")
        lib.hipModuleLoad.restype = ctypes.c_int
        lib.hipModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.hipModuleGetFunction.restype = ctypes.c_int
        lib.hipModuleGetFunction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.hipModuleLaunchKernel.restype = ctypes.c_int
        lib.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        lib.hipGetErrorString.restype = ctypes.c_char_p
        self.lib = lib
        module = ctypes.c_void_p()
        self._check(
            lib.hipModuleLoad(ctypes.byref(module), code_object.encode()),
            "hipModuleLoad",
        )
        self.functions = {}
        for name, _ in _KERNELS.values():
            fn = ctypes.c_void_p()
            self._check(
                lib.hipModuleGetFunction(ctypes.byref(fn), module, name.encode()),
                f"hipModuleGetFunction {name}",
            )
            self.functions[name] = fn

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(
                f"M3.1 fused router: {what} failed: {rc} "
                f"({self.lib.hipGetErrorString(rc).decode()})"
            )

    def launch(self, name: str, grid: int, args: _Args, stream: int) -> None:
        size = ctypes.c_size_t(ctypes.sizeof(args))
        # HIP_LAUNCH_PARAM_BUFFER_POINTER, BUFFER_SIZE, END
        extra = (ctypes.c_void_p * 5)(
            1,
            ctypes.cast(ctypes.byref(args), ctypes.c_void_p).value,
            2,
            ctypes.cast(ctypes.byref(size), ctypes.c_void_p).value,
            3,
        )
        self._check(
            self.lib.hipModuleLaunchKernel(
                self.functions[name],
                grid,
                1,
                1,
                256,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                None,
                ctypes.cast(extra, ctypes.c_void_p),
            ),
            "hipModuleLaunchKernel",
        )


def _loaded_library(stem: str) -> str | None:
    """Path of the ``stem`` library this process mapped. ROCm can ship a second
    libamdhip64; a fresh dlopen of the unversioned name may pick that one, and a
    launch on a torch stream then fails with hipErrorContextIsDestroyed."""
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.split()[-1]
                if os.path.basename(path).startswith(stem):
                    return path
    except OSError:
        pass
    return None


def _code_object() -> pathlib.Path:
    """``m31_router.hip`` built once per source, flags and ROCm, so restarts and
    the other ranks load it instead of running hipcc again."""
    digest = hashlib.sha256(repr((_HIPCC_FLAGS, torch.version.hip)).encode())
    digest.update(_SRC.read_bytes())
    cache_dir = pathlib.Path(os.path.expanduser("~/.cache/atom/hipcc"))
    co = cache_dir / f"m31_router-{digest.hexdigest()[:16]}.co"
    if co.is_file():
        return co
    cache_dir.mkdir(parents=True, exist_ok=True)
    hipcc = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")
    with tempfile.TemporaryDirectory(dir=cache_dir) as build_dir:
        built = os.path.join(build_dir, "router.co")
        subprocess.run(
            [hipcc, *_HIPCC_FLAGS, "-o", built, str(_SRC)],
            check=True,
            capture_output=True,
        )
        # ranks building at once each publish the same bytes
        os.replace(built, co)
    return co


@functools.cache
def _runtime() -> _Hip | None:
    try:
        hip = _Hip(str(_code_object()))
    except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
        detail = getattr(e, "stderr", b"") or b""
        logger.warning(
            "M3.1 fused router disabled: %s %s", e, detail.decode(errors="replace")
        )
        return None
    logger.info("M3.1 fused router: loaded %s", _KERNELS)
    return hip


_workspaces: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Split partials (always fully overwritten) and the self-resetting ticket
    counter, one pair per device shared by every layer: the launches of one
    stream run in order."""
    ws = _workspaces.get(device)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ws = _workspaces[device] = (
            torch.empty(
                _MAX_SPLIT * BLOCK_M * NUM_EXPERTS, dtype=torch.float32, device=device
            ),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
    return ws


def fused_router_enabled() -> bool:
    from atom.utils import envs

    return envs.ATOM_M3_FUSED_ROUTER and register_presorted_moe is not None


def fused_router_supported(
    x: torch.Tensor, gate_weight: torch.Tensor, bias: torch.Tensor | None
) -> bool:
    """Whether ``m31_route`` serves these operands (the build is checked at the
    launch)."""
    m = x.shape[0]
    return (
        x.dim() == 2
        and 1 <= m <= MAX_TOKENS
        and x.shape[1] == HIDDEN
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and gate_weight.dtype == torch.bfloat16
        and gate_weight.shape == (NUM_EXPERTS, HIDDEN)
        and gate_weight.is_contiguous()
        and bias is not None
        and bias.dtype == torch.float32
        and bias.shape == (NUM_EXPERTS,)
        and bias.is_contiguous()
        and torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
        == "gfx950"
    )


def m31_route(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    n_local: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict] | None:
    """``(topk_weights [M, 4] fp32, topk_ids [M, 4] int32, presort)`` for
    ``fused_router_supported`` operands, or None while the build or the
    workspace is missing (a first call under graph capture, a build failure).

    ``presort`` is the ``aiter.register_presorted_moe`` registration for a
    ``fused_moe(..., topk_ids, topk_weights)`` call with no ``num_local_tokens``.
    ``n_local`` (int32 device scalar): rows from it on are padding -- no routes,
    zero weights; without it every row is routed. Weights are multiplied by
    ``routed_scaling_factor``.
    """
    hip = _runtime()
    ws = _workspace(x.device)
    if hip is None or ws is None:
        return None
    m = x.shape[0]
    dev = x.device
    topk_w = torch.empty(m, TOPK, dtype=torch.float32, device=dev)
    topk_ids = torch.empty(m, TOPK, dtype=torch.int32, device=dev)
    # aiter's BM=16 sort capacity, a whole number of blocks
    max_sorted = -(-(m * TOPK + NUM_EXPERTS * BLOCK_M - TOPK) // BLOCK_M) * BLOCK_M
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=dev)
    sorted_w = torch.empty(max_sorted, dtype=torch.float32, device=dev)
    sorted_eids = torch.empty(max_sorted // BLOCK_M, dtype=torch.int32, device=dev)
    num_valid = torch.empty(2, dtype=torch.int32, device=dev)
    m_indices = torch.empty(max_sorted, dtype=torch.int32, device=dev)
    reverse_sorted = torch.empty(m * TOPK, dtype=torch.int32, device=dev)
    sorted_ids_ms = torch.empty(max_sorted, dtype=torch.int32, device=dev)
    # the MoE output, zeroed in the launch for a GEMM2 that accumulates into it
    moe_buf = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev)
    tensors = (
        x,
        gate_weight,
        bias,
        *ws,
        topk_w,
        topk_ids,
        sorted_ids,
        sorted_w,
        sorted_eids,
        num_valid,
        m_indices,
        reverse_sorted,
    )
    args = _Args(
        *(t.data_ptr() for t in tensors),
        sorted_ids_ms.data_ptr(),
        None if n_local is None else n_local.data_ptr(),
        moe_buf.data_ptr(),
        m,
        x.stride(0),
        float(routed_scaling_factor),
    )
    name, grid = _KERNELS[8 if m <= 8 else 16]
    hip.launch(name, grid, args, torch.cuda.current_stream(dev).cuda_stream)
    presort = {
        "sorted_token_ids": sorted_ids,
        "sorted_weights": sorted_w,
        "sorted_expert_ids": sorted_eids,
        "num_valid_ids": num_valid,
        "m_indices": m_indices,
        "reverse_sorted": reverse_sorted,
        # moe_sorting's layout of the same ids (padding (topk << 24) | M), for a
        # pipeline that takes no aux sort, and the output the launch zeroed
        "sorted_token_ids_moe_sorting": sorted_ids_ms,
        "moe_buf": moe_buf,
        "moe_buf_zeroed": True,
        "moe_buf_ptr": moe_buf.data_ptr(),
        "topk_ids_ptr": topk_ids.data_ptr(),
        "topk_weights_ptr": topk_w.data_ptr(),
        "M": m,
        "block_m": BLOCK_M,
        "num_experts": NUM_EXPERTS,
        "topk": TOPK,
    }
    return topk_w, topk_ids, presort
