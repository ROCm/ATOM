# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HIP kernels of the NVFP4 MLA decode path (nvfp4_decode_prep.cu), built on
first use (`torch.utils.cpp_extension.load_inline`; ``ATOM_NVFP4_HIP_BUILD_DIR``
picks the build directory, shared by the ranks under the extension's file
lock)."""

import functools
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _opus_include() -> str:
    """aiter's csrc/include, for opus/opus.hpp (the TDM descriptors)."""
    import aiter

    inc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(aiter.__file__))), "csrc", "include")
    if not os.path.exists(os.path.join(inc, "opus", "opus.hpp")):
        raise RuntimeError(f"NVFP4 decode prep needs aiter's opus headers; {inc}/opus/opus.hpp is missing")
    return inc


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load_inline

    build = os.environ.get("ATOM_NVFP4_HIP_BUILD_DIR")
    if build:
        os.makedirs(build, exist_ok=True)
    with open(os.path.join(_HERE, "nvfp4_decode_prep.cu")) as f:
        src = f.read()
    return load_inline(
        name="atom_nvfp4_decode_prep",
        cpp_sources=(
            "void q_rope_quant(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
            "torch::Tensor, torch::Tensor, bool);\n"
            "void decode_prep(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
            "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
            "torch::Tensor, torch::Tensor, torch::Tensor, bool, int64_t, int64_t, bool, bool);"
        ),
        cuda_sources=src,
        functions=["q_rope_quant", "decode_prep"],
        # The quantization reproduces the Triton kernel's roundings; no FMA
        # contraction may change them.
        extra_cuda_cflags=["-O3", "-ffp-contract=off", "--offload-arch=gfx1250"],
        extra_include_paths=[_opus_include()],
        build_directory=build or None,
        verbose=False,
    )


def hip_decode_prep_available() -> bool:
    """gfx1250 only: the kernels use its E2M1 conversions."""
    if not torch.cuda.is_available():
        return False
    return "gfx1250" in getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


def warm(device) -> None:
    """Build or load the extension and run its kernel once, eagerly: its code
    object must be loaded before a CUDA graph captures the first decode step."""
    if not hip_decode_prep_available():
        return
    q_rope = torch.zeros((1, 128, 64), dtype=torch.bfloat16, device=device)
    q_aux = torch.zeros((1, 128, 128), dtype=torch.uint8, device=device)
    rows = torch.zeros(1, dtype=torch.int64, device=device)
    q_rope_quantize(q_rope, q_aux, rows, rows)
    torch.cuda.synchronize(device)


def q_rope_quantize(q_rope, q_aux, positions, slots, cos=None, sin=None) -> None:
    """The RoPE part of ``q_aux [B, 128, 128]`` (bytes 0..63 and 96..127) from
    ``q_rope [B, 128, 64]``, in place, as `decode_prep` writes it with
    ``q_planes``: pre-RoPE and rotated here with the BF16 cos/sin caches, or
    already roped without them; a padding row (slot -1) gets a zero Q.
    Graph-safe."""
    fused = cos is not None
    _ext().q_rope_quant(q_rope, q_aux, positions, slots, cos if fused else q_rope,
                        sin if fused else q_rope, fused)


def decode_prep(q_rope, q_aux, ring, req_ids, k_nope, k_rope, slots, kv_nope, kv_aux,
                positions, cos=None, sin=None) -> None:
    """`q_rope_quantize` and, in the same launch, `decode_prep`'s K part with
    the N16 AUX layout and a 16-token ring: each token into its tail row, its
    tile requantized into the page, its RoPE part into the AUX tile; padding
    rows (slot -1) write nothing. Same bytes as the Triton kernel. Graph-safe.
    ``ATOM_NVFP4_PREP_TDM=0`` keeps the per-lane loads (the TDM path needs order
    0 / 1, one K block per token and q_rope rows 128 head strides apart)."""
    fused = cos is not None
    order = int(os.environ.get("ATOM_NVFP4_PREP_ORDER", "1"))
    k_split = int(os.environ.get("ATOM_NVFP4_PREP_KSPLIT", "1"))
    rcp = os.environ.get("ATOM_NVFP4_PREP_RCP", "1") == "1"
    tdm = os.environ.get("ATOM_NVFP4_PREP_TDM", "1") == "1"
    _ext().decode_prep(q_rope, q_aux, ring, req_ids, k_nope, k_rope, kv_nope, kv_aux,
                       positions, slots, cos if fused else q_rope, sin if fused else q_rope, fused,
                       order, k_split, rcp, tdm)
