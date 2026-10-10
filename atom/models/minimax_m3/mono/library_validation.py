# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate borrowed weights and caches without converting their storage."""

import torch

from atom.models.minimax_m3.mono.library_types import CacheSpec, LayerSpec


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"MiniMax-M3 ATOM mono: {message}")


def checked_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    require(
        tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tensor.device == device
        and tensor.is_contiguous()
        and tensor.data_ptr() % 16 == 0,
        f"{name}: expected contiguous aligned {shape} {dtype} on {device}; "
        f"got {tuple(tensor.shape)} {tensor.dtype} {tensor.device} "
        f"stride={tensor.stride()} offset={tensor.storage_offset()}",
    )
    return tensor


def validate_layer(spec: LayerSpec, device: torch.device) -> None:
    """Check a prepared layer before any runtime allocation or IPC handshake."""

    def check(tensor, name, shape, dtype=torch.bfloat16):
        checked_tensor(tensor, f"layer {spec.layer_id} {name}", shape, dtype, device)

    check(spec.w_qkv, "QKV", (2560, 6144), torch.float8_e4m3fn)
    check(spec.s_qkv, "QKV scales", (2560,), torch.float32)
    check(spec.w_o, "O", (6144, 2048), torch.float8_e4m3fn)
    check(spec.s_o, "O scales", (6144,), torch.float32)
    check(spec.gate, "router", (128, 6144))
    check(spec.g_in, "input norm", (6144,))
    for name in ("g_q", "g_k", "g_iq", "g_ik"):
        check(getattr(spec, name), name, (128,))
    check(spec.cos_sin, "cos/sin", (spec.cos_sin.shape[0], 64))
    check(spec.g_post, "post norm", (6144,))
    check(spec.bias, "router bias", (128,), torch.float32)
    check(spec.w13, "w13", (129, 1536, 3072), torch.float4_e2m1fn_x2)
    check(spec.s13, "s13", (129, 1536, 192), torch.uint8)
    check(spec.w2, "w2", (129, 6144, 384), torch.float4_e2m1fn_x2)
    check(spec.s2, "s2", (129, 6144, 24), torch.uint8)


def validate_cache(cache: CacheSpec, device: torch.device):
    require(cache.layout == "packed_page16_scalar_fp8", "unsupported cache layout")
    main, k, v = cache.main, cache.k, cache.v
    require(
        main.ndim == 4
        and tuple(main.shape[1:]) == (2, 128, 128)
        and main.shape[0] > 0
        and main.element_size() == 1
        and main.is_contiguous()
        and main.device == device,
        "native packed K/V layout",
    )
    require(
        k.dtype == v.dtype == torch.float8_e4m3fn
        and k.device == v.device == device
        and k.is_contiguous()
        and v.is_contiguous()
        and k.shape[0] == main.shape[0] * 16
        and v.shape[0] == k.shape[0] - 8
        and k.data_ptr() == main.data_ptr()
        and v.data_ptr() - k.data_ptr() == 8 * 16 * 128
        and k.untyped_storage().data_ptr() == main.untyped_storage().data_ptr()
        and v.untyped_storage().data_ptr() == main.untyped_storage().data_ptr(),
        "page16 K/V views must share native storage",
    )
    checked_tensor(
        cache.index,
        "index cache",
        (cache.index.shape[0], 128, 128),
        torch.float8_e4m3fn,
        device,
    )
    for scale in (cache.k_scale, cache.v_scale):
        require(
            scale.numel() == 1
            and scale.dtype == torch.float32
            and scale.device == device
            and scale.is_contiguous()
            and bool(torch.isfinite(scale).all())
            and bool((scale > 0).all()),
            "cache scale must be a positive finite FP32 scalar",
        )
    require(
        0 < cache.max_context <= 16384,
        "context capacity exceeds scalar-cache qualification",
    )
