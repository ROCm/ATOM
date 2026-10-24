"""Optional AITER fused GDN decode (``fused_gdn_decode_qkvz``) for native ATOM Qwen3.5
on gfx1250.

A decode-only batch runs conv1d, gating, the delta-rule update and the gated
RMSNorm as one kernel. The kernel reads a V-major state, so the cache builder
exposes V-major storage as the logical KV view (zero-copy, as for Qwen4Exp's
FlyDSL decode). Its output is already normalized, so a layer that uses it also
applies the gated RMSNorm inside the attention op for every other batch.
"""

import functools
import logging

import torch

from atom.plugin.prepare import is_plugin_mode
from atom.utils import envs

from .layernorm_guard import layer_norm_fwd

logger = logging.getLogger("atom")


@functools.cache
def available():
    """Static gate for a model: the env switch, native ATOM, a gfx1250 device and
    the AITER op. The plugin frameworks bring their own GDN attention, which does
    not take the output gate."""
    if (
        not envs.ATOM_GDN_FUSED_DECODE
        or is_plugin_mode()
        or torch.version.hip is None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx1250"
    ):
        return False
    try:
        from aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz import (  # noqa: F401
            fused_gdn_decode_qkvz,
        )
    except ImportError as error:
        logger.warning("AITER fused GDN decode unavailable: %s", error)
        return False
    return True


@functools.cache
def _log_dispatch():
    logger.info("GDN decode: using the AITER fused kernel")


@functools.cache
def _log_fallback(reason):
    logger.warning("GDN decode: AITER fused kernel not used (%s)", reason)


def decode(layer, mixed_qkv, b, a, z, conv_state, ssm_state, state_indices, out):
    """Run a decode-only batch and write ``norm(core) * silu(z)`` into ``out``.

    Returns False, with nothing launched, for inputs outside the kernel's
    contract. The kernel reads q|k|v|z and b|a through one row pointer each, so
    each pair must be adjacent column slices of one projection output.
    """
    from aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz import (
        fused_gdn_decode_qkvz,
        fused_gdn_decode_qkvz_supported,
    )

    tokens, qkv_size = mixed_qkv.shape
    heads = b.shape[1]
    row = mixed_qkv.stride(0)
    one_row = tokens == 1  # a single row's stride is arbitrary (view() rewrites it)
    if not (
        mixed_qkv.stride(1) == b.stride(1) == a.stride(1) == 1
        and z.stride()[1:] == (z.shape[2], 1)
        and (one_row or (z.stride(0) == row and a.stride(0) == b.stride(0)))
        and z.data_ptr() == mixed_qkv.data_ptr() + qkv_size * mixed_qkv.element_size()
        and a.data_ptr() == b.data_ptr() + heads * b.element_size()
    ):
        _log_fallback("q|k|v|z or b|a are not adjacent slices of one projection")
        return False
    weight = layer.conv1d.weight
    args = (
        mixed_qkv.as_strided((tokens, qkv_size + z.shape[1] * z.shape[2]), (row, 1)),
        b.as_strided((tokens, 2 * heads), (b.stride(0), 1)),
        conv_state,
        ssm_state.transpose(-1, -2),  # the V-major storage behind the KV view
        state_indices,
        weight.view(weight.size(0), -1),
        getattr(layer.conv1d, "bias", None),
        layer.A_log,
        layer.dt_bias,
        layer.gated_norm.weight,
    )
    supported, reason = fused_gdn_decode_qkvz_supported(
        *args, qkvz_layout="flat", out=out
    )
    if not supported:
        _log_fallback(reason)
        return False
    _log_dispatch()
    fused_gdn_decode_qkvz(
        *args,
        scale=layer.head_k_dim**-0.5,
        norm_eps=layer.gated_norm.eps,
        qkvz_layout="flat",
        out=out,
    )
    return True


def gated_rmsnorm_(core, z, norm):
    """``norm(core) * silu(z)`` in place on ``[tokens, heads, dim]``, as one kernel."""
    rows = core.shape[0] * core.shape[1]
    layer_norm_fwd(
        core.view(rows, -1),
        norm.weight,
        None,
        norm.eps,
        z=z.reshape(rows, -1),
        out=core.view(rows, -1),
        norm_before_gate=True,
        is_rms_norm=True,
    )
