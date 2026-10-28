"""Optional AITER fused GDN decode (``fused_gdn_decode_qkvz``) and speculative verify
(``fused_gdn_verify_qkvz``) for native ATOM Qwen3.5 on gfx1250.

A decode-only batch runs conv1d, gating, the delta-rule update and the gated
RMSNorm as one kernel. The kernel reads a V-major state, so the cache builder
exposes V-major storage as the logical KV view (zero-copy, as for Qwen4Exp's
FlyDSL decode). Its output is already normalized, so a layer that uses it also
applies the gated RMSNorm inside the attention op for every other batch.

With ReplaySSM (speculative decoding), a batch of spec decodes runs the verify
kernel instead: conv1d over the verify window, gating, the state rebuild from the
checkpoint and the committed records, the update and the record of each token, and
the gated RMSNorm. It reads the K-major checkpoints of ReplaySSM as they are.
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
def verify_available():
    """Static gate for the speculative-verify kernel: its env switch, the gate above,
    and the AITER op, which is newer than the decode op."""
    if not envs.ATOM_GDN_FUSED_VERIFY or not available():
        return False
    try:
        from aiter.ops.triton.gated_delta_net.fused_gdn_verify_qkvz import (  # noqa: F401
            fused_gdn_verify_qkvz,
        )
    except ImportError as error:
        logger.warning("AITER fused GDN verify unavailable: %s", error)
        return False
    return True


@functools.cache
def _log_dispatch_verify():
    logger.info("GDN verify: using the AITER fused kernel")


@functools.cache
def _log_fallback(reason, step="decode"):
    logger.warning("GDN %s: AITER fused kernel not used (%s)", step, reason)


def _packed_rows(mixed_qkv, b, a, z, step):
    """The q|k|v|z and b|a rows, as the kernels read them: through one row pointer
    each. Each pair must be adjacent column slices of one projection output; None
    (after a warning) if not."""
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
        _log_fallback("q|k|v|z or b|a are not adjacent slices of one projection", step)
        return None
    return (
        mixed_qkv.as_strided((tokens, qkv_size + z.shape[1] * z.shape[2]), (row, 1)),
        b.as_strided((tokens, 2 * heads), (b.stride(0), 1)),
    )


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

    rows = _packed_rows(mixed_qkv, b, a, z, "decode")
    if rows is None:
        return False
    weight = layer.conv1d.weight
    args = (
        *rows,
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


def verify(layer, mixed_qkv, b, a, z, conv_state, ssm_state, layer_cache, md, out):
    """Run a speculative-verify batch on ReplaySSM state and write
    ``norm(core) * silu(z)`` into ``out``.

    The batch holds only spec decodes. The kernel reads the committed records that
    ``replayssm_commit`` counted for this forward, appends one record for each
    token, and writes the checkpoint only on a flush. Returns False, with nothing
    launched, when q|k|v|z and b|a are not adjacent slices of one projection.
    """
    from aiter.ops.triton.gated_delta_net.fused_gdn_verify_qkvz import (
        fused_gdn_verify_qkvz,
    )

    rows = _packed_rows(mixed_qkv, b, a, z, "verify")
    if rows is None:
        return False
    sequences = md.num_spec_decodes
    weight = layer.conv1d.weight
    args = (
        *rows,
        conv_state,
        md.spec_state_indices_tensor[:, 0][:sequences],
        md.num_accepted_tokens,
        md.spec_query_start_loc[: sequences + 1],
        ssm_state,
        layer_cache.replay_buf_k,
        layer_cache.replay_buf_u,
        layer_cache.replay_buf_g,
        md.write_pos,
        md.slot_idx[:sequences],
        weight.view(weight.size(0), -1),
        getattr(layer.conv1d, "bias", None),
        layer.A_log,
        layer.dt_bias,
        layer.gated_norm.weight,
    )
    _log_dispatch_verify()
    fused_gdn_verify_qkvz(
        *args,
        max_query_len=md.replayssm_max_query_len,
        scale=layer.head_k_dim**-0.5,
        norm_eps=layer.gated_norm.eps,
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
