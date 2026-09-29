# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL DSV4-Pro layer and MoE dispatch with native ATOM attention/mHC.

Prepared before child linears shuffle their weights. All query buckets share
one packed weight set; each owns its graph-stable scratch and IPC epochs.
"""

import logging
import socket
import weakref

logger = logging.getLogger("atom")


def shape_supported(context, num_tokens):
    """Host metadata only: the decision must agree across all TP ranks."""
    return (
        context is not None
        and not context.is_prefill
        and context.running_bs == 1
        and context.running_tokens == num_tokens
        and num_tokens in (1, 2, 3, 4)
        and context.ubatch_token_offset == 0
    )


def stable_compressor_projection_supported(adapter, fwd, num_tokens):
    """Use stable BF16 projection only while a supported layer bucket is live."""
    op = adapter.ops.get(num_tokens) if adapter is not None else None
    return (
        op is not None
        and not op._closed
        and shape_supported(fwd.context, num_tokens)
        and not fwd.context.is_dummy_run
        and fwd.attn_metadata is not None
        and fwd.ubatch_slices is None
    )


def bind_stable_compressors(block):
    """Bind without weight copies or global GEMM overrides; close disables it."""
    compressors = [block.attn.compressor]
    if block.attn.indexer is not None:
        compressors.append(block.attn.indexer.compressor)
    for compressor in compressors:
        if compressor is not None:
            compressor._dsv4_mono_adapter = weakref.ref(block.ffn._moe_mono)


class Dsv4MoeMono:
    def __init__(self, owner, args):
        self.owner = weakref.ref(owner)
        self.args = args
        self.ops = {}
        self.prepared = None
        self.attempted = False

    def prepare(self):
        if self.attempted:
            return
        self.attempted = True
        import torch
        import torch.distributed as dist
        from aiter import ActivationType, QuantType, dtypes
        from aiter.dist.parallel_state import get_tp_group
        from aiter.fused_moe import resolve_activation_dtype
        from aiter.ops.flydsl.moe_common import GateMode

        from atom.config import get_current_atom_config
        from atom.plugin.prepare import is_plugin_mode

        owner = self.owner()
        ac = get_current_atom_config()
        expert, shared = owner.experts, owner.shared_experts
        unsupported = (
            (not is_plugin_mode(), "native ATOM frontend required"),
            (not expert.use_ep and expert.dp_size == 1, "EP/DP unsupported"),
            (ac.pipeline_parallel_size == 1, "PP unsupported"),
            (ac.prefill_context_parallel_size == 1, "PCP unsupported"),
            (ac.decode_context_parallel_size == 1, "DCP unsupported"),
            (not (ac.enable_tbo or ac.enable_tbo_decode), "TBO unsupported"),
            (not getattr(ac, "eplb_enable", False), "EPLB unsupported"),
            (
                not getattr(ac, "fake_eplb", False),
                "synthetic load-balance routing unsupported",
            ),
            (
                not getattr(expert, "online_quant", False),
                "online requantization unsupported",
            ),
            (expert._comm_fused_moe is None, "another MoE backend is active"),
            (shared is not None, "native FP8 shared expert required"),
            (
                not getattr(expert.quant_method, "use_triton", True),
                "standard AITER A8W4 path required",
            ),
            (
                getattr(expert.quant_method, "is_guinterleave", False),
                "ATOM_MOE_GU_ITLV=1 required",
            ),
        )
        for ok, reason in unsupported:
            if not ok:
                logger.info("%s: DSV4 MoE mono disabled: %s", owner.prefix, reason)
                return
        if shared.use_fused_clamp_act_mul:
            logger.info(
                "%s: DSV4 MoE mono requires ATOM_V4_USE_TRITON_FUSION=0", owner.prefix
            )
            return
        selected = resolve_activation_dtype(
            QuantType.per_1x32,
            dtypes.fp4x2,
            activation=ActivationType.Silu,
            gate_mode=GateMode.INTERLEAVE,
            M=1,
        )
        if selected != dtypes.fp8:
            logger.info(
                "%s: DSV4 MoE mono requires AITER_BF16_FP8_MOE_BOUND=0", owner.prefix
            )
            return
        # The external kernel package is imported only for an enabled deployment.
        from kernels.monokernel.dsv4.config import Dsv4Config
        from kernels.monokernel.dsv4.moe import Dsv4MoeMonoKernel, PreparedMoeWeights

        cfg = Dsv4Config.from_atom_args(self.args)
        try:
            cfg.validate_pro()
            cfg.validate_moe(1, owner.tp_size)
        except ValueError as exc:
            logger.info("%s: DSV4 MoE mono disabled: %s", owner.prefix, exc)
            return
        group = get_tp_group()
        peers = [None] * owner.tp_size
        if owner.tp_size > 1:
            dist.all_gather_object(peers, socket.gethostname(), group=group.cpu_group)
            if len(set(peers)) != 1:
                logger.info("%s: DSV4 MoE mono requires local TP", owner.prefix)
                return

        def scale_bytes(scale):
            if scale.dtype == torch.uint8:
                return scale.contiguous()
            if scale.dtype == torch.float8_e8m0fnu:
                return scale.view(torch.uint8).contiguous()
            # Native FP8 scales may have been expanded to FP32 by the loader.
            if not torch.all(scale > 0):
                raise ValueError("DSV4 shared scales must be positive powers of two")
            encoded = scale.to(torch.float8_e8m0fnu)
            if not torch.equal(encoded.float(), scale.float()):
                raise ValueError("DSV4 shared scales are not exact E8M0 values")
            return encoded.view(torch.uint8).contiguous()

        su, sd = shared.gate_up_proj, shared.w2
        for linear in (owner.gate, su, sd):
            if getattr(linear.weight, "is_shuffled", False):
                raise RuntimeError(
                    "DSV4 mono preparation must precede child weight shuffling"
                )
        if not (su.blockscale_e8m0_scale and sd.blockscale_e8m0_scale):
            raise ValueError(
                "DSV4 mono requires the native E8M0 shared quantization path"
            )
        bias = getattr(owner.gate, "e_score_correction_bias", None)
        if bias is None:
            bias = torch.zeros(
                cfg.experts, dtype=torch.float32, device=owner.gate.weight.device
            )
        self.prepared = PreparedMoeWeights(
            owner.gate.weight,
            bias,
            expert.w13_weight.view(torch.uint8),
            scale_bytes(expert.w13_weight_scale),
            expert.w2_weight.view(torch.uint8),
            scale_bytes(expert.w2_weight_scale),
            tp=owner.tp_size,
            config=cfg,
            shared=(
                su.weight,
                scale_bytes(su.weight_scale),
                sd.weight,
                scale_bytes(sd.weight_scale),
            ),
            hash_table=getattr(owner.gate, "tid2eid", None),
            borrow_aiter_experts=(expert.w13_weight, expert.w2_weight),
        )
        self.ops = {
            seq: Dsv4MoeMonoKernel(
                prepared=self.prepared,
                seq_len=seq,
                config=cfg,
                tp=owner.tp_size,
                rank=expert.tp_rank,
                group=group.cpu_group,
                hash_routing=owner.is_hash_layer,
            )
            for seq in (1, 2, 3, 4)
        }
        logger.info(
            "%s: prepared DSV4-Pro A8W4 MoE mono, TP%d, bs1/seq1..4",
            owner.prefix,
            owner.tp_size,
        )

    def maybe_forward(self, x):
        from atom.utils.forward_context import get_forward_context

        fwd = get_forward_context()
        if (
            not self.ops
            or not shape_supported(fwd.context, x.shape[0])
            or fwd.ubatch_slices is not None
        ):
            return None
        if self.ops[x.shape[0]]._closed:
            return None
        owner = self.owner()
        ids = None
        if owner.is_hash_layer:
            input_ids = fwd.context.input_ids
            if input_ids is None or input_ids.numel() != x.shape[0]:
                return None
            return self.ops[x.shape[0]](x, token_ids=input_ids.reshape(-1))
        return self.ops[x.shape[0]](x, hash_ids=ids)

    def close(self):
        """Collective: close before destroying the ATOM TP process group."""
        for op in self.ops.values():
            op.close()
        self.ops.clear()


class Dsv4LayerMono:
    """Prepare per-query layer wrappers before graph capture, without weight copies."""

    def __init__(self, owner):
        self.owner = weakref.ref(owner)
        self.ops = {}

    def prepare(self):
        from aiter.dist.parallel_state import get_tp_group
        from kernels.monokernel.dsv4 import Dsv4MonoKernel

        if self.ops:
            return
        block = self.owner()
        adapter = block.ffn._moe_mono
        adapter.prepare()
        if not adapter.ops:
            return
        self.ops = {
            seq: Dsv4MonoKernel(
                block,
                seq,
                layer_idx=block.layer_id,
                rank=block.ffn.experts.tp_rank,
                npes=block.ffn.tp_size,
                group=get_tp_group().cpu_group,
            )
            for seq in (1, 2, 3, 4)
        }

    def supports(self, num_tokens):
        from atom.utils.forward_context import get_forward_context

        fwd = get_forward_context()
        return (
            num_tokens in self.ops
            and not self.ops[num_tokens].closed
            and not self.ops[num_tokens].moe._closed
            and shape_supported(fwd.context, num_tokens)
            and not fwd.context.is_dummy_run
            and fwd.attn_metadata is not None
            and fwd.ubatch_slices is None
        )

    def forward(self, hc_state, positions, *, unfused=False):
        seq = positions.numel()
        if not self.supports(seq):
            raise ValueError(
                "DSV4 mono_kernel_forward supports prepared bs1/seq1..4 decode layers only"
            )
        return self.ops[seq](hc_state, positions, unfused=unfused)

    def close(self):
        for op in self.ops.values():
            op.close()
        self.ops.clear()


def close_monokernels(model):
    """Collectively close IPC while ATOM's TP groups still exist."""
    for module in model.modules():
        layer = getattr(module, "_layer_mono", None)
        if layer is not None:
            layer.close()
        moe = getattr(module, "_moe_mono", None)
        if moe is not None:
            moe.close()
