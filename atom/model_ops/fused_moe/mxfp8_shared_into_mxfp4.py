# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""An MXFP8 shared expert fused into the MXFP4 slot of its FusedMoE.

The Quark MXFP4 export of MiniMax-M3.1 ships its routed experts in MXFP4 and
its shared expert in MXFP8. Under ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4 the shared
expert runs as one more expert of the fused MoE (129 experts, top-5), so its
slot needs MXFP4 bytes the checkpoint does not have. This is a load-time format
conversion, the same kind of step as ``FusedMoE._online_quant``: the shared
slot's checkpoint tensors are staged in MXFP8 while loading, then each shared
expert is dequantized to bf16 and quantized to MXFP4 into its slot.
"""

import torch
from aiter import QuantType, dtypes

from atom.model_ops.fused_moe.expert_layout import expert_shard_dim
from atom.quant_spec import will_online_requant
from atom.quantization.quark.utils import dequant_mxfp8, quant_weight_online

# The FusedMoE parameter each staged tensor stands in for, and its shards.
_STAGED_SHARDS = {
    "w13_weight": ("w1", "w3"),
    "w13_weight_scale": ("w1", "w3"),
    "w2_weight": ("w2",),
    "w2_weight_scale": ("w2",),
}


class Mxfp8SharedIntoMxfp4:
    """The staging of one FusedMoE's MXFP8 shared experts. The layer calls
    `load` from its weight_loader and `fuse` after its online quantization."""

    def __init__(self, layer, quant_config, hidden_size: int, inter: int):
        name = "ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4"
        if layer.expert_layout.shared_is_routed:
            # EPLB with all2all EP routes the shared expert as a logical expert
            # in the routed slots, so its tensors never reach a shared slot.
            raise NotImplementedError(
                f"{layer.prefix}: {name} does not support EPLB with all2all EP, "
                "which places the shared expert among the routed experts."
            )
        source = layer.layer_quant_config
        if will_online_requant(
            quant_config, layer.prefix, source.quant_type, source.quant_dtype
        ):
            # The shared slot is written as MXFP4, so the routed buffers must
            # stay the checkpoint's MXFP4. This also rules out streaming online
            # quant, whose buffers are still on meta here.
            raise NotImplementedError(
                f"{layer.prefix}: {name} needs the routed experts to stay MXFP4, "
                "but --online_quant_config re-quantizes this layer."
            )
        self.shared_lo = layer.expert_layout.routed_physical_per_rank
        self.num_shared = layer.num_fused_shared_experts
        n, device = self.num_shared, layer.w13_weight.device
        f8, u8 = torch.float8_e4m3fn, torch.uint8
        self.staged = {
            "w13_weight": torch.empty(
                n, 2 * inter, hidden_size, dtype=f8, device=device
            ),
            "w13_weight_scale": torch.empty(
                n, 2 * inter, hidden_size // 32, dtype=u8, device=device
            ),
            "w2_weight": torch.empty(n, hidden_size, inter, dtype=f8, device=device),
            "w2_weight_scale": torch.empty(
                n, hidden_size, inter // 32, dtype=u8, device=device
            ),
        }
        self.loaded = set()

    def load(
        self,
        layer,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        expert_id: int,
    ) -> bool:
        """Stage one checkpoint tensor of a shared expert.

        Returns whether ``expert_id`` (local) is a shared slot; its arrivals
        never touch the MXFP4 buffers. Weights and E8M0 scales are TP-sharded
        like any per_1x32 expert tensor.
        """
        if expert_id < self.shared_lo:
            return False
        if self.staged is None:
            # `fuse` freed the staging; keeping it for the life of the process
            # only to serve a reload is not worth its memory.
            raise NotImplementedError(
                f"{layer.prefix}: the MXFP8 shared expert was already "
                "re-quantized into its MXFP4 slot; reloading it under "
                "ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4 is not supported."
            )
        name = next((n for n in _STAGED_SHARDS if getattr(layer, n) is param), None)
        if name is None:
            raise ValueError(
                f"{layer.prefix}: unexpected fused MXFP8 shared-expert tensor "
                f"for expert {expert_id} (shard {shard_id!r})."
            )
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be w1/w2/w3, got {shard_id!r}.")
        layer._load_model_weight_or_group_weight_scale(
            shard_dim=expert_shard_dim(shard_id, False),
            expert_data=self.staged[name][expert_id - self.shared_lo],
            shard_id=shard_id,
            loaded_weight=loaded_weight,
            tp_rank=layer.tp_rank,
        )
        self.loaded.add((name, expert_id, shard_id))
        return True

    def fuse(self, layer) -> None:
        """Re-quantize the staged shared experts into their MXFP4 slots, then
        free the staging.

        Runs after loading and before the quant method's own post-processing
        (shuffle) covers every expert; the gate/up halves are laid out as the
        routed experts' (``[0, I)`` and ``[half, half + I)``)."""
        expected = {
            (name, self.shared_lo + i, shard)
            for i in range(self.num_shared)
            for name, shards in _STAGED_SHARDS.items()
            for shard in shards
        }
        missing = expected - self.loaded
        if missing:
            raise RuntimeError(
                f"{layer.prefix}: fused MXFP8 shared expert tensors were not "
                f"loaded from the checkpoint: {sorted(missing)}"
            )
        staged, self.staged = self.staged, None

        def quant_weight(w):
            return quant_weight_online(
                w, online_quant_type=QuantType.per_1x32, online_quant_dtype=dtypes.fp4x2
            )

        for i in range(self.num_shared):
            expert_id = self.shared_lo + i
            w13 = dequant_mxfp8(
                staged["w13_weight"][i], staged["w13_weight_scale"][i]
            ).to(torch.bfloat16)
            w13_dst = layer.w13_weight.data[expert_id]
            w13_scale_dst = layer.w13_weight_scale.data[expert_id]
            half, dst_half = w13.shape[0] // 2, w13_dst.shape[0] // 2
            for src_off, dst_off in ((0, 0), (half, dst_half)):
                q, sc = quant_weight(w13[src_off : src_off + half])
                layer._copy_quant_storage(
                    w13_dst[dst_off : dst_off + q.shape[0], : q.shape[1]], q
                )
                layer._copy_quant_storage(
                    w13_scale_dst[dst_off : dst_off + sc.shape[0], : sc.shape[1]], sc
                )
            del w13
            w2 = dequant_mxfp8(staged["w2_weight"][i], staged["w2_weight_scale"][i]).to(
                torch.bfloat16
            )
            q, sc = quant_weight(w2)
            layer._copy_quant_storage(
                layer.w2_weight.data[expert_id, : q.shape[0], : q.shape[1]], q
            )
            layer._copy_quant_storage(
                layer.w2_weight_scale.data[expert_id, : sc.shape[0], : sc.shape[1]], sc
            )
