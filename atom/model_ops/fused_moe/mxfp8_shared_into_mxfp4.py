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

With ``ptpc``, a per-channel FP8 shared expert joins routed experts that online
quantization makes per-channel FP8 (V4.1 FP4PTPC): same format, so the staged
tensors are copied into the slot once the routed buffers exist.
"""

import torch
from aiter import QuantType, dtypes

from atom.model_ops.fused_moe.expert_layout import expert_shard_dim
from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz
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

    def __init__(
        self, layer, quant_config, hidden_size: int, inter: int, ptpc: bool = False
    ):
        name = "ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4"
        if layer.expert_layout.shared_is_routed:
            # EPLB with all2all EP routes the shared expert as a logical expert
            # in the routed slots, so its tensors never reach a shared slot.
            raise NotImplementedError(
                f"{layer.prefix}: {name} does not support EPLB with all2all EP, "
                "which places the shared expert among the routed experts."
            )
        self.ptpc = ptpc
        source = layer.layer_quant_config
        if ptpc:
            if layer._stream_online_quant:
                # Its buffers are still on meta here.
                raise NotImplementedError(
                    f"{layer.prefix}: a fused per-channel FP8 shared expert does "
                    "not support streaming online quant."
                )
        elif will_online_requant(
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
        f8 = torch.float8_e4m3fn
        # One F32 scale per row, or E8M0 per 1x32 block.
        s13, s2 = (1, 1) if ptpc else (hidden_size // 32, inter // 32)
        sdt = torch.float32 if ptpc else torch.uint8
        self.staged = {
            "w13_weight": torch.empty(
                n, 2 * inter, hidden_size, dtype=f8, device=device
            ),
            "w13_weight_scale": torch.empty(
                n, 2 * inter, s13, dtype=sdt, device=device
            ),
            "w2_weight": torch.empty(n, hidden_size, inter, dtype=f8, device=device),
            "w2_weight_scale": torch.empty(
                n, hidden_size, s2, dtype=sdt, device=device
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
        # A per-channel scale is a row's: w13's shard by row, w2's stay whole.
        load = (
            layer._load_per_channel_weight_scale
            if self.ptpc and name.endswith("_scale")
            else layer._load_model_weight_or_group_weight_scale
        )
        load(
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
        if self.ptpc:
            self._copy(layer, staged)
            return

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

    def _copy(self, layer, staged) -> None:
        """The per-channel FP8 shared experts into their slots as they are."""

        def put(dst, src):
            region = dst[tuple(slice(n) for n in src.shape[: dst.dim()])]
            layer._copy_quant_storage(region, src.reshape(region.shape))

        fnuz = layer.w13_weight.dtype == torch.float8_e4m3fnuz
        for i in range(self.num_shared):
            expert_id = self.shared_lo + i
            w13, s13 = staged["w13_weight"][i], staged["w13_weight_scale"][i]
            w2, s2 = staged["w2_weight"][i], staged["w2_weight_scale"][i]
            if fnuz:
                # The online-quantized routed experts are FNUZ already.
                w13, s13, _ = normalize_e4m3fn_to_e4m3fnuz(w13, s13)
                w2, s2, _ = normalize_e4m3fn_to_e4m3fnuz(w2, s2)
            slot = [layer.w13_weight, layer.w13_weight_scale]
            slot += [layer.w2_weight, layer.w2_weight_scale]
            # `_online_quant` filled the slot from unloaded bytes.
            for p in slot:
                p.data[expert_id].zero_()
            w13_dst = layer.w13_weight.data[expert_id]
            w13_scale_dst = layer.w13_weight_scale.data[expert_id]
            half, dst_half = w13.shape[0] // 2, w13_dst.shape[0] // 2
            for src_off, dst_off in ((0, 0), (half, dst_half)):
                put(w13_dst[dst_off:], w13[src_off : src_off + half])
                put(w13_scale_dst[dst_off:], s13[src_off : src_off + half])
            put(layer.w2_weight.data[expert_id], w2)
            put(layer.w2_weight_scale.data[expert_id], s2)
