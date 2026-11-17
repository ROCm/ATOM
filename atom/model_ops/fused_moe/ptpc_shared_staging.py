# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A per-channel FP8 shared expert fused into the online-quantized FP8 slot of
its FusedMoE.

DeepSeek-V4.1's FP4PTPC checkpoint stores its shared expert in per-channel FP8
and its routed experts in FP4, which `--online_quant_config` makes per-channel
FP8 at load. Same format then, so the shared expert's tensors are staged while
loading and copied into the slot after that conversion, which would otherwise
overwrite them.
"""

import torch

from atom.config import get_current_atom_config
from atom.model_ops.fused_moe.expert_layout import expert_shard_dim
from atom.model_ops.fused_moe.mxfp8_shared_into_mxfp4 import _STAGED_SHARDS
from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz


class PtpcSharedStaging:
    """The staging of one FusedMoE's per-channel FP8 shared experts. The layer
    calls `load` from its weight_loader and `fuse` after its online
    quantization."""

    def __init__(self, layer, hidden_size: int, inter: int):
        # EPLB with all2all EP places the shared expert among the routed ones,
        # the FP8 online method has no all2all shared-expert dispatch, and
        # streaming buffers are still on meta here.
        if (
            layer.expert_layout.shared_is_routed
            or layer.expert_layout.uses_all2all_fusion
            or layer._stream_online_quant
        ):
            raise NotImplementedError(
                f"{layer.prefix}: a fused per-channel FP8 shared expert does not "
                "support all2all EP or streaming online quant."
            )
        self.shared_lo = layer.expert_layout.routed_physical_per_rank
        self.num_shared = layer.num_fused_shared_experts
        n, device = self.num_shared, layer.w13_weight.device
        f8, f32 = torch.float8_e4m3fn, torch.float32
        self.staged = {
            "w13_weight": torch.empty(
                n, 2 * inter, hidden_size, dtype=f8, device=device
            ),
            "w13_weight_scale": torch.empty(n, 2 * inter, 1, dtype=f32, device=device),
            "w2_weight": torch.empty(n, hidden_size, inter, dtype=f8, device=device),
            "w2_weight_scale": torch.empty(n, hidden_size, 1, dtype=f32, device=device),
        }
        self.loaded = set()

    def load(self, layer, param, loaded_weight, shard_id: str, expert_id: int) -> bool:
        """Stage one checkpoint tensor of a shared expert; returns whether
        ``expert_id`` (local) is a shared slot. TP-sharded like the routed
        experts': a scale with its row (w2's stay whole)."""
        if expert_id < self.shared_lo:
            return False
        if self.staged is None:
            raise NotImplementedError(
                f"{layer.prefix}: the shared expert is already in its fused "
                "slot; reloading it is not supported."
            )
        name = next((n for n in _STAGED_SHARDS if getattr(layer, n) is param), None)
        if name is None:
            raise ValueError(
                f"{layer.prefix}: unexpected fused shared-expert tensor "
                f"for expert {expert_id} (shard {shard_id!r})."
            )
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be w1/w2/w3, got {shard_id!r}.")
        load = (
            layer._load_per_channel_weight_scale
            if name.endswith("_scale")
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
        """Copy the staged shared experts into their (unpadded) slots, as FNUZ
        bytes with doubled scales where the routed experts are FNUZ, then free
        the staging."""
        if not self.loaded and get_current_atom_config().load_dummy:
            # `--load_dummy` loads nothing; the slot keeps its dummy values.
            self.staged = None
            return
        expected = {
            (name, self.shared_lo + i, shard)
            for i in range(self.num_shared)
            for name, shards in _STAGED_SHARDS.items()
            for shard in shards
        }
        missing = expected - self.loaded
        if missing:
            raise RuntimeError(
                f"{layer.prefix}: fused shared expert tensors were not "
                f"loaded from the checkpoint: {sorted(missing)}"
            )
        staged, self.staged = self.staged, None
        fnuz = layer.w13_weight.dtype == torch.float8_e4m3fnuz
        for i in range(self.num_shared):
            expert_id = self.shared_lo + i
            for w in ("w13", "w2"):
                q, s = staged[f"{w}_weight"][i], staged[f"{w}_weight_scale"][i]
                if fnuz:
                    q, s, _ = normalize_e4m3fn_to_e4m3fnuz(q, s)
                layer._copy_quant_storage(
                    getattr(layer, f"{w}_weight").data[expert_id], q
                )
                scale = getattr(layer, f"{w}_weight_scale").data[expert_id]
                scale.copy_(s.reshape(scale.shape))
