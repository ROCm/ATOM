# SPDX-License-Identifier: MIT
"""The DeepSeek-V4.1 serving model the plugin host builds.

`atom.models.deepseek_v41` is ATOM's own model and stays exactly as upstream
ships it. This package is the same model with the plugin seams and the
recipe-driven options in it: `model.py`, `attention.py`, `moe.py` and
`dspark.py` are its copies (ROCm/ATOM#3529 seams, ROCm/ATOM#3639/#3645
kernels), `v4.py` the V4 MoE / shared expert subclasses. The runtime entry
below reuses ATOM's own runtime wholesale (graphs, bounded replay, the guarded
ops) and only swaps the layer class.

`get_model` picks it when the process has a plugin plan or boots for hot-plug
(`plugin_host.model_class_for`); without one ATOM's own model serves.
"""

import os

import torch

from atom import plugin_host, plugin_transition
from atom.config import get_current_atom_config
from atom.models.deepseek_v41.runtime import DeepseekV41RuntimeModel
from atom.models.deepseek_v41.runtime import RuntimeBlock as _NativeRuntimeBlock
from atom.utils.forward_context import get_forward_context

from .model import Block


class PluginRuntimeBlock(Block):
    """The plugin layer behind ATOM's guarded runtime ops (`v41_attention`,
    `v41_engram`), which call back into this layer's own modules."""

    attention_forward = _NativeRuntimeBlock.attention_forward
    engram_forward = _NativeRuntimeBlock.engram_forward


class DeepseekV41PluginRuntimeModel(DeepseekV41RuntimeModel):
    block_cls = PluginRuntimeBlock

    def _transition_wanted(self) -> bool:
        return (
            plugin_host.model_options().block_transition
            or os.environ.get("ATOM_V41_TRANSITION_AB") == "1"
        )

    def _transition_forward(self, input_ids, positions, inputs_embeds):
        """Non-TBO TP4, eager or full-forward graph with the Python backbone."""
        from aiter.dist.parallel_state import get_tp_group

        cfg = get_current_atom_config()
        if (
            cfg.compilation_config.level != 0
            or cfg.tensor_parallel_size != 4
            or cfg.parallel_config.data_parallel_size != 1
            or cfg.enable_dp_attention
            or cfg.enable_expert_parallel
            or cfg.enable_tbo
            or cfg.prefill_context_parallel_size != 1
            or cfg.decode_context_parallel_size != 1
            or cfg.pipeline_parallel_size != 1
        ):
            return self.backbone(input_ids, positions, inputs_embeds)
        forward = get_forward_context()
        group = get_tp_group()
        if group.world_size != 4 or group.cpu_group is None:
            return self.backbone(input_ids, positions, inputs_embeds)

        def eligible(pairs):
            ctx = forward.context
            captured, bucket = plugin_host._graph_state()
            # Capture routes by padded capacity; it cannot specialize on the
            # scheduled row count of a future replay. Native kernels retain
            # their device-side valid-row handling.
            if (
                ctx.is_prefill
                or (not captured and ctx.scheduled_tokens != input_ids.numel())
                or plugin_host._routes("v41.block.ffn", captured, bucket)
            ):
                return False
            for src, dst in pairs:
                if dst >= len(self.layers) or self.layers[dst].engram is not None:
                    return False
                moe = self.layers[src].ffn
                if (
                    moe._use_comm_fused_dispatch
                    or moe._use_dual_stream
                    or moe.tp_size != 4
                ):
                    return False
            return True

        def agree(token):
            agreed = [None] * group.world_size
            torch.distributed.all_gather_object(agreed, token, group=group.cpu_group)
            return agreed

        with plugin_transition.scope(
            eligible=eligible, rows=input_ids.numel(), agree=agree
        ):
            return self.backbone(input_ids, positions, inputs_embeds)

    def forward(self, input_ids, positions, inputs_embeds=None):
        if (
            not self.replay
            and not torch.compiler.is_compiling()
            and self._transition_wanted()
        ):
            return self._transition_forward(input_ids, positions, inputs_embeds)
        return super().forward(input_ids, positions, inputs_embeds)
