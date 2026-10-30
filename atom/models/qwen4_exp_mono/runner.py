# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen decode runner with persistent S1 and weight-reusing TP1/TP8 batches."""

import logging

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.models.qwen4_exp_mono.config import enabled_widths
from atom.models.qwen4_exp_mono.plan import (
    BoundLayer,
    ExpertLayout,
    LayerKind,
    MonoBuild,
    make_plan,
)
from atom.models.qwen4_exp_mono.shapes import SUPPORTED_TP
from atom.models.qwen4_exp_mono.state import MonoState
from atom.models.qwen4_exp_mono.step_meta import (
    gdn_step_view,
    qsa_step_view,
)
from atom.models.qwen4_exp_mono.tp1 import TP1Execution
from atom.models.qwen4_exp_mono.tp8 import TP8Execution
from atom.models.qwen4_exp_mono.weights import bind_model
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class QwenMonoRunner:
    """Run persistent GDN or two-launch QSA layers for an accepted width."""

    def __init__(self, model, max_model_len=8192) -> None:
        self.model = model
        self.max_model_len = max_model_len
        self.rank = get_tensor_model_parallel_rank()
        self.tp = get_tensor_model_parallel_world_size()
        self.group = get_tp_group().cpu_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        bind_agreed(self._bind, self.group)
        self.state = MonoState(self, self.group, self.rank, self.tp, self.device)
        self.execution = TP8Execution(self) if self.tp == 8 else TP1Execution(self)
        self.widths = WidthBuilds(
            self._build_width,
            self.execution.kernels,
            self.group,
            "Qwen mono",
            check_resources=self.execution.check_resources,
        )
        self._closed = False
        self._running = False
        self.check = envs.ATOM_MONO_CHECK

    def _bind(self) -> None:
        if self.tp not in SUPPORTED_TP:
            raise MonoUnsupported(f"Qwen mono has no TP{self.tp} kernel ABI")
        props = torch.cuda.get_device_properties(self.device)
        arch = getattr(props, "gcnArchName", "").split(":")[0]
        if torch.version.hip is None or arch != "gfx950":
            raise MonoUnsupported(
                f"Qwen mono requires ROCm gfx950, got {arch or 'unknown'}"
            )
        if props.multi_processor_count != BLOCKS:
            raise MonoUnsupported(
                f"{props.multi_processor_count} CUs: Qwen mono needs all "
                f"{BLOCKS} CTAs co-resident"
            )
        if self.model.model.return_hc_state:
            raise MonoUnsupported("Qwen mono does not serve speculative target output")
        self.layers = self.model.model.layers
        self.plan = make_plan(self.layers)
        self.weights = bind_model(self.model, self.plan, self.tp)
        self.bound_layers = tuple(
            BoundLayer(task, layer, weights)
            for task, layer, weights in zip(
                self.plan.layers, self.layers, self.weights, strict=True
            )
        )

    def _build_key(self, tokens: int, kind: LayerKind) -> MonoBuild:
        return MonoBuild(
            tokens,
            kind,
            tp=self.tp,
            expert_layout=ExpertLayout.for_tp(self.tp),
            max_model_len=self.max_model_len if kind is LayerKind.QSA else 8192,
        )

    def _build_width(self, tokens: int):
        if tokens not in enabled_widths(self.tp):
            raise ValueError(f"Qwen MonoKernel has no {tokens}-token build")
        return self.execution.build(tokens)

    def prepare(self, tokens: int) -> bool:
        if tokens not in enabled_widths(self.tp):
            return False
        return self.widths.prepare(tokens)

    def close(self) -> None:
        if self._closed:
            return
        if self._running:
            raise RuntimeError("cannot close an active Qwen mono runner")
        self.state.close()
        self._closed = True

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        if self._closed:
            raise RuntimeError("closed Qwen mono runner")
        if self._running:
            raise RuntimeError("nested Qwen mono forward")
        self._running = True
        try:
            model = self.model.model
            tokens = input_ids.numel()
            if tokens == 1:
                view = self.state.view(tokens)
                self.state.begin_step(tokens)
            else:
                view = self.widths[tokens]
            context = get_forward_context()
            # Validate all step views before the first layer can mutate model state.
            steps = tuple(
                (
                    gdn_step_view(context, bound.task.layer_id, tokens, tp=self.tp)
                    if bound.task.is_gdn
                    else (
                        qsa_step_view(
                            context,
                            bound.layer.self_attn,
                            tokens,
                            tp=self.tp,
                            max_model_len=self.max_model_len,
                        )
                        if tokens == 1
                        else None
                    )
                )
                for bound in self.bound_layers
            )
            hidden_states = model.get_input_embeddings(input_ids).repeat(
                1, model.hc_count
            )
            pending = None
            for bound, step in zip(self.bound_layers, steps, strict=True):
                hidden_states, pending = self.execution.layer(
                    bound.task,
                    bound.layer,
                    bound.weights,
                    step,
                    positions,
                    hidden_states,
                    input_ids,
                    pending,
                    view,
                )
            self.execution.finish_step(pending, view)
            mixed, _ = self.execution.mix(
                model.hyper_connection_mixer, hidden_states, pending, view
            )
            return mixed
        finally:
            self._running = False
