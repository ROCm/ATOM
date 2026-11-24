# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""R3 (record / replay router) orchestration for MoE training and rollout."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from torch import Tensor, nn

from atomrl.core.config import R3Config
from atomrl.core.protocol import DataProto
from atomrl.moe.router_recorder import RouterRecorder
from atomrl.moe.router_replayer import RouterReplayer

logger = logging.getLogger(__name__)


class R3Manager:
    """Owns :class:`RouterRecorder` and :class:`RouterReplayer` lifecycles for R3."""

    def __init__(self, config: R3Config) -> None:
        self._cfg = config
        self._recorder = RouterRecorder()
        self._replayer = RouterReplayer()

    @property
    def recorder(self) -> RouterRecorder:
        return self._recorder

    @property
    def replayer(self) -> RouterReplayer:
        return self._replayer

    @contextmanager
    def record_phase(self, model: nn.Module) -> Iterator[RouterRecorder]:
        """Install recorder hooks for the duration of the context block."""
        if not self._cfg.enabled or not self._cfg.record_router_logits:
            logger.debug("R3 record_phase: disabled or record_router_logits=False.")
            yield self._recorder
            return
        with self._recorder.recording(model):
            logger.debug(
                "R3 record_phase: hooks active (replay_mode=%s).", self._cfg.replay_mode
            )
            yield self._recorder

    @contextmanager
    def replay_phase(
        self,
        model: nn.Module,
        distributions: dict[int, Tensor],
    ) -> Iterator[RouterReplayer]:
        """Install replayer hooks that inject recorded logits."""
        if not self._cfg.enabled:
            logger.debug("R3 replay_phase: disabled.")
            yield self._replayer
            return
        self._replayer.install_hooks(model, distributions)
        try:
            logger.debug(
                "R3 replay_phase: replaying %d layer distributions.", len(distributions)
            )
            yield self._replayer
        finally:
            self._replayer.remove_hooks()

    @staticmethod
    def transfer_distributions(
        data: DataProto, recorded: dict[int, Tensor]
    ) -> DataProto:
        """Attach ``recorded`` router tensors to ``data`` using the DataProto R3 helpers."""
        out = DataProto(tensors=dict(data.tensors), meta=dict(data.meta))
        for layer_idx, logits in recorded.items():
            out.add_router_distributions(int(layer_idx), logits)
        return out

    def clear(self) -> None:
        """Clear recorder buffers and ensure hooks are removed."""
        self._recorder.clear()
        self._recorder.remove_hooks()
        self._replayer.remove_hooks()
