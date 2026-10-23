# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8 execution for the shared Qwen Mono runner."""

from dataclasses import dataclass

import torch

from atom.models.qwen4_exp_mono import batch, contract
from atom.models.qwen4_exp_mono.execution import LayerExecution
from atom.models.qwen4_exp_mono.kernels import collective, finish, gdn_tp8, qsa_tp8
from atom.models.qwen4_exp_mono.plan import LayerKind
from atom.models.qwen4_exp_mono.shapes import BATCH_ROWS
from atom.models.qwen4_exp_mono.state import StateView


@dataclass(frozen=True)
class TP8Build:
    tp8_finish: object
    tp8_pull: object
    tp8_gdn_k1: object
    tp8_gdn_k2: object
    tp8_qsa_k1: object
    tp8_qsa_k2: object


class TP8Execution(LayerExecution):
    """Persistent S1 and ordinary B2/B4 builds for the shared model runner."""

    def build(self, tokens: int) -> TP8Build | batch.BatchBuild:
        if tokens in BATCH_ROWS:
            return batch.build(tokens, self.runner.device)
        key = self.runner._build_key(tokens, LayerKind.GDN)
        return TP8Build(
            tp8_finish=finish.build_finish(key),
            tp8_pull=collective.build_pull(tokens, self.runner.tp),
            tp8_gdn_k1=gdn_tp8.build_gdn_k1(key),
            tp8_gdn_k2=gdn_tp8.build_gdn_k2(key),
            tp8_qsa_k1=qsa_tp8.build_qsa_k1(
                self.runner._build_key(tokens, LayerKind.QSA)
            ),
            tp8_qsa_k2=qsa_tp8.build_qsa_k2(
                self.runner._build_key(tokens, LayerKind.QSA)
            ),
        )

    @staticmethod
    def kernels(build):
        if isinstance(build, batch.BatchBuild):
            return list(build.kernels())
        return [
            (build.tp8_finish, finish.TP8_ABI),
            (build.tp8_pull, collective.PULL_ABI),
            (build.tp8_gdn_k1, gdn_tp8.K1_ABI),
            (build.tp8_gdn_k2, gdn_tp8.K2_ABI),
            (build.tp8_qsa_k1, qsa_tp8.ABI),
            (build.tp8_qsa_k2, qsa_tp8.K2_ABI),
        ]

    @staticmethod
    def check_resources(build, tokens):
        if isinstance(build, batch.BatchBuild):
            contract.check_batch(build)
            return
        contract.check_tp8(build)

    def _tp8_pull(self, layer_id: int, output: torch.Tensor, view: StateView) -> None:
        build = self.runner.widths[view.tokens]
        build.tp8_pull(
            self.runner.state.peers.addresses.data_ptr(),
            output.data_ptr(),
            layer_id,
            stream=torch.cuda.current_stream(),
        )
