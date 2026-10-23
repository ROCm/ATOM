# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP1 execution for the shared Qwen Mono runner."""

from dataclasses import dataclass

from atom.models.qwen4_exp_mono import batch, contract
from atom.models.qwen4_exp_mono.execution import LayerExecution
from atom.models.qwen4_exp_mono.kernels import gdn_layer, qsa_post, qsa_pre
from atom.models.qwen4_exp_mono.plan import LayerKind
from atom.models.qwen4_exp_mono.shapes import (
    BATCH_ROWS,
    QSA_DIRECT_CONTEXT,
)


@dataclass(frozen=True)
class TP1Build:
    gdn_layer: object
    qsa_pre: object
    qsa_post: object


class TP1Execution(LayerExecution):
    """TP1 kernel construction and layer launches for the shared model runner."""

    def build(self, tokens: int) -> TP1Build | batch.BatchBuild:
        if tokens in BATCH_ROWS:
            return batch.build(tokens, self.runner.device, tp=1)
        gdn = gdn_layer.build_gdn_layer(self.runner._build_key(tokens, LayerKind.GDN))
        pre = qsa_pre.build_qsa_pre(self.runner._build_key(tokens, LayerKind.QSA))
        post = qsa_post.build_qsa_post(
            self.runner._build_key(tokens, LayerKind.QSA),
            direct=self.runner.max_model_len <= QSA_DIRECT_CONTEXT,
        )
        return TP1Build(gdn, pre, post)

    @staticmethod
    def kernels(build):
        if isinstance(build, batch.BatchBuild):
            return list(build.kernels())
        return [
            (build.gdn_layer, gdn_layer.ABI),
            (build.qsa_pre, qsa_pre.ABI),
            (build.qsa_post, qsa_post.ABI),
        ]

    def check_resources(self, build, tokens):
        if isinstance(build, batch.BatchBuild):
            contract.check_batch(build)
            return
        contract.check_launcher(
            build.gdn_layer, max_sgpr_spill=contract.sgpr_spill_limit(tokens, "gdn")
        )
        contract.check_launcher(
            build.qsa_pre, max_sgpr_spill=contract.sgpr_spill_limit(tokens, "qsa_pre")
        )
        capacity = self.runner.max_model_len
        stage = (
            "qsa_direct"
            if capacity <= QSA_DIRECT_CONTEXT
            else "qsa_full" if capacity <= 8192 else "qsa_tree"
        )
        contract.check_launcher(
            build.qsa_post, max_sgpr_spill=contract.sgpr_spill_limit(tokens, stage)
        )
