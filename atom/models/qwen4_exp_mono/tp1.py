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
    qsa_post: dict[int, object]


class TP1Execution(LayerExecution):
    """TP1 kernel construction and layer launches for the shared model runner."""

    def build(self, tokens: int) -> TP1Build | batch.BatchBuild:
        if tokens in BATCH_ROWS:
            return batch.build(tokens, self.runner.device, tp=1)
        gdn = gdn_layer.build_gdn_layer(self.runner._build_key(tokens, LayerKind.GDN))
        pre = qsa_pre.build_qsa_pre(self.runner._build_key(tokens, LayerKind.QSA))
        # Context buckets share one launcher for each numerical schedule.
        posts = {}
        for context in self.runner.context_buckets:
            direct = context <= QSA_DIRECT_CONTEXT
            if direct not in posts:
                posts[direct] = qsa_post.build_qsa_post(
                    self.runner._build_key(tokens, LayerKind.QSA), direct=direct
                )
        return TP1Build(
            gdn,
            pre,
            {
                context: posts[context <= QSA_DIRECT_CONTEXT]
                for context in self.runner.context_buckets
            },
        )

    @staticmethod
    def kernels(build):
        if isinstance(build, batch.BatchBuild):
            return list(build.kernels())
        return [
            (build.gdn_layer, gdn_layer.ABI),
            (build.qsa_pre, qsa_pre.ABI),
            *[
                (launcher, qsa_post.ABI)
                for launcher in dict.fromkeys(build.qsa_post.values())
            ],
        ]

    @staticmethod
    def check_resources(build, tokens):
        if isinstance(build, batch.BatchBuild):
            contract.check_batch(build)
            return
        contract.check_launcher(
            build.gdn_layer, max_sgpr_spill=contract.sgpr_spill_limit(tokens, "gdn")
        )
        contract.check_launcher(
            build.qsa_pre, max_sgpr_spill=contract.sgpr_spill_limit(tokens, "qsa_pre")
        )
        for launcher in dict.fromkeys(build.qsa_post.values()):
            # The shared launcher is compiled for deployment capacity.
            context = max(
                context
                for context, bound in build.qsa_post.items()
                if bound is launcher
            )
            stage = (
                "qsa_direct"
                if context <= QSA_DIRECT_CONTEXT
                else "qsa_full" if context <= 8192 else "qsa_tree"
            )
            contract.check_launcher(
                launcher, max_sgpr_spill=contract.sgpr_spill_limit(tokens, stage)
            )
