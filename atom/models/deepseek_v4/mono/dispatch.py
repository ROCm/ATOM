# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 Mono installation and deployment admission.

The executor stays in :mod:`runner`, its serving lifecycle in :mod:`session`,
and deployment contracts in :mod:`profile`. This module applies the routing
policy and installs a session only for an admitted deployment.
"""

from __future__ import annotations

import logging

from torch import nn

from atom.models.deepseek_v4.mono.profile import (
    deployment_refusal,
    select_profile,
)
from atom.utils import envs

logger = logging.getLogger("atom")


class MonoDecodeModel(nn.Module):
    """Route published V4 steps outside the native model's compile boundary.

    Like V4.1, keep the loaded model in ``model`` and delegate native methods.
    The private session owns V4 cache publication and graph lifetimes.
    """

    def __init__(self, model, session):
        super().__init__()
        self.model = model
        self._mono = session

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, input_ids, positions):
        mono = self._mono
        try:
            mono.builder.prepare_mono()
            if mono.adapter is None:
                return self.model(input_ids, positions)
            mono.engine.prepare_model_forward(input_ids, positions)
            if mono.armed:
                output = mono.forward(input_ids, positions)
                mono.engine.model_forwarded()
                return output
            return self.model(input_ids, positions)
        except BaseException:
            if mono.engine._capture_selected is not None:
                mono.disabled = True
                mono.engine._capture_key = mono.engine._capture_selected = None
                mono.finish_capture_publication()
            raise
        finally:
            # Capture warmups deliberately retain their publication for the
            # actual recording. Live eager authorization is always one-use.
            if mono.engine._capture_selected is None:
                mono.engine.abort_step()

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)


def install_mono_decode(model, atom_config, drafter, metadata_builder):
    """Install the pure-TP8 executor under the configured routing policy."""
    if not envs.ATOM_MONO_ENABLE:
        logger.info("V4 Mono off: ATOM_MONO_ENABLE")
        return model
    # Use the same refusal as V4.1: bind-time weight copies are not a
    # qualified weight-sync/sleep contract, so rollout keeps Native.
    from atom.mono.runtime.deployment import _takes_weight_updates

    if _takes_weight_updates(atom_config):
        logger.info("V4 Mono off: rollout weight updates")
        return model
    profile = select_profile(atom_config, drafter)
    why = deployment_refusal(atom_config, profile, model_args=model.args)
    if why is not None:
        logger.info("V4 Mono auto kept native path: %s", why)
        return model
    from atom.models.deepseek_v4.mono.session import V4MonoSession

    session = V4MonoSession(model, atom_config, metadata_builder, profile)
    # Repack before native warmup and KV sizing so the existing memory profile
    # includes persistent Mono weights alongside peak native activations.
    # Cache/aux binding and execution ownership stay deferred until KV is bound.
    if not session.prepare_weights():
        return model
    # Native's generic lookup schema conservatively marks weights mutable.
    # Preserve the loaded parameter and exact kernel, but use a read-only
    # schema so graph recording cannot invalidate the weight binding.
    from atom.models.deepseek_v4.mono.embedding import MonoEmbedding
    from atom.models.deepseek_v4.mono.projection import MonoIndexerProjection

    model.model.embed = MonoEmbedding(model.model.embed)
    for layer in model.model.layers:
        if layer.attn.indexer is not None:
            layer.attn.indexer.weights_proj = MonoIndexerProjection(
                layer.attn.indexer.weights_proj
            )
    if drafter is not None:
        drafter.model.share_with_target(model, set())
    metadata_builder._mono_session = session
    if profile.name.startswith("dspark_"):
        from atom.models.deepseek_v4.mono.proposer import (
            MonoDSparkProposer,
        )

        metadata_builder.model_runner.drafter = MonoDSparkProposer(drafter, session)
    return MonoDecodeModel(model, session)
