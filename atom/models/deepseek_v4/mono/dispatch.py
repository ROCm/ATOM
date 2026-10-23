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
from atom.models.deepseek_v4.mono.session import V4MonoSession
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
        mono.engine.prepare_model_forward()
        if mono.armed:
            output = mono.forward(input_ids, positions)
            mono.engine.model_forwarded()
            return output
        return self.model(input_ids, positions)

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
    session = V4MonoSession(model, atom_config, metadata_builder, profile)
    # The engine keeps one private execution capability, not a V4 model probe.
    # Installation uses the builder's existing owner; no extra installer args.
    metadata_builder.model_runner._mono_execution = session.engine
    if profile.name.startswith("dspark_"):
        from atom.models.deepseek_v4.mono.proposer import (
            MonoDSparkProposer,
            MonoMetadataBuilder,
        )

        builder = MonoMetadataBuilder(metadata_builder, session)
        session.builder = builder
        metadata_builder.model_runner.attn_metadata_builder = builder
        metadata_builder.model_runner.drafter = MonoDSparkProposer(drafter, session)
    return MonoDecodeModel(model, session)
