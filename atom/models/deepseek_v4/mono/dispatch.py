# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 Mono installation and deployment admission.

The executor stays in :mod:`runner`, its serving lifecycle in :mod:`session`,
and deployment contracts in :mod:`profile`. The native model runner calls this
installer after loading the model and drafter, just as for V4.1 and Kimi-K3.
Only an admitted deployment gets model and metadata-builder wrappers.
"""

from __future__ import annotations

import logging

from torch import nn

from atom.models.deepseek_v4.mono.profile import (
    deployment_refusal,
    select_profile,
)
from atom.mono.runtime.consensus import bind_agreed
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


def install_mono_decode(model, atom_config, drafter, *, metadata_builder=None):
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
    if metadata_builder is None:
        raise ValueError("V4 Mono installation requires the native metadata builder")
    from atom.models.deepseek_v4.mono.metadata import MonoMetadataBuilder
    from atom.models.deepseek_v4.mono.session import V4MonoSession

    metadata_builder = MonoMetadataBuilder(metadata_builder)
    session = V4MonoSession(model, atom_config, metadata_builder, profile)
    # Native's generic lookup schema conservatively marks weights mutable.
    # Preserve the loaded parameter and exact kernel, but use a read-only
    # schema so graph recording cannot invalidate the weight binding.
    from atom.models.deepseek_v4.mono.embedding import MonoEmbedding
    from atom.models.deepseek_v4.mono.projection import MonoIndexerProjection

    replacements = []

    def prepare_wrappers():
        # Construct and validate every wrapper before replacing any module.
        # A quantized indexer must refuse on every rank without partial edits.
        embedding = MonoEmbedding(model.model.embed)
        replacements.append((model.model, "embed", model.model.embed, embedding))
        for layer in model.model.layers:
            if layer.attn.indexer is not None:
                owner = layer.attn.indexer
                replacements.append(
                    (
                        owner,
                        "weights_proj",
                        owner.weights_proj,
                        MonoIndexerProjection(owner.weights_proj),
                    )
                )
        if drafter is not None:
            # Retain the original references for both native draft layouts.
            if profile.name.startswith("dspark_"):
                from atom.models.deepseek_v4.mono.norm import MonoRMSNorm

                inner = drafter.model.model
                replacements.append((inner, "embed", inner.embed, embedding))
                last = inner.mtp[-1]
                replacements.append((last, "norm", last.norm, MonoRMSNorm(last.norm)))
            else:
                for block in drafter.model.model.mtp:
                    replacements.append((block, "embed", block.embed, embedding))

    def replace_modules(index):
        for replacement in replacements:
            owner, name = replacement[:2]
            setattr(owner, name, replacement[index])

    def install_modules():
        replace_modules(3)
        if profile.name.startswith("dspark_"):
            from atom.models.deepseek_v4.mono.proposer import MonoDSparkProposer

            # The proposer rebinds native callbacks. Snapshot them before its
            # constructor so a peer's refusal rolls back the entire draft too.
            block = drafter.block
            for name in ("forward", "epilogue", "warmup_inputs"):
                original = getattr(block, name)
                replacements.append((block, name, original, original))
            runner = metadata_builder.model_runner
            replacements.append((runner, "drafter", drafter, drafter))
            runner.drafter = MonoDSparkProposer(drafter, session)

    installed = prepared = False
    try:
        bind_agreed(prepare_wrappers, session.group)
        installed = True
        bind_agreed(install_modules, session.group)
        # Binding sees the final read-only modules and the shared draft owner.
        # Any later geometry/device refusal rolls both back before Native runs.
        prepared = session.prepare_storage()
        if not prepared:
            return model
    except Exception:
        logger.exception("V4 Mono installation refused before publication")
        return model
    finally:
        if not prepared:
            try:
                if installed:
                    bind_agreed(lambda: replace_modules(2), session.group)
            finally:
                session.close()
    metadata_builder._mono_session = session
    # Publish only after preparation succeeds. Refused deployments retain the
    # exact native builder, whose buffers were allocated once by ModelRunner.
    metadata_builder.model_runner.attn_metadata_builder = metadata_builder
    return MonoDecodeModel(model, session)
