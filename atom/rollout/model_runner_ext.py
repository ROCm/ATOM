# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from atom.model_engine.model_runner import ModelRunner
from atom.rollout.memory_manager import MemoryManagerMixin
from atom.rollout.weight_updater import WeightUpdaterMixin


class RLHFModelRunner(ModelRunner, WeightUpdaterMixin, MemoryManagerMixin):
    """ModelRunner with RLHF extensions (weight sync + memory lifecycle).

    Only used when ATOM is driven by an external RLHF framework (e.g. verl).
    Pure inference deployments use the base ModelRunner, which carries no
    RLHF-specific code.
    """
    pass
