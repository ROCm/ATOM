# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Public API for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atom.models.glm52.mono.kernels.op import Glm5MonoKernel
    from atom.models.glm52.mono.kernels.tp4_op import Glm5TP4MonoKernel
    from atom.mono.weights.layer import LayerWeights

__all__ = ["Glm5MonoKernel", "Glm5TP4MonoKernel", "LayerWeights"]


def __getattr__(name: str):
    """Load GPU wrappers only when callers request them."""

    if name == "Glm5MonoKernel":
        from atom.models.glm52.mono.kernels.op import Glm5MonoKernel

        return Glm5MonoKernel
    if name == "Glm5TP4MonoKernel":
        from atom.models.glm52.mono.kernels.tp4_op import Glm5TP4MonoKernel

        return Glm5TP4MonoKernel
    if name == "LayerWeights":
        from atom.mono.weights.layer import LayerWeights

        return LayerWeights
    raise AttributeError(name)
