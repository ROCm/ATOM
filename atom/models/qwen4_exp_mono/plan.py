# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile-time layer kinds and the persistent task coverage of a Qwen model."""

from dataclasses import dataclass
from enum import Enum

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.shapes import LAYERS, PERSISTENT_ROWS
from atom.mono.runtime.consensus import MonoUnsupported


class LayerKind(Enum):
    GDN = "gdn"
    PLE_GDN = "ple_gdn"
    QSA = "qsa"


class ExpertLayout(Enum):
    REPLICATED = "replicated"
    EXPERT_ID = "expert_id"

    @staticmethod
    def for_tp(tp: int) -> "ExpertLayout":
        return ExpertLayout.REPLICATED if tp == 1 else ExpertLayout.EXPERT_ID


@dataclass(frozen=True)
class MonoBuild:
    """One compile-time token width and Qwen layer kind."""

    tokens: int
    kind: LayerKind
    tp: int = 1
    expert_layout: ExpertLayout = ExpertLayout.REPLICATED
    max_model_len: int = 8192

    def __post_init__(self) -> None:
        if self.tokens not in PERSISTENT_ROWS:
            raise ValueError(f"unsupported Qwen mono width {self.tokens}")
        dims = Dims(self.tp)
        if self.max_model_len <= 0:
            raise ValueError("Qwen mono deployment capacity must be positive")
        if self.expert_layout.value != dims.expert_layout:
            raise ValueError(
                f"TP{self.tp} requires {dims.expert_layout} expert layout, got "
                f"{self.expert_layout.value}"
            )

    @property
    def dims(self) -> Dims:
        return Dims(self.tp)


@dataclass(frozen=True)
class LayerTask:
    layer_id: int
    kind: LayerKind

    @property
    def is_gdn(self) -> bool:
        return self.kind is not LayerKind.QSA

    @property
    def ple(self) -> bool:
        return self.kind is LayerKind.PLE_GDN


@dataclass(frozen=True)
class BoundLayer:
    """One immutable host record; loaded modules and weights retain ownership."""

    task: LayerTask
    layer: object
    weights: object


@dataclass(frozen=True)
class TaskPlan:
    layers: tuple[LayerTask, ...]

    def validate(self) -> None:
        if tuple(task.layer_id for task in self.layers) != tuple(range(LAYERS)):
            raise MonoUnsupported("Qwen mono task plan does not cover layers 0..47")
        if sum(task.is_gdn for task in self.layers) != 36:
            raise MonoUnsupported("Qwen mono task plan does not have 36 GDN layers")
        if sum(task.ple for task in self.layers) != 1:
            raise MonoUnsupported("Qwen mono task plan does not have one PLE-GDN layer")
        if sum(task.kind is LayerKind.QSA for task in self.layers) != 12:
            raise MonoUnsupported("Qwen mono task plan does not have 12 QSA layers")


def classify_layer(layer, layer_id: int) -> LayerTask:
    kind = getattr(layer, "layer_type", None)
    ple = getattr(layer, "ple", None) is not None
    if kind == "linear_attention":
        build_kind = LayerKind.PLE_GDN if ple else LayerKind.GDN
    elif kind in ("full_attention", "qwen_sparse_attention"):
        if ple:
            raise MonoUnsupported(f"layer {layer_id}: PLE on QSA is unsupported")
        build_kind = LayerKind.QSA
    else:
        raise MonoUnsupported(f"layer {layer_id}: unknown attention kind {kind!r}")
    return LayerTask(layer_id, build_kind)


def make_plan(layers) -> TaskPlan:
    plan = TaskPlan(tuple(classify_layer(layer, i) for i, layer in enumerate(layers)))
    plan.validate()
    return plan
