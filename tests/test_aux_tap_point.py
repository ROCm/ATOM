# SPDX-License-Identifier: MIT
"""A target's own tap point for a layer's post-layer hidden state
(``aux_tap_point``): ``install_aux_taps`` hooks it for specs that declare that
meaning (``post_layer_hidden``), and the layer itself otherwise.

Not CPU-only, like test_pp_aux_relay.py: `spec_decode.drafter` imports the
model loader and so the real aiter; skips where it cannot be imported."""

import pytest
import torch
from torch import nn

pytest.importorskip(
    "atom.spec_decode.drafter",
    reason="requires AITER to import the drafter chain",
    exc_type=ImportError,
)

from atom.spec_decode.drafter import AuxCaptureSpec, install_aux_taps


class _Target(nn.Module):
    def __init__(self, n_layers: int, with_tap_points: bool):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(n_layers))
        self.points = nn.ModuleList(nn.Identity() for _ in range(n_layers))
        if with_tap_points:
            # layer k's output is read from point k + 1, the last layer's from
            # the final point; layer 1 keeps its own tap
            self.aux_tap_point = lambda lid: (
                None if lid == 1 else (self.points[lid], lambda out, _m: out)
            )


def _spec(post_layer_hidden: bool, capture: str = "output") -> AuxCaptureSpec:
    return AuxCaptureSpec(
        layer_ids=(0, 1, 2),
        hidden_size=2,
        extract=lambda out, _m: out,
        capture=capture,
        post_layer_hidden=post_layer_hidden,
    )


def _hooked(model: _Target):
    hooked = {
        name
        for name, m in model.named_modules()
        if m._forward_hooks or m._forward_pre_hooks
    }
    return sorted(hooked)


def _install(model: _Target, spec: AuxCaptureSpec):
    install_aux_taps(model, spec, spec.layer_ids, lambda lid, extract: lambda *a: None)


def test_tap_points_replace_the_layers_for_post_layer_hidden_specs():
    model = _Target(3, with_tap_points=True)
    _install(model, _spec(post_layer_hidden=True))
    assert _hooked(model) == ["layers.1", "points.0", "points.2"]


def test_other_specs_keep_the_layer_taps():
    model = _Target(3, with_tap_points=True)
    _install(model, _spec(post_layer_hidden=False))
    assert _hooked(model) == ["layers.0", "layers.1", "layers.2"]
    model = _Target(3, with_tap_points=True)
    _install(model, _spec(post_layer_hidden=True, capture="input"))
    assert _hooked(model) == ["layers.0", "layers.1", "layers.2"]


def test_targets_without_tap_points_keep_the_layer_taps():
    model = _Target(3, with_tap_points=False)
    _install(model, _spec(post_layer_hidden=True))
    assert _hooked(model) == ["layers.0", "layers.1", "layers.2"]
    assert torch.is_tensor(model.layers[0].weight)
