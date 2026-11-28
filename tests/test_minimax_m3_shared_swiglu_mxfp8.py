# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which MiniMax-M3 MLP calls take the SwiGLU-OAI + MXFP8 kernel (CPU)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.models.minimax_m3 import model as m3_model
from atom.models.minimax_m3.model import MiniMaxM3MLP


class _GateUp:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def __call__(
        self, x: torch.Tensor, x_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.calls.append((x, x_scale))
        return x


class _DownProj:
    def __init__(
        self,
        *,
        native_group_rows: int | None = 1,
        runs_bf16: bool = False,
    ) -> None:
        self.quant_type = m3_model.QuantType.per_1x32
        self.params_dtype = m3_model.dtypes.fp8
        self.native_a8_group_rows = native_group_rows
        self.runs_bf16 = runs_bf16
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def __call__(
        self, x: torch.Tensor, x_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.calls.append((x, x_scale))
        return x


def _record_swiglu(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def split(
        gate_up: torch.Tensor,
        alpha: float,
        beta: float,
        limit: float | None,
    ) -> torch.Tensor:
        del alpha, beta, limit
        calls.append("split")
        return gate_up[:, : gate_up.shape[-1] // 2]

    def fused(
        gate_up: torch.Tensor,
        alpha: float,
        beta: float,
        limit: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del alpha, beta, limit
        calls.append("fused")
        intermediate = gate_up.shape[-1] // 2
        return (
            torch.zeros((gate_up.shape[0], intermediate), dtype=torch.uint8),
            torch.zeros((gate_up.shape[0], intermediate // 32), dtype=torch.uint8),
        )

    monkeypatch.setattr(m3_model, "swiglu_oai_split", split)
    monkeypatch.setattr(m3_model, "swiglu_oai_split_mxfp8", fused)
    return calls


def _mlp(*, enabled: bool, native_group_rows: int | None = 1) -> SimpleNamespace:
    return SimpleNamespace(
        gate_up_proj=_GateUp(),
        down_proj=_DownProj(native_group_rows=native_group_rows),
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
        fuse_swiglu_mxfp8=enabled,
    )


@pytest.mark.parametrize(
    ("enabled", "input_eligible", "expected"),
    [
        (False, True, "split"),
        (True, False, "split"),
        (True, True, "fused"),
    ],
)
def test_fusion_needs_gfx1250_and_an_eligible_input(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    input_eligible: bool,
    expected: str,
) -> None:
    calls = _record_swiglu(monkeypatch)
    monkeypatch.setattr(
        m3_model, "swiglu_oai_mxfp8_supported", lambda _: input_eligible
    )
    mlp = _mlp(enabled=enabled)

    MiniMaxM3MLP.forward(mlp, torch.randn(3, 64))

    assert calls == [expected]
    _, scale = mlp.down_proj.calls[0]
    assert (scale is not None) is (expected == "fused")


def test_ineligible_down_projection_keeps_existing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_swiglu(monkeypatch)
    monkeypatch.setattr(m3_model, "swiglu_oai_mxfp8_supported", lambda _: True)
    mlp = _mlp(enabled=True, native_group_rows=None)
    x_scale = torch.ones(2, 1)

    MiniMaxM3MLP.forward(mlp, torch.randn(2, 64), x_scale=x_scale)

    assert calls == ["split"]
    assert mlp.gate_up_proj.calls[0][1] is x_scale
    assert mlp.down_proj.calls[0][1] is None


def test_native_down_projection_contract_is_conservative() -> None:
    down = _DownProj()
    assert m3_model._linear_consumes_native_mxfp8(down)

    down.native_a8_group_rows = 32
    assert not m3_model._linear_consumes_native_mxfp8(down)
    down.native_a8_group_rows = 1
    down.runs_bf16 = True
    assert not m3_model._linear_consumes_native_mxfp8(down)
    down.runs_bf16 = False
    down.params_dtype = torch.bfloat16
    assert not m3_model._linear_consumes_native_mxfp8(down)
