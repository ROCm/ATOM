# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for MiniMax-M3 SwiGLU-OAI MXFP8 fusion."""

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
def test_shared_fusion_requires_opt_in_and_eligible_input(
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


@pytest.mark.parametrize(("value", "enabled"), [("0", False), ("1", True)])
def test_shared_expert_construction_reads_common_fusion_flag(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    enabled: bool,
) -> None:
    monkeypatch.setenv("ATOM_MINIMAX_M3_SWIGLU_MXFP8", value)
    monkeypatch.setattr(m3_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(m3_model, "ReplicatedLinear", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        m3_model,
        "FusedMoE",
        lambda **kwargs: SimpleNamespace(
            quant_method=None,
            num_fused_shared_experts=0,
        ),
    )
    mlp_calls: list[dict[str, object]] = []

    def shared_mlp(**kwargs: object) -> object:
        mlp_calls.append(kwargs)
        return object()

    monkeypatch.setattr(m3_model, "MiniMaxM3MLP", shared_mlp)
    config = SimpleNamespace(
        hidden_size=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        intermediate_size=32,
        n_shared_experts=1,
    )

    m3_model.MiniMaxM3MoE(config, layer_id=0)

    assert mlp_calls[0]["fuse_swiglu_mxfp8"] is enabled


@pytest.mark.parametrize(("value", "enabled"), [("0", False), ("1", True)])
def test_dense_mlp_construction_reads_common_fusion_flag(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    enabled: bool,
) -> None:
    monkeypatch.setenv("ATOM_MINIMAX_M3_SWIGLU_MXFP8", value)
    monkeypatch.setattr(m3_model, "_sparse_attention_layer_ids", lambda _: set())
    monkeypatch.setattr(m3_model, "MiniMaxM3Attention", lambda **kwargs: object())
    monkeypatch.setattr(m3_model, "GemmaRMSNorm", lambda *args, **kwargs: object())
    mlp_calls: list[dict[str, object]] = []

    def dense_mlp(**kwargs: object) -> object:
        mlp_calls.append(kwargs)
        return object()

    monkeypatch.setattr(m3_model, "MiniMaxM3MLP", dense_mlp)
    config = SimpleNamespace(
        hidden_size=64,
        dense_intermediate_size=3072,
        moe_layer_freq=[0],
        rms_norm_eps=1e-6,
    )

    m3_model.MiniMaxM3DecoderLayer(config, prefix="", layer_num=0)

    assert mlp_calls[0]["fuse_swiglu_mxfp8"] is enabled
