# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments and which steps the gfx1250 DeepSeek-V4.1 mono decode takes."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.deepseek_v41.mono import dispatch as common_dispatch
from atom.models.deepseek_v41.mono.gfx1250 import dspark_config as config
from atom.models.deepseek_v41.mono.gfx1250 import dspark_dispatch as dispatch
from atom.mono.runtime import deployment
from atom.mono.runtime.consensus import MonoUnsupported


def atom_config(**overrides):
    fields = dict(
        tensor_parallel_size=4,
        enable_expert_parallel=False,
        enable_dp_attention=False,
        parallel_config=NS(data_parallel_size=1),
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_tbo=False,
        enable_tbo_decode=False,
        compilation_config=NS(level=0),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
        speculative_config=NS(method="dspark", num_speculative_tokens=5),
        dspark=NS(confidence_schedule=False, ragged=False),
        max_model_len=1 << 20,
        hf_config=NS(
            sliding_window=128,
            index_topk=512,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            index_n_heads=32,
            index_head_dim=128,
        ),
    )
    fields.update(overrides)
    return NS(**fields)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(deployment, "_gfx", lambda: "gfx1250")
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for key, value in {
        "ATOM_MONO_ENABLE": "1",
        "ATOM_DSV41_SIDE_STREAMS": "0",
        "ATOM_MONO_CHECK": "0",
        "ATOM_MONO_DEBUG": "0",
        "ATOM_MONO_TIMELINE": "",
    }.items():
        monkeypatch.setenv(key, value)


def context(monkeypatch, **changes):
    step = NS(
        decode=True,
        width=6,
        max_q_len=6,
        cu_seqlens_q=torch.tensor([0, 6]),
        requests=[],
        tentative=True,
    )
    fwd = NS(
        context=NS(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=NS(step=step, image_mask=None),
    )
    for name, value in changes.items():
        if name == "prefill":
            fwd.context.is_prefill = value
        elif name == "ubatch":
            fwd.ubatch_slices = value
        elif name == "missing_step":
            fwd.attn_metadata.step = None
        elif name == "image_mask":
            fwd.attn_metadata.image_mask = value
        else:
            setattr(step, name, value)
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: fwd)
    return fwd


def test_default_config_and_capacity():
    assert config.config_refusal(atom_config()) is None
    assert (
        config.MAX_REQUESTS == 8 and config.MAX_ROWS == 48 and config.MAX_MOE_ROWS == 96
    )
    assert config.TARGET_WIDTHS == (6, 12, 48) and config.DRAFT_WIDTHS == (5, 10, 40)
    assert dispatch.V41MonoDecodeRunner.mhtjstest_widths[-1] == 48


@pytest.mark.parametrize(
    "fields",
    [
        {"tensor_parallel_size": 2},
        {"tensor_parallel_size": 8},
        {"kv_cache_dtype": "fp8"},
        {"index_cache_dtype": "bf16"},
        {"speculative_config": None},
        {"speculative_config": NS(method="mtp", num_speculative_tokens=5)},
        {"speculative_config": NS(method="dspark", num_speculative_tokens=3)},
        {"dspark": NS(confidence_schedule=True, ragged=False)},
        {"dspark": NS(confidence_schedule=False, ragged=True)},
        {"compilation_config": NS(level=2)},
        {"enable_expert_parallel": True},
        {"parallel_config": NS(data_parallel_size=2)},
    ],
)
def test_unsupported_deployments_fall_back(fields):
    assert config.config_refusal(atom_config(**fields))


@pytest.mark.parametrize(
    "key,value",
    [
        ("ATOM_MONO_ENABLE", "0"),
        ("ATOM_DSV41_SIDE_STREAMS", "1"),
        ("ATOM_MONO_CHECK", "1"),
        ("ATOM_MONO_DEBUG", "1"),
        ("ATOM_MONO_TIMELINE", "/task/timeline"),
    ],
)
def test_conflicting_switches_fall_back(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    assert config.config_refusal(atom_config())


def test_plugin_falls_back(monkeypatch):
    monkeypatch.setattr(deployment, "is_vllm", lambda: True)
    assert config.config_refusal(atom_config()) == "plugin mode"


@pytest.mark.parametrize("cu", [[0, 6], [0, 3, 6], [0, 1, 2, 3, 4, 5, 6]])
def test_decode_steps_route_on_graph_width(monkeypatch, cu):
    """As gfx950: the graph width, however its rows split into requests."""
    context(monkeypatch, cu_seqlens_q=torch.tensor(cu))
    assert dispatch.step_supported(torch.zeros(6, dtype=torch.int64), None)


@pytest.mark.parametrize("rows", [1, 5, 12, 24, 48])
def test_every_decode_width_up_to_48_is_supported(monkeypatch, rows):
    context(monkeypatch, width=rows, max_q_len=rows)
    assert dispatch.step_supported(torch.zeros(rows, dtype=torch.int64), None)


@pytest.mark.parametrize("rows", [54, 96])
def test_wider_steps_stay_native(monkeypatch, rows):
    context(monkeypatch, width=rows)
    assert not dispatch.step_supported(torch.zeros(rows, dtype=torch.int64), None)


@pytest.mark.parametrize(
    "changes",
    [
        {"prefill": True},
        {"ubatch": []},
        {"missing_step": True},
        {"image_mask": torch.zeros(6)},
        {"decode": False},
        {"width": 12},
    ],
)
def test_non_decode_steps_stay_native(monkeypatch, changes):
    context(monkeypatch, **changes)
    assert not dispatch.step_supported(torch.zeros(6, dtype=torch.int64), None)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [None] * 40
        self.calls = []

    def forward(self, ids, positions, inputs_embeds=None):
        self.calls.append((ids.numel(), inputs_embeds))
        return "native-target"


class Draft(nn.Module):
    block_size = 5

    def __init__(self):
        super().__init__()
        self.mtp = [None] * 3

    def block_backbone(self, ids, positions, width):
        return "native-draft"


@pytest.fixture
def runners(monkeypatch):
    prepared = []

    class TargetRunner:
        def __init__(self, *args):
            self.kind = "target"

        def prepare(self, rows):
            prepared.append((self.kind, rows))
            return True

        def forward(self, *args):
            return "mono-target"

    class DraftRunner(TargetRunner):
        def __init__(self, *args):
            self.kind = "draft"

        def backbone(self, *args):
            return "mono-draft"

    monkeypatch.setattr(dispatch, "V41MonoDecodeRunner", TargetRunner)
    monkeypatch.setattr(dispatch, "DraftMonoRunner", DraftRunner)
    return prepared


def test_regular_installer_prepares_enabled_widths_and_routes(monkeypatch, runners):
    model, draft = Model(), Draft()
    drafter = NS(model=draft)
    wrapped = common_dispatch.install_mono_decode(model, atom_config(), drafter)
    assert isinstance(wrapped, dispatch.MonoDecodeModel)
    assert isinstance(drafter.model, dispatch.MonoDraftModel)
    assert runners == [
        ("target", 6),
        ("target", 12),
        ("target", 48),
        ("draft", 5),
        ("draft", 10),
        ("draft", 40),
    ]
    for requests in (1, 2, 8):
        rows = 6 * requests
        context(monkeypatch, width=rows, cu_seqlens_q=torch.arange(requests + 1) * 6)
        assert wrapped(torch.zeros(rows), torch.zeros(rows)) == "mono-target"
        ids = torch.zeros(requests)
        assert drafter.model.block_backbone(ids, ids, None) == "mono-draft"
    for requests in (3, 4, 7):
        rows = 6 * requests
        context(monkeypatch, width=rows, cu_seqlens_q=torch.arange(requests + 1) * 6)
        assert wrapped(torch.zeros(rows), torch.zeros(rows)) == "native-target"
        ids = torch.zeros(requests)
        assert drafter.model.block_backbone(ids, ids, 5) == "native-draft"
    assert (
        drafter.model.block_backbone(torch.zeros(1), torch.zeros(1), 3)
        == "native-draft"
    )
    context(monkeypatch)
    embeds = torch.zeros(6, 2)
    assert wrapped(torch.zeros(6), torch.zeros(6), embeds) == "native-target"
    assert model.calls[-1][1] is embeds


def test_disabled_switch_skips_installer(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    monkeypatch.setattr(
        dispatch,
        "install_mono_decode",
        lambda *a, **k: pytest.fail("disabled installer called"),
    )
    model = Model()
    assert (
        common_dispatch.install_mono_decode(model, atom_config(), NS(model=Draft()))
        is model
    )


@pytest.mark.parametrize("failure", ["target", "draft"])
def test_refusal_preserves_original_wrappers(monkeypatch, failure):
    class Runner:
        def __init__(self, *args):
            if failure == "target":
                raise MonoUnsupported("test target refusal")

        def prepare(self, rows):
            return rows != 5

    monkeypatch.setattr(dispatch, "V41MonoDecodeRunner", Runner)
    monkeypatch.setattr(dispatch, "DraftMonoRunner", Runner)
    model, draft = Model(), Draft()
    drafter = NS(model=draft)
    assert dispatch.install_mono_decode(model, atom_config(), drafter) is model
    assert drafter.model is draft


def test_wrong_model_topology_stays_native(runners):
    model = Model()
    model.layers.pop()
    assert (
        dispatch.install_mono_decode(model, atom_config(), NS(model=Draft())) is model
    )
    assert not runners
