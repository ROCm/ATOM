# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for SGLang-to-ATOM forward normalization."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from atom.models.utils import IntermediateTensors


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_forward_context_module():
    @contextmanager
    def bind_current_forward_batch(_forward_batch):
        yield

    fake_modules = {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.model_executor": _package("sglang.srt.model_executor"),
        "sglang.srt.model_executor.forward_batch_info": ModuleType(
            "sglang.srt.model_executor.forward_batch_info"
        ),
        "atom.plugin": _package("atom.plugin"),
        "atom.plugin.sglang": _package("atom.plugin.sglang"),
        "atom.plugin.sglang.runtime": _package("atom.plugin.sglang.runtime"),
        "atom.plugin.sglang.runtime.context": ModuleType(
            "atom.plugin.sglang.runtime.context"
        ),
    }
    fake_modules["sglang.srt.model_executor.forward_batch_info"].ForwardBatch = object
    fake_modules["atom.plugin.sglang.runtime.context"].bind_current_forward_batch = (
        bind_current_forward_batch
    )

    module_name = "_test_sglang_runtime_forward_context"
    module_path = (
        Path(__file__).parents[2]
        / "atom"
        / "plugin"
        / "sglang"
        / "runtime"
        / "forward_context.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


def test_idle_runtime_trims_pipeline_intermediate_tensors():
    module = _load_forward_context_module()
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: True),
        positions=torch.empty(0, dtype=torch.long),
        seq_lens=torch.empty(0, dtype=torch.int32),
        seq_lens_cpu=torch.empty(0, dtype=torch.int32),
        batch_size=0,
        seq_lens_sum=0,
    )
    runtime = module.SGLangPluginRuntime(
        atom_config=SimpleNamespace(),
        forward_batch=forward_batch,
        positions=forward_batch.positions,
        input_ids=torch.empty(0, dtype=torch.long),
        set_forward_context=False,
    )

    with runtime:
        assert runtime.positions.shape == (1,)
        assert runtime.forward_batch.seq_lens.tolist() == [1]

        output = IntermediateTensors(
            {
                "hidden_states": torch.ones(1, 4),
                "residual": torch.ones(1, 4),
            }
        )
        trimmed = runtime.trim_output(output)

    assert isinstance(trimmed, IntermediateTensors)
    assert trimmed["hidden_states"].shape == (0, 4)
    assert trimmed["residual"].shape == (0, 4)


def test_idle_runtime_preserves_mrope_shape_without_sequence_lengths():
    module = _load_forward_context_module()
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: True),
        positions=torch.empty((3, 0), dtype=torch.long),
        seq_lens=None,
        batch_size=0,
        seq_lens_sum=0,
    )
    runtime = module.SGLangPluginRuntime(
        atom_config=SimpleNamespace(),
        forward_batch=forward_batch,
        positions=forward_batch.positions,
        input_ids=None,
        set_forward_context=False,
    )

    with runtime:
        assert runtime.positions.shape == (3, 1)
        assert runtime.forward_batch.seq_lens.tolist() == [1]
        assert runtime.forward_batch.seq_lens_cpu.tolist() == [1]
        assert runtime.forward_batch.seq_lens_cpu.device.type == "cpu"


def test_dp_token_counts_materialize_every_idle_rank():
    module = _load_forward_context_module()
    atom_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=4))
    forward_batch = SimpleNamespace(global_num_tokens_cpu=[0, 3, 0, 2])

    token_counts = module._resolve_num_tokens_across_dp(
        atom_config, forward_batch, num_tokens=1
    )

    assert token_counts.dtype == torch.int32
    assert token_counts.device.type == "cpu"
    assert token_counts.tolist() == [1, 3, 1, 2]
