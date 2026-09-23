# SPDX-License-Identifier: MIT
"""Forward-context metadata must work without the GPU communication runtime."""

import ast
import logging
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.distributed import ulysses_sp as sp
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence
from atom.utils.forward_context import (
    get_forward_context,
    reset_forward_context,
    set_forward_context,
)


@pytest.fixture
def runner_type():
    # Execute the actual methods without importing the GPU attention stack.
    path = Path(__file__).resolve().parents[2] / "atom/model_engine/model_runner.py"
    module = ast.parse(path.read_text())
    runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    runner.body = [
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"sp_local_tokens", "sp_graph_input_ids", "prepare_inputs", "warmup_model"}
    ]
    module.body = [ast.parse("from __future__ import annotations").body[0], runner]
    namespace = {
        "torch": torch,
        "np": np,
        "time": time,
        "logger": logging.getLogger(__name__),
        "get_dp_group": lambda: SimpleNamespace(world_size=1),
        "Sequence": Sequence,
        "ScheduledBatch": ScheduledBatch,
        "Context": SimpleNamespace,
        "set_forward_context": set_forward_context,
        **{
            name: getattr(sp, name)
            for name in (
                "sp_is_enabled",
                "sp_pad_len",
                "get_sp_world_size",
                "sp_local_slice",
            )
        },
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace["ModelRunner"]


@pytest.mark.parametrize("world", [1, 4])
def test_sp_warmup_profiles_the_full_global_token_budget(
    runner_type, monkeypatch, world
):
    monkeypatch.setattr(sp, "_SP_WORLD_SIZE", world)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    runner = runner_type()
    runner.config = SimpleNamespace(
        max_num_batched_tokens=17,
        max_model_len=17,
        max_num_seqs=1,
        enable_dp_attention=False,
        prefill_context_parallel_size=world,
    )
    runner.block_size = 16
    runner.label = "test"
    batches = []
    runner.forward = batches.append
    runner.tokenID_processor = SimpleNamespace(clean=lambda: None)
    runner.warmup_model()
    assert len(batches) == 1
    assert batches[0].total_tokens_num == 17


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("capacity", [1, 3, 17])
def test_graph_input_staging_pads_each_rank_at_buffer_capacity(
    runner_type, monkeypatch, world, capacity
):
    monkeypatch.setattr(sp, "_SP_WORLD_SIZE", world)
    runner = runner_type()
    source = torch.arange(1, capacity + 1)
    runner.forward_vars = {
        "input_ids": SimpleNamespace(gpu=source),
        "sp_input_ids": torch.full((sp.sp_pad_len(capacity) // world,), -1),
    }
    pointer = runner.forward_vars["sp_input_ids"].data_ptr()
    for rank in range(world):
        monkeypatch.setattr(sp, "get_sp_rank", lambda rank=rank: rank)
        for count in (capacity, max(1, capacity - 2), capacity):
            result = runner.sp_graph_input_ids(count)
            assert result.data_ptr() == pointer
            assert torch.equal(result, sp.sp_split_tokens(source[:count]))
    assert torch.equal(source, torch.arange(1, capacity + 1))


@pytest.mark.parametrize("world", [1, 4])
def test_runner_publishes_padded_token_counts_for_sp_moe(
    runner_type, monkeypatch, world
):
    monkeypatch.setattr(sp, "_SP_WORLD_SIZE", world)
    runner = runner_type()
    runner.config = SimpleNamespace(
        prefill_context_parallel_size=world,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    runner.attn_metadata_builder = SimpleNamespace(
        build=lambda **kwargs: (None, None),
        prepare_model_inputs=lambda *args: None,
    )
    runner._maybe_create_tbo_slices = lambda *args: None
    batch = SimpleNamespace(
        total_tokens_num_prefill=0,
        total_tokens_num=17,
        total_seqs_num=17,
        num_scheduled_tokens=[1] * 17,
        is_dummy_run=False,
    )
    mode = SimpleNamespace(
        sync=None,
        tbo_collective_active=False,
        running_tokens_are_unified=True,
        running_bs=32,
        running_tokens=32,
        max_seqlen_q=1,
        scheduled_bs=17,
    )
    try:
        runner.prepare_inputs(batch, torch.arange(32), mode)
        expected = (8,) * world if world > 1 else None
        assert get_forward_context().context.running_tokens_across_dp == expected
    finally:
        reset_forward_context()


def test_forward_context_without_aiter_or_triton():
    # A fresh interpreter also checks this on GPU developer machines, where a
    # previously imported AITER module would otherwise hide the CI regression.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import sys
                from types import SimpleNamespace

                import torch

                sys.modules["aiter"] = None
                sys.modules["triton"] = None
                torch.cuda.is_available = lambda: False

                from atom.distributed import ulysses_sp as sp
                from atom.utils.forward_context import (
                    get_forward_context,
                    set_forward_context,
                )

                config = SimpleNamespace(
                    parallel_config=SimpleNamespace(data_parallel_size=1),
                    compilation_config=SimpleNamespace(static_forward_context={}),
                )
                context = SimpleNamespace()
                sp.set_sp_world_size(4)
                set_forward_context(None, config, context, num_tokens=17)
                assert get_forward_context().context is context
                assert context.running_tokens_across_dp == (5, 5, 5, 5)

                set_forward_context(None, config, context, num_tokens=8)
                assert context.running_tokens_across_dp == (2, 2, 2, 2)
                set_forward_context(None, config, context)
                assert context.running_tokens_across_dp is None

                sp.set_sp_world_size(1)
                counts = torch.tensor([3, 7], dtype=torch.int32)
                set_forward_context(None, config, context, num_tokens_across_dp=counts)
                assert context.running_tokens_across_dp == (3, 7)
                set_forward_context(None, config, context, num_tokens=17)
                assert context.running_tokens_across_dp is None
                assert sp.get_sp_rank() == 0
                assert sys.modules["aiter"] is None
                assert sys.modules["triton"] is None
                """),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
