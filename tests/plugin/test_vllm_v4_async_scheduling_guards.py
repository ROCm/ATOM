# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Guards that keep DeepSeek-V4 alive under `--async-scheduling` and DP > 1.

Async scheduling drops vLLM's sync on the sampled token ids. That is what makes
the CPU run ahead of the H2D copies, and what lets the scheduler's `-1` spec
placeholder reach a gather before the worker overwrites it. Each test below
pins one of the guards that removed a reproduced GPU fault.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from atom.plugin.vllm.deepseek_v4_bridge import (
    _is_runtime_dummy_decode,
    _V4DecodeMetaBuffers,
)
from atom.plugin.vllm.spec_decode_patch import _patch_dspark_markov_embed_bounds
from atom.utils.forward_context import running_tokens_from_bs

BRIDGE_PATH = Path(__file__).parents[2] / "atom/plugin/vllm/deepseek_v4_bridge.py"
EMBED_HEAD_PATH = Path(__file__).parents[2] / "atom/model_ops/embed_head.py"

# execute_dummy_batch() sends uniform_decode_query_len == 1 + num_spec_tokens.
K = 5


def _batch(max_query_len):
    return SimpleNamespace(max_query_len=max_query_len)


# --------------------------------------------------------------------------
# 1. The runtime dummy decode batch an idle DP rank steps through.
# --------------------------------------------------------------------------


def test_idle_dp_rank_dummy_decode_is_repaired_like_a_capture_batch():
    # req_ids == [] plus a decode-width query: exactly what execute_dummy_batch
    # produces. Without the repair the rank faults on its first idle step.
    assert _is_runtime_dummy_decode(_batch(1 + K), [], K)
    assert _is_runtime_dummy_decode(_batch(1), [], 0)


def test_real_decode_batch_is_never_repaired():
    # A real step has requests, so its context is coherent and rewriting
    # seq_lens/positions would corrupt it.
    assert not _is_runtime_dummy_decode(_batch(1 + K), ["req-0"], K)


def test_prefill_shaped_dummy_is_left_alone():
    # Profiling dummies are prefill-shaped and already carry a coherent context.
    assert not _is_runtime_dummy_decode(_batch(4096), [], K)
    assert not _is_runtime_dummy_decode(_batch(2 + K), [], K)


def test_missing_passthrough_patch_does_not_trigger_the_repair():
    # None means "patch not installed" (standalone/tests), not "no requests".
    assert not _is_runtime_dummy_decode(_batch(1 + K), None, K)


def test_zero_width_batch_is_not_a_decode():
    assert not _is_runtime_dummy_decode(_batch(0), [], K)


# --------------------------------------------------------------------------
# 2. The depth-1 gate on pinned staging-buffer reuse.
# --------------------------------------------------------------------------


class _RecordingEvent:
    def __init__(self):
        self.calls = []

    def synchronize(self):
        self.calls.append("synchronize")

    def record(self):
        self.calls.append("record")


def _bufs_with(event):
    bufs = object.__new__(_V4DecodeMetaBuffers)
    bufs._h2d_done = event
    return bufs


def test_gate_and_mark_pair_up_outside_capture():
    event = _RecordingEvent()
    bufs = _bufs_with(event)

    bufs.gate_staging_reuse()
    bufs.mark_h2d_enqueued()

    assert event.calls == ["synchronize", "record"]


def test_gate_and_mark_are_skipped_inside_a_capture_region(monkeypatch):
    # Event record/synchronize are illegal inside a capture region, and capture
    # has no CPU run-ahead to guard.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    event = _RecordingEvent()
    bufs = _bufs_with(event)

    bufs.gate_staging_reuse()
    bufs.mark_h2d_enqueued()

    assert event.calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA event")
def test_gate_blocks_until_the_previous_h2d_has_executed():
    bufs = _bufs_with(torch.cuda.Event())
    pinned = torch.zeros(1 << 22, dtype=torch.int32, pin_memory=True)
    gpu = torch.empty_like(pinned, device="cuda")

    # A never-recorded event passes immediately, so the first build is free.
    bufs.gate_staging_reuse()

    gpu.copy_(pinned, non_blocking=True)
    bufs.mark_h2d_enqueued()
    bufs.gate_staging_reuse()

    # Returning means the DMA out of `pinned` is done, so the next build may
    # overwrite it. This is the invariant; the assert just proves we got here
    # with the copy retired rather than merely enqueued.
    assert bufs._h2d_done.query()


def _persistent_decode_block():
    tree = ast.parse(BRIDGE_PATH.read_text())
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_atom_v4_attention_metadata"
    )
    return next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "decode_persistent"
    )


def test_every_staging_h2d_sits_between_the_gate_and_the_mark():
    # One event covers a whole build, so an H2D added outside the pair would be
    # silently unguarded -- the exact shape of the original bug.
    gate, mark, staging = [], [], []
    for node in ast.walk(_persistent_decode_block()):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        if attr == "gate_staging_reuse":
            gate.append(node.lineno)
        elif attr == "mark_h2d_enqueued":
            mark.append(node.lineno)
        elif attr in ("stage", "copy_to_gpu"):
            staging.append(node.lineno)

    assert len(gate) == 1, "the gate must be entered exactly once per build"
    assert len(mark) == 1, "one event per build, recorded after the last copy"
    assert staging, "no staging copies found -- did the persistent path move?"
    assert gate[0] < min(staging)
    assert mark[0] > max(staging)


# --------------------------------------------------------------------------
# 3. Embedding gathers that must survive the `-1` spec placeholder.
# --------------------------------------------------------------------------


def test_tp1_vocab_embedding_does_not_use_a_raw_gather():
    # The tp_size > 1 branch masked; tp_size == 1 fell through to F.embedding,
    # which reads the row before the table for the -1 placeholder. Both branches
    # must now go through a masked kernel, which returns zeros out of range and
    # is bit-identical for every valid token.
    tree = ast.parse(EMBED_HEAD_PATH.read_text())
    forward = next(
        node
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "VocabParallelEmbedding"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }

    assert "replicated_embedding" in called
    assert "masked_embedding" in called
    assert "embedding" not in called, "raw F.embedding is back in the tp1 branch"


class _FakeDSpark:
    """Stands in for DSparkDeepseekV4ForCausalLM's bare nn.Embedding lookup."""

    num_rows = 8

    def __init__(self):
        self.model = SimpleNamespace(
            markov_head=SimpleNamespace(
                markov_w1=SimpleNamespace(num_embeddings=self.num_rows)
            )
        )
        self.seen = None

    def markov_embed(self, token_ids):
        self.seen = token_ids
        # A real F.embedding would fault here on a negative id.
        assert int(token_ids.min()) >= 0
        assert int(token_ids.max()) < self.num_rows
        return token_ids


@pytest.fixture
def patched_dspark(monkeypatch):
    module = ModuleType("vllm.models.deepseek_v4.amd.dspark")
    module.DSparkDeepseekV4ForCausalLM = _FakeDSpark
    for name in (
        "vllm.models",
        "vllm.models.deepseek_v4",
        "vllm.models.deepseek_v4.amd",
    ):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name, ModuleType(name)))
    monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v4.amd.dspark", module)
    original = _FakeDSpark.markov_embed
    _patch_dspark_markov_embed_bounds()
    yield _FakeDSpark
    _FakeDSpark.markov_embed = original


def test_markov_embed_clamps_the_spec_placeholder(patched_dspark):
    model = patched_dspark()
    ids = torch.tensor([-1, 0, 3, -1])

    out = model.markov_embed(ids)

    assert out.tolist() == [0, 0, 3, 0]
    # Out-of-place: `prev` is the caller's live loop variable in
    # _sample_sequential, reassigned on every one of the K steps.
    assert ids.tolist() == [-1, 0, 3, -1]


def test_markov_embed_clamps_ids_past_the_table(patched_dspark):
    model = patched_dspark()

    out = model.markov_embed(torch.tensor([7, 8, 99]))

    assert out.tolist() == [7, 7, 7]


def test_markov_embed_leaves_valid_ids_untouched(patched_dspark):
    model = patched_dspark()
    ids = torch.arange(patched_dspark.num_rows)

    assert model.markov_embed(ids).tolist() == ids.tolist()


def test_markov_embed_patch_is_idempotent(patched_dspark):
    _patch_dspark_markov_embed_bounds()
    _patch_dspark_markov_embed_bounds()

    assert patched_dspark().markov_embed(torch.tensor([-1])).tolist() == [0]


# --------------------------------------------------------------------------
# 4. The profile stub has no attention geometry to answer with.
# --------------------------------------------------------------------------


def test_running_tokens_falls_back_to_the_request_count_without_metadata():
    # force_dummy passes None because the profile stub cannot answer
    # max_seqlen_q; is_prefill is pinned False there, which used to route the
    # stub down the decode branch and kill the worker during profile_run.
    assert running_tokens_from_bs(8, is_prefill=False, attn_metadata=None) == 8
    assert running_tokens_from_bs(
        8, is_prefill=False, attn_metadata=SimpleNamespace(max_seqlen_q=1 + K)
    ) == 8 * (1 + K)


def test_force_dummy_never_hands_the_profile_stub_to_running_tokens():
    source = BRIDGE_PATH.read_text()
    assert "attn_metadata=None if force_dummy else attn_metadata," in source
