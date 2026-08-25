# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GDN/KDA prefill `has_initial_state` must reflect real forwarded-token counts.

`prepare_gdn_metadata` used to overwrite the per-seq context lengths with
`torch.zeros(...)` before deriving `has_initial_state = context_lens > 0`, so the
mask was unconditionally all-False. Downstream that means:

  * `causal_conv1d_fn` takes the `load_init_state == False` branch and treats
    every prefill chunk as if it had no prior tokens (conv window zeroed);
  * `attention_gdn.py` runs `initial_state[~has_initial_state] = 0`, wiping the
    recurrent ssm_state gathered from the working slot.

Both are silent wrong-output bugs the moment a prompt is split across chunks
(`enable_chunked_prefill` defaults True) or resumes from a prefix-cache hit.

These tests pin the contract at the source level (no GPU required) plus a
numeric check of the mask derivation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GDN_ATTN = REPO / "atom" / "model_ops" / "attentions" / "gdn_attn.py"
ENGINE_CORE = REPO / "atom" / "model_engine" / "engine_core.py"


def _func(tree: ast.Module, cls_name: str, fn_name: str) -> ast.FunctionDef:
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    return next(
        n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef) and n.name == fn_name
    )


def _str_constants(node: ast.AST) -> set[str]:
    """String literals in `node`, excluding docstrings."""
    docstrings = {
        ast.get_docstring(n, clean=False)
        for n in ast.walk(node)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
    }
    return {
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value not in docstrings
    }


# ── source contract ────────────────────────────────────────────────────────


def test_prepare_gdn_metadata_does_not_zero_context_lens():
    """No unconditional zero-fill may shadow the real per-seq lengths."""
    tree = ast.parse(GDN_ATTN.read_text())
    fn = _func(tree, "GDNAttentionMetadataBuilder", "prepare_gdn_metadata")
    zero_calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "zeros"
    ]
    assert not zero_calls, (
        "prepare_gdn_metadata must not synthesize zeroed context lengths — "
        "that made has_initial_state always False and dropped recurrent state "
        f"across prefill chunks (gdn_attn.py:{zero_calls[0].lineno})"
        if zero_calls
        else ""
    )


def test_has_initial_state_derives_from_num_cached_tokens():
    """The mask must key off tokens ALREADY forwarded, not total context.

    `batch.context_lens` is num_cached_tokens + this chunk's tokens, so it is
    > 0 for every prefill and would mark fresh prompts as having prior state.
    """
    tree = ast.parse(GDN_ATTN.read_text())
    fn = _func(tree, "GDNAttentionMetadataBuilder", "prepare_has_initial_state")
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "num_cached_tokens" in attrs
    assert "context_lens" not in attrs


# ── mask semantics (GPU) ───────────────────────────────────────────────────

torch = pytest.importorskip("torch")

gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/ROCm"
)


@pytest.fixture(scope="module")
def real_gdn():
    """The real `gdn_attn` module + a buffer factory, past conftest's stubs.

    conftest stubs `atom.config` so the pure-Python engine tests import without
    HuggingFace/GPU. The builder pulls in the real config through
    `atom.model_ops.linear`, so drop the stubbed `atom.*` entries for the
    duration of this module and restore them afterwards — otherwise every
    later test in the session sees the heavyweight real modules.
    """
    import sys
    import types

    saved = {
        k: v for k, v in sys.modules.items() if k == "atom" or k.startswith("atom.")
    }
    for name in saved:
        del sys.modules[name]
    try:
        from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
        from atom.utils import CpuGpuBuffer

        def stub(max_bs=8):
            # Minimal stand-in exposing only what prepare_has_initial_state
            # touches, so the test skips full ModelRunner / HF-config setup.
            return types.SimpleNamespace(
                num_computed_tokens=CpuGpuBuffer(
                    (max_bs,), dtype=torch.int32, device=torch.device("cuda")
                )
            )

        yield GDNAttentionMetadataBuilder.prepare_has_initial_state, stub
    finally:
        for name in [k for k in sys.modules if k == "atom" or k.startswith("atom.")]:
            del sys.modules[name]
        sys.modules.update(saved)


def _batch(num_cached):
    import types

    return types.SimpleNamespace(num_cached_tokens=num_cached)


@gpu_only
@pytest.mark.parametrize(
    "num_cached, expected",
    [
        ([0, 0, 0], [False, False, False]),  # three fresh prompts
        ([0, 16, 0], [False, True, False]),  # mixed fresh + resumed chunk
        ([64, 128], [True, True]),  # both mid-prefill / prefix-cache hit
    ],
)
def test_mask_semantics(real_gdn, num_cached, expected):
    """`num_cached_tokens > 0` is exactly 'slot already holds usable state'."""
    fn, stub = real_gdn
    got = fn(stub(), _batch(num_cached), len(num_cached))
    assert got.dtype is torch.bool
    assert got.device.type == "cuda"
    assert got.tolist() == expected


@gpu_only
def test_mask_does_not_leak_across_batches(real_gdn):
    """The persistent buffer must be sliced to this batch's prefill count.

    Otherwise a small batch following a large one would read stale entries
    from the previous step and claim state slots it never wrote.
    """
    fn, stub = real_gdn
    builder = stub()

    fn(builder, _batch([99] * 8), 8)
    got = fn(builder, _batch([0, 0]), 2)

    assert tuple(got.shape) == (2,)
    assert got.tolist() == [False, False]


# ── prefix caching gate ────────────────────────────────────────────────────


def test_prefix_caching_gate_is_conditional_on_the_checkpoint_pool():
    """The gate must turn prefix caching off only when the pool is INACTIVE.

    Checkpoints are what make a paged hit sound for a recurrent model: the hit
    is bounded to a boundary whose conv/ssm state was saved
    (`StateCachePool.bounded_hit`) and that state is copied into the working
    slot before the forward (`restore_state`). With an active pool the gate
    must therefore stand down, or the whole feature is dead code.
    """
    tree = ast.parse(ENGINE_CORE.read_text())
    fn = _func(tree, "EngineCore", "__init__")
    gate = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and "_has_recurrent_state" in ast.unparse(n.test)
        and "enable_prefix_caching" in ast.unparse(n.test)
    )
    test_src = ast.unparse(gate.test)
    assert "state_cache_block_size" in test_src or "state_ckpt_active" in test_src, (
        "the recurrent-model prefix-caching gate ignores the checkpoint pool — "
        "an active pool makes the hit safe and must lift the gate"
    )
    # And the body still disables it when the pool is absent.
    assert "config.enable_prefix_caching = False" in ast.unparse(gate.body)


def test_pool_is_active_only_when_both_size_and_capacity_are_set():
    """Either half alone is not a pool: M > 0 with zero blocks has nowhere to
    write, and blocks with M == 0 have no boundary to write at."""
    tree = ast.parse(ENGINE_CORE.read_text())
    fn = _func(tree, "EngineCore", "__init__")
    assign = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "state_ckpt_active"
    )
    src = ast.unparse(assign.value)
    assert "state_cache_block_size > 0" in src
    assert "num_state_cache_blocks > 0" in src
    assert isinstance(assign.value, ast.BoolOp) and isinstance(assign.value.op, ast.And)


def test_prefix_caching_disabled_for_all_recurrent_models():
    """Every linear-attention hybrid — not just Kimi-K3 — must be gated.

    A paged prefix-cache hit lets a sequence skip forwarding tokens. For a
    recurrent model those tokens' conv/ssm state was never computed into the
    request's working slot, so the hit is only sound unless a checkpoint pool
    is active. Qwen3-Next / Qwen3.5 were previously ungated (the check named
    only Kimi-K3), which combined with the always-False mask above to hide the
    bug.
    """
    tree = ast.parse(ENGINE_CORE.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_has_recurrent_state"
    )
    listed = _str_constants(fn)
    for model_type in ("qwen3_next", "qwen3_5_text", "qwen3_5_moe_text", "kimi_linear"):
        assert model_type in listed, f"{model_type} not gated in _has_recurrent_state"


def test_recurrent_state_gate_matches_per_req_cache_types():
    """Keep the gate in sync with the per-req-cache model_type registry.

    `InputOutputProcessor._per_req_cache_model_types()` is the single source of
    truth for "this model keeps state outside the paged pool". deepseek_v4 is
    the one member that is exempt: its compressor state is handled by the SWA
    pool, which is already prefix-cache safe.
    """
    io_tree = ast.parse((REPO / "atom" / "model_engine" / "llm_engine.py").read_text())
    io_fn = _func(io_tree, "InputOutputProcessor", "_per_req_cache_model_types")
    per_req = _str_constants(io_fn)

    core_tree = ast.parse(ENGINE_CORE.read_text())
    core_fn = next(
        n
        for n in core_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_has_recurrent_state"
    )
    gated = _str_constants(core_fn)

    missing = per_req - gated - {"deepseek_v4"}
    assert not missing, (
        f"model_type(s) {sorted(missing)} keep per-request state but are not "
        "gated in engine_core._has_recurrent_state — a prefix-cache hit would "
        "skip forwarding tokens whose recurrent state was never computed"
    )
