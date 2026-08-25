# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The device-to-device half of recurrent-state checkpointing.

`GDNAttentionMetadataBuilder.save_state_checkpoints` copies a working slot out
to its reserved checkpoint slot after the forward, and `restore_state` copies a
checkpoint back into a working slot before it. Between them they are the entire
write/read path — there is no kernel involvement, because the scheduler aligns
each prefill chunk so the forward's one leftover state IS the checkpoint (see
`StateCachePool.trim_chunk_to_boundary`).

Two properties matter and are pinned here:
  * the slot pairing. `state_restore_slots` / `state_ckpt_write_slots` are
    index-aligned with `state_slots` — mispairing would hand a request another
    request's recurrent state, which no shape check would catch.
  * the conv-state width. A checkpoint holds the COMMITTED rows only; the
    speculative rollback rows above them are working-slot scratch and must be
    left untouched in both directions.

The source-level checks run anywhere; the copy semantics need a GPU.

STALE. `save_state_checkpoints` / `restore_state` / `state_restore_slots` no
longer exist in `atom/` — this whole module has been failing at collection
since that path was replaced, independently of any change here. Left in place
rather than deleted because the two properties above still hold of whatever
replaces it, and the `num_spec` cases are the only written record of the
conv-width rule. The `group * (1 + num_spec)` arithmetic it used to assert is
gone for good: slots are handed out one at a time and a request's set is not
contiguous.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BACKENDS = REPO / "atom" / "model_ops" / "attentions" / "backends.py"
GDN_ATTN = REPO / "atom" / "model_ops" / "attentions" / "gdn_attn.py"
MODEL_RUNNER = REPO / "atom" / "model_engine" / "model_runner.py"


def _func(tree: ast.Module, fn_name: str, cls_name: str | None = None):
    scope: ast.AST = tree
    if cls_name is not None:
        scope = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == cls_name
        )
    return next(
        n
        for n in ast.walk(scope)
        if isinstance(n, ast.FunctionDef) and n.name == fn_name
    )


# ── hook contract ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["allocate_state_checkpoints", "restore_state", "save_state_checkpoints"]
)
def test_backends_defaults_every_hook_to_a_no_op(name):
    """Stateless attentions opt OUT by default so the runner can call the hooks
    unconditionally — the same shape as the paged-SWA and per-req-cache hooks."""
    fn = _func(ast.parse(BACKENDS.read_text()), name)
    assert len(fn.body) == 2  # docstring + return
    ret = fn.body[-1]
    assert isinstance(ret, ast.Return)
    assert ret.value is None or ast.literal_eval(ret.value) in ({}, None)


def test_checkpoints_use_the_committed_conv_shape():
    """`_state_shape` folds num_spec into the conv-state length, so allocating
    the checkpoint pool at the working-slot width would waste num_spec rows per
    layer AND store rollback scratch as if it were committed state.

    Matched against the code only (docstring stripped): a docstring that
    *describes* the contract would otherwise satisfy this forever.
    """
    fn = _func(
        ast.parse(GDN_ATTN.read_text()),
        "allocate_state_checkpoints",
        cls_name="GDNAttentionMetadataBuilder",
    )
    assert "num_spec=0" in "\n".join(ast.unparse(n) for n in fn.body[1:])


def test_runner_brackets_the_forward_with_the_hooks():
    """Restore must land after prepare_model claims this step's working slots
    and before the forward reads them; save must see what the forward left.

    Instrumenting at the call site is mandatory here: the models these hooks
    serve are @support_torch_compile and must not be edited.
    """
    fn = _func(ast.parse(MODEL_RUNNER.read_text()), "forward", cls_name="ModelRunner")
    wanted = ("prepare_model", "restore_state", "run_model", "save_state_checkpoints")
    # ast.walk yields no particular order, so sort by source position.
    seq = [
        name
        for _, name in sorted(
            (n.lineno, n.func.attr)
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in wanted
        )
    ]
    assert seq == [
        "prepare_model",
        "restore_state",
        "run_model",
        "save_state_checkpoints",
    ], f"hooks are misordered around the forward: {seq}"


# ── copy semantics (GPU) ───────────────────────────────────────────────────

torch = pytest.importorskip("torch")

gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/ROCm"
)


@pytest.fixture(scope="module")
def builder_cls():
    """The real GDN builder class, past conftest's `atom.*` stubs.

    Same dance as `test_gdn_has_initial_state.real_gdn`: conftest stubs
    `atom.config` for the pure-Python engine tests, but this module needs the
    real one, so drop the stubs for the duration and restore them after.
    """
    saved = {
        k: v for k, v in sys.modules.items() if k == "atom" or k.startswith("atom.")
    }
    for name in saved:
        del sys.modules[name]
    try:
        from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder

        yield GDNAttentionMetadataBuilder
    finally:
        for name in [k for k in sys.modules if k == "atom" or k.startswith("atom.")]:
            del sys.modules[name]
        sys.modules.update(saved)


CONV_ROWS, CONV_DIM = 3, 8  # committed conv rows, conv width
SSM_SHAPE = (2, 4, 4)
LAYERS = 2


def make_builder(builder_cls, num_spec=0, num_slots=4, num_ckpt=4):
    """A builder wired to real tensors, bypassing __init__'s HF/GPU setup.

    Only `device`, `num_spec` and `model_runner`'s four tensors are read by the
    three methods under test, so constructing the object directly keeps this a
    test of the copy logic rather than of ModelRunner startup.
    """
    b = object.__new__(builder_cls)
    b.device = torch.device("cuda")
    b.num_spec = num_spec
    work_conv = (CONV_ROWS + num_spec, CONV_DIM)
    b.model_runner = types.SimpleNamespace(
        mamba_k_cache=torch.zeros((LAYERS, num_slots) + work_conv, device="cuda"),
        mamba_v_cache=torch.zeros((LAYERS, num_slots) + SSM_SHAPE, device="cuda"),
        state_ckpt_k=torch.zeros(
            (LAYERS, num_ckpt, CONV_ROWS, CONV_DIM), device="cuda"
        ),
        state_ckpt_v=torch.zeros((LAYERS, num_ckpt) + SSM_SHAPE, device="cuda"),
    )
    return b


def batch(groups, restore=None, writes=None):
    n = len(groups)
    return types.SimpleNamespace(
        state_slots=[[g] for g in groups],
        state_restore_slots=list(restore if restore is not None else [-1] * n),
        state_ckpt_write_slots=list(writes if writes is not None else [-1] * n),
    )


@gpu_only
class TestSaveAndRestore:
    def test_save_then_restore_round_trips_the_state(self, builder_cls):
        b = make_builder(builder_cls)
        r = b.model_runner
        r.mamba_k_cache[:, 1] = 7.0
        r.mamba_v_cache[:, 1] = 9.0

        b.save_state_checkpoints(batch([1], writes=[2]))
        r.mamba_k_cache.zero_()
        r.mamba_v_cache.zero_()
        b.restore_state(batch([3], restore=[2]))

        assert torch.all(r.mamba_k_cache[:, 3] == 7.0)
        assert torch.all(r.mamba_v_cache[:, 3] == 9.0)

    def test_each_request_gets_its_own_checkpoint(self, builder_cls):
        """The pairing is what a mismatch would corrupt silently, so drive
        several requests at once with distinguishable contents."""
        b = make_builder(builder_cls, num_slots=4, num_ckpt=4)
        r = b.model_runner
        for slot in range(3):
            r.mamba_v_cache[:, slot] = float(slot + 1)

        b.save_state_checkpoints(batch([0, 1, 2], writes=[2, 0, 1]))

        assert torch.all(r.state_ckpt_v[:, 2] == 1.0)
        assert torch.all(r.state_ckpt_v[:, 0] == 2.0)
        assert torch.all(r.state_ckpt_v[:, 1] == 3.0)

    def test_negative_slots_are_skipped(self, builder_cls):
        """-1 means "no checkpoint this step"; those working slots must be
        left exactly as the forward left them."""
        b = make_builder(builder_cls)
        r = b.model_runner
        r.state_ckpt_v[:, 0] = 5.0
        r.mamba_v_cache[:, 1] = 1.0
        r.mamba_v_cache[:, 2] = 2.0

        b.save_state_checkpoints(batch([1, 2], writes=[-1, 0]))
        assert torch.all(r.state_ckpt_v[:, 0] == 2.0)  # only slot 2 was saved

        b.restore_state(batch([1, 2], restore=[-1, -1]))
        assert torch.all(r.mamba_v_cache[:, 1] == 1.0)  # untouched
        assert torch.all(r.mamba_v_cache[:, 2] == 2.0)

    def test_working_slot_is_group_times_slots_per_req(self, builder_cls):
        """With speculative decoding a group owns `1 + num_spec` consecutive
        working slots and the committed one is the first — the same base
        `prepare_state_indices` computes."""
        b = make_builder(builder_cls, num_spec=2, num_slots=9, num_ckpt=4)
        r = b.model_runner
        r.mamba_v_cache[:, 3] = 4.0  # group 1's committed slot: 1 * (1 + 2)
        b.save_state_checkpoints(batch([1], writes=[0]))
        assert torch.all(r.state_ckpt_v[:, 0] == 4.0)

    def test_speculative_rollback_rows_are_not_checkpointed(self, builder_cls):
        """The conv state's trailing `num_spec` rows are scratch: saving must
        read only the committed prefix, and restoring must write only it."""
        b = make_builder(builder_cls, num_spec=2, num_slots=3, num_ckpt=2)
        r = b.model_runner
        assert r.mamba_k_cache.shape[2] == CONV_ROWS + 2
        r.mamba_k_cache[:, 0, :CONV_ROWS] = 1.0
        r.mamba_k_cache[:, 0, CONV_ROWS:] = 99.0  # rollback scratch

        b.save_state_checkpoints(batch([0], writes=[0]))
        assert torch.all(r.state_ckpt_k[:, 0] == 1.0)  # no 99s made it in

        r.mamba_k_cache.zero_()
        r.mamba_k_cache[:, 0, CONV_ROWS:] = 42.0
        b.restore_state(batch([0], restore=[0]))
        assert torch.all(r.mamba_k_cache[:, 0, :CONV_ROWS] == 1.0)
        assert torch.all(r.mamba_k_cache[:, 0, CONV_ROWS:] == 42.0)

    def test_absent_pool_makes_both_hooks_inert(self, builder_cls):
        """Every model without a checkpoint pool runs these hooks each step."""
        b = make_builder(builder_cls)
        b.model_runner.state_ckpt_k = None
        b.model_runner.state_ckpt_v = None
        b.restore_state(batch([0], restore=[0]))
        b.save_state_checkpoints(batch([0], writes=[0]))

    def test_empty_batch_touches_nothing(self, builder_cls):
        b = make_builder(builder_cls)
        b.restore_state(batch([]))
        b.save_state_checkpoints(batch([]))
        assert torch.all(b.model_runner.state_ckpt_v == 0.0)

    def test_checkpoint_dtypes_follow_the_working_slots(self, builder_cls):
        """`fla.ops.kda` asserts `initial_state.dtype == torch.float32`, so a
        checkpoint stored at a narrower dtype would fail on restore. Matching
        the working slots keeps every copy a straight D2D move."""
        fn = _func(
            ast.parse(GDN_ATTN.read_text()),
            "allocate_state_checkpoints",
            cls_name="GDNAttentionMetadataBuilder",
        )
        src = ast.unparse(fn)
        assert "_state_dtypes()" in src
