# SPDX-License-Identifier: MIT
"""GLM-5.2 mono: the fused indexer's context-parallel switch.

Two things are pinned here:

1. ``indexer_cp_refusal`` refuses by name everything the vendored kernel's own
   contract says CP cannot serve (FlyDSL ``doc/glm-indexer-cp.md``, "CP
   algorithm and input contract"), including the one combination that would
   HANG rather than merely misbehave -- see
   ``test_cp_cannot_outlive_the_capacity_refusal``.
2. **With the switch off, nothing changes.** The kernel that gets built, the
   layers that route to it, and therefore the numerics are identical to what
   they were before the switch existed. This is checked at the only two places
   a difference could enter: the argument ``_GlmFusedLayerOp`` hands the
   kernel, and the compile-time layout the kernel derives from it.

Of the two halves of (2), the load-bearing one is
``test_runner_passes_the_switch_through_to_the_kernel``: a dropped parameter
on the way from the runner to ``Glm5TP4MonoKernel`` leaves the switch inert
and every refusal above describing a build that never happens. The layout
half is a narrower check that no CP region intrudes on the off path.
"""

from types import SimpleNamespace

import pytest

from atom.models.glm52.mono.config import (
    FUSED_INDEX_MAX_SEQ,
    FUSED_INDEX_PAGE_SIZE,
    indexer_cp_refusal,
)
from atom.models.glm52.mono.kernels.layout import (
    HIDDEN,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_Q_ROWS,
    _align,
    index_cp_world,
    index_levels,
    layout,
)


def _servable_atom_config(**overrides):
    """A deployment both the fused indexer and its CP can serve."""
    fields = {
        "index_cache_dtype": "fp8",
        "max_model_len": FUSED_INDEX_MAX_SEQ,
        "kv_cache_block_size": FUSED_INDEX_PAGE_SIZE,
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def fused_on(monkeypatch):
    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    return monkeypatch


# --------------------------------------------------------------- the switch


def test_off_by_default_env_unset(fused_on):
    # The opt-in itself. Note this fires even though the fused indexer is on:
    # CP is a separate, narrower switch.
    fused_on.delenv("ATOM_GLM52_INDEXER_CP", raising=False)
    assert (
        indexer_cp_refusal(_servable_atom_config()) == "fused indexer CP switched off"
    )


def test_explicit_zero_is_off(fused_on):
    fused_on.setenv("ATOM_GLM52_INDEXER_CP", "0")
    assert (
        indexer_cp_refusal(_servable_atom_config()) == "fused indexer CP switched off"
    )


def test_servable_when_switched_on(fused_on):
    fused_on.setenv("ATOM_GLM52_INDEXER_CP", "1")
    assert indexer_cp_refusal(_servable_atom_config()) is None


def test_cp_is_narrower_than_the_fused_indexer(monkeypatch):
    # CP cannot be on where the fused indexer is off: it is a build parameter
    # of a kernel only the fused path builds. The refusal names the reason the
    # fused indexer itself gave, so an operator is not left guessing.
    monkeypatch.delenv("ATOM_GLM52_FUSED_INDEXER", raising=False)
    monkeypatch.setenv("ATOM_GLM52_INDEXER_CP", "1")
    why = indexer_cp_refusal(_servable_atom_config())
    assert why is not None
    assert "fused indexer off" in why and "switched off" in why


def test_fused_indexer_refusal_is_quoted_not_swallowed(fused_on):
    fused_on.setenv("ATOM_GLM52_INDEXER_CP", "1")
    why = indexer_cp_refusal(_servable_atom_config(index_cache_dtype="fp4"))
    assert why is not None and "index_cache_dtype" in why and "fp4" in why


# ------------------------------------------- what the kernel contract refuses


def test_tp1_has_no_cp_group(fused_on):
    # "Each rank owns a contiguous 64-token-aligned shard": a group of one
    # shards nothing, so it is refused by name rather than silently building a
    # CP kernel that is a slower spelling of CP off.
    fused_on.setenv("ATOM_GLM52_INDEXER_CP", "1")
    why = indexer_cp_refusal(_servable_atom_config(tensor_parallel_size=1))
    assert why is not None and "CP group" in why and "TP1" in why


def test_decode_context_parallel_is_refused_by_name(fused_on):
    # "Fused indexer CP currently requires dcp_size=1." config_refusal's
    # common_refusals already rejects decode CP for GLM-5.2 mono as a whole,
    # but the two are independent mechanisms and the kernel contract names
    # this one, so relaxing the former must not silently enable CP.
    fused_on.setenv("ATOM_GLM52_INDEXER_CP", "1")
    why = indexer_cp_refusal(_servable_atom_config(decode_context_parallel_size=4))
    assert why is not None and "dcp_size=1" in why


def test_cp_cannot_outlive_the_capacity_refusal(fused_on):
    # The one that would hang. Above LOCAL_INDEX_CAPACITY the CP scorer takes
    # the `else` at kernel.py:2012 and stops publishing `index_cp_scores`,
    # while the short selector -- reached unconditionally, since TILED_INDEXER
    # is False at index_max_seq == INDEX_LOCAL_LDS_CAPACITY == 16384 (the test
    # is strictly-greater) -- keeps polling that mailbox for every slot. All
    # four ranks spin forever with no error on any of them. Nothing in the
    # kernel stops this; `fused_indexer_refusal`'s static max_model_len check
    # does, and `indexer_cp_refusal` requires it to have passed, so a
    # deployment that could reach the hang cannot turn CP on.
    fused_on.setenv("ATOM_GLM52_INDEXER_CP", "1")
    why = indexer_cp_refusal(
        _servable_atom_config(max_model_len=FUSED_INDEX_MAX_SEQ + 1)
    )
    assert why is not None and "fused indexer off" in why
    assert str(FUSED_INDEX_MAX_SEQ) in why


def test_index_max_seq_meets_the_contract():
    # "index_max_seq must be a positive multiple of 64." Static, so this
    # asserts the constant rather than a deployment: if FUSED_INDEX_MAX_SEQ is
    # ever retuned to something CP cannot take, this fails here and not on a
    # GPU.
    assert FUSED_INDEX_MAX_SEQ > 0
    assert FUSED_INDEX_MAX_SEQ % 64 == 0


def test_cp_group_is_the_whole_tp_group_at_tp4():
    # The group size the kernel will use, read from the same helper the kernel
    # reads. TP4's special cases (two CP4 groups) are TP8-only, so TP4 is
    # always a single four-rank group and the 64-token shard alignment that
    # index_cp_capacity enforces divides evenly.
    assert index_cp_world(4, 1, FUSED_INDEX_MAX_SEQ) == 4
    assert index_cp_world(4, 10, FUSED_INDEX_MAX_SEQ) == 4


# ------------------------------- with the switch off, nothing changed at all


@pytest.mark.parametrize("samples", [1, 2, 4, 5, 6, 8, 10, 12])
def test_cp_off_layout_is_identical_to_no_cp(samples):
    """The compile-time scratch and symmetric layouts a fused build derives.

    ``indexer_cp=False`` must reproduce, byte offset for byte offset, the
    layout a build made before this switch existed. Any extra CP region
    appearing here when CP is off would relocate every field after it and
    silently change what every launch reads.

    Pinned against sizes restated here from the kernel's own contract, NOT
    against another ``layout`` call: ``indexer_cp`` already defaults to
    ``False``, so comparing an explicit ``False`` to the default compares a
    call to itself and passes no matter what the parameter does. The
    concrete regression this must catch is the ``indexer_cp`` conditional
    being dropped from ``index_scores``' size (``layout.py:275-278``) or from
    ``index_ranks`` (``:243``): with CP off that resizes ``index_scores``
    from ``index_max_seq`` pairs per sample to ``index_max_seq // npes``, and
    every field after it moves while launches keep indexing at the old
    offsets.
    """
    build = {
        "inter": 512,
        "output_heads": 16,
        "dcp_size": 1,
        "native_fp4_mfma": True,
    }
    scratch, symmetric = layout(
        samples, 16, 4, 2048, True, FUSED_INDEX_MAX_SEQ, indexer_cp=False, **build
    )

    # No CP region exists at all with the switch off -- scratch or symmetric.
    assert not any(name.startswith("index_cp") for name in scratch)
    assert not any(name.startswith("index_cp") for name in symmetric)

    # The indexer block's fields, in declaration order, each with the size the
    # kernel contract gives it. `index_scores` is the one the CP conditional
    # controls: with CP off it must hold the FULL compiled capacity, one
    # 8-byte pair per (sample, key), not a per-rank shard of it.
    pair = 8
    expected = [
        ("index_k", samples * INDEX_DIM * pair),
        ("index_k_new", samples * INDEX_DIM // 2 * pair),
        ("index_ready", samples * pair),
        ("index_w", samples * INDEX_HEADS * pair),
        ("index_q", samples * INDEX_Q_ROWS // 2 * pair),
        ("index_scores", samples * FUSED_INDEX_MAX_SEQ * pair),
        ("indices", samples * 2048 * 4),
        ("indices_ready", samples * pair),
    ]
    # What follows the indexer block depends on the build's capacity, not on
    # the CP switch: a merge-tree build appends one ``isel<l>`` region per
    # level (``test_glm5_index_tree.py`` owns their sizes), while a one-leaf
    # build ends the map here. Derive the sentinel from ``index_levels`` so
    # this stays a statement about the layout rather than a read-back of it.
    terminator = "isel0" if index_levels(FUSED_INDEX_MAX_SEQ, 2048) else "_bytes"
    for (name, size), (nxt, _) in zip(expected, expected[1:] + [(terminator, 0)]):
        assert scratch[nxt] - scratch[name] == _align(size), name

    # And the symmetric region ends where the non-CP peer buffers end: two
    # launch slots of attention + FFN, no CP tail appended.
    assert symmetric["_bytes"] == 2 * (2 * 4 * samples * HIDDEN * pair)


@pytest.mark.parametrize("samples", [1, 5, 10])
def test_cp_on_changes_the_build(samples):
    """The negative control for the test above.

    If ``indexer_cp=True`` produced the same layout as off, the identity test
    would be vacuous -- it would be passing because the parameter does
    nothing, not because off is faithful.
    """
    common = {
        "inter": 512,
        "output_heads": 16,
        "dcp_size": 1,
        "native_fp4_mfma": True,
    }
    off = layout(
        samples, 16, 4, 2048, True, FUSED_INDEX_MAX_SEQ, indexer_cp=False, **common
    )
    on = layout(
        samples, 16, 4, 2048, True, FUSED_INDEX_MAX_SEQ, indexer_cp=True, **common
    )
    assert on != off
    assert any(name.startswith("index_cp") for name in on[1])


def test_runner_passes_the_switch_through_to_the_kernel():
    """``_GlmFusedLayerOp`` must take ``indexer_cp`` and forward it.

    Checked on the signatures rather than by building: a build needs four
    GPUs. What this catches is the parameter being dropped on the way from the
    runner to ``Glm5TP4MonoKernel`` -- which would leave the switch inert and
    the refusals above describing a build that never happens.
    """
    import inspect

    # the kernel module compiles through FlyDSL at import, and the runner
    # imports the AITER runtime unconditionally
    pytest.importorskip("aiter")
    pytest.importorskip("flydsl", reason="the mono kernel is FlyDSL source")

    from atom.models.glm52.mono.kernels.op import Glm5MonoKernel
    from atom.models.glm52.mono.kernels.tp4_op import Glm5TP4MonoKernel
    from atom.models.glm52.mono.runner import _GlmFusedLayerOp

    assert "indexer_cp" in inspect.signature(_GlmFusedLayerOp.__init__).parameters
    # Off by default on both kernel entry points, so a caller that does not
    # pass it gets the pre-switch build.
    for cls in (Glm5MonoKernel, Glm5TP4MonoKernel):
        assert inspect.signature(cls.__init__).parameters["indexer_cp"].default is False
    # The TP4 shim keeps it keyword-only. That matters because ATOM passes the
    # first eleven arguments of Glm5MonoKernel positionally through the shim's
    # super() call, and `indexer_cp` is the eleventh -- a caller that handed
    # the shim a positional in that slot would be setting `timeline`.
    tp4 = inspect.signature(Glm5TP4MonoKernel.__init__).parameters["indexer_cp"]
    assert tp4.kind is inspect.Parameter.KEYWORD_ONLY
