# SPDX-License-Identifier: MIT
"""GLM-5.2 mono: the fused TP4 indexer's own refusal, independent of
``config_refusal`` -- a deployment GLM-5.2 mono already serves may still
refuse the fused indexer alone, in which case "full" layers fall back to the
existing external-indexer path."""

from types import SimpleNamespace

from atom.models.glm52.mono.config import (
    FUSED_INDEX_MAX_SEQ,
    FUSED_INDEX_PAGE_SIZE,
    fused_indexer_refusal,
)


def _servable_atom_config(**overrides):
    """A deployment the fused indexer can serve, absent ``overrides``."""
    fields = {
        "index_cache_dtype": "fp8",
        "max_model_len": FUSED_INDEX_MAX_SEQ,
        "kv_cache_block_size": FUSED_INDEX_PAGE_SIZE,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_off_by_default_env_unset(monkeypatch):
    # The opt-in itself: with ATOM_GLM52_FUSED_INDEXER unset, the fused
    # indexer is refused even for an otherwise-servable deployment.
    monkeypatch.delenv("ATOM_GLM52_FUSED_INDEXER", raising=False)
    why = fused_indexer_refusal(_servable_atom_config())
    assert why == "fused indexer switched off"


def test_servable_when_switched_on(monkeypatch):
    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    assert fused_indexer_refusal(_servable_atom_config()) is None


def test_fp4_index_cache_dtype_is_refused_by_name(monkeypatch):
    # FP4 is a two-plane format, out of scope for this build.
    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    why = fused_indexer_refusal(_servable_atom_config(index_cache_dtype="fp4"))
    assert why is not None and "index_cache_dtype" in why and "fp4" in why


def test_max_model_len_past_capacity_is_refused_by_name(monkeypatch):
    # The capacity is enforced HERE, statically, because it cannot be enforced
    # per step: GLM-5.2 serves at --cudagraph-mode FULL, where a data-dependent
    # Python branch in the captured forward is impossible (the .item() it needs
    # raises under capture, and a baked branch would be frozen for every
    # replay). The kernel does not cover the overflow either -- TILED_INDEXER
    # is off at this capacity, so the per-query use_long_index dispatch is
    # compiled out. What happens without this refusal: with CP off, a silent
    # wrong answer (only keys [0, FUSED_INDEX_MAX_SEQ) are scored, excluding
    # the request's own newest token); with CP on, a four-rank HANG with no
    # error, because scoring stops writing index_cp_scores above
    # LOCAL_INDEX_CAPACITY while the short selector still polls it for every
    # slot. See fused_indexer_refusal's docstring.
    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    why = fused_indexer_refusal(
        _servable_atom_config(max_model_len=FUSED_INDEX_MAX_SEQ + 1)
    )
    assert why is not None and str(FUSED_INDEX_MAX_SEQ) in why


def test_max_model_len_equal_to_capacity_is_servable(monkeypatch):
    # Equality must not refuse: the highest key index the kernel addresses is
    # FUSED_INDEX_MAX_SEQ - 1, and a request whose context is exactly
    # FUSED_INDEX_MAX_SEQ tokens reaches exactly that.
    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    assert (
        fused_indexer_refusal(_servable_atom_config(max_model_len=FUSED_INDEX_MAX_SEQ))
        is None
    )


def test_mismatched_page_size_is_refused_by_name(monkeypatch):
    # The fused indexer shares block_tables with ATOM's KV cache: the index
    # page size must equal kv_cache_block_size exactly, not merely divide it.
    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    why = fused_indexer_refusal(_servable_atom_config(kv_cache_block_size=32))
    assert why is not None and "32" in why and str(FUSED_INDEX_PAGE_SIZE) in why


def test_refusals_fire_in_the_order_they_are_checked(monkeypatch):
    # The switch is checked before any deployment field: with it off, no
    # other field's value can appear in the refusal reason.
    monkeypatch.delenv("ATOM_GLM52_FUSED_INDEXER", raising=False)
    why = fused_indexer_refusal(_servable_atom_config(index_cache_dtype="fp4"))
    assert why == "fused indexer switched off"


def test_build_capacity_tracks_the_deployment_not_the_ceiling():
    """``FUSED_INDEX_MAX_SEQ`` is what the fused indexer SUPPORTS; it is not
    what every deployment compiles against.

    The scoring work map is enumerated from the compiled capacity and not
    from the request (``layout.py:448``: ``samples * ceil(capacity / 64)``
    tasks per fused layer). The keys each task reads are bounded by the
    request's own context, but the tasks are dispatched and retired either
    way -- so building a 16 k deployment at the 1 M ceiling costs 64x the
    tasks to do identical work. Measured at +11.14 % against the external
    indexer at a 1 M build versus -0.09 % at a 16 k build.
    """

    from atom.models.glm52.mono.config import fused_index_capacity
    from atom.models.glm52.mono.kernels.layout import INDEX_SEG

    # A capped deployment pays for what it capped, not for the ceiling.
    assert fused_index_capacity(INDEX_SEG) == INDEX_SEG
    assert fused_index_capacity(INDEX_SEG) != FUSED_INDEX_MAX_SEQ
    # A genuine 1 M deployment gets 1 M -- the capability is not withdrawn.
    assert fused_index_capacity(FUSED_INDEX_MAX_SEQ) == FUSED_INDEX_MAX_SEQ
    # Whole leaves only: a capacity under ``topk`` would make the merge
    # tree's fan 0 and ``index_levels`` would never terminate.
    assert fused_index_capacity(1) == INDEX_SEG
    assert fused_index_capacity(INDEX_SEG + 1) == 2 * INDEX_SEG
    assert fused_index_capacity(3 * INDEX_SEG) % INDEX_SEG == 0
    # Never above the ceiling, whatever the deployment asks for.
    assert fused_index_capacity(8 * FUSED_INDEX_MAX_SEQ) == FUSED_INDEX_MAX_SEQ


def test_build_capacity_is_legal_for_every_servable_deployment():
    """Every value the capacity function returns must satisfy what the kernel
    asserts: a whole number of ``INDEX_KEYS_PER_TASK`` (64) and of
    ``FUSED_INDEX_PAGE_SIZE`` (16), and a merge-tree depth that terminates."""

    from atom.models.glm52.mono.config import fused_index_capacity
    from atom.models.glm52.mono.kernels.layout import index_levels

    for max_model_len in (1, 4096, 16384, 16385, 32768, 131072, 1048576):
        capacity = fused_index_capacity(max_model_len)
        assert capacity % 64 == 0, max_model_len
        assert capacity % FUSED_INDEX_PAGE_SIZE == 0, max_model_len
        assert 0 <= index_levels(capacity, 2048) <= 2, max_model_len
