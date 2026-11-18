import itertools

import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    FINISH_PACKED_FRONT_ABI,
    build_packed_finish,
)


def _rows(*rows):
    """``rows``, those past the token cap skipped until it is raised."""
    from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS

    return [
        pytest.param(
            r,
            marks=pytest.mark.skipif(r > GFX1250_MAX_TOKENS, reason="token cap"),
        )
        for r in rows
    ]


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=320 * 1024,
        tokens=rows,
        nblocks=4,
    )


@pytest.mark.parametrize("rows", (1, 16, 17, 32, 33, 48))
@pytest.mark.parametrize("gu_interleave", (False, True))
@pytest.mark.parametrize("include_shared", (False, True))
@pytest.mark.parametrize("include_routed", (False, True))
def test_persistent_finish_is_one_launch_with_static_resident_grid(
    rows, gu_interleave, include_shared, include_routed
):
    candidate = build_packed_finish(
        _plan(rows),
        gu_interleave=gu_interleave,
        include_shared=include_shared,
        include_routed=include_routed,
    )
    assert candidate.grid == (256,)
    assert candidate.block == (256,)
    assert candidate.launch_count == 1
    assert candidate.required_lds_bytes == 113_664
    assert f"_gu{int(gu_interleave)}_sh{int(include_shared)}_" in candidate.name
    assert candidate.name.endswith(f"_r{int(include_routed)}")


def test_persistent_finish_abi_is_packed_and_rank_local():
    candidate = build_packed_finish(_plan(48))
    assert candidate.abi == FINISH_PACKED_FRONT_ABI
    names = candidate.abi.names
    assert names[0] == "front_gathered"
    assert "routed_intermediate_fp4" in names
    assert "routed_intermediate_scales" in names
    assert "latent_fp4" in names
    assert "latent_scales" in names
    assert not any(
        forbidden in name
        for name in names
        for forbidden in ("peer", "ipc", "xgmi", "collective")
    )


def test_fp8_shared_down_is_the_grouped_moe_finish_only():
    candidate = build_packed_finish(
        _plan(40), include_routed=False, replicated_front=True, fp8_shared_down=True
    )
    assert candidate.name.endswith("_rf_f8")
    names = candidate.abi.names
    assert names.index("shared_down_scale") == names.index("shared_down_weight") + 1
    with pytest.raises(ValueError, match="FP8 shared down"):
        build_packed_finish(_plan(40), fp8_shared_down=True)


@pytest.mark.parametrize("rows", _rows(20, 40, 48, 50, 55, 60, 63, 64))
def test_precomputed_herd_scratch_holds_every_output_apart(rows):
    from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
        herd_scratch_layout,
    )

    candidate = build_packed_finish(
        _plan(rows),
        include_routed=False,
        replicated_front=True,
        herd=True,
        precompute_herd=True,
    )
    assert candidate.name.endswith("_rf_herd_hp")
    spans = sorted(
        (at, at + 4 * words) for at, words in herd_scratch_layout(rows).values()
    )
    assert spans[0][0] >= 8  # past the grid barrier's epoch
    assert all(end <= start for (_, end), (start, _) in itertools.pairwise(spans))
    assert spans[-1][1] <= candidate.scratch_bytes
    with pytest.raises(ValueError, match="HERD grouped-MoE finish"):
        build_packed_finish(_plan(rows), precompute_herd=True)


@pytest.mark.parametrize("herd", (False, True))
@pytest.mark.parametrize("rolled", (False, True))
def test_a_rolled_finish_is_a_build_of_its_own(herd, rolled):
    candidate = build_packed_finish(
        _plan(40),
        include_routed=False,
        replicated_front=True,
        herd=herd,
        fp8_shared_down=True,
        precompute_herd=herd,
        rolled=rolled,
    )
    assert ("_b256_rl_" in candidate.name) == rolled
    assert candidate.name.endswith("_rf_herd_f8_hp" if herd else "_rf_f8")


def test_herd_finalize_keeps_its_48_row_layout_and_grows_past_it():
    from atom.models.kimi_k3.mono.kernels.stages import moe_gfx1250

    at48 = moe_gfx1250.herd_finalize_words(48)
    assert all(moe_gfx1250.herd_finalize_words(r)["slots"] == at48["slots"] for r in (1, 20, 48))
    assert moe_gfx1250.herd_finalize_words(64)["slots"] == 64 * 16
    with pytest.raises(ValueError):
        moe_gfx1250.herd_finalize_words(65)


@pytest.mark.parametrize("rows", _rows(48, 62, 63, 64))
@pytest.mark.parametrize("parallel", (False, True))
def test_the_herd_cta_s_shared_down_tasks_find_takers(rows, parallel):
    candidate = build_packed_finish(
        _plan(rows),
        gu_interleave=True,
        include_routed=False,
        replicated_front=True,
        herd=True,
        fp8_shared_down=True,
        precompute_herd=True,
        parallel_herd=parallel,
    )
    assert candidate.grid == (256,)
