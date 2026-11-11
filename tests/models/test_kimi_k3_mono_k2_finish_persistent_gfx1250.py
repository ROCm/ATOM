import itertools

import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    FINISH_PACKED_FRONT_ABI,
    build_packed_finish,
)


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


@pytest.mark.parametrize("rows", (20, 40, 48))
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
