from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_front_role_lds_gfx1250 import (
    ATTENTION_CTAS,
    ROLE_LDS_FRONT_ABI,
    build_role_lds_front,
)


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=320 * 1024,
        tokens=rows,
        nblocks=4,
    )


def test_role_lds_front_abi_has_no_full_attnres_workspace():
    candidate = build_role_lds_front(_plan(1))
    assert candidate.abi == ROLE_LDS_FRONT_ABI
    names = candidate.abi.names
    assert "attnres_values" not in names
    assert "attnres_mixed" not in names
    assert "attnres_partials" not in names
    assert "attnres_mean_squares" not in names
    assert "front_payload" in names
    assert "router_local" not in names
    assert "latent_local" in names
    assert names[3:5] == ("blk_sm", "blk_sr")
    assert candidate.grid == (256,)
    assert candidate.block == (256,)
    assert candidate.launch_count == 1
    assert candidate.required_lds_bytes == 237_824
    assert ATTENTION_CTAS == 56


def test_role_lds_front_scales_to_s48_without_more_launches():
    s1 = build_role_lds_front(_plan(1))
    s48 = build_role_lds_front(_plan(48))
    assert s1.launch_count == s48.launch_count == 1
    assert s48.scratch_bytes > s1.scratch_bytes


def test_role_lds_front_specializes_block_write_prefix_reset():
    normal = build_role_lds_front(_plan(16))
    reset = build_role_lds_front(_plan(16), reset=True)
    projection = build_role_lds_front(_plan(16), include_attention=False)
    assert normal.name.endswith("_a1_p1_r0")
    assert reset.name.endswith("_a1_p1_r1")
    assert projection.name.endswith("_a0_p1_r0")
    assert normal.abi == reset.abi
