import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_finish_persistent_gfx1250 as fin
from atom.models.kimi_k3.mono.kernels.stages import moe_tdm_gfx1250 as mt


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=4
    )


def _finish(rows, **kwargs):
    return fin.build_packed_finish(
        _plan(rows),
        gu_interleave=True,
        include_routed=False,
        replicated_front=True,
        herd=True,
        fp8_shared_down=True,
        precompute_herd=True,
        **kwargs,
    )


@pytest.mark.parametrize("rows", (16, 20, 32, 33, 40, 48))
def test_a_parallel_herd_finish_is_a_build_of_its_own(rows):
    base = _finish(rows)
    candidate = _finish(rows, parallel_herd=True)
    assert candidate.name == base.name + "_ph"
    assert candidate.abi == base.abi
    assert candidate.grid == base.grid
    assert candidate.scratch_bytes == base.scratch_bytes


def test_parallel_herd_needs_the_precomputed_herd():
    with pytest.raises(ValueError, match="in parallel"):
        fin.build_packed_finish(
            _plan(40), include_routed=False, replicated_front=True, parallel_herd=True
        )


@pytest.mark.parametrize("rows", (1, 20, 40, 48))
def test_the_routing_ctas_leave_every_shared_task_to_the_rest(rows):
    shared_ctas = 256 - rows
    slots = -(-fin.SHARED_DOWN_TASKS // shared_ctas)
    assert slots * shared_ctas >= fin.SHARED_DOWN_TASKS
    assert slots <= 3


@pytest.mark.parametrize("herd", (False, True))
def test_a_profile_finish_stamps_past_its_scratch(herd):
    kwargs = {"herd": herd, "precompute_herd": herd}
    plain = fin.build_packed_finish(
        _plan(40), include_routed=False, replicated_front=True, **kwargs
    )
    profiled = fin.build_packed_finish(
        _plan(40), include_routed=False, replicated_front=True, profile=True, **kwargs
    )
    assert profiled.name == plain.name + "_profile"
    at = fin.profile_offset(40, herd)
    assert at % 64 == 0 and at >= plain.scratch_bytes
    assert profiled.scratch_bytes == at + 4 * 256 * fin.PROFILE_WORDS
    assert len(fin.PROFILE_STAMPS) <= fin.PROFILE_WORDS


def test_a_profile_moe_holds_its_first_weights_stamp():
    assert mt.profile_words() == mt.MOE_STATE_WORDS + 256 * 12
    assert mt.build_moe(20, profile=True).name.endswith("_profile")


def test_the_parallel_herd_switch_defaults_on(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_PARALLEL_HERD", raising=False)
    assert envs.ATOM_KIMI_MONO_PARALLEL_HERD is True
    monkeypatch.setenv("ATOM_KIMI_MONO_PARALLEL_HERD", "0")
    assert envs.ATOM_KIMI_MONO_PARALLEL_HERD is False


def test_only_a_parallel_herd_finish_bounds_its_polls():
    assert _finish(40, parallel_herd=True, herd_poll_limit=1024).name.endswith("_ph_pl1024")
    with pytest.raises(ValueError, match="bounds its candidate polls"):
        _finish(40, herd_poll_limit=1024)
