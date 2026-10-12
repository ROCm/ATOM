"""Every name the scheduler patches call must exist.

Three edits in this branch have deleted a function its own callers use,
because the edit replaced a region "from def A to def B" and something
lived between them. Each time the whole unit suite stayed green and an
engine boot paid for it -- most recently `NameError: name
'_log_cap_decision' is not defined`, in the EngineCore process, on the
first request.
"""

from atom.plugin.vllm import scheduler as mod

LIVE = [
    "_image_only_cap",
    "_log_cap_decision",
    "apply_vllm_v41_local_hit_cap_patch",
]


def test_every_live_name_is_present_and_callable():
    missing = [n for n in LIVE if not callable(getattr(mod, n, None))]
    assert not missing, f"gone from the module its callers use: {missing}"


def test_the_hit_cap_wrapper_can_be_installed():
    """Installing is what proves the names it closes over resolve."""
    mod.apply_vllm_v41_local_hit_cap_patch()
    from vllm.v1.core.sched.scheduler import Scheduler

    assert getattr(
        Scheduler._get_local_prefix_cache_hit, "_atom_v41_hit_cap_patched", False
    )


def test_the_decision_log_takes_both_shapes():
    """It is called twice per hit: once before the early return with no
    `capped`, once after with it. A signature that only fits one of those
    is a NameError's quieter cousin."""
    mod._log_cap_decision(0, None)
    mod._log_cap_decision(1280, object(), 1280)
