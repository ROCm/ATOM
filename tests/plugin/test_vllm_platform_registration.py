import pytest

register = pytest.importorskip("atom.plugin.vllm.register")

ATOM_PLATFORM = "atom.plugin.vllm.platform.ATOMPlatform"


def test_platform_is_returned_while_vllm_is_still_importing(monkeypatch):
    def import_cycle():
        raise ImportError(
            "cannot import name 'direct_register_custom_op' from partially "
            "initialized module 'vllm.utils.torch_utils'"
        )

    monkeypatch.setattr(register, "disable_vllm_plugin", False)
    monkeypatch.setattr(register, "_apply_platform_patches", import_cycle)

    assert register.register_platform() == ATOM_PLATFORM


def test_platform_patches_still_run_when_vllm_has_loaded(monkeypatch):
    applied = []
    monkeypatch.setattr(register, "disable_vllm_plugin", False)
    monkeypatch.setattr(register, "_apply_platform_patches", lambda: applied.append(1))

    assert register.register_platform() == ATOM_PLATFORM
    assert applied == [1]


def test_other_platform_patch_failures_are_not_hidden(monkeypatch):
    def broken():
        raise RuntimeError("a real bug in a patch")

    monkeypatch.setattr(register, "disable_vllm_plugin", False)
    monkeypatch.setattr(register, "_apply_platform_patches", broken)

    with pytest.raises(RuntimeError):
        register.register_platform()
