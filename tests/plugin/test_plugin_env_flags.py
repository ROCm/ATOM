import importlib


def test_disable_vllm_plugin_flag_disables_platform(monkeypatch):
    # ATOM_DISABLE_VLLM_PLUGIN takes precedence:
    # when it is 1, vLLM should not get ATOM platform/attention at all.
    for disable_attention in ("0", "1"):
        monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN", "1")
        monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN_ATTENTION", disable_attention)

        import atom.plugin.vllm.platform as platform_module
        import atom.plugin.vllm.register as register_module

        importlib.reload(platform_module)
        importlib.reload(register_module)

        assert platform_module.ATOMPlatform is None
        assert register_module.register_platform() is None
