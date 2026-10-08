"""SGLang MiniMax mono must refuse plugin mode before reading cudagraph_mode."""

from types import SimpleNamespace

from atom.models.minimax_m3.mono import dispatch
from atom.plugin.sglang.minimax_m3_bridge import install_minimax_mono_plugin_refusal


def test_sglang_mono_refusal_ignores_missing_cudagraph_mode():
    original = dispatch._config_refusal
    flag = getattr(dispatch, "_atom_sglang_mono_plugin_refusal", False)
    try:
        dispatch._atom_sglang_mono_plugin_refusal = False
        install_minimax_mono_plugin_refusal()
        why = dispatch._config_refusal(
            SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=None)),
            SimpleNamespace(),
        )
        assert why == "plugin mode"
        install_minimax_mono_plugin_refusal()
        assert dispatch._config_refusal(None, None) == "plugin mode"
    finally:
        dispatch._config_refusal = original
        dispatch._atom_sglang_mono_plugin_refusal = flag
