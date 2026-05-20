"""No-monkey-patch guards for GLM5 RTP MLA M1.5 forward."""

from pathlib import Path


_ATOM_ROOT = Path(__file__).resolve().parents[2]


def _read_plugin_file(relative_path: str) -> str:
    return (_ATOM_ROOT / relative_path).read_text()


def test_rtp_mla_prepare_no_longer_contains_deepseek_forward_monkey_patch():
    assert not (
        _ATOM_ROOT / "atom/plugin/rtpllm/attention_backend/rtp_mla_prepare.py"
    ).exists()


def test_glm5_wrapper_does_not_import_or_call_deepseek_mla_patch():
    source = _read_plugin_file("atom/plugin/rtpllm/models/glm5.py")

    assert "apply_deepseek_mla_rtpllm_patch" not in source

