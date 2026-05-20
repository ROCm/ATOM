"""Static guards for the GLM5 rtp-llm plugin path."""

import ast
from pathlib import Path


_ATOM_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORT_TIME_SPARSE_KERNELS = {
    "flashmla_sparse",
    "flash_mla",
    "sparse_mla",
    "attention_mla_sparse",
}


def _read_plugin_file(relative_path: str) -> str:
    return (_ATOM_ROOT / relative_path).read_text()


def test_glm5_wrapper_does_not_use_mha_or_qwen_patches():
    source = _read_plugin_file("atom/plugin/rtpllm/models/glm5.py")

    assert "RTPFullAttention" not in source
    assert "apply_attention_mha_rtpllm_patch" not in source
    assert "apply_attention_gdn_rtpllm_patch" not in source
    assert "apply_qwen3_next_rtpllm_patch" not in source


def test_glm5_wrapper_does_not_reference_deepseek_mla_patch():
    source = _read_plugin_file("atom/plugin/rtpllm/models/glm5.py")

    assert "apply_deepseek_mla_rtpllm_patch" not in source


def test_rtp_mla_prepare_does_not_keep_native_forward_mirror_helpers():
    assert not (
        _ATOM_ROOT / "atom/plugin/rtpllm/attention_backend/rtp_mla_prepare.py"
    ).exists()


def test_glm5_mla_backend_is_not_full_attention_adapter():
    source = _read_plugin_file("atom/plugin/rtpllm/attention_backend/rtp_mla_attention.py")

    assert "class RTPMLAAttention" in source
    assert "use_mla" in source
    assert "RTPFullAttention" not in source


def test_sparse_mla_backend_has_no_import_time_cuda_sparse_kernel_dependencies():
    backend_path = _ATOM_ROOT / "atom/plugin/rtpllm/attention_backend/rtp_sparse_mla_backend.py"
    assert backend_path.exists()

    tree = ast.parse(backend_path.read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert not any(
        forbidden in module_name.split(".")
        for module_name in imported_modules
        for forbidden in _FORBIDDEN_IMPORT_TIME_SPARSE_KERNELS
    )

