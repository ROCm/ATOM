import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_without_test_stubs(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_kimi_k3_plugin_registries_are_synchronized():
    # Out-of-process like its neighbours, and for a sharper reason than theirs.
    # This is the first import of torch in some pytest sessions, and several
    # plugin tests import torch inside a `patch.dict(sys.modules, ...)` whose
    # exit deletes the whole torch tree from `sys.modules` while torch's
    # C-level state stays initialized. Re-importing it in-process then dies
    # somewhere unrelated -- "_has_torch_function already has a docstring", or
    # an aiter JIT KeyError -- naming this file for damage another one did.
    # Running in a clean interpreter makes the check independent of what ran
    # before it, which is the only property that makes the result mean
    # anything.
    _run_without_test_stubs("""
        from atom.plugin.vllm.model_wrapper import _ATOM_MODEL_CLASSES
        from atom.plugin.vllm.register import _VLLM_MODEL_REGISTRY_OVERRIDES

        arch = "KimiK3ForConditionalGeneration"
        assert (
            _VLLM_MODEL_REGISTRY_OVERRIDES[arch]
            == "atom.plugin.vllm.models.kimi_k3:KimiK3ForCausalLMVllm"
        )
        assert (
            _ATOM_MODEL_CLASSES[arch]
            == "atom.plugin.vllm.models.kimi_k3:KimiK3ForCausalLM"
        )
        """)


def test_kimi_k3_temporal_state_uses_fp32():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.plugin.vllm.models.kimi_k3 import _get_k3_state_dtype

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(
                mamba_cache_dtype="auto",
                mamba_ssm_cache_dtype="auto",
            ),
        )
        conv_dtype, temporal_dtype = _get_k3_state_dtype(vllm_config)
        assert conv_dtype == torch.bfloat16
        assert temporal_dtype == torch.float32
        """)


def test_dense_mla_decode_pads_small_head_count():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.plugin.vllm.attention import layer_mla

        seen = {}

        def fake_mla_decode_fwd(q, _kv, output, *_args, **_kwargs):
            seen["num_heads"] = q.shape[1]
            output.fill_(1)
            return output, None

        layer_mla.mla_decode_fwd = fake_mla_decode_fwd
        attention = SimpleNamespace(
            head_repeat_factor=1,
            head_pad=4,
            kv_lora_rank=8,
            dcp_world_size=1,
            scale=1.0,
            _q_scale=None,
            _k_scale=None,
            _pad_query_heads=lambda q: torch.nn.functional.pad(
                q, (0, 0, 0, 4)
            ),
            _restore_query_heads=lambda output, num_heads: output[:, :num_heads],
        )
        decode = SimpleNamespace(
            attn_out_dtype=torch.bfloat16,
            use_persistent_metadata=False,
            paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
            paged_kv_indices=torch.tensor([0], dtype=torch.int32),
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
            fold_factor=None,
            max_qo_len=1,
        )
        output, lse = layer_mla.AttentionForVllmMLA._forward_decode(
            attention,
            torch.zeros(1, 12, 8, dtype=torch.bfloat16),
            torch.zeros(1, 8, dtype=torch.bfloat16),
            SimpleNamespace(decode=decode),
        )
        assert seen["num_heads"] == 16
        assert output.shape == (1, 12, 8)
        assert lse is None
        """)
