from pathlib import Path


def test_dsv4_launch_preserves_explicit_aiter_setting() -> None:
    script = (
        Path(__file__).parents[2] / "examples" / "GRPO" / "run_grpo_dsv4.sh"
    ).read_text(encoding="utf-8")

    assert 'export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"' in script
