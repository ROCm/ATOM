# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Source-level parity tests for the Lumen shell launchers."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LUMEN_ROOT = REPO_ROOT / "lumen"

DEV_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.dev"
LUMEN_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.lumen"
LLAMA31_LAUNCHER = LUMEN_ROOT / "examples" / "llama31" / "run_pretrain.sh"
LLAMA31_CONFIG = LUMEN_ROOT / "examples" / "llama31" / "config_MI355X_1x8x1.sh"
LLAMA2_LAUNCHER = LUMEN_ROOT / "examples" / "llama2" / "run_finetune.sh"
LLAMA2_CONFIG = LUMEN_ROOT / "examples" / "llama2" / "config_MI355X_1x8x1.sh"
DSV4_TWO_NODE_LAUNCHER = (
    LUMEN_ROOT / "examples" / "dsv4" / "launch_dsv4_flash_pretrain_2node.sh"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _between(text: str, start_marker: str, end_marker: str) -> str:
    return text.split(start_marker, 1)[1].split(end_marker, 1)[0]


class TestLauncherParity:
    def test_example_dockerfiles_align_root_training_stack_contract(self):
        for dockerfile in (DEV_DOCKERFILE, LUMEN_DOCKERFILE):
            source = _source(dockerfile)
            assert "ldconfig" in source
            assert "torchao" in source
            assert "requirements.txt" in source
            assert "LUMEN_BUILD_EXTENSIONS=1 pip install" in source
        lumen_source = _source(LUMEN_DOCKERFILE)
        assert "apt-get update &&" in lumen_source
        assert "libopenmpi-dev libibverbs-dev rdma-core libpci-dev" in lumen_source

    def test_example_launchers_expose_use_sdma_switch(self):
        for launcher in (LLAMA2_LAUNCHER, LLAMA31_LAUNCHER):
            source = _source(launcher)
            assert 'if [ "${USE_SDMA}" -gt 0 ]; then' in source
            assert 'CMD_SUFFIX="${CMD_SUFFIX} --use-sdma"' in source
            assert 'export MORI_ENABLE_SDMA="${USE_SDMA}"' in source

    def test_launchers_resolve_packaged_lumen_source(self):
        for launcher in (LLAMA2_LAUNCHER, LLAMA31_LAUNCHER):
            source = _source(launcher)
            assert 'ATOM_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"' in source
            assert 'LUMEN_SOURCE_ROOT="${ATOM_ROOT}/atomrl/atomrl/lumen"' in source
            assert '"${LUMEN_SOURCE_ROOT}/models/perf_env.sh"' in source
            assert "${REPO_ROOT}/lumen/" not in source

    def test_dsv4_two_node_syncs_packaged_source_fail_fast(self):
        source = _source(DSV4_TWO_NODE_LAUNCHER)
        assert 'ATOM_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"' in source
        assert 'LUMEN_SOURCE_ROOT="${ATOM_ROOT}/atomrl/atomrl/lumen"' in source
        assert '"${LUMEN_SOURCE_ROOT}/${_rel}"' in source
        assert '"${WORKER_SSH}:${LUMEN_SOURCE_ROOT}/${_rel}"' in source
        assert 'rsync -a -e "${_rsync_rsh}"' in source
        assert "2>/dev/null || true" not in _between(
            source, "_rsync_rsh=", "docker rm -f"
        )

    def test_llama2_launcher_builds_shared_checkpoint_suffix(self):
        source = _source(LLAMA2_LAUNCHER)
        assert "CMD_SUFFIX" in source
        assert "--use-ckpt" in source
        assert "--resume-from-hf" in source
        assert "--save-ckpt" in source
        assert "--tag" in source
        assert "--fp8-params" in source

    def test_llama2_launcher_computes_eval_interval_from_eval_every(self):
        source = _source(LLAMA2_LAUNCHER)
        assert (
            'if [ "${EVAL_EVERY}" -gt 0 ] && [ "${EVAL_INTERVAL}" -eq 0 ]; then'
            in source
        )
        assert "EVAL_INTERVAL=$(( (EVAL_EVERY + GBS - 1) / GBS ))" in source

    def test_llama2_megatron_launcher_emits_shared_experiment_flags(self):
        source = _source(LLAMA2_LAUNCHER)
        megatron_block = _between(
            source,
            "run_megatron() {",
            "###############################################################################\n# FSDP BACKEND",
        )
        for flag in (
            "--seed",
            "--eval-iters ${EVAL_ITERS}",
            "--continual-ckpt-path",
            "--target-log-ppl",
            "--step-time-atol",
            "--ckpt-start-step",
            "--eval-every",
            "--start-eval-at",
        ):
            assert flag in megatron_block

    def test_llama2_fsdp_launcher_emits_shared_contract_flags(self):
        source = _source(LLAMA2_LAUNCHER)
        fsdp_block = _between(
            source,
            "run_fsdp() {",
            "###############################################################################\n# DISPATCH",
        )
        for flag in (
            "--lr-warmup-steps",
            "--continual-ckpt-path",
            "--target-log-ppl",
            "--step-time-atol",
            "--ckpt-start-step",
            "--eval-every",
            "--start-eval-at",
        ):
            assert flag in fsdp_block
        for flag in (
            "--primus-turbo-fp8-attention",
            "--primus-turbo-mxfp8-attention",
            "--dbg-attn-output",
        ):
            assert flag not in fsdp_block

    def test_llama2_config_defines_shared_launcher_env_defaults(self):
        source = _source(LLAMA2_CONFIG)
        for env_name in (
            "SEED",
            "EVAL_ITERS",
            "USE_SDMA",
            "LR_WARMUP_STEPS",
            "USE_CKPT",
            "FROM_HF",
            "SAVE_CKPT",
            "CONTINUAL_CKPT",
            "CKPT_START_STEP",
            "FP8_PARAMS",
            "TARGET_LOG_PPL",
            "STEP_TIME_ATOL",
            "EVAL_EVERY",
            "START_EVAL_AT",
        ):
            assert f"export {env_name}" in source
        for env_name in (
            "PRIMUS_FP8_ATTN",
            "PRIMUS_MXFP8_ATTN",
            "DBG_ATTN_OUTPUT",
        ):
            assert f"export {env_name}" not in source

    def test_llama31_config_defines_use_sdma_default(self):
        source = _source(LLAMA31_CONFIG)
        assert "export USE_SDMA=0" in source
