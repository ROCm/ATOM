# shellcheck shell=bash
# d_env_akmoe.sh: the DECODE role of recipe-audit/d_env.sh plus ganyi's C17 amd-kernels MoE block (user decision
# 2026-11-25: on). Same MoE numerics as the InferenceX vLLM MI355X image (amd_kernels mxfp4_moe, a4w4 at every M,
# FP32-gate router; vLLM benenzhu/vllm@d7e8adaf55 models/minimax_m3/amd/akr_moe.py + amd_kernels fd233ef, verified
# 11-25). The five flags act only together (excluded.txt A1). Mono stays off: it serves 1-request steps only and the
# engine blocks it on a P/D decode (ROCm/ATOM#3369).
source "$(dirname "${BASH_SOURCE[0]}")/d_env.sh"
export ATOM_M3_AK_MOE=1
export ATOM_P1_MOE_TILES=1
export ATOM_P1_MOE_IDROWS=1 ATOM_P1_MBF_ASM=1 ATOM_P1_MBF_BACK=1
for _n in ATOM_P1_MOE_TILES ATOM_P1_MOE_IDROWS ATOM_P1_MBF_ASM ATOM_P1_MBF_BACK; do
  case " ${D_ENV_NAMES[*]} " in *" ${_n} "*) ;; *) D_ENV_NAMES+=("${_n}") ;; esac
done
unset _n
