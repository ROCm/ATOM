#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# 统一 DAPO 启动：MODE=bf16|fp8|atomfp8|atombf16, TRAIN_FP8=0|1, STEPS=N。路径取容器内 $RL_ROOT/$DATA_ROOT。
set -uo pipefail
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"
MODE="${MODE:-bf16}"; TRAIN_FP8="${TRAIN_FP8:-0}"; STEPS="${STEPS:-1000}"
MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/Qwen3-8B-Base}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet}"
RUN_ID="${RUN_ID:-${MODE}$([ "$TRAIN_FP8" = 1 ] && echo -e2e)-ray-vllm-8b-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG:-$DATA_ROOT/logs/${RUN_ID}.log}"
USER_PYTHONPATH="${PYTHONPATH:-}"
ATOMRL_DIR="${ATOMRL_DIR:-$RL_ROOT/ATOM/atomrl}"
LUMEN_DIR="${LUMEN_DIR:-$ATOMRL_DIR/lumen}"
if [ ! -f "$LUMEN_DIR/lumen/config.py" ]; then
  LUMEN_DIR="$RL_ROOT/../rl_base/lumen"
fi
AITER_DIR="${AITER_DIR:-$RL_ROOT/aiter}"
if [ ! -d "$AITER_DIR/aiter" ]; then
  AITER_DIR="$RL_ROOT/../rl_base/aiter"
fi
ATOM_DIR="${ATOM_DIR:-$RL_ROOT/ATOM}"
if [ ! -f "$ATOM_DIR/atom/rollout/async_engine.py" ]; then
  ATOM_DIR="$RL_ROOT/../rl_base/ATOM"
fi
cd "$ATOMRL_DIR"

# ---- 通用 env（BF16/FP8 共用）----
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
# Honor a caller-provided PYTORCH_CUDA_ALLOC_CONF, including an explicit empty value
# (`PYTORCH_CUDA_ALLOC_CONF= ... bash run_dapo.sh`) to drop expandable_segments entirely.
_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"
if [ -n "$_ALLOC_CONF" ]; then
  export PYTORCH_CUDA_ALLOC_CONF="$_ALLOC_CONF"
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 CUDA_DEVICE_MAX_CONNECTIONS=1
# Same opt-out shape as PYTORCH_CUDA_ALLOC_CONF above, and for a sharper reason.
# On ROCm 7.14 / RCCL 2.28.9 / torch 2.12 this knob breaks
# intra-node reduce-scatter, which is how Megatron's distributed optimizer
# reduces gradients: the coalesced form hangs and the plain form returns correct
# numbers while leaving the HIP context in an error state, so the step dies in
# an unrelated one-element allocation with "CUDA error: invalid argument".
# Bisected against every other knob on the line above plus NCCL_CUMEM_ENABLE;
# this one reproduces on its own. all-to-all is unaffected.
_FRAG_ALLOC="${HSA_DISABLE_FRAGMENT_ALLOCATOR-1}"
if [ -n "$_FRAG_ALLOC" ]; then
  export HSA_DISABLE_FRAGMENT_ALLOCATOR="$_FRAG_ALLOC"
else
  unset HSA_DISABLE_FRAGMENT_ALLOCATOR
fi
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN ATOM_DISABLE_VLLM_PLUGIN=1
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export LUMEN_DISABLE_HF_ATTN_PATCH=1 MODEL_NAME="$MODEL_PATH"
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb" ATOMRL_LOG_LEVEL=INFO
# MEGATRON_PATH first so Ray actors import this tree instead of a pip megatron-core.
_PP="$ATOMRL_DIR:$AITER_DIR:$ATOM_DIR:${PYTHONPATH:-}"
if [ -n "${MEGATRON_PATH:-}" ]; then
  _PP="$MEGATRON_PATH:$_PP"
fi
export PYTHONPATH="$_PP"
for _wandb_key in "$RL_ROOT/wandb.key" "$RL_ROOT/../wandb.key"; do
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$_wandb_key" ]; then
    export WANDB_API_KEY="$(cut -d= -f2- "$_wandb_key" | tr -d '[:space:]')"
  fi
done

EXTRA_ARGS=()
if [ "$MODE" = "atomfp8" ] || [ "$MODE" = "atom_fp8" ] || \
   [ "$MODE" = "atombf16" ] || [ "$MODE" = "atom_bf16" ]; then
  ATOM_BF16=0
  if [ "$MODE" = "atombf16" ] || [ "$MODE" = "atom_bf16" ]; then
    ATOM_BF16=1
    CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_bf16_longrun.yaml
  elif [ "${ATOM_DEBUG:-0}" = "1" ]; then
    CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_debug.yaml
  else
    CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_longrun.yaml
  fi
  unset ATOM_DISABLE_VLLM_PLUGIN
  export ATOMRL_ATOM_AITER_SRC="${ATOMRL_ATOM_AITER_SRC:-$AITER_DIR}"
  _PP="$ATOMRL_DIR:$AITER_DIR:$ATOM_DIR:$USER_PYTHONPATH"
  if [ -n "${MEGATRON_PATH:-}" ]; then
    _PP="$MEGATRON_PATH:$_PP"
  fi
  export PYTHONPATH="$_PP"
  # ATOM FP8 正式方案：no-eager + compilation level=3。dynamo 只有 ATOM rollout 需要，
  # 由 ATOMReplicaManager 通过 Ray runtime_env 注入 TORCHDYNAMO_DISABLE=0，这里保持全局
  # TORCHDYNAMO_DISABLE=1，避免训练 actor 被动继承。每个 colocated replica 仍用独立
  # torch compile cache，避免 8 个 rank0 并发写同一路径触发 Inductor rename race。
  # ATOM_ISOLATE_TORCH_COMPILE_CACHE 只按 replica 隔离，不按量化模式隔离。若 atomfp8 与
  # atombf16 共用同一个 cache root，后跑的一方会加载另一方编译出的图，在 AOTAutograd 里
  # 报 IndexError: list index out of range（runtime_wrappers.py args[i] 参数个数不匹配），
  # 表现为 ATOMRayServer 初始化阶段 SHUTDOWN。因此把 MODE 加进默认路径。
  export ATOM_ISOLATE_TORCH_COMPILE_CACHE=1
  export ATOM_TORCH_COMPILE_CACHE_ROOT="${ATOM_TORCH_COMPILE_CACHE_ROOT:-/tmp/atom_torch_compile_cache/$MODE}"
  # ATOM 的汇编 paged-decode kernel 在一条序列的上下文恰好占满 16 页、最后一页不满时
  # 返回有限但错误的结果，于是每次运行都有极少数 token 拿到任意 logprob。它躲得过
  # mismatch/abs_diff（分歧 token 的数量与平均幅度都不变），只在二次型的
  # mismatch/chi2_token 上现形：例子 4 第 3 步 5761.69，上一步 0.0138。改走 Triton。
  # 设在这里而不是只设在 examples/DAPO/examples.yaml 的例子表里，是因为这条要求属于 ATOM
  # 这条路径本身，不属于某几个例子编号——手工调 run_dapo.sh 的人同样漏不掉。
  # ATOM 自己的修复进入所钉的 commit 后即可删除。
  export ATOM_FORCE_ATTN_TRITON="${ATOM_FORCE_ATTN_TRITON:-1}"
  export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0
  # Match the vLLM fp8 training-side configuration exactly: standard Lumen FP8
  # blockwise2d linear + norm, no HF attention patch and no rollout-specific
  # early-return path. Rollout is ATOM; actor training stays on the vLLM fp8 path.
  #
  # Opt-in exception: a caller-provided LUMEN_ROLLOUT=ATOM (verl FP8 recipe knob)
  # aligns the actor forward (norm/sdpa/linear/mlp) with ATOM inference. That path
  # returns early in LumenConfig.enable(), so actor FP8 quantization is skipped and
  # the actor trains in BF16 regardless of TRAIN_FP8.
  if [ -z "${LUMEN_ROLLOUT:-}" ]; then
    unset LUMEN_ROLLOUT
  else
    export LUMEN_ROLLOUT
  fi
  export LUMEN_DISABLE_HF_ATTN_PATCH=1
  if [ "$ATOM_BF16" = "1" ]; then
    # Pure BF16 baseline: keep both actor and ATOM rollout unquantized. Avoid
    # importing the Lumen/AITER norm patch in actor initialization; the HF
    # Qwen3 RMSNorm is already model-sensitive and this keeps BF16 comparable
    # with MODE=bf16 while retaining the ATOM rollout backend.
    _KEEP_NORM="${LUMEN_NORM:-}"
    unset LUMEN_NORM LUMEN_FP8 FP8_PARAM_MANAGER LUMEN_FP8_SCALING
    unset LUMEN_FP8_FORMAT LUMEN_FP8_BLOCK_SIZE LUMEN_FP8_ATTN
    unset LUMEN_FP8_QUANT_TYPE LUMEN_ATTN_BACKEND LUMEN_FP8_WGRAD
    [ -n "$_KEEP_NORM" ] && export LUMEN_NORM="$_KEEP_NORM"
  else
    export LUMEN_NORM=1
  fi
  if [ "$ATOM_BF16" = "0" ] && [ "$TRAIN_FP8" = "1" ]; then
    export LUMEN_FP8=1 FP8_PARAM_MANAGER=0
    export LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_FORMAT=fp8_e4m3 LUMEN_FP8_BLOCK_SIZE=128
    export LUMEN_FP8_ATTN=none LUMEN_FP8_QUANT_TYPE=blockwise LUMEN_ATTN_BACKEND=auto
    export LUMEN_FP8_WGRAD="${LUMEN_FP8_WGRAD:-0}"
  fi
  # no-eager level=3 正式方案：enforce_eager=false + compilation_config.level=3 + sleep2
  # （sleep_mode 训练前释放 rollout KV/weights/CUDA graph，避免 backward OOM）。
  EXTRA_ARGS+=(
    policy.generation.vllm_cfg.enforce_eager=false
    policy.generation.atom_cfg.engine_kwargs.enforce_eager=false
    policy.generation.atom_cfg.engine_kwargs.compilation_config.level=3
    policy.generation.vllm_cfg.enable_sleep_mode=true
    policy.generation.vllm_cfg.sleep_level=2
  )
  if [ "$ATOM_BF16" = "0" ]; then
    ATOM_ONLINE_QUANT="${ATOM_ONLINE_QUANT:-per_block_fp8}"
    EXTRA_ARGS+=(
      policy.generation.atom_cfg.online_quant_config.global_quant_config="$ATOM_ONLINE_QUANT"
    )
  fi
elif [ "$MODE" = "fp8" ]; then
  CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml
  # rollout per_block_fp8 + AITER unified attention
  export ATOMRL_FP8_PER_BLOCK=1
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_AITER_LINEAR=0
  if [ "$TRAIN_FP8" = "1" ]; then    # FP8 E2E 训练（blockwise2d，param manager 必须关）
    export LUMEN_FP8=1 FP8_PARAM_MANAGER=0 LUMEN_NORM=1
    export LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_FORMAT=fp8_e4m3 LUMEN_FP8_BLOCK_SIZE=128 LUMEN_FP8_ATTN=none
  fi
else
  CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_longrun.yaml
  export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0
fi
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

# Same flags as Megatron-LM examples/qwen3/train_qwen3.sh ENABLE_MORI=true.
if [ "${ENABLE_MORI:-false}" = "true" ] || [ "${ENABLE_MORI:-0}" = "1" ]; then
  export MORI_SHMEM_LOG_LEVEL="${MORI_SHMEM_LOG_LEVEL:-INFO}"
  echo "[INFO] MORI EP: MORI_SHMEM_MODE=${MORI_SHMEM_MODE:-}"
  echo "[INFO] MORI EP: MORI_SHMEM_LOG_LEVEL=${MORI_SHMEM_LOG_LEVEL}"
  EXTRA_ARGS+=(
    policy.training.megatron_cfg.moe_token_dispatcher_type=flex
    policy.training.megatron_cfg.moe_flex_dispatcher_backend=mori
  )
fi

echo "$LOG" > /tmp/run_dapo_log.txt
echo "=== MODE=$MODE TRAIN_FP8=$TRAIN_FP8 STEPS=$STEPS  CONFIG=$CONFIG  LOG=$LOG ==="

# Multi-node: one Ray cluster across the nodes, started from this script so
# every raylet -- and so every Ray worker -- carries the environment set above.
#   RAY_ROLE=head    start the Ray head on NODE_IP, then exit
#   RAY_ROLE=worker  join HEAD_IP from NODE_IP, then exit
#   RAY_ROLE=driver  run the trainer against that cluster (cluster.ray_address=auto)
# Unset: single node, the trainer starts its own Ray.
case "${RAY_ROLE:-}" in
  head|worker)
    ulimit -n 524288 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
    if [ "$RAY_ROLE" = head ]; then
      ray start --head --node-ip-address="${NODE_IP:?}" --port="${RAY_PORT:-6379}" \
        --num-gpus="${GPUS_PER_NODE:-8}" --dashboard-host=0.0.0.0
    else
      ray start --address="${HEAD_IP:?}:${RAY_PORT:-6379}" --node-ip-address="${NODE_IP:?}" \
        --num-gpus="${GPUS_PER_NODE:-8}"
    fi
    exit $? ;;
  driver|"") ;;
  *) echo "unknown RAY_ROLE=$RAY_ROLE (head|worker|driver)" >&2; exit 2 ;;
esac

# 清理旧进程
[ "${RAY_ROLE:-}" = driver ] || ray stop --force >/dev/null 2>&1 || true
python3 - <<'PY'
import os
import signal
import subprocess

patterns = (
    "atomrl.trainer.main",
    "VLLMRayServer",
    "ATOMRayServer",
    "VLLM::EngineCore",
    "EngineCore",
    "spawn_main",
    "torch/_inductor/compile_worker",
    "multiprocessing.resource_tracker",
)
skip = {os.getpid(), os.getppid()}
out = subprocess.check_output(["ps", "-eo", "pid,ppid,stat,cmd"], text=True)
for line in out.splitlines()[1:]:
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        continue
    pid = int(parts[0])
    stat = parts[2]
    cmd = parts[3]
    if pid in skip or "Z" in stat:
        continue
    if any(p in cmd for p in patterns):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY
sleep 8

python3 -u -m atomrl.trainer.main --config "$CONFIG" \
  policy.model_name="$MODEL_PATH" reward.dataset="$TRAIN_FILE" val_dataset="$VAL_FILE" \
  num_training_steps="$STEPS" seed=10086 "${EXTRA_ARGS[@]}" ${EXTRA_OVERRIDE:-} > "$LOG" 2>&1
exit_code=$?
echo "=== exit=$exit_code ==="
exit "$exit_code"
