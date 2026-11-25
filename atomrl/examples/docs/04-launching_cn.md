> [Examples README](../README_cn.md) > 启动

# 4. 启动

## 4.1 Config 与规模

全部在 `examples/DAPO/configs/`：

```text
# 1  8B BF16
dapo_qwen3_8b_ray_vllm_smoke.yaml                     resp=512
dapo_qwen3_8b_ray_vllm_longrun.yaml

# 2、3  8B vLLM FP8（共用 config，训练精度由 TRAIN_FP8 控制）
dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml                 resp=512
dapo_qwen3_8b_ray_vllm_fp8_4k_smoke.yaml              resp=4096
dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml

# 4  8B ATOM FP8
dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml              resp=4096
dapo_qwen3_8b_ray_atom_fp8_longrun.yaml

# 5  8B ATOM BF16（规模与例子 4 逐字段相同，只关掉量化）
dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml             resp=4096
dapo_qwen3_8b_ray_atom_bf16_longrun.yaml

# 6  MoE FSDP2
dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml
dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml

# 7  MoE Megatron EP=8
dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml
dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml
dapo_qwen3moe_a3b_ray_megatron_verlref_4k_longrun.yaml   # 压缩版，几小时出结论
```

**8B 长跑规模**（例子 1-5 相同）：1000 步、`train_global_batch_size=512`（32 prompt x 16）、
`gen_batch_size=96`、`max_response_length=20480`、`max_total_sequence_length=21504`、
lr 1e-6 / warmup 10 / wd 0.1 / clip_grad 1.0、clip 0.2/0.28/10 + token-mean、
`overlong_buffer` 512/1.0、`filter_groups` acc / max 10 轮、`rollout_is=token` 阈值 2.0、
`val_steps=10` / `save_steps=50` / seed 10086。
BF16 与 FP8 的 config **只差 `vllm_cfg.quantization` 一行**。

**MoE 长跑规模**（例子 6、7 相同）：prompt=2048、resp=20480、
**128 prompt x 16 = 2048 序列**、`gen_batch_size=384`、**lr warmup = 0**、1000 步。
两个 config 除了 `policy.training_backend` 和 `megatron_cfg` **逐字段相同**，
所以两条线的任何差异都只能来自训练后端。

> **注意单位**：`train_global_batch_size` 是**序列数**（2048），`gen_batch_size` 是
> **prompt 数**（384）。框架用 `train_prompts = train_global_batch_size // num_generations`
> 反推 prompt 数。

**例子 7 的 `megatron_cfg`（长跑）**：

```yaml
use_distributed_optimizer: true
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
context_parallel_size: 1
expert_model_parallel_size: 8       # 128 专家分到 8 卡，每卡 16 个
sequence_parallel: false
moe_grouped_gemm: true
moe_permute_fusion: true
moe_aux_loss_coeff: 0.0
moe_router_dtype: fp32              # 与 ATOMRL_FP32_MOE_ROUTER=1 配对
recompute_granularity: full         # resp=20480 必需
recompute_method: uniform
recompute_num_layers: 1
log_probs_chunk_size: 1024
enable_dynamic_batch: true
max_tokens_per_gpu: 8192            # 不是 22528，见排障
```

**拓扑为什么选 EP=8**：`DP = 8 / (TP x PP x CP) = 8`，和 FSDP2 的 DP8 一致，每个 rank 仍然
看到 2048/8 = 256 条序列。任何缩小 DP 的改动都会让 distributed optimizer 的 state 每卡翻倍
（DP 8->4 多约 8.5 GB），把激活上省下来的又吃回去 —— **CP=2 实测当场 OOM**，比 CP=1 更早死。

---

## 4.2 环境变量

`run_dapo.sh` 的开关全部走环境变量，**不需要改脚本内容**：

- `MODE`（默认 `bf16`）：`bf16` / `fp8` / `atomfp8` / `atombf16`，选 config + rollout 引擎与精度。
- `TRAIN_FP8`（默认 `0`）：`1` = 训练侧 Lumen FP8 blockwise2d，自动带 `FP8_PARAM_MANAGER=0`。
- `STEPS`（默认 `1000`）：覆盖 `num_training_steps`。
- `CONFIG_OVERRIDE`（默认按 `MODE` 推导）：直接指定 config，**跑 smoke 必须用它**。
- `EXTRA_OVERRIDE`（默认空）：追加任意 Hydra 覆盖，空格分隔。
- `MODEL_PATH` / `TRAIN_FILE` / `VAL_FILE`：换模型或数据，默认走 `$DATA_ROOT` 标准布局。
- `LOG`：日志路径，默认 `$DATA_ROOT/logs/$RUN_ID.log`，同时写进 `/tmp/run_dapo_log.txt`。
- `ATOMRL_FP32_MOE_ROUTER`（默认 `1`）：**例子 6、7 必须显式给**，见下。
- `PYTORCH_CUDA_ALLOC_CONF`：**启动时置空**，ROCm/HIP allocator 不支持 `expandable_segments`。

> **脚本是唯一来源，被误改就还原它**：`git -C "$RL_ROOT/ATOM/atomrl" checkout -- examples/DAPO/run_dapo.sh`。

所有命令统一带这段前缀。`export VAR=` 是"设为空值"，脚本据此把它 `unset`：

```bash
S=$RL_ROOT/ATOM/atomrl/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' PYTORCH_CUDA_ALLOC_CONF=;"
```

> `run_dapo.sh` 开头有 `: "${RL_ROOT:?}"`，容器内 `RL_ROOT` 为空会直接退出。
> `ENVX` 就是为防这个坑——detached exec 不要依赖起容器时的 `-e` 注入。

---

## 4.3 例子 1-5：Qwen3-8B-Base

先跑 smoke（前台等结果）。**smoke 必须用 `CONFIG_OVERRIDE` 指到 `*_smoke.yaml`**，
只设 `STEPS=1` 会继续用长跑 config（resp=20480、batch=512），那不是 smoke：

```bash
# 例子 1
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  STEPS=1 MODE=bf16 LOG=$DATA_ROOT/logs/smoke-bf16.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# 例子 2（TRAIN_FP8=0，只验 rollout fp8_per_block）
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml \
  STEPS=1 MODE=fp8 LOG=$DATA_ROOT/logs/smoke-fp8.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# 例子 4（4k 配置；先做完预编译）
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml \
  STEPS=1 MODE=atomfp8 LOG=$DATA_ROOT/logs/smoke-atomfp8.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# 例子 5（同样先做完预编译；不需要 RMSNorm patch）
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml \
  STEPS=1 MODE=atombf16 LOG=$DATA_ROOT/logs/smoke-atombf16.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""
```

再起长跑（detached，防中断）：

```bash
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=bf16                 bash '$S'"  # 1
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8                  bash '$S'"  # 2
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8      TRAIN_FP8=1 bash '$S'"  # 3
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atomfp8  TRAIN_FP8=1 bash '$S'"  # 4
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atombf16             bash '$S'"  # 5
```

> 例子 5 不要传 `TRAIN_FP8=1`：`MODE=atombf16` 会把 `LUMEN_FP8` / `FP8_PARAM_MANAGER` /
> `LUMEN_FP8_SCALING` 等一并 unset，训练侧无条件是 BF16。它也**不导入 Lumen/AITER 的 norm
> patch**（HF Qwen3 的 RMSNorm 本身就是 model-sensitive），这样才和例子 1 严格可比。

> 建议先 `STEPS=30` 起一版确认显存/指标健康，再上 1000 步。
> W&B 可选：把 `WANDB_API_KEY=xxxx` 放进 `$RL_ROOT/wandb.key`，脚本自动读。
> 换 ckpt 落盘频率用 `EXTRA_OVERRIDE='checkpointing.save_steps=10 checkpointing.save_total_limit=2'`；
> 8B FSDP2 单个 checkpoint 约 90 GB，先 `df -h`。

确认已经在跑：

```bash
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); sleep 200
  grep -aE "setup .ray-controller. complete|filter_groups round|View run" "$L" | tail -3
  grep -aiE "Traceback|OutOfMemory|CUDA error" "$L" | tail'
```

---

## 4.4 例子 6：MoE + FSDP2

```bash
ENVX_MOE="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
ATOMRL_FP32_MOE_ROUTER=0 PYTORCH_CUDA_ALLOC_CONF=;"

# smoke（4k，3 步，约 10 分钟，其中 5 分钟是 8 个 actor 各自加载 57GB 模型）
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-4k-smoke.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""

# 长跑（先看磁盘：至少 400G 可用）
df -h "$DATA_ROOT"
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe.log bash '$S'"
```

> **`MODEL_PATH` 必须显式给** —— `run_dapo.sh` 的默认值是 8B。

> **`ATOMRL_FP32_MOE_ROUTER=0` 必须给。** 框架默认是 fp32，而这条线要求 router 走 BF16：
> FSDP2 和 vLLM 跑的是**同一个 PyTorch router op、同一种布局**，BF16 舍入会让两边落到同一组
> top-8 专家，两侧对齐比单侧提精度更重要。日志里应看到
> `[atomrl] MoE router patched on 48 gates (fp32=False)`，`True` 说明忘了传。

> **`SCRATCH_ROOT` 必须导出**：config 用 `${oc.env:SCRATCH_ROOT}` 解析
> `model_name` / `checkpoint_dir`，解析不到 omegaconf 直接退出。**即使关掉落盘也要给。**

**新机器第一次跑 MoE，先做一次权重同步的端到端确认。** transformers 5.x 的融合专家张量
（约 57 GB、**93% 的参数**）一旦匹配不上 vLLM 的 `expert_params_mapping`，会走 vLLM 的静默
`continue` 分支：不报错、不加载，rollout 引擎的专家永远停在磁盘加载值。覆盖率断言
（`ATOMRL_WEIGHT_SYNC_CHECK=error`）默认开着，再加一次逐位比对更稳：

```bash
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE ATOMRL_WEIGHT_SYNC_VERIFY=1 \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-verify.log bash '$S'"
```

> 判据是**没有异常**：exit 0 就说明 96 个融合张量 x 8 replica x 3 次同步全部逐位一致。
> 失败会抛 `weight sync verify failed for ... shard w1/w3/w2` 或
> `weight sync (colocate-ipc) left N/M rollout parameters untouched: ...`。

顺带跑一遍纯 CPU 单测，确认代码完整：

```bash
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT/ATOM/atomrl" &&
  python3 -m pytest atomrl/tests/atomrl/test_moe_weight_sync.py &&      # 11 项，融合专家同步
  python3 -m atomrl.tests.test_rollout_routing &&      #  9 项
  python3 -m atomrl.tests.test_dataproto_ragged &&     # 10 项
  python3 -m atomrl.tests.test_mismatch_metrics'       #  4 项
```

---

## 4.5 例子 7：MoE + Megatron EP=8

```bash
# smoke：config 的 moe_router_dtype 是 null，所以这里是 =0
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-4k-smoke-megatron.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""

# 长跑：config 的 moe_router_dtype 是 fp32，所以这里必须翻成 =1
ENVX_MEGA="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
ATOMRL_FP32_MOE_ROUTER=1 PYTORCH_CUDA_ALLOC_CONF=;"

df -h "$DATA_ROOT"     # Megatron dist-checkpoint 约 400GB
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX_MEGA \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe-megatron.log bash '$S'"
```

> **`ATOMRL_FP32_MOE_ROUTER` 只作用于 vLLM worker**，训练侧 Megatron 读的是
> `megatron_cfg.moe_router_dtype`。**两处必须一起翻。**

**为什么这条线的 router 是 fp32，而例子 6 是 BF16**：Megatron 走的是它自己的 `TopKRouter`
喂 grouped-GEMM，和 vLLM 不是同一个实现，BF16 下两者会在"近乎平票"的 token 上选出不同专家，
而翻一个专家会让那个 token 的 log-prob 变很多。实测 `moe_router_dtype: null` 时
`rollout_corr/kl` 到 step 77 一直平在 6.5e-4，然后每步约 +16% 爬到 step 110 的 2.4e-2；
换成 fp32 之后到 step 185 都还是 7e-4。

长跑 config 的 `save_steps: 5` 很激进（9.3 min/步 -> 约 46 分钟一次 400GB 落盘），
按容错需求调大：`EXTRA_OVERRIDE='checkpointing.save_steps=20'`。

启动后先确认三件事再放手：`MoE+EP spec ... EP=8 ... router_dtype=fp32`、
无 `Traceback` / `HSA_STATUS`、首步在约 14 分钟内出 `callbacks: step=1`。

### Checkpoint 验证（例子 7 / Megatron）

完整的 Megatron distributed checkpoint 必须包含：
- 8 个 model shard（`model_world_size_8_rank_*.pt`）
- 8 个 optimizer metadata shard（`optim_world_size_8_rank_*.pt`）
- 8 个 extra-state shard（`extra_state_world_size_8_rank_*.pt`）
- 8 个 optimizer parameter-state shard（`optim_parameter_state_world_size_8_rank_*.pt`，每个约 41-45 GiB）

首次保存后验证：

```bash
# 填入启动日志中的 RUN_ID
RUN_ID=<your-run-id>
P="$DATA_ROOT/ckpts/$RUN_ID/global_step_5/actor"
ls -lh "$P"/model_world_size_8_rank_*.pt
ls -lh "$P"/optim_world_size_8_rank_*.pt
ls -lh "$P"/optim_parameter_state_world_size_8_rank_*.pt
ls -lh "$P"/extra_state_world_size_8_rank_*.pt
```

自动数量检查：

```bash
sudo docker exec -e RUN_ID="$RUN_ID" "$CONTAINER" bash -lc '
python3 - <<PY
import os
from pathlib import Path

p = Path(os.environ["DATA_ROOT"]) / "ckpts" / os.environ["RUN_ID"] / "global_step_5" / "actor"
for pattern in (
    "model_world_size_8_rank_*.pt",
    "optim_world_size_8_rank_*.pt",
    "optim_parameter_state_world_size_8_rank_*.pt",
    "extra_state_world_size_8_rank_*.pt",
):
    files = list(p.glob(pattern))
    print(pattern, len(files), sum(x.stat().st_size for x in files))
    assert len(files) == 8
print("checkpoint verification passed")
PY
'
```

> **严禁从不完整的 checkpoint 恢复。** 缺少大体积 `optim_parameter_state_*` shard 的
> checkpoint 加载不报错，但下一步 optimizer step 会产生 NaN —— FP32 master weights 和
> Adam moments 缺失。详见[多节点 RDMA](05-multinode-rdma_cn.md#54-checkpoint-损坏记录)。

---

## 4.6 关掉 checkpoint 落盘（磁盘不够时）

```bash
EXTRA_OVERRIDE='checkpointing.save_steps=1000000 checkpointing.resume=false'
```

> **不要写 `checkpointing.checkpoint_dir=`**。Hydra 会把空值解析成 `None`，
> omegaconf 立刻报 `Incompatible value 'None' for field of type 'str'` 并退出。
> 用一个跑不到的大 `save_steps` 才是干净做法。崩了只能从头跑，先想清楚。

---

## 4.7 健康判据

**Smoke 通过的硬判据**：exit 0、无 `Traceback`、无 `HSA_STATUS`、日志里有
`RLTrainer.setup (ray-controller) complete: ... actor_workers=8`。

各例子的 `rollout_corr/kl` / 显存 / step 时间（`resp=20480`）/ checkpoint 大小：

- **例子 1**：kl ~0.001，`mem/actor_allocated_gb` 11.6 GB，4-5 min/步，ckpt ~90 GB。
  `grad_norm` ~0.85，`ppo_kl` ~0。
- **例子 2、3**：kl **~0.003-0.004**（FP8 gap，正常；逼近 TIS 阈值 2.0 才警惕），
  显存与 step 时间同例子 1，ckpt ~90 GB。
- **例子 4**：kl ~0.004（比 vLLM FP8 略高），显存同上；no-eager level=3 主要加速 rollout，
  但 sleep/wake + 权重同步会增加固定开销。ckpt ~90 GB。
- **例子 5**：kl 应落在例子 1 的量级（~0.001）而不是例子 4 的 ~0.004 —— 量化关掉了，
  剩下的只是 ATOM 与训练侧的实现差异。**这条就是判断 ATOM 对齐是否正确的判据**：
  如果 ATOM BF16 的 kl 也在 0.004 量级，说明差异不来自 FP8，去查[排障](06-troubleshooting_cn.md)的 ATOM RMSNorm 对齐。
  显存、step 时间、ckpt 与例子 4 同量级。
- **例子 6**：kl ~1.5e-3，`mem/actor_max_reserved_gb` 75-115 GB，~11 min/步，ckpt **~342 GB**。
  第 1 步的 `lr` 就是 `9.99998e-07`（满值），说明 warmup 确实是 0；看到 `2e-07` 说明用错了 config。
- **例子 7**：kl ~1.5e-3（健康区间 6e-4 ~ 1.8e-3），allocated 72 GB（4k）/ 130 GB（20k）、
  `max_reserved` 128-140 GB，~9.3 min/步（首步约 14 min，含 vLLM 加载），ckpt **~400 GB**。
  日志应有 `MoE+EP spec: num_experts=128 topk=8 moe_ffn=768 | tp=1 pp=1 cp=1 EP=8 etp=1
  -> local_experts/rank=16 | grouped_gemm=True router_dtype=fp32 pre_softmax=False`。

通用：`timing/weight_sync_s` 1.1-1.7 s 且**不随步数增长**；`mem/actor_allocated_gb` 恒定
（`max_reserved` 随每步 batch 波动是正常的，**存活内存在动才是泄漏**）。

**最重要的一条：`rollout_corr/kl` 不随步数单调爬升。** 它降下去是正常的（策略收敛变确定，
log 空间分歧自然缩小）；爬上去按概率排有三种原因：MoE router 精度两侧不匹配、
权重同步漏参数（用 `ATOMRL_WEIGHT_SYNC_VERIFY=1` 复查）、或者出现了新的对齐类 bug。

**长度崩塌看 `seq/max_len`。** 它在预算上限附近波动是健康的（说明每步都有序列打满），
单调往下走就是崩了。

### 实测参考曲线

**例子 6**（101 步 / 21.6 小时）：`reward/accuracy` 0.136 -> 0.494（step 50）-> **0.581**。
AIME-2024 在线验证（每 10 步，greedy）`val-core/acc/mean@1` 从 step 10 的 0.041 涨到
step 90 的 **0.361**，`val/response_length_mean` 从 2407 涨到 10389 ——模型学会想更久，
这就是这条线跑通的证据。

**例子 7**（`verlref_4k_longrun` 压缩配方，91 步）：`reward/accuracy` 0.168 -> 0.42、
`seq/mean_response_len` 773 -> 925、`rollout_corr/kl` 0.00136 -> 0.00060（在降，正确）、
AIME `mean@1` 0.086 -> 0.199。

> **已知的熵坍缩不是 bug**：例子 6、7 的 config 就是 `entropy_coeff=0`，所以 entropy
> 单调下降（101 步 0.844 -> 0.094）是配置的必然结果。只有在"entropy 掉到 0.05 以下
> **且** 长度开始缩"同时出现时才需要警惕。要治得先加 `entropy_coeff`。

---

## 4.8 监控 / 停止 / 续跑

```bash
# 监控
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt)
  grep -aE "callbacks: step=" "$L" | tail -5
  grep -aiE "Traceback|OutOfMemory|CUDA error|HSA_STATUS" "$L" | tail'

# 停止（连 Ray actor 一起清）
sudo docker exec "$CONTAINER" bash -lc '
  ray stop --force 2>/dev/null
  pkill -9 -f "[l]umenrl.trainer.main"; pkill -9 -f "[V]LLMRayServer"; pkill -9 -f "[E]ngineCore"
  sleep 10; rocm-smi --showmeminfo vram | grep -i used | head -1'   # 应回到 ~298MB/卡
```

**续跑**：config 里 `resume: true`，重跑同一条长跑命令即从最近 checkpoint 恢复。
新机器目录为空时就是从 step 0 开始。
