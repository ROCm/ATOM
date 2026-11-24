> [Examples README](../README_cn.md) > 多节点 RDMA

# 5. 多节点 RDMA 验证

本文档覆盖**多节点 RDMA 权重传输部署**的专项测试与验证流程（例如双节点 Qwen3-30B-A3B，
节点 1 Megatron 训练 + 节点 2 vLLM rollout，通过 RoCE RDMA 连接）。

单节点配置（单台 8 卡机器上的例子 1-7）请参阅[启动](04-launching_cn.md)。

---

## 5.1 RDMA / 网络预检

在**两个节点**的容器内都执行。确认 RDMA 设备可见且配置正确。

### 设备存在性

```bash
docker exec \
  -e RDMA_HCA="$RDMA_HCA" \
  -e RDMA_IFACE="$RDMA_IFACE" \
  "$CONTAINER" bash -lc '
ls -l /dev/infiniband
test -e /dev/infiniband/uverbs0
test -d "/sys/class/infiniband/$RDMA_HCA"
ip -4 addr show dev "$RDMA_IFACE"
'
```

### NCCL/RCCL 环境变量

训练启动环境中必须设置：

```bash
export NCCL_SOCKET_IFNAME="$RDMA_IFACE"
export NCCL_IB_HCA="$RDMA_HCA"
export NCCL_IB_GID_INDEX="$RDMA_GID_INDEX"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_DMABUF_ENABLE=0
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=INFO
```

### 日志确认

RDMA 只有在训练日志中**同时出现**以下两行才算验证通过：

```text
NCCL INFO Using network IB
NCCL INFO ... via NET/IB/0/GDRDMA ...
```

如果只出现 socket/TCP 日志，则 RDMA 未生效——检查 HCA 名称、GID index、以及容器是否挂载了
`/dev/infiniband`。

---

## 5.2 启动验证

多节点训练启动后，检查日志中以下序列。

### 启动序列

这些行必须按顺序出现：

```text
Created 1 placement groups for pool 'rollout' ... node_ip=${ROLLOUT_NODE_IP}
Created 1 placement groups for pool 'actor' ... node_ip=${TRAIN_NODE_IP}
RDMA preflight: ...
RDMA weight group ready: ... world=9
NCCL INFO Using network IB
NCCL INFO ... NET/IB/0/GDRDMA ...
```

`world=9` 确认了 9-rank 进程组（1 个 Megatron sender + 8 个 vLLM TP receiver）。
`GDRDMA` 行确认 GPU Direct RDMA 已激活。

### 每步确认

每个训练步必须同时输出：

```text
RDMA weight sync committed:
callbacks: step=N
```

### 重点监控指标

```text
rollout_corr/kl
rollout_corr/rollout_is_eff_sample_size
actor/loss
actor/grad_norm
actor/entropy
weight_sync/gbps
timing/weight_sync_rdma_s
```

**任一核心指标出现 `nan` 必须立即停止，不得继续写下一个 checkpoint。**

---

## 5.3 Ray 集群验证

两个 Ray 节点都启动后，确认集群健康：

```bash
docker exec "$CONTAINER" ray status
```

必须显示：

```text
Active: 2 nodes
Total: 16 GPU
```

---

## 5.4 Checkpoint 验证（Megatron distributed）

Qwen3-30B-A3B 的完整 Megatron distributed-optimizer checkpoint 必须包含：

- 8 个 model shard
- 8 个 optimizer metadata shard
- 8 个 extra-state shard
- 8 个大体积 optimizer parameter-state shard（每个约 41-45 GiB）
- controller 侧存在 `checkpoint_N.pt` 与 `latest_checkpointed_iteration.txt`

### 文件清单

在 trainer 节点，step 5 保存后检查：

```bash
RUN_ID=<your-run-id>
P="$RUNTIME_HOST_DIR/ckpts/$RUN_ID/global_step_5/actor"
ls -lh "$P"/model_world_size_8_rank_*.pt
ls -lh "$P"/optim_world_size_8_rank_*.pt
ls -lh "$P"/optim_parameter_state_world_size_8_rank_*.pt
ls -lh "$P"/extra_state_world_size_8_rank_*.pt
```

### 自动数量检查

```bash
docker exec -i -e RUN_ID="$RUN_ID" "$CONTAINER" python3 - <<'PY'
import os
from pathlib import Path

p = Path("/runtime/ckpts") / os.environ["RUN_ID"] / "global_step_5" / "actor"
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
```

### Checkpoint 损坏记录

历史 v1 任务在保存 checkpoint 时磁盘写满。旧逻辑只保存了约 2 KiB 的
`optimizer.state_dict()` metadata，没有 Megatron distributed optimizer 的 FP32 master 和 Adam
moments；这种 checkpoint 加载模型后继续更新会出现 NaN。

当前可恢复 checkpoint 必须同时覆盖：

```text
optimizer.state_dict()
optimizer.save_parameter_state(...)
optimizer.load_parameter_state(...)
optimizer.reload_model_params()
```

**严禁把只有小型 optimizer metadata 文件的 checkpoint 视为可续训 checkpoint。**

---

## 5.5 已验证基线

以下数值来自已验证的 RDMA smoke 跑通记录，仅用于判断新部署是否明显退化，不能作为配置来源。

**RDMA smoke（连续 3 步）：**

| 指标 | 数值 |
|---|---|
| 每步广播 | 61.1 GB，58 buckets |
| 同步耗时 | 2.51-3.90 秒 |
| 有效吞吐 | 134-215 Gb/s |
| `rollout_corr/kl` | 0.0008425、0.0008446、0.0007718 |
| ESS | 0.998361、0.998328、0.998397 |
| 动态权重覆盖 | 每个 TP worker 均已校验 |

**基线 checkpoint 大小：** ~402 GiB（8 model + 8 optim-metadata + 8 extra-state
+ 8 optim-parameter-state shard，每个约 41-45 GiB）。
