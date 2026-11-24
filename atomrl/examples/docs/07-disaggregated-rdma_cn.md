> [Examples README](../README_cn.md) > 训推分离双节点 RDMA

# 7. 训推分离双节点 Megatron + vLLM RDMA 部署

在两台 8 卡 AMD GPU 机器上从零部署 **Qwen3-30B-A3B** MoE RL 训练：节点 1 运行 Megatron
训练，节点 2 运行 vLLM rollout，通过 RCCL/RoCE GPU Direct RDMA 同步权重。

> 本文档覆盖的部署模式与例子 1-7（单节点 co-located）完全不同。单节点配置请参阅
> [启动](04-launching_cn.md)。适用于所有多节点 RDMA 部署的验证流程请参阅
> [多节点 RDMA](05-multinode-rdma_cn.md)。

**一句话流程**：收集路径 -> 自动发现网络 -> 拉代码 -> 构建 Docker 镜像 -> 起容器 ->
装依赖（如需要）-> 下载模型/数据 -> RDMA 预检 -> 启动 Ray 集群 -> 生成配置 ->
smoke（3 步）-> longrun（200 步）。

---

## 7.1 架构概述

| 角色 | 节点 | GPU | 关键软件 |
|---|---|---|---|
| Rollout / Ray head | `${ROLLOUT_NODE_IP}` | 8x MI308X (gfx942) | vLLM TP=2 x 4 replicas |
| Megatron 训练 | `${TRAIN_NODE_IP}` | 8x MI308X (gfx942) | Megatron TP=4, EP=8, ETP=1 |

权重同步使用独立的 9-rank `torch.distributed` 进程组：

- rank 0：Megatron sender
- rank 1-8：vLLM TP receivers
- ROCm 的 `backend="nccl"` 运行时自动映射为 RCCL
- 传输：RoCE RDMA，使用自动探测到的 `${RDMA_HCA}` / `${RDMA_IFACE}` / `${RDMA_GID_INDEX}`
- 日志必须出现 `Using network IB` 和 `NET/IB/0/GDRDMA`

训练与 rollout：

- 训练权重：BF16 compute + Megatron distributed optimizer 的 FP32 master/Adam state
- Rollout 权重：BF16；KV cache `auto`（当前稳定 BF16 baseline）
- R3：vLLM 记录 top-k expert IDs，Megatron `RouterReplay` 执行 hard assignment replay
- 算法：GRPO/DAPO，32 prompts x 8 generations，全局 batch 256
- 目标：200 步

本流程不使用 ATOM rollout、ZMQ CUDA-IPC 或跨节点 safetensors 作为主路径。可选的 ATOM 替代方案
见[附录 A](#附录-a-atom-rollout)。

---

## 7.2 宿主机前置条件

在**两台主机**上执行。起点是已安装 AMD GPU/RDMA 内核驱动和 Docker 的机器。

```bash
for cmd in docker git ip python3 ping; do
  command -v "$cmd" >/dev/null || { echo "missing: $cmd"; exit 1; }
done
docker info >/dev/null
test -e /dev/kfd
test -d /dev/dri
test -d /dev/infiniband
test -d /sys/class/infiniband
rocminfo | grep -c 'Name:.*gfx942'
```

最后一条应发现 8 个 GPU agent。数量不是 8 时先修复主机驱动/设备权限，不要进入容器部署。

---

## 7.3 路径变量与环境文件

### 7.3.1 两台主机分别填写路径

以下变量为宿主机路径。两台机器分别填写，容器内挂载点保持一致。

```bash
read -r -p "代码根目录 WORK_ROOT: " WORK_ROOT
read -r -p "模型根目录 MODEL_HOST_DIR（其下放 Qwen3-30B-A3B）: " MODEL_HOST_DIR
read -r -p "数据根目录 DATASET_HOST_DIR: " DATASET_HOST_DIR
read -r -p "日志/checkpoint 目录 RUNTIME_HOST_DIR: " RUNTIME_HOST_DIR
read -r -p "共享目录 SHARED_HOST_DIR（没有则填本地空目录）: " SHARED_HOST_DIR
read -r -p "rollout 镜像 [rocm/atom-dev:vllm-latest]: " ROLLOUT_IMAGE
read -r -p "trainer 镜像 [rocm/atom-dev:latest]: " TRAIN_IMAGE
read -r -p "W&B project [AtomRL]: " WANDB_PROJECT
read -r -p "W&B entity（留空使用默认账号）: " WANDB_ENTITY

export WORK_ROOT MODEL_HOST_DIR DATASET_HOST_DIR RUNTIME_HOST_DIR
export SHARED_HOST_DIR
export ATOMRL_HOST_DIR="${WORK_ROOT}/ATOM/atomrl"
export LUMEN_HOST_DIR="${WORK_ROOT}/ATOM/atomrl/lumen"
export MEGATRON_HOST_DIR="${WORK_ROOT}/Megatron-LM"
export CONTAINER="${CONTAINER:-qwen3-30b-rl}"
export ROLLOUT_IMAGE="${ROLLOUT_IMAGE:-rocm/atom-dev:vllm-latest}"
export TRAIN_IMAGE="${TRAIN_IMAGE:-rocm/atom-dev:latest}"
export WANDB_PROJECT="${WANDB_PROJECT:-AtomRL}"
export WANDB_ENTITY

for p in "$WORK_ROOT" "$MODEL_HOST_DIR" "$DATASET_HOST_DIR" \
         "$RUNTIME_HOST_DIR" "$SHARED_HOST_DIR"; do
  test -n "$p" || { echo "路径不能为空"; exit 1; }
  mkdir -p "$p"
done
mkdir -p "$RUNTIME_HOST_DIR/logs" "$RUNTIME_HOST_DIR/ckpts"
```

### 7.3.2 保存环境文件

将每台主机的值保存到仅当前用户可读的环境文件。后续命令先
`source "$HOME/qwen3-rdma-node.env"`。

```bash
ENV_FILE="${HOME}/qwen3-rdma-node.env"
umask 077
cat > "$ENV_FILE" <<EOF
export WORK_ROOT='$WORK_ROOT'
export MODEL_HOST_DIR='$MODEL_HOST_DIR'
export DATASET_HOST_DIR='$DATASET_HOST_DIR'
export RUNTIME_HOST_DIR='$RUNTIME_HOST_DIR'
export SHARED_HOST_DIR='$SHARED_HOST_DIR'
export ATOMRL_HOST_DIR='$ATOMRL_HOST_DIR'
export LUMEN_HOST_DIR='$LUMEN_HOST_DIR'
export MEGATRON_HOST_DIR='$MEGATRON_HOST_DIR'
export CONTAINER='$CONTAINER'
export ROLLOUT_IMAGE='$ROLLOUT_IMAGE'
export TRAIN_IMAGE='$TRAIN_IMAGE'
export WANDB_PROJECT='$WANDB_PROJECT'
export WANDB_ENTITY='$WANDB_ENTITY'
EOF
echo "saved $ENV_FILE"
```

### 7.3.3 容器内挂载点

| 宿主机路径 | 容器内路径 | 用途 |
|---|---|---|
| `$ATOMRL_HOST_DIR` | `/workspace/ATOM/atomrl` | AtomRL 源码 |
| `$LUMEN_HOST_DIR` | `/workspace/Lumen` | Lumen 依赖源码 |
| `$MEGATRON_HOST_DIR` | `/workspace/Megatron-LM` | ROCm Megatron-LM（megatron.training + RouterReplay）|
| `$AITER_HOST_DIR` | `/workspace/aiter` | aiter GPU kernel 源码 |
| `$FA_HOST_DIR` | `/workspace/flash-attention` | ROCm flash-attention 源码 |
| `$MODEL_HOST_DIR` | `/root/models` | 模型权重 |
| `$DATASET_HOST_DIR` | `/root/data_cached` | 已过滤数据集 |
| `$RUNTIME_HOST_DIR` | `/runtime` | 日志与 checkpoint |
| `$SHARED_HOST_DIR` | `/shared` | 可选 shared folder fallback |
| (设备) | `/dev/infiniband` | RoCE verbs 设备 |

---

## 7.4 自动发现网络

在**每台主机上分别执行**。不要从示例、旧日志或其他集群复制 IP。

### 7.4.1 自动发现 Ray node IP

```bash
AUTO_RAY_IP=$(
  ip -4 route get "${RAY_PROBE_TARGET:-1.1.1.1}" |
    awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}'
)
if [ -z "$AUTO_RAY_IP" ]; then
  AUTO_RAY_IP=$(ip -o -4 addr show scope global | awk 'NR==1 {split($4,a,"/"); print a[1]}')
fi

echo "自动选择: $AUTO_RAY_IP"
echo "全部候选:"
ip -o -4 addr show scope global |
  awk '{split($4,a,"/"); printf "  iface=%-16s ip=%s\n",$2,a[1]}'

read -r -p "确认 Ray node IP [${AUTO_RAY_IP}]: " NODE_RAY_IP
export NODE_RAY_IP="${NODE_RAY_IP:-$AUTO_RAY_IP}"
test -n "$NODE_RAY_IP"
```

在 rollout 节点记录：`export ROLLOUT_NODE_IP="$NODE_RAY_IP"`

在 trainer 节点记录：`export TRAIN_NODE_IP="$NODE_RAY_IP"`

交换两个节点的自动探测结果后，在两台节点输入：

```bash
read -r -p "rollout 节点探测到的 Ray IP: " ROLLOUT_NODE_IP
read -r -p "trainer 节点探测到的 Ray IP: " TRAIN_NODE_IP
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export ROLLOUT_NODE_IP='$ROLLOUT_NODE_IP'
export TRAIN_NODE_IP='$TRAIN_NODE_IP'
EOF
```

验证：

```bash
source "$HOME/qwen3-rdma-node.env"
test "$ROLLOUT_NODE_IP" != "$TRAIN_NODE_IP"
ping -c 2 "$ROLLOUT_NODE_IP"
ping -c 2 "$TRAIN_NODE_IP"
```

若默认路由选出的地址不能跨节点访问，应从"全部候选"中选择实际互通的地址。

### 7.4.2 自动发现 RoCE HCA、网卡和 GID

以下脚本从 sysfs 枚举 active RDMA port，优先选择带 IPv4 地址的 RoCE v2 GID。两台主机分别执行：

```bash
eval "$(
python3 - <<'PY'
import json
import pathlib
import subprocess
import sys

root = pathlib.Path("/sys/class/infiniband")
addrs = json.loads(subprocess.check_output(["ip", "-j", "-4", "addr", "show"]))
ipv4 = {
    item["ifname"]: [
        a["local"] for a in item.get("addr_info", [])
        if a.get("family") == "inet" and a.get("scope") == "global"
    ]
    for item in addrs
}

candidates = []
for hca_dir in sorted(root.glob("*")):
    for port_dir in sorted((hca_dir / "ports").glob("*")):
        if (port_dir / "state").read_text().strip().split(":", 1)[0] != "4":
            continue
        gids = port_dir / "gids"
        for gid_file in sorted(gids.glob("*"), key=lambda p: int(p.name)):
            idx = gid_file.name
            ndev_file = port_dir / "gid_attrs" / "ndevs" / idx
            type_file = port_dir / "gid_attrs" / "types" / idx
            if not ndev_file.exists() or not type_file.exists():
                continue
            iface = ndev_file.read_text().strip()
            gid_type = type_file.read_text().strip()
            ips = ipv4.get(iface, [])
            if not iface or not ips or "v2" not in gid_type.lower():
                continue
            gid = gid_file.read_text().strip()
            candidates.append((hca_dir.name, port_dir.name, int(idx), iface, ips[0], gid_type, gid))

if not candidates:
    raise SystemExit("没有找到 active RoCE v2 + IPv4 candidate")

for item in candidates:
    print(
        "candidate:",
        f"hca={item[0]} port={item[1]} gid_index={item[2]}",
        f"iface={item[3]} ip={item[4]} type={item[5]} gid={item[6]}",
        file=sys.stderr,
    )

hca, port, gid_index, iface, ip, gid_type, gid = candidates[0]
print(f"export RDMA_HCA={hca!r}")
print(f"export RDMA_PORT={port!r}")
print(f"export RDMA_GID_INDEX={gid_index!r}")
print(f"export RDMA_IFACE={iface!r}")
print(f"export RDMA_IP={ip!r}")
print(f"export RDMA_GID={gid!r}")
print(f"export RDMA_GID_TYPE={gid_type!r}")
PY
)"

printf 'HCA=%s port=%s iface=%s IP=%s GID_INDEX=%s type=%s GID=%s\n' \
  "$RDMA_HCA" "$RDMA_PORT" "$RDMA_IFACE" "$RDMA_IP" \
  "$RDMA_GID_INDEX" "$RDMA_GID_TYPE" "$RDMA_GID"
```

如果脚本选择的 RoCE 网络不是两节点互联网络，设置 `RDMA_IFACE` 为正确候选后重新选择。

在 rollout 节点记录：`export ROLLOUT_RDMA_IP="$RDMA_IP"`

在 trainer 节点记录：`export TRAIN_RDMA_IP="$RDMA_IP"`

两台节点当前要求 HCA 名、网卡名和 GID index 一致。交换自动探测结果后，在两台节点输入：

```bash
read -r -p "rollout 节点探测到的 RoCE IP: " ROLLOUT_RDMA_IP
read -r -p "trainer 节点探测到的 RoCE IP: " TRAIN_RDMA_IP
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export RDMA_HCA='$RDMA_HCA'
export RDMA_PORT='$RDMA_PORT'
export RDMA_IFACE='$RDMA_IFACE'
export RDMA_GID_INDEX='$RDMA_GID_INDEX'
export ROLLOUT_RDMA_IP='$ROLLOUT_RDMA_IP'
export TRAIN_RDMA_IP='$TRAIN_RDMA_IP'
EOF
```

验证 RoCE 专网：

```bash
source "$HOME/qwen3-rdma-node.env"
ip -4 addr show dev "$RDMA_IFACE"

# rollout 节点
ping -I "$RDMA_IFACE" -c 3 "$TRAIN_RDMA_IP"

# trainer 节点
ping -I "$RDMA_IFACE" -c 3 "$ROLLOUT_RDMA_IP"
```

最终角色表（由本次探测结果填写）：

| 角色 | Ray node IP | RoCE IP | GPU | 容器镜像 |
|---|---|---|---|---|
| Rollout / Ray head | `${ROLLOUT_NODE_IP}` | `${ROLLOUT_RDMA_IP}` | 8 | `${ROLLOUT_IMAGE}` |
| Megatron 训练 | `${TRAIN_NODE_IP}` | `${TRAIN_RDMA_IP}` | 8 | `${TRAIN_IMAGE}` |

---

## 7.5 源码

所有组件从源码安装。两台节点执行相同的克隆操作。

| 组件 | 仓库 | 分支 |
|---|---|---|
| AtomRL | `https://github.com/ROCm/ATOM.git` | `main` |
| Lumen | `${WORK_ROOT}/ATOM/atomrl/lumen` | 内置 |
| Megatron-LM | `https://github.com/ROCm/Megatron-LM.git` | `rocm_dev` |
| aiter | `https://github.com/ROCm/aiter.git` | `adbb6ac8b841` |
| flash-attention | `https://github.com/ROCm/flash-attention.git` | `main` |

```bash
source "$HOME/qwen3-rdma-node.env"

# AtomRL
if [ ! -d "${WORK_ROOT}/ATOM/.git" ]; then
  git clone -b main https://github.com/ROCm/ATOM.git "${WORK_ROOT}/ATOM"
fi
cd "${WORK_ROOT}/ATOM"
git fetch origin main
git switch main 2>/dev/null \
  || git switch -c main --track origin/main
git merge --ff-only origin/main

# Lumen 已导入 AtomRL 源码树。
LUMEN_HOST_DIR="${WORK_ROOT}/ATOM/atomrl/lumen"
test -d "$LUMEN_HOST_DIR/lumen"

# Megatron-LM（ROCm fork，含 megatron.training 和 RouterReplay）
if [ ! -d "$MEGATRON_HOST_DIR/.git" ]; then
  git clone https://github.com/ROCm/Megatron-LM.git "$MEGATRON_HOST_DIR"
fi
cd "$MEGATRON_HOST_DIR"
git fetch origin rocm_dev
git switch rocm_dev 2>/dev/null \
  || git switch -c rocm_dev --track origin/rocm_dev
git merge --ff-only origin/rocm_dev

# aiter
AITER_HOST_DIR="${WORK_ROOT}/aiter"
if [ ! -d "$AITER_HOST_DIR/.git" ]; then
  git clone https://github.com/ROCm/aiter.git "$AITER_HOST_DIR"
fi
cd "$AITER_HOST_DIR"
git fetch --depth 1 origin adbb6ac8b84110134e0cf4dc513b250d2cc0a75c
git checkout --detach FETCH_HEAD

# flash-attention
FA_HOST_DIR="${WORK_ROOT}/flash-attention"
if [ ! -d "$FA_HOST_DIR/.git" ]; then
  git clone https://github.com/ROCm/flash-attention.git "$FA_HOST_DIR"
fi

# 保存路径
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export AITER_HOST_DIR='$AITER_HOST_DIR'
export FA_HOST_DIR='$FA_HOST_DIR'
EOF
```

验证两台节点代码一致：

```bash
source "$HOME/qwen3-rdma-node.env"
echo "AtomRL:    $(cd "$ATOMRL_HOST_DIR" && git rev-parse HEAD)"
echo "Lumen:      $(cd "$LUMEN_HOST_DIR" && git rev-parse HEAD)"
echo "Megatron:   $(cd "$MEGATRON_HOST_DIR" && git rev-parse HEAD)"
echo "aiter:      $(cd "$AITER_HOST_DIR" && git rev-parse HEAD)"
echo "fa:         $(cd "$FA_HOST_DIR" && git rev-parse HEAD)"
```

两台节点的五行输出必须分别相同。

---

## 7.6 Docker 镜像

### 7.6.1 GRPO Dockerfile（优先）

如果 GRPO Dockerfile 存在，用它构建角色专用镜像：

```bash
source "$HOME/qwen3-rdma-node.env"
DOCKERFILE="$ATOMRL_HOST_DIR/examples/GRPO/docker/Dockerfile"
test -f "$DOCKERFILE" || {
  echo "GRPO Dockerfile 不存在，使用 §7.8 fallback 安装流程"
  export USE_GRPO_DOCKERFILE=0
}
```

| Target | 默认基础镜像 | 输出镜像 | 用途 |
|---|---|---|---|
| `rollout` | `rocm/atom-dev:vllm-latest` | `qwen3-30b-a3b:rollout` | Ray head + vLLM TP2 x 4 |
| `trainer` | `rocm/atom-dev:latest` | `qwen3-30b-a3b:trainer` | Megatron TP4/EP8 |

两个 target 不能合并成同一运行镜像。rollout 的 vLLM/NumPy/Triton build 与 trainer 不同；
强行统一会破坏已验证的组合。

Docker build context 必须是 **ATOM 仓库根目录**，不能是 `atomrl/examples/GRPO/`。

#### rollout 节点构建

只在 rollout 节点执行：

```bash
source "$HOME/qwen3-rdma-node.env"
ATOM_ROOT="$(cd "$ATOMRL_HOST_DIR/../.." && pwd)"
cd "$ATOM_ROOT"

docker build --network=host --progress=plain \
  -f atomrl/examples/GRPO/docker/Dockerfile \
  --target rollout \
  -t qwen3-30b-a3b:rollout \
  .

export ROLLOUT_IMAGE=qwen3-30b-a3b:rollout
export USE_GRPO_DOCKERFILE=1
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export ROLLOUT_IMAGE='$ROLLOUT_IMAGE'
export USE_GRPO_DOCKERFILE=1
EOF
```

#### trainer 节点构建

只在 trainer 节点执行：

```bash
source "$HOME/qwen3-rdma-node.env"
ATOM_ROOT="$(cd "$ATOMRL_HOST_DIR/../.." && pwd)"
cd "$ATOM_ROOT"

docker build --network=host --progress=plain \
  -f atomrl/examples/GRPO/docker/Dockerfile \
  --target trainer \
  -t qwen3-30b-a3b:trainer \
  .

export TRAIN_IMAGE=qwen3-30b-a3b:trainer
export USE_GRPO_DOCKERFILE=1
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export TRAIN_IMAGE='$TRAIN_IMAGE'
export USE_GRPO_DOCKERFILE=1
EOF
```

首次构建会编译 gfx942 的 aiter、flash-attention 和 Lumen HIP extension，耗时较长。
后续相同源码和 build args 会复用 Docker layer cache。

固定基础镜像 digest 和源码 ref 以确保可复现：

```bash
docker build --network=host --progress=plain \
  -f atomrl/examples/GRPO/docker/Dockerfile \
  --target rollout \
  --build-arg ROLLOUT_BASE_IMAGE='rocm/atom-dev@sha256:<digest>' \
  --build-arg MEGATRON_REF='<commit>' \
  --build-arg AITER_SHA='<commit>' \
  --build-arg FLASH_ATTN_REF='<commit>' \
  -t qwen3-30b-a3b:rollout \
  .
```

核对构建输出：

```bash
docker image inspect qwen3-30b-a3b:rollout \
  --format 'rollout {{.Id}} {{index .Config.Labels "org.opencontainers.image.title"}}' \
  2>/dev/null || true
docker image inspect qwen3-30b-a3b:trainer \
  --format 'trainer {{.Id}} {{index .Config.Labels "org.opencontainers.image.title"}}' \
  2>/dev/null || true
```

如果只在一台 build 主机构建，可推送到内部 registry 或使用 `docker save` / `docker load`。
两节点必须使用各自角色的 target 镜像。

### 7.6.2 已验证软件版本

| 包 | 版本 |
|---|---|
| Python | `3.12.3` |
| PyTorch | `2.10.0+rocm7.2.4.git3d3aa833` |
| HIP | `7.2.53211` |
| Ray | `2.56.1` |
| Megatron-LM (ROCm fork) | `rocm_dev` 分支 |
| flash-attn | `2.8.4` |
| amd-aiter | `0.1.0` |
| Transformers | `5.2.0` |
| Datasets | `5.0.0` |
| Accelerate | `1.14.0` |
| Safetensors | `0.8.0` |
| OmegaConf | `2.3.1` |
| math-verify | `0.3.3` |
| W&B | `0.28.1` |

Rollout 专有：vLLM `0.22.1.dev0+g0b3ba88f1.d20260629.rocm724`。

Trainer 专有：vLLM **未安装**（预期状态）。

---

## 7.7 启动容器

### 7.7.1 rollout 节点

```bash
source "$HOME/qwen3-rdma-node.env"
SOURCE_MOUNTS=()
if [ "${USE_GRPO_DOCKERFILE:-0}" != "1" ]; then
  SOURCE_MOUNTS=(
    -v "$ATOMRL_HOST_DIR":/workspace/ATOM/atomrl
    -v "$LUMEN_HOST_DIR":/workspace/Lumen
    -v "$MEGATRON_HOST_DIR":/workspace/Megatron-LM
    -v "$AITER_HOST_DIR":/workspace/aiter
    -v "$FA_HOST_DIR":/workspace/flash-attention
  )
fi
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add=video \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  "${SOURCE_MOUNTS[@]}" \
  -v "$DATASET_HOST_DIR":/root/data_cached \
  -v "$MODEL_HOST_DIR":/root/models \
  -v "$SHARED_HOST_DIR":/shared \
  -v "$RUNTIME_HOST_DIR":/runtime \
  "$ROLLOUT_IMAGE" -lc 'sleep infinity'
```

### 7.7.2 trainer 节点

```bash
source "$HOME/qwen3-rdma-node.env"
SOURCE_MOUNTS=()
if [ "${USE_GRPO_DOCKERFILE:-0}" != "1" ]; then
  SOURCE_MOUNTS=(
    -v "$ATOMRL_HOST_DIR":/workspace/ATOM/atomrl
    -v "$LUMEN_HOST_DIR":/workspace/Lumen
    -v "$MEGATRON_HOST_DIR":/workspace/Megatron-LM
    -v "$AITER_HOST_DIR":/workspace/aiter
    -v "$FA_HOST_DIR":/workspace/flash-attention
  )
fi
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add=video \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  "${SOURCE_MOUNTS[@]}" \
  -v "$DATASET_HOST_DIR":/root/data_cached \
  -v "$MODEL_HOST_DIR":/root/models \
  -v "$SHARED_HOST_DIR":/shared \
  -v "$RUNTIME_HOST_DIR":/runtime \
  "$TRAIN_IMAGE" -lc 'sleep infinity'
```

### 7.7.3 关键约束

- 必须映射整个 `/dev/infiniband`，只映射 `/dev/kfd` 与 `/dev/dri` 不足以使用 verbs/GDRDMA。
- rollout 节点必须使用包含正确 ROCm vLLM build 的镜像。
- trainer 节点不需要安装 vLLM。
- 两节点容器必须使用 `--network=host`，否则 Ray IP、RoCE IP 与 rendezvous 地址需要重新配置。
- `USE_GRPO_DOCKERFILE=1` 时不要再挂载宿主机源码覆盖 `/workspace/*`，否则运行的不是构建时已验证的代码。

---

## 7.8 Fallback：从源码安装依赖

> **如果 §7.6 Dockerfile 构建成功，跳过本节。** 不要在角色镜像启动后再 `pip install`，
> 否则会破坏 Dockerfile 构建末尾验证过的版本矩阵。

仅当 `USE_GRPO_DOCKERFILE=0`（Dockerfile 不可用）时执行。

### 7.8.1 aiter（两节点，含 HIP C++ 编译）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
cd /workspace/aiter
/opt/venv/bin/pip install -e . 2>&1 | tail -5
/opt/venv/bin/python -c "import aiter; print(\"aiter ok:\", aiter.__file__)"
'
```

首次编译约 15-30 分钟。编译产物缓存在宿主机挂载的源码目录。

### 7.8.2 flash-attention（两节点，含 HIP C++ 编译）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
cd /workspace/flash-attention
GPU_ARCHS="gfx942" /opt/venv/bin/pip install -e . 2>&1 | tail -5
/opt/venv/bin/python -c "from flash_attn import flash_attn_varlen_func; print(\"flash_attn ok\")"
'
```

`GPU_ARCHS="gfx942"` 限制只编译 MI308X 架构，大幅缩短编译时间。

### 7.8.3 统一安装 ATOM / AtomRL / Lumen（两节点）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
# 保留该案例验证过的 Transformers 5.2.0；ATOM 元数据固定为 5.16.1。
LUMEN_BUILD_EXTENSIONS=1 /opt/venv/bin/pip install --no-build-isolation --no-deps -e /workspace/ATOM
/opt/venv/bin/pip install --no-build-isolation --no-deps -e /workspace/ATOM/atomrl
/opt/venv/bin/python -c "import atomrl, atomrl.lumen; print(\"atomrl/lumen ok:\", atomrl.__file__, atomrl.lumen.__file__)"
'
```

### 7.8.4 Python 依赖（两节点）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
P=/opt/venv/bin/pip
$P install "ray[default]==2.56.1" \
  "transformers==5.2.0" "datasets==5.0.0" "accelerate==1.14.0" \
  "safetensors==0.8.0" "omegaconf==2.3.1" \
  "math_verify==0.3.3" "wandb==0.28.1" \
  "numpy>=1.26" "pybind11>=3.0"
'
```

### 7.8.6 Megatron-LM 通过 .pth 引入（两节点）

**不能用 PyPI 的 `megatron-core`** 或 `pip install -e .`。PyPI 包只含 `megatron.core.*`，
不含 `megatron.training`。ROCm fork 的 `pyproject.toml` 也只把 `megatron.core` 列为可安装包。
Lumen 在模块顶层 `from megatron.training import get_args`，且 ROCm fork 包含
`RouterReplay`——R3 MoE replay 的核心依赖。

通过 `.pth` 文件指向 `/workspace/Megatron-LM`：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
/opt/venv/bin/pip uninstall -y megatron-core 2>/dev/null || true
SITE=$(/opt/venv/bin/python -c "import site; print(site.getsitepackages()[0])")
echo "/workspace/Megatron-LM" > "$SITE/megatron-lm-source.pth"
echo "wrote $SITE/megatron-lm-source.pth"
'
```

验证 `.pth` 生效（无需设置 `PYTHONPATH`）：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
/opt/venv/bin/python -c "
import megatron.core; print(\"megatron.core ok:\", megatron.core.__file__)
from megatron.training import get_args; print(\"megatron.training ok\")
from megatron.core.transformer.moe.router_replay import RouterReplay; print(\"RouterReplay ok\")
"
'
```

### 7.8.7 验证完整导入链（两节点）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i -e HIP_VISIBLE_DEVICES=0 \
  "$CONTAINER" /opt/venv/bin/python - <<'PY'
import sys
checks = []

import aiter; checks.append(("aiter", aiter.__file__))
import flash_attn; checks.append(("flash_attn", flash_attn.__file__))
import atomrl.lumen; checks.append(("lumen", atomrl.lumen.__file__))
import atomrl; checks.append(("atomrl", atomrl.__file__))
import megatron.core; checks.append(("megatron.core", megatron.core.__file__))

try:
    from megatron.training import get_args
    checks.append(("megatron.training", "ok"))
except ImportError:
    checks.append(("megatron.training", "MISSING — need ROCm Megatron-LM, not PyPI megatron-core"))

try:
    from megatron.core.transformer.moe.router_replay import RouterReplay
    checks.append(("RouterReplay", "ok"))
except ImportError:
    checks.append(("RouterReplay", "MISSING — need ROCm fork with router_replay"))

try:
    import vllm; checks.append(("vllm", vllm.__file__))
except ImportError:
    checks.append(("vllm", "NOT INSTALLED (ok for trainer node)"))

for name, path in checks:
    print(f"  {name:12s} {path}")

for name, expected_prefix in [
    ("lumen", "/workspace/Lumen/"),
    ("atomrl", "/workspace/ATOM/atomrl/"),
    ("megatron", "/workspace/Megatron-LM/"),
    ("aiter", "/workspace/aiter/"),
    ("flash_attn", "/workspace/flash-attention/"),
]:
    mod = sys.modules.get(name)
    if mod and hasattr(mod, "__file__") and mod.__file__:
        assert mod.__file__.startswith(expected_prefix), \
            f"{name} imported from {mod.__file__}, expected {expected_prefix}"
print("all source installs verified")
PY
```

### 7.8.8 flash-attn ABI 验证

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i -e HIP_VISIBLE_DEVICES=0 "$CONTAINER" /opt/venv/bin/python - <<'PY'
import torch
from flash_attn import flash_attn_varlen_func

q = torch.randn(12, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
cu = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_func(q, k, v, cu, cu, 7, 7, causal=True)
out.float().sum().backward()
torch.cuda.synchronize()
print("flash_varlen_forward_backward_ok", tuple(out.shape))
PY
```

期望输出：`flash_varlen_forward_backward_ok (12, 2, 128)`

---

## 7.9 模型与数据

### 7.9.1 下载模型

如果模型目录尚未就绪，在任一节点下载。两台节点的 `$MODEL_HOST_DIR` 都必须最终包含同一模型，
或使用真正共享的模型目录。

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i "$CONTAINER" /opt/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-30B-A3B",
    local_dir="/root/models/Qwen3-30B-A3B",
)
PY
```

### 7.9.2 准备过滤后的数据集

如果还没有过滤后的 parquet，在任一节点生成，再把结果同步到另一节点相同的容器内路径：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i "$CONTAINER" /opt/venv/bin/python - <<'PY'
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

out = Path("/root/data_cached/qwen3-30b-a3b-maxprompt1024")
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained("/root/models/Qwen3-30B-A3B")

def prompt_len(row):
    prompt = row["prompt"]
    return len(tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True))

def normalize_prompt(row):
    prompt = (
        row.get("prompt")
        or row.get("question")
        or row.get("problem")
        or row.get("input")
        or ""
    )
    if not isinstance(prompt, list):
        prompt = [{"role": "user", "content": str(prompt)}]
    return {"prompt": prompt}

jobs = (
    ("BytedTsinghua/DAPO-Math-17k", "train",
     out / "dapo-math-17k.filtered.parquet"),
    ("HuggingFaceH4/aime-2024", "train",
     out / "aime-2024.filtered.parquet"),
)
for repo, split, dst in jobs:
    ds = load_dataset(repo, split=split)
    ds = ds.map(normalize_prompt, num_proc=16)
    ds = ds.filter(lambda row: prompt_len(row) <= 1024, num_proc=16)
    ds.to_parquet(dst)
    print(dst, len(ds))
PY
```

### 7.9.3 启动前验证

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
test -f /root/models/Qwen3-30B-A3B/config.json
test -f /root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
test -f /root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
'
```

---

## 7.10 RDMA 预检

在**两个节点**容器内执行：

```bash
source "$HOME/qwen3-rdma-node.env"
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

NCCL/RCCL 环境变量（在训练启动环境中设置）：

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

RDMA 只有在训练日志中**同时出现**以下两行才算验证通过：

```text
NCCL INFO Using network IB
NCCL INFO ... via NET/IB/0/GDRDMA ...
```

只出现 socket/TCP 日志则 RDMA 未生效。

---

## 7.11 Ray 集群

### 7.11.1 启动 head（rollout 节点）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec \
  -e ROLLOUT_NODE_IP="$ROLLOUT_NODE_IP" \
  -e NCCL_SOCKET_IFNAME="$RDMA_IFACE" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_IB_GID_INDEX="$RDMA_GID_INDEX" \
  "$CONTAINER" bash -lc '
ulimit -n 524288
/opt/venv/bin/ray stop --force || true
/opt/venv/bin/ray start --head \
  --node-ip-address="$ROLLOUT_NODE_IP" \
  --port=6379 \
  --num-gpus=8 \
  --num-cpus=64 \
  --dashboard-host=0.0.0.0
'
```

### 7.11.2 加入 worker（trainer 节点）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec \
  -e ROLLOUT_NODE_IP="$ROLLOUT_NODE_IP" \
  -e TRAIN_NODE_IP="$TRAIN_NODE_IP" \
  -e NCCL_SOCKET_IFNAME="$RDMA_IFACE" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_IB_GID_INDEX="$RDMA_GID_INDEX" \
  "$CONTAINER" bash -lc '
ulimit -n 524288
/opt/venv/bin/ray stop --force || true
/opt/venv/bin/ray start \
  --address="$ROLLOUT_NODE_IP:6379" \
  --node-ip-address="$TRAIN_NODE_IP" \
  --num-gpus=8 \
  --num-cpus=64
'
```

### 7.11.3 验证

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" /opt/venv/bin/ray status
```

必须显示：

```text
Active: 2 nodes
Total: 16 GPU
```

---

## 7.12 生成部署配置

不要直接编辑仓库中的基准 YAML。只在 **rollout/driver 节点**执行，用自动发现的值生成
`/runtime/configs/` 配置：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i \
  -e ROLLOUT_NODE_IP="$ROLLOUT_NODE_IP" \
  -e TRAIN_NODE_IP="$TRAIN_NODE_IP" \
  -e RDMA_HCA="$RDMA_HCA" \
  -e RDMA_IFACE="$RDMA_IFACE" \
  -e RDMA_GID_INDEX="$RDMA_GID_INDEX" \
  -e WANDB_PROJECT="$WANDB_PROJECT" \
  -e WANDB_ENTITY="$WANDB_ENTITY" \
  "$CONTAINER" /opt/venv/bin/python - <<'PY'
import copy
import os
from pathlib import Path
from omegaconf import OmegaConf

src = Path("/workspace/ATOM/atomrl/examples/GRPO/configs/grpo_qwen3_30b_a3b_vllm_ep8_longrun.yaml")
out_dir = Path("/runtime/configs")
out_dir.mkdir(parents=True, exist_ok=True)

cfg = OmegaConf.load(src)
cfg.cluster.num_nodes = 2
cfg.cluster.gpus_per_node = 8
cfg.cluster.ray_address = "auto"
cfg.controller.ray.actor.topology_tags = {"node_ip": os.environ["TRAIN_NODE_IP"]}
cfg.controller.ray.rollout.topology_tags = {"node_ip": os.environ["ROLLOUT_NODE_IP"]}
cfg.weight_sync.backend = "rdma"
cfg.weight_sync.shared_folder = "/shared/atomrl_weight_sync/qwen3-30b-a3b"
cfg.weight_sync.rdma.backend = "rccl"
cfg.weight_sync.rdma.require_rdma = True
cfg.weight_sync.rdma.hca = os.environ["RDMA_HCA"]
cfg.weight_sync.rdma.interface = os.environ["RDMA_IFACE"]
cfg.weight_sync.rdma.gid_index = int(os.environ["RDMA_GID_INDEX"])
cfg.weight_sync.rdma.gdr_mode = "auto"
cfg.policy.model_name = "/root/models/Qwen3-30B-A3B"
cfg.reward.dataset = (
    "/root/data_cached/qwen3-30b-a3b-maxprompt1024/"
    "dapo-math-17k.filtered.parquet"
)
cfg.val_dataset = (
    "/root/data_cached/qwen3-30b-a3b-maxprompt1024/"
    "aime-2024.filtered.parquet"
)
cfg.checkpointing.checkpoint_dir = "/runtime/ckpts/qwen3-30b-a3b-rdma-longrun"
cfg.checkpointing.resume = False
cfg.eval.enabled = True
cfg.logger.wandb.project = os.environ["WANDB_PROJECT"]
cfg.logger.wandb.entity = os.environ.get("WANDB_ENTITY") or None
cfg.num_training_steps = 200

longrun = out_dir / "qwen3-30b-a3b-rdma-longrun.yaml"
OmegaConf.save(cfg, longrun)

smoke_cfg = copy.deepcopy(cfg)
smoke_cfg.policy.max_total_sequence_length = 128
smoke_cfg.policy.max_response_length = 64
smoke_cfg.policy.train_global_batch_size = 8
smoke_cfg.policy.gen_batch_size = 8
smoke_cfg.policy.max_token_len_per_gpu = 512
smoke_cfg.policy.generation.vllm_cfg.max_model_len = 128
smoke_cfg.val_steps = 0
smoke_cfg.eval.enabled = False
smoke_cfg.checkpointing.checkpoint_dir = "/runtime/ckpts/qwen3-30b-a3b-rdma-smoke"
smoke_cfg.checkpointing.save_steps = 9999
smoke_cfg.checkpointing.resume = False
smoke_cfg.logger.wandb_enabled = False
smoke_cfg.num_training_steps = 3
smoke = out_dir / "qwen3-30b-a3b-rdma-smoke.yaml"
OmegaConf.save(smoke_cfg, smoke)

print(longrun)
print(smoke)
PY
```

核对生成值——输出中不得出现旧机器 IP 或示例 RoCE 名称：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i "$CONTAINER" /opt/venv/bin/python - <<'PY'
from omegaconf import OmegaConf

for path in (
    "/runtime/configs/qwen3-30b-a3b-rdma-smoke.yaml",
    "/runtime/configs/qwen3-30b-a3b-rdma-longrun.yaml",
):
    cfg = OmegaConf.load(path)
    print(path)
    print("  actor:", cfg.controller.ray.actor.topology_tags.node_ip)
    print("  rollout:", cfg.controller.ray.rollout.topology_tags.node_ip)
    print("  RDMA:", cfg.weight_sync.rdma.hca, cfg.weight_sync.rdma.interface,
          cfg.weight_sync.rdma.gid_index)
PY
```

> `weight_sync.rdma.backend: rccl` 是配置语义；PyTorch API 仍使用 `backend="nccl"`，
> ROCm 运行时自动映射至 RCCL。

---

## 7.13 启动

### 7.13.1 W&B key（可选）

将 key 放到 rollout 节点的 `${RUNTIME_HOST_DIR}/wandb.key`：

```text
WANDB_API_KEY=...
```

不要把 key 放进 `docker exec -e WANDB_API_KEY=...` 的命令参数，否则会出现在 `ps` 输出。

### 7.13.2 Smoke（3 步，前台）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec \
  -e NCCL_SOCKET_IFNAME="$RDMA_IFACE" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_IB_GID_INDEX="$RDMA_GID_INDEX" \
  "$CONTAINER" bash -lc '
export PATH=/opt/venv/bin:$PATH
export RL_ROOT=/workspace
export DATA_ROOT=/runtime
export AITER_DIR=/workspace/aiter
export MODEL_PATH=/root/models/Qwen3-30B-A3B
export TRAIN_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
export VAL_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
export MODE=smoke
export STEPS=3
export BACKEND=vllm
export CONFIG_OVERRIDE=/runtime/configs/qwen3-30b-a3b-rdma-smoke.yaml
export ATOMRL_KEEP_RAY_CLUSTER=1
export RUN_ID=qwen3-30b-a3b-rdma-smoke3
export LOG=/runtime/logs/qwen3-30b-a3b-rdma-smoke3.log
export CKPT_DIR=/runtime/ckpts/qwen3-30b-a3b-rdma-smoke3
export RESUME_OVERRIDE=false
export WEIGHT_SYNC_BACKEND=rdma

bash /workspace/ATOM/atomrl/examples/GRPO/run_grpo.sh
'
```

### 7.13.3 Longrun（200 步，后台）

确认 3 步 smoke 通过 §7.14 验证后，再分离启动 200 步 longrun：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -d \
  -e NCCL_SOCKET_IFNAME="$RDMA_IFACE" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_IB_GID_INDEX="$RDMA_GID_INDEX" \
  "$CONTAINER" bash -lc '
export PATH=/opt/venv/bin:$PATH
export RL_ROOT=/workspace
export DATA_ROOT=/runtime
export AITER_DIR=/workspace/aiter
export MODEL_PATH=/root/models/Qwen3-30B-A3B
export TRAIN_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
export VAL_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
export MODE=longrun
export STEPS=200
export BACKEND=vllm
export CONFIG_OVERRIDE=/runtime/configs/qwen3-30b-a3b-rdma-longrun.yaml
export ATOMRL_KEEP_RAY_CLUSTER=1
export RUN_ID="qwen3-30b-a3b-rdma-longrun-$(date +%Y%m%d-%H%M%S)"
export LOG="/runtime/logs/${RUN_ID}.log"
export CKPT_DIR="/runtime/ckpts/${RUN_ID}"
export WANDB_RUN_NAME="$RUN_ID"
export RESUME_OVERRIDE=false
export WEIGHT_SYNC_BACKEND=rdma

if [ -f /runtime/wandb.key ]; then
  export WANDB_API_KEY="$(cut -d= -f2- /runtime/wandb.key | tr -d "[:space:]")"
fi

echo "$RUN_ID" > /runtime/current_run_id.txt
echo "$LOG" > /runtime/current_run_log.txt
echo "$CKPT_DIR" > /runtime/current_ckpt_dir.txt
bash /workspace/ATOM/atomrl/examples/GRPO/run_grpo.sh
'
```

> - `BACKEND=vllm` 与生成的 `CONFIG_OVERRIDE` 必须同时明确设置。`run_grpo.sh` 历史默认值可能仍是 `atom`。
> - 首次从 step 0 启动必须使用 `RESUME_OVERRIDE=false`。
> - 只有从 §7.15 验证完整的 checkpoint 恢复时才使用 `RESUME_OVERRIDE=true`。

---

## 7.14 启动验证

### 7.14.1 启动序列

日志中必须按顺序出现：

```text
Created 1 placement groups for pool 'rollout' ... node_ip=${ROLLOUT_NODE_IP}
Created 1 placement groups for pool 'actor' ... node_ip=${TRAIN_NODE_IP}
RDMA preflight: ...
RDMA weight group ready: ... world=9
NCCL INFO Using network IB
NCCL INFO ... NET/IB/0/GDRDMA ...
```

`world=9` 确认了 9-rank 进程组。`GDRDMA` 确认 GPU Direct RDMA 已激活。

### 7.14.2 每步确认

每个训练步必须同时输出：

```text
RDMA weight sync committed:
callbacks: step=N
```

### 7.14.3 重点指标

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

## 7.15 Checkpoint 验证

每 5 步保存一次。完整的 Megatron distributed-optimizer checkpoint（Qwen3-30B-A3B）必须包含：

- 8 个 model shard
- 8 个 optimizer metadata shard
- 8 个 extra-state shard
- 8 个大体积 optimizer parameter-state shard（每个约 41-45 GiB）
- controller 侧存在 `checkpoint_N.pt` 与 `latest_checkpointed_iteration.txt`

### 7.15.1 文件清单

在 trainer 节点，step 5 保存后检查：

```bash
source "$HOME/qwen3-rdma-node.env"
RUN_ID=<your-run-id>
P="$RUNTIME_HOST_DIR/ckpts/$RUN_ID/global_step_5/actor"
ls -lh "$P"/model_world_size_8_rank_*.pt
ls -lh "$P"/optim_world_size_8_rank_*.pt
ls -lh "$P"/optim_parameter_state_world_size_8_rank_*.pt
ls -lh "$P"/extra_state_world_size_8_rank_*.pt
```

### 7.15.2 自动数量检查

```bash
source "$HOME/qwen3-rdma-node.env"
RUN_ID=<your-run-id>
docker exec -i -e RUN_ID="$RUN_ID" "$CONTAINER" /opt/venv/bin/python - <<'PY'
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

### 7.15.3 Checkpoint 损坏记录

历史 v1 任务在保存 checkpoint 时磁盘写满。旧逻辑只保存了约 2 KiB 的
`optimizer.state_dict()` metadata，没有 Megatron distributed optimizer 的 FP32 master
和 Adam moments；这种 checkpoint 加载模型后继续更新会出现 NaN。

当前可恢复 checkpoint 必须同时覆盖：

```text
optimizer.state_dict()
optimizer.save_parameter_state(...)
optimizer.load_parameter_state(...)
optimizer.reload_model_params()
```

**严禁把只有小型 optimizer metadata 文件的 checkpoint 视为可续训 checkpoint。**

基线 checkpoint 大小：约 402 GiB（8 model + 8 optim-metadata + 8 extra-state + 8
optim-parameter-state shard，每个约 41-45 GiB）。

---

## 7.16 监控

### 7.16.1 训练状态

```bash
source "$HOME/qwen3-rdma-node.env"
LOG="$RUNTIME_HOST_DIR/logs/$(basename "$(cat "$RUNTIME_HOST_DIR/current_run_log.txt")")"

grep -a "atomrl.trainer.callbacks: step=" "$LOG" | tail -1
grep -a "RDMA weight sync committed" "$LOG" | tail -1
grep -aiE "Training failed|Traceback|OutOfMemory|NCCL.*timeout|SIGABRT|=nan" "$LOG" | tail
```

### 7.16.2 进程和 Ray

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" pgrep -af '[l]umenrl.trainer.main'
docker exec "$CONTAINER" /opt/venv/bin/ray status
```

### 7.16.3 磁盘

在 trainer 节点执行：

```bash
source "$HOME/qwen3-rdma-node.env"
RUN_ID=<your-run-id>
df -h "$RUNTIME_HOST_DIR"
du -sh "$RUNTIME_HOST_DIR/ckpts/$RUN_ID"/global_step_*
```

单个完整 checkpoint 约 402 GiB。`save_total_limit=3` 需要约 1.2 TiB，加上模型、日志和
Ray 临时文件必须预留安全余量。

### 7.16.4 W&B

W&B 页面使用 §7.3 输入的 `${WANDB_ENTITY}` / `${WANDB_PROJECT}` 和启动时生成的 `RUN_ID`。
不要复制历史任务的 run URL。

在线 history 的最新 global step 应与本地 callback 日志基本一致。若 run state 为 `crashed` 或
线上 step 长时间不增长，先检查本地 W&B 日志与网络，不要仅根据网页判断训练进程状态。

---

## 7.17 停止与恢复

### 7.17.1 停止

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" pkill -TERM -f '[l]umenrl.trainer.main' || true
```

### 7.17.2 恢复

恢复前必须先确认最新 checkpoint 含大体积
`optim_parameter_state_world_size_8_rank_*.pt`。然后使用相同 `CKPT_DIR`，设置：

```bash
export RESUME_OVERRIDE=true
```

日志必须出现：

```text
Resuming Ray actor checkpoint from ... global_step_N/actor
Ray resume complete. Next training log will be global_step=N+1
```

恢复后至少观察两个完整 step：

- 第一个 step 验证 checkpoint model 可加载。
- 第二个 step 验证 optimizer update 后权重仍为有限值。

只有连续两步 KL、ESS、loss、grad norm、entropy 全部有限，才视为恢复成功。

---

## 7.18 排障

### `ModuleNotFoundError: vllm`

原因：Ray 把 rollout placement group 调度到 trainer 节点。

处理：
- 确认 YAML 的 `rollout.topology_tags.node_ip=${ROLLOUT_NODE_IP}`。
- 确认 `actor.topology_tags.node_ip=${TRAIN_NODE_IP}`。
- 日志必须打印本次探测并写入 YAML 的两个 node IP。

### RDMA 退化成 TCP

```bash
source "$HOME/qwen3-rdma-node.env"
ls -l /dev/infiniband
test -d "/sys/class/infiniband/$RDMA_HCA"
```

确认 `NCCL_SOCKET_IFNAME=$RDMA_IFACE`、`NCCL_IB_HCA=$RDMA_HCA`、
`NCCL_IB_GID_INDEX=$RDMA_GID_INDEX`。没有 `NET/IB/.../GDRDMA` 日志就不能宣称
GPU Direct RDMA 已启用。

### flash-attn `varlen_fwd()` 参数不兼容

原因：Python wrapper 带 `num_splits`，native ROCm extension 是旧 21 参数 ABI。

处理：使用 `run_grpo.sh` 的幂等 ABI patch，并运行 §7.8.8 kernel 测试。

### Checkpoint 写满磁盘

症状：`RuntimeError: basic_ios::clear: iostream error` / `No space left on device`。

处理：
- 删除不完整的当前 step 目录。
- 保留至少一个已验证完整 checkpoint。
- 确认 actor 节点 `prune_checkpoints` 在保存前执行。
- 不要保留只有 model、没有 optimizer parameter state 的历史 checkpoint。

### 恢复后第二步 NaN

通常表示 optimizer 没有完整恢复，或 FP32 main parameters 未与 model 同步。

检查：
- `optim_parameter_state_world_size_*.pt` 是否存在且为几十 GiB。
- 是否调用 `load_parameter_state()`。
- 是否调用 `reload_model_params()`。
- 不要用 2 KiB 的 optimizer metadata 文件代替 distributed parameter state。

### W&B 没更新

检查：
- `/runtime/wandb.key` 是否存在且容器内可读。
- 本地 callback 是否继续输出 step。
- W&B run name 是否与本次 `RUN_ID` 一致。
- resume 时不要让 global step 回退，否则 W&B 会拒绝旧 step。

### `Too many open files` / Ray socket EOF

容器启动和训练脚本均设置：

```bash
ulimit -n 524288
```

必要时停止两节点 Ray 后重新按 §7.11 顺序启动。

---

## 7.19 关键源码

```text
atomrl/core/config.py
atomrl/utils/independent_process_group.py
atomrl/engine/inference/rdma_weight_transfer.py
atomrl/engine/inference/vllm_colocate_worker_ext.py
atomrl/engine/inference/vllm_ray_server.py
atomrl/workers/actor_worker.py
atomrl/controller/ray_worker_group.py
atomrl/trainer/rl_trainer.py
atomrl/trainer/callbacks.py
atomrl/engine/training/megatron_engine.py
examples/GRPO/run_grpo.sh
examples/GRPO/configs/grpo_qwen3_30b_a3b_vllm_ep8_longrun.yaml
```

---

## 附录 A: ATOM rollout

本附录说明可选的 ATOM rollout 替代方案。正文中的 200 步部署使用 vLLM + RCCL/RoCE RDMA。

### A.1 支持范围和限制

ATOM 与当前 RDMA backend 不能直接组合：

- `policy.generation_backend: atom` 会创建 `ATOMReplicaManager`。
- 当前 ATOM manager 没有初始化 9-rank RCCL weight group，也没有 RDMA receiver。
- 因此 ATOM 配置不能设置 `weight_sync.backend: rdma`。
- 两节点 ATOM 必须使用 `shared_folder`；同节点可使用 `auto`。

`weight_sync.backend: auto` 的实际选择规则：

- 同节点、ATOM TP=1、8 replicas 对应 8 actor workers：使用 ZMQ CUDA-IPC。
- 同节点、ATOM TP=2、4 replicas 少于 8 actor workers：自动回退到 safetensors。
- ATOM 与 trainer 分布在不同节点：自动回退到 safetensors。

Qwen3-30B-A3B 当前推荐 ATOM TP=2，因此实际稳定路径是：

```text
Megatron BF16 shard
  → actor 聚合并转换为 HF 名称
  → safetensors 导出
  → ATOM TP=2 × 4 replicas reload
```

不要把该路径标记为 RDMA，也不要期待日志出现 `NET/IB/.../GDRDMA`。

### A.2 部署形态

推荐先做单节点 colocated smoke：

```text
单节点 8 × MI308X
├── Megatron actor workers：8
├── ATOM replicas：4
├── 每个 ATOM replica：TP=2
├── ATOM 权重：BF16
├── KV cache：FP8
└── sleep mode：level 2
```

同一批 GPU 在 rollout 和 training 间切换：

1. ATOM wake weights/KV cache。
2. 生成 response，并记录 MoE router logits。
3. ATOM sleep level 2，释放可回收显存。
4. Megatron 使用 R3 distribution replay 计算 log-prob 并更新参数。
5. actor 导出 HF safetensors。
6. ATOM wake weights，reload 新参数，再恢复 KV cache。

单节点同时容纳 ATOM CUDA graph、Megatron 参数和 distributed optimizer，显存压力明显高于正文的
两节点方案。出现 OOM 时优先降低 response length、batch size、ATOM
`gpu_memory_utilization`，不要关闭 checkpoint 的 optimizer parameter-state 保存。

### A.3 ATOM 源码和环境

如果使用自定义 ATOM 源码，先提供宿主机目录：

```bash
read -r -p "ATOM 源码目录 ATOM_HOST_DIR: " ATOM_HOST_DIR
export ATOM_HOST_DIR
test -f "$ATOM_HOST_DIR/atom/__init__.py"
```

创建容器时额外添加：

```bash
-v "$ATOM_HOST_DIR":/workspace/ATOM
```

不挂载源码时使用镜像内 ATOM。挂载后启动脚本会把 `/workspace/ATOM` 放到 `PYTHONPATH`。
验证实际导入位置：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
PYTHONPATH=/workspace/ATOM:/workspace/ATOM/atomrl \
  /opt/venv/bin/python - <<PY
import atom
from atom.rollout.async_engine import AsyncLLMEngine
print("atom module:", atom.__file__)
print("AsyncLLMEngine:", AsyncLLMEngine)
PY
'
```

期望 `atom.__file__` 位于 `/workspace/ATOM/atom/`，而不是意外导入其他 site-packages 版本。

ATOM TP>1 时保留以下环境：

```bash
export ATOMRL_DISABLE_CUSTOM_AR=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
export ATOM_ISOLATE_TORCH_COMPILE_CACHE=1
export ATOM_TORCH_COMPILE_CACHE_ROOT=/tmp/atom_torch_compile_cache
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USE_AITER_LINEAR=0
export USE_ROCM_AITER_ROPE_BACKEND=0
```

将 ATOM 的公开引擎选项 `disable_custom_all_reduce` 设为 `true`，可将 TP
路径从 HIP IPC custom all-reduce 切换为 RCCL。这里的 RCCL 只用于 ATOM TP
collective，不是正文所述的跨节点 RDMA 权重广播。

### A.4 ATOM YAML

基准文件：`examples/GRPO/configs/grpo_qwen3_30b_a3b_atom_ep8_longrun.yaml`

关键配置：

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8

weight_sync:
  backend: auto
  shared_folder: /shared/atomrl_weight_sync/atom-qwen3-30b-a3b
  bucket_size_mb: 1024
  timeout_s: 600

controller:
  ray:
    enabled: true
    fuse_actor_ref: false
    actor:
      num_workers: 8

policy:
  training_backend: megatron
  generation_backend: atom
  training:
    megatron_cfg:
      tensor_parallel_size: 4
      expert_parallel_size: 8
      expert_tensor_parallel_size: 1
      use_distributed_optimizer: true
      sequence_parallel: true
      attention_backend: flash
  generation:
    atom_cfg:
      tensor_parallel_size: 2
      expert_parallel_size: 1
      kv_cache_dtype: fp8
      engine_kwargs:
        enforce_eager: false
        compilation_config:
          level: 3
    vllm_cfg:
      gpu_memory_utilization: 0.60
      max_model_len: 9216
      dtype: bfloat16
      quantization: ""
      enable_sleep_mode: true
      sleep_level: 2

moe:
  r3:
    enabled: true
    record_router_logits: true
    replay_mode: distribution
```

注意：

- `atom_cfg` 控制 ATOM engine；当前实现仍从 `vllm_cfg` 读取部分通用 generation/sleep 参数。
- `quantization: ""` 表示 BF16 权重，不启用 `per_block_fp8`。
- FP8 只用于 KV cache。Qwen3-30B-A3B 的 MoE intermediate size 为 768，在线
  `per_block_fp8` 已观测到输出乱码。
- `grpo_qwen3_30b_a3b_atom_ep8_smoke.yaml` 当前内容实际是
  `generation_backend: vllm`，不能作为 ATOM smoke 直接使用；应复制 longrun 文件并缩小参数。

ATOM smoke 建议覆盖：

```yaml
policy:
  max_total_sequence_length: 1088
  max_response_length: 1024
  train_global_batch_size: 64
  gen_batch_size: 8
  max_token_len_per_gpu: 4096

checkpointing:
  save_steps: 9999
  resume: false

logger:
  wandb_enabled: false

num_training_steps: 3
```

### A.5 启动

先准备本地同步目录：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
mkdir -p /shared/atomrl_weight_sync/atom-qwen3-30b-a3b
rm -f /shared/atomrl_weight_sync/atom-qwen3-30b-a3b/*.tmp
'
```

使用单独制作的 ATOM smoke YAML 启动：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
export PATH=/opt/venv/bin:$PATH
export RL_ROOT=/workspace
export DATA_ROOT=/runtime
export MODEL_PATH=/root/models/Qwen3-30B-A3B
export TRAIN_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
export VAL_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
export MODE=smoke
export STEPS=3
export BACKEND=atom
export CONFIG_OVERRIDE=examples/GRPO/configs/grpo_qwen3_30b_a3b_atom_ep8_smoke_local.yaml
export RUN_ID=qwen3-30b-a3b-atom-smoke3
export LOG=/runtime/logs/qwen3-30b-a3b-atom-smoke3.log
export CKPT_DIR=/runtime/ckpts/qwen3-30b-a3b-atom-smoke3
export WEIGHT_SYNC_BACKEND=auto
export RESUME_OVERRIDE=false

bash /workspace/ATOM/atomrl/examples/GRPO/run_grpo.sh
'
```

> `BACKEND=atom` 不足以修正一个内容仍为 vLLM 的 `CONFIG_OVERRIDE`；必须确认最终 YAML
> 中明确写有 `policy.generation_backend: atom`。

### A.6 ATOM smoke 验证

启动日志应出现：

```text
ATOMReplicaManager: launched 4 colocated rollout replicas (atom_tp=2, workers=8)
Ray ATOM rollout ready: 4 colocated replicas
R3Manager: router recording ENABLED
```

TP=2×4 时权重同步应出现 safetensors export/reload 日志，而不是 RDMA 日志。每步必须确认
`callbacks: step=N`。

检查：

```bash
source "$HOME/qwen3-rdma-node.env"
LOG="$RUNTIME_HOST_DIR/logs/qwen3-30b-a3b-atom-smoke3.log"
grep -aE "ATOMReplicaManager|Ray ATOM rollout ready|reloaded weights|callbacks: step=" "$LOG" | tail -30
grep -aiE "Traceback|OutOfMemory|illegal memory access|ca_comm|=nan" "$LOG" | tail
```

通过标准：

- 连续完成至少 3 步。
- reward、loss、entropy、grad norm、KL 和 ESS 全部有限。
- R3 router-logit coverage 完整。
- 每轮训练后 ATOM 成功 reload 权重。
- 无乱码、非法显存访问、OOM 或 stale CUDA/HIP IPC handle。

ATOM smoke 通过后，才可切换到 `grpo_qwen3_30b_a3b_atom_ep8_longrun.yaml`。正式长跑仍应采用
§7.15 的完整 distributed optimizer checkpoint 规则；不要恢复不完整 checkpoint。

---

## 参考

- AtomRL：`https://github.com/ROCm/ATOM/tree/main/atomrl`
- Lumen：`${WORK_ROOT}/ATOM/atomrl/lumen`（内置）
- Megatron-LM（ROCm fork）：`https://github.com/ROCm/Megatron-LM/tree/rocm_dev`
- aiter：`https://github.com/ROCm/aiter/tree/adbb6ac8b84110134e0cf4dc513b250d2cc0a75c`
- flash-attention：`https://github.com/ROCm/flash-attention`
- Qwen3-30B-A3B：`https://huggingface.co/Qwen/Qwen3-30B-A3B`
