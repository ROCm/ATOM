> [Examples README](../README.md) > Disaggregated 2-node RDMA

# 7. Disaggregated 2-Node Megatron + vLLM RDMA Deployment

Deploy **Qwen3-30B-A3B** MoE RL training from scratch on two 8-GPU AMD machines with
disaggregated training and inference: Megatron trainer on node 1, vLLM rollout on node 2,
connected via RCCL/RoCE GPU Direct RDMA weight sync.

> This guide covers a fundamentally different deployment from examples 1-7 (single-node
> co-located). For single-node setups, see [Launching](04-launching.md). For RDMA
> verification procedures applicable to any multi-node deployment, see
> [Multi-node RDMA](05-multinode-rdma.md).

**One-line summary**: collect paths -> auto-discover network -> clone repos -> build
Docker images -> start containers -> install deps (if needed) -> download model/data ->
RDMA pre-checks -> start Ray cluster -> generate config -> smoke (3 steps) -> longrun
(200 steps).

---

## 7.1 Architecture overview

| Role | Node | GPU | Key software |
|---|---|---|---|
| Rollout / Ray head | `${ROLLOUT_NODE_IP}` | 8x MI308X (gfx942) | vLLM TP=2 x 4 replicas |
| Megatron trainer | `${TRAIN_NODE_IP}` | 8x MI308X (gfx942) | Megatron TP=4, EP=8, ETP=1 |

Weight sync uses an independent 9-rank `torch.distributed` process group:

- rank 0: Megatron sender
- rank 1-8: vLLM TP receivers
- ROCm `backend="nccl"` maps to RCCL at runtime
- Transport: RoCE RDMA via auto-discovered `${RDMA_HCA}` / `${RDMA_IFACE}` / `${RDMA_GID_INDEX}`
- Log must show `Using network IB` and `NET/IB/0/GDRDMA`

Training and rollout:

- Training weights: BF16 compute + Megatron distributed optimizer FP32 master/Adam state
- Rollout weights: BF16; KV cache `auto` (BF16 baseline)
- R3: vLLM records top-k expert IDs, Megatron `RouterReplay` executes hard assignment replay
- Algorithm: GRPO/DAPO, 32 prompts x 8 generations, global batch 256
- Target: 200 steps

This flow does not use ATOM rollout, ZMQ CUDA-IPC, or cross-node safetensors as the
primary path. For the optional ATOM alternative, see [Appendix A](#appendix-a-atom-rollout).

---

## 7.2 Host prerequisites

Run on **both hosts** before anything else. The starting point is two machines with AMD
GPU drivers, RDMA kernel drivers, and Docker already installed.

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

The last command should find 8 GPU agents. If the count is not 8, fix the host
driver/device permissions before proceeding.

---

## 7.3 Path variables and env file

### 7.3.1 Collect paths on both hosts

These are host-side paths. Each machine fills in its own values; container mount points
are kept consistent.

```bash
read -r -p "Code root WORK_ROOT: " WORK_ROOT
read -r -p "Model root MODEL_HOST_DIR (Qwen3-30B-A3B goes under here): " MODEL_HOST_DIR
read -r -p "Dataset root DATASET_HOST_DIR: " DATASET_HOST_DIR
read -r -p "Log/checkpoint dir RUNTIME_HOST_DIR: " RUNTIME_HOST_DIR
read -r -p "Shared dir SHARED_HOST_DIR (use an empty local dir if none): " SHARED_HOST_DIR
read -r -p "Rollout image [rocm/atom-dev:vllm-latest]: " ROLLOUT_IMAGE
read -r -p "Trainer image [rocm/atom-dev:latest]: " TRAIN_IMAGE
read -r -p "W&B project [AtomRL]: " WANDB_PROJECT
read -r -p "W&B entity (leave empty for default account): " WANDB_ENTITY

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
  test -n "$p" || { echo "path must not be empty"; exit 1; }
  mkdir -p "$p"
done
mkdir -p "$RUNTIME_HOST_DIR/logs" "$RUNTIME_HOST_DIR/ckpts"
```

### 7.3.2 Save env file

Save each host's values to a user-readable env file. All subsequent commands begin with
`source "$HOME/qwen3-rdma-node.env"`.

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

### 7.3.3 Container mount points

| Host path | Container path | Purpose |
|---|---|---|
| `$ATOMRL_HOST_DIR` | `/workspace/ATOM/atomrl` | AtomRL source |
| `$LUMEN_HOST_DIR` | `/workspace/Lumen` | Lumen dependency source |
| `$MEGATRON_HOST_DIR` | `/workspace/Megatron-LM` | ROCm Megatron-LM (megatron.training + RouterReplay) |
| `$AITER_HOST_DIR` | `/workspace/aiter` | aiter GPU kernel source |
| `$FA_HOST_DIR` | `/workspace/flash-attention` | ROCm flash-attention source |
| `$MODEL_HOST_DIR` | `/root/models` | Model weights |
| `$DATASET_HOST_DIR` | `/root/data_cached` | Filtered datasets |
| `$RUNTIME_HOST_DIR` | `/runtime` | Logs and checkpoints |
| `$SHARED_HOST_DIR` | `/shared` | Optional shared folder fallback |
| (device) | `/dev/infiniband` | RoCE verbs devices |

---

## 7.4 Network auto-discovery

Run on **each host separately**. Do not copy IPs from examples, old logs, or other clusters.

### 7.4.1 Discover Ray node IP

```bash
AUTO_RAY_IP=$(
  ip -4 route get "${RAY_PROBE_TARGET:-1.1.1.1}" |
    awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}'
)
if [ -z "$AUTO_RAY_IP" ]; then
  AUTO_RAY_IP=$(ip -o -4 addr show scope global | awk 'NR==1 {split($4,a,"/"); print a[1]}')
fi

echo "Auto-selected: $AUTO_RAY_IP"
echo "All candidates:"
ip -o -4 addr show scope global |
  awk '{split($4,a,"/"); printf "  iface=%-16s ip=%s\n",$2,a[1]}'

read -r -p "Confirm Ray node IP [${AUTO_RAY_IP}]: " NODE_RAY_IP
export NODE_RAY_IP="${NODE_RAY_IP:-$AUTO_RAY_IP}"
test -n "$NODE_RAY_IP"
```

On the rollout node: `export ROLLOUT_NODE_IP="$NODE_RAY_IP"`

On the trainer node: `export TRAIN_NODE_IP="$NODE_RAY_IP"`

Exchange the auto-detected IPs between the two nodes, then on both nodes:

```bash
read -r -p "Rollout node Ray IP: " ROLLOUT_NODE_IP
read -r -p "Trainer node Ray IP: " TRAIN_NODE_IP
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export ROLLOUT_NODE_IP='$ROLLOUT_NODE_IP'
export TRAIN_NODE_IP='$TRAIN_NODE_IP'
EOF
```

Verify:

```bash
source "$HOME/qwen3-rdma-node.env"
test "$ROLLOUT_NODE_IP" != "$TRAIN_NODE_IP"
ping -c 2 "$ROLLOUT_NODE_IP"
ping -c 2 "$TRAIN_NODE_IP"
```

If the auto-selected address cannot reach the other node, pick an actually reachable
address from the "All candidates" list.

### 7.4.2 Discover RoCE HCA, interface, and GID

The following script enumerates active RDMA ports from sysfs and selects the first RoCE
v2 GID with an IPv4 address. Run on each host separately:

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
    raise SystemExit("no active RoCE v2 + IPv4 candidate found")

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

If the script picks a RoCE network that is not the interconnect between the two nodes,
set `RDMA_IFACE` to the correct candidate and re-select via the corresponding sysfs GID.

On the rollout node: `export ROLLOUT_RDMA_IP="$RDMA_IP"`

On the trainer node: `export TRAIN_RDMA_IP="$RDMA_IP"`

Both nodes currently require matching HCA name, interface name, and GID index. Exchange
the auto-detected RoCE IPs, then on both nodes:

```bash
read -r -p "Rollout node RoCE IP: " ROLLOUT_RDMA_IP
read -r -p "Trainer node RoCE IP: " TRAIN_RDMA_IP
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export RDMA_HCA='$RDMA_HCA'
export RDMA_PORT='$RDMA_PORT'
export RDMA_IFACE='$RDMA_IFACE'
export RDMA_GID_INDEX='$RDMA_GID_INDEX'
export ROLLOUT_RDMA_IP='$ROLLOUT_RDMA_IP'
export TRAIN_RDMA_IP='$TRAIN_RDMA_IP'
EOF
```

Verify the RoCE network:

```bash
source "$HOME/qwen3-rdma-node.env"
ip -4 addr show dev "$RDMA_IFACE"

# on rollout node
ping -I "$RDMA_IFACE" -c 3 "$TRAIN_RDMA_IP"

# on trainer node
ping -I "$RDMA_IFACE" -c 3 "$ROLLOUT_RDMA_IP"
```

Final role table (filled from this deployment's auto-discovery):

| Role | Ray node IP | RoCE IP | GPU | Container image |
|---|---|---|---|---|
| Rollout / Ray head | `${ROLLOUT_NODE_IP}` | `${ROLLOUT_RDMA_IP}` | 8 | `${ROLLOUT_IMAGE}` |
| Megatron trainer | `${TRAIN_NODE_IP}` | `${TRAIN_RDMA_IP}` | 8 | `${TRAIN_IMAGE}` |

---

## 7.5 Source code

All components are installed from source. Both nodes execute the same cloning procedure.

| Component | Repository | Branch |
|---|---|---|
| AtomRL | `https://github.com/ROCm/ATOM.git` | `main` |
| Lumen | `${WORK_ROOT}/ATOM/atomrl/lumen` | in-tree |
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

# Lumen is imported in the AtomRL tree.
LUMEN_HOST_DIR="${WORK_ROOT}/ATOM/atomrl/lumen"
test -d "$LUMEN_HOST_DIR/lumen"

# Megatron-LM (ROCm fork with megatron.training and RouterReplay)
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

# save paths
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export AITER_HOST_DIR='$AITER_HOST_DIR'
export FA_HOST_DIR='$FA_HOST_DIR'
EOF
```

Verify both nodes have identical commits:

```bash
source "$HOME/qwen3-rdma-node.env"
echo "AtomRL:    $(cd "$ATOMRL_HOST_DIR" && git rev-parse HEAD)"
echo "Lumen:      $(cd "$LUMEN_HOST_DIR" && git rev-parse HEAD)"
echo "Megatron:   $(cd "$MEGATRON_HOST_DIR" && git rev-parse HEAD)"
echo "aiter:      $(cd "$AITER_HOST_DIR" && git rev-parse HEAD)"
echo "fa:         $(cd "$FA_HOST_DIR" && git rev-parse HEAD)"
```

The five lines of output must match between the two nodes.

---

## 7.6 Docker images

### 7.6.1 GRPO Dockerfile (preferred)

If the GRPO Dockerfile exists, use it to build separate role-specific images:

```bash
source "$HOME/qwen3-rdma-node.env"
DOCKERFILE="$ATOMRL_HOST_DIR/examples/GRPO/docker/Dockerfile"
test -f "$DOCKERFILE" || {
  echo "GRPO Dockerfile not found — use §7.8 fallback installation"
  export USE_GRPO_DOCKERFILE=0
}
```

| Target | Default base image | Output image | Purpose |
|---|---|---|---|
| `rollout` | `rocm/atom-dev:vllm-latest` | `qwen3-30b-a3b:rollout` | Ray head + vLLM TP2 x 4 |
| `trainer` | `rocm/atom-dev:latest` | `qwen3-30b-a3b:trainer` | Megatron TP4/EP8 |

The two targets cannot be merged into a single image. The rollout vLLM/NumPy/Triton
build differs from the trainer's; forcing unification breaks the verified combination.

The Docker build context must be the **ATOM repository root**, not `atomrl/examples/GRPO/`.

#### Rollout node build

Run only on the rollout node:

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

#### Trainer node build

Run only on the trainer node:

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

First-time builds compile gfx942 kernels for aiter, flash-attention, and Lumen HIP
extensions — expect significant build time. Subsequent builds with the same source and
build args reuse Docker layer cache.

To pin a specific base image digest and source refs for reproducibility:

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

Verify build output:

```bash
docker image inspect qwen3-30b-a3b:rollout \
  --format 'rollout {{.Id}} {{index .Config.Labels "org.opencontainers.image.title"}}' \
  2>/dev/null || true
docker image inspect qwen3-30b-a3b:trainer \
  --format 'trainer {{.Id}} {{index .Config.Labels "org.opencontainers.image.title"}}' \
  2>/dev/null || true
```

If building on one machine only, push to an internal registry or use `docker save` /
`docker load`. Each node must run its own role's target image.

### 7.6.2 Verified software versions

| Package | Version |
|---|---|
| Python | `3.12.3` |
| PyTorch | `2.10.0+rocm7.2.4.git3d3aa833` |
| HIP | `7.2.53211` |
| Ray | `2.56.1` |
| Megatron-LM (ROCm fork) | `rocm_dev` branch |
| flash-attn | `2.8.4` |
| amd-aiter | `0.1.0` |
| Transformers | `5.2.0` |
| Datasets | `5.0.0` |
| Accelerate | `1.14.0` |
| Safetensors | `0.8.0` |
| OmegaConf | `2.3.1` |
| math-verify | `0.3.3` |
| W&B | `0.28.1` |

Rollout-only: vLLM `0.22.1.dev0+g0b3ba88f1.d20260629.rocm724`.

Trainer-only: vLLM is **not installed** (expected).

---

## 7.7 Container startup

### 7.7.1 Rollout node

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

### 7.7.2 Trainer node

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

### 7.7.3 Key constraints

- Must map the entire `/dev/infiniband`; only `/dev/kfd` + `/dev/dri` is not sufficient for verbs/GDRDMA.
- The rollout node must use an image with the correct ROCm vLLM build.
- The trainer node does not need vLLM installed.
- Both containers must use `--network=host`; otherwise Ray IP, RoCE IP, and rendezvous addresses need reconfiguration.
- When `USE_GRPO_DOCKERFILE=1`, do not mount host source code over `/workspace/*` — that would override the build-time verified code.

---

## 7.8 Fallback: install dependencies from source

> **Skip this section entirely if §7.6 Dockerfile build succeeded.** Do not `pip install`
> into a container launched from the role-specific images — it will break the version
> matrix verified at Dockerfile build time.

This fallback applies only when `USE_GRPO_DOCKERFILE=0` (Dockerfile not available).

### 7.8.1 aiter (both nodes, HIP C++ compilation)

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
cd /workspace/aiter
/opt/venv/bin/pip install -e . 2>&1 | tail -5
/opt/venv/bin/python -c "import aiter; print(\"aiter ok:\", aiter.__file__)"
'
```

First-time compilation takes ~15-30 minutes. Build artifacts are cached in the
host-mounted source directory.

### 7.8.2 flash-attention (both nodes, HIP C++ compilation)

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
cd /workspace/flash-attention
GPU_ARCHS="gfx942" /opt/venv/bin/pip install -e . 2>&1 | tail -5
/opt/venv/bin/python -c "from flash_attn import flash_attn_varlen_func; print(\"flash_attn ok\")"
'
```

`GPU_ARCHS="gfx942"` limits compilation to the MI308X architecture, significantly
reducing build time.

### 7.8.3 Unified ATOM / AtomRL / Lumen install (both nodes)

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
# Keep this case's validated Transformers 5.2.0; ATOM metadata pins 5.16.1.
LUMEN_BUILD_EXTENSIONS=1 /opt/venv/bin/pip install --no-build-isolation --no-deps -e /workspace/ATOM
/opt/venv/bin/pip install --no-build-isolation --no-deps -e /workspace/ATOM/atomrl
/opt/venv/bin/python -c "import atomrl, atomrl.lumen; print(\"atomrl/lumen ok:\", atomrl.__file__, atomrl.lumen.__file__)"
'
```

### 7.8.4 Python dependencies (both nodes)

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

### 7.8.6 Megatron-LM via .pth (both nodes)

**Do not use PyPI `megatron-core`** or `pip install -e .`. The PyPI package only contains
`megatron.core.*`, not `megatron.training`. The ROCm fork's `pyproject.toml` also only
lists `megatron.core` as installable. Lumen imports `from megatron.training import
get_args` at module top-level, and the ROCm fork contains `RouterReplay` — the R3 MoE
replay core dependency.

Install via `.pth` file pointing to `/workspace/Megatron-LM`:

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

Verify the `.pth` file works (no `PYTHONPATH` needed):

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

### 7.8.7 Import chain verification (both nodes)

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

### 7.8.8 flash-attn ABI verification

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

Expected: `flash_varlen_forward_backward_ok (12, 2, 128)`

---

## 7.9 Models and data

### 7.9.1 Download model

If the model directory is not yet populated, download on either node. Both nodes'
`$MODEL_HOST_DIR` must ultimately contain the same model, or use a truly shared directory.

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

### 7.9.2 Prepare filtered datasets

If filtered parquets are not already available, generate on either node and sync the
results to the other node at the same container path:

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

### 7.9.3 Verify before launch

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
test -f /root/models/Qwen3-30B-A3B/config.json
test -f /root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
test -f /root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
'
```

---

## 7.10 RDMA pre-checks

Run inside containers on **both nodes**:

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

NCCL/RCCL environment variables (set in the training launch environment):

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

RDMA is only verified when the training log contains **both**:

```text
NCCL INFO Using network IB
NCCL INFO ... via NET/IB/0/GDRDMA ...
```

Socket/TCP-only logs mean RDMA is not active.

---

## 7.11 Ray cluster

### 7.11.1 Start head (rollout node)

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

### 7.11.2 Join worker (trainer node)

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

### 7.11.3 Verify

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" /opt/venv/bin/ray status
```

Must show:

```text
Active: 2 nodes
Total: 16 GPU
```

---

## 7.12 Config generation

Do not edit the base YAML files in the repository. Generate deployment-specific configs
with auto-discovered values. Run only on the **rollout/driver node**:

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

Verify the generated values — no old-machine IPs or example RoCE names should appear:

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

> `weight_sync.rdma.backend: rccl` is the config semantics; the PyTorch API still uses
> `backend="nccl"`, which ROCm automatically maps to RCCL.

---

## 7.13 Launching

### 7.13.1 W&B key (optional)

Place the key at `${RUNTIME_HOST_DIR}/wandb.key` on the rollout node:

```text
WANDB_API_KEY=...
```

Do not pass the key via `docker exec -e WANDB_API_KEY=...` — it would appear in `ps` output.

### 7.13.2 Smoke (3-step, foreground)

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

### 7.13.3 Longrun (200-step, background)

Only launch after the 3-step smoke passes §7.14 verification:

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

> - `BACKEND=vllm` and the generated `CONFIG_OVERRIDE` must both be explicitly set. The
>   `run_grpo.sh` historical default may still be `atom`.
> - First launch from step 0 must use `RESUME_OVERRIDE=false`.
> - Use `RESUME_OVERRIDE=true` only when resuming from a §7.15-verified complete checkpoint.

---

## 7.14 Launch verification

### 7.14.1 Startup sequence

These lines must appear in order in the log:

```text
Created 1 placement groups for pool 'rollout' ... node_ip=${ROLLOUT_NODE_IP}
Created 1 placement groups for pool 'actor' ... node_ip=${TRAIN_NODE_IP}
RDMA preflight: ...
RDMA weight group ready: ... world=9
NCCL INFO Using network IB
NCCL INFO ... NET/IB/0/GDRDMA ...
```

`world=9` confirms the 9-rank process group. `GDRDMA` confirms GPU Direct RDMA.

### 7.14.2 Per-step confirmation

Every training step must emit both:

```text
RDMA weight sync committed:
callbacks: step=N
```

### 7.14.3 Key metrics

```text
rollout_corr/kl
rollout_corr/rollout_is_eff_sample_size
actor/loss
actor/grad_norm
actor/entropy
weight_sync/gbps
timing/weight_sync_rdma_s
```

**If any core metric is `nan`, stop immediately. Do not write the next checkpoint.**

---

## 7.15 Checkpoint verification

Checkpoints are saved every 5 steps. A complete Megatron distributed-optimizer checkpoint
for Qwen3-30B-A3B requires:

- 8 model shards
- 8 optimizer metadata shards
- 8 extra-state shards
- 8 large optimizer parameter-state shards (each ~41-45 GiB)
- On the controller side: `checkpoint_N.pt` and `latest_checkpointed_iteration.txt`

### 7.15.1 File listing

On the trainer node, after step 5 saves:

```bash
source "$HOME/qwen3-rdma-node.env"
RUN_ID=<your-run-id>
P="$RUNTIME_HOST_DIR/ckpts/$RUN_ID/global_step_5/actor"
ls -lh "$P"/model_world_size_8_rank_*.pt
ls -lh "$P"/optim_world_size_8_rank_*.pt
ls -lh "$P"/optim_parameter_state_world_size_8_rank_*.pt
ls -lh "$P"/extra_state_world_size_8_rank_*.pt
```

### 7.15.2 Automated count check

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

### 7.15.3 Checkpoint corruption history

A previous v1 run ran out of disk during checkpoint saving. The old code only saved
~2 KiB of `optimizer.state_dict()` metadata, missing the Megatron distributed optimizer's
FP32 master weights and Adam moments. Loading such a checkpoint and continuing updates
produces NaN.

A valid resumable checkpoint must call all four methods:

```text
optimizer.state_dict()
optimizer.save_parameter_state(...)
optimizer.load_parameter_state(...)
optimizer.reload_model_params()
```

**Any checkpoint with only small optimizer metadata files must not be treated as
resumable.**

Baseline checkpoint size: ~402 GiB (8 model + 8 optim-metadata + 8 extra-state + 8
optim-parameter-state shards at ~41-45 GiB each).

---

## 7.16 Monitoring

### 7.16.1 Training status

```bash
source "$HOME/qwen3-rdma-node.env"
LOG="$RUNTIME_HOST_DIR/logs/$(basename "$(cat "$RUNTIME_HOST_DIR/current_run_log.txt")")"

grep -a "atomrl.trainer.callbacks: step=" "$LOG" | tail -1
grep -a "RDMA weight sync committed" "$LOG" | tail -1
grep -aiE "Training failed|Traceback|OutOfMemory|NCCL.*timeout|SIGABRT|=nan" "$LOG" | tail
```

### 7.16.2 Process and Ray

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" pgrep -af '[l]umenrl.trainer.main'
docker exec "$CONTAINER" /opt/venv/bin/ray status
```

### 7.16.3 Disk

On the trainer node:

```bash
source "$HOME/qwen3-rdma-node.env"
RUN_ID=<your-run-id>
df -h "$RUNTIME_HOST_DIR"
du -sh "$RUNTIME_HOST_DIR/ckpts/$RUN_ID"/global_step_*
```

A single complete checkpoint is ~402 GiB. With `save_total_limit=3`, expect ~1.2 TiB
plus safety margin for the model, logs, and Ray temp files.

### 7.16.4 W&B

Use the `${WANDB_ENTITY}` / `${WANDB_PROJECT}` from §7.3 and the `RUN_ID` generated at
launch time. Do not copy historical run URLs.

The online history's latest global step should roughly match the local callback log. If
the run state is `crashed` or the online step stops advancing, check local W&B logs and
network first — do not judge training process state solely from the web UI.

---

## 7.17 Stop and resume

### 7.17.1 Stop

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" pkill -TERM -f '[l]umenrl.trainer.main' || true
```

### 7.17.2 Resume

Before resuming, confirm the latest checkpoint contains the large
`optim_parameter_state_world_size_8_rank_*.pt` files. Then use the same `CKPT_DIR` and set:

```bash
export RESUME_OVERRIDE=true
```

The log must show:

```text
Resuming Ray actor checkpoint from ... global_step_N/actor
Ray resume complete. Next training log will be global_step=N+1
```

After resuming, observe at least two complete steps:

- The first step verifies the checkpoint model loads correctly.
- The second step verifies optimizer updates still produce finite values.

Only when KL, ESS, loss, grad norm, and entropy are all finite for two consecutive steps
is the resume considered successful.

---

## 7.18 Troubleshooting

### `ModuleNotFoundError: vllm`

Ray scheduled the rollout placement group onto the trainer node.

Fix:
- Confirm the YAML has `rollout.topology_tags.node_ip=${ROLLOUT_NODE_IP}`.
- Confirm `actor.topology_tags.node_ip=${TRAIN_NODE_IP}`.
- The log must print both node IPs as detected and written to the YAML.

### RDMA falling back to TCP

```bash
source "$HOME/qwen3-rdma-node.env"
ls -l /dev/infiniband
test -d "/sys/class/infiniband/$RDMA_HCA"
```

Confirm `NCCL_SOCKET_IFNAME=$RDMA_IFACE`, `NCCL_IB_HCA=$RDMA_HCA`,
`NCCL_IB_GID_INDEX=$RDMA_GID_INDEX`. No `NET/IB/.../GDRDMA` in the log means GPU Direct
RDMA is not active.

### flash-attn `varlen_fwd()` argument mismatch

The Python wrapper passes `num_splits` but the native ROCm extension uses the old
21-argument ABI. Use the idempotent ABI patch in `run_grpo.sh` and run the §7.8.8 kernel
test.

### Checkpoint disk full

Symptoms: `RuntimeError: basic_ios::clear: iostream error` / `No space left on device`.

Fix:
- Delete the incomplete current-step directory.
- Keep at least one verified complete checkpoint.
- Confirm `prune_checkpoints` runs on the actor node before saving.
- Do not keep checkpoints that only have model shards without optimizer parameter state.

### NaN after resume (second step)

Usually means the optimizer was not fully restored, or FP32 main parameters were not
synced with the model.

Check:
- `optim_parameter_state_world_size_*.pt` exists and is tens of GiB.
- `load_parameter_state()` was called.
- `reload_model_params()` was called.
- Do not substitute the 2 KiB optimizer metadata file for the distributed parameter state.

### W&B not updating

Check:
- `/runtime/wandb.key` exists and is readable inside the container.
- Local callbacks still output step numbers.
- The W&B run name matches this run's `RUN_ID`.
- On resume, do not let the global step go backward — W&B rejects old steps.

### `Too many open files` / Ray socket EOF

Both container startup and the training script must set:

```bash
ulimit -n 524288
```

If needed, stop Ray on both nodes and restart in the §7.11 order.

---

## 7.19 Key source files

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

## Appendix A: ATOM rollout

This appendix describes the **optional** ATOM rollout alternative. The primary 200-step
deployment above uses vLLM + RCCL/RoCE RDMA.

### A.1 Scope and limitations

ATOM and the current RDMA backend cannot be directly combined:

- `policy.generation_backend: atom` creates `ATOMReplicaManager`.
- The current ATOM manager does not initialize the 9-rank RCCL weight group or RDMA receiver.
- ATOM configs cannot set `weight_sync.backend: rdma`.
- Two-node ATOM must use `shared_folder`; same-node can use `auto`.

`weight_sync.backend: auto` selection rules:

- Same node, ATOM TP=1, 8 replicas matching 8 actor workers: ZMQ CUDA-IPC.
- Same node, ATOM TP=2, 4 replicas (fewer than 8 actor workers): falls back to safetensors.
- ATOM and trainer on different nodes: falls back to safetensors.

Qwen3-30B-A3B currently recommends ATOM TP=2, so the stable path is:

```text
Megatron BF16 shard
  -> actor aggregates and converts to HF names
  -> safetensors export
  -> ATOM TP=2 x 4 replicas reload
```

Do not label this path as RDMA, and do not expect `NET/IB/.../GDRDMA` in the log.

### A.2 Deployment topology

Recommended starting point: single-node co-located smoke.

```text
Single node 8 x MI308X
├── Megatron actor workers: 8
├── ATOM replicas: 4
├── Each ATOM replica: TP=2
├── ATOM weights: BF16
├── KV cache: FP8
└── Sleep mode: level 2
```

The same GPUs alternate between rollout and training:

1. ATOM wakes weights/KV cache.
2. Generates responses and records MoE router logits.
3. ATOM sleeps to level 2, releasing reclaimable memory.
4. Megatron uses R3 distribution replay to compute log-prob and update parameters.
5. Actor exports HF safetensors.
6. ATOM wakes, reloads new parameters, restores KV cache.

A single node hosting ATOM CUDA graphs, Megatron parameters, and the distributed
optimizer simultaneously has significantly higher memory pressure than the two-node
deployment. On OOM, reduce response length, batch size, or ATOM
`gpu_memory_utilization` first — do not disable optimizer parameter-state saving.

### A.3 ATOM source and environment

If using custom ATOM source code, provide the host directory:

```bash
read -r -p "ATOM source dir ATOM_HOST_DIR: " ATOM_HOST_DIR
export ATOM_HOST_DIR
test -f "$ATOM_HOST_DIR/atom/__init__.py"
```

Add to the container creation command:

```bash
-v "$ATOM_HOST_DIR":/workspace/ATOM
```

Without the mount, the container uses its bundled ATOM. With the mount, the launch script
puts `/workspace/ATOM` on `PYTHONPATH`. Verify the actual import location:

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

Expected: `atom.__file__` under `/workspace/ATOM/atom/`, not an unexpected site-packages
version.

ATOM TP>1 requires these environment variables:

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

Set ATOM's public `disable_custom_all_reduce: true` engine option to switch the
TP path from HIP IPC custom all-reduce to RCCL. This RCCL is only for ATOM TP
collectives — not the cross-node RDMA weight broadcast described in the main
guide.

### A.4 ATOM YAML

Base file: `examples/GRPO/configs/grpo_qwen3_30b_a3b_atom_ep8_longrun.yaml`

Key configuration:

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

Notes:

- `atom_cfg` controls the ATOM engine; the current implementation still reads some
  common generation/sleep parameters from `vllm_cfg`.
- `quantization: ""` means BF16 weights, no `per_block_fp8`.
- FP8 is only used for KV cache. Qwen3-30B-A3B's MoE intermediate size is 768; online
  `per_block_fp8` has been observed to produce garbled output.
- `grpo_qwen3_30b_a3b_atom_ep8_smoke.yaml` currently contains
  `generation_backend: vllm` — it cannot be used as an ATOM smoke directly. Copy the
  longrun file and reduce parameters.

ATOM smoke overrides:

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

### A.5 ATOM launch

Prepare the local sync directory:

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
mkdir -p /shared/atomrl_weight_sync/atom-qwen3-30b-a3b
rm -f /shared/atomrl_weight_sync/atom-qwen3-30b-a3b/*.tmp
'
```

Launch with a dedicated ATOM smoke YAML:

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

> `BACKEND=atom` alone does not fix a `CONFIG_OVERRIDE` whose content still says
> `generation_backend: vllm`. The final YAML must explicitly contain
> `policy.generation_backend: atom`.

### A.6 ATOM smoke verification

The startup log should show:

```text
ATOMReplicaManager: launched 4 colocated rollout replicas (atom_tp=2, workers=8)
Ray ATOM rollout ready: 4 colocated replicas
R3Manager: router recording ENABLED
```

With TP=2 x 4, weight sync should show safetensors export/reload logs, not RDMA logs.
Each step must output `callbacks: step=N`.

Check:

```bash
source "$HOME/qwen3-rdma-node.env"
LOG="$RUNTIME_HOST_DIR/logs/qwen3-30b-a3b-atom-smoke3.log"
grep -aE "ATOMReplicaManager|Ray ATOM rollout ready|reloaded weights|callbacks: step=" "$LOG" | tail -30
grep -aiE "Traceback|OutOfMemory|illegal memory access|ca_comm|=nan" "$LOG" | tail
```

Pass criteria:

- 3 consecutive steps complete.
- Reward, loss, entropy, grad norm, KL, and ESS are all finite.
- R3 router-logit coverage is complete.
- ATOM successfully reloads weights after each training round.
- No garbled output, illegal memory access, OOM, or stale CUDA/HIP IPC handles.

After ATOM smoke passes, switch to `grpo_qwen3_30b_a3b_atom_ep8_longrun.yaml` for the
full run. Apply the same §7.15 checkpoint rules — do not resume from incomplete
checkpoints.

---

## References

- AtomRL: `https://github.com/ROCm/ATOM/tree/main/atomrl`
- Lumen: `${WORK_ROOT}/ATOM/atomrl/lumen` (in-tree)
- Megatron-LM (ROCm fork): `https://github.com/ROCm/Megatron-LM/tree/rocm_dev`
- aiter: `https://github.com/ROCm/aiter/tree/adbb6ac8b84110134e0cf4dc513b250d2cc0a75c`
- flash-attention: `https://github.com/ROCm/flash-attention`
- Qwen3-30B-A3B: `https://huggingface.co/Qwen/Qwen3-30B-A3B`
