# DSV4 Flash Full-Model Runbook (2-node MI308X)

Operations guide for Lumen native GRPO full finetune (43 layers) on a **head + worker** dual-node 16-GPU cluster. Script details: [README.md](./README.md).

---

## 1. Cluster and environment

| Item | Description |
|------|-------------|
| Head | `NODE_RANK=0`, IP = `${MASTER_ADDR}` |
| Worker | `NODE_RANK=1`, SSH = `${WORKER_SSH}` |
| Docker image | `lumen/dsv4-lumen:mi308x` |
| Parallelism | TP=4, PP=4, EP=4 (11+11+11+10 layers) |
| Network | Set `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` per cluster (MI308X Banff often uses `ens14np0`) |
| NCCL workaround | `NCCL_IB_GDR_LEVEL=0`, `NCCL_NET_GDR_LEVEL=LOC`, `MEGATRON_NO_BATCH_P2P_COMM=1` |

**On worker:** AIter under `${LUMEN_DIR}` and `/workspace/aiter` must match head.

### 1.1 Local checkpoints (recommended; default launch example)

Head/worker use checkpoints on local NVMe (**do not** load the 532GB dist ckpt from NFS — init will be extremely slow):

| Node | Launch variable | Example |
|------|-----------------|---------|
| head | `MODEL_DIR` | `/data1/${USER}/models` |
| worker | `WORKER_MODEL_DIR` | `/mnt/nvme0n1/${USER}/models` |

Both map inside the container to `/root/models/${MODEL_NAME}_torch_dist`.

---

## 2. Paths (NFS + local)

Defaults from `dsv4_paths.sh` (`DATA_ROOT` auto-detects `/nfs/data/${USER}`, etc.):

| Purpose | Host path | Container path |
|---------|-----------|----------------|
| Model dir (**local NVMe recommended**) | head: `/data1/${USER}/models`<br>worker: `/mnt/nvme0n1/${USER}/models` | `/root/models` |
| Dataset | `${DATA_DIR}` | `/root/datasets` |
| Logs | `${LOG_DIR}` | — |
| Lumen code | `${LUMEN_DIR}` | `/workspace/Lumen` |
| **Shared rollout (dual-node, NFS)** | `${DATA_ROOT}/models/fake_rollout.pt` | Same path (`DATA_ROOT` bind-mount) |

If `MODEL_DIR` is unset, `dsv4_paths.sh` falls back to `${DATA_ROOT}/models` (NFS) — fine for small models or existing ckpts; **for 43L flash finetune always set local `MODEL_DIR` / `WORKER_MODEL_DIR`**.

### 2.1 Pretrained checkpoint (finetune **load**)

| Path | Description |
|------|-------------|
| `${MODEL_DIR}/${MODEL_NAME}_torch_dist` | Primary checkpoint (torch_dist) |
| `${MODEL_DIR}/${MODEL_NAME}_torch_dist_hc${DSV4_HC_MULT}` | fallback |

Verify: `latest_checkpointed_iteration.txt` exists.

### 2.2 GRPO rollout data

| Path | Description |
|------|-------------|
| `${DATA_ROOT}/models/fake_rollout.pt` | debug-train-only fake rollout (shared dual-node) |

Dual-node: rank0 generates/reuses on **NFS `${DATA_ROOT}/models/`**; worker reads the same NFS rollout even if ckpt is on local NVMe. Skipped if already present.

Single-node default: `/root/models/fake_rollout.pt`.

### 2.3 Finetune **output** checkpoint

**Not saved by default.** Scripts omit `--save`; `--save-interval 1000000` effectively disables saving.

To save finetune weights, add to `DSV4_FINETUNE_TORCHRUN_ARGS` in `dsv4_finetune_common.sh`, e.g.:

```bash
--save /root/models/DeepSeek-V4-Flash-FP8-finetune_torch_dist
--save-interval 1
```

---

## 3. Pre-launch checks

### 3.1 Sync code to worker

On **head**:

```bash
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.nfs*' \
  "${LUMEN_DIR}/" "${WORKER_SSH}:${LUMEN_DIR}/" \
  -e "ssh ${SSH_KEY:+-i ${SSH_KEY} -o IdentitiesOnly=yes} -o BatchMode=yes"
```

### 3.2 Confirm worker 8 GPUs are idle

```bash
ssh ${SSH_KEY:+-i ${SSH_KEY} -o IdentitiesOnly=yes} -o BatchMode=yes "${WORKER_SSH}" \
  'for i in 0 1 2 3 4 5 6 7; do echo -n "GPU$i: "; \
   rocm-smi -d $i --showmeminfo vram 2>/dev/null | grep Used; done'
```

### 3.3 Clean old containers

```bash
docker rm -f lumen-dsv4-flash-finetune-node0 lumen-dsv4-flash-finetune-node1 2>/dev/null || true
ssh ${SSH_KEY:+-i ${SSH_KEY} -o IdentitiesOnly=yes} -o BatchMode=yes "${WORKER_SSH}" \
  'docker rm -f lumen-dsv4-flash-finetune-node0 lumen-dsv4-flash-finetune-node1 2>/dev/null || true'
```

---

## 4. Launch: DSV4 43-layer GRPO full finetune

On **head** (recommended one-shot dual-node):

```bash
cd "${LUMEN_DIR}"

MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-ip> \
MODEL_DIR=/data1/${USER}/models \
WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models \
DATA_ROOT=/nfs/data/${USER} \
SKIP_PREPARE=1 DSV4_HC_MULT=4 \
V4_INDEXER_IMPL=aiter V4_SPARSE_MLA_BACKEND=triton \
OPTIMIZER_OFFLOAD_FRACTION=0.75 \
NCCL_IB_GDR_LEVEL=0 NCCL_NET_GDR_LEVEL=LOC MEGATRON_NO_BATCH_P2P_COMM=1 \
HSA_OVERRIDE_GFX_VERSION=9.4.2 NCCL_SOCKET_IFNAME=ens14np0 GLOO_SOCKET_IFNAME=ens14np0 \
IMAGE=lumen/dsv4-lumen:mi308x \
bash examples/dsv4/launch_dsv4_2node.sh
```

Default finetune batch: **`GBS=256`**, **`SEQ_LEN=4096`**, **`NUM_ROLLOUT=10`** (`launch_dsv4_2node.sh` / `dsv4_finetune_common.sh`).

**Optional bisect smoke** (connectivity / PP debug; not default production config):

```bash
GBS=8 DSV4_KEEP_GBS=1 SEQ_LEN=512 DSV4_KEEP_SEQ_LEN=1 NUM_ROLLOUT=2 \
ROLLOUT_N_PROMPTS=1 ROLLOUT_N_PER_PROMPT=8 SMOKE_LEGACY_FAKE_ROLLOUT=1 \
# …same remaining env as above (MODEL_DIR / WORKER_MODEL_DIR)…
bash examples/dsv4/launch_dsv4_2node.sh
```

### 4.1 Key parameters

| Variable | Default / recommended | Description |
|----------|----------------------|-------------|
| `GBS` | **`256`** | Must equal rollout sample count (32×8) |
| `SEQ_LEN` | **`4096`** | Megatron training sequence length |
| `MBS` | `1` | micro-batch |
| `NUM_ROLLOUT` | `10` | GRPO rollout / train iters |
| `DSV4_HC_MULT` | `4` | MHC multiplier |
| `SKIP_PREPARE` | `1` (launch) | Skip HF→torch_dist conversion |
| `MODEL_DIR` | **`/data1/${USER}/models` (head)** | Local NVMe ckpt; do not load full model from NFS |
| `WORKER_MODEL_DIR` | **`/mnt/nvme0n1/${USER}/models`** | Worker local ckpt |
| `DATA_ROOT` | `/nfs/data/${USER}` | Shared rollout path |
| `V4_INDEXER_IMPL` | `aiter` | DSA indexer (aiter triton kernel) |
| `V4_SPARSE_MLA_BACKEND` | `triton` | sparse MLA |
| MHC | AIter | Uses AIter DSV4 fused API directly |
| `MEGATRON_NO_BATCH_P2P_COMM` | `1` | Avoid PP P2P hang |
| `NCCL_IB_GDR_LEVEL` | `0` | IB GDR workaround |
| `NCCL_NET_GDR_LEVEL` | `LOC` | Works with above |
| `OPTIMIZER_OFFLOAD_FRACTION` | `0.75` | CPU Adam offload |
| `SSH_KEY` | — | Worker SSH private key (optional; default ssh-agent key if unset) |

### 4.2 Manual single-node launch (fallback)

Head:

```bash
cd "${LUMEN_DIR}"
NODE_RANK=0 MASTER_ADDR=<head-ip> \
MODEL_DIR=/data1/${USER}/models \
SKIP_PREPARE=1 DSV4_HC_MULT=4 \
V4_INDEXER_IMPL=aiter V4_SPARSE_MLA_BACKEND=triton \
OPTIMIZER_OFFLOAD_FRACTION=0.75 \
NCCL_IB_GDR_LEVEL=0 NCCL_NET_GDR_LEVEL=LOC MEGATRON_NO_BATCH_P2P_COMM=1 \
HSA_OVERRIDE_GFX_VERSION=9.4.2 NCCL_SOCKET_IFNAME=ens14np0 GLOO_SOCKET_IFNAME=ens14np0 \
IMAGE=lumen/dsv4-lumen:mi308x \
DSV4_PROFILE=flash bash examples/dsv4/run_dsv4.sh
```

Worker:

```bash
cd "${LUMEN_DIR}"
NODE_RANK=1 MASTER_ADDR=<head-ip> \
MODEL_DIR=/mnt/nvme0n1/${USER}/models \
# same remaining env as head (GBS/SEQ/NCCL/…)
DSV4_PROFILE=flash bash examples/dsv4/run_dsv4.sh
```

---

## 5. Logs and monitoring

| Log | Path |
|-----|------|
| Head training | `${LOG_DIR}/lumen_dsv4_flash_finetune_node0_*.log` |
| Worker training | `${LOG_DIR}/lumen_dsv4_flash_finetune_node1_*.log` |
| Launch head | `${LOG_DIR}/lumen_dsv4_flash_finetune_launch_head_*.log` |
| Launch worker | `${LOG_DIR}/lumen_dsv4_flash_finetune_launch_worker_*.log` |
| Preflight | `${LOG_DIR}/.dsv4_preflight/runs/<PREFLIGHT_ID>/` |

```bash
tail -f "${LOG_DIR}"/lumen_dsv4_flash_finetune_node0_*.log
docker ps --filter name=lumen-dsv4-flash-finetune
```

### 5.1 Training success indicators

- `rollout/num_samples`, `train/loss`, `perf/actor_train_time`
- Completion: `=== [done] Lumen DSV4 flash native GRPO finetune completed ===`

### 5.2 Normal but slow phases

- Checkpoint load: `q_norm/kv_norm ... will skip` (requires `LUMEN_DSV4_SKIP_OPTIONAL_NORMS=1`, on by default)
- Optimizer CPU offload init: high host memory; may see no new log lines for tens of minutes

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Checkpoint load extremely slow | Confirm `MODEL_DIR` / `WORKER_MODEL_DIR` point to local NVMe, not `${DATA_ROOT}/models` (NFS) |
| `TRAIN_ITERS: unbound variable` | Use latest `preflight_dsv4_flash_multinode.sh` |
| Missing rollout data | Confirm dual-node `FAKE_ROLLOUT_DATA=${DATA_ROOT}/models/fake_rollout.pt`; or `SMOKE_LEGACY_FAKE_ROLLOUT=1` |
| Worker rank OOM | Worker GPUs occupied by other processes; check `rocm-smi --showpids` and free them |
| NCCL hang / config mismatch | Launch both nodes from same `launch_dsv4_2node.sh`; preflight validates GBS/TP/NCCL |
| `NET/IB : Unable to open device mlx5_*` | WARN when IB unavailable; may fall back to socket |
| Worker unreachable | Check SSH key, `WORKER_SSH`, reservation / security group |

---

## 7. Related scripts

```text
examples/dsv4/
├── runbook.md                              # This document
├── launch_dsv4_2node.sh                    # Dual-node finetune one-shot launch (recommended)
├── run_dsv4.sh                             # Single-rank launcher
├── run_dsv4_inner.sh                       # In-container torchrun GRPO
├── finetune_dsv4_megatron.py               # Python entry point
├── dsv4_finetune_common.sh                 # batch / ckpt / rollout helpers
├── tools/gen_fake_rollout_data.py          # fake_rollout.pt
├── preflight_dsv4_flash_multinode.sh       # Dual-node config validation
└── dsv4_paths.sh                           # MODEL_DIR / LOG_DIR / etc.
```
