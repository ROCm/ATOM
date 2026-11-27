# DSV4 on MI308X — GRPO full finetune

**Entry point:** `run_dsv4.sh` — native torchrun + Lumen `get_dsv4_spec` + GRPO policy loss.

Model implementation: `lumen/models/dsv4/`

---

## Training paths

| Profile | Hardware | Description |
|---------|----------|-------------|
| `DSV4_PROFILE=4layer` (default) | Single node 8×MI308X | 4-layer GRPO smoke |
| `DSV4_PROFILE=flash` | 2 nodes, 16 GPUs | 43-layer full-model GRPO |

Load a pretrained checkpoint → update nearly all weights with **GRPO policy loss** (MoE router gate / e-score bias frozen). Default `DEBUG_TRAIN_ONLY=1` (pre-generated `fake_rollout.pt`, no SGLang / Ray).

**Finetune default batch** (same for 4-layer and flash, set by `dsv4_finetune_common.sh`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GBS` | `256` | Must equal rollout sample count (32 prompts × 8) |
| `SEQ_LEN` | `4096` | Megatron `--seq-length` |
| `MBS` | `1` | micro-batch |

Pretrain-only scripts (`run_dsv4_*pretrain*`) still default to `GBS=8`, `SEQ_LEN=2048`, which differs from finetune.

---

## Prerequisites

| Item | Description |
|------|-------------|
| Hardware | 8× MI308X (flash needs 2 nodes × 8 GPUs); host must access `/dev/kfd` |
| Docker | Installed; current user can run containers |
| Base image | `lumen/tests:latest` (used to build the DSV4 image) |
| AIter | `/workspace/aiter` inside the training container; provides DSV4 MHC Triton API |

Default paths are resolved by `dsv4_paths.sh`:

```text
WORKSPACE_ROOT=..
DATA_ROOT=                     # /nfs/data/$USER → /mnt/data/$USER → ${WORKSPACE_ROOT}/dsv4-data
MODEL_DIR=${DATA_ROOT}/models
LOG_DIR=${DATA_ROOT}/logs
BOOTSTRAP_DIR=${DATA_ROOT}/lumen-dsv4-bootstrap
```

---

## Directory layout

```text
examples/dsv4/
├── run_dsv4.sh                         # Host Docker launcher
├── run_dsv4_inner.sh                   # In-container torchrun GRPO
├── launch_dsv4_2node.sh                # Flash 2-node one-shot launch
├── dsv4_megatron_args.sh               # 4layer/flash model args
├── dsv4_flash_mi300x_parallel.sh       # Flash TP4/PP4/EP4 parallelism
├── dsv4_finetune_common.sh             # batch / ckpt / rollout helpers
├── prepare_dsv4_checkpoint.py          # HF → BF16 → torch_dist
├── finetune_dsv4_megatron.py           # GRPO main program
├── patch_megatron_source.py            # SOURCE patch entry (host, no torch required)
├── patch_rocm_megatron_dsv4.py         # Deprecated alias for patch_megatron_source.py
├── PATCHES.md                          # Patch registry quick reference
├── tools/gen_fake_rollout_data.py      # fake_rollout.pt generation
└── …
```

---

## Quick start

### 1. Build the image

```bash
cd ~/ATOM/atomrl/lumen
bash examples/dsv4/build_dsv4_lumen_image.sh
```

### 2. 4-layer GRPO finetune (single node, 8 GPUs)

```bash
# First run: auto-prepare checkpoint + GSM8K + fake rollout
bash examples/dsv4/run_dsv4.sh

# Existing checkpoint + rollout
SKIP_PREPARE=1 DSV4_HC_MULT=4 bash examples/dsv4/run_dsv4.sh
```

Logs: `${LOG_DIR}/lumen_dsv4_4layer_finetune_*.log`

Success indicator:

```text
=== [done] Lumen DSV4 4layer native GRPO finetune completed ===
```

### 3. Flash full-model GRPO finetune (2×8, 16 GPUs)

**Recommended: one-shot head launch**

```bash
MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-host> \
MODEL_DIR=/data1/${USER}/models \
WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models \
SKIP_PREPARE=1 DSV4_HC_MULT=4 \
  bash examples/dsv4/launch_dsv4_2node.sh
```

Defaults: `GBS=256`, `SEQ_LEN=4096`, `NUM_ROLLOUT=10` (see launch script). Put checkpoints on local NVMe per node (`MODEL_DIR` / `WORKER_MODEL_DIR` above); rollout still uses NFS `${DATA_ROOT}/models/fake_rollout.pt`.

**Manual two-node launch**

```bash
# head
NODE_RANK=0 MASTER_ADDR=<head-ip> MODEL_DIR=/data1/${USER}/models \
  SKIP_PREPARE=1 DSV4_HC_MULT=4 DSV4_PROFILE=flash bash examples/dsv4/run_dsv4.sh

# worker (start after head preflight passes)
NODE_RANK=1 MASTER_ADDR=<head-ip> MODEL_DIR=/mnt/nvme0n1/${USER}/models \
  SKIP_PREPARE=1 DSV4_HC_MULT=4 DSV4_PROFILE=flash bash examples/dsv4/run_dsv4.sh
```

Logs: `${LOG_DIR}/lumen_dsv4_flash_finetune_node{0,1}_*.log`

---

## Common environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DSV4_PROFILE` | `4layer` | `4layer` or `flash` |
| `NUM_ROLLOUT` | `10` | GRPO training steps |
| `GBS` | `256` | Must equal rollout sample count |
| `SEQ_LEN` | `4096` | Training sequence length (finetune) |
| `MBS` | `1` | micro-batch |
| `DSV4_HC_MULT` | `4` (4layer) / `4` (flash) | Must match checkpoint dir `_torch_dist_hc{N}` |
| `SKIP_PREPARE` | `0` | `1` = skip HF→torch_dist |
| `DEBUG_TRAIN_ONLY` | `1` | Must be 1 (no SGLang live rollout) |
| `FAKE_ROLLOUT_DATA` | `${DATA_ROOT}/models/fake_rollout.pt` (dual-node) or `/root/models/fake_rollout.pt` | debug-train-only rollout |
| `V4_INDEXER_IMPL` | `aiter` | DSA indexer (aiter triton kernel; `tilelang` is a compat alias only) |
| `V4_SPARSE_MLA_BACKEND` | `triton` | sparse MLA |
| MHC | AIter | Hyper-Connection calls AIter DSV4 fused API directly; no runtime backend switch |
| `OPTIMIZER_OFFLOAD_FRACTION` | `0.75` | Flash full-model CPU Adam offload |

---

## Architecture

```text
run_dsv4.sh (host)
  └─ docker run lumen/dsv4-lumen:mi308x
       └─ run_dsv4_inner.sh
            ├─ setup_container_env.sh
            ├─ prepare_dsv4_checkpoint.py  (optional)
            ├─ gen_fake_rollout_data.prepare()  (rank0)
            └─ torchrun finetune_dsv4_megatron.py
                 └─ atomrl.lumen.models.dsv4.megatron.spec.get_dsv4_spec
```

Multi-node details: [runbook.md](./runbook.md).

`tile_kernels` is only used by FP8 QAT quant kernels in `lumen/ops/dsv4/qat.py`;
it no longer participates in Hyper-Connection.

---

## Megatron patch registry

DSV4 changes to Megatron split into **SOURCE (on-disk)** and **runtime (import/build time)** layers.

- **Full patch catalog (recommended):** [Megatron Patch Registry](../../lumen-docs/docs/source/advance/patch_registry.md)
- **Quick reference:** [PATCHES.md](./PATCHES.md)

Common commands:

```bash
# Host / CI (no PyTorch required)
PYTHONPATH=~/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4
PYTHONPATH=~/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py --list --tag rocm

# Inside container
python3 -m atomrl.lumen.patches --list --tag dsv4
```

Bootstrap script `prepare_rocm_megatron.sh` applies SOURCE patches automatically after clone.
