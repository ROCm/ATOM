> [Examples README](../README.md) > Multi-node RDMA

# 5. Multi-node RDMA Verification

This document covers testing and verification procedures specific to **multi-node
deployments with RDMA weight transfer** (e.g. two-node Qwen3-30B-A3B with Megatron
training on node 1 and vLLM rollout on node 2, connected via RoCE RDMA).

For single-node setups (examples 1-7 on a single 8-GPU machine), see
[Launching](04-launching.md).

---

## 5.1 RDMA / network pre-checks

Run on **both nodes**, inside the running container. These checks confirm the RDMA device
is visible and correctly configured.

### Device presence

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

### NCCL/RCCL environment variables

These must be set in the training launch environment:

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

### Log confirmation

RDMA is only verified when the training log contains **both** of these lines:

```text
NCCL INFO Using network IB
NCCL INFO ... via NET/IB/0/GDRDMA ...
```

If only socket/TCP logs appear, RDMA is not active — check the HCA name, GID index, and
whether the container has `/dev/infiniband` mounted.

---

## 5.2 Launch verification

After starting a multi-node training run, inspect the log for the following sequence.

### Startup sequence

These lines must appear in order:

```text
Created 1 placement groups for pool 'rollout' ... node_ip=${ROLLOUT_NODE_IP}
Created 1 placement groups for pool 'actor' ... node_ip=${TRAIN_NODE_IP}
RDMA preflight: ...
RDMA weight group ready: ... world=9
NCCL INFO Using network IB
NCCL INFO ... NET/IB/0/GDRDMA ...
```

The `world=9` confirms the 9-rank process group (1 Megatron sender + 8 vLLM TP receivers).
The `GDRDMA` line confirms GPU Direct RDMA is active.

### Per-step confirmation

Every training step must emit both:

```text
RDMA weight sync committed:
callbacks: step=N
```

### Key metrics to monitor

```text
rollout_corr/kl
rollout_corr/rollout_is_eff_sample_size
actor/loss
actor/grad_norm
actor/entropy
weight_sync/gbps
timing/weight_sync_rdma_s
```

**If any core metric is `nan`, stop the run immediately. Do not write the next checkpoint
on a NaN run.**

---

## 5.3 Ray cluster verification

After starting both Ray nodes, confirm the cluster is healthy:

```bash
docker exec "$CONTAINER" ray status
```

Must show:

```text
Active: 2 nodes
Total: 16 GPU
```

---

## 5.4 Checkpoint verification (Megatron distributed)

A complete Megatron distributed-optimizer checkpoint for Qwen3-30B-A3B requires exactly:

- 8 model shards
- 8 optimizer metadata shards
- 8 extra-state shards
- 8 large optimizer parameter-state shards (each ~41-45 GiB)
- On the controller side: `checkpoint_N.pt` and `latest_checkpointed_iteration.txt`

### File listing

On the trainer node, after step 5 has saved:

```bash
RUN_ID=<your-run-id>
P="$RUNTIME_HOST_DIR/ckpts/$RUN_ID/global_step_5/actor"
ls -lh "$P"/model_world_size_8_rank_*.pt
ls -lh "$P"/optim_world_size_8_rank_*.pt
ls -lh "$P"/optim_parameter_state_world_size_8_rank_*.pt
ls -lh "$P"/extra_state_world_size_8_rank_*.pt
```

### Automated count check

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

### Checkpoint corruption history

The v1 run ran out of disk space during checkpoint saving. The old code only saved
`optimizer.state_dict()` metadata (~2 KiB), not the Megatron distributed optimizer's
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

---

## 5.5 Verified baselines

These numbers are from the verified RDMA smoke run and serve only as a regression
baseline for new deployments — they are not configuration values.

**RDMA smoke (3 consecutive steps):**

| Metric | Value |
|---|---|
| Broadcast per step | 61.1 GB, 58 buckets |
| Sync duration | 2.51-3.90 seconds |
| Effective throughput | 134-215 Gb/s |
| `rollout_corr/kl` | 0.0008425, 0.0008446, 0.0007718 |
| ESS | 0.998361, 0.998328, 0.998397 |
| Dynamic weight coverage | verified on every TP worker |

**Baseline checkpoint size:** ~402 GiB total (8 model + 8 optim-metadata + 8 extra-state
+ 8 optim-parameter-state shards at ~41-45 GiB each).
