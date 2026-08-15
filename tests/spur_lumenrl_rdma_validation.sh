#!/usr/bin/env bash
set -euo pipefail

IMAGE="docker.io/jiaolyu/qwen3-30b-a3b-vllm-rollout:rocm724-20260729-step54"
MOUNTS="/home/jiaolyu/lumenrl-rdma-work/ATOM:/workspace/ATOM,/home/jiaolyu/lumenrl-rdma-work/Lumen-RL:/workspace/Lumen-RL"
NODE_A=${1:?first validation node is required}
NODE_B=${2:?second validation node is required}
NODELIST="${NODE_A},${NODE_B}"

echo "VALIDATION_JOB=${SLURM_JOB_ID}"
echo "VALIDATION_NODES=${SLURM_JOB_NODELIST}"
echo "REQUESTED_NODES=${NODELIST}"

spur run \
  --jobid "${SLURM_JOB_ID}" \
  --overlap \
  --nodes 1 \
  --ntasks 1 \
  --gpus 1 \
  --nodelist "${NODE_A}" \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-workdir /workspace/ATOM \
  /bin/bash -lc \
  'PYTHONPATH=/workspace/ATOM:/workspace/Lumen-RL python -m pytest -q tests/test_rdma_weight_receiver.py tests/test_collective_rpc.py /workspace/Lumen-RL/tests/unit/engine/test_atom_ray_server_rdma.py'

spur run \
  --jobid "${SLURM_JOB_ID}" \
  --overlap \
  --nodes 2 \
  --ntasks 5 \
  --gpus-per-task 1 \
  --distribution block \
  --nodelist "${NODELIST}" \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-workdir /workspace/ATOM \
  /bin/bash -lc \
  "export PYTHONPATH=/workspace/ATOM:/workspace/Lumen-RL; export RDMA_TEST_MASTER_ADDR=${NODE_A}; export NCCL_IB_DISABLE=0; export NCCL_IB_HCA=ionic_0; export NCCL_IB_GID_INDEX=1; export NCCL_SOCKET_IFNAME=ens3; export NCCL_NET_GDR_LEVEL=PHB; export NCCL_DMABUF_ENABLE=1; export NCCL_DEBUG=INFO; export NCCL_DEBUG_SUBSYS=INIT,NET; python tests/run_lumenrl_rdma_compat.py"
