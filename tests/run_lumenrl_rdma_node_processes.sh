#!/usr/bin/env bash
set -u

RANK_BASE=${RDMA_TEST_RANK_BASE:?rank base is required}
RANK_COUNT=${RDMA_NODE_RANK_COUNT:?rank count is required}
LOG_DIR=${RDMA_LOG_DIR:?log directory is required}

pids=()
global_ranks=()
for ((local_rank = 0; local_rank < RANK_COUNT; local_rank++)); do
  global_rank=$((RANK_BASE + local_rank))
  echo "starting global_rank=${global_rank} local_rank=${local_rank} host=$(hostname)"
  SLURM_PROCID=${local_rank} \
  SLURM_LOCALID=${local_rank} \
  SLURM_NTASKS=${RANK_COUNT} \
    python tests/run_lumenrl_rdma_compat.py >"${LOG_DIR}/rank-${global_rank}.log" 2>&1 &
  pids+=("$!")
  global_ranks+=("${global_rank}")
done

rc=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "global_rank=${global_ranks[${index}]} rc=0"
  else
    rank_rc=$?
    echo "global_rank=${global_ranks[${index}]} rc=${rank_rc}"
    rc=1
  fi
done
exit "${rc}"
