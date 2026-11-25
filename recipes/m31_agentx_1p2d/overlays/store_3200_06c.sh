# Sourced last in every rank env: the 06c screen's Store owner plan, passed explicitly (CONTROL 06:27): 3,200 GiB =
# 06a's total and 06a's 800 GiB per pool. Node A NUMA1 reached only 748 GiB of free 2 MiB blocks after page-cache
# eviction (memory pressure) + 420 s of THP compaction, so it holds 4 x 176 instead of 06a's 4 x 192, and node B NUMA0
# 4 x 240 instead of 4 x 224 (same per-pool and total capacity).
export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS='0:240:rdma0;0:240:rdma1;0:240:rdma2;0:240:rdma3;1:176:rdma0;1:176:rdma1;1:176:rdma2;1:176:rdma3'
export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS='0:240:rdma0;0:240:rdma1;0:240:rdma2;0:240:rdma3;1:144:rdma0;1:144:rdma1;1:144:rdma2;1:144:rdma3'
