# Sourced last in every rank env: 06a's exact Store owner plan (3,200 GiB; 4 per-NIC pools x 800 GiB), passed explicitly
# instead of the harness's auto plan (CONTROL 06:27). Node A (P + D0): NUMA0 4 x 240 + NUMA1 4 x 192; node B (D1 + owners):
# NUMA0 4 x 224 + NUMA1 4 x 144.
export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS='0:240:rdma0;0:240:rdma1;0:240:rdma2;0:240:rdma3;1:192:rdma0;1:192:rdma1;1:192:rdma2;1:192:rdma3'
export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS='0:224:rdma0;0:224:rdma1;0:224:rdma2;0:224:rdma3;1:144:rdma0;1:144:rdma1;1:144:rdma2;1:144:rdma3'
