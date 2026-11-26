# Sourced last in every rank env: the hour point's Store owners, an EXAMPLE for nodes like its own (g52 + g27 after prep,
# 2026-11-26 03:46 UTC): 3,392 GiB = 848 GiB per rdma pool, 4 pools. Node A (P + D0): NUMA0 4 x 224 + NUMA1 4 x 272;
# node B (D1 + owners): NUMA0 4 x 144 + NUMA1 4 x 208. The plan depends on each NUMA node's free 2 MiB blocks after the
# prep, so size yours from fresh probes: ../prep/ and ../harness/bin/store_plan.py (see ../README.md).
export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS='0:224:rdma0;0:224:rdma1;0:224:rdma2;0:224:rdma3;1:272:rdma0;1:272:rdma1;1:272:rdma2;1:272:rdma3'
export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS='0:144:rdma0;0:144:rdma1;0:144:rdma2;0:144:rdma3;1:208:rdma0;1:208:rdma1;1:208:rdma2;1:208:rdma3'
