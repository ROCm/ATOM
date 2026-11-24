# ARM C8M (milestone5 round 18, DIAGNOSTIC variant, not gated): C8 + two flag-only memory mitigations after C8 died
# (m5_C8_035_1: HSA out of resources, torch reserved 274.4 GiB with TBO pairs vs ~215 plain) and hung (gate needle,
# 23954-token 12K+12K pair):
#  - ATOM_P1_SCHED_BALANCE_TBO_MAX=16384: no pair above the decode-time cap (no 12K+12K, no 22K idle pairs)
#  - MEM 0.75 (gpu-memory-utilization): KV pool 140.45 -> ~126 GB (C3's live KV max is 29-36% of 140 GB), ~14 GiB more
#    headroom for the TBO path's extra ~50-58 GiB of cached activations
#  - ATOM_P1_SP_TBO_HOSTLOG=1: rank-0 log of every SP forward (reserved / alloc retries / pair sizes), no device sync
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC8.sh"
export ATOM_P1_SCHED_BALANCE_TBO_MAX=16384 ATOM_P1_SP_TBO_HOSTLOG=1
export MEM=0.75
