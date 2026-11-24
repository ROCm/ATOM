# ARM C17b = C17 with MEM raised just enough to hold the KV pool at C16's ~90.96K blocks (sourced). Same tree as C17.
# C16 (MEM 0.85): budget 244.79 GiB, peak_torch 72.26, available_for_kv 154.84 GiB, 1,827,840 B/block -> 90957 blocks.
# sp-local-shared raised peak_torch by +2.74 GiB in C11 vs C3 (74.96 vs 72.22: +3.1 resident, -0.38 activations), so
# Predicted: C17 needs +2.74 GiB of budget (+0.0095 MEM). MEASURED (C17 smoke c17_C17_smoke900_034): peak_torch 75.00,
# available_for_kv 152.23 GiB, 89425 blocks -> +2.61 GiB = +2.61/287.98 = +0.00906 MEM -> 0.8591 (-> ~90.96K blocks).
# Expected device VRAM peak = C17's + ~2.7 GiB (C16 276.0 -> ~278.7 GiB if C17 matches C16), below the 282 GiB rule.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC17.sh"
export MEM=0.8591
ARM=C17b
export FLYDSL_RUNTIME_CACHE_DIR=$NLOC/flydsl_armC17b
FLYDSL_SEED="$NLOC/flydsl_armC17 $FLYDSL_SEED"
