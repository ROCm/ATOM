# ARM B80 = the milestone's arm B (p1/integrated5 @ 5f26263a8, unchanged flags) at gpu-memory-utilization 0.80
# (stable int5 reference: B at 0.85 crashed/stalled in milestone2).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armB.sh"
export MEM=0.80
ARM=B80
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-$NLOC/flydsl_armB}
