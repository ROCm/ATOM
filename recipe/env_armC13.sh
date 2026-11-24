# ARM C13 = C12 + KV capacity (sourced). Same tree as C12 (p1-prefix-cache @ 9bfa1c6c4).
# 1. everything arm C12 sets
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC12.sh"
# 2. capacity: gpu-memory-utilization 0.80 -> 0.85 = +14.4 GiB of KV = ~91.0K blocks (82.5K at 0.80; vLLM 90.6K).
#    Safety: the official-3600 C3 runs grew device VRAM only in intervals that ran prefill steps > 16384 tokens
#    (warm-up storm, idle bursts: 324 of 4903 batches, all 32768/29K/22K); intervals with only <= 16384-token steps
#    (incl. 700-815K-token contexts) never grew it. ATOM_P1_SCHED_IDLE_PREFILL_CAP=16384 caps the steps taken while no
#    decode runs (decode steps are already capped at 16384 by ATOM_SCHED_DECODE_PREFILL_CAP), so no step exceeds
#    16384 tokens; max-num-batched-tokens stays 32768 (profiling forward, delayer fill, chunk geometry unchanged).
#    Projected device peak <= 224.9 + 14.5 + 23.7 = ~263 GiB of 288 (C3 at 0.80: 261.5). UNVALIDATED on a GPU.
export MEM=0.85 ATOM_P1_SCHED_IDLE_PREFILL_CAP=16384
ARM=C13
ARM_REPO_EXPECT=9bfa1c6c4
export FLYDSL_RUNTIME_CACHE_DIR=$NLOC/flydsl_armC12
FLYDSL_SEED=$NLOC/flydsl_armC3
