# ARM A (control) = int1_both_c32 EXACTLY as the previous team ran it (sourced).
# Provenance: /shared_nfs/ygan/max/int1/chain2.sh case both_c32 (gate int1_g1 PASSed, so VAR="") ->
#   gpurun.sh 4 bash int1/point.sh int1_both_c32 32 "ATOM_PREFILL_DECODE_INTERVAL=16 ATOM_PREFILL_DELAYER_TARGET_FILL=0.225"
#   point.sh: export the KEY=VALs, source int1/recipe.sh, then export EXTRA ATOM_MONO_STATS_EVERY=200
#   ATOM_ENABLE_METRICS_DEVICE_TIMER=1; serve.sh with CONC=32 (max-num-seqs 64), MEM 0.85, AL 4.04, k=7.
#   Server log: /shared_nfs/ygan/max/logs/server_int1_both_c32.log (2026-11-19 06:30 UTC, 035 GPUs 0-3);
#   result: /shared_nfs/ygan/max/results/int1_both_c32 (54668 tok/s/GPU @ ix_p90 141.1).
# int1_recipe.sh is a byte copy of /shared_nfs/ygan/max/int1/recipe.sh (sha256 6ece7176..., mtime 05:36 UTC,
# i.e. the version that run sourced). Tree: /shared_nfs/ygan/work/max-int1 @ 60870c2b4 (int1).
unset MONO DUTY AK_MOE EXTRA_MORE EXTRA TIER CPU_GB MEM MAX_SEQS AL DSPARK_K
export ATOM_PREFILL_DECODE_INTERVAL=16 ATOM_PREFILL_DELAYER_TARGET_FILL=0.225
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/int1_recipe.sh"
export EXTRA ATOM_MONO_STATS_EVERY=200 ATOM_ENABLE_METRICS_DEVICE_TIMER=1
ARM=A
ARM_REPO_EXPECT=60870c2b4
