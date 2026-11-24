#!/bin/bash
# Run several points back to back on one GPU set of THIS node, each as its OWN capped gpurun lease (run on the node HOST):
#   GPUS=0,1,2,3 CAP=2400 bash /shared_nfs/ygan/p1/c17/recipe/series.sh C2:m3_C2_034_1 C-noSP:m3_CnoSP_034_1
# The chain stops early (no further lease) if a point aborted (logs/abort_<TAG>) or its GPUs were not freed,
# or if /shared_nfs/ygan/p1/c17/STOP exists (budget brake). Log: logs/series_<first TAG>.out
set -u
MS=/shared_nfs/ygan/p1/c17; RCP=$MS/recipe
GPUS=${GPUS:?explicit GPU list}; CAP=${CAP:?explicit cap in seconds (a 3600 s point: CAP=5700, as official-3600)}
for at in "$@"; do t=${at#*:}; [ -e "$MS/logs/lease_$t.out" ] && { echo "lease_$t.out exists"; exit 2; }; done
first=${1#*:}
inner=""
for at in "$@"; do
  a=${at%%:*}; t=${at#*:}
  inner+="[ -e $MS/STOP ] && { echo \"### \$(date -Is) STOP file: not starting $t\"; exit 0; }; "
  inner+="echo \"### \$(date -Is) lease $a $t\"; GPU_TAG=p1/c17/$t bash /shared_nfs/ygan/max/bin/gpurun.sh $GPUS timeout -k 300 $CAP bash $RCP/point.sh $a $t 32 > $MS/logs/lease_$t.out 2>&1; "
  inner+="echo \"### \$(date -Is) $t rc=\$?\"; [ -e $MS/logs/abort_$t ] && { echo '### aborted: chain stopped'; exit 1; }; "
  inner+="grep -q 'GPUS_FREE FAIL' $MS/logs/lease_$t.out && { echo '### GPUs not free: chain stopped'; exit 1; }; "
done
docker exec -d p1 bash -c "ulimit -c 0; cd /tmp; { $inner } > $MS/logs/series_$first.out 2>&1" < /dev/null
echo "launched series $* gpus=$GPUS cap=${CAP}s on $(hostname -s); log $MS/logs/series_$first.out"
