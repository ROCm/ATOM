#!/bin/bash
# Launch one milestone job in the p1 container of THIS compute node, as ONE capped gpurun lease (run on the node HOST):
#   ~/.superagent/bin/crsn 034 <<'X'
#   GPUS=4 CAP=2100 bash /shared_nfs/ygan/p1/c17/recipe/launch.sh point C m2_C_c32_1
#   X
# usage: launch.sh point <A|B|C|C-noSP|C-noWIDE> <TAG> [CONC=32]  |  launch.sh gate <same arms> <GTAG>
# env:   GPUS  gpurun request (default 4 = a free NUMA half; or an explicit list, e.g. 4,5,6,7 on 035 whose 4-7 are
#              in the reserved file and are never auto-picked)
#        CAP   hard cap in seconds of the leased command (default 2100 = 35 min; above the 30-min lease rule, so it is
#              explicit; int1_both_c32 took 23 min, gate int1_g1 24 min). timeout sends TERM to the whole job (server
#              and its TP workers included), KILL 300 s later; the job stops its server and checks the GPUs are free.
# Lease output: /shared_nfs/ygan/p1/c17/logs/lease_<TAG>.out ; watch: bash /shared_nfs/ygan/max/bin/gpustat.sh
set -u
JOB=${1:?point|gate}; ARM=${2:?A|B}; TAG=${3:?TAG}; CONC=${4:-32}
RCP=/shared_nfs/ygan/p1/c17/recipe; MS=/shared_nfs/ygan/p1/c17
GPUS=${GPUS:-4}; CAP=${CAP:-2100}
case "$ARM" in A|B|B0|B80|C|C-noSP|C-noWIDE|C2|C3|C4|C4T|C8|C12|C13|C14|C14b|C15|C15-noback|C16|C17|C17b) ;; *) echo "arm must be A, B, B0, B80, C, C-noSP, C-noWIDE, C2, C3, C4, C4T, C8, C12, C13, C14, C14b, C15, C15-noback, C16, C17 or C17b"; exit 2 ;; esac
case "$JOB" in
  point) CMD="bash $RCP/point.sh $ARM $TAG $CONC" ;;
  gate)  CMD="bash $RCP/gate_job.sh $ARM $TAG" ;;
  *) echo "job must be point or gate"; exit 2 ;;
esac
[ -e "$MS/logs/lease_$TAG.out" ] && { echo "lease_$TAG.out exists: pick a new TAG"; exit 2; }
docker exec -d -e GPU_TAG=p1/c17/$TAG p1 bash -c \
  "ulimit -c 0; cd /tmp; bash /shared_nfs/ygan/max/bin/gpurun.sh $GPUS timeout -k 300 $CAP $CMD > $MS/logs/lease_$TAG.out 2>&1" < /dev/null
echo "launched $JOB arm=$ARM tag=$TAG gpus=$GPUS cap=${CAP}s on $(hostname -s); log $MS/logs/lease_$TAG.out"
