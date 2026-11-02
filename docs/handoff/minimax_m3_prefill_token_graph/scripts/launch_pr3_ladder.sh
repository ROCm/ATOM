#!/bin/bash
# Host: AgentX c2/c10 on the 10-28 stack (ATOM 47ca9958e + aiter fd51011f7) plus aiter #7077/#7079/#7205/#7209,
# same ladder settings as logs/agentic/mi455_m3stack_perf (OFFLOAD=0, no accuracy gate, 3600 s per point).
# Runs in its own container, created from a snapshot of ruili_m3_stack with the two trees swapped.
set -u
R=/home/ruili/code/m3_fp8kv
IMG=ruili/m3stack-env:1030
C=ruili_m3_pr3
TAG=mi455_m3stack_pr7079_7205_7209
ATOM_DIR=$R/ATOM_1028
AITER_DIR=$R/aiter_pr3
log() { echo "$(date '+%F %T') $*"; }

while pgrep -f '[d]ocker commit .*ruili/m3stack-env:1030' >/dev/null; do sleep 10; done
docker image inspect $IMG >/dev/null 2>&1 || { log "image $IMG missing"; exit 1; }
log "image ready: $(docker image inspect -f '{{.Id}}' $IMG | cut -c1-19)"

if ! docker inspect $C >/dev/null 2>&1; then
  docker run -d --name $C --network host --ipc host --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --device /dev/kfd --device /dev/dri --shm-size 64g \
    -v $R/logs:/logs -v /data/models:/models -v $R/scripts:/scripts \
    -v $ATOM_DIR:/app/ATOM -v $AITER_DIR:/app/aiter -v $R:$R \
    -w /app/ATOM $IMG >/dev/null || { log "docker run failed"; exit 1; }
fi
log "container $C up"
docker exec $C bash -c 'env | grep -E "^(PYTHONPATH|HIP_VISIBLE_DEVICES)="; env | grep -iE "nccl|gluon" || echo "no NCCL/GLUON env";
  python3 -c "import aiter, atom; print(aiter.__file__, atom.__file__)" 2>&1 | tail -n 1;
  ls /app/aiter/aiter/jit/*.so | wc -l; ls -d /opt/aiperf /root/.cache/huggingface'

cd $R/scripts/agentic
C=$C TAG=$TAG ROOT=$R ATOM_DIR=$ATOM_DIR AITER_DIR=$AITER_DIR OFFLOAD=0 ACC_FIRST=0 bash run_ladder.sh 2 10
log "ladder exited rc=$?"
