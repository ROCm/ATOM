#!/usr/bin/env bash
# agx_teardown.sh <run dir>: stop everything one run started, on every node of the run, and check
# that nothing of it is left. Only this run's own containers (atomesh-<cell>-<jid>-<rank>[-phase])
# and processes (pid files in the run dir, checked against their command line / LOG_ROOT) are
# touched; never another container. Prints per node: containers left, GPUs still holding VRAM.
set -uo pipefail
RUN="${1:?run dir}"
read -r CELL JID HOSTS < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]+"/run.json")); print(d["cell_id"], d["jid"], ",".join(d["hosts"]))' "${RUN}")
touch "${RUN}/STOP"
IFS=, read -r -a hosts <<< "${HOSTS}"
pat="^atomesh-${CELL//./\\.}-${JID}-[0-9]+(-benchmark|-eval)?\$"
for i in "${!hosts[@]}"; do
  h=${hosts[i]}; short=${h#"${AGX_HOST_PREFIX-pit2-p03-}"}
  pidfiles="${RUN}/rank-${i}.pid ${RUN}/nic-${short}.pid"
  [[ "${i}" == 0 ]] && pidfiles+=" ${RUN}/scrape.pid"
  # Containers first: a worker's pd_slurm_job.sh runs its EXIT trap (rank-rc, rm) once its
  # docker run returns. Then the worker, sampler and scraper pids, if still this run's.
  ssh -o BatchMode=yes "${h}" "
    for c in \$(docker ps -a --format '{{.Names}}' | grep -E '${pat}'); do
      timeout 60 docker stop -t 30 \"\$c\" >/dev/null 2>&1; timeout 30 docker rm -f \"\$c\" >/dev/null 2>&1; echo \"  ${short}: removed \$c\"
    done
    sleep 5
    for f in ${pidfiles}; do
      [ -f \"\$f\" ] || continue
      p=\$(cat \"\$f\"); [ -n \"\$p\" ] && [ -d /proc/\$p ] || continue
      if tr '\\0' ' ' < /proc/\$p/cmdline 2>/dev/null | grep -qF '${RUN}' \
         || tr '\\0' '\\n' < /proc/\$p/environ 2>/dev/null | grep -qxF 'LOG_ROOT=${RUN}/logs'; then
        kill \$p 2>/dev/null && echo \"  ${short}: stopped pid \$p (\$(basename \$f))\"
      fi
    done
  " 2>&1
done
sleep 10
for h in "${hosts[@]}"; do
  ssh -o BatchMode=yes "${h}" "
    left=\$(docker ps -a --format '{{.Names}}' | grep -cE '${pat}')
    busy=\$(amd-smi monitor -v 2>/dev/null | awk 'NR>1 && \$3+0 > 2048 {printf \"%s \", \$1}')
    echo \"\$(date -u +%H:%M:%S) ${h}: containers_left=\${left} gpus_with_vram=[\${busy}] all_containers=[\$(docker ps --format '{{.Names}}' | paste -sd, -)]\"
  " 2>&1
done | tee -a "${RUN}/teardown.log"
