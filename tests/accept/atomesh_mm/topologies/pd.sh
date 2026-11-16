# PD: an ATOM prefill server and an ATOM decode server behind ATOMesh's PD
# router, the KV moving from P to D over Mooncake; checked against single.
# The profile places them (PREFILL_NODE/GPUS, DECODE_NODE/GPUS). Each server
# registers its KV on the HCAs local to its GPUs, read from hca/<node>.map,
# and the JSON line carries how many RDMA bytes those HCAs sent and received
# during the check (topology_counters). The run fails when P's HCAs sent no
# RDMA bytes or D's received none: the KV then moved off RDMA, or not at all.

REF_TOPOLOGY=single
# Below the ports Mooncake's TCP transport picks (about 15000-17000); each
# server binds it plus its TP rank.
HANDSHAKE_PORT=${HANDSHAKE_PORT:-6301}

# role_hcas <role>: the HCAs local to the role's GPUs, comma-separated.
role_hcas() {
  local map gpu hca gpus hcas=()
  map=$HERE/hca/$(role_var "$1" NODE).map
  [[ -f $map ]] || { log "no HCA map ${map#"$HERE"/}"; return 1; }
  IFS=, read -ra gpus <<<"$(role_var "$1" GPUS)"
  for gpu in "${gpus[@]}"; do
    hca=$(awk -v gpu="$gpu" '$1 == gpu { print $2 }' "$map")
    [[ -n $hca ]] || { log "${map#"$HERE"/} names no HCA for GPU $gpu"; return 1; }
    hcas+=("$hca")
  done
  (IFS=,; echo "${hcas[*]}")
}

# launch_rdma <role> <port> <engine_port> [server args...]: launch_atom with
# Mooncake on the role's HCAs, its container given the node's RDMA devices and
# the host's ionic verbs provider, as .github/workflows/atomesh-benchmark.yaml
# runs PD. MC_DISABLE_HIP=1 keeps Mooncake off its intra-node HIP transport,
# which garbles M3.1's output.
launch_rdma() {
  local role=$1 hcas mounts m
  hcas=$(role_hcas "$role")
  log "$role registers its KV on $hcas"
  mapfile -t mounts < <(on_node "$(role_var "$role" NODE)" '
    lib=/usr/lib/x86_64-linux-gnu/libionic.so.1
    [[ ! -e $lib ]] || echo "$(readlink -f $lib):$lib:ro"
    for f in /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so /etc/libibverbs.d/ionic.driver; do
      [[ ! -e $f ]] || echo "$f:$f:ro"
    done')
  local docker_args=(--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1)
  for m in "${mounts[@]}"; do docker_args+=(-v "$m"); done
  local env=("${ATOM_ENV[@]}" MC_DISABLE_HIP=1 MC_GID_INDEX=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1
    "ATOM_MOONCAKE_IB_DEVICE=$hcas")
  local DOCKER_ARGS=("${docker_args[@]}") ATOM_ENV=("${env[@]}")
  launch_atom "$@"
}

# launch_pd <role> <kv_role> <port> <engine_port>: an ATOM server with the
# Mooncake KV connector.
launch_pd() {
  launch_rdma "$1" "$3" "$4" --kv-transfer-config \
    "{\"kv_connector\": \"mooncake\", \"kv_role\": \"$2\", \"handshake_port\": $HANDSHAKE_PORT}"
}

topology_up() {
  launch_pd prefill kv_producer "$PREFILL_PORT" "$PREFILL_ENGINE_PORT"
  launch_pd decode kv_consumer "$DECODE_PORT" "$DECODE_ENGINE_PORT"
  launch_mesh mesh "$MESH_PORT" --backend atom --pd-disaggregation --policy cache_aware \
    --request-timeout-secs 14400 --prefill "$(role_url prefill "$PREFILL_PORT")" \
    --decode "$(role_url decode "$DECODE_PORT")"
  ENDPOINT=$(role_url mesh "$MESH_PORT")
}

# hca_bytes <role> <rx|tx>: RDMA bytes the role's HCAs have received or sent.
# Fails, rather than summing what it got, when a counter cannot be read: a
# short sum reads as KV that never crossed RDMA.
hca_bytes() {
  local hcas hca paths=() values sum=0 v
  hcas=$(role_hcas "$1") || return 1
  IFS=, read -ra hcas <<<"$hcas"
  for hca in "${hcas[@]}"; do
    paths+=("/sys/class/infiniband/$hca/ports/1/hw_counters/$2_rdma_ucast_bytes")
  done
  values=($(on_node "$(role_var "$1" NODE)" "cat ${paths[*]}")) || {
    log "$1: cannot read the $2 counters of $(IFS=,; echo "${hcas[*]}")"
    return 1
  }
  if ((${#values[@]} != ${#hcas[@]})); then
    log "$1: ${#values[@]} $2 counters for ${#hcas[@]} HCAs"
    return 1
  fi
  for v in "${values[@]}"; do
    sum=$((sum + v))
  done
  echo "$sum"
}

# topology_counters: one JSON object of counters, for each of HCA_ROLES;
# accept.sh reads it before and after the check and adds the differences to
# the check's JSON line.
HCA_ROLES=(prefill decode)
topology_counters() {
  local role dir bytes sep="" out="{"
  for role in "${HCA_ROLES[@]}"; do
    for dir in tx rx; do
      bytes=$(hca_bytes "$role" "$dir") || return 1
      out+="$sep\"${role}_hca_${dir}_bytes\": $bytes"
      sep=", "
    done
  done
  echo "$out}"
}

# kv_failures <counter deltas>: a JSON list of the reasons the counters show
# the KV off RDMA.
kv_failures() {
  python3 - "$1" <<'EOF'
import json
import sys

deltas = json.loads(sys.argv[1])
failures = []
if not deltas["prefill_hca_tx_bytes"]:
    failures.append("P's HCAs sent no RDMA bytes: the KV left RDMA")
if not deltas["decode_hca_rx_bytes"]:
    failures.append("D's HCAs received no RDMA bytes: the KV left RDMA")
print(json.dumps(failures))
EOF
}

# topology_findings <counter deltas>: the run's failures.
topology_findings() {
  echo "{\"failures\": $(kv_failures "$1")}"
}
