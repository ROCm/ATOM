# Launch library for accept.sh: one container per role, on the node and GPUs
# the model profile names (<ROLE>_NODE, <ROLE>_GPUS), named
# epd-<ACCEPT_ISSUE>-<role>, running ATOM and ATOMesh from the checkout at
# $ROOT. Every container is removed at exit. Progress goes to stderr, logs to
# $RUN_DIR.

CONTAINERS=()
# Extra docker run options for the next start_container; a topology sets it
# as a local around the launch that needs it.
DOCKER_ARGS=()

log() { echo "[accept] $*" >&2; }

# on_node <node> <command>: run a shell command on a node, over ssh unless it
# is this host.
on_node() {
  local node=$1
  shift
  if [[ $node == "$(hostname -s)" || $node == "$(hostname -f)" ]]; then
    bash -c "$*"
  else
    ssh -o BatchMode=yes "$node" "$*"
  fi
}

role_var() { # role_var <role> <NODE|GPUS>
  local name="${1^^}_$2"
  echo "${!name:-}"
}

cname() { echo "epd-${ACCEPT_ISSUE}-$1"; }

cleanup_containers() {
  local role
  for role in "${CONTAINERS[@]}"; do
    on_node "$(role_var "$role" NODE)" "docker rm -f $(cname "$role") >/dev/null 2>&1" || true
  done
}

# start_container <role>: a detached container idling in the image, with the
# checkout, the profile's mounts and, when the role has GPUs, the GPU devices.
start_container() {
  local role=$1 node gpus name
  node=$(role_var "$role" NODE)
  gpus=$(role_var "$role" GPUS)
  name=$(cname "$role")
  [[ -n $node ]] || { log "no ${role^^}_NODE in the model profile"; return 1; }
  local args=(run -d --name "$name" --network host --ipc host --shm-size 64g
    -e "HF_HOME=$HF_HOME" -e PYTHONUNBUFFERED=1 -w "$ROOT" "${DOCKER_ARGS[@]}")
  local m covered=0
  for m in "${MOUNTS[@]}"; do
    args+=(-v "$m:$m")
    [[ $ROOT/ == "$m"/* ]] && covered=1
  done
  ((covered)) || args+=(-v "$ROOT:$ROOT")
  if [[ -n $gpus ]]; then
    args+=(--device /dev/kfd --device /dev/dri --group-add video -e "HIP_VISIBLE_DEVICES=$gpus")
  fi
  log "starting $name on $node${gpus:+ (GPUs $gpus)}"
  CONTAINERS+=("$role")
  on_node "$node" "$(printf '%q ' docker "${args[@]}" "$IMAGE" sleep infinity) >/dev/null"
}

# in_role <role> [docker exec options --] <command...>: run in the role's container.
in_role() {
  local role=$1
  shift
  local opts=()
  if [[ " $* " == *" -- "* ]]; then
    while [[ $1 != -- ]]; do
      opts+=("$1")
      shift
    done
    shift
  fi
  on_node "$(role_var "$role" NODE)" "$(printf '%q ' docker exec "${opts[@]}" "$(cname "$role")" "$@")"
}

# wait_http <role> <url> <timeout_s> <process pattern>: poll from inside the
# role's container until the url answers; fail early when the server died.
wait_http() {
  local role=$1 url=$2 timeout=$3 pattern=$4 waited=0
  until in_role "$role" curl -sf -o /dev/null "$url"; do
    if ! in_role "$role" pgrep -f "$pattern" >/dev/null; then
      log "$role exited before $url answered"
      return 1
    fi
    if ((waited >= timeout)); then
      log "$role: $url did not answer in ${timeout}s"
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

role_url() { echo "http://$(role_var "$1" NODE):$2"; }

# standalone: whether ATOM_ENV launches ATOM's server through ATOMesh's
# standalone entrypoint (USE_ATOMESH_ENTRYPOINTS=1), which makes it ATOMesh too.
standalone() { [[ " ${ATOM_ENV[*]} " == *" USE_ATOMESH_ENTRYPOINTS=1 "* ]]; }

# launch_atom <role> <server_port> <engine_port> [extra server args...]:
# ATOM's OpenAI server from the checkout, per the profile's ATOM_ENV/ATOM_ARGS.
launch_atom() {
  local role=$1 port=$2 engine_port=$3
  shift 3
  start_container "$role"
  # The image predates the jsonschema dependency in pyproject.toml.
  in_role "$role" pip install -q "jsonschema>=4.25,<5" >&2
  local where
  where=$(in_role "$role" python3 -c 'import atom, os; print(os.path.dirname(atom.__file__))')
  if [[ $where != "$ROOT/atom" ]]; then
    log "atom resolves to $where, not $ROOT/atom"
    return 1
  fi
  local env_opts=() e port_args=(--port "$engine_port")
  for e in "${ATOM_ENV[@]}"; do env_opts+=(-e "$e"); done
  # Under USE_ATOMESH_ENTRYPOINTS=1 the server imports ATOMesh's Python module,
  # which the image does not carry, and reads --port as its HTTP port.
  if standalone; then
    build_mesh "$role"
    in_role "$role" mkdir -p "$MESH_TARGET/pylib"
    in_role "$role" ln -sf ../release/libmesh.so "$MESH_TARGET/pylib/atomesh_runner.so"
    env_opts+=(-e "PYTHONPATH=$MESH_TARGET/pylib")
    port_args=()
  fi
  local cmd
  cmd=$(printf '%q ' python3 -m atom.entrypoints.openai_server "${ATOM_ARGS[@]}" \
    --server-port "$port" "${port_args[@]}" "$@")
  in_role "$role" -d "${env_opts[@]}" -- bash -c "$cmd > $RUN_DIR/$role.log 2>&1"
  log "waiting for $role (log $RUN_DIR/$role.log)"
  wait_http "$role" "http://127.0.0.1:$port/health" 3600 atom.entrypoints.openai_server
}

# build_mesh <role>: build ATOMesh from the checkout in the role's container,
# into MESH_TARGET (one per checkout).
MESH_TARGET=$ACCEPT_DIR/cargo/$(echo -n "$ROOT" | md5sum | cut -c1-12)
build_mesh() {
  local role=$1
  log "building atomesh into $MESH_TARGET (log $RUN_DIR/$role-build.log)"
  if ! in_role "$role" -e "CARGO_TARGET_DIR=$MESH_TARGET" -- bash -c \
    "cargo build --release --manifest-path $ROOT/atom/mesh/Cargo.toml > $RUN_DIR/$role-build.log 2>&1"; then
    log "atomesh build failed"
    return 1
  fi
}

# launch_mesh <role> <port> [atomesh launch args...]: build ATOMesh from the
# checkout and run it, the decision event module at debug, into a kept log.
launch_mesh() {
  local role=$1 port=$2
  shift 2
  start_container "$role"
  build_mesh "$role"
  local cmd
  cmd=$(printf '%q ' "$MESH_TARGET/release/atomesh" launch --host 0.0.0.0 --port "$port" \
    --prometheus-port "$MESH_METRICS_PORT" --model-path "$MODEL_PATH" "$@")
  in_role "$role" -d -e "RUST_LOG=mesh=info,mesh::core::placement::policy_apply=debug" -- \
    bash -c "$cmd > $RUN_DIR/$role.log 2>&1"
  log "waiting for $role (log $RUN_DIR/$role.log)"
  wait_http "$role" "http://127.0.0.1:$port/readiness" 600 "$MESH_TARGET/release/atomesh"
}

# start_client: the container the checks run in, with lmms-eval installed
# from the pinned source CI uses.
start_client() {
  start_container client
  log "installing lmms-eval (log $RUN_DIR/client-install.log)"
  # The shared source tree belongs to another uid than the container's root.
  in_role client -e "LMMS_EVAL_SRC_DIR=$ACCEPT_DIR/lmms-eval" -e GIT_CONFIG_COUNT=1 \
    -e GIT_CONFIG_KEY_0=safe.directory -e "GIT_CONFIG_VALUE_0=$ACCEPT_DIR/lmms-eval" -- bash -c \
    "bash $ROOT/.github/scripts/install_lmms_eval_no_deps.sh > $RUN_DIR/client-install.log 2>&1"
}
