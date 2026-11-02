#!/bin/bash
# prep_container.sh: create the server container if it does not exist, start it, and check it imports the mounted
# ATOM and aiter. Safe to re-run. Not --pid=host on purpose: down.sh's pkill then reaches this container's processes
# only, never another user's.
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
mkdir -p "$WORK/logs"
for d in "$ATOM_DIR" "$AITER_DIR" "$MODEL_PATH" "$DRAFT_PATH"; do
  [ -d "$d" ] || { echo "missing $d (config.env)" >&2; exit 1; }
done
if ! docker inspect "$CONTAINER" > /dev/null 2>&1; then
  # shellcheck disable=SC2086
  docker run -d --name "$CONTAINER" --network host --ipc host --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --device /dev/kfd --device /dev/dri --shm-size 64g \
    -e PYTHONPATH=/app/aiter:/app/ATOM \
    -v "$MODEL_PATH":"/models/$MODEL":ro -v "$DRAFT_PATH":"/models/$DRAFT":ro -v "$WORK/logs":/logs -v "$PWD":/ep:ro \
    -v "$ATOM_DIR":/app/ATOM -v "$AITER_DIR":/app/aiter $EXTRA_DOCKER_ARGS \
    -w /app/ATOM --entrypoint sleep "$IMAGE" infinity > /dev/null
  echo "created $CONTAINER from $IMAGE"
fi
docker start "$CONTAINER" > /dev/null
docker exec "$CONTAINER" python3 -c "import aiter, atom; print(aiter.__file__, atom.__file__)" 2>&1 | tail -n 1
echo "CONTAINER READY: $CONTAINER"
