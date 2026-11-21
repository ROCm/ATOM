#!/usr/bin/env bash
# Start the container used by every later step: vllm/vllm-openai-rocm:v0.27.0 with $WORK mounted at the same path.
# Usage (host): WORK=/path/to/work bash launch_container.sh
set -euo pipefail
: "${WORK:?set WORK to a directory with >= 1 TB free}"
NAME=${CONTAINER_NAME:-qwen-image-rl}
IMAGE=${IMAGE:-vllm/vllm-openai-rocm:v0.27.0}

docker run -d --init --name "$NAME" \
  --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$WORK:$WORK" -e WORK="$WORK" -e HF_HOME="$WORK/hf_home" -w "$WORK" \
  --entrypoint "" "$IMAGE" sleep infinity
docker inspect -f 'started {{.Name}} image={{.Config.Image}}' "$NAME"
