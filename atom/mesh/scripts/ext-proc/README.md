# Envoy + Atomesh + ATOM Engine Quick Test

Run on a Linux host with Docker, Python 3, curl, a local model directory and two
AMD GPUs. Atomesh and Engine share an ATOM image containing `atomesh` built with
the `ext-proc` feature. The startup script uses the image directly.

Run the commands below from the repository root.

## Build the Docker Image

Use the existing release Dockerfile and enable the Mesh `ext-proc` feature:

```bash
ATOM_SOURCE_REPO=https://github.com/ROCm/ATOM.git
ATOM_SOURCE_REF=$(git rev-parse HEAD)
ATOM_IMAGE=atom-extproc:test

docker build -f docker/atom_release.dockerfile --target atom_image \
  --build-arg ATOM_REPO="$ATOM_SOURCE_REPO" \
  --build-arg ATOM_COMMIT="$ATOM_SOURCE_REF" \
  --build-arg ATOM_MESH_FEATURES=ext-proc \
  --ulimit nofile=65536:65536 \
  -t "$ATOM_IMAGE" .
```

Set `ATOM_SOURCE_REPO` to your fork if needed. The Dockerfile clones that repository,
so `ATOM_SOURCE_REF` must exist there; uncommitted local changes are not included.
The build image must provide `protoc` (`protobuf-compiler` on Debian/Ubuntu).

You can also use `-f docker/Dockerfile` with the same arguments. Retain any
base-image or GPU-architecture arguments required by your environment. To skip
AITER kernel precompilation, add `--build-arg PREBUILD_KERNELS=0`; kernels will
compile when needed at runtime.

## Start Services

```bash
ATOM_IMAGE=atom-extproc:test \
MODEL_PATH=/models/Qwen3-0.6B \
GPU_DEVICES=0,1 \
bash atom/mesh/scripts/ext-proc/test_envoy_atom.sh
```

Defaults: Envoy `v1.37.0`, `TP=1`, `DP_SIZE=2`, served model `smoke-model`.
With one GPU, set `DP_SIZE=1 GPU_DEVICES=0`. Send inference requests to Envoy on
port `10016`; Mesh management uses port `10013`.

For other options:

```bash
bash atom/mesh/scripts/ext-proc/test_envoy_atom.sh --help
```

The Envoy template disables automatic request ID generation and preserves client
request IDs. Keep `generate_request_id: false` and
`preserve_external_request_id: true` when adapting the configuration so Mesh can
apply its configured `request_id_headers` priority. The selected ID is forwarded
upstream and returned in the `x-request-id` response header. Header mutations use
`allow_envoy: true` to remove client-supplied `x-envoy-*` headers, and
`disallow_is_error: true` so rejected filtering operations fail the request.

## Send Requests

Non-streaming completion:

```bash
curl --noproxy '*' --max-time 900 http://127.0.0.1:10016/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smoke-model","prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":false}'
```

Streaming completion:

```bash
curl --noproxy '*' -N --max-time 900 http://127.0.0.1:10016/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smoke-model","prompt":"Say hello.","max_tokens":32,"temperature":0,"stream":true}'
```

## View Logs and Clean Up

```bash
docker logs -f <container-name>
```

When finished, run the `docker rm -f ...` command printed by the startup script.
