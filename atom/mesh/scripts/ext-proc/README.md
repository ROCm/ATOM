# Envoy + Atomesh + ATOM Engine Manual Testing

Run three containers on a single Linux host, with Atomesh and Engine sharing the
same `ATOM_IMAGE` (default: `atom-extproc:test`). Override `ATOM_IMAGE` to use your
own existing image. Envoy uses `ENVOY_IMAGE` (default: `envoyproxy/envoy:v1.37.0`).
The startup script uses these images directly without rebuilding or compiling.
Engine uses the small dense model `Qwen/Qwen3-0.6B` with
`TP=1, DP=2`, placing each replica on one AMD GPU. This is the smallest topology
that exercises multiple DP ranks; DP here means data parallelism.

```text
Test client ──HTTP :10016──> Envoy v1.37.0 ──HTTP :10010──> ATOM Engine
                              │                              ├─ DP rank 0
                              │ ext-proc gRPC :10014          └─ DP rank 1
                              ▼
                            Atomesh
                              └─ HTTP :10013: management and health checks
```

All three containers use host networking, with services listening on `127.0.0.1`.
The script reuses
[`tests/fixtures/ext-proc/envoy.yaml`](../../tests/fixtures/ext-proc/envoy.yaml),
adjusting the ports and inference timeouts while preserving the `FULL_DUPLEX_STREAMED` and
`ORIGINAL_DST` configuration.

## Prerequisites

- Install Docker, Python 3 and curl on the host, and make two AMD GPUs available.
- Prepare a complete local Qwen3-0.6B model directory containing `config.json`,
  weights and tokenizer files. All files and symlink targets must be accessible
  inside the container after mounting.
- `ATOM_IMAGE` must contain `/usr/local/bin/atomesh` with the current ext-proc
  implementation and an ATOM Engine supporting `--data-parallel-size`,
  `--served-model-name` and `/server_info`. Before loading the model, the script
  checks that `atomesh launch --help` includes `--ext-proc`.

## Start the Services

Run from the repository root using the default images:

```bash
MODEL_PATH=/models/Qwen3-0.6B \
GPU_DEVICES=0,1 \
bash atom/mesh/scripts/ext-proc/test_envoy_atom.sh
```

To select other existing images, set the environment variables when starting:

```bash
ATOM_IMAGE=my-registry/atom:extproc \
ENVOY_IMAGE=envoyproxy/envoy:v1.37.0 \
MODEL_PATH=/models/Qwen3-0.6B \
bash atom/mesh/scripts/ext-proc/test_envoy_atom.sh
```

Changing these variables does not require rebuilding an image. The selected
ATOM image must already support ext-proc as described above.

The script starts the three containers and waits for service readiness. It
leaves them running for you to test and prints their cleanup command.
It does not send inference requests or validate generated responses.

`GPU_DEVICES` is passed to Engine through `ROCR_VISIBLE_DEVICES`. Defaults are
`DP_SIZE=2`, `TP=1`, BF16 KV cache, eager execution, a context length of 1024,
at most 4 sequences, and a GPU memory utilization fraction of 0.85. The model is
mounted at `/model` inside the container and served as `smoke-model`.
Engine uses the original Python HTTP entrypoint with `USE_ATOMESH_ENTRYPOINTS=0`
explicitly set. Atomesh uses
`--backend atom --dp-aware --policy round_robin --ext-proc`.

The script passes `ATOM_DP_LM_HEAD_MODE=default` to Engine, using a replicated
LM head on each DP rank. Some Engine images enable `all2all` by default; their
DP-sharded LM head can stall when one rank prefills and another runs a dummy
decode. The replicated mode retains TP=1 / DP=2 and avoids the LM head's
cross-DP collectives. Set `ATOM_DP_LM_HEAD_MODE=all2all` or `allgather` explicitly
only when testing that optimization with a compatible Engine image.

To preview the commands and generate the Envoy configuration without starting
containers:

```bash
MODEL_PATH=/models/Qwen3-0.6B \
bash atom/mesh/scripts/ext-proc/test_envoy_atom.sh --dry-run
```

Use `--help` to see all options. With only one GPU, set `DP_SIZE=1 GPU_DEVICES=0`.
This checks the path through a single replica; it does not exercise multiple DP ranks.

### Cold Inference and Larger Models

Engine readiness and startup warmup do not cover every inference kernel shape.
For models such as Qwen3.5-27B-FP8, the first real request can trigger Triton
autotuning and AITER compilation and take more than 120 seconds.
`REQUEST_TIMEOUT` sets the inference wait budget used to configure Mesh and Envoy
(default: 900s). Their idle timeouts are at least this value plus 30 seconds, and
the Envoy route timeout is extended when needed. Configure your client's timeout
separately; the manual commands below use `curl --max-time 900`.
`WAIT_TIMEOUT` controls service startup only.

For example, from this directory:

```bash
MODEL_PATH="$HOME/workspace/models/Qwen3.5-27B-FP8" \
GPU_DEVICES=0,1 GPU_MEMORY_UTIL=0.85 \
ATOM_DP_LM_HEAD_MODE=default \
REQUEST_TIMEOUT=900 \
bash test_envoy_atom.sh
```

TP remains fixed at 1: each DP replica loads a full model on its own GPU.
Set `GPU_MEMORY_UTIL` to fit the model and leave room for cache on each GPU.
The launcher prints an Engine `docker logs -f ...` command before startup so
compilation progress can be followed in another terminal. If an inference request
times out, inspect the Engine log before increasing the timeout further. These script
changes do not require rebuilding the image. Compilation caches stored only in
the Engine container are lost when that container is removed.

## Manual Checks

These examples use the default ports. Run them on the same host after startup.

Inspect Engine metadata and registered Mesh workers:

```bash
curl --noproxy '*' --max-time 5 http://127.0.0.1:10010/server_info
curl --noproxy '*' --max-time 5 http://127.0.0.1:10013/workers
```

Expect `tp_size=1`, `dp_size=2` and `model_id=smoke-model`. By default, Mesh
should list two healthy workers, `http://127.0.0.1:10010@0` and
`http://127.0.0.1:10010@1`. With `DP_SIZE=1`, expect only rank 0.

Send a non-streaming completion through Envoy:

```bash
curl --noproxy '*' --max-time 900 http://127.0.0.1:10016/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smoke-model","prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":false}'
```

Check that `choices[0].text` contains generated text and
`usage.completion_tokens` is positive. With no other inference traffic, run this
command twice to cover a full round-robin cycle across two DP workers.

Send a streaming chat request:

```bash
curl --noproxy '*' -N --max-time 900 http://127.0.0.1:10016/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smoke-model","messages":[{"role":"user","content":"Say hello."}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":32,"temperature":0,"stream":true}'
```

Expect SSE events containing generated content and a final `data: [DONE]`.
Query `/workers` again after requests finish; each worker's `load` should return
to zero. The Mesh management port (`10013`) returns 404 for inference routes;
send inference requests to Envoy (`10016`).

### Continuous Requests

Run the optional client script from this directory to continuously send random
arithmetic prompts. It prints each prompt, the JSON response, HTTP status and
elapsed time. Requests are sequential, with a one-second pause after each one.
Stop with Ctrl+C.

```bash
bash continuous_curl.sh
```

Set `INTERVAL`, `MAX_TOKENS`, `BASE_URL`, `MODEL` or `REQUEST_TIMEOUT` as needed:

```bash
INTERVAL=0.2 MAX_TOKENS=128 bash continuous_curl.sh
```

This client is run manually and is not invoked by the service startup script.

## Logs and Cleanup

After successful startup, all three containers remain running. No
`KEEP_RUNNING` setting is needed. The generated Envoy configuration and startup
log snapshots are saved in the printed `/tmp/atomesh-envoy-smoke.XXXXXX/`
directory. Follow current logs with `docker logs -f <container-name>`.

When finished, run the exact `docker rm -f ...` command printed by the script.
If startup fails, the script removes the containers it created and retains their
logs. Port conflicts are reported before startup; adjust the port variables
listed by `--help` as needed.

## Port Conflicts

If a port cannot be bound, the script reports its address and the environment
variable that controls it, then exits before starting any containers. For example:

```text
Port preflight failed; no containers were started:
  DP_MASTER_PORT=10012: cannot bind 127.0.0.1:10012: [Errno 98] Address already in use
```

Inspect existing listeners with `ss -ltnp`, then set the reported variable to an
unused port. For the example above, run from this directory:

```bash
MODEL_PATH=/models/Qwen3-0.6B GPU_DEVICES=0,1 \
DP_MASTER_PORT=10017 bash test_envoy_atom.sh
```

The default ports are:

| Variable | Port | Service |
| --- | --- | --- |
| `ENGINE_PORT` | 10010 | Engine HTTP |
| `ENGINE_INTERNAL_PORT` | 10011 | Engine internal communication |
| `DP_MASTER_PORT` | 10012 | DP rendezvous |
| `MESH_PORT` | 10013 | Mesh HTTP management |
| `EXT_PROC_PORT` | 10014 | Mesh ext-proc gRPC |
| `METRICS_PORT` | 10015 | Mesh Prometheus |
| `ENVOY_PORT` | 10016 | Envoy HTTP |

Changing these variables only requires rerunning the script; the image does not
need to be rebuilt.

## Optional: Build an Image

Skip this section when you already have a compatible image. If you need to
package the current Atomesh code, build a derived image once and reuse it on
subsequent runs. Run from the repository root, replacing `ATOM_BASE_IMAGE` as needed:

```bash
docker build --build-arg ATOM_BASE_IMAGE=rocm/atom-dev:latest \
  --ulimit nofile=65536:65536 \
  -t atom-extproc:test \
  -f atom/mesh/scripts/ext-proc/Dockerfile atom/mesh
```

The build reuses the base image's Python Engine and Rust toolchain and updates
the Atomesh binary. Both Engine and Mesh containers then use `atom-extproc:test`.
If `protoc` is already available, the build reuses it and skips apt to avoid
triggering dependency conflicts with the base image's custom RCCL build.
Installation is attempted only when `protoc` is missing. The build downloads any
missing Cargo dependencies.

The build sets the file descriptor limit to 65536 and uses 4 concurrent Cargo
jobs by default to avoid `Too many open files (os error 24)` on hosts with many
CPU cores. Use `--build-arg CARGO_BUILD_JOBS=2` to reduce concurrency further.
The `--ulimit` option sets both soft and hard limits for Docker build containers.

If you are already in `atom/mesh/scripts/ext-proc`, use:

```bash
docker build -f Dockerfile \
  --build-arg ATOM_BASE_IMAGE=rocm/atom-dev:latest \
  --ulimit nofile=65536:65536 \
  -t atom-extproc:test ../..
```

## Directory Contents

| File | Purpose |
| --- | --- |
| [Dockerfile](Dockerfile) | Build a test image with the current Atomesh on top of an ATOM image; the build context is `atom/mesh` |
| [test_envoy_atom.sh](test_envoy_atom.sh) | Start the three services for manual testing and print the cleanup command |
| [continuous_curl.sh](continuous_curl.sh) | Continuously send random prompts using curl until interrupted |

The protocol dependency update tool is located at
[proto/vendor_ext_proc.py](../../proto/vendor_ext_proc.py). It updates the
official protocol files and licenses and is run only when maintaining protocol versions.
