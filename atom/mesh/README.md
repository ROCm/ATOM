# Atomesh

High-performance model routing gateway for **PD (Prefill–Decode) disaggregated** LLM serving. It routes inference across heterogeneous worker fleets with cache-aware load balancing, a gRPC pipeline that keeps tokenization in Rust, and built-in reliability primitives.

Atomesh can also run in an **ATOM standalone** mode. In this mode, Python owns the ATOM engine (LLMEngine and its CoreManager), while Atomesh serves the OpenAI API itself: it renders the chat template, tokenizes, detokenizes, runs the reasoning and tool-call parsers and writes the SSE, and hands token ids to the engine through a PyO3 token bridge (`routers/atom_bridge`, see `docs/atomesh_rust_frontend.md`). This allows the same Atomesh serving layer to support both distributed worker routing and a single-process ATOM deployment path.

## Features

- **PD disaggregation**: Separate prefill and decode workers, independent routing policies, and optional bootstrap ports for KV transfer
- **Regular mode**: Single-pool routing as a performance baseline
- **Dual protocol**: HTTP and gRPC through a shared reliability layer
- **gRPC pipeline**: Rust-side tokenization, reasoning parsing, and tool-call execution for high throughput
- **Load balancing**: Random, round-robin, cache-aware (prefix tree), power-of-two, and prefix-hash strategies, with DP-aware scheduling where applicable
- **Reliability**: Retries with exponential backoff, per-worker circuit breakers, rate limiting, and request queuing
- **Observability**: 40+ Prometheus metrics and structured logging
- **Multi-backend**: SGLang worker backend

### Supported Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions with streaming and tool calls |
| `POST /v1/completions` | Text completions |
| `POST /generate` | SGLang generate API |
| `POST /v1/messages` | HTTP-backend Messages proxy, JSON and SSE |
| `POST /v1/responses` | HTTP-backend Responses proxy, JSON and SSE; existing background/query support |
| `POST /v1/tokenize` / `/v1/detokenize` | Tokenization with batch support |
| `POST /parse/reasoning` / `/parse/function_call` | Reasoning and tool-call parsing |
| `GET /health` / `/readiness` / `/liveness` | Health probes |
| `GET /engine_metrics` | Aggregated worker engine Prometheus metrics |
| `GET /metrics` | Mesh self metrics, the same registry as the Prometheus exporter |
| `GET /v1/models` | Model metadata |

Responses resource operations (GET, DELETE, cancel and input_items) probe regular
HTTP backends that support `/v1/responses`. A `mesh.apis` label restricts supported
APIs using comma-separated paths, such as `/v1/chat/completions,/v1/responses`;
omitting the label preserves the default of accepting all supported API routes.

A Responses request that refers to stored state (`previous_response_id`,
`conversation`, or an `item_reference` input item) must reach the backend that
holds that state. When all candidate workers share one backend address, it is
routed normally. When candidates span several addresses, every candidate must
carry the `mesh.responses_state=shared` label, which declares that those backends
read one shared state store; otherwise Mesh returns
`501 stateful_routing_unsupported`. Unhealthy candidates count too, so failover
cannot move a request to a different state store. Requests without these fields
need no label.

A configured worker API key replaces client credentials and is sent as
`authorization: Bearer <key>`; Messages requests also carry it as `x-api-key`.
Backends without a worker key receive the client's `authorization` and
`x-api-key` headers. Resource operations use the same credential rules as
inference.
Workers sharing one backend address must have matching credential configuration.
The first successful probe is returned immediately. If all probes fail, non-404
client errors take precedence over server errors and unreachable backends
(`502 backend_unreachable`), which outrank 404 because such a backend may own the
resource. 405 and 501 from backends without these routes rank last. Errors with
equal priority are selected by backend address order.

DP-aware forwarding writes the selected worker's rank to `data_parallel_rank` in
the request body for Chat Completions, Completions and Generate, overriding any
client value. Messages and Responses receive the rank only when the DP worker's
`mesh.apis` label explicitly lists `/v1/messages` or `/v1/responses`; without the
label they are forwarded unchanged, so the backend chooses the rank and DP
placement is only a load-balancing estimate. The backend must consume the field
to route to the selected rank. Verify support for each backend and API before
listing it; accepting an unknown JSON field does not establish rank-routing support.

## Installation

### Prerequisites

- **Rust and Cargo**
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  ```

- **Protocol Buffers compiler (`protoc`)**, for every build: the `smg-grpc-client`
  dependency's build script runs it (this crate's own protos, including the optional
  `ext-proc` Envoy bindings, are compiled in-process by protox):
  ```bash
  sudo apt-get install protobuf-compiler
  ```
- **Python development libraries** for the existing PyO3 bridge; select the interpreter
  with `PYO3_PYTHON` when needed.

### Build from source

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Release build with optional Envoy ext-proc support
cargo build --release --features ext-proc
```

Default builds exclude ext-proc code, its protobuf generation step, and the
`--ext-proc*` CLI options. Enabling the Cargo feature makes those options
available; pass `--ext-proc` at runtime to start the listener.

Artifacts: `target/release/atomesh`, `target/release/libmesh.so`.

### Build during ATOM package install

ATOM package installation does not build Atomesh by default. To compile Atomesh during
package installation and include `libmesh.so` in the installed package, enable
the Atomesh build hook:

```bash
ATOM_MESH_BUILD=1 python -m pip install .
```

For editable installs:

```bash
ATOM_MESH_BUILD=1 python -m pip install -e .
```

The hook runs `cargo build --release` under `atom/mesh` before Python package
files are collected. To include ext-proc, pass its Cargo feature explicitly:

```bash
ATOM_MESH_BUILD=1 ATOM_MESH_FEATURES=ext-proc python -m pip install -e .
```

`ATOM_MESH_FEATURES` accepts Cargo's comma- or space-separated feature list and
defaults to empty. Both `docker/Dockerfile` and `docker/atom_release.dockerfile`
accept `--build-arg ATOM_MESH_FEATURES=ext-proc` for the same opt-in. The build
image must provide `protoc` either way, for `smg-grpc-client`.

## Usage

### ATOM standalone

ATOM standalone mode starts the ATOM engine in Python, while Rust Mesh serves
the OpenAI API (template, tokenizer, parsers, SSE) and hands each request's
token ids to the engine through the token bridge. A model whose chat template
is Python (DeepSeek-V4's encoder, Kimi-K3's tokenizer) has its prompt rendered
by ATOM instead. Enable the Atomesh entrypoint through
`USE_ATOMESH_ENTRYPOINTS=1`:

```bash
USE_ATOMESH_ENTRYPOINTS=1 python -m atom.entrypoints.openai_server \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 30000
```

Do not pass `mesh-only` for standalone mode. The `mesh-only` subcommand is only
used when the Python entrypoint should run Atomesh as a router for external
workers.


### Regular HTTP routing

Routing mode can be started through either the Rust binary or the Python
entrypoint. For the Python entrypoint, `mesh-only` selects Atomesh routing;
without `mesh-only`, it defaults to ATOM standalone mode.

```bash
./target/release/atomesh launch \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware \
  --backend atom

USE_ATOMESH_ENTRYPOINTS=1 python -m atom.entrypoints.openai_server mesh-only \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware
```

ATOM workers need `--backend atom`, in regular mode as in PD mode. Without it
the router keys an image chat as it would for SGLang: two chats that differ only
in an image past `prefix_hash`'s token window, or after a long shared text, get
the same `cache_aware` or `prefix_hash` key, though ATOM keys every KV block of
a media prompt by all its media and shares no block between them. The binary
defaults to `--backend sglang`; the Python entrypoint's `mesh-only` passes
`--backend atom` unless the command line sets `--backend` (or its alias
`--runtime`) itself, so put `--backend sglang` there to route to SGLang workers.

`--backend atom` also changes how the router discovers an HTTP worker's model:
it reads `/v1/models` instead of SGLang's `/server_info` and `/model_info`, and
the model id it returns names the worker's model. For a worker serving under an
alias (`--served-model-name`) that id is the alias, so the router loads the
worker's tokenizer from `--tokenizer-path` or `--model-path` when either is
set, and from the model id only otherwise.

### Envoy external processing

Build with `--features ext-proc`, then add `--ext-proc` at runtime to use the
tonic ext-proc gRPC listener for inference. The Axum
HTTP port retains management, health and auxiliary APIs; direct HTTP inference
routes are disabled. Envoy forwards Regular requests directly to selected HTTP
workers; Prefill/Decode mode uses a separate execution listener for the selected
pair. See [the ext-proc smoke test guide](scripts/ext-proc/README.md) for image
build and integration test instructions, and the
[complete Envoy configuration](tests/fixtures/ext-proc/envoy.yaml).

### Prefill / decode disaggregation

```bash
./target/release/atomesh launch --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --prefill http://prefill2:30002 \
  --decode http://decode1:30011 \
  --decode http://decode2:30012 \
  --prefill-policy cache_aware --decode-policy power_of_two

USE_ATOMESH_ENTRYPOINTS=1 python -m atom.entrypoints.openai_server mesh-only \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --prefill http://prefill2:30002 \
  --decode http://decode1:30011 \
  --decode http://decode2:30012 \
  --prefill-policy cache_aware --decode-policy power_of_two
```

Prefill entries may include an optional bootstrap port (e.g. for Mooncake KV cache transfer).

With `--backend atom`, `--atom-pd-concurrent` (default off) sends a relay-mode
`/v1/chat/completions` request to decode while prefill runs, as a two-line body
on `/v1/pd/late_kv/chat/completions`: the request, then prefill's
`kv_transfer_params` once prefill answers. Decode needs `ATOM_PD_LATE_KV=1`; a
decode that answers 404 gets the serial relay's request instead (logged once).
A prefill failure aborts decode's body before the second line, so decode
enqueues nothing.

Messages and Responses have no PD adapter: in PD mode `/v1/messages` and
`/v1/responses` return `501 unsupported_api_topology`. Serve them from regular
HTTP workers.

### gRPC routing

```bash
./target/release/atomesh launch \
  --worker-urls grpc://worker1:31001 grpc://worker2:31002 \
  --tokenizer-path /path/to/tokenizer.json \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json

USE_ATOMESH_ENTRYPOINTS=1 python -m atom.entrypoints.openai_server mesh-only \
  --worker-urls grpc://worker1:31001 grpc://worker2:31002 \
  --tokenizer-path /path/to/tokenizer.json \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json
```

Supported reasoning parsers: `deepseek-r1`, `qwen3`, `qwen3-thinking`, `kimi`, `glm45`, `glm47`, `step3`, `minimax`.  
Supported tool parsers: `json`, `python`, `xml`.

## Architecture

### Control plane

- **Worker registry**: Central registration with model-based indexing and a consistent hash ring
- **Worker manager**: Validates workers, discovers capabilities, tracks load
- **Job queue**: Async add/remove with status via `/workers/{id}`
- **Health checker**: Background probes feeding circuit breakers and policies

### Data plane

- **HTTP router**: Regular and PD routing with per-model policy overrides
- **gRPC router**: Rust-native tokenizer, reasoning parser, and tool parser pipeline
- **Resilience layer**: Rate limiting, queuing, retries, circuit breakers

### Worker APIs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/workers` | Register worker (async, returns 202) |
| `GET` | `/workers` | List workers with health and load |
| `GET` / `PUT` / `DELETE` | `/workers/{id}` | Inspect, update, or remove worker |
| `POST` | `/flush_cache` | Flush cache across HTTP workers |
| `GET` | `/get_loads` | Sample current worker loads |
| `GET` | `/kv_events/sources` | With `--kv-events`: each followed worker rank's KV event stream and what it carried. A worker not followed, as with the engine's KV events off (`ATOM_KV_EVENTS_ENABLE`) or sources that cannot be read, has no row, as if not registered; the log says why |
| `GET` | `/kv_storage/sources` | With `--kv-events`: each Mooncake Store the workers report offloading to, its readers, and each pool master's answer, asked meanwhile at `--mooncake-admin-port`: tainted or not |

## Load balancing

| Policy | Description |
|--------|-------------|
| `random` | Uniform random selection |
| `round_robin` | Sequential rotation with atomic counters |
| `cache_aware` | Prefix-tree matching for cache reuse, with configurable balance thresholds; with `--kv-events`, a request the router can tokenize goes by the blocks each worker's device cache holds, once one holds more of it than `cache_threshold`. Requests with media, imbalanced load and workers without indexed events stay on the tree |
| `power_of_two` | Chooses the lighter of two randomly sampled workers |
| `prefix_hash` | Consistent prefix hashing |
| `agentic` | Prefill and regular pools. Lowest prefill cost: queued prefill tokens, plus the prompt tokens the worker's GPU cache does not hold (from `--kv-events`, or a text tree for requests the index cannot match), plus `--agentic-request-token-equivalent` (512) per in-flight request. See [agentic routing](docs/agentic_cost.md) |

In PD mode, use `--prefill-policy` and `--decode-policy` for per-mode overrides.

`prefix_hash` routes on exact request tokens. Messages and Responses have no
registered token-routing template, so under `prefix_hash` they return
`501 token_routing_unsupported`. So do chats for a model whose tokenizer
llm-tokenizer renders with a built-in encoder chosen from `config.json`, not a
chat template, such as DeepSeek V3.2 and V4
([#3377](https://github.com/ROCm/ATOM/issues/3377)). Use a policy that does not
need tokens, such as `cache_aware` or `round_robin`, for these models and APIs.

The router renders a chat it tokenizes with the model's chat template and the
request's `reasoning_effort` and `chat_template_kwargs`. A checkpoint whose
`config.json` has `model_type` `deepseek_v41` (DeepSeek-V4.1) renders with
llm-tokenizer's port of V4.1's encoder instead, and the router gives it the
effort and mode as ATOM's V4.1 encoder takes them: `minimal` as `low`, `medium`
and `xhigh` as `high`, `none` as chat mode, and the mode from `thinking_mode`,
or chat mode where `enable_thinking` is false. Other controls, such as a
message's list of text blocks, can still render otherwise than on ATOM; ATOM
renders the chat again, so the difference costs prefix hits, not a different
prompt.

## Tokenization

The router tokenizes requests with their model's HuggingFace tokenizer, or with
[fastokens](https://github.com/crusoecloud/fastokens) for the models listed in
`--fastokens-models`, by the name or source their tokenizer registers under, or
`*` for every model. fastokens encodes a `tokenizer.json` BPE faster than
HuggingFace `tokenizers`, and splits a long text between special tokens across
a pool of its own threads (one per logical core up to 32;
`FASTOKENS_BPE_THREADS` sets the number). Decoding, chat templates and vocabulary lookups stay with the
HuggingFace tokenizer.

A model encodes with fastokens only if fastokens encodes a set of probe texts
exactly as its HuggingFace tokenizer does, with special tokens and without. The
probe covers added tokens, digits, whitespace, several scripts, a chat, a 1 MiB
text and a 256 KiB one with no newline. A model that fails the probe keeps
HuggingFace, with a warning, or under `--fastokens-strict` its tokenizer fails
to register, with no retries. A `tokenizer.json` that fastokens cannot find or
download fails no probe: the model keeps HuggingFace, with a warning, or under
`--fastokens-strict` its registration fails and is retried.

fastokens loads `tokenizer.json` alone, as HuggingFace `tokenizers` does, so
both encode the tokens only `tokenizer_config.json` declares (Qwen3.5's audio
tokens, for one) as text, where `transformers`, and so the engine, encodes them
as tokens. With gRPC workers the router sends
the ids it encodes as the prompt, so the model runs on fastokens' ids, not only
routes by them. `mesh_fastokens_tokenizers_total{outcome}` counts tokenizers as
`loaded`, `fallback` or `refused`.

The prompts of these models tokenize up to `--fastokens-max-tokenize-bytes`
(default: the larger of 64 MiB and the other models' limit), and so do their
`/v1/tokenize` batches. `--fastokens-segment-cache-mb` sizes
fastokens' cache of the ids of the text between special tokens, per tokenizer
(default: 1024; 0 turns it off). An agent re-sends its history with each
request, and the cache keeps the router from encoding it again. It replaces
`--tokenizer-cache-*`, which does not apply to these models.

`test_requests_tokenize_as_on_atom` checks that requests tokenize for routing as
on ATOM, with either backend, and times encoding. `scripts/atom_token_parity.py`
tokenizes a set of requests, or recorded ones, with ATOM's own chat and
completion endpoints, in process and without a GPU, and the test compares the
router's tokens with those, in the file `$ATOM_TOKEN_PARITY` names, so it runs
only when asked for. CI's mesh unit tests run the two for a DeepSeek-V4.1-Flash
revision, whose tokenizer files and chat encoder the script downloads from
HuggingFace, or `$HF_ENDPOINT`, and the test with
`cargo test --lib -- --ignored routers::ingress::token_parity`. The script's
docstring shows how to run them for another checkpoint.

## Reliability

- **Request preparation**: Parsing, template rendering and tokenization use a fixed
  thread pool. By default, Mesh splits Rust's `available_parallelism()` count
  between preparation and Tokio: 224 available CPUs give 112 preparation threads
  and 112 Tokio workers. An odd spare CPU goes to Tokio; with one available CPU,
  each pool still has one worker. `--prepare-workers` and `TOKIO_WORKER_THREADS`
  independently override their respective defaults. The OS chooses CPU placement,
  and auxiliary threads may also exist.
  The waiting queue defaults to 10 jobs per resolved preparation thread: 1120 jobs
  with 112 threads, or 20 with `--prepare-workers 2`.
  `--prepare-queue-capacity` overrides that limit. Queued jobs still have a
  `--prepare-queue-timeout-ms` deadline (default: 250 ms); canceled waiters release
  their slot immediately. All request parsing runs on this pool. Each request's
  preparation shares one `--prepare-timeout-ms` deadline (default: 5000 ms), and
  tokenization input is limited by `--prepare-max-tokenize-bytes` (default: 1 MiB),
  or for the models fastokens encodes by `--fastokens-max-tokenize-bytes`
  (default: the larger of that limit and 64 MiB).
  When ext-proc is enabled, the legacy `--ext-proc-parser-concurrency` and
  `--ext-proc-max-tokenize-bytes` options apply only if the corresponding
  `--prepare-*` option is unset; neither has a default of its own. On shutdown,
  Mesh waits up to 5 seconds for preparation threads after transport draining.
  Ext-proc's separate `--ext-proc-max-buffered-bytes` limit covers its request
  buffers through rewriting and dispatch.
- **Retries**: Up to 5 attempts with exponential backoff and jitter (`--retry-max-retries`, `--retry-initial-backoff-ms`)
- **Circuit breakers**: Per-worker failure/success thresholds (`--cb-failure-threshold`, `--cb-timeout-duration-secs`)
- **Rate limiting**: Token bucket via `--max-concurrent-requests`, optional queue (`--queue-size`, `--queue-timeout-secs`)
- **Health checks**: Configurable interval, timeout, and failure thresholds (`--health-check-interval-secs`)

## Observability

### Prometheus metrics

Default bind: `0.0.0.0:29000` (`--prometheus-host` / `--prometheus-port`)

Mesh self metrics are exposed by the Prometheus exporter above and by the
router's own `/metrics` endpoint, which returns 503 when the exporter is not
running. The router port serves the whole registry whatever `--prometheus-host`
is set to, so binding the exporter to a private address does not hide these
metrics from clients that can reach the router. Worker engine metrics remain
available through the router's `/engine_metrics` endpoint, which scrapes backend
workers, injects `worker_addr`, and aggregates their Prometheus text without
changing worker metric names.

| Layer | Prefix | Description |
|-------|--------|-------------|
| HTTP | `mesh_http_*` | Request counts, duration, connections, rate limiting |
| Router | `mesh_router_*` | Requests by model/endpoint, latency, errors |
| Inference | `mesh_router_ttft/tpot/tokens_*` | TTFT, TPOT, token counts (gRPC) |
| Worker | `mesh_worker_*` | Pool size, connections, health, selection |
| Routing | `mesh_route_*`, `mesh_session_*` | Policy decisions, session worker changes, engine-reported cached tokens against the router estimate |
| Circuit breaker | `mesh_worker_cb_*` | State, transitions, outcomes |
| Retry | `mesh_worker_retries_*` | Attempts, exhaustion, backoff |
| Tokenization | `mesh_fastokens_*` | Tokenizers that encode with fastokens, keep HuggingFace or fail to register |

[Routing metrics](docs/metrics.md) covers each routing metric's labels, the
policies and branches that record it, how to read the values, and example
queries.

### Logging

Structured logging via `tracing`, optional file sink (`--log-dir`), and log level (`--log-level`).

`--route-log-sample-rate` (0 to 1, default 0 = off) logs the routing decisions of
that fraction of inference requests at info level under target `mesh::route`.
While it is above 0, a request carrying the `x-mesh-route-trace` header is always
logged. HTTP routers only: gRPC workers and Envoy external processing do not
emit these logs.

## Security

### TLS

Enable HTTPS by providing both a PEM certificate chain and a PEM private key:

```bash
./target/release/atomesh launch \
  --tls-cert-path /path/to/fullchain.pem \
  --tls-key-path /path/to/privkey.pem \
  --worker-urls http://worker1:8000 http://worker2:8000

USE_ATOMESH_ENTRYPOINTS=1 python -m atom.entrypoints.openai_server mesh-only \
  --tls-cert-path /path/to/fullchain.pem \
  --tls-key-path /path/to/privkey.pem \
  --worker-urls http://worker1:8000 http://worker2:8000
```

`--tls-cert-path` and `--tls-key-path` must be provided together. When TLS is
enabled, the router serves HTTPS on the configured `--host` and `--port`.

### API key

Set `ATOM_MESH_API_KEY` in your environment, then pass it to the router:

```bash
./target/release/atomesh launch --api-key "${ATOM_MESH_API_KEY:?Set ATOM_MESH_API_KEY}" \
  --worker-urls http://worker1:8000 http://worker2:8000

USE_ATOMESH_ENTRYPOINTS=1 python -m atom.entrypoints.openai_server mesh-only --api-key "${ATOM_MESH_API_KEY:?Set ATOM_MESH_API_KEY}" \
  --worker-urls http://worker1:8000 http://worker2:8000
```

Clients send `Authorization: Bearer <key>`. Workers declared on the CLI inherit the router key.

## Acknowledgments

Atomesh was derived from [sgl-model-gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) and has since been substantially reworked into an independent serving layer for the **ATOM** stack and **AMD** hardware. Its routing, scheduling, runtime integration, and performance-sensitive paths are designed around AMD accelerator deployments and ATOM-specific serving requirements.
