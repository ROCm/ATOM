# Atomesh

High-performance model routing gateway for **PD (Prefill–Decode) disaggregated** LLM serving. It routes inference across heterogeneous worker fleets with cache-aware load balancing, a gRPC pipeline that keeps tokenization in Rust, and built-in reliability primitives.

Atomesh can also run in an **ATOM standalone** mode. In this mode, Python owns the ATOM engine, tokenizer, and OpenAI-compatible serving semantics, while Atomesh provides the HTTP server, request routing, lifecycle management, observability, and PyO3 bridge. This allows the same Atomesh serving layer to support both distributed worker routing and a single-process ATOM deployment path.

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
| `POST /v1/responses` | Background responses with status tracking |
| `POST /v1/tokenize` / `/v1/detokenize` | Tokenization with batch support |
| `POST /parse/reasoning` / `/parse/function_call` | Reasoning and tool-call parsing |
| `GET /health` / `/readiness` / `/liveness` | Health probes |
| `GET /engine_metrics` | Aggregated worker engine Prometheus metrics |
| `GET /v1/models` | Model metadata |

## Installation

### Prerequisites

- **Rust and Cargo**
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  ```

### Build from source

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

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
files are collected.

## Usage

### ATOM standalone

ATOM standalone mode starts the ATOM engine and tokenizer in Python, while Rust
Mesh provides the HTTP server, lifecycle handling, observability, and PyO3
request bridge. Enable the Atomesh entrypoint through
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
  --policy cache_aware

USE_ATOMESH_ENTRYPOINTS=1 python -m atom.entrypoints.openai_server mesh-only \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware
```

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

## Load balancing

| Policy | Description |
|--------|-------------|
| `random` | Uniform random selection |
| `round_robin` | Sequential rotation with atomic counters |
| `cache_aware` | Prefix-tree matching for cache reuse, with configurable balance thresholds |
| `power_of_two` | Chooses the lighter of two randomly sampled workers |
| `prefix_hash` | Consistent prefix hashing |

In PD mode, use `--prefill-policy` and `--decode-policy` for per-mode overrides.

## Reliability

- **Retries**: Up to 5 attempts with exponential backoff and jitter (`--retry-max-retries`, `--retry-initial-backoff-ms`)
- **Circuit breakers**: Per-worker failure/success thresholds (`--cb-failure-threshold`, `--cb-timeout-duration-secs`)
- **Rate limiting**: Token bucket via `--max-concurrent-requests`, optional queue (`--queue-size`, `--queue-timeout-secs`)
- **Health checks**: Configurable interval, timeout, and failure thresholds (`--health-check-interval-secs`)

## Observability

### Prometheus metrics

Default bind: `0.0.0.0:29000` (`--prometheus-host` / `--prometheus-port`)

Mesh self metrics are exposed by the Prometheus exporter above. Worker engine
metrics remain available through the router's `/engine_metrics` endpoint, which
scrapes backend workers, injects `worker_addr`, and aggregates their Prometheus
text without changing worker metric names.

| Layer | Prefix | Description |
|-------|--------|-------------|
| HTTP | `mesh_http_*` | Request counts, duration, connections, rate limiting |
| Router | `mesh_router_*` | Requests by model/endpoint, latency, errors |
| Inference | `mesh_router_ttft/tpot/tokens_*` | TTFT, TPOT, token counts (gRPC) |
| Worker | `mesh_worker_*` | Pool size, connections, health, selection |
| Circuit breaker | `mesh_worker_cb_*` | State, transitions, outcomes |
| Retry | `mesh_worker_retries_*` | Attempts, exhaustion, backoff |

### Logging

Structured logging via `tracing`, optional file sink (`--log-dir`), and log level (`--log-level`).

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

Optional API key protection for router endpoints:

```bash
./target/release/atomesh launch --api-key "your-secret-key" \
  --worker-urls http://worker1:8000 http://worker2:8000

USE_ATOMESH_ENTRYPOINTS=1 python -m atom.entrypoints.openai_server mesh-only --api-key "your-secret-key" \
  --worker-urls http://worker1:8000 http://worker2:8000
```

Clients send `Authorization: Bearer <key>`. Workers declared on the CLI inherit the router key.

## Acknowledgments

Atomesh was derived from [sgl-model-gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) and has since been substantially reworked into an independent serving layer for the **ATOM** stack and **AMD** hardware. Its routing, scheduling, runtime integration, and performance-sensitive paths are designed around AMD accelerator deployments and ATOM-specific serving requirements.