# ATOM Mesh

High-performance model routing gateway for **PD (Prefill–Decode) disaggregated** LLM serving. It routes inference across heterogeneous worker fleets with cache-aware load balancing, a gRPC pipeline that keeps tokenization in Rust, and built-in reliability primitives.

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

Artifacts: `target/release/atom-mesh`, `target/release/mesh`.

### Verify

```bash
./target/release/mesh --version
```

## Usage

### Regular HTTP routing

```bash
mesh launch --worker-urls http://worker1:8000 http://worker2:8000 --policy cache_aware
```

### Prefill / decode disaggregation

```bash
mesh launch --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --prefill http://prefill2:30002 \
  --decode http://decode1:30011 \
  --decode http://decode2:30012 \
  --prefill-policy cache_aware --decode-policy power_of_two
```

Prefill entries may include an optional bootstrap port (e.g. for Mooncake KV cache transfer).

### gRPC routing

```bash
mesh launch \
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

Optional API key protection for router endpoints:

```bash
mesh launch --api-key "your-secret-key" \
  --worker-urls http://worker1:8000 http://worker2:8000
```

Clients send `Authorization: Bearer <key>`. Workers declared on the CLI inherit the router key.

## Development

```bash
cargo build           # debug
cargo build --release # release
```

## Acknowledgments and upstream

[**sgl-model-gateway**](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) is an excellent reference implementation for disaggregated model routing and high-throughput serving. ATOM Mesh builds on that design, then adapts it for the **ATOM** stack and **AMD** hardware: scheduling, defaults, and performance-related paths are tuned for AMD accelerators and typical AMD deployment constraints so the gateway remains a strong fit for AMD-centric clusters.
