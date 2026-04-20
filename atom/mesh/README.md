# ATOM Mesh

> Forked from [sgl-model-gateway v0.3.2](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway).

High-performance model routing gateway for PD (Prefill-Decode) disaggregated LLM serving. Routes inference requests across heterogeneous worker fleets with cache-aware load balancing, gRPC pipeline with Rust-native tokenization, and built-in reliability primitives.

## Features

- **PD Disaggregation**: Separate prefill and decode workers with independent routing policies and bootstrap port handling
- **Regular Mode**: Non-disaggregated routing as performance baseline
- **Dual Protocol**: HTTP and gRPC routing with shared reliability layer
- **gRPC Pipeline**: Fully Rust tokenization, reasoning parsing, and tool-call execution for high-throughput serving
- **Load Balancing**: Random, round-robin, cache-aware (prefix tree), power-of-two, prefix-hash strategies with DP-aware scheduling
- **Reliability**: Retries with exponential backoff, per-worker circuit breakers, rate limiting, and request queuing
- **Observability**: 40+ Prometheus metrics, structured logging
- **Multi-Backend**: SGLang worker backend

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
| `GET /v1/models` | Model info |

## Installation

### Prerequisites

- **Rust and Cargo**
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  ```
### Build from Source

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

Binaries: `target/release/atom-mesh`, `target/release/mesh`

### Verify

```bash
./target/release/mesh --version
```

## Usage

### Regular HTTP Routing

```bash
mesh launch --worker-urls http://worker1:8000 http://worker2:8000 --policy cache_aware
```

### Prefill/Decode Disaggregation

```bash
mesh launch --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --prefill http://prefill2:30002 \
  --decode http://decode1:30011 \
  --decode http://decode2:30012 \
  --prefill-policy cache_aware --decode-policy power_of_two
```

Prefill entries accept an optional bootstrap port (for Mooncake KV cache transfer).

### gRPC Routing

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

### Control Plane

- **Worker Registry**: Centralized registration with model-based indexing and consistent hash ring
- **Worker Manager**: Validates workers, discovers capabilities, tracks load
- **Job Queue**: Async add/remove operations with status tracking via `/workers/{id}`
- **Health Checker**: Background probes feeding circuit breakers and policies

### Data Plane

- **HTTP Router**: Regular and PD routing with per-model policy overrides
- **gRPC Router**: Rust-native tokenizer, reasoning parser, and tool parser pipeline
- **Resilience Layer**: Rate limiting, queuing, retries, circuit breakers

### Worker APIs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/workers` | Register worker (async, returns 202) |
| `GET` | `/workers` | List workers with health and load |
| `GET/PUT/DELETE` | `/workers/{id}` | Inspect, update, or remove worker |
| `POST` | `/flush_cache` | Flush cache across HTTP workers |
| `GET` | `/get_loads` | Sample current worker loads |

## Load Balancing

| Policy | Description |
|--------|-------------|
| `random` | Uniform random selection |
| `round_robin` | Sequential rotation with atomic counters |
| `cache_aware` | Prefix tree matching for cache reuse, with configurable balance thresholds |
| `power_of_two` | Picks lighter worker among two random candidates |
| `prefix_hash` | Consistent prefix hashing |

Per-mode overrides via `--prefill-policy` and `--decode-policy` in PD mode.

## Reliability

- **Retries**: Max 5 with exponential backoff and jitter (`--retry-max-retries`, `--retry-initial-backoff-ms`)
- **Circuit Breakers**: Per-worker with configurable failure/success thresholds (`--cb-failure-threshold`, `--cb-timeout-duration-secs`)
- **Rate Limiting**: Token bucket via `--max-concurrent-requests` with optional request queue (`--queue-size`, `--queue-timeout-secs`)
- **Health Checks**: Configurable interval, timeout, and failure thresholds (`--health-check-interval-secs`)

## Observability

### Prometheus Metrics

Default endpoint: `0.0.0.0:29000` (`--prometheus-host` / `--prometheus-port`)

| Layer | Prefix | Description |
|-------|--------|-------------|
| HTTP | `mesh_http_*` | Request counts, duration, connections, rate limiting |
| Router | `mesh_router_*` | Requests by model/endpoint, latency, errors |
| Inference | `mesh_router_ttft/tpot/tokens_*` | TTFT, TPOT, token counts (gRPC) |
| Worker | `mesh_worker_*` | Pool size, connections, health, selection |
| Circuit Breaker | `mesh_worker_cb_*` | State, transitions, outcomes |
| Retry | `mesh_worker_retries_*` | Attempts, exhausted, backoff |

### Logging

Structured logging via `tracing` with optional file sink (`--log-dir`) and configurable level (`--log-level`).

## Security

Simple API key protection for router endpoints:

```bash
mesh launch --api-key "your-secret-key" \
  --worker-urls http://worker1:8000 http://worker2:8000
```

Clients must provide `Authorization: Bearer <key>`. Workers declared via CLI inherit the router key.

## Development

```bash
cargo build                              # build
cargo build --release                    # release build
```
