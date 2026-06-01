# Atomesh Mocker

`atom/mesh/mocker` is a standalone crate for fixture-driven Atomesh mock and benchmark workflows.

## Build

From `atom/mesh/mocker`:

```bash
cargo build
```

From `atom/mesh`:

```bash
cargo build --manifest-path mocker/Cargo.toml
```

## Workflow

Run the workflow in three terminals.

### 1. Start Virtual Workers

Start one or more virtual backend workers first:

```bash
cd ATOM/atom/mesh/mocker && ./target/debug/atomesh-mocker virtual-workers \
  --ip 127.0.0.1 \
  --base-port 30010 \
  --workers 1 \
  fixtures/http_regular_chat.json
```

Workers bind to `base-port + worker_index`. For example, `--base-port 30010 --workers 2` starts workers on `30010` and `30011`.

### 2. Start Atomesh

Start Atomesh and point it at the virtual worker URL:

```bash
cd ATOM/atom/mesh && ./target/debug/atom-mesh \
  --host 127.0.0.1 \
  --port 30000 \
  --worker http://127.0.0.1:30010
```

Adjust the Atomesh launch arguments to match the router mode you want to test. For PD mode, start multiple virtual workers and pass the corresponding prefill/decode worker URLs to Atomesh.

### 3. Start Benchmark Requests

Run the benchmark request producer/consumer pipeline against Atomesh:

```bash
cd ATOM/atom/mesh/mocker && ./target/debug/atomesh-mocker benchmark-request \
  --base-url http://127.0.0.1:30000 \
  --producer-threads 1 \
  --consumer-threads 4 \
  fixtures/http_regular_chat.json
```

`benchmark-request` keeps producing requests until `Ctrl-C`. Metrics are printed every 5 seconds.

## Test Harness

`test_harness.rs` provides fixture-driven integration tests for the mocker crate. It starts virtual workers, creates an in-process Atomesh app/router, sends the fixture request through Atomesh, and validates the response plus runtime routing state.

Run all harness cases from `atom/mesh/mocker`:

```bash
cargo test test_atomesh_harness
```

Run one focused case:

```bash
cargo test test_atomesh_harness_http_regular_chat
```

From `atom/mesh`, use the mocker manifest explicitly:

```bash
cargo test --manifest-path mocker/Cargo.toml test_atomesh_harness
```

The harness does not require a separately running Atomesh server or virtual worker process; it owns those resources inside the test.

## Fixtures

Fixture JSON files live directly under `fixtures/`. Pass fixture paths explicitly:

```bash
./target/debug/atomesh-mocker benchmark-request \
  --base-url http://127.0.0.1:3000 \
  fixtures/http_regular_chat.json \
  fixtures/http_regular_generate.json
```

## Commands

```bash
./target/debug/atomesh-mocker virtual-workers --help
./target/debug/atomesh-mocker benchmark-request --help
```
