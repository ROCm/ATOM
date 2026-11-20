//! `RlWorkerInfo::new` and `entry` on workers built directly, covering what an
//! HTTP mock cannot: health, gRPC workers and PD worker types.

use mesh::{
    config::BackendType,
    core::{
        worker_registry::WorkerId, BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker,
        WorkerType,
    },
    rl::discovery::{entry, RlWorkerInfo},
};
use serde_json::{json, Value};

const URL: &str = "http://worker:8000";

fn row(worker: &dyn Worker, backend: BackendType) -> Value {
    let info = RlWorkerInfo::new(&WorkerId::from_string("w0".to_string()), worker, backend);
    serde_json::to_value(entry(&info, 1)).unwrap()
}

#[test]
fn health_follows_the_worker() {
    let worker = BasicWorkerBuilder::new(URL).build();
    worker.set_healthy(false);
    assert_eq!(row(&worker, BackendType::Atom)["health"], "not_ready");
    worker.set_healthy(true);
    assert_eq!(row(&worker, BackendType::Atom)["health"], "ready");
}

#[test]
fn http_worker_takes_the_backend_as_engine_and_itself_as_control_url() {
    let worker = BasicWorkerBuilder::new(URL).build();
    let row = row(&worker, BackendType::Sglang);
    assert_eq!(row["id"], "w0");
    assert_eq!(row["engine"], "sglang");
    assert_eq!(row["connection_mode"], "http");
    assert_eq!(row["control_url"], URL);
    assert_eq!(row["worker_type"], "regular");
}

#[test]
fn grpc_worker_takes_its_runtime_as_engine_and_has_no_control_url() {
    let worker = BasicWorkerBuilder::new(URL)
        .connection_mode(ConnectionMode::Grpc { port: Some(9000) })
        .runtime_type(RuntimeType::Vllm)
        .build();
    let row = row(&worker, BackendType::Sglang);
    assert_eq!(row["engine"], "vllm", "the runtime, not the --backend");
    assert_eq!(row["connection_mode"], "grpc");
    assert_eq!(row["control_url"], Value::Null);
    assert_eq!(
        row["capabilities"]["pause_modes"],
        json!(["abort", "wait", "keep"]),
        "the vllm capability row"
    );
}

#[test]
fn pd_workers_report_their_type() {
    for (worker_type, label) in [
        (
            WorkerType::Prefill {
                bootstrap_port: Some(8998),
            },
            "prefill",
        ),
        (WorkerType::Decode, "decode"),
    ] {
        let worker = BasicWorkerBuilder::new_with_type(URL, worker_type).build();
        assert_eq!(row(&worker, BackendType::Sglang)["worker_type"], label);
    }
}
