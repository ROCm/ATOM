#![allow(unused_imports)]

use super::test_support::*;

#[tokio::test]
async fn h01_http_regular_planner_single_dispatches_to_worker_url() {
    todo!()
}

#[tokio::test]
async fn h02_http_regular_no_workers_returns_503() {
    todo!()
}

#[tokio::test]
async fn h03_http_regular_model_not_found_returns_503_with_model_name() {
    todo!()
}

#[tokio::test]
async fn h04_grpc_stage_regular_writes_single_worker_selection() {
    use crate::core::placement::types::{PlacementPlan, PlacementTrace};
    use crate::routers::grpc::context::WorkerSelection;
    use crate::routers::grpc::common::stages::worker_selection::plan_to_worker_selection;

    let worker = make_regular_grpc("http://w-1:8000", "m");
    let plan = PlacementPlan::Single {
        worker: worker.clone(),
        policy_name: "round_robin",
        trace: PlacementTrace::for_single(Some("m"), 1, 1, worker.url(), "round_robin", Some("m")),
    };

    let selection = plan_to_worker_selection(plan);
    match selection {
        WorkerSelection::Single { worker: w } => {
            assert_eq!(w.url(), "http://w-1:8000");
        }
        _ => panic!("expected Single"),
    }
}

#[tokio::test]
async fn h05_grpc_stage_pd_writes_dual_worker_selection() {
    use crate::core::placement::types::{PlacementPlan, PlacementTrace};
    use crate::routers::grpc::context::WorkerSelection;
    use crate::routers::grpc::common::stages::worker_selection::plan_to_worker_selection;

    let prefill = make_prefill_grpc("http://p:8000", "m", Some(8998));
    let decode = make_decode_grpc("http://d:8000", "m");
    let plan = PlacementPlan::Pair {
        prefill: prefill.clone(),
        decode: decode.clone(),
        prefill_policy: "round_robin",
        decode_policy: "round_robin",
        trace: PlacementTrace::for_pair(
            Some("m"),
            2,
            2,
            prefill.url(),
            decode.url(),
            "round_robin",
            "round_robin",
            Some("m"),
        ),
    };

    let selection = plan_to_worker_selection(plan);
    match selection {
        WorkerSelection::Dual { prefill: p, decode: d } => {
            assert_eq!(p.url(), "http://p:8000");
            assert_eq!(d.url(), "http://d:8000");
        }
        _ => panic!("expected Dual"),
    }
}

#[tokio::test]
async fn h06_grpc_stage_planner_err_returns_service_unavailable() {
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use crate::core::placement::types::PlacementError;
    use crate::routers::grpc::common::stages::worker_selection::placement_err_to_response;

    let resp = placement_err_to_response(
        PlacementError::NoAvailableWorkers,
        Some("test_model"),
    );
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(
        body_str.contains("test_model"),
        "expected body to mention model_id, got: {}",
        body_str
    );
}

#[tokio::test]
async fn h07_http_pd_sglang_dual_dispatch_to_prefill_and_decode() {
    todo!()
}

#[tokio::test]
async fn h08_http_pd_vllm_pair_uses_shared_transfer_id() {
    todo!()
}

#[tokio::test]
async fn h09_http_pd_sglang_batch_writes_length_n_arrays() {
    todo!()
}

#[tokio::test]
async fn h10_http_pd_retry_preserves_text_headers_tokens() {
    todo!()
}
