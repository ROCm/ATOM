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
    todo!()
}

#[tokio::test]
async fn h05_grpc_stage_pd_writes_dual_worker_selection() {
    todo!()
}

#[tokio::test]
async fn h06_grpc_stage_planner_err_returns_service_unavailable() {
    todo!()
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
