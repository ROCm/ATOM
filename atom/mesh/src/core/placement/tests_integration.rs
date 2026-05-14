#![allow(unused_imports)]

use super::test_support::*;

#[tokio::test]
async fn h01_http_regular_planner_single_dispatches_to_worker_url() {
    use std::sync::Arc;
    use crate::core::placement::planner::DefaultPlanner;
    use crate::core::placement::traits::PdPlanner;
    use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};

    let src = MockWorkerSource::new().add_worker(make_regular_http("http://w-1:8000", "m"));
    let policies = MockPolicySource::new();
    let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

    let descriptor = RequestDescriptor {
        model_id: Some("m"),
        protocol: Some(Protocol::Http),
        ..Default::default()
    };

    let plan = planner.plan(&descriptor).await.expect("plan should succeed");
    match plan {
        PlacementPlan::Single { worker, .. } => {
            assert_eq!(worker.url(), "http://w-1:8000");
        }
        _ => panic!("expected Single"),
    }
}

#[tokio::test]
async fn h02_http_regular_no_workers_returns_503() {
    use std::sync::Arc;
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use crate::core::placement::planner::DefaultPlanner;
    use crate::core::placement::traits::PdPlanner;
    use crate::core::placement::types::{PlacementError, Protocol, RequestDescriptor};
    use crate::routers::shared::placement_response::placement_err_to_response;

    let src = MockWorkerSource::new();
    let policies = MockPolicySource::new();
    let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

    let descriptor = RequestDescriptor {
        model_id: None,
        protocol: Some(Protocol::Http),
        ..Default::default()
    };

    let err = planner.plan(&descriptor).await.expect_err("plan should fail");
    assert_eq!(err, PlacementError::NoWorkers);

    let resp = placement_err_to_response(err, None);
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(
        body_str.contains("no_workers"),
        "expected body to contain 'no_workers', got: {}",
        body_str
    );
}

#[tokio::test]
async fn h03_http_regular_model_not_found_returns_503_with_model_name() {
    use std::sync::Arc;
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use crate::core::placement::planner::DefaultPlanner;
    use crate::core::placement::traits::PdPlanner;
    use crate::core::placement::types::{PlacementError, Protocol, RequestDescriptor};
    use crate::routers::shared::placement_response::placement_err_to_response;

    let src = MockWorkerSource::new().add_worker(make_regular_http("http://w-other:8000", "other"));
    let policies = MockPolicySource::new();
    let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

    let descriptor = RequestDescriptor {
        model_id: Some("requested_model"),
        protocol: Some(Protocol::Http),
        ..Default::default()
    };

    let err = planner.plan(&descriptor).await.expect_err("plan should fail");
    assert_eq!(
        err,
        PlacementError::ModelNotFound {
            model_id: "requested_model".to_string()
        }
    );

    let resp = placement_err_to_response(err, Some("requested_model"));
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(
        body_str.contains("requested_model"),
        "expected body to contain 'requested_model', got: {}",
        body_str
    );
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
    use crate::routers::shared::placement_response::placement_err_to_response;

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
    use std::sync::Arc;
    use serde_json::json;
    use crate::core::placement::backend::sglang::SglangAdapter;
    use crate::core::placement::backend::BackendAdapter;
    use crate::core::placement::planner::DefaultPlanner;
    use crate::core::placement::traits::PdPlanner;
    use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};

    let prefill = make_prefill_http("http://prefill-1:8000", "m", Some(8998));
    let decode = make_decode_http("http://decode-1:8000", "m");
    let src = MockWorkerSource::new()
        .add_worker(prefill.clone())
        .add_worker(decode.clone());
    let policies = MockPolicySource::new();
    let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

    let descriptor = RequestDescriptor {
        model_id: Some("m"),
        protocol: Some(Protocol::Http),
        ..Default::default()
    };
    let plan = planner.plan(&descriptor).await.expect("plan");
    let (p, d) = match plan {
        PlacementPlan::Pair { prefill, decode, .. } => (prefill, decode),
        _ => panic!("expected Pair"),
    };
    assert_eq!(p.url(), "http://prefill-1:8000");
    assert_eq!(d.url(), "http://decode-1:8000");

    let adapter = SglangAdapter;
    let ctx = adapter.prepare_pair(p.as_ref(), d.as_ref()).expect("prepare");
    let mut prefill_body = json!({"prompt": "hello"});
    let mut decode_body = prefill_body.clone();
    adapter
        .inject_prefill_fields(&mut prefill_body, &ctx)
        .unwrap();
    adapter
        .inject_decode_fields(&mut decode_body, &ctx)
        .unwrap();
    assert_eq!(prefill_body["bootstrap_host"], json!("prefill-1"));
    assert_eq!(prefill_body["bootstrap_port"], json!(8998));
    assert!(prefill_body["bootstrap_room"].is_u64());
    assert_eq!(decode_body, json!({"prompt": "hello"}));
}

#[tokio::test]
async fn h08_http_pd_vllm_pair_uses_shared_transfer_id() {
    use std::collections::HashMap;
    use std::sync::Arc;
    use serde_json::json;
    use crate::core::placement::backend::vllm::{VllmAdapter, VllmPrefillInfo};
    use crate::core::placement::backend::BackendAdapter;
    use crate::core::placement::planner::DefaultPlanner;
    use crate::core::placement::traits::PdPlanner;
    use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};

    let prefill_url = "http://prefill-1:8000";
    let prefill = make_prefill_http(prefill_url, "m", None);
    let decode = make_decode_http("http://decode-1:8000", "m");
    let src = MockWorkerSource::new()
        .add_worker(prefill.clone())
        .add_worker(decode.clone());
    let policies = MockPolicySource::new();
    let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

    let descriptor = RequestDescriptor {
        model_id: Some("m"),
        protocol: Some(Protocol::Http),
        ..Default::default()
    };
    let plan = planner.plan(&descriptor).await.expect("plan");
    let (p, d) = match plan {
        PlacementPlan::Pair { prefill, decode, .. } => (prefill, decode),
        _ => panic!("expected Pair"),
    };

    let mut bootstrap_addrs = HashMap::new();
    bootstrap_addrs.insert(prefill_url.to_string(), "http://10.0.0.1:9000".to_string());
    let mut engine_ids = HashMap::new();
    let mut per_rank = HashMap::new();
    per_rank.insert(0usize, "engine-xyz".to_string());
    engine_ids.insert(prefill_url.to_string(), per_rank);
    let info = Arc::new(VllmPrefillInfo {
        bootstrap_addrs,
        engine_ids,
    });
    let adapter = VllmAdapter::new(info);
    let ctx = adapter.prepare_pair(p.as_ref(), d.as_ref()).expect("prepare");

    let mut prefill_body = json!({"prompt": "hi"});
    let mut decode_body = prefill_body.clone();
    adapter
        .inject_prefill_fields(&mut prefill_body, &ctx)
        .unwrap();
    adapter
        .inject_decode_fields(&mut decode_body, &ctx)
        .unwrap();

    let prefill_kv = &prefill_body["kv_transfer_params"];
    let decode_kv = &decode_body["kv_transfer_params"];
    assert_eq!(prefill_kv["do_remote_decode"], json!(true));
    assert_eq!(prefill_kv["do_remote_prefill"], json!(false));
    assert_eq!(decode_kv["do_remote_decode"], json!(false));
    assert_eq!(decode_kv["do_remote_prefill"], json!(true));
    assert_eq!(decode_kv["remote_engine_id"], json!("engine-xyz"));
    let pid = prefill_kv["transfer_id"].as_str().unwrap();
    let did = decode_kv["transfer_id"].as_str().unwrap();
    assert_eq!(pid, did);
}

#[tokio::test]
async fn h09_http_pd_sglang_batch_writes_length_n_arrays() {
    use std::sync::Arc;
    use serde_json::json;
    use crate::core::placement::backend::sglang::SglangAdapter;
    use crate::core::placement::backend::BackendAdapter;
    use crate::core::placement::planner::DefaultPlanner;
    use crate::core::placement::traits::PdPlanner;
    use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};

    let prefill = make_prefill_http("http://prefill-1:8000", "m", Some(8998));
    let decode = make_decode_http("http://decode-1:8000", "m");
    let src = MockWorkerSource::new()
        .add_worker(prefill)
        .add_worker(decode);
    let policies = MockPolicySource::new();
    let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

    let descriptor = RequestDescriptor {
        model_id: Some("m"),
        protocol: Some(Protocol::Http),
        ..Default::default()
    };
    let (p, d) = match planner.plan(&descriptor).await.expect("plan") {
        PlacementPlan::Pair { prefill, decode, .. } => (prefill, decode),
        _ => panic!("expected Pair"),
    };

    let adapter = SglangAdapter;
    let ctx = adapter.prepare_pair(p.as_ref(), d.as_ref()).unwrap();
    let mut body = json!({"prompt": ["a", "b", "c"]});
    adapter
        .inject_batch_prefill_fields(&mut body, &ctx, 3)
        .unwrap();
    assert_eq!(body["bootstrap_host"].as_array().unwrap().len(), 3);
    assert_eq!(body["bootstrap_port"].as_array().unwrap().len(), 3);
    assert_eq!(body["bootstrap_room"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn h10_http_pd_retry_preserves_text_headers_tokens() {
    use std::sync::Arc;
    use http::HeaderMap;
    use crate::core::placement::planner::DefaultPlanner;
    use crate::core::placement::traits::PdPlanner;
    use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};

    let prefill = make_prefill_http("http://prefill-1:8000", "m", Some(8998));
    let decode = make_decode_http("http://decode-1:8000", "m");
    let src = MockWorkerSource::new()
        .add_worker(prefill)
        .add_worker(decode);
    let recording = Arc::new(RecordingPolicy::round_robin());
    let policies = MockPolicySource::new()
        .with_prefill(recording.clone())
        .with_decode(recording.clone());
    let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

    let mut headers = HeaderMap::new();
    headers.insert("x-trace", "abc".parse().unwrap());
    let tokens = vec![10u32, 20, 30];
    let descriptor = RequestDescriptor {
        model_id: Some("m"),
        protocol: Some(Protocol::Http),
        text: Some("hello"),
        tokens: Some(&tokens),
        headers: Some(&headers),
        stream: false,
        return_logprob: false,
    };

    for _ in 0..2 {
        let plan = planner.plan(&descriptor).await.expect("plan");
        assert!(matches!(plan, PlacementPlan::Pair { .. }));
    }

    let calls = recording.calls();
    assert_eq!(calls.len(), 4);
    for call in &calls {
        assert_eq!(call.request_text.as_deref(), Some("hello"));
        assert_eq!(call.tokens.as_deref(), Some(&[10u32, 20, 30][..]));
        assert_eq!(
            call.headers.as_ref().and_then(|h| h.get("x-trace")),
            Some(&"abc".parse().unwrap())
        );
    }
}
