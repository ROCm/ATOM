use std::sync::Arc;

use super::planner::DefaultPlanner;
use super::test_support::*;
use super::traits::{PdPlanner, PolicySource, WorkerSource};
use super::types::{PlacementError, PlacementPlan, Protocol, RequestDescriptor};
use crate::policies::{LoadBalancingPolicy, PrefixHashConfig, PrefixHashPolicy};

fn make_planner(src: MockWorkerSource, policies: MockPolicySource) -> DefaultPlanner {
    DefaultPlanner::new(
        Arc::new(src) as Arc<dyn WorkerSource>,
        Arc::new(policies) as Arc<dyn PolicySource>,
    )
}

#[tokio::test]
async fn c01_single_worker_for_specified_model() {
    let src = MockWorkerSource::new().add_worker(make_regular_http("http://m1-w:8000", "m1"));
    let planner = make_planner(src, MockPolicySource::new());

    let plan = planner
        .plan(&make_descriptor(Some("m1"), None, None, None))
        .await
        .unwrap();

    match plan {
        PlacementPlan::Single { worker, .. } => {
            assert_eq!(worker.url(), "http://m1-w:8000");
        }
        other => panic!("expected Single, got {:?}", other),
    }
}

#[tokio::test]
async fn c02_model_id_none_falls_back_to_default_policy() {
    let src = MockWorkerSource::new().add_worker(make_regular_http("http://w:8000", "m"));
    let planner = make_planner(src, MockPolicySource::new());

    let plan = planner
        .plan(&make_descriptor(None, None, None, None))
        .await
        .unwrap();

    match plan {
        PlacementPlan::Single { worker, .. } => assert_eq!(worker.url(), "http://w:8000"),
        other => panic!("expected Single, got {:?}", other),
    }
}

#[tokio::test]
async fn c03_model_not_found_returns_typed_error() {
    let src = MockWorkerSource::new().add_worker(make_regular_http("http://w:8000", "m_present"));
    let planner = make_planner(src, MockPolicySource::new());

    let err = planner
        .plan(&make_descriptor(Some("m_missing"), None, None, None))
        .await
        .unwrap_err();

    assert_eq!(
        err,
        PlacementError::ModelNotFound {
            model_id: "m_missing".to_string()
        }
    );
}

#[tokio::test]
async fn c04_cross_model_isolation_100_iterations() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://m1-a:8000", "m1"))
        .add_worker(make_regular_http("http://m1-b:8000", "m1"))
        .add_worker(make_regular_http("http://m2-a:8000", "m2"))
        .add_worker(make_regular_http("http://m2-b:8000", "m2"));
    let planner = make_planner(src, MockPolicySource::new());

    for _ in 0..100 {
        let plan = planner
            .plan(&make_descriptor(Some("m1"), None, None, None))
            .await
            .unwrap();
        match plan {
            PlacementPlan::Single { worker, .. } => {
                assert_eq!(worker.model_id(), "m1", "leaked m2 worker into m1 routing");
            }
            other => panic!("expected Single, got {:?}", other),
        }
    }
}

#[tokio::test]
async fn c05_hash_ring_called_with_real_model_id() {
    let workers = vec![
        make_regular_http("http://w1:8000", "m1"),
        make_regular_http("http://w2:8000", "m1"),
    ];
    let ring = Arc::new(crate::core::HashRing::new(&workers));
    let src = MockWorkerSource::new()
        .add_worker(workers[0].clone())
        .add_worker(workers[1].clone())
        .with_hash_ring("m1", ring);
    let call_log = src.hash_ring_calls.clone();

    let policies = MockPolicySource::new()
        .with_regular(Arc::new(PrefixHashPolicy::new(PrefixHashConfig::default())));
    let planner = make_planner(src, policies);

    let _ = planner
        .plan(&RequestDescriptor {
            model_id: Some("m1"),
            protocol: None,
            text: None,
            tokens: Some(&[1u32, 2, 3]),
            headers: None,
            stream: false,
        })
        .await
        .unwrap();

    let calls = call_log.lock().unwrap().clone();
    assert!(
        calls.iter().any(|c| c == "m1"),
        "hash_ring not queried with real model_id; calls={:?}",
        calls
    );
    assert!(
        calls.iter().all(|c| c != crate::core::UNKNOWN_MODEL_ID),
        "hash_ring queried with UNKNOWN_MODEL_ID; calls={:?}",
        calls
    );
}

#[tokio::test]
async fn c06_grpc_single_worker_excludes_http() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://h:8000", "m"))
        .add_worker(make_regular_grpc("http://g:8000", "m"));
    let planner = make_planner(src, MockPolicySource::new());

    let plan = planner
        .plan(&RequestDescriptor {
            model_id: Some("m"),
            protocol: Some(Protocol::Grpc),
            text: None,
            tokens: None,
            headers: None,
            stream: false,
        })
        .await
        .unwrap();

    match plan {
        PlacementPlan::Single { worker, .. } => assert_eq!(worker.url(), "http://g:8000"),
        other => panic!("expected gRPC Single, got {:?}", other),
    }
}

#[tokio::test]
async fn c07_grpc_model_id_none_falls_back_to_default() {
    let src = MockWorkerSource::new().add_worker(make_regular_grpc("http://g:8000", "m"));
    let planner = make_planner(src, MockPolicySource::new());

    let plan = planner
        .plan(&RequestDescriptor {
            model_id: None,
            protocol: Some(Protocol::Grpc),
            text: None,
            tokens: None,
            headers: None,
            stream: false,
        })
        .await
        .unwrap();

    match plan {
        PlacementPlan::Single { worker, .. } => assert_eq!(worker.url(), "http://g:8000"),
        other => panic!("expected Single, got {:?}", other),
    }
}

#[tokio::test]
async fn c08_all_unhealthy_returns_no_available_workers() {
    let w1 = make_regular_http("http://w1:8000", "m");
    let w2 = make_regular_http("http://w2:8000", "m");
    w1.set_healthy(false);
    w2.set_healthy(false);
    let src = MockWorkerSource::new().add_worker(w1).add_worker(w2);
    let planner = make_planner(src, MockPolicySource::new());

    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::NoAvailableWorkers);
}

#[tokio::test]
async fn c09_empty_registry_model_id_none_returns_no_workers() {
    let src = MockWorkerSource::new();
    let planner = make_planner(src, MockPolicySource::new());

    let err = planner
        .plan(&make_descriptor(None, None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::NoWorkers);
}

#[tokio::test]
async fn c10_pd_mode_all_unhealthy_returns_no_available_workers() {
    let p = make_prefill_http("http://p:8000", "m", Some(8998));
    let d = make_decode_http("http://d:8000", "m");
    p.set_healthy(false);
    d.set_healthy(false);
    let src = MockWorkerSource::new().add_worker(p).add_worker(d);
    let planner = make_planner(src, MockPolicySource::new());

    let err = planner
        .plan(&make_descriptor(None, None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::NoAvailableWorkers);
}

#[tokio::test]
async fn c11_policy_returned_none_propagates() {
    let src = MockWorkerSource::new().add_worker(make_regular_http("http://w:8000", "m"));
    let policies = MockPolicySource::new()
        .with_regular(Arc::new(AlwaysNonePolicy) as Arc<dyn LoadBalancingPolicy>);
    let planner = make_planner(src, policies);

    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::PolicyReturnedNone);
}
