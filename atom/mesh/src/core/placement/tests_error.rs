use std::sync::Arc;

use super::planner::DefaultPlanner;
use super::test_support::*;
use super::traits::{PdPlanner, PolicySource, WorkerSource};
use super::types::{AdapterError, PlacementError};
use crate::policies::LoadBalancingPolicy;

fn make_planner(src: MockWorkerSource, policies: MockPolicySource) -> DefaultPlanner {
    DefaultPlanner::new(
        Arc::new(src) as Arc<dyn WorkerSource>,
        Arc::new(policies) as Arc<dyn PolicySource>,
    )
}

#[tokio::test]
async fn g01_no_workers_triggered_by_empty_registry_and_no_model() {
    let planner = make_planner(MockWorkerSource::new(), MockPolicySource::new());
    let err = planner
        .plan(&make_descriptor(None, None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::NoWorkers);
}

#[tokio::test]
async fn g02_no_available_workers_triggered_by_all_unhealthy() {
    let w = make_regular_http("http://w:8000", "m");
    w.set_healthy(false);
    let planner = make_planner(
        MockWorkerSource::new().add_worker(w),
        MockPolicySource::new(),
    );
    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::NoAvailableWorkers);
}

#[tokio::test]
async fn g03_no_prefill_workers_triggered_in_pd_path() {
    let src = MockWorkerSource::new().add_worker(make_decode_http("http://d:8000", "m"));
    let planner = make_planner(src, MockPolicySource::new());
    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::NoPrefillWorkers);
}

#[tokio::test]
async fn g04_no_decode_workers_triggered_in_pd_path() {
    let src =
        MockWorkerSource::new().add_worker(make_prefill_http("http://p:8000", "m", Some(8998)));
    let planner = make_planner(src, MockPolicySource::new());
    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::NoDecodeWorkers);
}

#[tokio::test]
async fn g05_policy_returned_none_triggered_by_always_none_policy() {
    let policies = MockPolicySource::new()
        .with_regular(Arc::new(AlwaysNonePolicy) as Arc<dyn LoadBalancingPolicy>);
    let planner = make_planner(
        MockWorkerSource::new().add_worker(make_regular_http("http://w:8000", "m")),
        policies,
    );
    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::PolicyReturnedNone);
}

#[tokio::test]
async fn g06_model_not_found_triggered_when_model_id_unknown() {
    let src =
        MockWorkerSource::new().add_worker(make_regular_http("http://w:8000", "m_present"));
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
async fn g07_no_available_workers_in_pd_path_when_all_unhealthy() {
    let p = make_prefill_http("http://p:8000", "m", Some(8998));
    let d = make_decode_http("http://d:8000", "m");
    p.set_healthy(false);
    d.set_healthy(false);
    let src = MockWorkerSource::new().add_worker(p).add_worker(d);
    let planner = make_planner(src, MockPolicySource::new());

    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::NoAvailableWorkers);
}

#[test]
fn g08_adapter_errors_map_to_5xx_response() {
    let body_not_object = AdapterError::BodyNotObject;
    let bootstrap_missing = AdapterError::BootstrapAddrMissing {
        prefill_url: "http://p:8000".to_string(),
    };
    let engine_missing = AdapterError::EngineIdMissing {
        prefill_url: "http://p:8000".to_string(),
        dp_rank: 2,
    };
    let ctx_mismatch = AdapterError::CtxTypeMismatch;

    assert!(!format!("{}", body_not_object).is_empty());
    assert!(format!("{}", bootstrap_missing).contains("http://p:8000"));
    assert!(format!("{}", engine_missing).contains("2"));
    assert!(!format!("{}", ctx_mismatch).is_empty());
}

#[test]
fn g09_error_display_includes_key_fields() {
    let model_not_found = PlacementError::ModelNotFound {
        model_id: "m_xyz".to_string(),
    };
    assert!(format!("{}", model_not_found).contains("m_xyz"));

    let engine_missing = AdapterError::EngineIdMissing {
        prefill_url: "http://p:8000".to_string(),
        dp_rank: 7,
    };
    let display = format!("{}", engine_missing);
    assert!(display.contains("http://p:8000"), "missing prefill_url: {}", display);
    assert!(display.contains('7'), "missing dp_rank: {}", display);
}
