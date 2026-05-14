use std::sync::Arc;

use http::HeaderMap;

use super::planner::DefaultPlanner;
use super::test_support::*;
use super::traits::{PdPlanner, PolicySource, WorkerSource};
use super::types::{PlacementError, PlacementPlan, Protocol, RequestDescriptor};
use crate::core::WorkerType;
use crate::policies::{LoadBalancingPolicy, RandomPolicy, RoundRobinPolicy};

fn make_planner(src: MockWorkerSource, policies: MockPolicySource) -> DefaultPlanner {
    DefaultPlanner::new(
        Arc::new(src) as Arc<dyn WorkerSource>,
        Arc::new(policies) as Arc<dyn PolicySource>,
    )
}

fn one_p_one_d_http(model: &str) -> MockWorkerSource {
    MockWorkerSource::new()
        .add_worker(make_prefill_http(
            &format!("http://{}-p:8000", model),
            model,
            Some(8998),
        ))
        .add_worker(make_decode_http(
            &format!("http://{}-d:8000", model),
            model,
        ))
}

fn one_p_one_d_grpc(model: &str) -> MockWorkerSource {
    MockWorkerSource::new()
        .add_worker(make_prefill_grpc(
            &format!("http://{}-p:8000", model),
            model,
            Some(8998),
        ))
        .add_worker(make_decode_grpc(
            &format!("http://{}-d:8000", model),
            model,
        ))
}

#[tokio::test]
async fn d01_http_1p1d_pair() {
    let planner = make_planner(one_p_one_d_http("m"), MockPolicySource::new());
    let plan = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap();

    match plan {
        PlacementPlan::Pair {
            prefill, decode, ..
        } => {
            assert_eq!(prefill.url(), "http://m-p:8000");
            assert_eq!(decode.url(), "http://m-d:8000");
        }
        other => panic!("expected Pair, got {:?}", other),
    }
}

#[tokio::test]
async fn d02_grpc_1p1d_pair() {
    let planner = make_planner(one_p_one_d_grpc("m"), MockPolicySource::new());
    let plan = planner
        .plan(&RequestDescriptor {
            model_id: Some("m"),
            protocol: Some(Protocol::Grpc),
            text: None,
            tokens: None,
            headers: None,
            stream: false,
            return_logprob: false,
        })
        .await
        .unwrap();

    match plan {
        PlacementPlan::Pair {
            prefill, decode, ..
        } => {
            assert_eq!(prefill.url(), "http://m-p:8000");
            assert_eq!(decode.url(), "http://m-d:8000");
        }
        other => panic!("expected Pair, got {:?}", other),
    }
}

#[tokio::test]
async fn d03_pd_cross_model_isolation() {
    let src = MockWorkerSource::new()
        .add_worker(make_prefill_http("http://m1-p:8000", "m1", Some(8998)))
        .add_worker(make_decode_http("http://m1-d:8000", "m1"))
        .add_worker(make_prefill_http("http://m2-p:8000", "m2", Some(8998)))
        .add_worker(make_decode_http("http://m2-d:8000", "m2"));
    let planner = make_planner(src, MockPolicySource::new());

    for model in ["m1", "m2", "m1"] {
        let plan = planner
            .plan(&make_descriptor(Some(model), None, None, None))
            .await
            .unwrap();
        match plan {
            PlacementPlan::Pair {
                prefill, decode, ..
            } => {
                assert_eq!(prefill.model_id(), model);
                assert_eq!(decode.model_id(), model);
            }
            other => panic!("expected Pair, got {:?}", other),
        }
    }
}

#[tokio::test]
async fn d04_pd_hash_ring_keyed_by_real_model_id() {
    let src = MockWorkerSource::new()
        .add_worker(make_prefill_http("http://m1-p:8000", "m1", Some(8998)))
        .add_worker(make_decode_http("http://m1-d:8000", "m1"));
    let call_log = src.hash_ring_calls.clone();
    let planner = make_planner(src, MockPolicySource::new());

    let _ = planner
        .plan(&make_descriptor(Some("m1"), None, None, None))
        .await
        .unwrap();

    let calls = call_log.lock().unwrap().clone();
    assert!(
        calls.iter().any(|c| c == "m1"),
        "hash_ring not queried with model_id; calls={:?}",
        calls
    );
    assert!(
        calls.iter().all(|c| c != crate::core::UNKNOWN_MODEL_ID),
        "hash_ring queried with UNKNOWN_MODEL_ID; calls={:?}",
        calls
    );
}

#[tokio::test]
async fn d05_zero_prefill_returns_no_prefill_workers() {
    let src = MockWorkerSource::new().add_worker(make_decode_http("http://d:8000", "m"));
    let planner = make_planner(src, MockPolicySource::new());

    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::NoPrefillWorkers);
}

#[tokio::test]
async fn d06_zero_decode_returns_no_decode_workers() {
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
async fn d07_grpc_pd_uses_separated_policies() {
    let prefill_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let decode_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RandomPolicy::new())));

    let prefill_calls = prefill_recorder.calls.clone();
    let decode_calls = decode_recorder.calls.clone();

    let policies = MockPolicySource::new()
        .with_prefill(prefill_recorder as Arc<dyn LoadBalancingPolicy>)
        .with_decode(decode_recorder as Arc<dyn LoadBalancingPolicy>);

    let planner = make_planner(one_p_one_d_grpc("m"), policies);
    let _ = planner
        .plan(&RequestDescriptor {
            model_id: Some("m"),
            protocol: Some(Protocol::Grpc),
            text: None,
            tokens: None,
            headers: None,
            stream: false,
            return_logprob: false,
        })
        .await
        .unwrap();

    assert_eq!(prefill_calls.lock().unwrap().len(), 1);
    assert_eq!(decode_calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn d08_http_pd_uses_separated_policies() {
    let prefill_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let decode_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RandomPolicy::new())));

    let prefill_calls = prefill_recorder.calls.clone();
    let decode_calls = decode_recorder.calls.clone();

    let policies = MockPolicySource::new()
        .with_prefill(prefill_recorder as Arc<dyn LoadBalancingPolicy>)
        .with_decode(decode_recorder as Arc<dyn LoadBalancingPolicy>);

    let planner = make_planner(one_p_one_d_http("m"), policies);
    let _ = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap();

    assert_eq!(prefill_calls.lock().unwrap().len(), 1);
    assert_eq!(decode_calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn d09_prefill_none_short_circuits_decode() {
    let decode_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let decode_calls = decode_recorder.calls.clone();

    let policies = MockPolicySource::new()
        .with_prefill(Arc::new(AlwaysNonePolicy) as Arc<dyn LoadBalancingPolicy>)
        .with_decode(decode_recorder as Arc<dyn LoadBalancingPolicy>);

    let planner = make_planner(one_p_one_d_http("m"), policies);
    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();

    assert_eq!(err, PlacementError::PolicyReturnedNone);
    assert_eq!(
        decode_calls.lock().unwrap().len(),
        0,
        "decode policy must not be called after prefill returns None"
    );
}

#[tokio::test]
async fn d10_decode_none_returns_policy_returned_none_with_prefill_in_trace() {
    let prefill_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let prefill_calls = prefill_recorder.calls.clone();

    let policies = MockPolicySource::new()
        .with_prefill(prefill_recorder as Arc<dyn LoadBalancingPolicy>)
        .with_decode(Arc::new(AlwaysNonePolicy) as Arc<dyn LoadBalancingPolicy>);

    let planner = make_planner(one_p_one_d_http("m"), policies);
    let err = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap_err();

    assert_eq!(err, PlacementError::PolicyReturnedNone);
    assert_eq!(
        prefill_calls.lock().unwrap().len(),
        1,
        "prefill must be selected before decode policy is consulted"
    );
}

#[tokio::test]
async fn d11_tokens_pass_to_both_pd_policies() {
    let prefill_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let decode_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let prefill_calls = prefill_recorder.calls.clone();
    let decode_calls = decode_recorder.calls.clone();

    let policies = MockPolicySource::new()
        .with_prefill(prefill_recorder as Arc<dyn LoadBalancingPolicy>)
        .with_decode(decode_recorder as Arc<dyn LoadBalancingPolicy>);

    let planner = make_planner(one_p_one_d_http("m"), policies);
    let tokens = [11u32, 22, 33];
    let _ = planner
        .plan(&make_descriptor(Some("m"), None, Some(&tokens), None))
        .await
        .unwrap();

    assert_eq!(
        prefill_calls.lock().unwrap()[0].tokens.as_deref(),
        Some(&tokens[..])
    );
    assert_eq!(
        decode_calls.lock().unwrap()[0].tokens.as_deref(),
        Some(&tokens[..])
    );
}

#[tokio::test]
async fn d12_text_passes_to_both_pd_policies() {
    let prefill_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let decode_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let prefill_calls = prefill_recorder.calls.clone();
    let decode_calls = decode_recorder.calls.clone();

    let policies = MockPolicySource::new()
        .with_prefill(prefill_recorder as Arc<dyn LoadBalancingPolicy>)
        .with_decode(decode_recorder as Arc<dyn LoadBalancingPolicy>);

    let planner = make_planner(one_p_one_d_http("m"), policies);
    let _ = planner
        .plan(&make_descriptor(Some("m"), Some("hello"), None, None))
        .await
        .unwrap();

    assert_eq!(
        prefill_calls.lock().unwrap()[0].request_text.as_deref(),
        Some("hello")
    );
    assert_eq!(
        decode_calls.lock().unwrap()[0].request_text.as_deref(),
        Some("hello")
    );
}

#[tokio::test]
async fn d13_headers_pass_to_both_pd_policies() {
    let prefill_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let decode_recorder = Arc::new(RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new())));
    let prefill_calls = prefill_recorder.calls.clone();
    let decode_calls = decode_recorder.calls.clone();

    let policies = MockPolicySource::new()
        .with_prefill(prefill_recorder as Arc<dyn LoadBalancingPolicy>)
        .with_decode(decode_recorder as Arc<dyn LoadBalancingPolicy>);

    let planner = make_planner(one_p_one_d_http("m"), policies);
    let mut hm = HeaderMap::new();
    hm.insert("x-trace", "abc".parse().unwrap());
    let _ = planner
        .plan(&make_descriptor(Some("m"), None, None, Some(&hm)))
        .await
        .unwrap();

    let p_headers = prefill_calls.lock().unwrap()[0].headers.clone().unwrap();
    let d_headers = decode_calls.lock().unwrap()[0].headers.clone().unwrap();
    assert_eq!(p_headers.get("x-trace"), Some(&"abc".parse().unwrap()));
    assert_eq!(d_headers.get("x-trace"), Some(&"abc".parse().unwrap()));
}

#[tokio::test]
async fn d14_pair_preserves_prefill_bootstrap_port() {
    let planner = make_planner(one_p_one_d_http("m"), MockPolicySource::new());
    let plan = planner
        .plan(&make_descriptor(Some("m"), None, None, None))
        .await
        .unwrap();

    match plan {
        PlacementPlan::Pair { prefill, .. } => match prefill.worker_type() {
            WorkerType::Prefill { bootstrap_port } => {
                assert_eq!(*bootstrap_port, Some(8998));
            }
            other => panic!("expected Prefill worker_type, got {:?}", other),
        },
        other => panic!("expected Pair, got {:?}", other),
    }
}
