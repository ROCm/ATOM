use std::sync::Arc;

use http::HeaderMap;

use super::policy_apply::apply_policy;
use super::test_support::*;
use super::types::PlacementError;
use crate::core::{HashRing, Worker};
use crate::policies::{
    CacheAwarePolicy, LoadBalancingPolicy, PowerOfTwoPolicy, PrefixHashConfig, PrefixHashPolicy,
    RandomPolicy, RoundRobinPolicy,
};

fn two_workers() -> Vec<Arc<dyn Worker>> {
    vec![
        make_regular_http("http://w1:8000", "m"),
        make_regular_http("http://w2:8000", "m"),
    ]
}

#[tokio::test]
async fn b01_round_robin_invoked_once_idx_in_range() {
    let candidates = two_workers();
    let recorder = RecordingPolicy::wrap(Arc::new(RoundRobinPolicy::new()));
    let descriptor = make_descriptor(Some("m"), None, None, None);

    let chosen = apply_policy(&candidates, &recorder, &descriptor, None)
        .await
        .unwrap();
    assert_eq!(recorder.call_count(), 1);
    assert!(candidates.iter().any(|w| Arc::ptr_eq(w, &chosen)));
}

#[tokio::test]
async fn b02_random_invoked_once_returns_valid_idx() {
    let candidates = two_workers();
    let recorder = RecordingPolicy::wrap(Arc::new(RandomPolicy::new()));
    let descriptor = make_descriptor(Some("m"), None, None, None);

    let chosen = apply_policy(&candidates, &recorder, &descriptor, None)
        .await
        .unwrap();
    assert_eq!(recorder.call_count(), 1);
    assert!(candidates.iter().any(|w| Arc::ptr_eq(w, &chosen)));
}

#[tokio::test]
async fn b03_power_of_two_invoked_once() {
    let candidates = two_workers();
    let recorder = RecordingPolicy::wrap(Arc::new(PowerOfTwoPolicy::new()));
    let descriptor = make_descriptor(Some("m"), None, None, None);

    let chosen = apply_policy(&candidates, &recorder, &descriptor, None)
        .await
        .unwrap();
    assert_eq!(recorder.call_count(), 1);
    assert!(candidates.iter().any(|w| Arc::ptr_eq(w, &chosen)));
}

#[tokio::test]
async fn b04_cache_aware_receives_request_text() {
    let candidates = two_workers();
    let recorder = RecordingPolicy::wrap(Arc::new(CacheAwarePolicy::new()));
    let descriptor = make_descriptor(Some("m"), Some("hello world"), None, None);

    let _ = apply_policy(&candidates, &recorder, &descriptor, None).await;
    let calls = recorder.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].request_text.as_deref(), Some("hello world"));
    assert!(recorder.needs_request_text());
}

#[tokio::test]
async fn b05_prefix_hash_receives_tokens() {
    let candidates = two_workers();
    let policy = PrefixHashPolicy::new(PrefixHashConfig::default());
    let recorder = RecordingPolicy::wrap(Arc::new(policy));
    let tokens = [10u32, 20, 30];
    let descriptor = make_descriptor(Some("m"), None, Some(&tokens), None);

    let _ = apply_policy(&candidates, &recorder, &descriptor, None).await;
    let calls = recorder.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tokens.as_deref(), Some(&tokens[..]));
}

#[tokio::test]
async fn b06_hash_ring_keyed_by_real_model_id() {
    let candidates = two_workers();
    let ring = Arc::new(HashRing::new(&candidates));
    let recorder = RecordingPolicy::round_robin();
    let descriptor = make_descriptor(Some("m1"), None, None, None);

    let _ = apply_policy(&candidates, &recorder, &descriptor, Some(ring.clone()))
        .await
        .unwrap();
    let calls = recorder.calls();
    assert_eq!(calls.len(), 1);
    let received = calls[0].hash_ring.as_ref().expect("hash_ring forwarded");
    assert!(Arc::ptr_eq(received, &ring));
}

#[tokio::test]
async fn b07_no_hash_ring_for_model_falls_back_gracefully() {
    let candidates = two_workers();
    let recorder = RecordingPolicy::round_robin();
    let descriptor = make_descriptor(Some("m_no_ring"), None, None, None);

    let chosen = apply_policy(&candidates, &recorder, &descriptor, None)
        .await
        .unwrap();
    let calls = recorder.calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].hash_ring.is_none());
    assert!(candidates.iter().any(|w| Arc::ptr_eq(w, &chosen)));
}

#[tokio::test]
async fn b08_request_text_passes_through() {
    let candidates = two_workers();
    let recorder = RecordingPolicy::round_robin();
    let descriptor = make_descriptor(Some("m"), Some("hello"), None, None);

    let _ = apply_policy(&candidates, &recorder, &descriptor, None).await;
    let calls = recorder.calls();
    assert_eq!(calls[0].request_text.as_deref(), Some("hello"));
}

#[tokio::test]
async fn b09_tokens_pass_through() {
    let candidates = two_workers();
    let recorder = RecordingPolicy::round_robin();
    let tokens = [1u32, 2, 3];
    let descriptor = make_descriptor(Some("m"), None, Some(&tokens), None);

    let _ = apply_policy(&candidates, &recorder, &descriptor, None).await;
    let calls = recorder.calls();
    assert_eq!(calls[0].tokens.as_deref(), Some(&tokens[..]));
}

#[tokio::test]
async fn b10_headers_pass_through() {
    let candidates = two_workers();
    let recorder = RecordingPolicy::round_robin();
    let mut hm = HeaderMap::new();
    hm.insert("x-trace", "abc".parse().unwrap());
    let descriptor = make_descriptor(Some("m"), None, None, Some(&hm));

    let _ = apply_policy(&candidates, &recorder, &descriptor, None).await;
    let calls = recorder.calls();
    let received = calls[0].headers.as_ref().expect("headers passed through");
    assert_eq!(received.get("x-trace"), Some(&"abc".parse().unwrap()));
}

#[tokio::test]
async fn b11_policy_returns_none_yields_typed_error() {
    let candidates = two_workers();
    let descriptor = make_descriptor(Some("m"), None, None, None);

    let err = apply_policy(&candidates, &AlwaysNonePolicy, &descriptor, None)
        .await
        .unwrap_err();
    assert_eq!(err, PlacementError::PolicyReturnedNone);
}

#[tokio::test]
async fn b12_empty_candidates_does_not_call_policy() {
    let candidates: Vec<Arc<dyn Worker>> = Vec::new();
    let recorder = RecordingPolicy::round_robin();
    let descriptor = make_descriptor(Some("m"), None, None, None);

    let err = apply_policy(&candidates, &recorder, &descriptor, None)
        .await
        .unwrap_err();
    assert_eq!(recorder.call_count(), 0);
    assert_eq!(err, PlacementError::NoAvailableWorkers);
}

#[test]
fn b13_pd_needs_request_text_aggregated_from_policies() {
    use super::traits::PolicySource;

    let yes: Arc<dyn LoadBalancingPolicy> = Arc::new(StaticNeedsTextPolicy::new("yes", true));
    let no: Arc<dyn LoadBalancingPolicy> = Arc::new(StaticNeedsTextPolicy::new("no", false));

    let none_needs = MockPolicySource::new()
        .with_prefill(no.clone())
        .with_decode(no.clone());
    assert!(!none_needs.pd_needs_request_text());

    let prefill_needs = MockPolicySource::new()
        .with_prefill(yes.clone())
        .with_decode(no.clone());
    assert!(prefill_needs.pd_needs_request_text());

    let decode_needs = MockPolicySource::new().with_prefill(no).with_decode(yes);
    assert!(decode_needs.pd_needs_request_text());
}

#[tokio::test]
async fn b14_returned_worker_belongs_to_candidate_set() {
    let candidates = two_workers();
    let candidate_urls: Vec<String> = candidates.iter().map(|w| w.url().to_string()).collect();
    let descriptor = make_descriptor(Some("m"), None, None, None);
    let policy = RandomPolicy::new();

    for _ in 0..32 {
        let chosen = apply_policy(&candidates, &policy, &descriptor, None)
            .await
            .unwrap();
        assert!(candidate_urls.iter().any(|u| u == chosen.url()));
    }
}
