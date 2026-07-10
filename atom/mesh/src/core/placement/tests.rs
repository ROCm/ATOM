mod a_candidate {
    use std::sync::Arc;

    use super::super::test_support::*;
    use super::super::traits::WorkerSource;
    use crate::core::{ConnectionMode, Worker, WorkerType};

    fn filter_candidates(
        src: &dyn WorkerSource,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
    ) -> Vec<Arc<dyn Worker>> {
        src.workers_filtered(model_id, worker_type, connection_mode)
            .into_iter()
            .filter(|w| w.is_available())
            .collect()
    }

    #[test]
    fn test_filters_to_specified_model_only() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://m1-w1:8000", "m1"))
            .add_worker(make_regular_http("http://m1-w2:8000", "m1"))
            .add_worker(make_regular_http("http://m2-w1:8000", "m2"));

        let result = filter_candidates(&src, Some("m1"), None, None);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|w| w.model_id() == "m1"));
    }

    #[test]
    fn test_model_id_none_returns_all_workers() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://w1:8000", "m1"))
            .add_worker(make_regular_http("http://w2:8000", "m2"));

        let result = filter_candidates(&src, None, None, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_unknown_model_returns_empty() {
        let src = MockWorkerSource::new().add_worker(make_regular_http("http://w:8000", "m1"));

        let result = filter_candidates(&src, Some("m_missing"), None, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_no_cross_model_contamination() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://m1-w:8000", "m1"))
            .add_worker(make_regular_http("http://m2-w:8000", "m2"));

        let result = filter_candidates(&src, Some("m1"), None, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url(), "http://m1-w:8000");
        assert!(result.iter().all(|w| w.model_id() != "m2"));
    }

    #[test]
    fn test_worker_type_regular_excludes_pd() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://r:8000", "m"))
            .add_worker(make_prefill_http("http://p:8000", "m", Some(8998)))
            .add_worker(make_decode_http("http://d:8000", "m"));

        let result = filter_candidates(&src, None, Some(WorkerType::Regular), None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url(), "http://r:8000");
    }

    #[test]
    fn test_worker_type_prefill_excludes_regular_and_decode() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://r:8000", "m"))
            .add_worker(make_prefill_http("http://p:8000", "m", Some(8998)))
            .add_worker(make_decode_http("http://d:8000", "m"));

        let result = filter_candidates(
            &src,
            None,
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            None,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url(), "http://p:8000");
    }

    #[test]
    fn test_worker_type_decode_excludes_regular_and_prefill() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://r:8000", "m"))
            .add_worker(make_prefill_http("http://p:8000", "m", Some(8998)))
            .add_worker(make_decode_http("http://d:8000", "m"));

        let result = filter_candidates(&src, None, Some(WorkerType::Decode), None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url(), "http://d:8000");
    }

    #[test]
    fn test_connection_mode_http_excludes_grpc() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://h:8000", "m"))
            .add_worker(make_regular_grpc("http://g:8000", "m"));

        let result = filter_candidates(&src, None, None, Some(ConnectionMode::Http));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url(), "http://h:8000");
    }

    #[test]
    fn test_connection_mode_grpc_excludes_http() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://h:8000", "m"))
            .add_worker(make_regular_grpc("http://g:8000", "m"));

        let result = filter_candidates(&src, None, None, Some(ConnectionMode::Grpc { port: None }));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url(), "http://g:8000");
    }

    #[test]
    fn test_all_healthy_all_pass() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://w1:8000", "m"))
            .add_worker(make_regular_http("http://w2:8000", "m"));

        let result = filter_candidates(&src, None, None, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_all_unhealthy_empty_set() {
        let w1 = make_regular_http("http://w1:8000", "m");
        let w2 = make_regular_http("http://w2:8000", "m");
        w1.set_healthy(false);
        w2.set_healthy(false);
        let src = MockWorkerSource::new().add_worker(w1).add_worker(w2);

        let result = filter_candidates(&src, None, None, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mixed_health_only_healthy_pass() {
        let healthy = make_regular_http("http://h:8000", "m");
        let unhealthy = make_regular_http("http://u:8000", "m");
        unhealthy.set_healthy(false);
        let src = MockWorkerSource::new()
            .add_worker(healthy)
            .add_worker(unhealthy);

        let result = filter_candidates(&src, None, None, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url(), "http://h:8000");
    }

    #[test]
    fn test_empty_registry_returns_empty_no_panic() {
        let src = MockWorkerSource::new();
        let result = filter_candidates(&src, Some("m"), None, None);
        assert!(result.is_empty());
        let result2 = filter_candidates(&src, None, None, None);
        assert!(result2.is_empty());
    }

    #[test]
    fn test_prefill_bootstrap_port_variants_both_match() {
        let src = MockWorkerSource::new()
            .add_worker(make_prefill_http("http://sg:8000", "m", Some(8998)))
            .add_worker(make_prefill_http("http://vl:8000", "m", None));

        let result = filter_candidates(
            &src,
            None,
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            None,
        );
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_combined_filters_all_apply() {
        let unhealthy = make_regular_http("http://m1-u:8000", "m1");
        unhealthy.set_healthy(false);
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://m1-h:8000", "m1"))
            .add_worker(unhealthy)
            .add_worker(make_regular_http("http://m2-h:8000", "m2"))
            .add_worker(make_regular_grpc("http://m1-g:8000", "m1"))
            .add_worker(make_prefill_http("http://m1-p:8000", "m1", Some(8998)));

        let result = filter_candidates(
            &src,
            Some("m1"),
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url(), "http://m1-h:8000");
    }

    #[test]
    fn test_model_id_some_worker_type_none_returns_all_for_model() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://m1-r:8000", "m1"))
            .add_worker(make_prefill_http("http://m1-p:8000", "m1", Some(8998)))
            .add_worker(make_decode_http("http://m1-d:8000", "m1"))
            .add_worker(make_regular_http("http://m2:8000", "m2"));

        let result = filter_candidates(&src, Some("m1"), None, None);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|w| w.model_id() == "m1"));
    }

    #[test]
    fn test_dp_aware_same_url_different_dp_rank_both_pass() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://shared:8000", "m"))
            .add_worker(make_regular_http("http://shared:8000", "m"));

        let result = filter_candidates(&src, None, None, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_filter_scales_linearly_with_worker_count() {
        let mut src = MockWorkerSource::new();
        for i in 0..256 {
            src = src.add_worker(make_regular_http(&format!("http://w{}:8000", i), "m"));
        }

        let result = filter_candidates(&src, Some("m"), Some(WorkerType::Regular), None);
        assert_eq!(result.len(), 256);

        let mut urls: Vec<&str> = result.iter().map(|w| w.url()).collect();
        urls.sort();
        urls.dedup();
        assert_eq!(
            urls.len(),
            256,
            "filter_candidates dropped or duplicated workers"
        );
    }
}

mod b_policy_apply {
    use std::sync::Arc;

    use http::HeaderMap;

    use super::super::policy_apply::apply_policy;
    use super::super::test_support::*;
    use super::super::types::PlacementError;
    use crate::core::{HashRing, Worker};
    use crate::policies::{
        CacheAwarePolicy, LoadBalancingPolicy, PowerOfTwoPolicy, PrefixHashConfig,
        PrefixHashPolicy, RandomPolicy, RoundRobinPolicy,
    };

    fn two_workers() -> Vec<Arc<dyn Worker>> {
        vec![
            make_regular_http("http://w1:8000", "m"),
            make_regular_http("http://w2:8000", "m"),
        ]
    }

    #[tokio::test]
    async fn test_round_robin_invoked_once_idx_in_range() {
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
    async fn test_random_invoked_once_returns_valid_idx() {
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
    async fn test_power_of_two_invoked_once() {
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
    async fn test_cache_aware_receives_request_text() {
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
    async fn test_prefix_hash_receives_tokens() {
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
    async fn test_hash_ring_keyed_by_real_model_id() {
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
    async fn test_no_hash_ring_for_model_falls_back_gracefully() {
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
    async fn test_request_text_passes_through() {
        let candidates = two_workers();
        let recorder = RecordingPolicy::round_robin();
        let descriptor = make_descriptor(Some("m"), Some("hello"), None, None);

        let _ = apply_policy(&candidates, &recorder, &descriptor, None).await;
        let calls = recorder.calls();
        assert_eq!(calls[0].request_text.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn test_tokens_pass_through() {
        let candidates = two_workers();
        let recorder = RecordingPolicy::round_robin();
        let tokens = [1u32, 2, 3];
        let descriptor = make_descriptor(Some("m"), None, Some(&tokens), None);

        let _ = apply_policy(&candidates, &recorder, &descriptor, None).await;
        let calls = recorder.calls();
        assert_eq!(calls[0].tokens.as_deref(), Some(&tokens[..]));
    }

    #[tokio::test]
    async fn test_headers_pass_through() {
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
    async fn test_policy_returns_none_yields_typed_error() {
        let candidates = two_workers();
        let descriptor = make_descriptor(Some("m"), None, None, None);

        let err = apply_policy(&candidates, &AlwaysNonePolicy, &descriptor, None)
            .await
            .unwrap_err();
        assert_eq!(err, PlacementError::PolicyReturnedNone);
    }

    #[tokio::test]
    async fn test_empty_candidates_does_not_call_policy() {
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
    fn test_pd_needs_request_text_aggregated_from_policies() {
        use super::super::traits::PolicySource;

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
    async fn test_returned_worker_belongs_to_candidate_set() {
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
}

mod c_regular_planning {
    use std::sync::Arc;

    use super::super::planner::DefaultPlanner;
    use super::super::test_support::*;
    use super::super::traits::{PdPlanner, PolicySource, WorkerSource};
    use super::super::types::{PlacementError, PlacementPlan, Protocol, RequestDescriptor};
    use crate::policies::{LoadBalancingPolicy, PrefixHashConfig, PrefixHashPolicy};

    fn make_planner(src: MockWorkerSource, policies: MockPolicySource) -> DefaultPlanner {
        DefaultPlanner::new(
            Arc::new(src) as Arc<dyn WorkerSource>,
            Arc::new(policies) as Arc<dyn PolicySource>,
        )
    }

    #[tokio::test]
    async fn test_single_worker_for_specified_model() {
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
    async fn test_model_id_none_falls_back_to_default_policy() {
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
    async fn test_model_not_found_returns_typed_error() {
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
    async fn test_cross_model_isolation_100_iterations() {
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
    async fn test_hash_ring_called_with_real_model_id() {
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
    async fn test_grpc_single_worker_excludes_http() {
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
    async fn test_grpc_model_id_none_falls_back_to_default() {
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
    async fn test_all_unhealthy_returns_no_available_workers() {
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
    async fn test_empty_registry_model_id_none_returns_no_workers() {
        let src = MockWorkerSource::new();
        let planner = make_planner(src, MockPolicySource::new());

        let err = planner
            .plan(&make_descriptor(None, None, None, None))
            .await
            .unwrap_err();
        assert_eq!(err, PlacementError::NoWorkers);
    }

    #[tokio::test]
    async fn test_pd_mode_all_unhealthy_returns_no_available_workers() {
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
    async fn test_policy_returned_none_propagates() {
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
}

mod d_pd_planning {
    use std::sync::Arc;

    use http::HeaderMap;

    use super::super::planner::DefaultPlanner;
    use super::super::test_support::*;
    use super::super::traits::{PdPlanner, PolicySource, WorkerSource};
    use super::super::types::{PlacementError, PlacementPlan, Protocol, RequestDescriptor};
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
            .add_worker(make_decode_http(&format!("http://{}-d:8000", model), model))
    }

    fn one_p_one_d_grpc(model: &str) -> MockWorkerSource {
        MockWorkerSource::new()
            .add_worker(make_prefill_grpc(
                &format!("http://{}-p:8000", model),
                model,
                Some(8998),
            ))
            .add_worker(make_decode_grpc(&format!("http://{}-d:8000", model), model))
    }

    #[tokio::test]
    async fn test_http_1p1d_pair() {
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
    async fn test_grpc_1p1d_pair() {
        let planner = make_planner(one_p_one_d_grpc("m"), MockPolicySource::new());
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
    async fn test_pd_cross_model_isolation() {
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
    async fn test_pd_hash_ring_keyed_by_real_model_id() {
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
    async fn test_zero_prefill_returns_no_prefill_workers() {
        let src = MockWorkerSource::new().add_worker(make_decode_http("http://d:8000", "m"));
        let planner = make_planner(src, MockPolicySource::new());

        let err = planner
            .plan(&make_descriptor(Some("m"), None, None, None))
            .await
            .unwrap_err();
        assert_eq!(err, PlacementError::NoPrefillWorkers);
    }

    #[tokio::test]
    async fn test_zero_decode_returns_no_decode_workers() {
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
    async fn test_grpc_pd_uses_separated_policies() {
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
            })
            .await
            .unwrap();

        assert_eq!(prefill_calls.lock().unwrap().len(), 1);
        assert_eq!(decode_calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_http_pd_uses_separated_policies() {
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
    async fn test_prefill_none_short_circuits_decode() {
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
    async fn test_decode_none_returns_policy_returned_none_with_prefill_in_trace() {
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
    async fn test_tokens_pass_to_both_pd_policies() {
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
    async fn test_text_passes_to_both_pd_policies() {
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
    async fn test_headers_pass_to_both_pd_policies() {
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
    async fn test_pair_preserves_prefill_bootstrap_port() {
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
}

mod e_adapter {
    use std::collections::HashMap;
    use std::sync::Arc;

    use serde_json::{json, Value};

    use super::super::backend::sglang::SglangAdapter;
    use super::super::backend::vllm::{VllmAdapter, VllmPrefillInfo};
    use super::super::backend::BackendAdapter;
    use super::super::test_support::*;
    use super::super::types::AdapterError;

    fn sglang_pair() -> (SglangAdapter, super::super::backend::PairCtx) {
        let prefill = make_prefill_http("http://prefill-1:8000", "m", Some(8998));
        let decode = make_decode_http("http://decode-1:8000", "m");
        let adapter = SglangAdapter;
        let ctx = adapter
            .prepare_pair(prefill.as_ref(), decode.as_ref())
            .expect("prepare_pair");
        (adapter, ctx)
    }

    #[test]
    fn test_sglang_inject_prefill_writes_three_keys() {
        let (adapter, ctx) = sglang_pair();
        let mut body = json!({"prompt": "hi"});
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        let obj = body.as_object().unwrap();
        assert_eq!(obj["bootstrap_host"], json!("prefill-1"));
        assert_eq!(obj["bootstrap_port"], json!(8998));
        assert!(obj["bootstrap_room"].is_u64());
        assert_eq!(obj["prompt"], json!("hi"));
    }

    #[test]
    fn test_sglang_inject_prefill_port_none_writes_null() {
        let prefill = make_prefill_http("http://prefill-2:8000", "m", None);
        let decode = make_decode_http("http://decode-2:8000", "m");
        let adapter = SglangAdapter;
        let ctx = adapter
            .prepare_pair(prefill.as_ref(), decode.as_ref())
            .unwrap();
        let mut body = json!({});
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        assert_eq!(body["bootstrap_port"], Value::Null);
        assert!(body.as_object().unwrap().contains_key("bootstrap_port"));
    }

    #[test]
    fn test_sglang_inject_decode_is_noop() {
        let (adapter, ctx) = sglang_pair();
        let mut body = json!({"prompt": "hi"});
        let before = body.clone();
        adapter.inject_decode_fields(&mut body, &ctx).unwrap();
        assert_eq!(body, before);
    }

    #[test]
    fn test_sglang_inject_batch_writes_three_arrays_of_size_n() {
        let (adapter, ctx) = sglang_pair();
        let mut body = json!({});
        adapter
            .inject_batch_prefill_fields(&mut body, &ctx, 3)
            .unwrap();
        let obj = body.as_object().unwrap();
        assert_eq!(obj["bootstrap_host"].as_array().unwrap().len(), 3);
        assert_eq!(obj["bootstrap_port"].as_array().unwrap().len(), 3);
        assert_eq!(obj["bootstrap_room"].as_array().unwrap().len(), 3);
        for v in obj["bootstrap_host"].as_array().unwrap() {
            assert_eq!(v, &json!("prefill-1"));
        }
        for v in obj["bootstrap_port"].as_array().unwrap() {
            assert_eq!(v, &json!(8998));
        }
    }

    #[test]
    fn test_sglang_inject_batch_room_ids_are_distinct() {
        let (adapter, ctx) = sglang_pair();
        let mut body = json!({});
        adapter
            .inject_batch_prefill_fields(&mut body, &ctx, 3)
            .unwrap();
        let rooms: Vec<u64> = body["bootstrap_room"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let unique: std::collections::HashSet<_> = rooms.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_sglang_inject_on_non_object_returns_body_not_object() {
        let (adapter, ctx) = sglang_pair();
        let mut body = json!([1, 2, 3]);
        let before = body.clone();
        let err = adapter.inject_prefill_fields(&mut body, &ctx).unwrap_err();
        assert_eq!(err, AdapterError::BodyNotObject);
        assert_eq!(body, before);

        let err = adapter
            .inject_batch_prefill_fields(&mut body, &ctx, 2)
            .unwrap_err();
        assert_eq!(err, AdapterError::BodyNotObject);
        assert_eq!(body, before);
    }

    fn vllm_info_with(prefill_url: &str, dp_rank: usize) -> Arc<VllmPrefillInfo> {
        let mut bootstrap_addrs = HashMap::new();
        bootstrap_addrs.insert(prefill_url.to_string(), "10.0.0.1:9000".to_string());
        let mut engine_ids = HashMap::new();
        let mut per_rank = HashMap::new();
        per_rank.insert(dp_rank, "engine-abc".to_string());
        engine_ids.insert(prefill_url.to_string(), per_rank);
        Arc::new(VllmPrefillInfo {
            bootstrap_addrs,
            engine_ids,
        })
    }

    fn vllm_pair_for(
        prefill_url: &str,
        dp_rank: usize,
    ) -> (VllmAdapter, super::super::backend::PairCtx) {
        let prefill = make_prefill_http(prefill_url, "m", None);
        let decode = make_decode_http("http://decode:8000", "m");
        let adapter = VllmAdapter::new(vllm_info_with(prefill_url, dp_rank));
        let ctx = adapter
            .prepare_pair(prefill.as_ref(), decode.as_ref())
            .expect("prepare_pair");
        (adapter, ctx)
    }

    #[test]
    fn test_vllm_inject_prefill_kv_and_force_prefill_rewrites() {
        let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
        let mut body = json!({
            "prompt": "hi",
            "stream": true,
            "max_tokens": 256,
            "max_completion_tokens": 100,
        });
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        let obj = body.as_object().unwrap();
        let kv = &obj["kv_transfer_params"];
        assert_eq!(kv["do_remote_decode"], json!(true));
        assert_eq!(kv["do_remote_prefill"], json!(false));
        assert!(kv["transfer_id"].as_str().unwrap().starts_with("xfer-"));
        assert_eq!(obj["stream"], json!(false));
        assert_eq!(obj["max_tokens"], json!(1));
        assert_eq!(obj["max_completion_tokens"], json!(1));
    }

    #[test]
    fn test_vllm_inject_decode_lookups_and_shared_transfer_id() {
        let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
        let mut prefill_body = json!({});
        adapter
            .inject_prefill_fields(&mut prefill_body, &ctx)
            .unwrap();
        let prefill_xfer = prefill_body["kv_transfer_params"]["transfer_id"]
            .as_str()
            .unwrap()
            .to_string();

        let mut decode_body = json!({});
        adapter
            .inject_decode_fields(&mut decode_body, &ctx)
            .unwrap();
        let kv = &decode_body["kv_transfer_params"];
        assert_eq!(kv["do_remote_decode"], json!(false));
        assert_eq!(kv["do_remote_prefill"], json!(true));
        assert_eq!(kv["remote_bootstrap_addr"], json!("10.0.0.1:9000"));
        assert_eq!(kv["remote_engine_id"], json!("engine-abc"));
        assert_eq!(kv["transfer_id"].as_str().unwrap(), prefill_xfer);
    }

    #[test]
    fn test_vllm_force_prefill_max_completion_tokens_only_overwrites() {
        let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
        let mut body = json!({"prompt": "hi"});
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        let obj = body.as_object().unwrap();
        assert!(!obj.contains_key("max_completion_tokens"));
        assert_eq!(obj["max_tokens"], json!(1));
        assert_eq!(obj["stream"], json!(false));
    }

    #[test]
    fn test_vllm_force_prefill_removes_stream_options() {
        let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
        let mut body = json!({
            "prompt": "hi",
            "stream_options": {"include_usage": true},
        });
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        assert!(!body.as_object().unwrap().contains_key("stream_options"));
    }

    #[test]
    fn test_vllm_prepare_pair_missing_bootstrap_addr() {
        let prefill = make_prefill_http("http://unknown:8000", "m", None);
        let decode = make_decode_http("http://decode:8000", "m");
        let adapter = VllmAdapter::new(vllm_info_with("http://other:8000", 0));
        let err = adapter
            .prepare_pair(prefill.as_ref(), decode.as_ref())
            .unwrap_err();
        assert_eq!(
            err,
            AdapterError::BootstrapAddrMissing {
                prefill_url: "http://unknown:8000".to_string()
            }
        );
    }

    #[test]
    fn vllm_inject_prefill_on_non_object_returns_body_not_object() {
        let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
        let mut body = json!([1, 2, 3]);
        let before = body.clone();
        let err = adapter.inject_prefill_fields(&mut body, &ctx).unwrap_err();
        assert_eq!(err, AdapterError::BodyNotObject);
        assert_eq!(body, before);
    }

    #[test]
    fn vllm_inject_decode_on_non_object_returns_body_not_object() {
        let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
        let mut body = json!("not-an-object");
        let before = body.clone();
        let err = adapter.inject_decode_fields(&mut body, &ctx).unwrap_err();
        assert_eq!(err, AdapterError::BodyNotObject);
        assert_eq!(body, before);
    }

    #[test]
    fn test_vllm_correlation_id_matches_transfer_id() {
        let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
        let mut body = json!({});
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        let body_xfer = body["kv_transfer_params"]["transfer_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(adapter.correlation_id(&ctx), Some(body_xfer));
    }

    #[test]
    fn test_sglang_correlation_id_matches_bootstrap_room() {
        let (adapter, ctx) = sglang_pair();
        let mut body = json!({});
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        let body_room = body["bootstrap_room"].as_u64().unwrap().to_string();
        assert_eq!(adapter.correlation_id(&ctx), Some(body_room));
    }

    #[test]
    fn test_vllm_prepare_pair_missing_engine_id() {
        let prefill = make_prefill_http("http://p:8000", "m", None);
        let decode = make_decode_http("http://decode:8000", "m");
        let adapter = VllmAdapter::new(vllm_info_with("http://p:8000", 7));
        let err = adapter
            .prepare_pair(prefill.as_ref(), decode.as_ref())
            .unwrap_err();
        assert_eq!(
            err,
            AdapterError::EngineIdMissing {
                prefill_url: "http://p:8000".to_string(),
                dp_rank: 0,
            }
        );
    }
}

mod f_atom_adapter {
    use std::collections::HashMap;
    use std::sync::Arc;

    use serde_json::{json, Value};

    use super::super::backend::atom::{AtomAdapter, AtomPairCtx, AtomPrefillInfo};
    use super::super::backend::BackendAdapter;
    use super::super::test_support::*;
    use super::super::types::AdapterError;

    fn atom_info_with(prefill_url: &str, tp_size: usize) -> Arc<AtomPrefillInfo> {
        let mut tp_sizes = HashMap::new();
        tp_sizes.insert(prefill_url.to_string(), tp_size);
        Arc::new(AtomPrefillInfo { tp_sizes })
    }

    fn atom_pair_for(
        prefill_url: &str,
        tp_size: usize,
    ) -> (AtomAdapter, super::super::backend::PairCtx) {
        let prefill = make_prefill_http(prefill_url, "m", None);
        let decode = make_decode_http("http://decode:8000", "m");
        let adapter = AtomAdapter::new(atom_info_with(prefill_url, tp_size));
        let ctx = adapter
            .prepare_pair(prefill.as_ref(), decode.as_ref())
            .expect("prepare_pair");
        (adapter, ctx)
    }

    #[test]
    fn test_atom_inject_prefill_writes_kv_and_force_prefill() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut body = json!({
            "prompt": "hi",
            "stream": true,
            "max_tokens": 256,
            "max_completion_tokens": 100,
        });
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        let obj = body.as_object().unwrap();
        let kv = &obj["kv_transfer_params"];
        assert_eq!(kv["do_remote_decode"], json!(true));
        assert_eq!(kv["do_remote_prefill"], json!(false));
        assert_eq!(obj["stream"], json!(false));
        assert_eq!(obj["max_tokens"], json!(1));
        assert_eq!(obj["max_completion_tokens"], json!(1));
    }

    #[test]
    fn test_atom_inject_prefill_removes_stream_options() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut body = json!({
            "prompt": "hi",
            "stream_options": {"include_usage": true},
        });
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        assert!(!body.as_object().unwrap().contains_key("stream_options"));
    }

    #[test]
    fn test_atom_inject_prefill_does_not_add_max_completion_tokens() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut body = json!({"prompt": "hi"});
        adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
        let obj = body.as_object().unwrap();
        assert!(!obj.contains_key("max_completion_tokens"));
        assert_eq!(obj["max_tokens"], json!(1));
    }

    #[test]
    fn test_atom_inject_decode_is_noop() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut body = json!({"prompt": "hi", "kv_transfer_params": {"x": 1}});
        let before = body.clone();
        adapter.inject_decode_fields(&mut body, &ctx).unwrap();
        assert_eq!(body, before);
    }

    #[test]
    fn test_atom_inject_prefill_on_non_object_returns_body_not_object() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut body = json!([1, 2, 3]);
        let before = body.clone();
        let err = adapter.inject_prefill_fields(&mut body, &ctx).unwrap_err();
        assert_eq!(err, AdapterError::BodyNotObject);
        assert_eq!(body, before);
    }

    #[test]
    fn test_atom_inject_batch_equals_single_for_batch_one() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut a = json!({"prompt": "hi"});
        let mut b = json!({"prompt": "hi"});
        adapter.inject_prefill_fields(&mut a, &ctx).unwrap();
        adapter
            .inject_batch_prefill_fields(&mut b, &ctx, 1)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_atom_correlation_id_matches_ctx_transfer_id() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let cid = adapter.correlation_id(&ctx).unwrap();
        let downcast = ctx
            .downcast_ref::<AtomPairCtx>()
            .expect("downcast atom ctx");
        assert!(cid.starts_with("xfer-"));
        assert_eq!(cid, downcast.transfer_id);
    }

    #[test]
    fn test_atom_prepare_pair_captures_default_dp_size_one() {
        let (_adapter, ctx) = atom_pair_for("http://p:8000", 4);
        let c = ctx.downcast_ref::<AtomPairCtx>().unwrap();
        assert_eq!(c.prefill_url, "http://p:8000");
        assert_eq!(c.prefill_dp_size, 1);
    }

    #[test]
    fn test_atom_enrich_decode_kv_single_dp() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut kv = json!({
            "do_remote_prefill": true,
            "do_remote_decode": false,
            "remote_block_ids": [0, 1, 2],
            "remote_engine_id": "10.24.112.168:6301",
            "transfer_id": 42,
        });
        adapter.enrich_decode_kv(&mut kv, &ctx).unwrap();
        assert_eq!(kv["remote_dp_size"], json!(1));
        assert_eq!(kv["remote_tp_size"], json!(8));
        assert_eq!(kv["transfer_id"], json!(42));
    }

    #[test]
    fn test_atom_enrich_decode_kv_multi_dp() {
        let info = atom_info_with("http://p:8000", 8);
        let adapter = AtomAdapter::new(info);
        let ctx: super::super::backend::PairCtx = Box::new(AtomPairCtx {
            transfer_id: "xfer-test".to_string(),
            prefill_url: "http://p:8000".to_string(),
            prefill_dp_size: 4,
            prefill_dp_rank: Some(2),
            decode_dp_rank: Some(2),
        });
        let mut kv = json!({});
        adapter.enrich_decode_kv(&mut kv, &ctx).unwrap();
        assert_eq!(kv["remote_dp_size"], json!(4));
        assert_eq!(kv["remote_dp_rank"], json!(2));
        assert_eq!(kv["remote_tp_size"], json!(8));
    }

    #[test]
    fn test_atom_enrich_decode_kv_maps_dp_rank_to_remote_dp_rank() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut kv = json!({
            "do_remote_prefill": true,
            "dp_rank": 3,
            "transfer_id": 7,
        });
        adapter.enrich_decode_kv(&mut kv, &ctx).unwrap();
        // The owning producer DP rank must be propagated so the decode side
        // sends its write_request to the correct side-channel port.
        assert_eq!(kv["remote_dp_rank"], json!(3));
        assert_eq!(kv["remote_dp_size"], json!(1));
        assert_eq!(kv["remote_tp_size"], json!(8));
    }

    #[test]
    fn test_atom_enrich_decode_kv_no_dp_rank_leaves_remote_dp_rank_unset() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut kv = json!({ "transfer_id": 1 });
        adapter.enrich_decode_kv(&mut kv, &ctx).unwrap();
        // Absent dp_rank: decode's ReqMeta default (0) applies, matching the
        // single-DP case; we must not invent a rank.
        assert!(kv.get("remote_dp_rank").is_none());
    }

    #[test]
    fn test_atom_enrich_decode_kv_null_dp_rank_leaves_remote_dp_rank_unset() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut kv = json!({ "dp_rank": serde_json::Value::Null });
        adapter.enrich_decode_kv(&mut kv, &ctx).unwrap();
        // A null dp_rank must not be propagated: decode would default to 0,
        // whereas copying null verbatim would crash its integer port math.
        assert!(kv.get("remote_dp_rank").is_none());
    }

    #[test]
    fn test_atom_enrich_decode_kv_missing_tp_size_errors() {
        let info = atom_info_with("http://other:8000", 8);
        let adapter = AtomAdapter::new(info);
        let ctx: super::super::backend::PairCtx = Box::new(AtomPairCtx {
            transfer_id: "x".to_string(),
            prefill_url: "http://unknown:8000".to_string(),
            prefill_dp_size: 1,
            prefill_dp_rank: None,
            decode_dp_rank: None,
        });
        let mut kv = json!({});
        let err = adapter.enrich_decode_kv(&mut kv, &ctx).unwrap_err();
        assert_eq!(
            err,
            AdapterError::TpSizeMissing {
                prefill_url: "http://unknown:8000".to_string()
            }
        );
        assert_eq!(kv, json!({}));
    }

    #[test]
    fn test_atom_enrich_decode_kv_non_object_errors() {
        let (adapter, ctx) = atom_pair_for("http://p:8000", 8);
        let mut kv = json!("not-an-object");
        let err = adapter.enrich_decode_kv(&mut kv, &ctx).unwrap_err();
        assert_eq!(err, AdapterError::BodyNotObject);
    }

    #[test]
    fn test_atom_enrich_decode_kv_wrong_ctx_errors() {
        let info = atom_info_with("http://p:8000", 8);
        let adapter = AtomAdapter::new(info);
        let wrong: super::super::backend::PairCtx = Box::new(42u32);
        let mut kv = json!({});
        let err = adapter.enrich_decode_kv(&mut kv, &wrong).unwrap_err();
        assert_eq!(err, AdapterError::CtxTypeMismatch);
    }
}

mod g_error {
    use std::sync::Arc;

    use super::super::planner::DefaultPlanner;
    use super::super::test_support::*;
    use super::super::traits::{PdPlanner, PolicySource, WorkerSource};
    use super::super::types::{AdapterError, PlacementError};
    use crate::policies::LoadBalancingPolicy;

    fn make_planner(src: MockWorkerSource, policies: MockPolicySource) -> DefaultPlanner {
        DefaultPlanner::new(
            Arc::new(src) as Arc<dyn WorkerSource>,
            Arc::new(policies) as Arc<dyn PolicySource>,
        )
    }

    #[tokio::test]
    async fn test_no_workers_triggered_by_empty_registry_and_no_model() {
        let planner = make_planner(MockWorkerSource::new(), MockPolicySource::new());
        let err = planner
            .plan(&make_descriptor(None, None, None, None))
            .await
            .unwrap_err();
        assert_eq!(err, PlacementError::NoWorkers);
    }

    #[tokio::test]
    async fn test_no_available_workers_triggered_by_all_unhealthy() {
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
    async fn test_no_prefill_workers_triggered_in_pd_path() {
        let src = MockWorkerSource::new().add_worker(make_decode_http("http://d:8000", "m"));
        let planner = make_planner(src, MockPolicySource::new());
        let err = planner
            .plan(&make_descriptor(Some("m"), None, None, None))
            .await
            .unwrap_err();
        assert_eq!(err, PlacementError::NoPrefillWorkers);
    }

    #[tokio::test]
    async fn test_no_decode_workers_triggered_in_pd_path() {
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
    async fn test_policy_returned_none_triggered_by_always_none_policy() {
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
    async fn test_model_not_found_triggered_when_model_id_unknown() {
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
    async fn test_no_available_workers_in_pd_path_when_all_unhealthy() {
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
    fn test_adapter_errors_map_to_5xx_response() {
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
    fn test_error_display_includes_key_fields() {
        let model_not_found = PlacementError::ModelNotFound {
            model_id: "m_xyz".to_string(),
        };
        assert!(format!("{}", model_not_found).contains("m_xyz"));

        let engine_missing = AdapterError::EngineIdMissing {
            prefill_url: "http://p:8000".to_string(),
            dp_rank: 7,
        };
        let display = format!("{}", engine_missing);
        assert!(
            display.contains("http://p:8000"),
            "missing prefill_url: {}",
            display
        );
        assert!(display.contains('7'), "missing dp_rank: {}", display);
    }
}

mod h_integration {
    #![allow(unused_imports)]

    use super::super::test_support::*;

    #[tokio::test]
    async fn test_http_regular_planner_single_dispatches_to_worker_url() {
        use crate::core::placement::planner::DefaultPlanner;
        use crate::core::placement::traits::PdPlanner;
        use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};
        use std::sync::Arc;

        let src = MockWorkerSource::new().add_worker(make_regular_http("http://w-1:8000", "m"));
        let policies = MockPolicySource::new();
        let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

        let descriptor = RequestDescriptor {
            model_id: Some("m"),
            protocol: Some(Protocol::Http),
            ..Default::default()
        };

        let plan = planner
            .plan(&descriptor)
            .await
            .expect("plan should succeed");
        match plan {
            PlacementPlan::Single { worker, .. } => {
                assert_eq!(worker.url(), "http://w-1:8000");
            }
            _ => panic!("expected Single"),
        }
    }

    #[tokio::test]
    async fn test_http_regular_no_workers_returns_503() {
        use crate::core::placement::planner::DefaultPlanner;
        use crate::core::placement::traits::PdPlanner;
        use crate::core::placement::types::{PlacementError, Protocol, RequestDescriptor};
        use crate::routers::comm::placement_response::placement_err_to_response;
        use axum::body::to_bytes;
        use axum::http::StatusCode;
        use std::sync::Arc;

        let src = MockWorkerSource::new();
        let policies = MockPolicySource::new();
        let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

        let descriptor = RequestDescriptor {
            model_id: None,
            protocol: Some(Protocol::Http),
            ..Default::default()
        };

        let err = planner
            .plan(&descriptor)
            .await
            .expect_err("plan should fail");
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
    async fn test_http_regular_model_not_found_returns_503_with_model_name() {
        use crate::core::placement::planner::DefaultPlanner;
        use crate::core::placement::traits::PdPlanner;
        use crate::core::placement::types::{PlacementError, Protocol, RequestDescriptor};
        use crate::routers::comm::placement_response::placement_err_to_response;
        use axum::body::to_bytes;
        use axum::http::StatusCode;
        use std::sync::Arc;

        let src =
            MockWorkerSource::new().add_worker(make_regular_http("http://w-other:8000", "other"));
        let policies = MockPolicySource::new();
        let planner = DefaultPlanner::new(Arc::new(src), Arc::new(policies));

        let descriptor = RequestDescriptor {
            model_id: Some("requested_model"),
            protocol: Some(Protocol::Http),
            ..Default::default()
        };

        let err = planner
            .plan(&descriptor)
            .await
            .expect_err("plan should fail");
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
    async fn test_grpc_stage_planner_err_returns_service_unavailable() {
        use crate::core::placement::types::PlacementError;
        use crate::routers::comm::placement_response::placement_err_to_response;
        use axum::body::to_bytes;
        use axum::http::StatusCode;

        let resp =
            placement_err_to_response(PlacementError::NoAvailableWorkers, Some("test_model"));
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
    async fn test_http_pd_sglang_dual_dispatch_to_prefill_and_decode() {
        use crate::core::placement::backend::sglang::SglangAdapter;
        use crate::core::placement::backend::BackendAdapter;
        use crate::core::placement::planner::DefaultPlanner;
        use crate::core::placement::traits::PdPlanner;
        use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};
        use serde_json::json;
        use std::sync::Arc;

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
            PlacementPlan::Pair {
                prefill, decode, ..
            } => (prefill, decode),
            _ => panic!("expected Pair"),
        };
        assert_eq!(p.url(), "http://prefill-1:8000");
        assert_eq!(d.url(), "http://decode-1:8000");

        let adapter = SglangAdapter;
        let ctx = adapter
            .prepare_pair(p.as_ref(), d.as_ref())
            .expect("prepare");
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
    async fn test_http_pd_vllm_pair_uses_shared_transfer_id() {
        use crate::core::placement::backend::vllm::{VllmAdapter, VllmPrefillInfo};
        use crate::core::placement::backend::BackendAdapter;
        use crate::core::placement::planner::DefaultPlanner;
        use crate::core::placement::traits::PdPlanner;
        use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};
        use serde_json::json;
        use std::collections::HashMap;
        use std::sync::Arc;

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
            PlacementPlan::Pair {
                prefill, decode, ..
            } => (prefill, decode),
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
        let ctx = adapter
            .prepare_pair(p.as_ref(), d.as_ref())
            .expect("prepare");

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
    async fn test_http_pd_sglang_batch_writes_length_n_arrays() {
        use crate::core::placement::backend::sglang::SglangAdapter;
        use crate::core::placement::backend::BackendAdapter;
        use crate::core::placement::planner::DefaultPlanner;
        use crate::core::placement::traits::PdPlanner;
        use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};
        use serde_json::json;
        use std::sync::Arc;

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
            PlacementPlan::Pair {
                prefill, decode, ..
            } => (prefill, decode),
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
    async fn test_http_pd_retry_preserves_text_headers_tokens() {
        use crate::core::placement::planner::DefaultPlanner;
        use crate::core::placement::traits::PdPlanner;
        use crate::core::placement::types::{PlacementPlan, Protocol, RequestDescriptor};
        use http::HeaderMap;
        use std::sync::Arc;

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
}
