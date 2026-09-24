use super::*;
use crate::core::{BasicWorkerBuilder, WorkerLoadGuard};
use http::{HeaderMap, HeaderValue};

fn workers() -> Vec<Arc<dyn Worker>> {
    (0..3)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://p{i}:8000"))
                    .worker_type(WorkerType::Prefill {
                        bootstrap_port: None,
                    })
                    .model_id("m")
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect()
}

async fn reserve(
    policy: &AdaptiveCacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    text: &str,
    session: Option<&str>,
) -> PrefillSelection {
    let mut headers = HeaderMap::new();
    if let Some(s) = session {
        headers.insert("x-session-id", HeaderValue::from_str(s).unwrap());
    }
    policy
        .select_prefill_worker(
            workers,
            &SelectWorkerInfo {
                request_text: Some(text),
                headers: Some(&headers),
                ..Default::default()
            },
        )
        .await
        .unwrap()
}

async fn seed(
    policy: &AdaptiveCacheAwarePolicy,
    worker: &Arc<dyn Worker>,
    text: &str,
    session: Option<&str>,
) {
    let mut selection = reserve(policy, &[worker.clone()], text, session).await;
    selection
        .reservation
        .as_mut()
        .unwrap()
        .complete(PrefillFeedback {
            input_tokens: Some(text.chars().count() as u64),
            cached_tokens: Some(0),
        });
}

fn pool(policy: &AdaptiveCacheAwarePolicy) -> Arc<Pool> {
    policy.pools.get("m").unwrap().clone()
}
fn outstanding(policy: &AdaptiveCacheAwarePolicy) -> f64 {
    pool(policy)
        .state
        .lock()
        .unwrap()
        .workers
        .values()
        .map(WorkerWork::queued_work)
        .sum()
}

#[tokio::test]
async fn adaptive_same_load_gap_moves_cheap_request_but_preserves_expensive_prefix() {
    for (prefix_len, expected) in [(10, 1), (1000, 0)] {
        let p = AdaptiveCacheAwarePolicy::new(0, 100000);
        let w = workers();
        let prefix = "p".repeat(prefix_len);
        seed(&p, &w[0], &prefix, Some("session")).await;
        let _busy = reserve(&p, &w[..1], &"q".repeat(100), None).await;
        let choice = reserve(&p, &w[..2], &(prefix + "x"), Some("session")).await;
        assert_eq!(choice.index, expected);
    }
}

#[tokio::test]
async fn adaptive_low_match_ratio_still_values_a_large_absolute_prefix() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let w = workers();
    let prefix = "p".repeat(1000);
    seed(&p, &w[0], &prefix, None).await;
    let _busy = reserve(&p, &w[..1], &"q".repeat(100), None).await;
    let choice = reserve(&p, &w, &(prefix + &"x".repeat(2000)), None).await;
    assert_eq!(choice.index, 0, "a 1/3 match must not bypass the cost gate");
}

#[tokio::test]
async fn adaptive_large_wait_saving_can_move_a_long_cached_request() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let w = workers();
    let prefix = "p".repeat(1000);
    seed(&p, &w[0], &prefix, Some("s")).await;
    let _busy = reserve(&p, &w[..1], &"q".repeat(2000), None).await;
    let choice = reserve(&p, &w, &(prefix + "x"), Some("s")).await;
    assert_ne!(
        choice.index, 0,
        "long requests have no blanket migration ban"
    );
}

#[tokio::test]
async fn adaptive_replica_cache_makes_relocation_cheap() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let w = workers();
    let prefix = "p".repeat(1000);
    seed(&p, &w[0], &prefix, Some("s")).await;
    seed(&p, &w[1], &prefix, None).await;
    let _busy = reserve(&p, &w[..1], &"q".repeat(100), None).await;
    assert_eq!(reserve(&p, &w, &(prefix + "x"), Some("s")).await.index, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn adaptive_concurrent_cold_burst_reserves_before_dispatch() {
    let p = Arc::new(AdaptiveCacheAwarePolicy::new(0, 100000));
    let w = workers();
    let mut tasks = Vec::new();
    for _ in 0..48 {
        let p = p.clone();
        let w = w.clone();
        tasks.push(tokio::spawn(async move {
            reserve(&p, &w, "cold request", None).await
        }));
    }
    let mut held = Vec::new();
    let mut counts = [0; 3];
    for task in tasks {
        let s = task.await.unwrap();
        counts[s.index] += 1;
        held.push(s);
    }
    assert_eq!(counts, [16, 16, 16]);
    assert!(outstanding(&p) > 0.0);
    assert!(w.iter().all(|worker| worker.load() == 0));
    drop(held);
    assert_eq!(outstanding(&p), 0.0);
}

#[tokio::test]
async fn adaptive_dispatched_work_is_not_double_counted_and_drop_releases() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let w = workers();
    let mut a = reserve(&p, &w[..1], &"a".repeat(100), None).await;
    let _b = reserve(&p, &w[1..2], &"b".repeat(150), None).await;
    let load = WorkerLoadGuard::new(w[0].clone(), None);
    a.reservation.as_mut().unwrap().mark_dispatched();
    let choice = reserve(&p, &w[..2], "cold", None).await;
    assert_eq!(
        choice.index, 0,
        "100 units on A must not become 200 after dispatch"
    );
    drop(choice);
    drop(a);
    drop(load);
    assert_eq!(outstanding(&p), 150.0);
    assert_eq!(w[0].load(), 0);
}

#[tokio::test]
async fn adaptive_cancelled_and_failed_attempts_do_not_confirm_cache_or_session() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let w = workers();
    drop(reserve(&p, &w[..1], "unconfirmed prompt", Some("s")).await);
    assert_eq!(outstanding(&p), 0.0);
    let pool = pool(&p);
    assert!(pool.state.lock().unwrap().sessions.is_empty());
    assert_eq!(
        pool.tree.prefix_match_counts_for_tenants(
            "unconfirmed prompt",
            &[&Identity::of(w[0].as_ref()).tenant()]
        ),
        vec![0]
    );
    // A retry can choose another rank without retaining the failed reservation.
    let _busy = reserve(&p, &w[..1], "busy", None).await;
    assert_eq!(
        reserve(&p, &w, "unconfirmed prompt", Some("s")).await.index,
        1
    );
}

#[tokio::test]
async fn adaptive_completion_is_idempotent_and_only_newer_success_updates_session() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let w = workers();
    let mut old = reserve(&p, &w[..1], "shared prefix", Some("s")).await;
    let mut new = reserve(&p, &w[1..2], "shared prefix", Some("s")).await;
    let feedback = PrefillFeedback {
        input_tokens: Some(13),
        cached_tokens: Some(0),
    };
    new.reservation.as_mut().unwrap().complete(feedback);
    new.reservation.as_mut().unwrap().complete(feedback);
    old.reservation.as_mut().unwrap().complete(feedback);
    drop(old);
    drop(new);
    assert_eq!(outstanding(&p), 0.0);
    assert_eq!(reserve(&p, &w, "shared prefix", Some("s")).await.index, 1);
}

#[tokio::test]
async fn adaptive_removed_worker_late_response_does_not_resurrect_cache() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let w = workers();
    let mut old = reserve(&p, &w[..1], "old prefix", Some("s")).await;
    p.remove_worker_by_url(w[0].url());
    let _new = reserve(&p, &w[..1], "new prefix", None).await;
    old.reservation
        .as_mut()
        .unwrap()
        .complete(PrefillFeedback::default());
    let pool = pool(&p);
    assert!(pool.state.lock().unwrap().sessions.is_empty());
    assert_eq!(
        pool.tree.prefix_match_counts_for_tenants(
            "old prefix",
            &[&Identity::of(w[0].as_ref()).tenant()]
        ),
        vec![0]
    );
    assert_eq!(outstanding(&p), 10.0);
}

#[tokio::test]
async fn adaptive_unrelated_hot_or_unhealthy_rank_cannot_trigger_cache_bypass() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let w = workers();
    seed(&p, &w[0], "valuable prefix", Some("s")).await;
    for _ in 0..100 {
        w[2].increment_load();
    }
    assert_eq!(
        reserve(&p, &w, "valuable prefix plus", Some("s"))
            .await
            .index,
        0
    );
    w[0].set_healthy(false);
    assert_ne!(
        reserve(&p, &w, "valuable prefix plus", Some("s"))
            .await
            .index,
        0
    );
}

#[tokio::test]
async fn adaptive_invalid_usage_is_not_used_as_a_training_sample() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let w = workers();
    for feedback in [
        PrefillFeedback::default(),
        PrefillFeedback {
            input_tokens: Some(10),
            cached_tokens: Some(11),
        },
    ] {
        let mut s = reserve(&p, &w[..1], "prefix", None).await;
        s.reservation.as_mut().unwrap().complete(feedback);
    }
    let pool = pool(&p);
    let state = pool.state.lock().unwrap();
    assert_eq!(state.workers[&Identity::of(w[0].as_ref())].observations, 0);
    assert_eq!(state.mean_work, None);
}

#[test]
fn adaptive_feedback_reads_actual_usage_and_preserves_missing_values() {
    let f = PrefillFeedback::from_response(
        &serde_json::json!({"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":80}}}),
    );
    assert_eq!((f.input_tokens, f.cached_tokens), (Some(100), Some(80)));
    let f = PrefillFeedback::from_response(
        &serde_json::json!({"usage":{"prompt_tokens":100,"prompt_cache_read_tokens":75}}),
    );
    assert_eq!(f.cached_tokens, Some(75));
    assert!(PrefillFeedback::from_response(&serde_json::json!({}))
        .cached_tokens
        .is_none());
}

#[tokio::test]
async fn adaptive_does_not_apply_prefill_work_model_to_decode_or_mixed_models() {
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let decode: Vec<Arc<dyn Worker>> = vec![Arc::new(
        BasicWorkerBuilder::new("http://d")
            .worker_type(WorkerType::Decode)
            .build(),
    )];
    assert!(p
        .select_prefill_worker(&decode, &SelectWorkerInfo::default())
        .await
        .is_none());
    let mut w = workers();
    w.push(Arc::new(
        BasicWorkerBuilder::new("http://other")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: None,
            })
            .model_id("other")
            .build(),
    ));
    assert!(p
        .select_prefill_worker(&w, &SelectWorkerInfo::default())
        .await
        .is_none());
}

#[tokio::test]
async fn adaptive_abandoned_plan_releases_work_before_any_http_dispatch() {
    use crate::core::placement::{
        planner::DefaultPlanner,
        test_support::{make_decode_http, MockPolicySource, MockWorkerSource},
        traits::PdPlanner,
        RequestDescriptor,
    };
    let policy = Arc::new(AdaptiveCacheAwarePolicy::new(0, 100000));
    let w = workers();
    let source = MockWorkerSource::new()
        .add_worker(w[0].clone())
        .add_worker(make_decode_http("http://d:8000", "m"));
    let planner = DefaultPlanner::new(
        Arc::new(source),
        Arc::new(MockPolicySource::new().with_prefill(policy.clone())),
    );
    let plan = planner
        .plan(&RequestDescriptor {
            model_id: Some("m"),
            text: Some("prefix"),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(outstanding(&policy), 6.0);
    drop(plan);
    assert_eq!(outstanding(&policy), 0.0);
}

#[derive(Debug)]
struct WaitingDecode(Arc<tokio::sync::Notify>);

#[async_trait]
impl LoadBalancingPolicy for WaitingDecode {
    async fn select_worker(
        &self,
        _: &[Arc<dyn Worker>],
        _: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.0.notify_one();
        std::future::pending().await
    }
    fn name(&self) -> &'static str {
        "waiting-test"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[tokio::test]
async fn adaptive_cancellation_during_decode_selection_releases_prefill_work() {
    use crate::core::placement::{
        planner::DefaultPlanner,
        test_support::{make_decode_http, MockPolicySource, MockWorkerSource},
        traits::PdPlanner,
        RequestDescriptor,
    };
    let policy = Arc::new(AdaptiveCacheAwarePolicy::new(0, 100000));
    let w = workers();
    let entered = Arc::new(tokio::sync::Notify::new());
    let source = MockWorkerSource::new()
        .add_worker(w[0].clone())
        .add_worker(make_decode_http("http://d:8000", "m"));
    let policies = MockPolicySource::new()
        .with_prefill(policy.clone())
        .with_decode(Arc::new(WaitingDecode(entered.clone())));
    let planner = DefaultPlanner::new(Arc::new(source), Arc::new(policies));
    let task = tokio::spawn(async move {
        planner
            .plan(&RequestDescriptor {
                model_id: Some("m"),
                text: Some("prefix"),
                ..Default::default()
            })
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .unwrap();
    assert_eq!(outstanding(&policy), 6.0);
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(outstanding(&policy), 0.0);
}

#[tokio::test]
async fn adaptive_cache_error_feedback_blocks_a_marginal_migration() {
    for (observe_miss, expected) in [(false, 1), (true, 0)] {
        let p = AdaptiveCacheAwarePolicy::new(0, 100000);
        let w = workers();
        let prefix = "p".repeat(100);
        seed(&p, &w[0], &prefix, Some("s")).await;
        if observe_miss {
            let mut selected = reserve(&p, &w[..1], &prefix, Some("s")).await;
            // The approximate tree predicted a hit; the worker reports a miss.
            selected
                .reservation
                .as_mut()
                .unwrap()
                .complete(PrefillFeedback {
                    input_tokens: Some(100),
                    cached_tokens: Some(0),
                });
        }
        let _busy = reserve(&p, &w[..1], &"q".repeat(101), None).await;
        assert_eq!(
            reserve(&p, &w[..2], &prefix, Some("s")).await.index,
            expected
        );
    }
}

#[tokio::test]
async fn adaptive_same_endpoint_dp_ranks_and_model_pools_have_separate_cache() {
    use crate::core::worker::DPAwareWorker;
    let p = AdaptiveCacheAwarePolicy::new(0, 100000);
    let make = |model: &str| -> Vec<Arc<dyn Worker>> {
        (0..2)
            .map(|rank| {
                let base = BasicWorkerBuilder::new("http://shared:8000")
                    .worker_type(WorkerType::Prefill {
                        bootstrap_port: None,
                    })
                    .model_id(model)
                    .build();
                Arc::new(DPAwareWorker::with_base_worker(
                    base,
                    "http://shared:8000".into(),
                    rank,
                    2,
                )) as Arc<dyn Worker>
            })
            .collect()
    };
    let w = make("m");
    seed(&p, &w[1], "valuable prefix", Some("same-session")).await;
    assert_eq!(
        reserve(&p, &w, "valuable prefix", Some("same-session"))
            .await
            .index,
        1
    );
    let other = make("other-model");
    assert_eq!(
        reserve(&p, &other, "valuable prefix", Some("same-session"))
            .await
            .index,
        0
    );
}
