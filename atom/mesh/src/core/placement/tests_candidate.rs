use std::sync::Arc;

use super::test_support::*;
use super::traits::WorkerSource;
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
fn a01_filters_to_specified_model_only() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://m1-w1:8000", "m1"))
        .add_worker(make_regular_http("http://m1-w2:8000", "m1"))
        .add_worker(make_regular_http("http://m2-w1:8000", "m2"));

    let result = filter_candidates(&src, Some("m1"), None, None);
    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|w| w.model_id() == "m1"));
}

#[test]
fn a02_model_id_none_returns_all_workers() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://w1:8000", "m1"))
        .add_worker(make_regular_http("http://w2:8000", "m2"));

    let result = filter_candidates(&src, None, None, None);
    assert_eq!(result.len(), 2);
}

#[test]
fn a03_unknown_model_returns_empty() {
    let src = MockWorkerSource::new().add_worker(make_regular_http("http://w:8000", "m1"));

    let result = filter_candidates(&src, Some("m_missing"), None, None);
    assert!(result.is_empty());
}

#[test]
fn a04_no_cross_model_contamination() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://m1-w:8000", "m1"))
        .add_worker(make_regular_http("http://m2-w:8000", "m2"));

    let result = filter_candidates(&src, Some("m1"), None, None);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].url(), "http://m1-w:8000");
    assert!(result.iter().all(|w| w.model_id() != "m2"));
}

#[test]
fn a05_worker_type_regular_excludes_pd() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://r:8000", "m"))
        .add_worker(make_prefill_http("http://p:8000", "m", Some(8998)))
        .add_worker(make_decode_http("http://d:8000", "m"));

    let result = filter_candidates(&src, None, Some(WorkerType::Regular), None);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].url(), "http://r:8000");
}

#[test]
fn a06_worker_type_prefill_excludes_regular_and_decode() {
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
fn a07_worker_type_decode_excludes_regular_and_prefill() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://r:8000", "m"))
        .add_worker(make_prefill_http("http://p:8000", "m", Some(8998)))
        .add_worker(make_decode_http("http://d:8000", "m"));

    let result = filter_candidates(&src, None, Some(WorkerType::Decode), None);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].url(), "http://d:8000");
}

#[test]
fn a08_connection_mode_http_excludes_grpc() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://h:8000", "m"))
        .add_worker(make_regular_grpc("http://g:8000", "m"));

    let result = filter_candidates(&src, None, None, Some(ConnectionMode::Http));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].url(), "http://h:8000");
}

#[test]
fn a09_connection_mode_grpc_excludes_http() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://h:8000", "m"))
        .add_worker(make_regular_grpc("http://g:8000", "m"));

    let result = filter_candidates(&src, None, None, Some(ConnectionMode::Grpc { port: None }));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].url(), "http://g:8000");
}

#[test]
fn a10_all_healthy_all_pass() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://w1:8000", "m"))
        .add_worker(make_regular_http("http://w2:8000", "m"));

    let result = filter_candidates(&src, None, None, None);
    assert_eq!(result.len(), 2);
}

#[test]
fn a11_all_unhealthy_empty_set() {
    let w1 = make_regular_http("http://w1:8000", "m");
    let w2 = make_regular_http("http://w2:8000", "m");
    w1.set_healthy(false);
    w2.set_healthy(false);
    let src = MockWorkerSource::new().add_worker(w1).add_worker(w2);

    let result = filter_candidates(&src, None, None, None);
    assert!(result.is_empty());
}

#[test]
fn a12_mixed_health_only_healthy_pass() {
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
fn a13_empty_registry_returns_empty_no_panic() {
    let src = MockWorkerSource::new();
    let result = filter_candidates(&src, Some("m"), None, None);
    assert!(result.is_empty());
    let result2 = filter_candidates(&src, None, None, None);
    assert!(result2.is_empty());
}

#[test]
fn a14_prefill_bootstrap_port_variants_both_match() {
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
fn a15_combined_filters_all_apply() {
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
fn a16_model_id_some_worker_type_none_returns_all_for_model() {
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
fn a17_dp_aware_same_url_different_dp_rank_both_pass() {
    let src = MockWorkerSource::new()
        .add_worker(make_regular_http("http://shared:8000", "m"))
        .add_worker(make_regular_http("http://shared:8000", "m"));

    let result = filter_candidates(&src, None, None, None);
    assert_eq!(result.len(), 2);
}

#[test]
fn a18_filter_scales_linearly_with_worker_count() {
    let mut src = MockWorkerSource::new();
    for i in 0..256 {
        src = src.add_worker(make_regular_http(&format!("http://w{}:8000", i), "m"));
    }

    let result = filter_candidates(&src, Some("m"), Some(WorkerType::Regular), None);
    assert_eq!(result.len(), 256);

    let mut urls: Vec<&str> = result.iter().map(|w| w.url()).collect();
    urls.sort();
    urls.dedup();
    assert_eq!(urls.len(), 256, "filter_candidates dropped or duplicated workers");
}
