//! Sampled `Selected worker` log of the routing decisions behind each dispatch,
//! and the router's cache estimate that the response's usage is reconciled with.
//!
//! While `--route-log-sample-rate` is above 0, inference requests pass through
//! [`scope`], and a sampled request gets a trace. Policies select and routers
//! dispatch on the request task, so both reach the trace through a task-local
//! instead of the placement API. Work spawned off the request task, such as the
//! detached Mooncake prefill, takes the trace and the request span along with
//! [`carry`]. The HTTP routers run each request's attempts in [`estimated`],
//! whatever the sample rate.

use std::{cell::RefCell, future::Future, sync::Arc};

use axum::{
    extract::{Request, State},
    http::HeaderMap,
    middleware::Next,
    response::Response,
};
use rand::Rng;
use tracing::{info, Instrument};

use crate::{
    core::Worker,
    observability::metrics::{metrics_labels, MeshMetrics},
};

/// Request header that logs the routing decisions of one request while the
/// sample rate is above 0.
pub const ROUTE_TRACE_HEADER: &str = "x-mesh-route-trace";

tokio::task_local! {
    static TRACE: RouteTrace;
    static ESTIMATE: RefCell<Option<Estimate>>;
}

/// The estimate of the latest selection in the pool that computes the prompt.
struct Estimate {
    worker: Arc<dyn Worker>,
    hit_ratio: f64,
}

/// Latest policy selection per pool; retries replace it.
#[derive(Default)]
struct RouteTrace {
    selections: RefCell<Vec<Selection>>,
}

#[derive(Clone)]
struct Selection {
    pool: &'static str,
    policy: &'static str,
    url: String,
    /// Router-tracked active requests on the worker when the policy selected it.
    active_requests: usize,
    branch: &'static str,
    match_ratio: Option<f64>,
}

/// Clients can send the header, so it only applies once the operator enables sampling.
fn should_log(headers: &HeaderMap, rate: f64) -> bool {
    rate > 0.0 && (headers.contains_key(ROUTE_TRACE_HEADER) || rand::rng().random::<f64>() < rate)
}

pub(crate) async fn scope(State(rate): State<f64>, request: Request, next: Next) -> Response {
    if should_log(request.headers(), rate) {
        TRACE.scope(RouteTrace::default(), next.run(request)).await
    } else {
        next.run(request).await
    }
}

/// Runs `future` in the current request span and, for a sampled request, with a
/// copy of its decisions, wherever the future is spawned.
pub(crate) fn carry<F: Future>(future: F) -> impl Future<Output = F::Output> {
    let selections = TRACE
        .try_with(|trace| trace.selections.borrow().clone())
        .ok();
    async move {
        match selections {
            Some(selections) => {
                let trace = RouteTrace {
                    selections: RefCell::new(selections),
                };
                TRACE.scope(trace, future).await
            }
            None => future.await,
        }
    }
    .instrument(tracing::Span::current())
}

/// Keeps the decision of a policy selection until the router dispatches it.
pub(crate) fn record_selection(
    pool: &'static str,
    policy: &'static str,
    worker: Option<&Arc<dyn Worker>>,
    branch: &'static str,
    match_ratio: Option<f64>,
) {
    let _ = TRACE.try_with(|trace| {
        let Some(worker) = worker else {
            info!(target: "mesh::route", pool, policy, branch, "No worker selected");
            return;
        };
        let mut selections = trace.selections.borrow_mut();
        selections.retain(|selection| selection.pool != pool);
        selections.push(Selection {
            pool,
            policy,
            url: worker.url().to_string(),
            active_requests: worker.load(),
            branch,
            match_ratio,
        });
    });
}

/// Runs the attempts of one routed request, and returns the cache hit ratio its
/// last selection expected on the worker that computes the prompt, unless the
/// request went to another worker.
pub(crate) async fn estimated<F: Future>(attempts: F) -> (F::Output, Option<f64>) {
    ESTIMATE
        .scope(RefCell::new(None), async {
            let output = attempts.await;
            let estimate = ESTIMATE.with(|estimate| estimate.take());
            (output, estimate.map(|estimate| estimate.hit_ratio))
        })
        .await
}

fn computes_prompt(pool: &'static str) -> bool {
    matches!(
        pool,
        metrics_labels::WORKER_REGULAR | metrics_labels::WORKER_PREFILL
    )
}

/// Keeps the estimate of a selection in the pool that computes the prompt; a
/// retry replaces it.
pub(crate) fn record_estimate(
    pool: &'static str,
    worker: Option<&Arc<dyn Worker>>,
    hit_ratio: Option<f64>,
) {
    if !computes_prompt(pool) {
        return;
    }
    let _ = ESTIMATE.try_with(|estimate| {
        *estimate.borrow_mut() = worker.zip(hit_ratio).map(|(worker, hit_ratio)| Estimate {
            worker: worker.clone(),
            hit_ratio,
        });
    });
}

/// Records the worker a request is sent to after rank mapping.
pub(crate) fn record_dispatch(pool: &'static str, worker: &Arc<dyn Worker>) {
    MeshMetrics::record_selected_dp_rank(pool, worker.base_url(), worker.dp_rank());
    if computes_prompt(pool) {
        let _ = ESTIMATE.try_with(|estimate| {
            let mut estimate = estimate.borrow_mut();
            // DP ranks keep separate KV caches, so a remapped request loses the estimate.
            if estimate
                .as_ref()
                .is_some_and(|e| e.worker.url() != worker.url())
            {
                *estimate = None;
            }
        });
    }
    let _ = TRACE.try_with(|trace| {
        let selections = trace.selections.borrow();
        let selection = selections.iter().find(|selection| selection.pool == pool);
        info!(
            target: "mesh::route",
            pool,
            policy = selection.map(|selection| selection.policy),
            branch = selection.map(|selection| selection.branch),
            worker = worker.base_url(),
            dp_rank = worker.dp_rank(),
            remapped_from = selection
                .filter(|selection| selection.url != worker.url())
                .map(|selection| selection.url.as_str()),
            match_ratio = selection.and_then(|selection| selection.match_ratio),
            active_requests = selection.map(|selection| selection.active_requests),
            "Selected worker"
        );
    });
}

#[cfg(test)]
mod tests {
    use std::io;

    use axum::{body::Body, routing::get, Router};
    use tower::ServiceExt;

    use super::*;
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType};

    #[derive(Clone, Default)]
    struct Capture(Arc<std::sync::Mutex<Vec<u8>>>);

    impl io::Write for Capture {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl Capture {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    fn prefill_rank(rank: usize) -> Arc<dyn Worker> {
        Arc::new(
            DPAwareWorkerBuilder::new("http://prefill:8000", rank, 2)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .build(),
        )
    }

    async fn logs(sampled: bool, run: impl Future<Output = ()>) -> String {
        let capture = Capture::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(capture.clone())
            .with_ansi(false)
            .finish();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        if sampled {
            TRACE.scope(RouteTrace::default(), run).await;
        } else {
            run.await;
        }
        capture.text()
    }

    #[test]
    fn header_forces_the_log_only_while_sampling() {
        let mut headers = HeaderMap::new();
        assert!(!should_log(&headers, 0.0));
        assert!(should_log(&headers, 1.0));
        headers.insert(ROUTE_TRACE_HEADER, "1".parse().unwrap());
        assert!(!should_log(&headers, 0.0));
        assert!(should_log(&headers, f64::MIN_POSITIVE));
    }

    #[tokio::test]
    async fn only_sampled_requests_get_a_trace() {
        let cases = [
            (1.0, false, "true"),
            (f64::MIN_POSITIVE, true, "true"),
            (f64::MIN_POSITIVE, false, "false"),
        ];
        for (rate, header, traced) in cases {
            let app = Router::new()
                .route(
                    "/",
                    get(|| async { TRACE.try_with(|_| ()).is_ok().to_string() }),
                )
                .layer(axum::middleware::from_fn_with_state(rate, scope));
            let mut request = Request::builder().uri("/");
            if header {
                request = request.header(ROUTE_TRACE_HEADER, "1");
            }

            let response = app
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap();
            let body = axum::body::to_bytes(response.into_body(), 64)
                .await
                .unwrap();
            assert_eq!(&body[..], traced.as_bytes(), "rate {rate} header {header}");
        }
    }

    #[test]
    fn dispatch_counts_the_rank_after_mapping() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let regular: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://regular:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        metrics::with_local_recorder(&recorder, || {
            record_dispatch("prefill", &prefill_rank(1));
            record_dispatch("regular", &regular);
        });

        let rendered = handle.render();
        for line in [
            r#"mesh_selected_dp_rank_total{pool="prefill",worker="http://prefill:8000",dp_rank="1"} 1"#,
            r#"mesh_selected_dp_rank_total{pool="regular",worker="http://regular:8000",dp_rank="none"} 1"#,
        ] {
            assert!(rendered.contains(line), "missing {line} in\n{rendered}");
        }
    }

    #[tokio::test]
    async fn the_estimate_follows_the_last_prompt_selection_until_a_remap() {
        let worker = |url: &str, worker_type| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .worker_type(worker_type)
                    .build(),
            )
        };
        let regular = worker("http://regular:8000", WorkerType::Regular);
        let decode = worker("http://decode:8000", WorkerType::Decode);

        let (_, kept) = estimated(async {
            record_estimate("prefill", Some(&prefill_rank(0)), Some(0.5));
            record_estimate("decode", Some(&decode), Some(0.9));
            record_dispatch("prefill", &prefill_rank(0));
            record_dispatch("decode", &decode);
        })
        .await;
        let (_, remapped) = estimated(async {
            record_estimate("prefill", Some(&prefill_rank(0)), Some(0.5));
            record_dispatch("prefill", &prefill_rank(1));
        })
        .await;
        let (_, retried) = estimated(async {
            record_estimate("regular", Some(&regular), Some(0.5));
            record_estimate("regular", Some(&regular), None);
            record_dispatch("regular", &regular);
        })
        .await;
        assert_eq!((kept, remapped, retried), (Some(0.5), None, None));
        // Outside `estimated` there is nothing to keep.
        record_estimate("regular", Some(&regular), Some(0.5));
        record_dispatch("regular", &regular);
    }

    #[tokio::test]
    async fn sampled_dispatch_logs_the_decision_and_remap() {
        let run = || {
            let selected = prefill_rank(0);
            record_selection(
                "prefill",
                "cache_aware",
                Some(&selected),
                "affinity",
                Some(0.75),
            );
            record_dispatch("prefill", &prefill_rank(1));
        };

        let sampled = logs(true, async { run() }).await;
        for field in [
            "Selected worker",
            r#"pool="prefill""#,
            r#"policy="cache_aware""#,
            r#"branch="affinity""#,
            r#"worker="http://prefill:8000""#,
            "dp_rank=1",
            r#"remapped_from="http://prefill:8000@0""#,
            "match_ratio=0.75",
            "active_requests=0",
        ] {
            assert!(sampled.contains(field), "missing {field} in\n{sampled}");
        }
        assert!(logs(false, async { run() }).await.is_empty());
    }

    #[tokio::test]
    async fn carried_work_dispatches_in_the_request_span_and_trace() {
        let sampled = logs(true, async {
            let selected = prefill_rank(0);
            record_selection("prefill", "cache_aware", Some(&selected), "affinity", None);
            async {
                let plain = async { record_dispatch("prefill", &prefill_rank(0)) };
                tokio::spawn(plain).await.unwrap();
                let carried = carry(async { record_dispatch("prefill", &prefill_rank(0)) });
                tokio::spawn(carried).await.unwrap();
            }
            .instrument(tracing::info_span!("request", id = 7))
            .await;
        })
        .await;

        assert_eq!(sampled.matches("Selected worker").count(), 1, "{sampled}");
        assert!(sampled.contains("request{id=7}"), "{sampled}");
        assert!(
            sampled.contains(r#"policy="cache_aware" branch="affinity""#),
            "{sampled}"
        );
    }

    #[tokio::test]
    async fn a_retry_replaces_the_pools_selection() {
        let sampled = logs(true, async {
            record_selection(
                "prefill",
                "cache_aware",
                Some(&prefill_rank(0)),
                "affinity",
                Some(0.75),
            );
            record_selection(
                "prefill",
                "round_robin",
                Some(&prefill_rank(1)),
                "round_robin",
                None,
            );
            record_dispatch("prefill", &prefill_rank(1));
        })
        .await;

        assert_eq!(sampled.matches("Selected worker").count(), 1, "{sampled}");
        assert!(
            sampled.contains(r#"policy="round_robin" branch="round_robin""#),
            "{sampled}"
        );
        assert!(!sampled.contains("remapped_from"), "{sampled}");
    }

    #[tokio::test]
    async fn a_selection_without_a_worker_logs_its_pool() {
        let sampled = logs(true, async {
            record_selection("decode", "round_robin", None, "no_healthy_workers", None);
        })
        .await;

        for field in [
            "No worker selected",
            r#"pool="decode""#,
            r#"policy="round_robin""#,
            r#"branch="no_healthy_workers""#,
        ] {
            assert!(sampled.contains(field), "missing {field} in\n{sampled}");
        }
    }

    #[tokio::test]
    async fn a_pool_without_candidates_logs_its_failure() {
        use crate::core::placement::{planner::DefaultPlanner, test_support::*, PdPlanner};

        let unhealthy = |worker: Arc<dyn Worker>| {
            worker.set_healthy(false);
            worker
        };
        let prefill = || make_prefill_http("http://p:8000", "m", Some(8998));
        let decode = || make_decode_http("http://d:8000", "m");
        let cases = [
            (
                "regular",
                vec![unhealthy(make_regular_http("http://w:8000", "m"))],
            ),
            ("prefill", vec![unhealthy(prefill()), decode()]),
            ("decode", vec![prefill(), unhealthy(decode())]),
            ("prefill", vec![decode()]),
            ("decode", vec![prefill()]),
        ];
        for (pool, workers) in cases {
            let source = workers
                .into_iter()
                .fold(MockWorkerSource::new(), MockWorkerSource::add_worker);
            let planner = DefaultPlanner::new(Arc::new(source), Arc::new(MockPolicySource::new()));
            let sampled = logs(true, async {
                let descriptor = make_descriptor(Some("m"), None, None, None);
                assert!(planner.plan(&descriptor).await.is_err());
            })
            .await;

            for field in [
                "No worker selected",
                &format!(r#"pool="{pool}""#),
                r#"policy="round_robin""#,
                r#"branch="no_healthy_workers""#,
            ] {
                assert!(sampled.contains(field), "missing {field} in\n{sampled}");
            }
        }
    }
}
