use std::time::Instant;

use crate::observability::request::RequestMetrics;

use super::{admission::AdmissionLease, routing::RoutingDecision};

#[derive(Clone, Copy)]
pub(super) enum Outcome {
    Completed,
    Canceled,
    ProxyDisconnected,
    Drained,
    Failed(&'static str),
}
impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Canceled => "canceled",
            Self::ProxyDisconnected => "proxy_disconnected",
            Self::Drained => "drain_timeout",
            Self::Failed(code) => code,
        }
    }
}

pub(super) struct RequestLifecycle {
    pub started: Instant,
    pub status: Option<u16>,
    pub response_started: bool,
    pub streaming: bool,
    outcome: Outcome,
    pub from_upstream: bool,
    upstream_failure: bool,
    lease: Option<AdmissionLease>,
    worker: Option<RoutingDecision>,
    pub observation: Option<RequestMetrics>,
}

impl RequestLifecycle {
    pub fn new() -> Self {
        metrics::gauge!("mesh_ext_proc_active_streams").increment(1.0);
        Self {
            started: Instant::now(),
            status: None,
            response_started: false,
            streaming: false,
            outcome: Outcome::Canceled,
            from_upstream: false,
            upstream_failure: false,
            lease: None,
            worker: None,
            observation: None,
        }
    }

    pub fn admit(&mut self, lease: AdmissionLease) {
        self.lease = Some(lease);
    }

    pub fn bind(&mut self, decision: RoutingDecision) {
        self.worker = Some(decision);
    }

    pub fn body(&mut self, bytes: &[u8]) {
        if let Some(observation) = &mut self.observation {
            observation.streaming = self.streaming;
            observation.body(bytes);
        }
    }

    pub fn finish(&mut self, outcome: Outcome) {
        self.outcome = outcome;
    }
    pub fn observe_attributes(
        &mut self,
        attributes: &std::collections::HashMap<String, prost_types::Struct>,
    ) {
        for namespace in attributes.values() {
            for (key, value) in &namespace.fields {
                if key != "response.code_details" {
                    continue;
                }
                if let Some(prost_types::value::Kind::StringValue(detail)) = &value.kind {
                    if detail.is_empty() {
                        continue;
                    }
                    let failure = detail == "response_timeout"
                        || detail.starts_with("upstream_reset")
                        || detail.starts_with("upstream_response_timeout")
                        || detail.starts_with("upstream_per_try_timeout");
                    self.upstream_failure |= failure;
                    self.from_upstream = detail == "via_upstream" || failure;
                }
            }
        }
    }

    pub fn expiration(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        self.worker
            .as_ref()
            .map(|w| w.target.expiration())
            .unwrap_or_else(|| Box::pin(std::future::pending()))
    }
}

impl Drop for RequestLifecycle {
    fn drop(&mut self) {
        // Complete usage frames remain valid even if the transport later aborts.
        if let Some(observation) = &mut self.observation {
            let error = match self.outcome {
                Outcome::Completed if self.upstream_failure => Some("upstream_failure"),
                Outcome::Completed => None,
                other => Some(other.label()),
            };
            observation.finish(self.status, error);
        }
        if let Some(decision) = self.worker.take() {
            let success = matches!(self.outcome, Outcome::Completed)
                && self.from_upstream
                && !self.upstream_failure
                && self.status.is_some_and(|s| (200..400).contains(&s));
            let health = if self.upstream_failure
                || (self.from_upstream && self.status.is_some_and(|s| s >= 500))
            {
                Some(false)
            } else if success {
                Some(true)
            } else {
                None
            };
            decision.target.complete(health, success);
        }
        metrics::counter!("mesh_ext_proc_streams_total", "outcome" => self.outcome.label())
            .increment(1);
        metrics::histogram!("mesh_ext_proc_stream_seconds")
            .record(self.started.elapsed().as_secs_f64());
        metrics::gauge!("mesh_ext_proc_active_streams").decrement(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::super::routing::ExecutionTarget;
    use super::*;
    use crate::core::{BasicWorkerBuilder, Worker, WorkerLoadGuard};
    use std::sync::Arc;

    fn bound() -> (RequestLifecycle, Arc<dyn Worker>) {
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://127.0.0.1:80").build());
        let mut lifecycle = RequestLifecycle::new();
        lifecycle.bind(RoutingDecision {
            address: "127.0.0.1:80".parse().unwrap(),
            authorization: None,
            target: ExecutionTarget::Single {
                _load: WorkerLoadGuard::new(worker.clone(), None),
                worker: worker.clone(),
                policy: Arc::new(crate::policies::RoundRobinPolicy::new()),
            },
        });
        (lifecycle, worker)
    }

    #[test]
    fn terminal_health_uses_response_origin_and_known_failures() {
        for (detail, status, outcome, successes, failures) in [
            ("via_upstream", 200, Outcome::Completed, 1, 0),
            ("via_upstream", 503, Outcome::Completed, 0, 1),
            ("via_upstream", 400, Outcome::Completed, 0, 0),
            ("via_upstream", 200, Outcome::Canceled, 0, 0),
            ("via_upstream", 200, Outcome::ProxyDisconnected, 0, 0),
            ("via_upstream", 200, Outcome::Drained, 0, 0),
            ("request_payload_too_large", 503, Outcome::Completed, 0, 0),
            ("upstream_response_timeout", 504, Outcome::Completed, 0, 1),
            ("response_timeout", 504, Outcome::Completed, 0, 1),
            (
                "upstream_reset_after_response_started{connection_failure}",
                200,
                Outcome::ProxyDisconnected,
                0,
                1,
            ),
        ] {
            let (mut lifecycle, worker) = bound();
            lifecycle.status = Some(status);
            lifecycle.observe_attributes(&std::collections::HashMap::from([(
                "envoy.filters.http.ext_proc".into(),
                prost_types::Struct {
                    fields: std::collections::BTreeMap::from([(
                        "response.code_details".into(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::StringValue(detail.into())),
                        },
                    )]),
                },
            )]));
            lifecycle.finish(outcome);
            drop(lifecycle);
            assert_eq!(worker.load(), 0, "{detail}");
            assert_eq!(
                worker.circuit_breaker().total_successes(),
                successes,
                "{detail}"
            );
            assert_eq!(
                worker.circuit_breaker().total_failures(),
                failures,
                "{detail}"
            );
        }
    }

    #[test]
    fn complete_usage_is_recorded_once_after_abnormal_termination() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            for outcome in [
                Outcome::Canceled,
                Outcome::ProxyDisconnected,
                Outcome::Drained,
                Outcome::Failed("stream_timeout"),
            ] {
                let mut lifecycle = RequestLifecycle::new();
                lifecycle.streaming = true;
                lifecycle.observation = Some(RequestMetrics::new(
                    "ext_proc",
                    "pd",
                    "test",
                    "/generate",
                    true,
                ));
                lifecycle
                    .body(b"data: {\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3}}\n\n");
                // A later incomplete frame must not erase the complete usage.
                lifecycle.body(b"data: {\"usage\":");
                lifecycle.finish(outcome);
                drop(lifecycle);
            }
        });
        let metrics = handle.render();
        assert!(
            metrics.contains("mesh_ext_proc_tokens_total{kind=\"prompt\"} 48"),
            "{metrics}"
        );
        assert!(
            metrics.contains("mesh_ext_proc_tokens_total{kind=\"completion\"} 12"),
            "{metrics}"
        );
    }
}
