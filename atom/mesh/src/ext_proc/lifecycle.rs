use std::time::Instant;

use crate::observability::ttft::FirstOutputSse;

use super::{admission::AdmissionLease, response::UsageObserver, routing::RoutingDecision};

pub(super) struct RequestLifecycle {
    pub started: Instant,
    pub status: Option<u16>,
    pub response_started: bool,
    pub streaming: bool,
    outcome: &'static str,
    lease: Option<AdmissionLease>,
    worker: Option<RoutingDecision>,
    detector: FirstOutputSse,
    usage: UsageObserver,
}

impl RequestLifecycle {
    pub fn new() -> Self {
        metrics::gauge!("mesh_ext_proc_active_streams").increment(1.0);
        Self {
            started: Instant::now(),
            status: None,
            response_started: false,
            streaming: false,
            outcome: "canceled",
            lease: None,
            worker: None,
            detector: FirstOutputSse::default(),
            usage: UsageObserver::default(),
        }
    }

    pub fn admit(&mut self, lease: AdmissionLease) {
        self.lease = Some(lease);
    }

    pub fn bind(&mut self, decision: RoutingDecision) {
        self.worker = Some(decision);
    }

    pub fn body(&mut self, bytes: &[u8]) {
        self.usage.feed(bytes, self.streaming);
        if self.streaming && self.detector.feed(bytes) {
            metrics::histogram!("mesh_ext_proc_ttft_seconds")
                .record(self.started.elapsed().as_secs_f64());
        }
    }

    pub fn finish(&mut self, outcome: &'static str) {
        self.outcome = outcome;
    }
}

impl Drop for RequestLifecycle {
    fn drop(&mut self) {
        if self.outcome == "completed" {
            self.usage.record(self.streaming);
            if let (Some(decision), Some(status)) = (&self.worker, self.status) {
                decision.target.complete(status);
            }
        }
        metrics::counter!("mesh_ext_proc_streams_total", "outcome" => self.outcome).increment(1);
        metrics::histogram!("mesh_ext_proc_stream_seconds")
            .record(self.started.elapsed().as_secs_f64());
        metrics::gauge!("mesh_ext_proc_active_streams").decrement(1.0);
    }
}
