#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use http::HeaderMap;

use super::traits::{PolicySource, WorkerSource};
use super::types::RequestDescriptor;
use crate::core::{BasicWorkerBuilder, ConnectionMode, HashRing, RuntimeType, Worker, WorkerType};
use crate::policies::{LoadBalancingPolicy, RoundRobinPolicy, SelectWorkerInfo};

#[derive(Debug, Clone)]
pub struct RecordedSelectInfo {
    pub request_text: Option<String>,
    pub tokens: Option<Vec<u32>>,
    pub headers: Option<HeaderMap>,
    pub hash_ring: Option<Arc<HashRing>>,
    pub candidate_urls: Vec<String>,
}

impl RecordedSelectInfo {
    pub fn hash_ring_present(&self) -> bool {
        self.hash_ring.is_some()
    }
}

#[derive(Default)]
pub struct MockWorkerSource {
    pub workers: Vec<Arc<dyn Worker>>,
    pub hash_rings: HashMap<String, Arc<HashRing>>,
    pub hash_ring_calls: Arc<Mutex<Vec<String>>>,
}

impl MockWorkerSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_worker(mut self, w: Arc<dyn Worker>) -> Self {
        self.workers.push(w);
        self
    }

    pub fn with_hash_ring(mut self, model_id: impl Into<String>, ring: Arc<HashRing>) -> Self {
        self.hash_rings.insert(model_id.into(), ring);
        self
    }

    pub fn hash_ring_call_log(&self) -> Vec<String> {
        self.hash_ring_calls.lock().unwrap().clone()
    }
}

impl WorkerSource for MockWorkerSource {
    fn workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
    ) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .filter(|w| match model_id {
                Some(m) => w.model_id() == m,
                None => true,
            })
            .filter(|w| match &worker_type {
                Some(WorkerType::Prefill { .. }) => {
                    matches!(w.worker_type(), WorkerType::Prefill { .. })
                }
                Some(wt) => w.worker_type() == wt,
                None => true,
            })
            .filter(|w| match &connection_mode {
                Some(cm) => w.connection_mode().matches(cm),
                None => true,
            })
            .cloned()
            .collect()
    }

    fn hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.hash_ring_calls
            .lock()
            .unwrap()
            .push(model_id.to_string());
        self.hash_rings.get(model_id).cloned()
    }
}

pub struct MockPolicySource {
    pub regular: Arc<dyn LoadBalancingPolicy>,
    pub regular_per_model: HashMap<String, Arc<dyn LoadBalancingPolicy>>,
    pub prefill: Arc<dyn LoadBalancingPolicy>,
    pub decode: Arc<dyn LoadBalancingPolicy>,
}

impl Default for MockPolicySource {
    fn default() -> Self {
        let rr: Arc<dyn LoadBalancingPolicy> = Arc::new(RoundRobinPolicy::new());
        Self {
            regular: rr.clone(),
            regular_per_model: HashMap::new(),
            prefill: rr.clone(),
            decode: rr,
        }
    }
}

impl MockPolicySource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_regular(mut self, p: Arc<dyn LoadBalancingPolicy>) -> Self {
        self.regular = p;
        self
    }

    pub fn with_regular_for_model(
        mut self,
        model_id: impl Into<String>,
        p: Arc<dyn LoadBalancingPolicy>,
    ) -> Self {
        self.regular_per_model.insert(model_id.into(), p);
        self
    }

    pub fn with_prefill(mut self, p: Arc<dyn LoadBalancingPolicy>) -> Self {
        self.prefill = p;
        self
    }

    pub fn with_decode(mut self, p: Arc<dyn LoadBalancingPolicy>) -> Self {
        self.decode = p;
        self
    }
}

impl PolicySource for MockPolicySource {
    fn regular_policy(&self, model_id: Option<&str>) -> Arc<dyn LoadBalancingPolicy> {
        match model_id.and_then(|m| self.regular_per_model.get(m)) {
            Some(p) => p.clone(),
            None => self.regular.clone(),
        }
    }
    fn prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.prefill.clone()
    }
    fn decode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.decode.clone()
    }
}

#[derive(Debug)]
pub struct RecordingPolicy {
    inner: Arc<dyn LoadBalancingPolicy>,
    pub calls: Arc<Mutex<Vec<RecordedSelectInfo>>>,
}

impl RecordingPolicy {
    pub fn wrap(inner: Arc<dyn LoadBalancingPolicy>) -> Self {
        Self {
            inner,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn round_robin() -> Self {
        Self::wrap(Arc::new(RoundRobinPolicy::new()))
    }

    pub fn calls(&self) -> Vec<RecordedSelectInfo> {
        self.calls.lock().unwrap().clone()
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl LoadBalancingPolicy for RecordingPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.calls.lock().unwrap().push(RecordedSelectInfo {
            request_text: info.request_text.map(|s| s.to_string()),
            tokens: info.tokens.map(|t| t.to_vec()),
            headers: info.headers.cloned(),
            hash_ring: info.hash_ring.clone(),
            candidate_urls: workers.iter().map(|w| w.url().to_string()).collect(),
        });
        self.inner.select_worker(workers, info).await
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn needs_request_text(&self) -> bool {
        self.inner.needs_request_text()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Default)]
pub struct AlwaysNonePolicy;

#[async_trait]
impl LoadBalancingPolicy for AlwaysNonePolicy {
    async fn select_worker(
        &self,
        _workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        None
    }

    fn name(&self) -> &'static str {
        "always_none"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub struct StaticNeedsTextPolicy {
    name: &'static str,
    needs: bool,
}

impl StaticNeedsTextPolicy {
    pub fn new(name: &'static str, needs: bool) -> Self {
        Self { name, needs }
    }
}

#[async_trait]
impl LoadBalancingPolicy for StaticNeedsTextPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        if workers.is_empty() {
            None
        } else {
            Some(0)
        }
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn needs_request_text(&self) -> bool {
        self.needs
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[allow(clippy::too_many_arguments)]
pub fn make_worker(
    url: &str,
    worker_type: WorkerType,
    healthy: bool,
    model_id: Option<&str>,
    api_key: Option<&str>,
    connection_mode: ConnectionMode,
    runtime_type: RuntimeType,
) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(url.to_string())
        .worker_type(worker_type)
        .connection_mode(connection_mode)
        .runtime_type(runtime_type);
    if let Some(m) = model_id {
        builder = builder.model_id(m.to_string());
    }
    if let Some(k) = api_key {
        builder = builder.api_key(k.to_string());
    }
    let w: Arc<dyn Worker> = Arc::new(builder.build());
    if !healthy {
        w.set_healthy(false);
    }
    w
}

pub fn make_regular_http(url: &str, model_id: &str) -> Arc<dyn Worker> {
    make_worker(
        url,
        WorkerType::Regular,
        true,
        Some(model_id),
        None,
        ConnectionMode::Http,
        RuntimeType::Sglang,
    )
}

pub fn make_regular_grpc(url: &str, model_id: &str) -> Arc<dyn Worker> {
    make_worker(
        url,
        WorkerType::Regular,
        true,
        Some(model_id),
        None,
        ConnectionMode::Grpc { port: None },
        RuntimeType::Sglang,
    )
}

pub fn make_prefill_http(
    url: &str,
    model_id: &str,
    bootstrap_port: Option<u16>,
) -> Arc<dyn Worker> {
    make_worker(
        url,
        WorkerType::Prefill { bootstrap_port },
        true,
        Some(model_id),
        None,
        ConnectionMode::Http,
        RuntimeType::Sglang,
    )
}

pub fn make_decode_http(url: &str, model_id: &str) -> Arc<dyn Worker> {
    make_worker(
        url,
        WorkerType::Decode,
        true,
        Some(model_id),
        None,
        ConnectionMode::Http,
        RuntimeType::Sglang,
    )
}

pub fn make_prefill_grpc(
    url: &str,
    model_id: &str,
    bootstrap_port: Option<u16>,
) -> Arc<dyn Worker> {
    make_worker(
        url,
        WorkerType::Prefill { bootstrap_port },
        true,
        Some(model_id),
        None,
        ConnectionMode::Grpc { port: None },
        RuntimeType::Sglang,
    )
}

pub fn make_decode_grpc(url: &str, model_id: &str) -> Arc<dyn Worker> {
    make_worker(
        url,
        WorkerType::Decode,
        true,
        Some(model_id),
        None,
        ConnectionMode::Grpc { port: None },
        RuntimeType::Sglang,
    )
}

pub fn make_descriptor<'a>(
    model_id: Option<&'a str>,
    text: Option<&'a str>,
    tokens: Option<&'a [u32]>,
    headers: Option<&'a HeaderMap>,
) -> RequestDescriptor<'a> {
    RequestDescriptor {
        model_id,
        protocol: None,
        text,
        tokens,
        headers,
        stream: false,
    }
}

#[cfg(test)]
mod fixture_smoke_tests {
    use super::*;

    #[test]
    fn mock_worker_source_filters_by_model_id() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://m1-w1:8000", "m1"))
            .add_worker(make_regular_http("http://m1-w2:8000", "m1"))
            .add_worker(make_regular_http("http://m2-w1:8000", "m2"));

        assert_eq!(src.workers_filtered(Some("m1"), None, None).len(), 2);
        assert_eq!(src.workers_filtered(Some("m2"), None, None).len(), 1);
        assert_eq!(src.workers_filtered(None, None, None).len(), 3);
    }

    #[test]
    fn mock_worker_source_filters_by_worker_type_and_connection_mode() {
        let src = MockWorkerSource::new()
            .add_worker(make_regular_http("http://r:8000", "m"))
            .add_worker(make_prefill_http("http://p:8000", "m", Some(8998)))
            .add_worker(make_decode_http("http://d:8000", "m"))
            .add_worker(make_regular_grpc("http://g:8000", "m"));

        assert_eq!(
            src.workers_filtered(None, Some(WorkerType::Regular), None)
                .len(),
            2
        );
        assert_eq!(
            src.workers_filtered(None, None, Some(ConnectionMode::Http))
                .len(),
            3
        );
        assert_eq!(
            src.workers_filtered(None, None, Some(ConnectionMode::Grpc { port: None }))
                .len(),
            1
        );
    }

    #[test]
    fn mock_worker_source_records_hash_ring_calls() {
        let src = MockWorkerSource::new();
        assert!(src.hash_ring("m1").is_none());
        assert!(src.hash_ring("m2").is_none());
        assert_eq!(src.hash_ring_call_log(), vec!["m1", "m2"]);
    }

    #[tokio::test]
    async fn recording_policy_records_select_worker_info() {
        let workers = vec![make_regular_http("http://w:8000", "m")];
        let policy = RecordingPolicy::round_robin();
        let mut hm = HeaderMap::new();
        hm.insert("x-test", "1".parse().unwrap());
        let info = SelectWorkerInfo {
            request_text: Some("hi"),
            tokens: Some(&[1, 2, 3]),
            headers: Some(&hm),
            hash_ring: None,
        };
        let _ = policy.select_worker(&workers, &info).await;
        let calls = policy.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].request_text.as_deref(), Some("hi"));
        assert_eq!(calls[0].tokens.as_deref(), Some(&[1u32, 2, 3][..]));
        assert_eq!(
            calls[0].headers.as_ref().and_then(|h| h.get("x-test")),
            Some(&"1".parse().unwrap())
        );
        assert_eq!(calls[0].candidate_urls, vec!["http://w:8000".to_string()]);
    }

    #[tokio::test]
    async fn always_none_policy_returns_none() {
        let workers = vec![make_regular_http("http://w:8000", "m")];
        let info = SelectWorkerInfo::default();
        assert_eq!(AlwaysNonePolicy.select_worker(&workers, &info).await, None);
    }

    #[test]
    fn mock_policy_source_per_model_lookup() {
        let p1: Arc<dyn LoadBalancingPolicy> = Arc::new(RoundRobinPolicy::new());
        let p2: Arc<dyn LoadBalancingPolicy> = Arc::new(AlwaysNonePolicy);
        let src = MockPolicySource::new()
            .with_regular(p1)
            .with_regular_for_model("m_special", p2);

        assert_eq!(src.regular_policy(Some("m_special")).name(), "always_none");
        assert_eq!(src.regular_policy(Some("m_other")).name(), "round_robin");
        assert_eq!(src.regular_policy(None).name(), "round_robin");
    }

    #[test]
    fn pd_needs_request_text_aggregates_prefill_and_decode() {
        let yes: Arc<dyn LoadBalancingPolicy> = Arc::new(StaticNeedsTextPolicy::new("yes", true));
        let no: Arc<dyn LoadBalancingPolicy> = Arc::new(StaticNeedsTextPolicy::new("no", false));

        let both_no = MockPolicySource::new()
            .with_prefill(no.clone())
            .with_decode(no.clone());
        assert!(!both_no.pd_needs_request_text());

        let prefill_yes = MockPolicySource::new()
            .with_prefill(yes.clone())
            .with_decode(no.clone());
        assert!(prefill_yes.pd_needs_request_text());

        let decode_yes = MockPolicySource::new().with_prefill(no).with_decode(yes);
        assert!(decode_yes.pd_needs_request_text());
    }

    #[test]
    fn make_worker_unhealthy_flag_takes_effect() {
        let w = make_worker(
            "http://x:8000",
            WorkerType::Regular,
            false,
            Some("m"),
            None,
            ConnectionMode::Http,
            RuntimeType::Sglang,
        );
        assert!(!w.is_healthy());
    }

    #[test]
    fn backend_adapter_is_object_safe() {
        use super::super::backend::sglang::SglangAdapter;
        use super::super::backend::vllm::{VllmAdapter, VllmPrefillInfo};
        use super::super::backend::BackendAdapter;
        let _sglang: Arc<dyn BackendAdapter> = Arc::new(SglangAdapter);
        let _vllm: Arc<dyn BackendAdapter> =
            Arc::new(VllmAdapter::new(Arc::new(VllmPrefillInfo::default())));
    }

    #[test]
    fn make_descriptor_round_trips_all_fields() {
        let headers = HeaderMap::new();
        let tokens = vec![1u32, 2, 3];
        let d = make_descriptor(Some("m"), Some("hi"), Some(&tokens), Some(&headers));
        assert_eq!(d.model_id, Some("m"));
        assert_eq!(d.text, Some("hi"));
        assert_eq!(d.tokens, Some(&tokens[..]));
        assert!(d.headers.is_some());
    }
}
