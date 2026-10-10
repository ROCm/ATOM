use std::{collections::HashSet, sync::Arc, time::Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use http::{HeaderMap, HeaderName, HeaderValue};
use prost_types::value::Kind;

use crate::{
    app_context::AppContext,
    core::prepare_pool::{InputLease, PrepareHandle},
    routers::ingress::IngressRouting,
};

use super::{core, error::ProcessingError, pb};
use crate::routers::prepare::inference::{InferenceMetadata, ParsedInference};

pub(super) struct RequestEnvelope {
    pub headers: HeaderMap,
    pub path: String,
    pub id: String,
    pub raw: Vec<u8>,
    pub trailer_mutation: Option<pb::HeaderMutation>,
    pub subset: Option<HashSet<String>>,
    buffered_bytes: usize,
    budget: Option<Arc<Semaphore>>,
    memory: Vec<OwnedSemaphorePermit>,
}

pub(super) struct RoutingInput {
    pub metadata: InferenceMetadata,
    pub tokens: Option<Vec<u32>>,
    pub state_reference: bool,
}
impl std::ops::Deref for RoutingInput {
    type Target = InferenceMetadata;
    fn deref(&self) -> &Self::Target {
        &self.metadata
    }
}

impl RequestEnvelope {
    /// Resolve correlation before validating the request so local errors can echo it.
    pub fn request_id(input: &pb::HttpHeaders, names: &[String]) -> String {
        let headers = input
            .headers
            .as_ref()
            .map(|h| h.headers.as_slice())
            .unwrap_or_default();
        let path = headers
            .iter()
            .find(|h| h.key == ":path")
            .and_then(|h| std::str::from_utf8(Self::header_bytes(h)).ok())
            .unwrap_or("")
            .split('?')
            .next()
            .unwrap_or("");
        crate::observability::request_id::resolve(names, path, |name| {
            let name = HeaderName::from_bytes(name.as_bytes()).ok()?;
            let header = headers
                .iter()
                .find(|h| h.key.eq_ignore_ascii_case(name.as_str()))?;
            HeaderValue::from_bytes(Self::header_bytes(header))
                .ok()?
                .to_str()
                .ok()
                .map(str::to_owned)
        })
    }

    pub fn new(input: pb::HttpHeaders, id: String) -> Result<Self, ProcessingError> {
        let mut headers = HeaderMap::new();
        let mut path = None;
        let mut method = None;
        for header in input.headers.unwrap_or_default().headers {
            let value = Self::header_bytes(&header);
            match header.key.as_str() {
                ":path" => {
                    if path
                        .replace(
                            String::from_utf8(value.to_vec())
                                .map_err(|_| ProcessingError::invalid("invalid path"))?,
                        )
                        .is_some()
                    {
                        return Err(ProcessingError::invalid("duplicate :path"));
                    }
                }
                ":method" => {
                    if method.replace(value.to_vec()).is_some() {
                        return Err(ProcessingError::invalid("duplicate :method"));
                    }
                }
                key if key.starts_with(':') => {}
                _ => {
                    let name = HeaderName::from_bytes(header.key.as_bytes())
                        .map_err(|_| ProcessingError::invalid("invalid header name"))?;
                    let value = HeaderValue::from_bytes(value)
                        .map_err(|_| ProcessingError::invalid("invalid header value"))?;
                    headers.append(name, value);
                }
            }
        }
        if method.as_deref() != Some(b"POST") {
            return Err(ProcessingError::new(
                405,
                "method_not_allowed",
                "ext-proc inference routes require POST",
            ));
        }
        let path = path.ok_or_else(|| ProcessingError::invalid("missing :path"))?;
        let route = path.split('?').next().unwrap_or("");
        if crate::routers::ingress::EndpointSpec::find(route).is_none() {
            return Err(ProcessingError::new(
                404,
                "unsupported_path",
                "unsupported inference API",
            ));
        }
        crate::routers::ingress::InferenceEnvelope::validate_headers(&headers)
            .map_err(ProcessingError::from)?;
        headers.insert(
            "x-request-id",
            HeaderValue::from_str(&id)
                .map_err(|_| ProcessingError::invalid("invalid request ID"))?,
        );
        headers.remove(super::mutation::Mutation::DESTINATION);
        Ok(Self {
            headers,
            path,
            id,
            raw: Vec::new(),
            trailer_mutation: None,
            subset: None,
            buffered_bytes: 0,
            budget: None,
            memory: Vec::new(),
        })
    }

    pub fn header_bytes(header: &core::HeaderValue) -> &[u8] {
        if header.raw_value.is_empty() {
            header.value.as_bytes()
        } else {
            &header.raw_value
        }
    }

    pub fn metadata(&mut self, metadata: Option<core::Metadata>) -> Result<(), ProcessingError> {
        let Some(metadata) = metadata else {
            return Ok(());
        };
        let Some(namespace) = metadata.filter_metadata.get("envoy.lb.subset_hint") else {
            return Ok(());
        };
        let Some(value) = namespace
            .fields
            .get("x-gateway-destination-endpoint-subset")
        else {
            return Ok(());
        };
        let Some(Kind::ListValue(list)) = &value.kind else {
            return Err(ProcessingError::invalid("endpoint subset must be a list"));
        };
        self.subset = if list.values.is_empty() {
            None
        } else {
            Some(
                list.values
                    .iter()
                    .map(|v| match &v.kind {
                        Some(Kind::StringValue(s)) => Ok(s.clone()),
                        _ => Err(ProcessingError::invalid(
                            "endpoint subset entries must be addresses",
                        )),
                    })
                    .collect::<Result<_, _>>()?,
            )
        };
        Ok(())
    }

    pub fn set_budget(&mut self, budget: Arc<Semaphore>) {
        self.budget = Some(budget);
    }

    fn reserve_buffer(&mut self, required: usize, limit: usize) -> Result<(), ProcessingError> {
        if required > self.raw.capacity() {
            let capacity = required.next_power_of_two().min(limit);
            if let Some(budget) = &self.budget {
                let permit = budget
                    .clone()
                    .try_acquire_many_owned((capacity - self.raw.capacity()) as u32)
                    .map_err(|_| {
                        ProcessingError::new(
                            503,
                            "buffer_budget_exhausted",
                            "global request buffer budget exhausted",
                        )
                    })?;
                self.memory.push(permit);
            }
            self.raw.reserve_exact(capacity - self.raw.len());
        }
        Ok(())
    }

    pub fn replace_body(&mut self, body: Vec<u8>, limit: usize) -> Result<(), ProcessingError> {
        if body.len() > limit {
            return Err(ProcessingError::new(
                413,
                "body_too_large",
                "prepared request body limit exceeded",
            ));
        }
        self.reserve_buffer(body.len(), limit)?;
        metrics::gauge!("mesh_ext_proc_buffered_request_bytes")
            .decrement(self.buffered_bytes as f64);
        self.raw.clear();
        self.raw.extend_from_slice(&body);
        self.buffered_bytes = body.len();
        metrics::gauge!("mesh_ext_proc_buffered_request_bytes")
            .increment(self.buffered_bytes as f64);
        Ok(())
    }

    pub fn append(&mut self, body: &[u8], limit: usize) -> Result<(), ProcessingError> {
        if body.len() > limit.saturating_sub(self.raw.len()) {
            return Err(ProcessingError::new(
                413,
                "body_too_large",
                "request body limit exceeded",
            ));
        }
        self.reserve_buffer(self.raw.len() + body.len(), limit)?;
        self.buffered_bytes += body.len();
        self.raw.extend_from_slice(body);
        metrics::gauge!("mesh_ext_proc_buffered_request_bytes").increment(body.len() as f64);
        Ok(())
    }
}

/// Parses owned requests on the shared preparation pool.
pub(super) struct RequestParser {
    routing: IngressRouting,
    pool: PrepareHandle,
    pub budget: Arc<Semaphore>,
}

struct PreparingRequest {
    request: RequestEnvelope,
    // Last: a canceled job releases its actual input before its preparation charge.
    lease: InputLease,
}

impl RequestParser {
    pub fn new(app: &AppContext) -> Self {
        Self {
            routing: IngressRouting::new(app),
            pool: app.prepare_pool.clone(),
            budget: Arc::new(Semaphore::new(
                app.router_config.ext_proc.max_buffered_bytes,
            )),
        }
    }

    pub async fn parse(
        &self,
        request: RequestEnvelope,
        decision_deadline: Instant,
    ) -> Result<(RequestEnvelope, RoutingInput), ProcessingError> {
        let deadline = decision_deadline.min(self.pool.prepare_deadline());
        let lease = self.pool.retain_input(request.raw.len())?;
        let preparing = PreparingRequest { request, lease };
        let routing = self.routing.clone();
        let (input, preparing) = self
            .pool
            .submit(deadline, move |context| {
                context.check()?;
                let parsed =
                    ParsedInference::parse(&preparing.request.path, &preparing.request.raw)
                        .map_err(ProcessingError::invalid)?;
                let (metadata, tokens) = routing.prepare(&parsed, context)?;
                let input = RoutingInput {
                    metadata,
                    tokens,
                    state_reference: parsed.requires_state_domain(),
                };
                Ok::<_, ProcessingError>((input, preparing))
            })
            .await?
            .wait()
            .await??;
        // Handoff occurs only after the worker's final deadline/cancellation
        // check succeeds. Forwarding retains its independent buffer budget.
        let PreparingRequest { request, lease } = preparing;
        drop(lease);
        Ok((request, input))
    }
}

impl Drop for RequestEnvelope {
    fn drop(&mut self) {
        metrics::gauge!("mesh_ext_proc_buffered_request_bytes")
            .decrement(self.buffered_bytes as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prepare_pool::PreparePoolRuntime;
    use crate::tokenizer::{
        traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer},
        MockTokenizer,
    };
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    fn envelope(budget: Arc<Semaphore>, size: usize) -> RequestEnvelope {
        let mut request = RequestEnvelope::new(
            pb::HttpHeaders {
                headers: Some(core::HeaderMap {
                    headers: [
                        (":method", "POST"),
                        (":path", "/generate"),
                        ("content-type", "application/json"),
                    ]
                    .into_iter()
                    .map(|(key, value)| core::HeaderValue {
                        key: key.into(),
                        value: value.into(),
                        ..Default::default()
                    })
                    .collect(),
                }),
                ..Default::default()
            },
            "buffer-test".into(),
        )
        .unwrap();
        request.set_budget(budget);
        request.append(&vec![b' '; size], 1024).unwrap();
        request
    }

    fn inference_request(budget: Arc<Semaphore>) -> RequestEnvelope {
        let mut body = br#"{"model":"test-model","text":"hello","vendor":{"kept":true}}"#.to_vec();
        body.resize(512, b' ');
        let mut request = envelope(budget, 0);
        request.replace_body(body, 1024).unwrap();
        request
    }

    async fn parser_with_budget(input_bytes: usize) -> (RequestParser, PreparePoolRuntime) {
        let mut config = crate::config::RouterConfig::default();
        config.policy = crate::config::PolicyConfig::RoundRobin;
        config.prepare_pool.workers = Some(1);
        config.prepare_pool.max_retained_input_bytes = input_bytes;
        config.ext_proc.max_buffered_bytes = 1024;
        let runtime = PreparePoolRuntime::new(config.resolved_prepare_pool()).unwrap();
        let app = AppContext::from_config(config, 5, runtime.handle())
            .await
            .unwrap();
        (RequestParser::new(&app), runtime)
    }

    #[test]
    fn buffer_budget_covers_capacity_growth_rewrites_and_drop() {
        let budget = Arc::new(Semaphore::new(1024));
        let mut first = envelope(budget.clone(), 300);
        assert_eq!(budget.available_permits(), 512);
        let second = envelope(budget.clone(), 400);
        assert_eq!(budget.available_permits(), 0);
        let error = first.append(&[b'x'; 300], 1024).unwrap_err();
        assert_eq!(error.code, "buffer_budget_exhausted");
        assert_eq!(first.raw.len(), 300);
        drop(second);
        first.replace_body(vec![b'x'; 700], 1024).unwrap();
        assert_eq!(budget.available_permits(), 0);
        drop(first);
        assert_eq!(budget.available_permits(), 1024);
    }

    #[tokio::test]
    async fn forwarding_buffers_do_not_hold_the_preparation_budget() {
        let (parser, mut runtime) = parser_with_budget(512).await;
        let pool = runtime.handle();
        let request = inference_request(parser.budget.clone());
        let original = request.raw.clone();
        let (first, _) = parser
            .parse(request, pool.prepare_deadline())
            .await
            .unwrap();
        assert_eq!(first.raw, original);
        assert_eq!(pool.stats().retained_input_bytes, 0);
        assert_eq!(parser.budget.available_permits(), 512);

        // The preparation budget fits only one request, but the first forwarded
        // request can remain buffered while the next request is prepared.
        let (second, _) = parser
            .parse(
                inference_request(parser.budget.clone()),
                pool.prepare_deadline(),
            )
            .await
            .unwrap();
        assert_eq!(second.raw, original);
        assert_eq!(pool.stats().retained_input_bytes, 0);
        assert_eq!(parser.budget.available_permits(), 0);
        drop(first);
        assert_eq!(parser.budget.available_permits(), 512);
        drop(second);
        assert_eq!(parser.budget.available_permits(), 1024);
        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn forwarding_body_rewrites_only_charge_the_transport_buffer_budget() {
        let (parser, mut runtime) = parser_with_budget(512).await;
        let pool = runtime.handle();
        let (mut request, _) = parser
            .parse(
                inference_request(parser.budget.clone()),
                pool.prepare_deadline(),
            )
            .await
            .unwrap();
        request.replace_body(vec![b'x'; 700], 1024).unwrap();
        assert_eq!(pool.stats().retained_input_bytes, 0);
        assert_eq!(parser.budget.available_permits(), 0);
        let error = request.replace_body(vec![b'y'; 1025], 2048).unwrap_err();
        assert_eq!(error.code, "buffer_budget_exhausted");
        assert_eq!(request.raw, vec![b'x'; 700]);
        assert_eq!(pool.stats().retained_input_bytes, 0);

        request.replace_body(vec![b'z'; 10], 1024).unwrap();
        // A rewrite can retain the old allocation; release conservatively at drop.
        assert_eq!(parser.budget.available_permits(), 0);
        drop(request);
        assert_eq!(parser.budget.available_permits(), 1024);
        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn tokenization_rejects_oversized_text_and_rendered_chat_prompt() {
        let mut config = crate::config::RouterConfig::default();
        config.policy = crate::config::PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        };
        config.prepare_pool.max_tokenize_bytes = Some(4);
        let _runtime = PreparePoolRuntime::new(config.resolved_prepare_pool()).unwrap();
        let mut app = AppContext::from_config(config, 5, _runtime.handle())
            .await
            .unwrap();
        app.tokenizer_registry =
            crate::routers::test_mocks::tokenizer::tokenizer_registry_with_hf("test-model");
        let parser = RequestParser::new(&app);
        for (path, body) in [
            (
                "/v1/completions",
                r#"{"model":"test-model","prompt":"abcde"}"#,
            ),
            (
                "/v1/chat/completions",
                r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#,
            ),
        ] {
            let mut request = envelope(Arc::new(Semaphore::new(1024)), 0);
            request.path = path.into();
            request
                .replace_body(body.as_bytes().to_vec(), 1024)
                .unwrap();
            let error = parser
                .parse(request, parser.pool.prepare_deadline())
                .await
                .err()
                .unwrap();
            assert_eq!(error.status, 413);
            assert_eq!(error.code, "tokenizer_input_too_large");
            assert_eq!(parser.pool.stats().retained_input_bytes, 0);
        }
    }

    struct GatedTokenizer {
        inner: MockTokenizer,
        started: parking_lot::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
        gate: parking_lot::Mutex<Option<std::sync::mpsc::Receiver<()>>>,
        calls: Arc<AtomicUsize>,
    }

    impl Encoder for GatedTokenizer {
        fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let gate = self.gate.lock().take();
            if let Some(gate) = gate {
                self.started.lock().take().unwrap().send(()).unwrap();
                gate.recv_timeout(Duration::from_secs(3)).unwrap();
            }
            self.inner.encode(input, add_special_tokens)
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            self.inner.encode_batch(inputs, add_special_tokens)
        }
    }

    impl Decoder for GatedTokenizer {
        fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
            self.inner.decode(tokens, skip_special_tokens)
        }
    }

    impl Tokenizer for GatedTokenizer {
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }
        fn get_special_tokens(&self) -> &SpecialTokens {
            self.inner.get_special_tokens()
        }
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.inner.token_to_id(token)
        }
        fn id_to_token(&self, token: u32) -> Option<String> {
            self.inner.id_to_token(token)
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[tokio::test]
    async fn cancellation_releases_waiting_buffers_but_keeps_running_buffers() {
        let mut config = crate::config::RouterConfig::default();
        config.policy = crate::config::PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        };
        config.prepare_pool.workers = Some(1);
        config.prepare_pool.max_retained_input_bytes = 1024;
        config.ext_proc.max_buffered_bytes = 1024;
        let mut runtime = PreparePoolRuntime::new(config.resolved_prepare_pool()).unwrap();
        let pool = runtime.handle();
        let app = AppContext::from_config(config, 5, pool.clone())
            .await
            .unwrap();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(GatedTokenizer {
            inner: MockTokenizer::new(),
            started: parking_lot::Mutex::new(Some(started_tx)),
            gate: parking_lot::Mutex::new(Some(release_rx)),
            calls: calls.clone(),
        });
        app.tokenizer_registry
            .load("gated-test", "test-model", "mock", || async {
                Ok(tokenizer)
            })
            .await
            .unwrap();
        let parser = Arc::new(RequestParser::new(&app));
        let request = inference_request(parser.budget.clone());
        let running_parser = parser.clone();
        let deadline = pool.prepare_deadline();
        let task = tokio::spawn(async move { running_parser.parse(request, deadline).await });
        started_rx.await.unwrap();
        task.abort();
        match task.await {
            Err(error) => assert!(error.is_cancelled()),
            Ok(_) => panic!("running parser was not canceled"),
        }
        assert_eq!(pool.stats().busy_workers, 1);
        assert_eq!(pool.stats().retained_input_bytes, 512);
        assert_eq!(parser.budget.available_permits(), 512);
        assert!(matches!(
            pool.retain_input(513),
            Err(crate::core::prepare_pool::PrepareError::BudgetExceeded)
        ));

        let queued_request = inference_request(parser.budget.clone());
        let mut queued = Box::pin(parser.parse(queued_request, pool.prepare_deadline()));
        assert!(futures_util::poll!(&mut queued).is_pending());
        assert_eq!(parser.budget.available_permits(), 0);
        assert_eq!(pool.stats().retained_input_bytes, 1024);
        assert_eq!(pool.stats().queued_jobs, 1);
        drop(queued);
        // A canceled waiter has not dispatched work, so both budgets release now.
        assert_eq!(parser.budget.available_permits(), 512);
        assert_eq!(pool.stats().retained_input_bytes, 512);
        assert_eq!(pool.stats().queued_jobs, 0);
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while pool.stats().retained_input_bytes != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(parser.budget.available_permits(), 1024);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        runtime.shutdown().await;
    }
}
