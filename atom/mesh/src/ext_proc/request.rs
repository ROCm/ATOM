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
    // Held until the request is dropped.
    input_leases: Vec<InputLease>,
    input_pool: Option<PrepareHandle>,
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
            input_leases: Vec::new(),
            input_pool: None,
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
        let retained: usize = self.input_leases.iter().map(InputLease::bytes).sum();
        let extra = match &self.input_pool {
            Some(pool) if body.len() > retained => Some(pool.retain_input(body.len() - retained)?),
            _ => None,
        };
        self.reserve_buffer(body.len(), limit)?;
        self.input_leases.extend(extra);
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
        mut request: RequestEnvelope,
        decision_deadline: Instant,
    ) -> Result<(RequestEnvelope, RoutingInput), ProcessingError> {
        let deadline = decision_deadline.min(self.pool.prepare_deadline());
        let lease = self.pool.retain_input(request.raw.len())?;
        request.input_leases.push(lease);
        request.input_pool = Some(self.pool.clone());
        let routing = self.routing.clone();
        self.pool
            .try_submit(deadline, move |context| {
                context.check()?;
                let parsed = ParsedInference::parse(&request.path, &request.raw)
                    .map_err(ProcessingError::invalid)?;
                let (metadata, tokens) = routing.prepare(&parsed, context)?;
                let input = RoutingInput {
                    metadata,
                    tokens,
                    state_reference: parsed.requires_state_domain(),
                };
                Ok::<_, ProcessingError>((request, input))
            })?
            .wait()
            .await?
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
    use std::{
        sync::atomic::{AtomicBool, Ordering},
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

    #[test]
    fn body_rewrite_keeps_both_input_budgets_until_drop() {
        let runtime = PreparePoolRuntime::new(crate::core::prepare_pool::PoolConfig {
            max_retained_input_bytes: 800,
            ..Default::default()
        })
        .unwrap();
        let pool = runtime.handle();
        let budget = Arc::new(Semaphore::new(1024));
        let mut request = envelope(budget.clone(), 300);
        request.input_leases.push(pool.retain_input(300).unwrap());
        request.input_pool = Some(pool.clone());

        request.replace_body(vec![b'x'; 700], 1024).unwrap();
        assert_eq!(pool.stats().retained_input_bytes, 700);
        assert_eq!(budget.available_permits(), 0);
        let error = request.replace_body(vec![b'y'; 801], 1024).unwrap_err();
        assert_eq!(error.code, "prepare_input_budget");
        assert_eq!(request.raw, vec![b'x'; 700]);
        assert_eq!(pool.stats().retained_input_bytes, 700);

        request.replace_body(vec![b'z'; 10], 1024).unwrap();
        // A rewrite can retain the old allocation; release conservatively at drop.
        assert_eq!(pool.stats().retained_input_bytes, 700);
        drop(request);
        assert_eq!(pool.stats().retained_input_bytes, 0);
        assert_eq!(budget.available_permits(), 1024);
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
        }
    }

    #[tokio::test]
    async fn canceled_jobs_keep_ext_proc_buffers_until_cleanup() {
        let runtime = PreparePoolRuntime::new(crate::core::prepare_pool::PoolConfig {
            workers: 1,
            ..Default::default()
        })
        .unwrap();
        let pool = runtime.handle();
        let budget = Arc::new(Semaphore::new(1024));
        let mut request = envelope(budget.clone(), 512);
        request.input_leases.push(pool.retain_input(512).unwrap());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let stages = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let completed_stages = stages.clone();
        let running = pool
            .try_submit(pool.prepare_deadline(), move |context| {
                let _request = request;
                started_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
                context.check()?;
                completed_stages.fetch_add(1, Ordering::SeqCst);
                Ok::<_, crate::core::prepare_pool::PrepareError>(())
            })
            .unwrap();
        let task = tokio::spawn(running.wait());
        started_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(pool.stats().busy_workers, 1);
        assert_eq!(pool.stats().retained_input_bytes, 512);
        assert_eq!(budget.available_permits(), 512);

        let mut queued_request = envelope(budget.clone(), 512);
        queued_request
            .input_leases
            .push(pool.retain_input(512).unwrap());
        let queued_ran = Arc::new(AtomicBool::new(false));
        let ran = queued_ran.clone();
        let queued = pool
            .try_submit(pool.prepare_deadline(), move |_| {
                let _request = queued_request;
                ran.store(true, Ordering::SeqCst);
            })
            .unwrap();
        drop(queued);
        // Cancellation retains queued buffers until dequeue.
        assert_eq!(budget.available_permits(), 0);
        assert_eq!(pool.stats().retained_input_bytes, 1024);
        assert_eq!(pool.stats().queued_jobs, 1);
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while pool.stats().retained_input_bytes != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(budget.available_permits(), 1024);
        assert_eq!(stages.load(Ordering::SeqCst), 0);
        assert!(!queued_ran.load(Ordering::SeqCst));
    }
}
