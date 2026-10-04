use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    task::JoinHandle,
};

use http::{HeaderMap, HeaderName, HeaderValue};
use prost_types::value::Kind;

use crate::{
    app_context::AppContext,
    core::placement::{registry_adapters::PolicyRegistryAdapter, traits::PolicySource},
    routers::prepare::chat_template::process_chat_messages,
};

use super::{core, error::ProcessingError, pb};
use crate::routers::prepare::inference::{InferenceMetadata, InferenceRequest, ParsedInference};

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
        if !matches!(
            route,
            "/v1/chat/completions" | "/v1/completions" | "/generate"
        ) {
            return Err(ProcessingError::new(
                404,
                "unsupported_path",
                "unsupported inference API",
            ));
        }
        if headers
            .get("content-encoding")
            .is_some_and(|v| v != "identity")
        {
            return Err(ProcessingError::new(
                415,
                "unsupported_encoding",
                "decompress requests before ext-proc",
            ));
        }
        if !headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| {
                v.split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .eq_ignore_ascii_case("application/json")
            })
        {
            return Err(ProcessingError::new(
                415,
                "unsupported_content_type",
                "application/json is required",
            ));
        }
        headers.insert(
            "x-request-id",
            HeaderValue::from_str(&id)
                .map_err(|_| ProcessingError::invalid("invalid request ID"))?,
        );
        headers.remove(super::mutation::Mutation::DESTINATION);
        Ok(Self {
            headers,
            path: route.to_owned(),
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

    pub fn needs_tokens(app: &AppContext, model: Option<&str>) -> bool {
        if app.router_config.mode.is_pd_mode() {
            app.policy_registry.get_prefill_policy().needs_tokens()
                || app.policy_registry.get_decode_policy().needs_tokens()
        } else {
            PolicyRegistryAdapter::new(app.policy_registry.clone())
                .regular_policy(model)
                .needs_tokens()
        }
    }

    pub fn parse(
        &self,
        app: &AppContext,
        canceled: &AtomicBool,
    ) -> Result<RoutingInput, ProcessingError> {
        check_canceled(canceled)?;
        let parsed =
            ParsedInference::parse(&self.path, &self.raw).map_err(ProcessingError::invalid)?;
        let metadata = parsed.metadata();
        let model = metadata.model.as_deref();
        let text = &metadata.text;
        check_canceled(canceled)?;
        if model.is_some_and(|model| model.trim().is_empty()) {
            return Err(ProcessingError::invalid("model is required"));
        }
        let tokens = if Self::needs_tokens(app, model) {
            if let Some(ids) = parsed.input_tokens().map_err(ProcessingError::invalid)? {
                Some(ids)
            } else {
                let tokenizer = model
                    .and_then(|model| app.tokenizer_registry.get(model))
                    .ok_or_else(|| {
                        ProcessingError::new(
                            503,
                            "tokenizer_unavailable",
                            "token routing requires input_ids or a model with a registered tokenizer",
                        )
                    })?;
                check_prompt_size(text, app.router_config.ext_proc.max_tokenize_bytes)?;
                check_canceled(canceled)?;
                let prompt = if let Some(chat) = parsed.chat() {
                    process_chat_messages(chat, &*tokenizer)
                        .map_err(ProcessingError::invalid)?
                        .text
                } else {
                    text.clone()
                };
                check_canceled(canceled)?;
                check_prompt_size(&prompt, app.router_config.ext_proc.max_tokenize_bytes)?;
                Some(
                    tokenizer
                        .encode(&prompt, false)
                        .map_err(|e| ProcessingError::invalid(e.to_string()))?
                        .token_ids()
                        .to_vec(),
                )
            }
        } else {
            None
        };
        check_canceled(canceled)?;
        Ok(RoutingInput { metadata, tokens })
    }
}

fn check_canceled(canceled: &AtomicBool) -> Result<(), ProcessingError> {
    if canceled.load(Ordering::Acquire) {
        Err(ProcessingError::new(
            499,
            "parser_canceled",
            "request parsing canceled",
        ))
    } else {
        Ok(())
    }
}

fn check_prompt_size(prompt: &str, limit: usize) -> Result<(), ProcessingError> {
    if prompt.len() > limit {
        Err(ProcessingError::new(
            413,
            "tokenizer_input_too_large",
            "prompt exceeds the synchronous tokenizer byte limit",
        ))
    } else {
        Ok(())
    }
}

/// Running synchronous work retains its slot and buffer budget until it exits.
/// Cancellation stops queued work and is checked between parsing stages; an
/// individual synchronous tokenizer call cannot be preempted.
pub(super) struct RequestParser {
    app: Arc<AppContext>,
    slots: Arc<Semaphore>,
    pub budget: Arc<Semaphore>,
}

struct BlockingTask<T> {
    task: JoinHandle<Result<T, ProcessingError>>,
    canceled: Arc<AtomicBool>,
}
impl<T> Drop for BlockingTask<T> {
    fn drop(&mut self) {
        self.canceled.store(true, Ordering::Release);
        self.task.abort(); // Also prevents a not-yet-started blocking job from running.
    }
}

impl RequestParser {
    pub fn new(app: Arc<AppContext>) -> Self {
        let count = app
            .router_config
            .ext_proc
            .parser_concurrency
            .min(app.router_config.ext_proc.max_streams);
        Self {
            slots: Arc::new(Semaphore::new(count)),
            budget: Arc::new(Semaphore::new(
                app.router_config.ext_proc.max_buffered_bytes,
            )),
            app,
        }
    }

    async fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&AtomicBool) -> Result<T, ProcessingError> + Send + 'static,
    ) -> Result<T, ProcessingError> {
        let permit =
            self.slots.clone().acquire_owned().await.map_err(|_| {
                ProcessingError::new(503, "parser_closed", "request parser is closed")
            })?;
        let canceled = Arc::new(AtomicBool::new(false));
        let cancel = canceled.clone();
        let mut task = BlockingTask {
            canceled,
            task: tokio::task::spawn_blocking(move || {
                let _permit = permit;
                check_canceled(&cancel)?;
                // Log inside the job as well: its caller may already have disconnected.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&cancel)))
                    .unwrap_or_else(|panic| {
                        let detail = panic
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| panic.downcast_ref::<&str>().copied())
                            .unwrap_or("non-string panic");
                        tracing::error!(panic = detail, "request parser panicked");
                        Err(ProcessingError::new(
                            500,
                            "parser_panicked",
                            "request parser panicked",
                        ))
                    })
            }),
        };
        (&mut task.task).await.map_err(|error| {
            tracing::error!(%error, "request parser task failed");
            ProcessingError::new(500, "parser_failed", "request parser task failed")
        })?
    }

    pub async fn parse(
        &self,
        request: RequestEnvelope,
    ) -> Result<(RequestEnvelope, RoutingInput), ProcessingError> {
        let app = self.app.clone();
        self.run(move |canceled| {
            let input = request.parse(&app, canceled)?;
            Ok((request, input))
        })
        .await
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
    use std::time::Duration;

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

    #[tokio::test]
    async fn tokenization_rejects_oversized_text_and_rendered_chat_prompt() {
        let mut config = crate::config::RouterConfig::default();
        config.policy = crate::config::PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        };
        config.ext_proc.max_tokenize_bytes = 4;
        let mut app = AppContext::from_config(config, 5).await.unwrap();
        app.tokenizer_registry =
            crate::routers::test_mocks::tokenizer::tokenizer_registry_with_hf("test-model");
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
            let error = request.parse(&app, &AtomicBool::new(false)).err().unwrap();
            assert_eq!(error.status, 413);
            assert_eq!(error.code, "tokenizer_input_too_large");
            assert_eq!(
                request
                    .parse(&app, &AtomicBool::new(true))
                    .err()
                    .unwrap()
                    .code,
                "parser_canceled"
            );
        }
    }

    #[tokio::test]
    async fn canceled_parser_holds_real_slot_and_buffer_until_job_exits() {
        let mut config = crate::config::RouterConfig::default();
        config.ext_proc.parser_concurrency = 1;
        let parser = Arc::new(RequestParser::new(Arc::new(
            AppContext::from_config(config, 5).await.unwrap(),
        )));
        let budget = Arc::new(Semaphore::new(1024));
        let request = envelope(budget.clone(), 512);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let stages = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let task = tokio::spawn({
            let parser = parser.clone();
            let stages = stages.clone();
            async move {
                parser
                    .run(move |canceled| {
                        let _request = request;
                        started_tx.send(()).unwrap();
                        release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
                        check_canceled(canceled)?;
                        stages.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                    .await
            }
        });
        started_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(parser.slots.available_permits(), 0);
        assert_eq!(budget.available_permits(), 512);

        let queued_request = envelope(budget.clone(), 512);
        let queued_ran = Arc::new(AtomicBool::new(false));
        let mut queued = Box::pin(parser.run({
            let ran = queued_ran.clone();
            move |_| {
                let _request = queued_request;
                ran.store(true, Ordering::SeqCst);
                Ok(())
            }
        }));
        assert!(futures_util::poll!(&mut queued).is_pending());
        drop(queued);
        assert_eq!(budget.available_permits(), 512);
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while parser.slots.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(budget.available_permits(), 1024);
        assert_eq!(stages.load(Ordering::SeqCst), 0);
        assert!(!queued_ran.load(Ordering::SeqCst));
        let error = parser
            .run::<()>(|_| panic!("test parser panic"))
            .await
            .unwrap_err();
        assert_eq!(error.code, "parser_panicked");
        assert_eq!(parser.slots.available_permits(), 1);
    }
}
