//! ATOM in-process adapter. Token ids are written straight into an LLMEngine
//! and the engine's output callback becomes the token stream.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll},
};

use async_trait::async_trait;
use axum::http::StatusCode;
use dashmap::{mapref::entry::Entry, DashMap};
use futures::Stream;
use pyo3::{
    exceptions::PyValueError,
    types::{PyAnyMethods, PyCFunction, PyDict, PyDictMethods, PyList},
    Bound, Py, PyAny, PyResult, Python,
};
use tokio::sync::mpsc;

use super::{
    backend::{Backend, BackendError, TokenRequest, TokenStream},
    prepare::generation_payload::SamplingParams,
};
use crate::{
    protocols::common::StringOrArray,
    routers::token_handle::{
        engine_error::EngineError,
        token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta},
        token_handle::{TokenHandle, TokenSource},
    },
};

pub type ChunkResult = Result<TokenChunk, EngineError>;

/// What the engine returns for one accepted request.
pub struct Submission {
    pub chunks: mpsc::UnboundedReceiver<ChunkResult>,
    /// Engine-side sequence ids, the handle `release` needs.
    pub seq_ids: Vec<i64>,
}

/// An engine that takes an already-encoded request.
pub trait TokenEngine: Send + Sync + 'static {
    fn submit(&self, request: &TokenRequest) -> Result<Submission, BackendError>;

    /// Called exactly once per accepted request, when its stream ends.
    /// `abort` is set when the engine may still be running the sequences.
    fn release(&self, seq_ids: &[i64], abort: bool);
}

/// Writes `request.token_ids()` into the wrapped engine and streams back what
/// the engine produces. Dropping the stream before it completes aborts the
/// request in the engine, so a render-side stop or a client disconnect frees
/// the engine's KV blocks.
pub struct StandaloneBackend<E> {
    engine: Arc<E>,
    /// Whoever removes a slot releases its sequences.
    inflight: Arc<DashMap<String, Slot>>,
    next_admission: AtomicU64,
}

struct Slot {
    /// Tells this admission apart from a later one that reused the id.
    admission: u64,
    /// `None` while the engine is still admitting the request.
    seq_ids: Option<Vec<i64>>,
}

impl<E: TokenEngine> StandaloneBackend<E> {
    pub fn new(engine: E) -> Self {
        Self {
            engine: Arc::new(engine),
            inflight: Arc::new(DashMap::new()),
            next_admission: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl<E: TokenEngine> Backend for StandaloneBackend<E> {
    async fn generate(&self, request: TokenRequest) -> Result<TokenStream, BackendError> {
        validate(&request)?;
        let request_id = request.payload.request_id.clone();
        let admission = self.next_admission.fetch_add(1, Ordering::Relaxed);
        match self.inflight.entry(request_id.clone()) {
            Entry::Occupied(_) => {
                return Err(BackendError::new(
                    StatusCode::CONFLICT,
                    "duplicate_request_id",
                    format!("request {request_id} is already running"),
                ))
            }
            Entry::Vacant(slot) => {
                slot.insert(Slot {
                    admission,
                    seq_ids: None,
                });
            }
        }
        let admit = Admit {
            engine: self.engine.clone(),
            inflight: self.inflight.clone(),
            request_id,
            admission,
        };
        // The task runs to the end even if this future is dropped; then the
        // stream it returns is dropped with it and aborts the request.
        tokio::task::spawn_blocking(move || admit.run(&request))
            .await
            .unwrap_or_else(|error| {
                Err(BackendError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "engine_submit_failed",
                    error.to_string(),
                ))
            })
    }

    async fn abort(&self, request_id: &str) -> Result<(), BackendError> {
        match self.inflight.remove(request_id) {
            Some((_, slot)) => {
                // A request still being admitted is released by `Admit::run`.
                if let Some(seq_ids) = slot.seq_ids {
                    self.engine.release(&seq_ids, true);
                }
                Ok(())
            }
            None => Err(BackendError::new(
                StatusCode::NOT_FOUND,
                "request_not_found",
                format!("no running request {request_id}"),
            )),
        }
    }
}

/// One admission, owning the reservation until the engine accepts the request.
struct Admit<E: TokenEngine> {
    engine: Arc<E>,
    inflight: Arc<DashMap<String, Slot>>,
    request_id: String,
    admission: u64,
}

impl<E: TokenEngine> Admit<E> {
    fn run(self, request: &TokenRequest) -> Result<TokenStream, BackendError> {
        let submission = self.engine.submit(request)?;
        let committed = match self.inflight.get_mut(&self.request_id) {
            Some(mut slot) if slot.admission == self.admission => {
                slot.seq_ids = Some(submission.seq_ids.clone());
                true
            }
            _ => false,
        };
        if !committed {
            self.engine.release(&submission.seq_ids, true);
            return Err(BackendError::new(
                StatusCode::CONFLICT,
                "request_aborted",
                format!(
                    "request {} was aborted while it was being admitted",
                    self.request_id
                ),
            ));
        }
        Ok(TokenStream::new(TokenHandle::new(EngineStream {
            chunks: submission.chunks,
            seq_ids: submission.seq_ids,
            request_id: self.request_id.clone(),
            admission: self.admission,
            engine: self.engine.clone(),
            inflight: self.inflight.clone(),
            finished: false,
        })))
    }
}

impl<E: TokenEngine> Drop for Admit<E> {
    /// Frees the id when admission failed or panicked.
    fn drop(&mut self) {
        self.inflight.remove_if(&self.request_id, |_, slot| {
            slot.admission == self.admission && slot.seq_ids.is_none()
        });
    }
}

struct EngineStream<E: TokenEngine> {
    chunks: mpsc::UnboundedReceiver<ChunkResult>,
    seq_ids: Vec<i64>,
    request_id: String,
    admission: u64,
    engine: Arc<E>,
    inflight: Arc<DashMap<String, Slot>>,
    finished: bool,
}

impl<E: TokenEngine> EngineStream<E> {
    /// Releases the request unless an explicit `abort` already did.
    fn end(&mut self, abort: bool) {
        self.finished = true;
        let admission = self.admission;
        if self
            .inflight
            .remove_if(&self.request_id, |_, slot| slot.admission == admission)
            .is_some()
        {
            self.engine.release(&self.seq_ids, abort);
        }
    }
}

impl<E: TokenEngine> Stream for EngineStream<E> {
    type Item = ChunkResult;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        // ATOM's output thread can hold the last callback, and with it the
        // sender, until its next output; the stream cannot wait for a close.
        if this.finished {
            return Poll::Ready(None);
        }
        match this.chunks.poll_recv(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if matches!(chunk, TokenChunk::Complete { .. }) {
                    this.end(false);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(error))) => {
                this.end(true);
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                this.end(true);
                Poll::Ready(Some(Err(EngineError::DecodeIncomplete)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<E: TokenEngine> TokenSource for EngineStream<E> {
    fn mark_completed(&mut self) {
        if !self.finished {
            self.end(false);
        }
    }
}

impl<E: TokenEngine> Drop for EngineStream<E> {
    fn drop(&mut self) {
        if !self.finished {
            self.end(true);
        }
    }
}

/// ATOM's `SamplingParams` has no field for these. Refusing them is better
/// than serving a request whose options were dropped.
fn validate(request: &TokenRequest) -> Result<(), BackendError> {
    let payload = &request.payload;
    let sampling = &payload.sampling;
    let unsupported = |name: &str| {
        Err(BackendError::new(
            StatusCode::BAD_REQUEST,
            "unsupported_sampling_param",
            format!("the ATOM engine does not support `{name}`"),
        ))
    };
    if payload.token_ids.is_empty() {
        return Err(BackendError::new(
            StatusCode::BAD_REQUEST,
            "empty_prompt",
            "the prompt encoded to no tokens",
        ));
    }
    if sampling.n != 1 {
        return unsupported("n");
    }
    if sampling.min_p != 0.0 {
        return unsupported("min_p");
    }
    if sampling.frequency_penalty != 0.0 {
        return unsupported("frequency_penalty");
    }
    if sampling.presence_penalty != 0.0 {
        return unsupported("presence_penalty");
    }
    if sampling.repetition_penalty != 1.0 {
        return unsupported("repetition_penalty");
    }
    if sampling.min_new_tokens != 0 {
        return unsupported("min_tokens");
    }
    if payload
        .stop
        .stop_token_ids
        .as_ref()
        .is_some_and(|ids| !ids.is_empty())
    {
        return unsupported("stop_token_ids");
    }
    if payload.logprob.return_logprob {
        return unsupported("logprobs");
    }
    if payload.tool_constraints.is_some() {
        return unsupported("tool_choice constraints");
    }
    if payload.pd_metadata.is_some() {
        return unsupported("prefill/decode bootstrap");
    }
    if payload.return_hidden_states {
        return unsupported("return_hidden_states");
    }
    Ok(())
}

/// PyO3 handle for `atom.model_engine.llm_engine.LLMEngine`.
///
/// It does what the engine's own server path does: `io_processor
/// .preprocess_fanout` builds the sequence from the token ids and a stream
/// callback, and `core_mgr.add_request` queues it. On the server path nothing
/// else removes a sequence from `io_processor.requests`, so `release` pops it
/// on every exit, after `core_mgr.abort_request` when the request is cut short.
pub struct PyLlmEngine {
    engine: Arc<Py<PyAny>>,
    sampling_params: Py<PyAny>,
    default_max_tokens: u32,
}

impl PyLlmEngine {
    /// Binds `engine` to ATOM's `SamplingParams` and to the output cap its
    /// OpenAI server applies when a request sets none.
    pub fn new(py: Python<'_>, engine: Py<PyAny>) -> PyResult<Self> {
        let sampling_params = py
            .import("atom.sampling_params")?
            .getattr("SamplingParams")?
            .unbind();
        let default_max_tokens = py
            .import("atom.entrypoints.openai.protocol")?
            .getattr("DEFAULT_MAX_TOKENS")?
            .extract()?;
        Ok(Self::with_sampling_params(
            engine,
            sampling_params,
            default_max_tokens,
        ))
    }

    fn with_sampling_params(
        engine: Py<PyAny>,
        sampling_params: Py<PyAny>,
        default_max_tokens: u32,
    ) -> Self {
        Self {
            engine: Arc::new(engine),
            sampling_params,
            default_max_tokens,
        }
    }
}

impl TokenEngine for PyLlmEngine {
    fn submit(&self, request: &TokenRequest) -> Result<Submission, BackendError> {
        Python::attach(|py| submit(py, self, request)).map_err(|error| {
            let status = Python::attach(|py| {
                if error.is_instance_of::<PyValueError>(py) {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            });
            BackendError::new(status, "engine_submit_failed", error.to_string())
        })
    }

    /// Runs off the async workers when there is a runtime: the engine's
    /// output threads hold the GIL for whole steps.
    fn release(&self, seq_ids: &[i64], abort: bool) {
        let engine = self.engine.clone();
        let seq_ids = seq_ids.to_vec();
        let work = move || Python::attach(|py| release(engine.bind(py), &seq_ids, abort));
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => drop(runtime.spawn_blocking(work)),
            Err(_) => work(),
        }
    }
}

fn release(engine: &Bound<'_, PyAny>, seq_ids: &[i64], abort: bool) {
    if abort {
        let aborted = engine.getattr("core_mgr").and_then(|core| {
            seq_ids
                .iter()
                .try_for_each(|id| core.call_method1("abort_request", (*id,)).map(drop))
        });
        if let Err(error) = aborted {
            tracing::warn!("ATOM abort_request failed: {error}");
        }
    }
    let popped = engine
        .getattr("io_processor")
        .and_then(|io| io.getattr("requests"))
        .and_then(|requests| {
            seq_ids.iter().try_for_each(|id| {
                requests
                    .call_method1("pop", (*id, engine.py().None()))
                    .map(drop)
            })
        });
    if let Err(error) = popped {
        tracing::warn!("ATOM io_processor.requests cleanup failed: {error}");
    }
}

/// Folds the engine's per-step outputs into chunks. A streaming request gets
/// each step as a `Partial`; a non-streaming one gets only the final
/// `Complete`, which the aggregator reads the whole output from.
struct Collector {
    stream: bool,
    request_id: String,
    prompt_tokens: u32,
    tokens: Vec<u32>,
    tx: mpsc::UnboundedSender<ChunkResult>,
}

impl Collector {
    fn on_output(
        &mut self,
        tokens: Vec<u32>,
        finished: bool,
        finish_reason: Option<String>,
        cached_tokens: u32,
    ) {
        if !finished {
            if tokens.is_empty() {
                return;
            }
            self.tokens.extend_from_slice(&tokens);
            if self.stream {
                let _ = self.tx.send(Ok(TokenChunk::Partial {
                    token_ids: tokens,
                    logprobs: None,
                }));
            }
            return;
        }
        if self.stream && !tokens.is_empty() {
            let _ = self.tx.send(Ok(TokenChunk::Partial {
                token_ids: tokens.clone(),
                logprobs: None,
            }));
        }
        self.tokens.extend_from_slice(&tokens);
        let completion = self.tokens.len() as u32;
        let _ = self.tx.send(Ok(TokenChunk::Complete {
            token_ids: std::mem::take(&mut self.tokens),
            finish_reason: finish_reason_of(finish_reason.as_deref()),
            matched_stop: None,
            usage: Usage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens: completion,
                total_tokens: self.prompt_tokens + completion,
            },
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta {
                request_id: self.request_id.clone(),
                weight_version: None,
                cached_tokens,
            },
        }));
    }
}

/// ATOM's leave reasons are `eos`, `stop_sequence`, `stop_<token_id>`,
/// `max_tokens`, `aborted`, and free text such as `unschedulable: ...`.
fn finish_reason_of(reason: Option<&str>) -> FinishReason {
    match reason {
        None | Some("eos" | "stop" | "stop_sequence") => FinishReason::Stop,
        Some("max_tokens" | "max_new_tokens" | "length") => FinishReason::Length,
        Some("aborted" | "abort") => FinishReason::Abort,
        Some(reason) if reason.starts_with("stop_") => FinishReason::Stop,
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

fn submit(py: Python<'_>, sink: &PyLlmEngine, request: &TokenRequest) -> PyResult<Submission> {
    let (tx, rx) = mpsc::unbounded_channel();
    let collector = Mutex::new(Collector {
        stream: request.payload.stream,
        request_id: request.payload.request_id.clone(),
        prompt_tokens: request.token_ids().len() as u32,
        tokens: Vec::new(),
        tx,
    });
    let callback = PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<()> {
        let output = args.get_item(0)?;
        let tokens: Vec<u32> = output.getattr("output_tokens")?.extract()?;
        let finished: bool = output.getattr("finished")?.extract()?;
        let reason: Option<String> = output.getattr("finish_reason")?.extract()?;
        let cached: u32 = output
            .getattr("num_cached_tokens")
            .and_then(|value| value.extract())
            .unwrap_or(0);
        collector
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .on_output(tokens, finished, reason, cached);
        Ok(())
    })?;

    let engine = sink.engine.bind(py);
    let ids = PyList::new(
        py,
        request
            .token_ids()
            .iter()
            .map(|id| i64::from(*id))
            .collect::<Vec<_>>(),
    )?;
    let sampling = sampling_params(
        sink.sampling_params.bind(py),
        &request.payload,
        sink.default_max_tokens,
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("stream_callback", callback)?;
    let io_processor = engine.getattr("io_processor")?;
    let seqs = io_processor.call_method("preprocess_fanout", (ids, sampling), Some(&kwargs))?;
    let seq_ids: Vec<i64> = seqs
        .try_iter()?
        .map(|seq| seq?.getattr("id")?.extract())
        .collect::<PyResult<_>>()?;
    let default_cap = request.payload.sampling.max_new_tokens.is_none();
    let admitted = fit_context(engine, &seqs, default_cap).and_then(|()| {
        engine
            .getattr("core_mgr")?
            .call_method1("add_request", (&seqs,))
            .map(drop)
    });
    if let Err(error) = admitted {
        release(engine, &seq_ids, false);
        return Err(error);
    }
    Ok(Submission {
        chunks: rx,
        seq_ids,
    })
}

/// The same checks ATOM's server runs after `preprocess`: a default output
/// cap shrinks to the context the prompt leaves, an explicit one has to fit,
/// and the prompt has to fit an empty KV pool.
fn fit_context(
    engine: &Bound<'_, PyAny>,
    seqs: &Bound<'_, PyAny>,
    default_cap: bool,
) -> PyResult<()> {
    let config = engine
        .getattr("config")
        .or_else(|_| engine.getattr("io_processor")?.getattr("config"));
    let limit = optional_int(config.and_then(|config| config.getattr("max_model_len")))?;
    let pool = optional_int(
        engine
            .getattr("core_mgr")
            .and_then(|core| core.getattr("max_pool_tokens")),
    )?;
    for seq in seqs.try_iter()? {
        let seq = seq?;
        let prompt: i64 = seq.getattr("num_prompt_tokens")?.extract()?;
        if let Some(limit) = limit {
            let mut max_tokens: i64 = seq.getattr("max_tokens")?.extract()?;
            if default_cap {
                let remaining = limit - prompt;
                if remaining <= 0 {
                    return Err(PyValueError::new_err(
                        "The prompt leaves no context space for output tokens",
                    ));
                }
                if max_tokens > remaining {
                    max_tokens = remaining;
                    seq.setattr("max_tokens", max_tokens)?;
                }
            }
            if prompt + max_tokens > limit {
                return Err(PyValueError::new_err(format!(
                    "This model's maximum context length is {limit} tokens. However, you \
                     requested {max_tokens} output tokens and your prompt contains {prompt} \
                     input tokens"
                )));
            }
        }
        if let Some(pool) = pool.filter(|pool| prompt > *pool) {
            return Err(PyValueError::new_err(format!(
                "This server's KV cache holds at most {pool} tokens for a single request, \
                 and your prompt contains {prompt} input tokens"
            )));
        }
    }
    Ok(())
}

/// A missing attribute or a Python `None` both mean the engine sets no limit.
fn optional_int(value: PyResult<Bound<'_, PyAny>>) -> PyResult<Option<i64>> {
    match value {
        Ok(value) => value.extract(),
        Err(_) => Ok(None),
    }
}

fn sampling_params<'py>(
    class: &Bound<'py, PyAny>,
    payload: &super::prepare::generation_payload::GenerationPayload,
    default_max_tokens: u32,
) -> PyResult<Bound<'py, PyAny>> {
    let sampling: &SamplingParams = &payload.sampling;
    let kwargs = PyDict::new(class.py());
    kwargs.set_item("temperature", sampling.temperature)?;
    kwargs.set_item("top_p", sampling.top_p)?;
    kwargs.set_item("top_k", sampling.top_k)?;
    kwargs.set_item("ignore_eos", sampling.ignore_eos)?;
    match sampling.max_new_tokens {
        Some(max_tokens) => kwargs.set_item("max_tokens", max_tokens)?,
        None => {
            kwargs.set_item("max_tokens", default_max_tokens)?;
            kwargs.set_item("max_tokens_is_default", true)?;
        }
    }
    if let Some(stop) = &payload.stop.stop {
        let stops = match stop {
            StringOrArray::String(stop) => vec![stop.clone()],
            StringOrArray::Array(stops) => stops.clone(),
        };
        kwargs.set_item("stop_strings", stops)?;
    }
    class.call((), Some(&kwargs))
}

#[cfg(test)]
mod tests {
    use std::sync::{mpsc as std_mpsc, Mutex as StdMutex};
    use std::time::Duration;

    use futures::StreamExt;

    use super::*;
    use crate::entrypoints::backend::test_request;

    /// Scripted engine: every submit hands back a channel the test feeds and
    /// a fresh pair of sequence ids.
    struct Scripted {
        senders: StdMutex<Vec<mpsc::UnboundedSender<ChunkResult>>>,
        submitted: StdMutex<Vec<Vec<u32>>>,
        released: StdMutex<Vec<(Vec<i64>, bool)>>,
        next_id: StdMutex<i64>,
        gate: StdMutex<Option<std_mpsc::Receiver<()>>>,
    }

    impl Scripted {
        fn new() -> Self {
            Self {
                senders: StdMutex::new(Vec::new()),
                submitted: StdMutex::new(Vec::new()),
                released: StdMutex::new(Vec::new()),
                next_id: StdMutex::new(40),
                gate: StdMutex::new(None),
            }
        }

        /// The next submit blocks until the returned sender fires.
        fn gated() -> (Self, std_mpsc::Sender<()>) {
            let (open, gate) = std_mpsc::channel();
            let engine = Self::new();
            *engine.gate.lock().unwrap() = Some(gate);
            (engine, open)
        }

        fn sender(&self, index: usize) -> mpsc::UnboundedSender<ChunkResult> {
            self.senders.lock().unwrap()[index].clone()
        }

        fn released(&self) -> Vec<(Vec<i64>, bool)> {
            self.released.lock().unwrap().clone()
        }
    }

    impl TokenEngine for Scripted {
        fn submit(&self, request: &TokenRequest) -> Result<Submission, BackendError> {
            self.submitted
                .lock()
                .unwrap()
                .push(request.token_ids().to_vec());
            let gate = self.gate.lock().unwrap().take();
            if let Some(gate) = gate {
                gate.recv().unwrap();
            }
            let (tx, rx) = mpsc::unbounded_channel();
            self.senders.lock().unwrap().push(tx);
            let mut next = self.next_id.lock().unwrap();
            *next += 2;
            Ok(Submission {
                chunks: rx,
                seq_ids: vec![*next, *next + 1],
            })
        }

        fn release(&self, seq_ids: &[i64], abort: bool) {
            self.released
                .lock()
                .unwrap()
                .push((seq_ids.to_vec(), abort));
        }
    }

    fn complete() -> TokenChunk {
        TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage::default(),
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta::default(),
        }
    }

    async fn until(mut ready: impl FnMut() -> bool) {
        for _ in 0..500 {
            if ready() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        panic!("condition not reached");
    }

    #[tokio::test]
    async fn the_stream_ends_at_complete_and_releases_without_abort() {
        let backend = StandaloneBackend::new(Scripted::new());
        let stream = backend.generate(test_request(vec![9, 8, 7])).await.unwrap();
        let (mut handle, _) = stream.into_parts();
        assert_eq!(
            *backend.engine.submitted.lock().unwrap(),
            vec![vec![9, 8, 7]]
        );
        let tx = backend.engine.sender(0);
        tx.send(Ok(TokenChunk::Partial {
            token_ids: vec![5],
            logprobs: None,
        }))
        .unwrap();
        tx.send(Ok(complete())).unwrap();
        assert!(matches!(
            handle.next().await,
            Some(Ok(TokenChunk::Partial { .. }))
        ));
        assert!(matches!(
            handle.next().await,
            Some(Ok(TokenChunk::Complete { .. }))
        ));
        // The sender is still alive, as it is while ATOM holds the callback.
        let end = tokio::time::timeout(Duration::from_secs(1), handle.next())
            .await
            .expect("the stream must end at Complete");
        assert!(end.is_none());
        handle.mark_completed();
        drop(handle);
        assert_eq!(backend.engine.released(), vec![(vec![42, 43], false)]);
        assert!(backend.inflight.is_empty());
    }

    #[tokio::test]
    async fn dropping_an_unfinished_stream_aborts_the_sequences() {
        let backend = StandaloneBackend::new(Scripted::new());
        let stream = backend.generate(test_request(vec![1])).await.unwrap();
        assert_eq!(backend.inflight.len(), 1);
        drop(stream);
        assert_eq!(backend.engine.released(), vec![(vec![42, 43], true)]);
        assert!(backend.inflight.is_empty());
    }

    #[tokio::test]
    async fn explicit_abort_releases_once_even_after_the_stream_drops() {
        let backend = StandaloneBackend::new(Scripted::new());
        let stream = backend.generate(test_request(vec![1])).await.unwrap();
        backend.abort("req-1").await.unwrap();
        let error = backend.abort("req-1").await.unwrap_err();
        assert_eq!(error.status, StatusCode::NOT_FOUND);
        drop(stream);
        assert_eq!(backend.engine.released(), vec![(vec![42, 43], true)]);
    }

    #[tokio::test]
    async fn a_reused_id_is_not_released_by_the_stream_it_replaced() {
        let backend = StandaloneBackend::new(Scripted::new());
        let first = backend.generate(test_request(vec![1])).await.unwrap();
        backend.abort("req-1").await.unwrap();
        let second = backend.generate(test_request(vec![2])).await.unwrap();
        drop(first);
        assert_eq!(
            backend.inflight.get("req-1").unwrap().seq_ids,
            Some(vec![44, 45])
        );
        backend.abort("req-1").await.unwrap();
        drop(second);
        assert_eq!(
            backend.engine.released(),
            vec![(vec![42, 43], true), (vec![44, 45], true)]
        );
    }

    #[tokio::test]
    async fn a_closed_engine_channel_is_an_incomplete_stream() {
        let backend = StandaloneBackend::new(Scripted::new());
        let stream = backend.generate(test_request(vec![1])).await.unwrap();
        let (mut handle, _) = stream.into_parts();
        backend.engine.senders.lock().unwrap().clear();
        assert!(matches!(
            handle.next().await,
            Some(Err(EngineError::DecodeIncomplete))
        ));
        assert!(handle.next().await.is_none());
        drop(handle);
        assert_eq!(backend.engine.released(), vec![(vec![42, 43], true)]);
    }

    #[tokio::test]
    async fn a_duplicate_request_id_is_rejected_until_the_first_ends() {
        let backend = StandaloneBackend::new(Scripted::new());
        let first = backend.generate(test_request(vec![1])).await.unwrap();
        let error = backend
            .generate(test_request(vec![2]))
            .await
            .err()
            .expect("the same id is already running");
        assert_eq!(error.status, StatusCode::CONFLICT);
        assert_eq!(error.code, "duplicate_request_id");
        drop(first);
        assert!(backend.generate(test_request(vec![3])).await.is_ok());
    }

    #[tokio::test]
    async fn the_id_is_reserved_while_the_engine_admits_the_request() {
        let (engine, open) = Scripted::gated();
        let backend = Arc::new(StandaloneBackend::new(engine));
        let first = tokio::spawn({
            let backend = backend.clone();
            async move { backend.generate(test_request(vec![1])).await }
        });
        until(|| !backend.engine.submitted.lock().unwrap().is_empty()).await;
        let error = backend
            .generate(test_request(vec![2]))
            .await
            .err()
            .expect("the first request holds the id");
        assert_eq!(error.code, "duplicate_request_id");
        open.send(()).unwrap();
        assert!(first.await.unwrap().is_ok());
        assert_eq!(backend.engine.submitted.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn an_abort_during_admission_releases_the_admitted_sequences() {
        let (engine, open) = Scripted::gated();
        let backend = Arc::new(StandaloneBackend::new(engine));
        let first = tokio::spawn({
            let backend = backend.clone();
            async move { backend.generate(test_request(vec![1])).await }
        });
        until(|| !backend.engine.submitted.lock().unwrap().is_empty()).await;
        backend.abort("req-1").await.unwrap();
        assert!(backend.engine.released().is_empty());
        open.send(()).unwrap();
        let error = first
            .await
            .unwrap()
            .err()
            .expect("aborted before admission");
        assert_eq!(error.code, "request_aborted");
        assert_eq!(backend.engine.released(), vec![(vec![42, 43], true)]);
        assert!(backend.inflight.is_empty());
    }

    #[tokio::test]
    async fn a_stale_admission_does_not_take_a_newer_reservation() {
        let (engine, open) = Scripted::gated();
        let backend = Arc::new(StandaloneBackend::new(engine));
        let first = tokio::spawn({
            let backend = backend.clone();
            async move { backend.generate(test_request(vec![1])).await }
        });
        until(|| !backend.engine.submitted.lock().unwrap().is_empty()).await;
        backend.abort("req-1").await.unwrap();
        let second = backend.generate(test_request(vec![2])).await.unwrap();
        open.send(()).unwrap();
        let error = first.await.unwrap().err().expect("the first was aborted");
        assert_eq!(error.code, "request_aborted");
        assert_eq!(backend.engine.released(), vec![(vec![44, 45], true)]);
        assert_eq!(
            backend.inflight.get("req-1").unwrap().seq_ids,
            Some(vec![42, 43])
        );
        drop(second);
        assert_eq!(
            backend.engine.released(),
            vec![(vec![44, 45], true), (vec![42, 43], true)]
        );
    }

    #[tokio::test]
    async fn a_caller_gone_during_admission_aborts_and_frees_the_id() {
        let (engine, open) = Scripted::gated();
        let backend = Arc::new(StandaloneBackend::new(engine));
        let caller = tokio::spawn({
            let backend = backend.clone();
            async move { backend.generate(test_request(vec![1])).await }
        });
        until(|| !backend.engine.submitted.lock().unwrap().is_empty()).await;
        caller.abort();
        assert!(caller.await.err().is_some_and(|error| error.is_cancelled()));
        open.send(()).unwrap();
        until(|| !backend.engine.released().is_empty()).await;
        assert_eq!(backend.engine.released(), vec![(vec![42, 43], true)]);
        until(|| backend.inflight.is_empty()).await;
        assert!(backend.generate(test_request(vec![2])).await.is_ok());
    }

    #[tokio::test]
    async fn options_the_engine_cannot_honor_are_refused_before_submit() {
        let backend = StandaloneBackend::new(Scripted::new());
        let cases: Vec<(&str, fn(&mut TokenRequest))> = vec![
            ("n", |r| r.payload.sampling.n = 2),
            ("min_p", |r| r.payload.sampling.min_p = 0.1),
            ("frequency_penalty", |r| {
                r.payload.sampling.frequency_penalty = 0.5
            }),
            ("presence_penalty", |r| {
                r.payload.sampling.presence_penalty = 0.5
            }),
            ("repetition_penalty", |r| {
                r.payload.sampling.repetition_penalty = 1.1
            }),
            ("min_tokens", |r| r.payload.sampling.min_new_tokens = 2),
            ("stop_token_ids", |r| {
                r.payload.stop.stop_token_ids = Some(vec![2])
            }),
            ("logprobs", |r| r.payload.logprob.return_logprob = true),
            ("tool_choice constraints", |r| {
                r.payload.tool_constraints = Some(("a".into(), "b".into()))
            }),
            ("return_hidden_states", |r| {
                r.payload.return_hidden_states = true
            }),
        ];
        for (name, mutate) in cases {
            let mut request = test_request(vec![1]);
            mutate(&mut request);
            let error = backend.generate(request).await.err().expect(name);
            assert_eq!(error.status, StatusCode::BAD_REQUEST, "{name}");
            assert_eq!(error.code, "unsupported_sampling_param", "{name}");
            assert!(error.message.contains(name), "{name}: {}", error.message);
        }
        let error = backend.generate(test_request(vec![])).await.err().unwrap();
        assert_eq!(error.code, "empty_prompt");
        assert!(backend.engine.submitted.lock().unwrap().is_empty());
        assert!(backend.inflight.is_empty());
    }

    fn collect(stream: bool) -> (Collector, mpsc::UnboundedReceiver<ChunkResult>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Collector {
                stream,
                request_id: "req-1".to_string(),
                prompt_tokens: 3,
                tokens: Vec::new(),
                tx,
            },
            rx,
        )
    }

    #[test]
    fn a_streaming_request_gets_partials_then_a_complete_with_usage() {
        let (mut collector, mut rx) = collect(true);
        collector.on_output(vec![10, 11], false, None, 0);
        collector.on_output(vec![], false, None, 0);
        collector.on_output(vec![12], true, Some("max_tokens".into()), 2);
        match rx.try_recv().unwrap().unwrap() {
            TokenChunk::Partial { token_ids, .. } => assert_eq!(token_ids, vec![10, 11]),
            other => panic!("{other:?}"),
        }
        match rx.try_recv().unwrap().unwrap() {
            TokenChunk::Partial { token_ids, .. } => assert_eq!(token_ids, vec![12]),
            other => panic!("{other:?}"),
        }
        match rx.try_recv().unwrap().unwrap() {
            TokenChunk::Complete {
                token_ids,
                finish_reason,
                usage,
                meta,
                ..
            } => {
                assert_eq!(token_ids, vec![10, 11, 12]);
                assert!(matches!(finish_reason, FinishReason::Length));
                assert_eq!(
                    (
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens
                    ),
                    (3, 3, 6)
                );
                assert_eq!(meta.cached_tokens, 2);
                assert_eq!(meta.request_id, "req-1");
            }
            other => panic!("{other:?}"),
        }
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn a_non_streaming_request_gets_one_complete_with_every_token() {
        let (mut collector, mut rx) = collect(false);
        collector.on_output(vec![10], false, None, 0);
        collector.on_output(vec![11], true, Some("eos".into()), 0);
        match rx.try_recv().unwrap().unwrap() {
            TokenChunk::Complete {
                token_ids,
                finish_reason,
                ..
            } => {
                assert_eq!(token_ids, vec![10, 11]);
                assert!(matches!(finish_reason, FinishReason::Stop));
            }
            other => panic!("{other:?}"),
        }
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn atom_finish_reasons_map_to_the_render_enum() {
        for (reason, expected) in [
            (Some("max_tokens"), "length"),
            (Some("eos"), "stop"),
            (Some("stop_sequence"), "stop"),
            (Some("stop_151645"), "stop"),
            (Some("aborted"), "abort"),
            (Some("abort"), "abort"),
            (None, "stop"),
        ] {
            let got = match finish_reason_of(reason) {
                FinishReason::Length => "length",
                FinishReason::Stop => "stop",
                FinishReason::Abort => "abort",
                other => panic!("{reason:?} -> {other:?}"),
            };
            assert_eq!(got, expected, "{reason:?}");
        }
        for reason in [
            "logprob_error",
            "unschedulable: needs 9 KV blocks. Reduce prompt length",
        ] {
            match finish_reason_of(Some(reason)) {
                FinishReason::Other(got) => assert_eq!(got, reason),
                other => panic!("{reason} -> {other:?}"),
            }
        }
    }

    /// The parts of ATOM's engine, `Sequence`, and `SamplingParams` the
    /// adapter touches, with the same names and defaults.
    const FAKE_ENGINE: &std::ffi::CStr = c"
from types import SimpleNamespace

class SamplingParams:
    def __init__(self, max_tokens=64, max_tokens_is_default=False, **kwargs):
        self.max_tokens = max_tokens
        self.max_tokens_is_default = max_tokens_is_default
        self.kwargs = kwargs

class Seq:
    def __init__(self, id, prompt, sampling):
        self.id = id
        self.num_prompt_tokens = len(prompt)
        self.max_tokens = sampling.max_tokens

class IoProcessor:
    def __init__(self):
        self.requests = {}
        self.calls = []

    def preprocess_fanout(self, prompt, sampling, **kwargs):
        self.calls.append((list(prompt), sampling, sorted(kwargs)))
        self.callback = kwargs['stream_callback']
        seq = Seq(77, prompt, sampling)
        self.requests[77] = seq
        return [seq]

class CoreMgr:
    def __init__(self, fail=False, max_pool_tokens=None):
        self.added = []
        self.aborted = []
        self.fail = fail
        self.max_pool_tokens = max_pool_tokens

    def add_request(self, seqs):
        if self.fail:
            raise RuntimeError('queue full')
        self.added.append([s.id for s in seqs])

    def abort_request(self, seq_id):
        self.aborted.append(seq_id)

class Engine:
    def __init__(self, fail=False, max_model_len=64, max_pool_tokens=None):
        self.config = SimpleNamespace(max_model_len=max_model_len)
        self.io_processor = IoProcessor()
        self.core_mgr = CoreMgr(fail, max_pool_tokens)

def out(tokens, finished, reason=None, cached=0):
    return SimpleNamespace(output_tokens=tokens, finished=finished, finish_reason=reason, num_cached_tokens=cached)
";

    struct Fake<'py> {
        globals: Bound<'py, PyDict>,
        engine: Bound<'py, PyAny>,
    }

    impl<'py> Fake<'py> {
        fn new(py: Python<'py>, fail: bool, max_model_len: i64, pool: Option<i64>) -> Self {
            let globals = PyDict::new(py);
            py.run(FAKE_ENGINE, Some(&globals), None).unwrap();
            let engine = globals
                .get_item("Engine")
                .unwrap()
                .unwrap()
                .call1((fail, max_model_len, pool))
                .unwrap();
            Self { globals, engine }
        }

        fn sink(&self) -> PyLlmEngine {
            let class = self.globals.get_item("SamplingParams").unwrap().unwrap();
            PyLlmEngine::with_sampling_params(self.engine.clone().unbind(), class.unbind(), 8192)
        }

        fn get(&self, path: &str) -> Bound<'py, PyAny> {
            path.split('.').fold(self.engine.clone(), |object, name| {
                object.getattr(name).unwrap()
            })
        }

        fn requests(&self) -> usize {
            self.get("io_processor.requests").len().unwrap()
        }
    }

    #[test]
    fn py_engine_relays_callbacks_and_pops_the_sequence_on_release() {
        Python::attach(|py| {
            let fake = Fake::new(py, false, 64, None);
            let sink = fake.sink();
            let mut request = test_request(vec![7, 8, 9]);
            request.payload.stream = true;
            request.payload.stop.stop = Some(StringOrArray::String("END".into()));
            let mut submission = sink.submit(&request).unwrap();
            assert_eq!(submission.seq_ids, vec![77]);

            let calls: Vec<(Vec<i64>, Py<PyAny>, Vec<String>)> =
                fake.get("io_processor.calls").extract().unwrap();
            assert_eq!(calls[0].0, vec![7, 8, 9]);
            // No `parent_request_id`: with n == 1 it becomes an external id
            // whose maps only the offline `postprocess` clears.
            assert_eq!(calls[0].2, vec!["stream_callback".to_string()]);
            let added: Vec<Vec<i64>> = fake.get("core_mgr.added").extract().unwrap();
            assert_eq!(added, vec![vec![77]]);
            let stops: Vec<String> = fake
                .get("io_processor.calls")
                .get_item(0)
                .unwrap()
                .get_item(1)
                .unwrap()
                .getattr("kwargs")
                .unwrap()
                .get_item("stop_strings")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(stops, vec!["END".to_string()]);

            let out = fake.globals.get_item("out").unwrap().unwrap();
            let callback = fake.get("io_processor.callback");
            callback
                .call1((out.call1((vec![5u32], false)).unwrap(),))
                .unwrap();
            callback
                .call1((out.call1((Vec::<u32>::new(), true, "max_tokens")).unwrap(),))
                .unwrap();
            match submission.chunks.try_recv().unwrap().unwrap() {
                TokenChunk::Partial { token_ids, .. } => assert_eq!(token_ids, vec![5]),
                other => panic!("{other:?}"),
            }
            match submission.chunks.try_recv().unwrap().unwrap() {
                TokenChunk::Complete {
                    token_ids,
                    finish_reason,
                    ..
                } => {
                    assert_eq!(token_ids, vec![5]);
                    assert!(matches!(finish_reason, FinishReason::Length));
                }
                other => panic!("{other:?}"),
            }

            sink.release(&[77], false);
            assert_eq!(fake.requests(), 0);
            let aborted: Vec<i64> = fake.get("core_mgr.aborted").extract().unwrap();
            assert!(aborted.is_empty());
        });
    }

    #[test]
    fn py_engine_abort_stops_the_sequence_and_pops_it() {
        Python::attach(|py| {
            let fake = Fake::new(py, false, 64, None);
            let sink = fake.sink();
            sink.submit(&test_request(vec![1])).unwrap();
            assert_eq!(fake.requests(), 1);
            sink.release(&[77], true);
            let aborted: Vec<i64> = fake.get("core_mgr.aborted").extract().unwrap();
            assert_eq!(aborted, vec![77]);
            assert_eq!(fake.requests(), 0);
        });
    }

    #[test]
    fn py_engine_default_output_cap_fits_the_context() {
        Python::attach(|py| {
            let fake = Fake::new(py, false, 16, None);
            let sink = fake.sink();
            let mut request = test_request(vec![1, 2, 3]);
            request.payload.sampling.max_new_tokens = None;
            sink.submit(&request).unwrap();
            let sampling = fake
                .get("io_processor.calls")
                .get_item(0)
                .unwrap()
                .get_item(1)
                .unwrap();
            let asked: u32 = sampling.getattr("max_tokens").unwrap().extract().unwrap();
            let is_default: bool = sampling
                .getattr("max_tokens_is_default")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!((asked, is_default), (8192, true));
            let seq = fake.get("io_processor.requests").get_item(77).unwrap();
            let max_tokens: i64 = seq.getattr("max_tokens").unwrap().extract().unwrap();
            assert_eq!(max_tokens, 13);
        });
    }

    #[test]
    fn py_engine_rejects_an_explicit_cap_past_the_context() {
        Python::attach(|py| {
            let fake = Fake::new(py, false, 8, None);
            let sink = fake.sink();
            let mut request = test_request(vec![1, 2, 3]);
            request.payload.sampling.max_new_tokens = Some(6);
            let error = sink.submit(&request).err().unwrap();
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert!(error.message.contains("maximum context length is 8"));
            assert_eq!(fake.requests(), 0);
            let added: Vec<Vec<i64>> = fake.get("core_mgr.added").extract().unwrap();
            assert!(added.is_empty());
        });
    }

    #[test]
    fn py_engine_failed_admission_drops_the_registered_sequence() {
        Python::attach(|py| {
            let fake = Fake::new(py, true, 64, None);
            let sink = fake.sink();
            let error = sink.submit(&test_request(vec![1])).err().unwrap();
            assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(error.code, "engine_submit_failed");
            assert!(error.message.contains("queue full"));
            assert_eq!(fake.requests(), 0);
        });
    }

    #[test]
    fn py_engine_rejects_a_prompt_larger_than_the_kv_pool() {
        Python::attach(|py| {
            let fake = Fake::new(py, false, 64, Some(2));
            let error = fake
                .sink()
                .submit(&test_request(vec![1, 2, 3]))
                .err()
                .unwrap();
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert!(error.message.contains("holds at most 2 tokens"));
            assert_eq!(fake.requests(), 0);
        });
    }

    #[test]
    fn py_engine_pops_the_sequence_even_when_abort_raises() {
        Python::attach(|py| {
            let fake = Fake::new(py, false, 64, None);
            let sink = fake.sink();
            sink.submit(&test_request(vec![1])).unwrap();
            let raising = py.eval(c"lambda seq_id: 1 / 0", None, None).unwrap();
            fake.get("core_mgr")
                .setattr("abort_request", raising)
                .unwrap();
            sink.release(&[77], true);
            assert_eq!(fake.requests(), 0);
        });
    }
}
