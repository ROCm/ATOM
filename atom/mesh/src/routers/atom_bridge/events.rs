//! A request's engine events as the `TokenChunk` stream render reads.
//!
//! Nothing here knows Python: a backend registers a request, delivers its
//! per-step tokens and its finish through `EventRegistry`, and says how to
//! stop it (`Aborter`). The token bridge is one such backend; a Rust client
//! of the EngineCores would be another.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

use dashmap::DashMap;
use futures::Stream;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use crate::routers::token_handle::{
    engine_error::EngineError,
    token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta},
    token_handle::TokenSource,
};

/// One engine step of one request, or the end of its stream.
#[derive(Debug)]
pub(crate) enum SeqEvent {
    Step {
        token_ids: Vec<u32>,
        finish: Option<SeqFinish>,
    },
    /// The backend gave up on the request (engine death, shutdown).
    Failed(String),
}

#[derive(Debug)]
pub(crate) struct SeqFinish {
    /// The engine's own reason (`eos`, `max_tokens`, `stop_sequence`, ...).
    pub reason: Option<String>,
    pub cached_tokens: u32,
}

/// Where a backend delivers events, by the key the request was registered
/// under. A key is registered before the request is submitted, so an event
/// can never arrive for a request this side does not know.
#[derive(Default)]
pub(crate) struct EventRegistry {
    next_key: AtomicU64,
    streams: DashMap<u64, mpsc::UnboundedSender<SeqEvent>>,
}

impl EventRegistry {
    pub(crate) fn register(&self) -> (u64, mpsc::UnboundedReceiver<SeqEvent>) {
        let key = self.next_key.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();
        self.streams.insert(key, tx);
        (key, rx)
    }

    pub(crate) fn deliver(&self, key: u64, event: SeqEvent) {
        let last = matches!(
            event,
            SeqEvent::Failed(_)
                | SeqEvent::Step {
                    finish: Some(_),
                    ..
                }
        );
        if last {
            if let Some((_, tx)) = self.streams.remove(&key) {
                let _ = tx.send(event);
            }
        } else if let Some(tx) = self.streams.get(&key) {
            let _ = tx.send(event);
        }
    }

    pub(crate) fn forget(&self, key: u64) {
        self.streams.remove(&key);
    }

    /// End every open stream, e.g. on shutdown.
    pub(crate) fn fail_all(&self, message: &str) {
        let keys: Vec<u64> = self.streams.iter().map(|entry| *entry.key()).collect();
        for key in keys {
            self.deliver(key, SeqEvent::Failed(message.to_string()));
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.streams.len()
    }
}

/// How a backend stops a request it was given.
pub(crate) trait Aborter: Send + Sync {
    fn abort(&self, key: u64);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Registered, handed to the backend, not yet accepted.
    Pending,
    /// The backend has it; dropping the request must stop it.
    Submitted,
    /// Dropped while pending: whoever finishes the submit must stop it.
    Cancelled,
    /// Ran to its end; nothing to stop.
    Finished,
}

/// The request's claim on the backend: stops it if the client goes away
/// before it finishes, whichever side of the submit that happens on.
pub(crate) struct Admission {
    key: u64,
    stage: Arc<Mutex<Stage>>,
    registry: Arc<EventRegistry>,
    aborter: Arc<dyn Aborter>,
}

impl Admission {
    pub(crate) fn new(key: u64, registry: Arc<EventRegistry>, aborter: Arc<dyn Aborter>) -> Self {
        Self {
            key,
            stage: Arc::new(Mutex::new(Stage::Pending)),
            registry,
            aborter,
        }
    }

    /// Handed to the code that submits, which calls `submitted` once the
    /// backend has the request.
    pub(crate) fn stage(&self) -> Arc<Mutex<Stage>> {
        Arc::clone(&self.stage)
    }

    fn finish(&self) {
        *self.stage.lock() = Stage::Finished;
    }
}

/// Record that the backend accepted the request. True when it was dropped
/// meanwhile, and the caller must stop it now: nobody else will.
pub(crate) fn submitted(stage: &Mutex<Stage>) -> bool {
    let mut stage = stage.lock();
    if *stage == Stage::Cancelled {
        return true;
    }
    *stage = Stage::Submitted;
    false
}

impl Drop for Admission {
    fn drop(&mut self) {
        let abort = {
            let mut stage = self.stage.lock();
            match *stage {
                Stage::Submitted => true,
                Stage::Pending => {
                    *stage = Stage::Cancelled;
                    false
                }
                Stage::Cancelled | Stage::Finished => false,
            }
        };
        self.registry.forget(self.key);
        if abort {
            self.aborter.abort(self.key);
        }
    }
}

/// One request's events as a `TokenSource`: a `Partial` per step with
/// tokens, then a `Complete` carrying all of them, as the gRPC engines send.
pub(crate) struct SeqEventSource {
    events: mpsc::UnboundedReceiver<SeqEvent>,
    admission: Admission,
    request_id: String,
    prompt_tokens: u32,
    output: Vec<u32>,
    pending_complete: Option<TokenChunk>,
    done: bool,
}

impl SeqEventSource {
    pub(crate) fn new(
        events: mpsc::UnboundedReceiver<SeqEvent>,
        admission: Admission,
        request_id: String,
        prompt_tokens: u32,
    ) -> Self {
        Self {
            events,
            admission,
            request_id,
            prompt_tokens,
            output: Vec::new(),
            pending_complete: None,
            done: false,
        }
    }

    fn complete(&mut self, finish: SeqFinish) -> TokenChunk {
        let completion_tokens = self.output.len() as u32;
        TokenChunk::Complete {
            token_ids: std::mem::take(&mut self.output),
            finish_reason: finish_reason(finish.reason.as_deref()),
            matched_stop: None,
            usage: Usage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens,
                total_tokens: self.prompt_tokens + completion_tokens,
            },
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta {
                request_id: self.request_id.clone(),
                weight_version: None,
                cached_tokens: finish.cached_tokens,
            },
        }
    }
}

/// The engine's reasons in OpenAI's terms (`Sequence.leave_reason`).
pub(crate) fn finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        None | Some("eos") | Some("stop_sequence") | Some("stop") => FinishReason::Stop,
        Some("max_tokens") | Some("length") => FinishReason::Length,
        Some("aborted") | Some("abort") => FinishReason::Abort,
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

impl Stream for SeqEventSource {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(complete) = self.pending_complete.take() {
            return Poll::Ready(Some(Ok(complete)));
        }
        if self.done {
            return Poll::Ready(None);
        }
        loop {
            let event = match self.events.poll_recv(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(event) => event,
            };
            match event {
                None => {
                    self.done = true;
                    return Poll::Ready(Some(Err(EngineError::DecodeIncomplete)));
                }
                Some(SeqEvent::Failed(message)) => {
                    self.done = true;
                    return Poll::Ready(Some(Err(EngineError::DecodeError(message))));
                }
                Some(SeqEvent::Step { token_ids, finish }) => {
                    self.output.extend_from_slice(&token_ids);
                    let Some(finish) = finish else {
                        if token_ids.is_empty() {
                            continue;
                        }
                        return Poll::Ready(Some(Ok(TokenChunk::Partial {
                            token_ids,
                            logprobs: None,
                        })));
                    };
                    self.done = true;
                    self.admission.finish();
                    let complete = self.complete(finish);
                    if token_ids.is_empty() {
                        return Poll::Ready(Some(Ok(complete)));
                    }
                    self.pending_complete = Some(complete);
                    return Poll::Ready(Some(Ok(TokenChunk::Partial {
                        token_ids,
                        logprobs: None,
                    })));
                }
            }
        }
    }
}

impl TokenSource for SeqEventSource {
    fn mark_completed(&mut self) {
        self.admission.finish();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::StreamExt;

    use super::*;

    #[derive(Default)]
    struct CountingAborter(AtomicUsize);

    impl Aborter for CountingAborter {
        fn abort(&self, _key: u64) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn steps_stream_as_partials_then_one_complete() {
        let registry = Arc::new(EventRegistry::default());
        let aborter = Arc::new(CountingAborter::default());
        let (key, events) = registry.register();
        let admission = Admission::new(key, Arc::clone(&registry), aborter.clone());
        assert!(!submitted(&admission.stage()));
        let mut source = SeqEventSource::new(events, admission, "req-1".into(), 4);

        registry.deliver(
            key,
            SeqEvent::Step {
                token_ids: vec![1, 2],
                finish: None,
            },
        );
        registry.deliver(
            key,
            SeqEvent::Step {
                token_ids: vec![3],
                finish: Some(SeqFinish {
                    reason: Some("eos".into()),
                    cached_tokens: 2,
                }),
            },
        );
        let mut items = Vec::new();
        while let Some(item) = source.next().await {
            items.push(item.unwrap());
        }
        assert!(matches!(&items[..], [
            TokenChunk::Partial { .. },
            TokenChunk::Partial { .. },
            TokenChunk::Complete { token_ids, finish_reason: FinishReason::Stop, usage, meta, .. },
        ] if token_ids == &[1, 2, 3] && usage.total_tokens == 7 && meta.cached_tokens == 2));
        drop(source);
        // It finished on its own: nothing to stop.
        assert_eq!(aborter.0.load(Ordering::SeqCst), 0);
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn a_dropped_request_is_stopped_on_whichever_side_of_the_submit() {
        let registry = Arc::new(EventRegistry::default());
        let aborter = Arc::new(CountingAborter::default());

        // Dropped after the submit: the drop stops it.
        let (key, _rx) = registry.register();
        let admission = Admission::new(key, Arc::clone(&registry), aborter.clone());
        assert!(!submitted(&admission.stage()));
        drop(admission);
        assert_eq!(aborter.0.load(Ordering::SeqCst), 1);

        // Dropped while pending: the submit that lands afterwards must.
        let (key, _rx) = registry.register();
        let admission = Admission::new(key, Arc::clone(&registry), aborter.clone());
        let stage = admission.stage();
        drop(admission);
        assert_eq!(aborter.0.load(Ordering::SeqCst), 1);
        assert!(submitted(&stage));
    }
}
