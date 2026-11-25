//! A request's engine events as the `TokenChunk` stream render reads.
//!
//! Nothing here knows Python: a backend registers a request, delivers its
//! per-step tokens and its finish through `EventRegistry`, and says how to
//! stop it (`Aborter`). The token bridge is one such backend; a Rust client
//! of the EngineCores would be another.

use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Instant,
};

use dashmap::DashMap;
use futures::Stream;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use crate::routers::token_handle::{
    engine_error::EngineError,
    token_chunk::{FinishReason, SeqTiming, TokenChunk, Usage, WorkerMeta},
    token_handle::TokenSource,
};

/// One engine step of one of a request's sequences, or the end of its
/// stream.
#[derive(Debug)]
pub(crate) enum SeqEvent {
    Step {
        /// The sibling's index, under `n > 1`; 0 otherwise.
        index: u32,
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

    /// A sibling's finish leaves the key open for the others: the request's
    /// `Admission` forgets it when the request is done. A failure ends it.
    pub(crate) fn deliver(&self, key: u64, event: SeqEvent) {
        if matches!(event, SeqEvent::Failed(_)) {
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

/// One request's events as a `TokenSource`, as the gRPC engines send them:
/// a `Partial` per step with tokens, then a `Complete` carrying all of them.
/// Under `n > 1` (never streamed) only the siblings' `Complete`s, in index
/// order.
pub(crate) struct SeqEventSource {
    events: mpsc::UnboundedReceiver<SeqEvent>,
    admission: Admission,
    request_id: String,
    prompt_tokens: u32,
    /// When the request was handed over: its timing runs from here.
    started_at: Instant,
    siblings: Vec<Sibling>,
    /// The next sibling whose `Complete` is due.
    next_complete: usize,
    ready: VecDeque<TokenChunk>,
    done: bool,
}

/// One of a request's sequences, as far as its steps have come.
#[derive(Default)]
struct Sibling {
    output: Vec<u32>,
    finish: Option<SeqFinish>,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    finished_at: Option<Instant>,
}

impl SeqEventSource {
    /// `siblings` is how many sequences the backend made of the request,
    /// handed over at `started_at`.
    pub(crate) fn new(
        events: mpsc::UnboundedReceiver<SeqEvent>,
        admission: Admission,
        request_id: String,
        prompt_tokens: u32,
        siblings: usize,
        started_at: Instant,
    ) -> Self {
        Self {
            events,
            admission,
            request_id,
            prompt_tokens,
            started_at,
            siblings: (0..siblings.max(1)).map(|_| Sibling::default()).collect(),
            next_complete: 0,
            ready: VecDeque::new(),
            done: false,
        }
    }

    /// Take in one step; queue what it makes ready.
    fn step(&mut self, index: u32, token_ids: Vec<u32>, finish: Option<SeqFinish>) {
        let streamed = self.siblings.len() == 1;
        let Some(sibling) = self.siblings.get_mut(index as usize) else {
            return;
        };
        let now = Instant::now();
        if !token_ids.is_empty() {
            sibling.first_token_at.get_or_insert(now);
            sibling.last_token_at = Some(now);
        }
        sibling.output.extend_from_slice(&token_ids);
        if streamed && !token_ids.is_empty() {
            self.ready.push_back(TokenChunk::Partial {
                token_ids,
                logprobs: None,
            });
        }
        if finish.is_some() {
            sibling.finish = finish;
            sibling.finished_at = Some(now);
        }
        while let Some(Sibling {
            finish: Some(_), ..
        }) = self.siblings.get(self.next_complete)
        {
            let sibling = std::mem::take(&mut self.siblings[self.next_complete]);
            // Each reports the prompt it shares with the others, as each
            // `/generate` item does; a chat counts it once (`AtomUsage`).
            let complete = self.complete(sibling);
            self.ready.push_back(complete);
            self.next_complete += 1;
        }
        if self.next_complete == self.siblings.len() {
            self.done = true;
            self.admission.finish();
        }
    }

    fn complete(&self, sibling: Sibling) -> TokenChunk {
        let prompt_tokens = self.prompt_tokens;
        let timing = self.timing(&sibling);
        let finish = sibling.finish.unwrap_or(SeqFinish {
            reason: None,
            cached_tokens: 0,
        });
        let completion_tokens = sibling.output.len() as u32;
        TokenChunk::Complete {
            token_ids: sibling.output,
            finish_reason: finish_reason(finish.reason.as_deref()),
            matched_stop: None,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta {
                request_id: self.request_id.clone(),
                weight_version: None,
                cached_tokens: finish.cached_tokens,
                timing: Some(timing),
            },
        }
    }

    /// As `api_server.py` times a sequence: to its first token, per output
    /// token after the first, and end to end.
    fn timing(&self, sibling: &Sibling) -> SeqTiming {
        let since = |at: Instant| at.duration_since(self.started_at).as_secs_f64();
        let tokens = sibling.output.len();
        SeqTiming {
            ttft: sibling.first_token_at.map_or(0.0, since),
            tpot: match (sibling.first_token_at, sibling.last_token_at) {
                (Some(first), Some(last)) if tokens > 1 => {
                    last.duration_since(first).as_secs_f64() / (tokens - 1) as f64
                }
                _ => 0.0,
            },
            latency: sibling.finished_at.map_or(0.0, since),
        }
    }
}

/// The engine's reason (`Sequence.leave_reason`) in OpenAI's terms, as the
/// Python server's `openai_stop_reason` maps it: a length cap is `length`,
/// and every other end -- `eos`, `stop_sequence`, `stop_<token_id>` (a model
/// with more than one EOS), `aborted`, `rejected`, `unschedulable: ...`, or
/// none -- is `stop`. Passing an unknown reason through would reach the
/// client as a `finish_reason` it does not know, and the render upgrades
/// only `stop` to `tool_calls` when the reply called a tool.
pub(crate) fn finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("max_tokens" | "max_new_tokens" | "length") => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

impl Stream for SeqEventSource {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(chunk) = self.ready.pop_front() {
                return Poll::Ready(Some(Ok(chunk)));
            }
            if self.done {
                return Poll::Ready(None);
            }
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
                Some(SeqEvent::Step {
                    index,
                    token_ids,
                    finish,
                }) => self.step(index, token_ids, finish),
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
        let mut source =
            SeqEventSource::new(events, admission, "req-1".into(), 4, 1, Instant::now());

        registry.deliver(
            key,
            SeqEvent::Step {
                index: 0,
                token_ids: vec![1, 2],
                finish: None,
            },
        );
        registry.deliver(
            key,
            SeqEvent::Step {
                index: 0,
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

    #[tokio::test]
    async fn siblings_complete_in_index_order_without_partials() {
        let registry = Arc::new(EventRegistry::default());
        let aborter = Arc::new(CountingAborter::default());
        let (key, events) = registry.register();
        let admission = Admission::new(key, Arc::clone(&registry), aborter.clone());
        assert!(!submitted(&admission.stage()));
        let mut source =
            SeqEventSource::new(events, admission, "req-n".into(), 4, 2, Instant::now());

        let finish = || {
            Some(SeqFinish {
                reason: None,
                cached_tokens: 0,
            })
        };
        for (index, token_ids, finish) in [
            (1, vec![7], finish()),
            (0, vec![5], None),
            (0, vec![6], finish()),
        ] {
            registry.deliver(
                key,
                SeqEvent::Step {
                    index,
                    token_ids,
                    finish,
                },
            );
        }
        let mut outputs = Vec::new();
        while let Some(item) = source.next().await {
            match item.unwrap() {
                TokenChunk::Complete { token_ids, .. } => outputs.push(token_ids),
                partial => panic!("unexpected {partial:?}"),
            }
        }
        assert_eq!(outputs, vec![vec![5, 6], vec![7]]);
        drop(source);
        assert_eq!(aborter.0.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn engine_reasons_map_as_the_python_server_maps_them() {
        for reason in ["max_tokens", "max_new_tokens", "length"] {
            assert!(matches!(finish_reason(Some(reason)), FinishReason::Length));
        }
        for reason in [
            "eos",
            "stop_sequence",
            "stop_151645",
            "aborted",
            "rejected",
            "unschedulable: prompt too long",
            "",
        ] {
            assert!(
                matches!(finish_reason(Some(reason)), FinishReason::Stop),
                "{reason}"
            );
        }
        assert!(matches!(finish_reason(None), FinishReason::Stop));
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
