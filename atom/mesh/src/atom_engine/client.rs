//! `EngineClient`: the one client of every EngineCore, as CoreManager is for
//! the Python runtime. It places and submits sequences, routes each Core's
//! outputs to the sequences they name, and aborts a sequence whose
//! `SeqHandle` is dropped before it ended.
//!
//! Threads: requests are sent from the caller's task (each Core's sockets sit
//! behind a lock); one reader thread per Core decodes its output socket.
//!
//! A Core that goes away unasked is lost: the client refuses new work from
//! then on and `wait_lost` resolves, so the server can exit (the service is
//! restarted as a whole; nothing here recovers a Core).

use std::{
    panic::AssertUnwindSafe,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
    thread::JoinHandle,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use dashmap::DashMap;
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

use super::{
    link::{process_alive, CoreLink, Interrupted},
    spec::EngineSpec,
    wire::{self, kind, reject, Inbound},
};

/// How long a reader blocks before checking for shutdown and Core liveness.
const READER_TICK_MS: i32 = 1000;

pub struct EngineClient {
    shared: Arc<Shared>,
    readers: Mutex<Vec<JoinHandle<()>>>,
}

/// What one sequence hears from its Core, in order: tokens, then one ending.
#[derive(Debug)]
pub enum SeqEvent {
    Tokens(Vec<u32>),
    Finished(Finished),
    Failed(SeqFailure),
}

#[derive(Debug)]
pub struct Finished {
    pub token_ids: Vec<u32>,
    pub logprobs: Option<Vec<f64>>,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub leave_reason: String,
    pub times: SeqTimes,
}

/// When a sequence was submitted, gave its first and last tokens, and ended.
#[derive(Debug, Clone, Copy)]
pub struct SeqTimes {
    pub submitted: Instant,
    pub first_token: Option<Instant>,
    pub last_token: Option<Instant>,
    pub ended: Instant,
}

#[derive(Debug)]
pub enum SeqFailure {
    /// Refused as INVALID: the request itself is at fault.
    Invalid(String),
    /// Refused as OVERLOADED, or the Core went away.
    Unavailable(String),
    /// Refused as INTERNAL.
    Internal(String),
}

/// One request, fanned out into `n` sibling sequences.
pub struct Submission<'a> {
    pub request_id: &'a str,
    pub prompt: &'a [u32],
    pub sampling: wire::Sampling,
    pub stop_token_sequences: Vec<Vec<i32>>,
    pub n: u32,
    pub session: Option<&'a str>,
    /// The DP rank the router placed the request on: every sibling goes there.
    pub rank: usize,
    pub stream: bool,
}

/// A submitted sequence's events; dropping it before its ending aborts it.
pub struct SeqHandle {
    seq_id: u64,
    events: mpsc::UnboundedReceiver<SeqEvent>,
    shared: Arc<Shared>,
    ended: bool,
}

struct Shared {
    links: Vec<CoreLink>,
    seqs: DashMap<u64, Slot>,
    /// Per DP rank, whether its Core is still serving. Registering a
    /// request and losing a Core both happen under this lock, so a request is
    /// either refused or failed by `core_lost`, never stranded.
    alive: Mutex<Vec<bool>>,
    next_seq_id: AtomicU64,
    max_pool_tokens: Option<u64>,
    /// `shutdown` has asked the Cores to stop: a `Stopped` is expected.
    shutting_down: AtomicBool,
    /// The readers are to return.
    stopping: AtomicBool,
    /// Why the first Core was lost, once one is.
    lost: watch::Sender<Option<String>>,
    _ctx: zmq::Context,
}

/// An in-flight sequence: where it runs and who is waiting on it.
struct Slot {
    rank: usize,
    stream: bool,
    /// Ends on its `Completion` (which alone carries logprobs) rather than
    /// on its last stream entry, which the Core sends a little earlier.
    needs_completion: bool,
    prompt_tokens: u32,
    /// The prefix-cache hit, as the latest stream entry reports it. A
    /// `Completion`'s own `num_cached_tokens` is the KV computed, not the hit.
    cached_tokens: u32,
    tokens: Vec<u32>,
    submitted: Instant,
    first_token: Option<Instant>,
    last_token: Option<Instant>,
    events: mpsc::UnboundedSender<SeqEvent>,
}

impl Slot {
    fn times(&self) -> SeqTimes {
        SeqTimes {
            submitted: self.submitted,
            first_token: self.first_token,
            last_token: self.last_token,
            ended: Instant::now(),
        }
    }
}

impl EngineClient {
    /// Bind every Core's sockets and wait until all of them are ready. Blocks,
    /// asking `interrupted` between polls.
    pub fn connect(spec: &EngineSpec, interrupted: Interrupted<'_>) -> Result<Self, String> {
        let ctx = zmq::Context::new();
        let mut links = Vec::with_capacity(spec.engines.len());
        let mut outputs = Vec::with_capacity(spec.engines.len());
        let mut max_pool_tokens: Option<u64> = None;
        for core in &spec.engines {
            let (link, output, pool) = CoreLink::connect(&ctx, core.clone(), interrupted)
                .map_err(|e| format!("EngineCore DP rank {}: {e}", core.dp_rank))?;
            // A prompt is only safe if it fits wherever it is placed.
            max_pool_tokens = match (max_pool_tokens, pool) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            links.push(link);
            outputs.push(output);
        }
        let shared = Arc::new(Shared {
            alive: Mutex::new(vec![true; links.len()]),
            links,
            seqs: DashMap::new(),
            next_seq_id: AtomicU64::new(0),
            max_pool_tokens,
            shutting_down: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            lost: watch::Sender::new(None),
            _ctx: ctx,
        });
        let readers = outputs
            .into_iter()
            .enumerate()
            .map(|(rank, output)| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("atom-core-out-{rank}"))
                    .spawn(move || {
                        // A reader that panics would strand its sequences.
                        let read = AssertUnwindSafe(|| shared.read_outputs(rank, output));
                        if std::panic::catch_unwind(read).is_err() {
                            shared.core_lost(rank, "output reader panicked");
                        }
                    })
                    .expect("spawn EngineCore output reader")
            })
            .collect();
        Ok(Self {
            shared,
            readers: Mutex::new(readers),
        })
    }

    /// The longest prompt every Core's KV pool can hold, if they reported it.
    pub fn max_pool_tokens(&self) -> Option<u64> {
        self.shared.max_pool_tokens
    }

    pub fn core_count(&self) -> usize {
        self.shared.links.len()
    }

    pub fn alive_core_count(&self) -> usize {
        self.shared
            .alive
            .lock()
            .iter()
            .filter(|&&alive| alive)
            .count()
    }

    /// Why a Core was lost, if one was.
    pub fn lost(&self) -> Option<String> {
        self.shared.lost.borrow().clone()
    }

    /// Resolves once a Core is lost, with why.
    pub async fn wait_lost(&self) -> String {
        let mut lost = self.shared.lost.subscribe();
        let reason = lost
            .wait_for(Option::is_some)
            .await
            .expect("the sender lives as long as self");
        reason.clone().unwrap_or_default()
    }

    /// Place the request's `n` sequences, then send each Core its share.
    pub fn submit(&self, sub: Submission<'_>) -> Result<Vec<SeqHandle>, SeqFailure> {
        let shared = &self.shared;
        if let Some(reason) = self.lost() {
            return Err(SeqFailure::Unavailable(format!(
                "the server is shutting down: {reason}"
            )));
        }
        let cores = shared.links.len();
        let rank = sub.rank;
        if rank >= cores {
            return Err(SeqFailure::Invalid(format!(
                "data_parallel_rank must be in [0, {cores}), got {rank}"
            )));
        }
        let n = sub.n.max(1);
        let first = shared.next_seq_id.fetch_add(n as u64, Ordering::Relaxed);
        let submitted = Instant::now();
        let seq_ids: Vec<u64> = (first..first + n as u64).collect();

        // Registered under the lock `core_lost` also takes: it either marks
        // the rank dead first, or finds the slots to fail. Registered before
        // sending, so no output can arrive for an unknown id.
        let mut handles = Vec::with_capacity(seq_ids.len());
        {
            let alive = shared.alive.lock();
            if !alive[rank] {
                return Err(SeqFailure::Unavailable(format!(
                    "EngineCore DP rank {rank} is gone"
                )));
            }
            for &seq_id in &seq_ids {
                let (tx, rx) = mpsc::unbounded_channel();
                let slot = Slot {
                    rank,
                    stream: sub.stream,
                    needs_completion: sub.sampling.logprobs.is_some(),
                    prompt_tokens: sub.prompt.len() as u32,
                    cached_tokens: 0,
                    tokens: Vec::new(),
                    submitted,
                    first_token: None,
                    last_token: None,
                    events: tx,
                };
                shared.seqs.insert(seq_id, slot);
                handles.push(SeqHandle {
                    seq_id,
                    events: rx,
                    shared: shared.clone(),
                    ended: false,
                });
            }
        }

        let prompt = wire::token_bytes(sub.prompt);
        let arrive_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or_default();
        let items: Vec<wire::SubmitItem> = seq_ids
            .iter()
            .enumerate()
            .map(|(sibling, &seq_id)| {
                sub.submit_item(seq_id, sibling as u32, rank, &prompt, arrive_time)
            })
            .collect();
        let submit = wire::encode(kind::ADD, &wire::Submit { items });
        if let Err(e) = shared.links[rank].send_input(submit) {
            error!(dp_rank = rank, "submit failed: {e}");
            // Dropping the handles aborts every sibling.
            return Err(SeqFailure::Unavailable(format!(
                "EngineCore send failed: {e}"
            )));
        }
        Ok(handles)
    }

    /// From here on a Core that stops or exits is expected to: the server is
    /// going down (a signal reaches the Cores' process group too).
    pub fn begin_shutdown(&self) {
        self.shared.shutting_down.store(true, Ordering::Release);
    }

    /// Ask every live Core to stop, wait for them to say so, then stop reading.
    /// Whatever is still in flight then ends as failed.
    pub fn shutdown(&self, timeout: Duration) {
        self.begin_shutdown();
        let shutdown = wire::encode(kind::SHUTDOWN, &wire::Empty {});
        for link in &self.shared.links {
            if let Err(e) = link.send_control(shutdown.clone()) {
                warn!(dp_rank = link.spec.dp_rank, "shutdown not delivered: {e}");
            }
        }
        let mut readers = std::mem::take(&mut *self.readers.lock());
        let deadline = Instant::now() + timeout;
        while readers.iter().any(|r| !r.is_finished()) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        self.shared.stopping.store(true, Ordering::Release);
        readers.drain(..).for_each(|r| {
            let _ = r.join();
        });
        self.shared.seqs.retain(|_, slot| {
            let failure = SeqFailure::Unavailable("the server is shutting down".into());
            let _ = slot.events.send(SeqEvent::Failed(failure));
            false
        });
    }
}

impl Submission<'_> {
    /// Sibling `sibling` of this request, as Core `rank` admits it.
    fn submit_item(
        &self,
        seq_id: u64,
        sibling: u32,
        rank: usize,
        prompt: &serde_bytes::ByteBuf,
        arrive_time: f64,
    ) -> wire::SubmitItem {
        let fanout = self.n > 1;
        wire::SubmitItem {
            seq_id,
            prompt: prompt.clone(),
            sampling: self.sampling.clone(),
            stop_token_sequences: self.stop_token_sequences.clone(),
            external_request_id: (!fanout).then(|| self.request_id.to_string()),
            parent_request_id: Some(self.request_id.to_string()),
            sibling_index: sibling,
            needs_independent_noise: fanout,
            data_parallel_rank: Some(rank as i64),
            dp_session_id: self.session.map(str::to_string),
            arrive_time,
            ..Default::default()
        }
    }
}

impl SeqHandle {
    pub fn poll_event(&mut self, cx: &mut Context<'_>) -> Poll<Option<SeqEvent>> {
        let polled = self.events.poll_recv(cx);
        if let Poll::Ready(Some(SeqEvent::Finished(_) | SeqEvent::Failed(_)) | None) = &polled {
            self.ended = true;
        }
        polled
    }
}

impl Drop for SeqHandle {
    fn drop(&mut self) {
        if !self.ended {
            self.shared.abort(self.seq_id);
        }
    }
}

impl Shared {
    fn abort(&self, seq_id: u64) {
        let Some((_, slot)) = self.seqs.remove(&seq_id) else {
            return;
        };
        let abort = wire::encode(
            kind::ABORT,
            &wire::Abort {
                seq_ids: vec![seq_id],
            },
        );
        if let Err(e) = self.links[slot.rank].send_control(abort) {
            warn!(seq_id, dp_rank = slot.rank, "abort not delivered: {e}");
        }
    }

    /// End `seq_id` with the event `ending` makes of it.
    fn end(&self, seq_id: u64, ending: impl FnOnce(&mut Slot) -> SeqEvent) {
        if let Some((_, mut slot)) = self.seqs.remove(&seq_id) {
            let event = ending(&mut slot);
            let _ = slot.events.send(event);
        }
    }

    fn read_outputs(&self, rank: usize, output: zmq::Socket) {
        let pid = self.links[rank].spec.pid;
        if let Err(e) = output.set_rcvtimeo(READER_TICK_MS) {
            return self.core_lost(rank, &format!("output socket failed: {e}"));
        }
        while !self.stopping.load(Ordering::Acquire) {
            let message = match output.recv_msg(0) {
                Ok(message) => Inbound::decode(&message),
                Err(zmq::Error::EAGAIN) if process_alive(pid) => continue,
                Err(zmq::Error::EAGAIN) => {
                    return self.core_lost(rank, &format!("process {pid} exited"));
                }
                Err(e) => {
                    return self.core_lost(rank, &format!("output socket failed: {e}"));
                }
            };
            // An output that does not decode may have carried a sequence's
            // last message, and which ones it was cannot be told: losing the
            // Core fails them rather than leaving them, and their load, hung.
            let message = match message {
                Ok(message) => message,
                Err(e) => return self.core_lost(rank, &format!("sent an undecodable output: {e}")),
            };
            if let Err(e) = self.dispatch(rank, &message) {
                return self.core_lost(rank, &format!("sent an undecodable output: {e}"));
            }
            if matches!(message, Inbound::Stopped) {
                return self.core_lost(rank, "stopped");
            }
        }
    }

    fn dispatch(&self, rank: usize, message: &Inbound) -> Result<(), String> {
        match message {
            Inbound::Stream(stream) => self.on_stream(stream),
            Inbound::Completions(done) => {
                // Item by item: one that does not decode fails alone.
                for item in &done.items {
                    if item.reject_code != reject::FINISHED {
                        self.on_refused(item);
                        continue;
                    }
                    let decoded = wire::tokens(&item.completion_tokens).and_then(|token_ids| {
                        // Only a sequence that asked for them has logprobs.
                        let logprobs = item
                            .logprobs_bytes
                            .as_ref()
                            .map(|b| wire::floats(b))
                            .transpose()?;
                        Ok((token_ids, logprobs))
                    });
                    self.end(item.seq_id, |slot| match decoded {
                        Ok((token_ids, logprobs)) => SeqEvent::Finished(Finished {
                            token_ids,
                            logprobs,
                            prompt_tokens: item.num_prompt_tokens,
                            cached_tokens: slot.cached_tokens,
                            leave_reason: item.leave_reason.clone(),
                            times: slot.times(),
                        }),
                        Err(e) => {
                            error!(
                                dp_rank = rank,
                                seq_id = item.seq_id,
                                "undecodable completion: {e}"
                            );
                            SeqEvent::Failed(SeqFailure::Internal(e))
                        }
                    });
                }
                Ok(())
            }
            Inbound::Ignored(kind) => {
                debug!(dp_rank = rank, kind, "ignoring output");
                Ok(())
            }
            Inbound::Ready(_) | Inbound::Stopped => Ok(()),
        }
    }

    /// A sequence the Core refused: it fails with the reason the Core gave.
    fn on_refused(&self, item: &wire::Completion) {
        let reason = item
            .leave_reason
            .strip_prefix("rejected: ")
            .unwrap_or(&item.leave_reason)
            .to_string();
        let failure = match item.reject_code {
            reject::INVALID => SeqFailure::Invalid(reason),
            reject::OVERLOADED => SeqFailure::Unavailable(reason),
            _ => SeqFailure::Internal(reason),
        };
        self.end(item.seq_id, |_| SeqEvent::Failed(failure));
    }

    /// Route one step's tokens. A sequence's last entry ends it, unless it
    /// waits for its `Completion`; that `Completion`, when it comes, finds
    /// nothing left to end.
    fn on_stream(&self, stream: &wire::Stream) -> Result<(), String> {
        let mut ended = Vec::new();
        for item in &stream.items {
            let Some(mut slot) = self.seqs.get_mut(&item.seq_id) else {
                continue;
            };
            slot.cached_tokens = item.num_cached_tokens as u32;
            let tokens = item.output_tokens.clone();
            if !tokens.is_empty() {
                let now = Instant::now();
                slot.first_token.get_or_insert(now);
                slot.last_token = Some(now);
            }
            slot.tokens.extend_from_slice(&tokens);
            if slot.stream {
                let _ = slot.events.send(SeqEvent::Tokens(tokens));
            }
            if item.finished && !slot.needs_completion {
                ended.push((item.seq_id, item.finish_reason.clone().unwrap_or_default()));
            }
        }
        for (seq_id, leave_reason) in ended {
            self.end(seq_id, |slot| {
                SeqEvent::Finished(Finished {
                    token_ids: std::mem::take(&mut slot.tokens),
                    logprobs: None,
                    prompt_tokens: slot.prompt_tokens,
                    cached_tokens: slot.cached_tokens,
                    leave_reason,
                    times: slot.times(),
                })
            });
        }
        Ok(())
    }

    /// Core `rank` is gone: fail what runs there and place nothing more on it.
    /// Unless `shutdown` asked it to stop, the client is lost with it.
    fn core_lost(&self, rank: usize, why: &str) {
        // Dead and its sequences failed before `lost` is published: whoever
        // `wait_lost` wakes sees the Core gone.
        {
            let mut alive = self.alive.lock();
            alive[rank] = false;
            self.seqs.retain(|_, slot| {
                if slot.rank != rank {
                    return true;
                }
                let failure = SeqFailure::Unavailable(format!("EngineCore DP rank {rank} is gone"));
                let _ = slot.events.send(SeqEvent::Failed(failure));
                false
            });
        }
        if self.shutting_down.load(Ordering::Acquire) {
            info!(dp_rank = rank, "EngineCore {why}");
        } else {
            error!(
                dp_rank = rank,
                "EngineCore {why} unasked; the server will exit"
            );
            let reason = format!("EngineCore DP rank {rank} {why}");
            self.lost.send_if_modified(|lost| {
                let first = lost.is_none();
                if first {
                    *lost = Some(reason);
                }
                first
            });
        }
    }
}
