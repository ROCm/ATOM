//! `EngineClient`: the one client of every EngineCore, as CoreManager is for
//! the Python runtime. It places and submits sequences, routes each Core's
//! outputs to the sequences they name, and aborts a sequence whose
//! `SeqHandle` is dropped before it ended.
//!
//! Threads: requests are sent from the caller's task (each Core's sockets sit
//! behind a lock); one reader thread per Core decodes its output socket.

use std::{
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
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::{
    balance::Balancer,
    link::{process_alive, recv_frames, CoreLink},
    spec::EngineSpec,
    wire::{self, pb, Inbound},
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
}

#[derive(Debug)]
pub enum SeqFailure {
    /// `Rejected` INVALID: the request itself is at fault.
    Invalid(String),
    /// `Rejected` OVERLOADED, or the Core went away.
    Unavailable(String),
    /// `Rejected` INTERNAL.
    Internal(String),
}

/// One request, fanned out into `n` sibling sequences.
pub struct Submission<'a> {
    pub request_id: &'a str,
    pub prompt: &'a [u32],
    pub sampling: pb::SamplingParams,
    pub stop_token_sequences: Vec<pb::TokenIds>,
    pub n: u32,
    pub session: Option<&'a str>,
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
    balancer: Mutex<Balancer>,
    next_seq_id: AtomicU64,
    max_pool_tokens: Option<u64>,
    stopping: AtomicBool,
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
    tokens: Vec<u32>,
    events: mpsc::UnboundedSender<SeqEvent>,
}

impl EngineClient {
    /// Bind every Core's sockets and wait until all of them are ready. Blocks.
    pub fn connect(spec: &EngineSpec) -> Result<Self, String> {
        let ctx = zmq::Context::new();
        let mut links = Vec::with_capacity(spec.engines.len());
        let mut outputs = Vec::with_capacity(spec.engines.len());
        let mut max_pool_tokens: Option<u64> = None;
        for core in &spec.engines {
            let (link, output, pool) = CoreLink::connect(&ctx, core.clone())
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
            balancer: Mutex::new(Balancer::new(spec.balance.clone(), links.len())),
            links,
            seqs: DashMap::new(),
            next_seq_id: AtomicU64::new(0),
            max_pool_tokens,
            stopping: AtomicBool::new(false),
            _ctx: ctx,
        });
        let readers = outputs
            .into_iter()
            .enumerate()
            .map(|(rank, output)| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("atom-core-out-{rank}"))
                    .spawn(move || shared.read_outputs(rank, output))
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
        self.shared.balancer.lock().alive_count()
    }

    /// In-flight requests and queued prefill tokens, per DP rank.
    pub fn rank_loads(&self) -> (Vec<u64>, Vec<u64>) {
        self.shared.balancer.lock().loads()
    }

    /// Place the request's `n` sequences, then send each Core its share.
    pub fn submit(&self, sub: Submission<'_>) -> Result<Vec<SeqHandle>, SeqFailure> {
        let shared = &self.shared;
        let n = sub.n.max(1);
        let first = shared.next_seq_id.fetch_add(n as u64, Ordering::Relaxed);
        let seq_ids: Vec<u64> = (first..first + n as u64).collect();

        let mut ranks = Vec::with_capacity(seq_ids.len());
        {
            let mut balancer = shared.balancer.lock();
            for &seq_id in &seq_ids {
                match balancer.admit(seq_id, sub.prompt.len() as u64, sub.session) {
                    Some(rank) => ranks.push(rank),
                    None => {
                        seq_ids.iter().for_each(|&s| balancer.release(s));
                        return Err(SeqFailure::Unavailable("no EngineCore is alive".into()));
                    }
                }
            }
        }

        // Registered before sending, so no output can arrive for an unknown id.
        let handles: Vec<SeqHandle> = seq_ids
            .iter()
            .zip(&ranks)
            .map(|(&seq_id, &rank)| {
                let (tx, rx) = mpsc::unbounded_channel();
                let slot = Slot {
                    rank,
                    stream: sub.stream,
                    needs_completion: sub.sampling.logprobs.is_some(),
                    prompt_tokens: sub.prompt.len() as u32,
                    tokens: Vec::new(),
                    events: tx,
                };
                shared.seqs.insert(seq_id, slot);
                SeqHandle {
                    seq_id,
                    events: rx,
                    shared: shared.clone(),
                    ended: false,
                }
            })
            .collect();

        let prompt = wire::token_frame(sub.prompt);
        let arrive_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or_default();
        for (rank, link) in shared.links.iter().enumerate() {
            let specs: Vec<pb::RequestSpec> = seq_ids
                .iter()
                .zip(&ranks)
                .enumerate()
                .filter(|(_, (_, &r))| r == rank)
                .map(|(sibling, (&seq_id, _))| {
                    sub.request_spec(seq_id, sibling as u32, rank, prompt.len(), arrive_time)
                })
                .collect();
            if specs.is_empty() {
                continue;
            }
            let body = pb::envelope::Body::Submit(pb::SubmitBatch { requests: specs });
            if let Err(e) = link.send_input(wire::encode(body, vec![prompt.clone()])) {
                error!(dp_rank = rank, "submit failed: {e}");
                // Dropping the handles aborts and releases every sibling.
                return Err(SeqFailure::Unavailable(format!(
                    "EngineCore send failed: {e}"
                )));
            }
        }
        Ok(handles)
    }

    /// Ask every live Core to stop, wait for them to say so, then stop reading.
    pub fn shutdown(&self, timeout: Duration) {
        let shutdown = wire::encode(pb::envelope::Body::Shutdown(pb::Shutdown {}), vec![]);
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
    }
}

impl Submission<'_> {
    /// Sibling `sibling` of this request, as Core `rank` admits it; its
    /// prompt is frame 1 of the message, shared by every sibling there.
    fn request_spec(
        &self,
        seq_id: u64,
        sibling: u32,
        rank: usize,
        prompt_bytes: usize,
        arrive_time: f64,
    ) -> pb::RequestSpec {
        let fanout = self.n > 1;
        pb::RequestSpec {
            seq_id,
            prompt_token_ids: Some(pb::BufferRef {
                frame: 1,
                offset: 0,
                nbytes: prompt_bytes as u64,
            }),
            sampling: Some(self.sampling.clone()),
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
        self.balancer.lock().release(seq_id);
        let abort = pb::envelope::Body::Abort(pb::Abort {
            seq_ids: vec![seq_id],
        });
        if let Err(e) = self.links[slot.rank].send_control(wire::encode(abort, vec![])) {
            warn!(seq_id, dp_rank = slot.rank, "abort not delivered: {e}");
        }
    }

    /// End `seq_id` with the event `ending` makes of it, releasing its charge.
    fn end(&self, seq_id: u64, ending: impl FnOnce(&mut Slot) -> SeqEvent) {
        if let Some((_, mut slot)) = self.seqs.remove(&seq_id) {
            self.balancer.lock().release(seq_id);
            let event = ending(&mut slot);
            let _ = slot.events.send(event);
        }
    }

    fn read_outputs(&self, rank: usize, output: zmq::Socket) {
        let pid = self.links[rank].spec.pid;
        if let Err(e) = output.set_rcvtimeo(READER_TICK_MS) {
            error!(dp_rank = rank, "EngineCore output socket failed: {e}");
            return self.core_lost(rank);
        }
        while !self.stopping.load(Ordering::Acquire) {
            let message = match recv_frames(&output) {
                Ok(frames) => Inbound::decode(frames),
                Err(zmq::Error::EAGAIN) if process_alive(pid) => continue,
                Err(zmq::Error::EAGAIN) => {
                    error!(dp_rank = rank, pid, "EngineCore process exited");
                    return self.core_lost(rank);
                }
                Err(e) => {
                    error!(dp_rank = rank, "EngineCore output socket failed: {e}");
                    return self.core_lost(rank);
                }
            };
            match message {
                Ok(message) => {
                    if let Err(e) = self.dispatch(rank, &message) {
                        error!(dp_rank = rank, "dropping undecodable output: {e}");
                    }
                    if matches!(message.body, pb::envelope::Body::Stopped(_)) {
                        info!(dp_rank = rank, "EngineCore stopped");
                        return self.core_lost(rank);
                    }
                }
                Err(e) => error!(dp_rank = rank, "dropping undecodable output: {e}"),
            }
        }
    }

    fn dispatch(&self, rank: usize, message: &Inbound) -> Result<(), String> {
        use pb::envelope::Body;
        match &message.body {
            Body::Stream(batch) => self.on_stream(batch),
            Body::Completions(done) => {
                for item in &done.items {
                    let finished = Finished {
                        token_ids: message.tokens(item.completion_token_ids.as_ref())?,
                        logprobs: match item.return_logprobs {
                            true => Some(message.floats(item.logprobs.as_ref())?),
                            false => None,
                        },
                        prompt_tokens: item.num_prompt_tokens,
                        cached_tokens: item.num_cached_tokens,
                        leave_reason: item.leave_reason.clone(),
                    };
                    self.end(item.seq_id, |_| SeqEvent::Finished(finished));
                }
                Ok(())
            }
            Body::Rejected(rejected) => {
                for item in &rejected.items {
                    let reason = item.reason.clone();
                    let failure = match pb::rejected_request::Code::try_from(item.code) {
                        Ok(pb::rejected_request::Code::Invalid) => SeqFailure::Invalid(reason),
                        Ok(pb::rejected_request::Code::Overloaded) => {
                            SeqFailure::Unavailable(reason)
                        }
                        _ => SeqFailure::Internal(reason),
                    };
                    self.end(item.seq_id, |_| SeqEvent::Failed(failure));
                }
                Ok(())
            }
            other => {
                debug!(
                    dp_rank = rank,
                    "ignoring output {:?}",
                    std::mem::discriminant(other)
                );
                Ok(())
            }
        }
    }

    /// Route one step's tokens. A sequence's last entry ends it, unless it
    /// waits for its `Completion`; that `Completion`, when it comes, finds
    /// nothing left to end.
    fn on_stream(&self, batch: &pb::StreamBatch) -> Result<(), String> {
        let mut offset = 0usize;
        let mut ended = Vec::new();
        {
            let mut balancer = self.balancer.lock();
            for (i, (&seq_id, &count)) in batch.seq_ids.iter().zip(&batch.token_counts).enumerate()
            {
                let end = offset + count as usize;
                let tokens = batch
                    .tokens
                    .get(offset..end)
                    .ok_or("stream token_counts overrun tokens")?;
                offset = end;
                balancer.prefill_done(seq_id);
                let finished = batch.finished.get(i).copied().unwrap_or(false);
                if finished {
                    balancer.release(seq_id);
                }
                let Some(mut slot) = self.seqs.get_mut(&seq_id) else {
                    continue;
                };
                let tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                slot.tokens.extend_from_slice(&tokens);
                if slot.stream {
                    let _ = slot.events.send(SeqEvent::Tokens(tokens));
                }
                if finished && !slot.needs_completion {
                    ended.push((seq_id, i as u32));
                }
            }
        }
        for (seq_id, i) in ended {
            let leave_reason = batch.finish_reasons.get(&i).cloned().unwrap_or_default();
            let cached_tokens = batch
                .num_cached_tokens
                .get(i as usize)
                .copied()
                .unwrap_or(0);
            self.end(seq_id, |slot| {
                SeqEvent::Finished(Finished {
                    token_ids: std::mem::take(&mut slot.tokens),
                    logprobs: None,
                    prompt_tokens: slot.prompt_tokens,
                    cached_tokens: cached_tokens as u32,
                    leave_reason,
                })
            });
        }
        Ok(())
    }

    /// Core `rank` is gone: fail what runs there and place nothing more on it.
    fn core_lost(&self, rank: usize) {
        let mut balancer = self.balancer.lock();
        balancer.mark_dead(rank);
        self.seqs.retain(|&seq_id, slot| {
            if slot.rank != rank {
                return true;
            }
            balancer.release(seq_id);
            let failure = SeqFailure::Unavailable(format!("EngineCore DP rank {rank} is gone"));
            let _ = slot.events.send(SeqEvent::Failed(failure));
            false
        });
    }
}
