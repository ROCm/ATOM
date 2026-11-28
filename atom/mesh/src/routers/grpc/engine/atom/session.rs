// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! The Session to an ATOM instance: one `AtomEngine.Session` call carries every
//! request to it as engine_v1 messages, and each request's outputs come back by seq_id.

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;
use std::time::{Duration, Instant};

use futures::future::{BoxFuture, FutureExt, WeakShared};
use futures::{stream, Stream, StreamExt};
use http::StatusCode;
use parking_lot::Mutex;
use prost::Message as _;
use serde_json::Value as Json;
use tokio::runtime;
use tokio::sync::{mpsc, oneshot};
use tonic::{Status, Streaming};
use tracing::warn;

use super::AtomClient;
use crate::atom_engine::grpc::{
    generate_response::Response, Frames, GenerateComplete, GenerateError, GenerateResponse,
    GenerateStreamChunk,
};
use crate::atom_engine::wire::{
    self,
    pb::{self, envelope::Body, rejected_request::Code, value::Kind},
    Inbound,
};
use crate::core::{Worker, WorkerRegistry};
use crate::routers::token_handle::engine_error::EngineError;

/// A Session's seq ids stay below 2^24, the window ATOM maps each Session's ids into:
/// 1 << `_SEQ_ID_SHIFT` (atom/entrypoints/openai/frontend_workers.py), which this must equal.
const SEQ_IDS: u64 = 1 << 24;

/// How long opening a Session may take, from connecting to ATOM's Ready.
const OPENING: Duration = Duration::from_secs(10);

/// How often requests look for Sessions whose instance has no registered worker.
const SWEEP: Duration = Duration::from_secs(1);

/// The request bytes (prompt and RequestSpec) a Session may hold unsent, a dropped one's
/// until the writer discards it; past them a request is refused with 503, or 400 alone.
const UNSENT_BYTES: usize = 256 << 20;

/// The outputs a streamed request may leave unread; past them it fails and is aborted.
const ROUTE_BUFFER: usize = 4096;

type Output = Result<GenerateResponse, Status>;

/// A Session being opened, which every request waiting for it shares.
type Opening = BoxFuture<'static, Result<Session, Arc<EngineError>>>;

/// Each ATOM instance's Session (base URL, all its ranks), opened by the first
/// request to it and again by the first request after it has failed.
pub(crate) struct AtomSessions {
    slots: Mutex<HashMap<String, Slot>>,
    /// Bounds each opening; OPENING but in tests.
    opening: Duration,
    /// With it, the Session of an instance none of whose workers is registered is
    /// released, at most every `sweep`.
    workers: Option<Arc<WorkerRegistry>>,
    sweep: Duration,
    swept: Mutex<Instant>,
}

impl Default for AtomSessions {
    fn default() -> Self {
        Self {
            slots: Mutex::default(),
            opening: OPENING,
            workers: None,
            sweep: SWEEP,
            swept: Mutex::new(Instant::now()),
        }
    }
}

enum Slot {
    Open(Session),
    /// Held weakly: when every request waiting for it has gone, the opening is dropped.
    Opening(WeakShared<Opening>),
}

impl AtomSessions {
    /// Sessions each released once its instance's workers are all removed from `workers`.
    pub(crate) fn releasing_removed(workers: Arc<WorkerRegistry>) -> Self {
        Self {
            workers: Some(workers),
            ..Self::default()
        }
    }

    /// Drops the Sessions of instances no registered worker belongs to; their requests'
    /// streams keep theirs until they end, and the call ends with the last.
    fn release_removed(&self) {
        let Some(workers) = &self.workers else {
            return;
        };
        {
            let mut swept = self.swept.lock();
            if swept.elapsed() < self.sweep {
                return;
            }
            *swept = Instant::now();
        }
        let registered: HashSet<String> = workers
            .get_all()
            .iter()
            .map(|worker| worker.base_url().trim_end_matches('/').to_string())
            .collect();
        self.slots.lock().retain(|url, _| registered.contains(url));
    }

    /// Sends `spec` for `prompt` on `worker`'s instance's Session, which sets its
    /// seq_id; the stream carries what Generate answers, streamed as `stream` says.
    pub(crate) async fn submit(
        &self,
        worker: &Arc<dyn Worker>,
        spec: pb::RequestSpec,
        prompt: Vec<u32>,
        stream: bool,
    ) -> Result<SessionStream, EngineError> {
        self.session(worker).await?.submit(spec, prompt, stream)
    }

    /// The instance's open Session, else the one being opened, else a new opening;
    /// no lock is held while it opens.
    async fn session(&self, worker: &Arc<dyn Worker>) -> Result<Session, EngineError> {
        self.release_removed();
        let url = worker.base_url().trim_end_matches('/').to_string();
        let opening = {
            let mut slots = self.slots.lock();
            let joined = match slots.get(&url) {
                Some(Slot::Open(session)) if session.is_open() => return Ok(session.clone()),
                Some(Slot::Opening(opening)) => opening.upgrade(),
                _ => None,
            };
            joined.unwrap_or_else(|| {
                let opening = Session::open(url.clone(), self.opening).boxed().shared();
                let weak = opening.downgrade().expect("an opening not yet polled");
                slots.insert(url.clone(), Slot::Opening(weak));
                opening
            })
        };
        let opened = opening.clone().await;
        let mut slots = self.slots.lock();
        let ours = matches!(slots.get(&url), Some(Slot::Opening(slot))
            if slot.upgrade().is_some_and(|slot| slot.ptr_eq(&opening)));
        match opened {
            Ok(session) => {
                if ours {
                    slots.insert(url, Slot::Open(session.clone()));
                }
                Ok(session)
            }
            Err(e) => {
                if ours {
                    slots.remove(&url);
                }
                Err(cloned(&e))
            }
        }
    }
}

/// An opening's error for each request that waited for it.
fn cloned(e: &EngineError) -> EngineError {
    match e {
        EngineError::Transport(status) => EngineError::Transport(status.clone()),
        EngineError::ConnectionAcquireFailed(url) => {
            EngineError::ConnectionAcquireFailed(url.clone())
        }
        other => EngineError::Transport(Status::internal(other.to_string())),
    }
}

/// One Session call, shared by its clones and its requests' streams; the call
/// ends once all of them have dropped.
#[derive(Clone)]
struct Session {
    routes: Arc<Mutex<Routes>>,
    commands: mpsc::UnboundedSender<Command>,
    /// Dropped with the last handle, which ends the reader whether or not ATOM ends the call.
    _owners: Arc<oneshot::Sender<()>>,
}

enum Command {
    Submit(Box<pb::RequestSpec>, Vec<u32>),
    /// A Submit whose prompt and bytes wait in `Routes::unsent` under this ticket.
    Queued(Box<pb::RequestSpec>, u64),
    Abort(u64),
}

impl Session {
    /// A handle on `routes` and `commands`; what it returns resolves once its last clone drops.
    fn new(
        routes: Arc<Mutex<Routes>>,
        commands: mpsc::UnboundedSender<Command>,
    ) -> (Self, oneshot::Receiver<()>) {
        let (owners, released) = oneshot::channel();
        let _owners = Arc::new(owners);
        (
            Self {
                routes,
                commands,
                _owners,
            },
            released,
        )
    }

    /// Opens the call on a connection and thread of its own, off the router's runtime;
    /// waits up to `bound` for ATOM's Ready, refusing one of another protocol version.
    /// Dropped before it opens, it ends the call and the thread.
    async fn open(url: String, bound: Duration) -> Result<Self, Arc<EngineError>> {
        let (commands, queued) = mpsc::unbounded_channel();
        let routes = Arc::new(Mutex::new(Routes::default()));
        let (opened, ready) = oneshot::channel();
        let (read_into, aborts) = (routes.clone(), commands.downgrade());
        let (session, released) = Self::new(routes, commands);
        thread::Builder::new()
            .name("atom-session".into())
            .spawn(move || run(url, bound, queued, read_into, aborts, opened, released))
            .map_err(|e| EngineError::Transport(Status::internal(e.to_string())))?;
        let gone = |_| EngineError::Transport(Status::internal("the Session thread ended"));
        ready.await.map_err(gone).and_then(|opened| opened)?;
        Ok(session)
    }

    fn is_open(&self) -> bool {
        !self.routes.lock().closed
    }

    fn submit(
        &self,
        mut spec: pb::RequestSpec,
        prompt: Vec<u32>,
        stream: bool,
    ) -> Result<SessionStream, EngineError> {
        let mut routes = self.routes.lock();
        let bytes = 4 * prompt.len() + spec.encoded_len();
        if bytes > routes.unsent_budget {
            return Err(EngineError::Http {
                status: StatusCode::BAD_REQUEST,
                message: format!(
                    "the request's {bytes} bytes exceed the {} an ATOM Session holds unsent",
                    routes.unsent_budget
                ),
            });
        }
        if routes.unsent_bytes + bytes > routes.unsent_budget {
            return Err(EngineError::Http {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "the ATOM Session is not reading the requests sent to it".to_string(),
            });
        }
        let seq_id = routes.free_seq_id().ok_or_else(|| EngineError::Http {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "every seq_id of the ATOM Session is in use".to_string(),
        })?;
        spec.seq_id = seq_id;
        let (tx, rx) = mpsc::channel(ROUTE_BUFFER);
        let route = Route {
            tx,
            overflowed: false,
            stream,
            prompt_tokens: prompt.len() as u32,
            completion_tokens: 0,
            cached_tokens: 0,
            ids: Vec::new(),
        };
        let ticket = routes.next_ticket;
        let command = Command::Queued(Box::new(spec), ticket);
        if routes.closed || self.commands.send(command).is_err() {
            return Err(EngineError::Transport(Status::unavailable(
                "the ATOM Session has ended",
            )));
        }
        routes.next_ticket += 1;
        routes.unsent_bytes += bytes;
        let prompt = Some(prompt);
        routes.unsent.insert(ticket, Unsent { prompt, bytes });
        routes.open.insert(seq_id, route);
        Ok(SessionStream {
            seq_id,
            ticket,
            rx,
            session: self.clone(),
        })
    }
}

/// The Session's thread: a runtime of its own, which connects, opens the call within
/// `bound` and reads it until it ends or is `released`; `opened` gets how the opening
/// went. With no one left to tell, it ends the call.
fn run(
    url: String,
    bound: Duration,
    queued: mpsc::UnboundedReceiver<Command>,
    routes: Arc<Mutex<Routes>>,
    aborts: mpsc::WeakUnboundedSender<Command>,
    mut opened: oneshot::Sender<Result<(), EngineError>>,
    released: oneshot::Receiver<()>,
) {
    let runtime = match runtime::Builder::new_current_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(e) => {
            let _ = opened.send(Err(EngineError::Transport(Status::internal(e.to_string()))));
            return;
        }
    };
    runtime.block_on(async move {
        let started = tokio::select! {
            started = tokio::time::timeout(bound, start(&url, queued, routes.clone())) => {
                started.unwrap_or_else(|_| Err(EngineError::Transport(Status::deadline_exceeded(
                    format!("ATOM Session: no Ready within {bound:?}"),
                ))))
            }
            _ = opened.closed() => return,
        };
        match started {
            Ok((_client, inbound)) => {
                if opened.send(Ok(())).is_ok() {
                    read(inbound, routes, aborts, url, released).await;
                }
            }
            Err(e) => {
                let _ = opened.send(Err(e));
            }
        }
    });
}

/// Connects to `url`, opens the call and reads ATOM's Ready; the client keeps the
/// connection the call runs on.
async fn start(
    url: &str,
    queued: mpsc::UnboundedReceiver<Command>,
    routes: Arc<Mutex<Routes>>,
) -> Result<(AtomClient, Streaming<Frames>), EngineError> {
    let client = AtomClient::connect(url)
        .await
        .map_err(|_| EngineError::ConnectionAcquireFailed(url.to_string()))?;
    let mut inbound = client
        .session(writer(queued, routes))
        .await
        .map_err(EngineError::Transport)?;
    let first = inbound.message().await.map_err(EngineError::Transport)?;
    match first.map(|message| Inbound::decode(message.frames).map(|m| m.body)) {
        Some(Ok(Body::Ready(_))) => Ok((client, inbound)),
        Some(Err(e)) => Err(refused(&e)),
        _ => Err(refused("the call did not start with Ready")),
    }
}

fn refused(reason: &str) -> EngineError {
    EngineError::Transport(Status::failed_precondition(format!(
        "ATOM Session refused: {reason}"
    )))
}

/// The writer: what is queued when it runs goes out as SubmitBatches and Aborts in
/// queue order, so an Abort never goes before its request nor after a later one. A
/// request dropped before its prompt was taken is not sent.
fn writer(
    queued: mpsc::UnboundedReceiver<Command>,
    routes: Arc<Mutex<Routes>>,
) -> impl Stream<Item = Frames> + Send + 'static {
    stream::unfold(queued, move |mut queued| {
        let routes = routes.clone();
        async move {
            let mut commands = vec![queued.recv().await?];
            while let Ok(command) = queued.try_recv() {
                commands.push(command);
            }
            let commands = routes.lock().take_unsent(commands);
            Some((encode(commands), queued))
        }
    })
    .flat_map(stream::iter)
}

/// `commands` as messages: each run of Submits then Aborts as one SubmitBatch then
/// one Abort, a Submit after an Abort starting the next run.
fn encode(commands: Vec<Command>) -> Vec<Frames> {
    let (mut requests, mut prompts, mut aborts) = (Vec::new(), Vec::new(), Vec::new());
    let mut messages = Vec::new();
    for command in commands {
        match command {
            Command::Submit(mut spec, prompt) => {
                if !aborts.is_empty() {
                    flush(&mut messages, &mut requests, &mut prompts, &mut aborts);
                }
                let frame = wire::token_frame(&prompt);
                spec.prompt_token_ids = Some(pb::BufferRef {
                    frame: prompts.len() as u32 + 1,
                    offset: 0,
                    nbytes: frame.len() as u64,
                });
                prompts.push(frame);
                requests.push(*spec);
            }
            Command::Abort(seq_id) => aborts.push(seq_id),
            Command::Queued(..) => unreachable!("the writer takes each Queued's prompt first"),
        }
    }
    flush(&mut messages, &mut requests, &mut prompts, &mut aborts);
    messages
}

/// The run so far as its SubmitBatch, then its Abort, onto `messages`.
fn flush(
    messages: &mut Vec<Frames>,
    requests: &mut Vec<pb::RequestSpec>,
    prompts: &mut Vec<Vec<u8>>,
    aborts: &mut Vec<u64>,
) {
    if !requests.is_empty() {
        let body = Body::Submit(pb::SubmitBatch {
            requests: std::mem::take(requests),
        });
        let frames = wire::encode(body, std::mem::take(prompts));
        messages.push(Frames { frames });
    }
    if !aborts.is_empty() {
        let body = Body::Abort(pb::Abort {
            seq_ids: std::mem::take(aborts),
        });
        let frames = wire::encode(body, Vec::new());
        messages.push(Frames { frames });
    }
}

/// The reader: hands each message's outputs to their requests, aborting those that
/// overflow, until the call ends or sends one it cannot read; then fails the
/// requests still open. Once the Session is `released`, it ends the call.
async fn read(
    mut inbound: Streaming<Frames>,
    routes: Arc<Mutex<Routes>>,
    aborts: mpsc::WeakUnboundedSender<Command>,
    url: String,
    mut released: oneshot::Receiver<()>,
) {
    let status = loop {
        let message = tokio::select! {
            message = inbound.message() => message,
            // No request is left to read for; ATOM aborts the call's requests.
            _ = &mut released => return,
        };
        match message {
            Ok(Some(message)) => {
                let routed = Inbound::decode(message.frames).and_then(|m| routes.lock().route(&m));
                match routed {
                    Ok(overflowed) => {
                        let commands = aborts.upgrade();
                        for seq_id in overflowed {
                            commands.as_ref().map(|c| c.send(Command::Abort(seq_id)));
                        }
                    }
                    Err(e) => break Status::internal(format!("ATOM Session: {e}")),
                }
            }
            Ok(None) => break Status::unavailable("the ATOM Session ended"),
            Err(status) => break status,
        }
    };
    warn!(url = %url, %status, "ATOM Session failed");
    routes.lock().fail(&status);
}

struct Routes {
    /// Requests by seq_id, each from its submit to its last message.
    open: HashMap<u64, Route>,
    next_seq_id: u64,
    /// The call has ended; no request is routed after.
    closed: bool,
    /// Requests queued and not yet taken by the writer, by ticket, and their bytes.
    unsent: HashMap<u64, Unsent>,
    unsent_bytes: usize,
    /// UNSENT_BYTES but in tests.
    unsent_budget: usize,
    next_ticket: u64,
}

impl Default for Routes {
    fn default() -> Self {
        Self {
            open: HashMap::new(),
            next_seq_id: 0,
            closed: false,
            unsent: HashMap::new(),
            unsent_bytes: 0,
            unsent_budget: UNSENT_BYTES,
            next_ticket: 0,
        }
    }
}

/// A queued request's prompt, gone once its stream drops, and the bytes it holds of the
/// budget, held until the writer takes its Queued with its spec.
struct Unsent {
    prompt: Option<Vec<u32>>,
    bytes: usize,
}

/// Where a request's outputs go, and what its Complete reports.
struct Route {
    tx: mpsc::Sender<Output>,
    /// Its stream left ROUTE_BUFFER outputs unread: it has its error, and is aborted.
    overflowed: bool,
    stream: bool,
    prompt_tokens: u32,
    completion_tokens: u32,
    /// The last nonzero prefix-cache hit, as ATOM's HTTP routes keep it.
    cached_tokens: u32,
    /// A request that does not stream: its ids so far, which its Complete carries.
    ids: Vec<u32>,
}

impl Route {
    /// Its last output; an overflowed request has its error already.
    fn send(&self, response: Response) {
        // A dropped stream has sent its Abort and reads nothing more.
        if !self.overflowed {
            let _ = self.tx.try_send(Ok(GenerateResponse {
                response: Some(response),
            }));
        }
    }

    /// One step's chunk; with one slot left, the request's error instead, the last
    /// slot kept for it. True when the request is to be aborted.
    fn stream(&mut self, chunk: GenerateStreamChunk) -> bool {
        if self.overflowed || self.tx.is_closed() {
            return false;
        }
        if self.tx.capacity() > 1 {
            self.send(Response::Chunk(chunk));
            return false;
        }
        self.overflowed = true;
        let unread = format!("the request left {ROUTE_BUFFER} outputs unread");
        let _ = self.tx.try_send(Err(Status::resource_exhausted(unread)));
        true
    }
}

impl Routes {
    /// The next seq_id below 2^24, in turn, that no open request holds.
    fn free_seq_id(&mut self) -> Option<u64> {
        let first = self.next_seq_id;
        let seq_id = (first..first + SEQ_IDS)
            .map(|id| id % SEQ_IDS)
            .find(|id| !self.open.contains_key(id))?;
        self.next_seq_id = (seq_id + 1) % SEQ_IDS;
        Some(seq_id)
    }

    /// Hands `message`'s outputs to their requests; the seq_ids of those it overflows,
    /// to abort. An Err fails the Session.
    fn route(&mut self, message: &Inbound<Vec<u8>>) -> Result<Vec<u64>, String> {
        match &message.body {
            Body::Stream(batch) => self.step(batch),
            Body::Completions(completions) => {
                // Every item is read before any request ends. An item carries no ids:
                // its request's StreamBatch entries did.
                let mut read = Vec::with_capacity(completions.items.len());
                for item in &completions.items {
                    let kv = item.kv_transfer_params_output.as_ref();
                    read.push((item, kv.map(json).transpose()?));
                }
                for (item, kv) in read {
                    let Some(mut route) = self.open.remove(&item.seq_id) else {
                        continue;
                    };
                    let output_ids = std::mem::take(&mut route.ids);
                    let (ttft_s, tpot_s, latency_s) = timing(item, route.completion_tokens);
                    route.send(Response::Complete(GenerateComplete {
                        completion_tokens: route.completion_tokens,
                        output_ids,
                        finish_reason: item.leave_reason.clone(),
                        prompt_tokens: item.num_prompt_tokens,
                        cached_tokens: route.cached_tokens,
                        kv_transfer_params_json: kv.map(|kv| kv.to_string()),
                        ttft_s,
                        tpot_s,
                        latency_s,
                        ..Default::default()
                    }));
                }
                Ok(Vec::new())
            }
            Body::Rejected(rejected) => {
                for item in &rejected.items {
                    if let Some(route) = self.open.remove(&item.seq_id) {
                        route.send(Response::Error(GenerateError {
                            message: item.reason.clone(),
                            http_status: http_status(item.code()),
                        }));
                    }
                }
                Ok(Vec::new())
            }
            _ => Err("a Session server sends Ready first, then only request outputs".into()),
        }
    }

    /// One engine step's entries: each request's ids, as one chunk when it streams,
    /// else kept for its Complete; the seq_ids of the streams it overflows.
    fn step(&mut self, batch: &pb::StreamBatch) -> Result<Vec<u64>, String> {
        let entries = batch.seq_ids.len();
        let counted: usize = batch.token_counts.iter().map(|&n| n as usize).sum();
        if batch.token_counts.len() != entries
            || batch.num_cached_tokens.len() != entries
            || counted != batch.tokens.len()
        {
            return Err("StreamBatch columns disagree in length".into());
        }
        let (mut start, mut overflowed) = (0, Vec::new());
        let columns = batch.token_counts.iter().zip(&batch.num_cached_tokens);
        for (seq_id, (&count, &cached)) in batch.seq_ids.iter().zip(columns) {
            let tokens = &batch.tokens[start..start + count as usize];
            start += count as usize;
            let Some(route) = self.open.get_mut(seq_id) else {
                continue;
            };
            route.completion_tokens += count;
            if cached > 0 {
                route.cached_tokens = cached as u32;
            }
            let token_ids = tokens.iter().map(|&id| id as u32);
            if route.stream {
                let chunk = GenerateStreamChunk {
                    token_ids: token_ids.collect(),
                    prompt_tokens: route.prompt_tokens,
                    completion_tokens: route.completion_tokens,
                    cached_tokens: route.cached_tokens,
                };
                if route.stream(chunk) {
                    overflowed.push(*seq_id);
                }
            } else if route.tx.is_closed() {
                // A dropped request's ids, which its Abort ends, are read by no one.
                route.ids = Vec::new();
            } else {
                route.ids.extend(token_ids);
            }
        }
        Ok(overflowed)
    }

    /// `commands` with each Queued's prompt taken and its bytes released; one whose
    /// prompt is gone is dropped.
    fn take_unsent(&mut self, commands: Vec<Command>) -> Vec<Command> {
        let take = |command| match command {
            Command::Queued(spec, ticket) => {
                let unsent = self.unsent.remove(&ticket)?;
                self.unsent_bytes -= unsent.bytes;
                Some(Command::Submit(spec, unsent.prompt?))
            }
            command => Some(command),
        };
        commands.into_iter().filter_map(take).collect()
    }

    /// Ends every open request with `status`; the Session takes no request after.
    fn fail(&mut self, status: &Status) {
        self.closed = true;
        self.unsent.clear();
        self.unsent_bytes = 0;
        for (_, route) in self.open.drain() {
            if !route.overflowed {
                let _ = route.tx.try_send(Err(status.clone()));
            }
        }
    }
}

/// A Completion's TTFT, TPOT and latency in seconds from its times, `tokens` long; as
/// Generate reports them, TTFT is 0 without a first token and TPOT without a second.
fn timing(item: &pb::Completion, tokens: u32) -> (f64, f64, f64) {
    let latency = item.leave_time - item.arrive_time;
    if item.first_token_time == 0.0 {
        return (0.0, 0.0, latency);
    }
    let ttft = item.first_token_time - item.arrive_time;
    let tpot = match tokens {
        0 | 1 => 0.0,
        n => (item.leave_time - item.first_token_time) / f64::from(n - 1),
    };
    (ttft, tpot, latency)
}

/// A Rejected code as the HTTP status ATOM's HTTP routes answer that refusal with.
fn http_status(code: Code) -> u32 {
    match code {
        Code::Invalid => 400,
        Code::Overloaded => 503,
        Code::Unspecified | Code::Internal => 500,
    }
}

/// A JSON-shaped engine_v1 Value, as ATOM's frames codec reads one.
fn json(value: &pb::Value) -> Result<Json, String> {
    Ok(match &value.kind {
        Some(Kind::NullValue(_)) => Json::Null,
        Some(Kind::BoolValue(b)) => Json::Bool(*b),
        Some(Kind::IntValue(i)) => Json::from(*i),
        Some(Kind::FloatValue(f)) => Json::from(*f),
        Some(Kind::StringValue(s)) => Json::String(s.clone()),
        Some(Kind::ListValue(list)) => list.items.iter().map(json).collect::<Result<_, _>>()?,
        Some(Kind::MapValue(map)) => {
            let mut object = serde_json::Map::new();
            for entry in &map.entries {
                let key = entry.key.as_ref().and_then(|key| key.kind.as_ref());
                let (Some(Kind::StringValue(key)), Some(value)) = (key, &entry.value) else {
                    return Err("a map entry needs a string key and a value".into());
                };
                object.insert(key.clone(), json(value)?);
            }
            Json::Object(object)
        }
        _ => return Err("a Value that is not JSON-shaped".into()),
    })
}

/// One request's outputs on its Session: the messages Generate sends it.
/// Dropped before its last message, it aborts the request.
pub(crate) struct SessionStream {
    seq_id: u64,
    /// Its ticket in `Routes::unsent` until the writer takes it.
    ticket: u64,
    rx: mpsc::Receiver<Output>,
    session: Session,
}

impl Stream for SessionStream {
    type Item = Output;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Output>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for SessionStream {
    fn drop(&mut self) {
        // Only its own route: once it has ended, a later request may hold its seq_id.
        self.rx.close();
        let mut routes = self.session.routes.lock();
        // Never sent: its prompt is freed and its seq_id released, with nothing to abort;
        // its spec stays queued, so its bytes stay held until the writer discards it.
        if let Some(unsent) = routes.unsent.get_mut(&self.ticket) {
            unsent.prompt = None;
            routes.open.remove(&self.seq_id);
            return;
        }
        if routes
            .open
            .get(&self.seq_id)
            .is_some_and(|route| route.tx.is_closed() && !route.overflowed)
        {
            let _ = self.session.commands.send(Command::Abort(self.seq_id));
        }
    }
}

#[cfg(test)]
#[path = "session_test.rs"]
mod tests;
