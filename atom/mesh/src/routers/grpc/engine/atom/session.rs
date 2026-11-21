//! The Session to an ATOM instance: one `AtomEngine.Session` call carries every
//! request to it as engine_v1 messages, and each request's outputs come back by seq_id.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;

use futures::{stream, Stream, StreamExt};
use http::StatusCode;
use parking_lot::Mutex;
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
use crate::core::Worker;
use crate::routers::token_handle::engine_error::EngineError;

/// A Session's seq ids stay below 2^24, the window ATOM maps each Session's ids into.
const SEQ_IDS: u64 = 1 << 24;

type Output = Result<GenerateResponse, Status>;

/// Each ATOM instance's Session (base URL, all its ranks), opened by the first
/// request to it and again by the first request after it has failed.
#[derive(Default)]
pub(crate) struct AtomSessions {
    slots: Mutex<HashMap<String, Arc<tokio::sync::Mutex<Option<Session>>>>>,
}

impl AtomSessions {
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

    async fn session(&self, worker: &Arc<dyn Worker>) -> Result<Session, EngineError> {
        let url = worker.base_url().trim_end_matches('/');
        let slot = self
            .slots
            .lock()
            .entry(url.to_string())
            .or_default()
            .clone();
        let mut slot = slot.lock().await;
        if let Some(session) = slot.as_ref().filter(|session| session.is_open()) {
            return Ok(session.clone());
        }
        Ok(slot.insert(Session::open(url).await?).clone())
    }
}

/// One Session call, shared by its clones and its requests' streams; the
/// client's half of the call ends once all of them have dropped.
#[derive(Clone)]
struct Session {
    routes: Arc<Mutex<Routes>>,
    commands: mpsc::UnboundedSender<Command>,
}

enum Command {
    Submit(Box<pb::RequestSpec>, Vec<u32>),
    Abort(u64),
}

impl Session {
    /// Opens the call on a connection and thread of its own, off the router's runtime;
    /// waits for ATOM's Ready, refusing one of another protocol version.
    async fn open(url: &str) -> Result<Self, EngineError> {
        let (commands, queued) = mpsc::unbounded_channel();
        let routes = Arc::new(Mutex::new(Routes::default()));
        let (opened, ready) = oneshot::channel();
        let (url, read_into) = (url.to_string(), routes.clone());
        thread::Builder::new()
            .name("atom-session".into())
            .spawn(move || run(url, queued, read_into, opened))
            .map_err(|e| EngineError::Transport(Status::internal(e.to_string())))?;
        let gone = |_| EngineError::Transport(Status::internal("the Session thread ended"));
        ready.await.map_err(gone)??;
        Ok(Self { routes, commands })
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
        let seq_id = routes.free_seq_id().ok_or_else(|| EngineError::Http {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "every seq_id of the ATOM Session is in use".to_string(),
        })?;
        spec.seq_id = seq_id;
        let (tx, rx) = mpsc::unbounded_channel();
        let route = Route {
            tx,
            stream,
            prompt_tokens: prompt.len() as u32,
            completion_tokens: 0,
            cached_tokens: 0,
            ids: Vec::new(),
        };
        let command = Command::Submit(Box::new(spec), prompt);
        if routes.closed || self.commands.send(command).is_err() {
            return Err(EngineError::Transport(Status::unavailable(
                "the ATOM Session has ended",
            )));
        }
        routes.open.insert(seq_id, route);
        Ok(SessionStream {
            seq_id,
            rx,
            session: self.clone(),
        })
    }
}

/// The Session's thread: a runtime of its own, which connects, opens the call and reads
/// it until it ends; `opened` gets how the opening went.
fn run(
    url: String,
    queued: mpsc::UnboundedReceiver<Command>,
    routes: Arc<Mutex<Routes>>,
    opened: oneshot::Sender<Result<(), EngineError>>,
) {
    let runtime = match runtime::Builder::new_current_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(e) => {
            let _ = opened.send(Err(EngineError::Transport(Status::internal(e.to_string()))));
            return;
        }
    };
    runtime.block_on(async move {
        match start(&url, queued).await {
            Ok((_client, inbound)) => {
                let _ = opened.send(Ok(()));
                read(inbound, routes, url).await;
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
) -> Result<(AtomClient, Streaming<Frames>), EngineError> {
    let client = AtomClient::connect(url)
        .await
        .map_err(|_| EngineError::ConnectionAcquireFailed(url.to_string()))?;
    let mut inbound = client
        .session(writer(queued))
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

/// The writer: what is queued when it runs goes out as one SubmitBatch, then
/// one Abort, so an Abort never goes before its request.
fn writer(queued: mpsc::UnboundedReceiver<Command>) -> impl Stream<Item = Frames> + Send + 'static {
    stream::unfold(queued, |mut queued| async move {
        let mut commands = vec![queued.recv().await?];
        while let Ok(command) = queued.try_recv() {
            commands.push(command);
        }
        Some((encode(commands), queued))
    })
    .flat_map(stream::iter)
}

fn encode(commands: Vec<Command>) -> Vec<Frames> {
    let (mut requests, mut prompts, mut aborts) = (Vec::new(), Vec::new(), Vec::new());
    for command in commands {
        match command {
            Command::Submit(mut spec, prompt) => {
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
        }
    }
    let mut messages = Vec::new();
    if !requests.is_empty() {
        let body = Body::Submit(pb::SubmitBatch { requests });
        messages.push(wire::encode(body, prompts));
    }
    if !aborts.is_empty() {
        let body = Body::Abort(pb::Abort { seq_ids: aborts });
        messages.push(wire::encode(body, Vec::new()));
    }
    messages
        .into_iter()
        .map(|frames| Frames { frames })
        .collect()
}

/// The reader: hands each message's outputs to their requests until the call
/// ends or sends one it cannot read, then fails the requests still open.
async fn read(mut inbound: Streaming<Frames>, routes: Arc<Mutex<Routes>>, url: String) {
    let status = loop {
        match inbound.message().await {
            Ok(Some(message)) => {
                let routed = Inbound::decode(message.frames).and_then(|m| routes.lock().route(&m));
                if let Err(e) = routed {
                    break Status::internal(format!("ATOM Session: {e}"));
                }
            }
            Ok(None) => break Status::unavailable("the ATOM Session ended"),
            Err(status) => break status,
        }
    };
    warn!(url = %url, %status, "ATOM Session failed");
    routes.lock().fail(&status);
}

#[derive(Default)]
struct Routes {
    /// Requests by seq_id, each from its submit to its last message.
    open: HashMap<u64, Route>,
    next_seq_id: u64,
    /// The call has ended; no request is routed after.
    closed: bool,
}

/// Where a request's outputs go, and what its Complete reports.
struct Route {
    tx: mpsc::UnboundedSender<Output>,
    stream: bool,
    prompt_tokens: u32,
    completion_tokens: u32,
    /// The last nonzero prefix-cache hit, as ATOM's HTTP routes keep it.
    cached_tokens: u32,
    /// A request that does not stream: its ids so far, which its Complete carries.
    ids: Vec<u32>,
}

impl Route {
    fn send(&self, response: Response) {
        // A dropped stream has sent its Abort and reads nothing more.
        let _ = self.tx.send(Ok(GenerateResponse {
            response: Some(response),
        }));
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

    /// Hands `message`'s outputs to their requests; an Err fails the Session.
    fn route(&mut self, message: &Inbound<Vec<u8>>) -> Result<(), String> {
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
                    route.send(Response::Complete(GenerateComplete {
                        completion_tokens: route.completion_tokens,
                        output_ids,
                        finish_reason: item.leave_reason.clone(),
                        prompt_tokens: item.num_prompt_tokens,
                        cached_tokens: route.cached_tokens,
                        kv_transfer_params_json: kv.map(|kv| kv.to_string()),
                    }));
                }
                Ok(())
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
                Ok(())
            }
            _ => Err("a Session server sends Ready first, then only request outputs".into()),
        }
    }

    /// One engine step's entries: each request's ids, as one chunk when it streams,
    /// else kept for its Complete.
    fn step(&mut self, batch: &pb::StreamBatch) -> Result<(), String> {
        let entries = batch.seq_ids.len();
        let counted: usize = batch.token_counts.iter().map(|&n| n as usize).sum();
        if batch.token_counts.len() != entries
            || batch.num_cached_tokens.len() != entries
            || counted != batch.tokens.len()
        {
            return Err("StreamBatch columns disagree in length".into());
        }
        let mut start = 0;
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
                route.send(Response::Chunk(GenerateStreamChunk {
                    token_ids: token_ids.collect(),
                    prompt_tokens: route.prompt_tokens,
                    completion_tokens: route.completion_tokens,
                    cached_tokens: route.cached_tokens,
                }));
            } else {
                route.ids.extend(token_ids);
            }
        }
        Ok(())
    }

    /// Ends every open request with `status`; the Session takes no request after.
    fn fail(&mut self, status: &Status) {
        self.closed = true;
        for (_, route) in self.open.drain() {
            let _ = route.tx.send(Err(status.clone()));
        }
    }
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
    rx: mpsc::UnboundedReceiver<Output>,
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
        // Its route lasts until its last message, so no other request holds its seq_id yet.
        let routes = self.session.routes.lock();
        if routes.open.contains_key(&self.seq_id) {
            let _ = self.session.commands.send(Command::Abort(self.seq_id));
        }
    }
}

#[cfg(test)]
#[path = "session_test.rs"]
mod tests;
