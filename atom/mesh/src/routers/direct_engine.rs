//! Multiplexed Rust client for one EngineCore IPC endpoint.
//!
//! A worker owns each ZMQ DEALER socket for its full lifetime. It serializes
//! socket access and fans received frames out to request-local Tokio channels,
//! mirroring Python CoreManager's one output thread per DP rank.

use std::{
    collections::HashMap,
    io,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread,
};

use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{
    atom_standalone::EngineCoreIpcEndpoint,
    token_handle::{
        engine_error::EngineError,
        token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta},
        token_handle::{TokenHandle, TokenSource},
    },
};

pub const DIRECT_ENGINE_PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Serialize)]
pub struct DirectSamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub ignore_eos: bool,
    pub n: u32,
    pub stop_strings: Option<Vec<String>>,
}

#[derive(Clone, Serialize)]
pub struct DirectEngineSubmit {
    pub version: u32,
    #[serde(rename = "type")]
    pub frame_type: &'static str,
    pub request_id: String,
    pub token_ids: Vec<u32>,
    pub sampling: DirectSamplingParams,
    pub stop_token_sequences: Vec<Vec<u32>>,
    pub kv_transfer_params: Option<serde_json::Value>,
    pub num_draft_tokens: u32,
    pub n: u32,
    pub data_parallel_rank: Option<usize>,
}

impl DirectEngineSubmit {
    pub fn new(
        request_id: String,
        token_ids: Vec<u32>,
        sampling: DirectSamplingParams,
        stop_token_sequences: Vec<Vec<u32>>,
    ) -> Self {
        Self {
            version: DIRECT_ENGINE_PROTOCOL_VERSION,
            frame_type: "submit",
            request_id,
            token_ids,
            n: sampling.n,
            sampling,
            stop_token_sequences,
            kv_transfer_params: None,
            num_draft_tokens: 0,
            data_parallel_rank: None,
        }
    }
}

#[derive(Serialize)]
struct AbortFrame<'a> {
    version: u32,
    #[serde(rename = "type")]
    frame_type: &'static str,
    request_id: &'a str,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Frame {
    #[serde(rename = "accepted")]
    Accepted {
        version: u32,
        request_id: String,
        seq_ids: Vec<u64>,
    },
    #[serde(rename = "token")]
    Token {
        seq_id: u64,
        token_ids: Vec<u32>,
        finished: bool,
        finish_reason: Option<String>,
        num_cached_tokens: u32,
    },
    #[serde(rename = "error")]
    Error {
        request_id: Option<String>,
        message: String,
    },
}

enum Command {
    Submit {
        request: DirectEngineSubmit,
        reply: oneshot::Sender<io::Result<DirectEngineStream>>,
    },
    Abort {
        request_id: String,
    },
}

#[derive(Clone)]
pub struct DirectEngineClient {
    endpoint: EngineCoreIpcEndpoint,
    commands: Sender<Command>,
}

impl DirectEngineClient {
    pub fn new(endpoint: EngineCoreIpcEndpoint) -> Self {
        let (commands, receiver) = mpsc::channel();
        let worker_endpoint = endpoint.clone();
        let worker_commands = commands.clone();
        thread::Builder::new()
            .name(format!("engine-core-ipc-dp{}", endpoint.dp_rank))
            .spawn(move || run_worker(worker_endpoint, receiver, worker_commands))
            .expect("failed to start EngineCore IPC worker");
        Self { endpoint, commands }
    }

    pub async fn submit(&self, request: &DirectEngineSubmit) -> io::Result<DirectEngineStream> {
        if self.endpoint.protocol_version != DIRECT_ENGINE_PROTOCOL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "EngineCore IPC protocol mismatch",
            ));
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands
            .send(Command::Submit {
                request: request.clone(),
                reply: reply_tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "IPC worker exited"))?;
        reply_rx
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "IPC worker exited"))?
    }
}

pub struct DirectEngineStream {
    receiver: UnboundedReceiverStream<Result<TokenChunk, EngineError>>,
    request_id: String,
    commands: Sender<Command>,
    completed: Arc<AtomicBool>,
}

struct Source {
    stream: DirectEngineStream,
}

impl Stream for Source {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.stream.receiver).poll_next(cx)
    }
}

impl TokenSource for Source {
    fn mark_completed(&mut self) {
        self.stream.completed.store(true, Ordering::Release);
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        if !self.stream.completed.load(Ordering::Acquire) {
            let _ = self.commands().send(Command::Abort {
                request_id: self.stream.request_id.clone(),
            });
        }
    }
}

impl Source {
    fn commands(&self) -> &Sender<Command> {
        &self.stream.commands
    }
}

pub fn into_token_handle(stream: DirectEngineStream) -> TokenHandle {
    TokenHandle::new(Source { stream })
}

struct RequestRoute {
    sender: tokio_mpsc::UnboundedSender<Result<TokenChunk, EngineError>>,
    request_id: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    generated_token_ids: Vec<u32>,
    completed: Arc<AtomicBool>,
}

fn run_worker(
    endpoint: EngineCoreIpcEndpoint,
    commands: Receiver<Command>,
    worker_commands: Sender<Command>,
) {
    let context = zmq::Context::new();
    let socket = match context.socket(zmq::DEALER).and_then(|socket| {
        socket.connect(&endpoint.address)?;
        socket.set_rcvtimeo(25)?;
        Ok(socket)
    }) {
        Ok(socket) => socket,
        Err(error) => {
            let message = error.to_string();
            while let Ok(command) = commands.recv() {
                if let Command::Submit { reply, .. } = command {
                    let _ = reply.send(Err(io::Error::new(
                        io::ErrorKind::ConnectionRefused,
                        message.clone(),
                    )));
                }
            }
            return;
        }
    };

    let mut pending = HashMap::new();
    let mut routes: HashMap<u64, RequestRoute> = HashMap::new();
    loop {
        while let Ok(command) = commands.try_recv() {
            match command {
                Command::Submit { request, reply } => {
                    let request_id = request.request_id.clone();
                    match send_frame(&socket, &request) {
                        Ok(()) => {
                            pending.insert(request_id, (request.token_ids.len() as u32, reply));
                        }
                        Err(error) => {
                            let _ = reply.send(Err(error));
                        }
                    }
                }
                Command::Abort { request_id } => {
                    let _ = send_frame(
                        &socket,
                        &AbortFrame {
                            version: DIRECT_ENGINE_PROTOCOL_VERSION,
                            frame_type: "abort",
                            request_id: &request_id,
                        },
                    );
                }
            }
        }

        match receive_frame(&socket) {
            Ok(frame) => route_frame(frame, &mut pending, &mut routes, &worker_commands),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
            Err(error) => {
                for (_, (_, reply)) in pending.drain() {
                    let _ = reply.send(Err(io::Error::new(error.kind(), error.to_string())));
                }
                for (_, route) in routes.drain() {
                    let _ = route
                        .sender
                        .send(Err(EngineError::DecodeError(error.to_string())));
                }
                return;
            }
        }
    }
}

fn route_frame(
    frame: Frame,
    pending: &mut HashMap<String, (u32, oneshot::Sender<io::Result<DirectEngineStream>>)>,
    routes: &mut HashMap<u64, RequestRoute>,
    commands: &Sender<Command>,
) {
    match frame {
        Frame::Accepted {
            version,
            request_id,
            seq_ids,
        } if version == DIRECT_ENGINE_PROTOCOL_VERSION => {
            let Some((prompt_tokens, reply)) = pending.remove(&request_id) else {
                return;
            };
            let (sender, receiver) = tokio_mpsc::unbounded_channel();
            let completed = Arc::new(AtomicBool::new(false));
            for seq_id in seq_ids {
                routes.insert(
                    seq_id,
                    RequestRoute {
                        sender: sender.clone(),
                        request_id: request_id.clone(),
                        prompt_tokens,
                        completion_tokens: 0,
                        generated_token_ids: Vec::new(),
                        completed: Arc::clone(&completed),
                    },
                );
            }
            let _ = reply.send(Ok(DirectEngineStream {
                receiver: UnboundedReceiverStream::new(receiver),
                request_id,
                commands: commands.clone(),
                completed,
            }));
        }
        Frame::Token {
            seq_id,
            token_ids,
            finished,
            finish_reason,
            num_cached_tokens,
        } => {
            let Some(route) = routes.get_mut(&seq_id) else {
                return;
            };
            route.completion_tokens += token_ids.len() as u32;
            route.generated_token_ids.extend_from_slice(&token_ids);
            let chunk = if finished {
                route.completed.store(true, Ordering::Release);
                TokenChunk::Complete {
                    token_ids: std::mem::take(&mut route.generated_token_ids),
                    finish_reason: parse_finish_reason(finish_reason),
                    matched_stop: None,
                    usage: Usage {
                        prompt_tokens: route.prompt_tokens,
                        completion_tokens: route.completion_tokens,
                        total_tokens: route.prompt_tokens + route.completion_tokens,
                    },
                    logprobs: None,
                    input_logprobs: None,
                    meta: WorkerMeta {
                        request_id: route.request_id.clone(),
                        weight_version: None,
                        cached_tokens: num_cached_tokens,
                    },
                }
            } else {
                TokenChunk::Partial {
                    token_ids,
                    logprobs: None,
                }
            };
            let _ = route.sender.send(Ok(chunk));
            if finished {
                routes.remove(&seq_id);
            }
        }
        Frame::Error {
            request_id,
            message,
        } => {
            if let Some(request_id) = request_id {
                if let Some((_, reply)) = pending.remove(&request_id) {
                    let _ = reply.send(Err(io::Error::new(io::ErrorKind::Other, message)));
                }
            }
        }
        Frame::Accepted { .. } => {}
    }
}

fn send_frame<T: Serialize>(socket: &zmq::Socket, value: &T) -> io::Result<()> {
    socket
        .send(
            rmp_serde::to_vec_named(value)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
            0,
        )
        .map_err(zmq_error)
}

fn receive_frame(socket: &zmq::Socket) -> io::Result<Frame> {
    socket.recv_bytes(0).map_err(zmq_error).and_then(|payload| {
        rmp_serde::from_slice(&payload)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    })
}

fn parse_finish_reason(reason: Option<String>) -> FinishReason {
    match reason.as_deref() {
        Some("length") | Some("max_tokens") => FinishReason::Length,
        Some("stop") | None => FinishReason::Stop,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("abort") => FinishReason::Abort,
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

fn zmq_error(error: zmq::Error) -> io::Error {
    if error == zmq::Error::EAGAIN {
        io::Error::new(io::ErrorKind::WouldBlock, error)
    } else {
        io::Error::new(io::ErrorKind::Other, error)
    }
}
