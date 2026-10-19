//! One EngineCore's sockets, bound and handshaken as the SOCKETS section of
//! `engine_v1.proto` prescribes. Knows frames, not what they mean.

use std::time::Duration;

use parking_lot::Mutex;
use tracing::info;

use super::{
    spec::CoreSpec,
    wire::{pb, Inbound},
};

/// How often a blocked wait rechecks that the Core process still exists.
const LIVENESS_POLL: Duration = Duration::from_secs(1);
/// `make_zmq_socket`'s buffer size on a large host.
const SOCKET_BUFFER_BYTES: i32 = 512 * 1024 * 1024;

pub(super) struct CoreLink {
    pub spec: CoreSpec,
    input: Mutex<Peer>,
    control: Mutex<Peer>,
}

/// A ROUTER socket and the one Core identity it routes to.
struct Peer {
    socket: zmq::Socket,
    identity: Vec<u8>,
}

impl CoreLink {
    /// Bind the Core's sockets, learn its identities and wait for its `Ready`.
    /// Returns the link, the output socket and `Ready.max_pool_tokens`.
    pub fn connect(
        ctx: &zmq::Context,
        spec: CoreSpec,
    ) -> Result<(Self, zmq::Socket, Option<u64>), String> {
        let output = bind(ctx, zmq::PULL, &spec.output)?;
        let input = Peer::accept(bind(ctx, zmq::ROUTER, &spec.input)?, spec.pid)?;
        let control = Peer::accept(bind(ctx, zmq::ROUTER, &spec.control)?, spec.pid)?;
        wait_readable(&output, spec.pid)?;
        let ready = match Inbound::decode(recv_frames(&output).map_err(zmq_err)?)?.body {
            pb::envelope::Body::Ready(ready) => ready,
            pb::envelope::Body::Stopped(_) => return Err("stopped before it was ready".into()),
            _ => return Err("its first output was not Ready".into()),
        };
        info!(dp_rank = spec.dp_rank, "EngineCore is ready");
        let link = Self {
            spec,
            input: Mutex::new(input),
            control: Mutex::new(control),
        };
        Ok((link, output, ready.max_pool_tokens))
    }

    /// SubmitBatch only, per the protocol.
    pub fn send_input(&self, frames: Vec<Vec<u8>>) -> Result<(), String> {
        self.input.lock().send(frames)
    }

    /// Abort and Shutdown.
    pub fn send_control(&self, frames: Vec<Vec<u8>>) -> Result<(), String> {
        self.control.lock().send(frames)
    }
}

impl Peer {
    /// The Core's first frame on a ROUTER carries its routing identity.
    fn accept(socket: zmq::Socket, pid: u32) -> Result<Self, String> {
        wait_readable(&socket, pid)?;
        let identity = recv_frames(&socket)
            .map_err(zmq_err)?
            .swap_remove(0)
            .to_vec();
        Ok(Self { socket, identity })
    }

    fn send(&self, frames: Vec<Vec<u8>>) -> Result<(), String> {
        let message = std::iter::once(self.identity.clone()).chain(frames);
        self.socket.send_multipart(message, 0).map_err(zmq_err)
    }
}

/// One whole multipart, frames kept as ZMQ messages.
pub(super) fn recv_frames(socket: &zmq::Socket) -> zmq::Result<Vec<zmq::Message>> {
    let mut frames = vec![socket.recv_msg(0)?];
    while socket.get_rcvmore()? {
        frames.push(socket.recv_msg(0)?);
    }
    Ok(frames)
}

/// Whether `pid` is still a running process (not gone, not a zombie).
pub(super) fn process_alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|stat| {
            let state = stat.rsplit_once(')')?.1.trim_start().chars().next()?;
            Some(!matches!(state, 'Z' | 'X'))
        })
        .unwrap_or(false)
}

/// Block until `socket` has a message, failing if Core `pid` dies first.
fn wait_readable(socket: &zmq::Socket, pid: u32) -> Result<(), String> {
    loop {
        if socket
            .poll(zmq::POLLIN, LIVENESS_POLL.as_millis() as i64)
            .map_err(zmq_err)?
            > 0
        {
            return Ok(());
        }
        if !process_alive(pid) {
            return Err(format!("EngineCore process {pid} exited"));
        }
    }
}

fn bind(ctx: &zmq::Context, kind: zmq::SocketType, address: &str) -> Result<zmq::Socket, String> {
    let socket = ctx.socket(kind).map_err(zmq_err)?;
    socket.set_rcvhwm(0).map_err(zmq_err)?;
    socket.set_sndhwm(0).map_err(zmq_err)?;
    socket.set_rcvbuf(SOCKET_BUFFER_BYTES).map_err(zmq_err)?;
    socket.set_sndbuf(SOCKET_BUFFER_BYTES).map_err(zmq_err)?;
    socket.set_linger(0).map_err(zmq_err)?;
    socket
        .bind(address)
        .map_err(|e| format!("bind {address}: {e}"))?;
    Ok(socket)
}

fn zmq_err(e: zmq::Error) -> String {
    e.to_string()
}
