//! One EngineCore's sockets, bound and handshaken as the EngineCore expects
//! (`EngineCore.process_input_sockets`): this side binds, the Core's DEALERs
//! send one empty frame each to name themselves, and its first output is
//! `Ready`. Knows messages, not what they mean.

use std::time::Duration;

use parking_lot::Mutex;
use tracing::info;

use super::{
    spec::CoreSpec,
    wire::{Inbound, PROTOCOL_VERSION},
};

/// How often a blocked wait rechecks that the Core process still exists.
const LIVENESS_POLL: Duration = Duration::from_secs(1);

/// Asked between polls while startup waits on a Core: true gives up (the
/// launcher was interrupted).
pub type Interrupted<'a> = &'a dyn Fn() -> bool;
/// `make_zmq_socket`'s buffer size on a large host.
const SOCKET_BUFFER_BYTES: i32 = 512 * 1024 * 1024;

pub struct CoreLink {
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
        interrupted: Interrupted<'_>,
    ) -> Result<(Self, zmq::Socket, Option<u64>), String> {
        let output = bind(ctx, zmq::PULL, &spec.output)?;
        let input = Peer::accept(bind(ctx, zmq::ROUTER, &spec.input)?, spec.pid, interrupted)?;
        let control = Peer::accept(
            bind(ctx, zmq::ROUTER, &spec.control)?,
            spec.pid,
            interrupted,
        )?;
        wait_readable(&output, spec.pid, interrupted)?;
        let ready = match Inbound::decode(&output.recv_msg(0).map_err(zmq_err)?)? {
            Inbound::Ready(ready) => ready,
            Inbound::Stopped => return Err("stopped before it was ready".into()),
            _ => return Err("its first output was not Ready".into()),
        };
        if ready.protocol_version != PROTOCOL_VERSION {
            return Err(format!(
                "it speaks protocol {}, this client {PROTOCOL_VERSION}",
                ready.protocol_version
            ));
        }
        info!(dp_rank = spec.dp_rank, "EngineCore is ready");
        let link = Self {
            spec,
            input: Mutex::new(input),
            control: Mutex::new(control),
        };
        Ok((link, output, ready.max_pool_tokens))
    }

    /// Submit only.
    pub fn send_input(&self, message: Vec<u8>) -> Result<(), String> {
        self.input.lock().send(message)
    }

    /// Abort and Shutdown: every client -> Core message except Submit. They
    /// travel apart from input so that none of them queues behind a large
    /// Submit.
    pub fn send_control(&self, message: Vec<u8>) -> Result<(), String> {
        self.control.lock().send(message)
    }
}

impl Peer {
    /// The Core's first frame on a ROUTER carries its routing identity.
    fn accept(socket: zmq::Socket, pid: u32, interrupted: Interrupted<'_>) -> Result<Self, String> {
        wait_readable(&socket, pid, interrupted)?;
        let identity = recv_frames(&socket)
            .map_err(zmq_err)?
            .swap_remove(0)
            .to_vec();
        Ok(Self { socket, identity })
    }

    fn send(&self, message: Vec<u8>) -> Result<(), String> {
        self.socket
            .send_multipart([self.identity.clone(), message], 0)
            .map_err(zmq_err)
    }
}

/// One whole multipart, frames kept as ZMQ messages.
pub(crate) fn recv_frames(socket: &zmq::Socket) -> zmq::Result<Vec<zmq::Message>> {
    let mut frames = vec![socket.recv_msg(0)?];
    while socket.get_rcvmore()? {
        frames.push(socket.recv_msg(0)?);
    }
    Ok(frames)
}

/// Whether `pid` is still a running process (not gone, not a zombie). Reads
/// `/proc`, so Linux only, as the EngineCores are.
pub(crate) fn process_alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|stat| {
            let state = stat.rsplit_once(')')?.1.trim_start().chars().next()?;
            Some(!matches!(state, 'Z' | 'X'))
        })
        .unwrap_or(false)
}

/// Block until `socket` has a message, failing if Core `pid` dies first or
/// the wait is interrupted.
fn wait_readable(
    socket: &zmq::Socket,
    pid: u32,
    interrupted: Interrupted<'_>,
) -> Result<(), String> {
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
        if interrupted() {
            return Err("interrupted while waiting for the EngineCores".into());
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
