use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use prost::Message;
use thiserror::Error;

use crate::proto::engine::{engine_core_envelope::Payload, EngineCoreEnvelope, ReadySignal};

use super::{EngineCoreEndpointTopology, ENGINE_CORE_WIRE_VERSION};

#[derive(Debug, Error)]
pub enum EngineCoreTransportError {
    #[error("ZeroMQ error for EngineCore dp_rank {dp_rank}: {source}")]
    Zmq {
        dp_rank: usize,
        #[source]
        source: zmq::Error,
    },
    #[error("invalid EngineCore handshake for dp_rank {dp_rank}: {message}")]
    InvalidHandshake { dp_rank: usize, message: String },
    #[error("invalid EngineCore protobuf for dp_rank {dp_rank}: {source}")]
    InvalidProtobuf {
        dp_rank: usize,
        #[source]
        source: prost::DecodeError,
    },
    #[error(
        "unsupported EngineCore wire version {actual} for dp_rank {dp_rank}; expected {expected}"
    )]
    UnsupportedWireVersion {
        dp_rank: usize,
        actual: u32,
        expected: u32,
    },
    #[error("expected READY from EngineCore dp_rank {dp_rank}, received {payload}")]
    ExpectedReady { dp_rank: usize, payload: String },
    #[error("EngineCore dp_rank {dp_rank} has not completed its {channel} handshake")]
    MissingIdentity {
        dp_rank: usize,
        channel: &'static str,
    },
    #[error("unknown EngineCore dp_rank {0}")]
    UnknownRank(usize),
    #[error("timed out waiting for EngineCore dp_rank {dp_rank} during {phase}")]
    StartupTimeout { dp_rank: usize, phase: &'static str },
}

pub(crate) struct EngineCoreRankSockets {
    input: zmq::Socket,
    control: zmq::Socket,
    output: zmq::Socket,
    input_identity: Option<Vec<u8>>,
    control_identity: Option<Vec<u8>>,
}

impl EngineCoreRankSockets {
    pub(crate) fn send_input(
        &self,
        dp_rank: usize,
        envelope: &EngineCoreEnvelope,
    ) -> Result<(), EngineCoreTransportError> {
        let identity =
            self.input_identity
                .as_ref()
                .ok_or(EngineCoreTransportError::MissingIdentity {
                    dp_rank,
                    channel: "input",
                })?;
        EngineCoreTransport::send_router_frame(&self.input, identity, envelope, dp_rank)
    }

    pub(crate) fn send_control(
        &self,
        dp_rank: usize,
        envelope: &EngineCoreEnvelope,
    ) -> Result<(), EngineCoreTransportError> {
        let identity =
            self.control_identity
                .as_ref()
                .ok_or(EngineCoreTransportError::MissingIdentity {
                    dp_rank,
                    channel: "control",
                })?;
        EngineCoreTransport::send_router_frame(&self.control, identity, envelope, dp_rank)
    }

    pub(crate) fn receive_output_nonblocking(
        &self,
        dp_rank: usize,
    ) -> Result<Option<EngineCoreEnvelope>, EngineCoreTransportError> {
        match self.output.recv_bytes(zmq::DONTWAIT) {
            Ok(frame) => EngineCoreTransport::decode_envelope(dp_rank, &frame).map(Some),
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(source) => Err(EngineCoreTransportError::Zmq { dp_rank, source }),
        }
    }
}

pub struct EngineCoreTransport {
    _context: zmq::Context,
    ranks: BTreeMap<usize, EngineCoreRankSockets>,
    startup_timeout: Duration,
}

impl std::fmt::Debug for EngineCoreTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EngineCoreTransport")
            .field("dp_ranks", &self.ranks.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl EngineCoreTransport {
    pub fn bind(
        topology: &EngineCoreEndpointTopology,
        receive_timeout_ms: i32,
    ) -> Result<Self, EngineCoreTransportError> {
        let context = zmq::Context::new();
        let mut ranks = BTreeMap::new();

        for endpoint in &topology.endpoints {
            let input = Self::bind_socket(
                &context,
                zmq::ROUTER,
                &endpoint.input_address,
                endpoint.dp_rank,
                receive_timeout_ms,
            )?;
            let control = Self::bind_socket(
                &context,
                zmq::ROUTER,
                &endpoint.control_address,
                endpoint.dp_rank,
                receive_timeout_ms,
            )?;
            let output = Self::bind_socket(
                &context,
                zmq::PULL,
                &endpoint.output_address,
                endpoint.dp_rank,
                receive_timeout_ms,
            )?;
            ranks.insert(
                endpoint.dp_rank,
                EngineCoreRankSockets {
                    input,
                    control,
                    output,
                    input_identity: None,
                    control_identity: None,
                },
            );
        }

        Ok(Self {
            _context: context,
            ranks,
            startup_timeout: Duration::from_millis(receive_timeout_ms as u64),
        })
    }

    fn bind_socket(
        context: &zmq::Context,
        socket_type: zmq::SocketType,
        address: &str,
        dp_rank: usize,
        receive_timeout_ms: i32,
    ) -> Result<zmq::Socket, EngineCoreTransportError> {
        let socket = context
            .socket(socket_type)
            .map_err(|source| EngineCoreTransportError::Zmq { dp_rank, source })?;
        socket
            .set_linger(0)
            .and_then(|_| socket.set_rcvtimeo(receive_timeout_ms))
            .and_then(|_| socket.set_sndtimeo(receive_timeout_ms.min(5_000)))
            .and_then(|_| socket.bind(address))
            .map_err(|source| EngineCoreTransportError::Zmq { dp_rank, source })?;
        if socket_type == zmq::ROUTER {
            socket
                .set_router_mandatory(true)
                .map_err(|source| EngineCoreTransportError::Zmq { dp_rank, source })?;
        }
        Ok(socket)
    }

    pub fn wait_until_all_connected(&mut self) -> Result<(), EngineCoreTransportError> {
        let deadline = Instant::now() + self.startup_timeout;
        for (&dp_rank, sockets) in &mut self.ranks {
            Self::apply_remaining_timeout(&sockets.input, deadline, dp_rank, "input handshake")?;
            sockets.input_identity =
                Some(Self::receive_identity(&sockets.input, dp_rank, "input")?);
            Self::apply_remaining_timeout(
                &sockets.control,
                deadline,
                dp_rank,
                "control handshake",
            )?;
            sockets.control_identity = Some(Self::receive_identity(
                &sockets.control,
                dp_rank,
                "control",
            )?);
        }
        Ok(())
    }

    fn apply_remaining_timeout(
        socket: &zmq::Socket,
        deadline: Instant,
        dp_rank: usize,
        phase: &'static str,
    ) -> Result<(), EngineCoreTransportError> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(EngineCoreTransportError::StartupTimeout { dp_rank, phase });
        }
        let timeout_ms = remaining.as_millis().clamp(1, i32::MAX as u128) as i32;
        socket
            .set_rcvtimeo(timeout_ms)
            .map_err(|source| EngineCoreTransportError::Zmq { dp_rank, source })
    }

    fn receive_identity(
        socket: &zmq::Socket,
        dp_rank: usize,
        channel: &'static str,
    ) -> Result<Vec<u8>, EngineCoreTransportError> {
        let frames = socket.recv_multipart(0).map_err(|source| {
            if source == zmq::Error::EAGAIN {
                EngineCoreTransportError::StartupTimeout {
                    dp_rank,
                    phase: if channel == "input" {
                        "input handshake"
                    } else {
                        "control handshake"
                    },
                }
            } else {
                EngineCoreTransportError::Zmq { dp_rank, source }
            }
        })?;
        if frames.len() != 2 || !frames[1].is_empty() {
            return Err(EngineCoreTransportError::InvalidHandshake {
                dp_rank,
                message: format!(
                    "{channel} ROUTER expected [identity, empty], got {} frame(s)",
                    frames.len()
                ),
            });
        }
        if frames[0].is_empty() {
            return Err(EngineCoreTransportError::InvalidHandshake {
                dp_rank,
                message: format!("{channel} DEALER identity is empty"),
            });
        }
        Ok(frames[0].clone())
    }

    pub fn wait_until_all_ready(
        &mut self,
    ) -> Result<BTreeMap<usize, Option<i64>>, EngineCoreTransportError> {
        let mut capacities = BTreeMap::new();
        let deadline = Instant::now() + self.startup_timeout;
        for (&dp_rank, sockets) in &mut self.ranks {
            Self::apply_remaining_timeout(&sockets.output, deadline, dp_rank, "READY signal")?;
            let frame = sockets.output.recv_bytes(0).map_err(|source| {
                if source == zmq::Error::EAGAIN {
                    EngineCoreTransportError::StartupTimeout {
                        dp_rank,
                        phase: "READY signal",
                    }
                } else {
                    EngineCoreTransportError::Zmq { dp_rank, source }
                }
            })?;
            let envelope = Self::decode_envelope(dp_rank, &frame)?;
            let ready = match envelope.payload {
                Some(Payload::Ready(ready)) => ready,
                payload => {
                    return Err(EngineCoreTransportError::ExpectedReady {
                        dp_rank,
                        payload: format!("{payload:?}"),
                    });
                }
            };
            capacities.insert(dp_rank, Self::ready_capacity(ready));
        }
        Ok(capacities)
    }

    fn ready_capacity(ready: ReadySignal) -> Option<i64> {
        ready.max_pool_tokens
    }

    pub fn send_input(
        &self,
        dp_rank: usize,
        envelope: &EngineCoreEnvelope,
    ) -> Result<(), EngineCoreTransportError> {
        let sockets = self
            .ranks
            .get(&dp_rank)
            .ok_or(EngineCoreTransportError::UnknownRank(dp_rank))?;
        let identity =
            sockets
                .input_identity
                .as_ref()
                .ok_or(EngineCoreTransportError::MissingIdentity {
                    dp_rank,
                    channel: "input",
                })?;
        Self::send_router_frame(&sockets.input, identity, envelope, dp_rank)
    }

    pub fn send_control(
        &self,
        dp_rank: usize,
        envelope: &EngineCoreEnvelope,
    ) -> Result<(), EngineCoreTransportError> {
        let sockets = self
            .ranks
            .get(&dp_rank)
            .ok_or(EngineCoreTransportError::UnknownRank(dp_rank))?;
        let identity =
            sockets
                .control_identity
                .as_ref()
                .ok_or(EngineCoreTransportError::MissingIdentity {
                    dp_rank,
                    channel: "control",
                })?;
        Self::send_router_frame(&sockets.control, identity, envelope, dp_rank)
    }

    fn send_router_frame(
        socket: &zmq::Socket,
        identity: &[u8],
        envelope: &EngineCoreEnvelope,
        dp_rank: usize,
    ) -> Result<(), EngineCoreTransportError> {
        socket
            .send_multipart([identity, envelope.encode_to_vec().as_slice()], 0)
            .map_err(|source| EngineCoreTransportError::Zmq { dp_rank, source })
    }

    pub fn send_shutdown_all(&self) -> Result<(), EngineCoreTransportError> {
        let envelope = EngineCoreEnvelope {
            wire_version: ENGINE_CORE_WIRE_VERSION,
            payload: Some(Payload::Shutdown(())),
        };
        for &dp_rank in self.ranks.keys() {
            self.send_control(dp_rank, &envelope)?;
        }
        Ok(())
    }

    pub fn dp_ranks(&self) -> Vec<usize> {
        self.ranks.keys().copied().collect()
    }

    pub(crate) fn take_rank_sockets(
        &mut self,
    ) -> Result<BTreeMap<usize, EngineCoreRankSockets>, EngineCoreTransportError> {
        for (&dp_rank, sockets) in &self.ranks {
            if sockets.input_identity.is_none() {
                return Err(EngineCoreTransportError::MissingIdentity {
                    dp_rank,
                    channel: "input",
                });
            }
            if sockets.control_identity.is_none() {
                return Err(EngineCoreTransportError::MissingIdentity {
                    dp_rank,
                    channel: "control",
                });
            }
        }
        Ok(std::mem::take(&mut self.ranks))
    }

    pub fn receive_output_nonblocking(
        &self,
        dp_rank: usize,
    ) -> Result<Option<EngineCoreEnvelope>, EngineCoreTransportError> {
        let sockets = self
            .ranks
            .get(&dp_rank)
            .ok_or(EngineCoreTransportError::UnknownRank(dp_rank))?;
        match sockets.output.recv_bytes(zmq::DONTWAIT) {
            Ok(frame) => Self::decode_envelope(dp_rank, &frame).map(Some),
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(source) => Err(EngineCoreTransportError::Zmq { dp_rank, source }),
        }
    }

    pub fn decode_envelope(
        dp_rank: usize,
        frame: &[u8],
    ) -> Result<EngineCoreEnvelope, EngineCoreTransportError> {
        let envelope = EngineCoreEnvelope::decode(frame)
            .map_err(|source| EngineCoreTransportError::InvalidProtobuf { dp_rank, source })?;
        if envelope.wire_version != ENGINE_CORE_WIRE_VERSION {
            return Err(EngineCoreTransportError::UnsupportedWireVersion {
                dp_rank,
                actual: envelope.wire_version,
                expected: ENGINE_CORE_WIRE_VERSION,
            });
        }
        Ok(envelope)
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use tempfile::TempDir;

    use super::*;
    use crate::routers::engine_core::EngineCoreEndpoint;

    fn ipc_address(directory: &TempDir, name: &str) -> String {
        format!("ipc://{}", directory.path().join(name).display())
    }

    #[test]
    fn reuses_planned_addresses_for_handshake_ready_and_shutdown() {
        let directory = TempDir::new().unwrap();
        let endpoint = EngineCoreEndpoint {
            dp_rank: 0,
            input_address: ipc_address(&directory, "input.sock"),
            control_address: ipc_address(&directory, "control.sock"),
            output_address: ipc_address(&directory, "output.sock"),
        };
        let topology = EngineCoreEndpointTopology::new(vec![endpoint.clone()]).unwrap();
        let mut transport = EngineCoreTransport::bind(&topology, 5_000).unwrap();

        let engine = thread::spawn(move || {
            let context = zmq::Context::new();
            let input = context.socket(zmq::DEALER).unwrap();
            let control = context.socket(zmq::DEALER).unwrap();
            let output = context.socket(zmq::PUSH).unwrap();
            control.set_rcvtimeo(5_000).unwrap();
            input.connect(&endpoint.input_address).unwrap();
            control.connect(&endpoint.control_address).unwrap();
            output.connect(&endpoint.output_address).unwrap();
            input.send(&[] as &[u8], 0).unwrap();
            control.send(&[] as &[u8], 0).unwrap();

            let ready = EngineCoreEnvelope {
                wire_version: ENGINE_CORE_WIRE_VERSION,
                payload: Some(Payload::Ready(ReadySignal {
                    max_pool_tokens: Some(4_096),
                })),
            };
            output.send(ready.encode_to_vec(), 0).unwrap();

            let shutdown = control.recv_bytes(0).unwrap();
            let shutdown = EngineCoreEnvelope::decode(shutdown.as_slice()).unwrap();
            assert!(matches!(shutdown.payload, Some(Payload::Shutdown(_))));
        });

        transport.wait_until_all_connected().unwrap();
        let capacities = transport.wait_until_all_ready().unwrap();
        assert_eq!(capacities.get(&0), Some(&Some(4_096)));
        transport.send_shutdown_all().unwrap();
        engine.join().unwrap();
    }

    #[test]
    fn rejects_wrong_wire_version() {
        let envelope = EngineCoreEnvelope {
            wire_version: ENGINE_CORE_WIRE_VERSION + 1,
            payload: Some(Payload::Shutdown(())),
        };
        let error = EngineCoreTransport::decode_envelope(3, &envelope.encode_to_vec()).unwrap_err();
        assert!(matches!(
            error,
            EngineCoreTransportError::UnsupportedWireVersion { dp_rank: 3, .. }
        ));
    }

    #[test]
    fn duplicate_bind_fails_immediately() {
        let directory = TempDir::new().unwrap();
        let endpoint = EngineCoreEndpoint {
            dp_rank: 0,
            input_address: ipc_address(&directory, "input.sock"),
            control_address: ipc_address(&directory, "control.sock"),
            output_address: ipc_address(&directory, "output.sock"),
        };
        let topology = EngineCoreEndpointTopology::new(vec![endpoint]).unwrap();
        let _owner = EngineCoreTransport::bind(&topology, 1_000).unwrap();
        assert!(matches!(
            EngineCoreTransport::bind(&topology, 1_000),
            Err(EngineCoreTransportError::Zmq { .. })
        ));
    }

    #[test]
    fn decodes_python_ready_and_shutdown_fixtures() {
        // Generated by EngineCoreIpcCodec using engine_core_pb2.
        let ready =
            EngineCoreTransport::decode_envelope(0, &[0x08, 0x01, 0x72, 0x03, 0x08, 0x80, 0x20])
                .unwrap();
        assert!(matches!(
            ready.payload,
            Some(Payload::Ready(ReadySignal {
                max_pool_tokens: Some(4_096)
            }))
        ));

        let shutdown =
            EngineCoreTransport::decode_envelope(0, &[0x08, 0x01, 0x8a, 0x01, 0x00]).unwrap();
        assert!(matches!(shutdown.payload, Some(Payload::Shutdown(_))));
    }
}
