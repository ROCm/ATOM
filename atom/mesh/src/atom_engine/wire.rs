//! The engine wire protocol: the messages generated from
//! `atom/model_engine/protocol/engine_v1.proto` (the spec) and the multipart
//! frames they ride in.

use prost::Message as _;

#[allow(clippy::all, unused_qualifications)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/atom.engine.v1.rs"));
}

/// `PROTOCOL_VERSION` in `atom/model_engine/protocol/wire.py`.
pub const PROTOCOL_VERSION: u32 = 1;

/// One message for a Core: frame 0 the envelope, later frames its buffers.
pub(super) fn encode(body: pb::envelope::Body, data_frames: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let envelope = pb::Envelope {
        protocol_version: PROTOCOL_VERSION,
        body: Some(body),
    };
    let mut frames = Vec::with_capacity(1 + data_frames.len());
    frames.push(envelope.encode_to_vec());
    frames.extend(data_frames);
    frames
}

/// A buffer a `BufferRef` can point at: int32 tokens, little-endian.
pub(super) fn token_frame(tokens: &[u32]) -> Vec<u8> {
    tokens
        .iter()
        .flat_map(|t| (*t as i32).to_le_bytes())
        .collect()
}

/// A received message: its envelope and the frames its `BufferRef`s index.
pub(super) struct Inbound {
    pub body: pb::envelope::Body,
    frames: Vec<zmq::Message>,
}

impl Inbound {
    pub fn decode(frames: Vec<zmq::Message>) -> Result<Self, String> {
        let head = frames.first().ok_or("empty message")?;
        let envelope = pb::Envelope::decode(&head[..]).map_err(|e| e.to_string())?;
        if envelope.protocol_version != PROTOCOL_VERSION {
            return Err(format!(
                "protocol version {}, expected {PROTOCOL_VERSION}",
                envelope.protocol_version
            ));
        }
        let body = envelope.body.ok_or("envelope carries no body")?;
        Ok(Self { body, frames })
    }

    /// The bytes `r` names; an absent ref is an empty buffer.
    fn buffer(&self, r: Option<&pb::BufferRef>) -> Result<&[u8], String> {
        let Some(r) = r else { return Ok(&[]) };
        let frame = self
            .frames
            .get(r.frame as usize)
            .filter(|_| r.frame >= 1)
            .ok_or_else(|| format!("buffer refers to missing frame {}", r.frame))?;
        let start = r.offset as usize;
        let end = start + r.nbytes as usize;
        frame
            .get(start..end)
            .ok_or_else(|| format!("buffer [{start}, {end}) overruns frame {}", r.frame))
    }

    pub fn tokens(&self, r: Option<&pb::BufferRef>) -> Result<Vec<u32>, String> {
        let bytes = self.buffer(r)?;
        if bytes.len() % 4 != 0 {
            return Err("an int32 buffer must be a multiple of 4 bytes".into());
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32)
            .collect())
    }

    pub fn floats(&self, r: Option<&pb::BufferRef>) -> Result<Vec<f64>, String> {
        let bytes = self.buffer(r)?;
        if bytes.len() % 8 != 0 {
            return Err("a float64 buffer must be a multiple of 8 bytes".into());
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}
