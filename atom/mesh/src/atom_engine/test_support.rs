//! Fake EngineCores on real ipc sockets, for the client's tests and the
//! frontend's: each fake connects the way EngineCore does (a DEALER on input
//! and control, each announcing itself with an empty frame; a PUSH on output
//! whose first message is `Ready`) and plays the Core's side by hand.

use std::{path::Path, thread, time::Duration};

use prost::Message as _;

use super::{
    client::EngineClient,
    spec::{BalanceSpec, BalanceStrategy, CoreSpec, EngineSpec, ModelSpec},
    wire::{self, pb},
};

pub(crate) const WAIT: Duration = Duration::from_secs(5);

/// One fake Core's sockets.
pub(crate) struct FakeCore {
    pub input: zmq::Socket,
    pub control: zmq::Socket,
    output: zmq::Socket,
    _ctx: zmq::Context,
}

impl FakeCore {
    fn connect(core: &CoreSpec, max_pool_tokens: Option<u64>) -> Self {
        let ctx = zmq::Context::new();
        let dealer = |address: &str| {
            let socket = ctx.socket(zmq::DEALER).unwrap();
            socket.set_linger(0).unwrap();
            socket.connect(address).unwrap();
            socket.send("", 0).unwrap();
            socket
        };
        let input = dealer(&core.input);
        let control = dealer(&core.control);
        let output = ctx.socket(zmq::PUSH).unwrap();
        output.set_linger(0).unwrap();
        output.connect(&core.output).unwrap();
        let fake = Self {
            input,
            control,
            output,
            _ctx: ctx,
        };
        fake.send(
            pb::envelope::Body::Ready(pb::Ready { max_pool_tokens }),
            vec![],
        );
        fake
    }

    pub fn send(&self, body: pb::envelope::Body, data_frames: Vec<Vec<u8>>) {
        self.output
            .send_multipart(wire::encode(body, data_frames), 0)
            .unwrap();
    }

    /// Frames sent as they are, encoded or not.
    pub fn send_frames(&self, frames: Vec<Vec<u8>>) {
        self.output.send_multipart(frames, 0).unwrap();
    }

    /// The next message on `socket` within `timeout`: its body and data frames.
    pub fn recv_on(
        socket: &zmq::Socket,
        timeout: Duration,
    ) -> Option<(pb::envelope::Body, Vec<Vec<u8>>)> {
        if socket
            .poll(zmq::POLLIN, timeout.as_millis() as i64)
            .unwrap()
            == 0
        {
            return None;
        }
        let mut frames = socket.recv_multipart(0).unwrap();
        let data = frames.split_off(1);
        let envelope = pb::Envelope::decode(&frames[0][..]).unwrap();
        assert_eq!(envelope.protocol_version, wire::PROTOCOL_VERSION);
        Some((envelope.body.unwrap(), data))
    }

    pub fn submitted(&self) -> Option<(Vec<pb::RequestSpec>, Vec<Vec<u8>>)> {
        Self::recv_on(&self.input, WAIT).map(|(body, data)| match body {
            pb::envelope::Body::Submit(batch) => (batch.requests, data),
            other => panic!("expected SubmitBatch on input, got {other:?}"),
        })
    }

    pub fn control_message(&self) -> pb::envelope::Body {
        Self::recv_on(&self.control, WAIT)
            .expect("a control message")
            .0
    }

    /// One step: `tokens` for each `(seq_id, tokens, finish_reason)`, a
    /// `Some` reason marking the entry finished. Every entry reports a
    /// 3-token prefix-cache hit.
    pub fn stream(&self, entries: &[(u64, &[i32], Option<&str>)]) {
        let mut batch = pb::StreamBatch::default();
        for (i, (seq_id, tokens, reason)) in entries.iter().enumerate() {
            batch.seq_ids.push(*seq_id);
            batch.token_counts.push(tokens.len() as u32);
            batch.tokens.extend_from_slice(tokens);
            batch.finished.push(reason.is_some());
            batch.num_cached_tokens.push(3);
            if let Some(reason) = reason {
                batch.finish_reasons.insert(i as u32, reason.to_string());
            }
        }
        self.send(pb::envelope::Body::Stream(batch), vec![]);
    }
}

/// A spec for `ranks` Cores on fresh ipc paths under `dir`, serving `model`.
pub(crate) fn engine_spec(dir: &Path, ranks: usize, model: ModelSpec) -> EngineSpec {
    let address =
        |rank: usize, socket: &str| format!("ipc://{}/core{rank}_{socket}", dir.display());
    EngineSpec {
        model,
        balance: BalanceSpec {
            strategy: BalanceStrategy::RoundRobin,
            req_equiv: 512,
            session_affinity: false,
        },
        engines: (0..ranks)
            .map(|rank| CoreSpec {
                dp_rank: rank,
                pid: std::process::id(),
                input: address(rank, "input"),
                control: address(rank, "control"),
                output: address(rank, "output"),
            })
            .collect(),
    }
}

pub(crate) fn model_spec(served_name: &str, tokenizer_path: &str) -> ModelSpec {
    ModelSpec {
        served_name: served_name.into(),
        tokenizer_path: tokenizer_path.into(),
        chat_template: None,
        default_chat_template_kwargs: Default::default(),
        max_model_len: None,
        default_max_tokens: 16,
        max_output_tokens: None,
        synthetic_acceptance: false,
        model_card: Default::default(),
        required_channel: None,
        reasoning_toggle: None,
    }
}

/// Connect a client to one fake Core per engine in `spec`, each announcing
/// its `pools` entry as its pool limit.
pub(crate) fn connect(spec: &EngineSpec, pools: &[Option<u64>]) -> (EngineClient, Vec<FakeCore>) {
    assert_eq!(spec.engines.len(), pools.len());
    // The client binds and blocks until every Core is ready, so the fakes
    // connect from their own threads.
    let fakes: Vec<_> = spec
        .engines
        .iter()
        .cloned()
        .zip(pools.iter().copied())
        .map(|(core, pool)| thread::spawn(move || FakeCore::connect(&core, pool)))
        .collect();
    let client = EngineClient::connect(spec, &|| false).unwrap();
    let cores = fakes.into_iter().map(|f| f.join().unwrap()).collect();
    (client, cores)
}

/// An int32 token buffer, as a data frame.
pub(crate) fn token_frame(tokens: &[i32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}
