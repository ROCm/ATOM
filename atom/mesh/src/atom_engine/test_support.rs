//! Fake EngineCores on real ipc sockets, for the client's tests and the
//! frontend's: each fake connects the way EngineCore does (a DEALER on input
//! and control, each announcing itself with an empty frame; a PUSH on output
//! whose first message is `Ready`) and plays the Core's side by hand.

use std::{path::Path, thread, time::Duration};

use serde::{de::DeserializeOwned, Serialize};

use super::{
    client::EngineClient,
    spec::{BalanceSpec, BalanceStrategy, CoreSpec, EngineSpec, ModelSpec},
    wire::{self, kind},
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
        let ready = wire::Ready {
            max_pool_tokens,
            ..Default::default()
        };
        Self::connect_with(core, wire::encode(kind::READY, &ready))
    }

    /// Connect, with `first` as the first message on output.
    pub fn connect_with(core: &CoreSpec, first: Vec<u8>) -> Self {
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
        fake.send_raw(first);
        fake
    }

    pub fn send<T: Serialize>(&self, kind: u8, body: &T) {
        self.send_raw(wire::encode(kind, body));
    }

    /// A message sent as it is, encoded or not.
    pub fn send_raw(&self, message: Vec<u8>) {
        self.output.send(message, 0).unwrap();
    }

    /// The next message on `socket` within `timeout`: its kind and body.
    pub fn recv_on(socket: &zmq::Socket, timeout: Duration) -> Option<(u8, Vec<u8>)> {
        if socket
            .poll(zmq::POLLIN, timeout.as_millis() as i64)
            .unwrap()
            == 0
        {
            return None;
        }
        let message = socket.recv_bytes(0).unwrap();
        let (kind, body) = wire::split(&message).unwrap();
        Some((kind, body.to_vec()))
    }

    pub fn submitted(&self) -> Option<Vec<wire::SubmitItem>> {
        Self::recv_on(&self.input, WAIT).map(|(kind, body)| {
            assert_eq!(kind, kind::ADD, "expected Submit on input");
            wire::body::<wire::Submit>(&body).unwrap().items
        })
    }

    /// The next control message: its kind and body.
    pub fn control_message(&self) -> (u8, Vec<u8>) {
        Self::recv_on(&self.control, WAIT).expect("a control message")
    }

    /// One step: `tokens` for each `(seq_id, tokens, finish_reason)`, a
    /// `Some` reason marking the entry finished. Every entry reports a
    /// 3-token prefix-cache hit.
    pub fn stream(&self, entries: &[(u64, &[i32], Option<&str>)]) {
        let items = entries
            .iter()
            .map(|(seq_id, tokens, reason)| wire::StreamItem {
                seq_id: *seq_id,
                output_tokens: tokens.iter().map(|&t| t as u32).collect(),
                finished: reason.is_some(),
                finish_reason: reason.map(str::to_string),
                num_cached_tokens: 3,
                ..Default::default()
            })
            .collect();
        self.send(kind::STREAM, &wire::Stream { items });
    }
}

/// `body` as a `T`: a message's body this test expects to be one.
pub(crate) fn decoded<T: DeserializeOwned>(body: &[u8]) -> T {
    wire::body(body).unwrap()
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
        synthetic_token_text: None,
        preserve_tool_call_arguments: false,
        model_card: Default::default(),
        required_channel: None,
        reasoning_toggle: None,
        template_efforts: vec!["low".into(), "high".into(), "max".into()],
        end_token_ids: vec![],
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

/// An int32 token buffer.
pub(crate) fn token_bytes(tokens: &[i32]) -> serde_bytes::ByteBuf {
    serde_bytes::ByteBuf::from(
        tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}
