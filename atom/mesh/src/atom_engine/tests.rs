//! `EngineClient` against fake EngineCores on real ipc sockets: each fake
//! connects the way EngineCore does (a DEALER on input and control, each
//! announcing itself with an empty frame; a PUSH on output whose first
//! message is `Ready`) and plays the Core's side of the protocol by hand.
//!
//! The protocol across languages: the Core's messages as Python encodes them
//! (`tests/fixtures/engine_v1/core_*`, written by
//! `tests/test_engine_v1_cross_language.py`) drive the client here, and the
//! client's own messages are kept as `client_*` there for Python to decode.
//! `ATOM_UPDATE_FIXTURES=1` rewrites the `client_*` ones.

use std::{future::poll_fn, path::PathBuf, thread, time::Duration};

use prost::Message as _;
use tempfile::TempDir;

use super::{
    client::{EngineClient, SeqEvent, SeqFailure, SeqHandle, Submission},
    spec::{BalanceSpec, BalanceStrategy, CoreSpec, EngineSpec, ModelSpec},
    wire::{self, pb},
};

const WAIT: Duration = Duration::from_secs(5);

/// One fake Core's sockets.
struct FakeCore {
    input: zmq::Socket,
    control: zmq::Socket,
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

    fn send(&self, body: pb::envelope::Body, data_frames: Vec<Vec<u8>>) {
        self.send_frames(wire::encode(body, data_frames));
    }

    fn send_frames(&self, frames: Vec<Vec<u8>>) {
        self.output.send_multipart(frames, 0).unwrap();
    }

    /// The next whole message on `socket`, frames as sent.
    fn recv_frames(socket: &zmq::Socket) -> Vec<Vec<u8>> {
        assert!(socket.poll(zmq::POLLIN, WAIT.as_millis() as i64).unwrap() > 0);
        socket.recv_multipart(0).unwrap()
    }

    /// The next message on `socket` within `timeout`: its body and data frames.
    fn recv_on(
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

    fn submitted(&self) -> Option<(Vec<pb::RequestSpec>, Vec<Vec<u8>>)> {
        Self::recv_on(&self.input, WAIT).map(|(body, data)| match body {
            pb::envelope::Body::Submit(batch) => (batch.requests, data),
            other => panic!("expected SubmitBatch on input, got {other:?}"),
        })
    }

    fn control_message(&self) -> pb::envelope::Body {
        Self::recv_on(&self.control, WAIT)
            .expect("a control message")
            .0
    }

    /// One step: `tokens` for each `(seq_id, tokens, finish_reason)`, a
    /// `Some` reason marking the entry finished.
    fn stream(&self, entries: &[(u64, &[i32], Option<&str>)]) {
        let mut batch = pb::StreamBatch::default();
        for (i, (seq_id, tokens, reason)) in entries.iter().enumerate() {
            batch.seq_ids.push(*seq_id);
            batch.request_ids.push(*seq_id as i64);
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

/// `cores` fake Cores on fresh ipc paths, with the client connected to them.
struct Harness {
    client: EngineClient,
    cores: Vec<FakeCore>,
    _dir: TempDir,
}

impl Harness {
    fn start(pools: &[Option<u64>]) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let address = |rank: usize, socket: &str| {
            format!("ipc://{}/core{rank}_{socket}", dir.path().display())
        };
        let engines: Vec<CoreSpec> = (0..pools.len())
            .map(|rank| CoreSpec {
                dp_rank: rank,
                pid: std::process::id(),
                input: address(rank, "input"),
                control: address(rank, "control"),
                output: address(rank, "output"),
            })
            .collect();
        let spec = EngineSpec {
            model: ModelSpec {
                served_name: "fake".into(),
                tokenizer_path: "fake".into(),
                chat_template: None,
                default_chat_template_kwargs: Default::default(),
                max_model_len: None,
                default_max_tokens: 16,
                max_output_tokens: None,
                speculative: false,
                required_channel: None,
                reasoning_toggle: None,
            },
            balance: BalanceSpec {
                strategy: BalanceStrategy::RoundRobin,
                req_equiv: 512,
                session_affinity: false,
            },
            engines: engines.clone(),
        };
        // The client binds and blocks until every Core is ready, so the
        // fakes connect from their own threads.
        let fakes: Vec<_> = engines
            .into_iter()
            .zip(pools.iter().copied())
            .map(|(core, pool)| thread::spawn(move || FakeCore::connect(&core, pool)))
            .collect();
        let client = EngineClient::connect(&spec).unwrap();
        let cores = fakes.into_iter().map(|f| f.join().unwrap()).collect();
        Self {
            client,
            cores,
            _dir: dir,
        }
    }

    fn submit(&self, n: u32, stream: bool, logprobs: bool) -> Vec<SeqHandle> {
        let sampling = pb::SamplingParams {
            temperature: Some(1.0),
            max_tokens: Some(8),
            n,
            logprobs: logprobs.then_some(pb::sampling_params::Logprobs::LogprobsFlag(true)),
            ..Default::default()
        };
        self.client
            .submit(Submission {
                request_id: "req",
                prompt: &[11, 12, 13],
                sampling,
                stop_token_sequences: vec![],
                n,
                session: None,
                stream,
            })
            .unwrap()
    }

    /// The Core the first submission went to, and what it received.
    fn first_submission(&self) -> (usize, Vec<pb::RequestSpec>, Vec<Vec<u8>>) {
        for (rank, core) in self.cores.iter().enumerate() {
            if let Some((specs, data)) = FakeCore::recv_on(&core.input, Duration::from_millis(500))
                .map(|(body, data)| match body {
                    pb::envelope::Body::Submit(batch) => (batch.requests, data),
                    other => panic!("expected SubmitBatch, got {other:?}"),
                })
            {
                return (rank, specs, data);
            }
        }
        panic!("no Core received the submission");
    }
}

async fn next_event(handle: &mut SeqHandle) -> SeqEvent {
    tokio::time::timeout(WAIT, poll_fn(|cx| handle.poll_event(cx)))
        .await
        .expect("an event within the timeout")
        .expect("the handle is still open")
}

fn token_frame(tokens: &[i32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}

#[tokio::test]
async fn ready_sets_the_pool_limit_and_a_stream_ends_on_its_finished_entry() {
    let harness = Harness::start(&[Some(1000), Some(600)]);
    assert_eq!(harness.client.max_pool_tokens(), Some(600));
    assert_eq!(harness.client.core_count(), 2);

    let mut handles = harness.submit(1, true, false);
    let (rank, specs, data) = harness.first_submission();
    let [spec] = &specs[..] else {
        panic!("expected one RequestSpec, got {specs:?}")
    };
    assert_eq!(spec.data_parallel_rank, Some(rank as i64));
    assert_eq!(spec.external_request_id.as_deref(), Some("req"));
    assert!(!spec.needs_independent_noise);
    let prompt = spec.prompt_token_ids.as_ref().unwrap();
    assert_eq!(prompt.frame, 1);
    assert_eq!(data[0], token_frame(&[11, 12, 13]));

    let core = &harness.cores[rank];
    core.stream(&[(spec.seq_id, &[7], None)]);
    core.stream(&[(spec.seq_id, &[8], Some("eos"))]);
    let handle = &mut handles[0];
    assert!(matches!(next_event(handle).await, SeqEvent::Tokens(t) if t == [7]));
    assert!(matches!(next_event(handle).await, SeqEvent::Tokens(t) if t == [8]));
    match next_event(handle).await {
        SeqEvent::Finished(done) => {
            assert_eq!(done.token_ids, [7, 8]);
            assert_eq!(done.leave_reason, "eos");
            assert_eq!((done.prompt_tokens, done.cached_tokens), (3, 3));
            assert!(done.logprobs.is_none());
        }
        other => panic!("expected Finished, got {other:?}"),
    }
}

#[tokio::test]
async fn a_logprobs_request_ends_on_its_completion() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, true);
    let (_, specs, _) = harness.first_submission();
    let seq_id = specs[0].seq_id;
    let core = &harness.cores[0];
    // Its finished stream entry does not end it: only the Completion has logprobs.
    core.stream(&[(seq_id, &[5, 6], Some("max_tokens"))]);
    let completion = pb::Completion {
        seq_id,
        completion_token_ids: Some(pb::BufferRef {
            frame: 1,
            offset: 0,
            nbytes: 8,
        }),
        logprobs: Some(pb::BufferRef {
            frame: 2,
            offset: 0,
            nbytes: 16,
        }),
        return_logprobs: true,
        num_prompt_tokens: 3,
        leave_reason: "max_tokens".into(),
        ..Default::default()
    };
    let logprobs: Vec<u8> = [-0.5f64, -1.25]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    core.send(
        pb::envelope::Body::Completions(pb::Completions {
            items: vec![completion],
        }),
        vec![token_frame(&[5, 6]), logprobs],
    );
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => {
            assert_eq!(done.token_ids, [5, 6]);
            assert_eq!(done.logprobs.as_deref(), Some(&[-0.5, -1.25][..]));
            assert_eq!(done.leave_reason, "max_tokens");
        }
        other => panic!("expected Finished, got {other:?}"),
    }
}

#[tokio::test]
async fn rejections_map_to_failures_by_code() {
    use pb::rejected_request::Code;
    let harness = Harness::start(&[None]);
    for (code, expect_invalid) in [(Code::Invalid, true), (Code::Overloaded, false)] {
        let mut handles = harness.submit(1, false, false);
        let (_, specs, _) = harness.first_submission();
        harness.cores[0].send(
            pb::envelope::Body::Rejected(pb::Rejected {
                items: vec![pb::RejectedRequest {
                    seq_id: specs[0].seq_id,
                    reason: "no".into(),
                    code: code as i32,
                }],
            }),
            vec![],
        );
        match next_event(&mut handles[0]).await {
            SeqEvent::Failed(SeqFailure::Invalid(reason)) if expect_invalid => {
                assert_eq!(reason, "no")
            }
            SeqEvent::Failed(SeqFailure::Unavailable(_)) if !expect_invalid => {}
            other => panic!("{code:?}: unexpected {other:?}"),
        }
    }
}

#[tokio::test]
async fn dropping_an_unfinished_handle_aborts_its_sequence() {
    let harness = Harness::start(&[None]);
    let handles = harness.submit(1, true, false);
    let (_, specs, _) = harness.first_submission();
    drop(handles);
    match harness.cores[0].control_message() {
        pb::envelope::Body::Abort(abort) => assert_eq!(abort.seq_ids, [specs[0].seq_id]),
        other => panic!("expected Abort, got {other:?}"),
    }
}

#[tokio::test]
async fn siblings_spread_over_ranks_and_share_one_prompt_frame() {
    let harness = Harness::start(&[None, None]);
    let handles = harness.submit(4, false, false);
    assert_eq!(handles.len(), 4);
    let mut siblings = Vec::new();
    for core in &harness.cores {
        let (specs, data) = core.submitted().expect("each rank gets its share");
        assert_eq!(data.len(), 1, "one prompt frame per message");
        assert_eq!(data[0], token_frame(&[11, 12, 13]));
        for spec in specs {
            assert!(spec.needs_independent_noise);
            assert_eq!(spec.external_request_id, None);
            assert_eq!(spec.parent_request_id.as_deref(), Some("req"));
            assert_eq!(spec.prompt_token_ids.as_ref().unwrap().frame, 1);
            siblings.push(spec.sibling_index);
        }
    }
    siblings.sort();
    assert_eq!(siblings, [0, 1, 2, 3]);
}

#[tokio::test]
async fn a_core_that_stops_unasked_loses_the_client() {
    let harness = Harness::start(&[None, None]);
    let mut handles = harness.submit(1, true, false);
    let (rank, _, _) = harness.first_submission();
    assert_eq!(harness.client.lost(), None);
    harness.cores[rank].send(pb::envelope::Body::Stopped(pb::Stopped {}), vec![]);
    assert!(matches!(
        next_event(&mut handles[0]).await,
        SeqEvent::Failed(SeqFailure::Unavailable(_))
    ));
    let reason = tokio::time::timeout(WAIT, harness.client.wait_lost())
        .await
        .expect("the loss is reported");
    assert_eq!(reason, format!("EngineCore DP rank {rank} stopped"));
    assert_eq!(harness.client.alive_core_count(), 1);
    // The live rank takes nothing more either: the server is exiting.
    let refused = harness.client.submit(Submission {
        request_id: "late",
        prompt: &[1],
        sampling: Default::default(),
        stop_token_sequences: vec![],
        n: 1,
        session: None,
        stream: false,
    });
    assert!(matches!(refused, Err(SeqFailure::Unavailable(_))));
    let live = &harness.cores[1 - rank];
    assert!(FakeCore::recv_on(&live.input, Duration::from_millis(200)).is_none());
}

#[tokio::test]
async fn shutdown_asks_every_core_to_stop_and_waits_for_them() {
    let harness = Harness::start(&[None, None]);
    // Answer each Shutdown with Stopped from a thread, as a Core would.
    let cores = harness.cores;
    let replies = thread::spawn(move || {
        for core in &cores {
            assert!(matches!(
                core.control_message(),
                pb::envelope::Body::Shutdown(_)
            ));
            core.send(pb::envelope::Body::Stopped(pb::Stopped {}), vec![]);
        }
        cores
    });
    harness.client.shutdown(WAIT);
    replies.join().unwrap();
    assert_eq!(harness.client.alive_core_count(), 0);
    // Stopping when asked to is not a loss.
    assert_eq!(harness.client.lost(), None);
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/engine_v1")
        .join(format!("{name}.frames"))
}

/// A message as `u32le length, bytes` per frame, as the Python test keeps it.
fn read_fixture(name: &str) -> Vec<Vec<u8>> {
    let path = fixture(name);
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut frames = Vec::new();
    let mut rest = &data[..];
    while !rest.is_empty() {
        let n = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        frames.push(rest[4..4 + n].to_vec());
        rest = &rest[4 + n..];
    }
    frames
}

/// Fail when the client's `frames` drifted from fixture `name`, which Python
/// decodes; rewrite it instead under `ATOM_UPDATE_FIXTURES`.
fn check_client_fixture(name: &str, frames: &[Vec<u8>]) {
    if std::env::var_os("ATOM_UPDATE_FIXTURES").is_some() {
        let mut data = Vec::new();
        for frame in frames {
            data.extend_from_slice(&(frame.len() as u32).to_le_bytes());
            data.extend_from_slice(frame);
        }
        std::fs::write(fixture(name), data).unwrap();
    }
    // Arrival times are the clock's; everything else must match.
    let envelope = |frames: &[Vec<u8>]| {
        let mut envelope = pb::Envelope::decode(&frames[0][..]).unwrap();
        if let Some(pb::envelope::Body::Submit(batch)) = &mut envelope.body {
            batch.requests.iter_mut().for_each(|r| r.arrive_time = 0.0);
        }
        envelope
    };
    let kept = read_fixture(name);
    assert!(
        envelope(frames) == envelope(&kept) && frames[1..] == kept[1..],
        "{name}.frames is stale: rerun with ATOM_UPDATE_FIXTURES=1, then \
         tests/test_engine_v1_cross_language.py against it"
    );
}

#[tokio::test]
async fn the_clients_messages_match_what_python_decodes() {
    let harness = Harness::start(&[None]);
    let sampling = pb::SamplingParams {
        temperature: Some(0.7),
        top_k: Some(20),
        top_p: Some(0.9),
        max_tokens: Some(64),
        stop_strings: Some(pb::StopStrings {
            value: Some(pb::stop_strings::Value::Many(pb::StringList {
                items: vec!["\n\n".into()],
            })),
        }),
        n: 2,
        logprobs: Some(pb::sampling_params::Logprobs::LogprobsFlag(true)),
        required_channel: Some(pb::RequiredChannel {
            after: vec![1],
            opening: vec![2, 3],
            opens_at: 1,
            end_ids: vec![4],
        }),
        ..Default::default()
    };
    let handles = harness
        .client
        .submit(Submission {
            request_id: "req",
            prompt: &[11, 12, 13],
            sampling,
            stop_token_sequences: vec![
                pb::TokenIds { ids: vec![5, 6] },
                pb::TokenIds { ids: vec![7] },
            ],
            n: 2,
            session: Some("sess"),
            stream: false,
        })
        .unwrap();
    let core = &harness.cores[0];
    check_client_fixture("client_submit", &FakeCore::recv_frames(&core.input));
    drop(handles);
    check_client_fixture("client_abort", &FakeCore::recv_frames(&core.control));
    let _second_abort = FakeCore::recv_frames(&core.control);
    harness.client.shutdown(Duration::from_millis(100));
    check_client_fixture("client_shutdown", &FakeCore::recv_frames(&core.control));
}

#[tokio::test]
async fn python_core_messages_drive_the_client() {
    let decode = |name| {
        wire::Inbound::decode(
            read_fixture(name)
                .into_iter()
                .map(zmq::Message::from)
                .collect(),
        )
        .unwrap()
        .body
    };
    assert!(matches!(
        decode("core_ready"),
        pb::envelope::Body::Ready(pb::Ready {
            max_pool_tokens: Some(4096)
        })
    ));
    assert!(matches!(
        decode("core_stopped"),
        pb::envelope::Body::Stopped(_)
    ));

    // Each harness's first sequence is 0, the one the fixtures name.
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, true, false);
    harness.cores[0].send_frames(read_fixture("core_stream"));
    assert!(matches!(next_event(&mut handles[0]).await, SeqEvent::Tokens(t) if t == [7, 8]));
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => {
            assert_eq!(done.token_ids, [7, 8]);
            assert_eq!(done.leave_reason, "eos");
            assert_eq!((done.prompt_tokens, done.cached_tokens), (3, 3));
        }
        other => panic!("expected Finished, got {other:?}"),
    }

    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, true);
    harness.cores[0].send_frames(read_fixture("core_completions"));
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => {
            assert_eq!(done.token_ids, [5, 6]);
            assert_eq!(done.logprobs.as_deref(), Some(&[-0.5, -1.25][..]));
            assert_eq!((done.prompt_tokens, done.cached_tokens), (3, 2));
            assert_eq!(done.leave_reason, "max_tokens");
        }
        other => panic!("expected Finished, got {other:?}"),
    }

    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, false);
    harness.cores[0].send_frames(read_fixture("core_rejected"));
    assert!(matches!(
        next_event(&mut handles[0]).await,
        SeqEvent::Failed(SeqFailure::Invalid(reason)) if reason == "bad"
    ));
}
