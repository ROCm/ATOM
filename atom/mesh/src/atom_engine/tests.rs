//! The wire framing, the launch spec, one Core's link and the client: a
//! message decodes back with its buffers and a malformed one is refused,
//! `EngineSpec` parses a hand-written spec, a `CoreLink` handshakes with a
//! fake Core on real ipc sockets and routes input and control to it, and gives
//! up on a Core that stops before it is ready or exits, or when interrupted,
//! and `EngineClient` drives fake Cores (`test_support`) through the
//! protocol's every ending.

use std::{future::poll_fn, thread, time::Duration};

use prost::Message as _;
use serde_json::json;
use tempfile::TempDir;

use super::{
    client::{EngineClient, SeqEvent, SeqFailure, SeqHandle, Submission},
    link::{self, CoreLink},
    spec::{BalanceStrategy, CoreSpec, EngineSpec},
    test_support::{self, token_frame, FakeCore, WAIT},
    wire::{self, pb, Inbound, PROTOCOL_VERSION},
};

fn inbound(frames: Vec<Vec<u8>>) -> Result<Inbound, String> {
    Inbound::decode(frames.into_iter().map(zmq::Message::from).collect())
}

fn buffer(frame: u32, offset: u64, nbytes: u64) -> Option<pb::BufferRef> {
    Some(pb::BufferRef {
        frame,
        offset,
        nbytes,
    })
}

fn envelope_frame(protocol_version: u32, body: Option<pb::envelope::Body>) -> Vec<u8> {
    pb::Envelope {
        protocol_version,
        body,
    }
    .encode_to_vec()
}

#[test]
fn a_message_decodes_back_with_the_buffers_its_refs_name() {
    let floats: Vec<u8> = [0.5f64, -1.25]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let completion = pb::Completion {
        completion_token_ids: buffer(1, 4, 8),
        logprobs: buffer(2, 0, 16),
        ..Default::default()
    };
    let body = pb::envelope::Body::Completions(pb::Completions {
        items: vec![completion],
    });
    let frames = wire::encode(body, vec![wire::token_frame(&[7, 8, u32::MAX]), floats]);
    assert_eq!(frames.len(), 3);

    let message = inbound(frames).unwrap();
    let pb::envelope::Body::Completions(batch) = &message.body else {
        panic!("decoded {:?}", message.body);
    };
    let completion = &batch.items[0];
    // u32::MAX crosses as int32 -1 and comes back as itself.
    assert_eq!(
        message
            .tokens(completion.completion_token_ids.as_ref())
            .unwrap(),
        vec![8, u32::MAX]
    );
    assert_eq!(
        message.floats(completion.logprobs.as_ref()).unwrap(),
        vec![0.5, -1.25]
    );
    assert!(message.tokens(None).unwrap().is_empty());
}

#[test]
fn a_malformed_message_or_an_out_of_bounds_ref_is_refused() {
    let stopped = || Some(pb::envelope::Body::Stopped(pb::Stopped {}));
    assert!(inbound(vec![]).is_err());
    assert!(inbound(vec![envelope_frame(PROTOCOL_VERSION, None)]).is_err());
    let err = inbound(vec![envelope_frame(PROTOCOL_VERSION + 1, stopped())])
        .err()
        .unwrap();
    assert!(err.contains("protocol version"), "{err}");

    let message = inbound(vec![
        envelope_frame(PROTOCOL_VERSION, stopped()),
        vec![0; 8],
    ])
    .unwrap();
    // Frame 0 is the envelope, never a buffer.
    let err = message.tokens(buffer(0, 0, 4).as_ref()).err().unwrap();
    assert!(err.contains("envelope"), "{err}");
    assert!(message.tokens(buffer(1, 4, 8).as_ref()).is_err());
    // Within the frame, but not a whole number of elements.
    assert!(message.tokens(buffer(1, 0, 6).as_ref()).is_err());
}

fn spec_json() -> serde_json::Value {
    json!({
        "model": {"served_name": "m", "tokenizer_path": "/m", "default_max_tokens": 16},
        "balance": {"strategy": "least_tokens", "req_equiv": 512},
        "engines": [
            {"dp_rank": 0, "pid": 10, "input": "ipc:///i0", "control": "ipc:///c0", "output": "ipc:///o0"},
            {"dp_rank": 1, "pid": 11, "input": "ipc:///i1", "control": "ipc:///c1", "output": "ipc:///o1"},
        ],
    })
}

#[test]
fn a_spec_lists_its_cores_and_refuses_cores_it_cannot_place() {
    let spec = EngineSpec::from_json(&spec_json().to_string()).unwrap();
    assert_eq!(spec.engines[1].pid, 11);
    assert_eq!(spec.model.served_name, "m");
    assert_eq!(spec.balance.strategy, BalanceStrategy::LeastTokens);
    // Fields added later default.
    assert!(!spec.balance.session_affinity);
    assert!(spec.model.reasoning_toggle.is_none());

    let mut no_cores = spec_json();
    no_cores["engines"] = json!([]);
    assert!(EngineSpec::from_json(&no_cores.to_string()).is_err());
    let mut swapped = spec_json();
    swapped["engines"].as_array_mut().unwrap().reverse();
    assert!(EngineSpec::from_json(&swapped.to_string()).is_err());
}

fn core_spec(dir: &TempDir, pid: u32) -> CoreSpec {
    let address = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    CoreSpec {
        dp_rank: 0,
        pid,
        input: address("input"),
        control: address("control"),
        output: address("output"),
    }
}

/// Connect as EngineCore does (a DEALER on input and control, each announcing
/// itself with an empty frame; a PUSH on output), send `first`, then hand
/// back the next message on input and on control, or none if either does not
/// come within the receive timeout.
fn fake_core(
    spec: CoreSpec,
    first: pb::envelope::Body,
) -> thread::JoinHandle<Option<[Inbound; 2]>> {
    thread::spawn(move || {
        let ctx = zmq::Context::new();
        let dealer = |address: &str| {
            let socket = ctx.socket(zmq::DEALER).unwrap();
            // A message the link sent elsewhere fails the test, not hangs it;
            // short, since a test the link sends nothing to waits it out.
            socket.set_rcvtimeo(1000).unwrap();
            socket.connect(address).unwrap();
            socket.send(Vec::<u8>::new(), 0).unwrap();
            socket
        };
        let input = dealer(&spec.input);
        let control = dealer(&spec.control);
        let output = ctx.socket(zmq::PUSH).unwrap();
        output.connect(&spec.output).unwrap();
        output
            .send_multipart(wire::encode(first, vec![]), 0)
            .unwrap();
        let receive = |socket: &zmq::Socket| Inbound::decode(link::recv_frames(socket).ok()?).ok();
        Some([receive(&input)?, receive(&control)?])
    })
}

fn never_interrupted() -> bool {
    false
}

#[test]
fn a_link_handshakes_and_routes_input_and_control_to_their_core() {
    let dir = TempDir::new().unwrap();
    let spec = core_spec(&dir, std::process::id());
    let ready = pb::envelope::Body::Ready(pb::Ready {
        max_pool_tokens: Some(4096),
    });
    let core = fake_core(spec.clone(), ready);

    let ctx = zmq::Context::new();
    let (link, _output, max_pool_tokens) =
        CoreLink::connect(&ctx, spec, &never_interrupted).unwrap();
    assert_eq!(max_pool_tokens, Some(4096));

    let tokens = wire::token_frame(&[1, 2, 3]);
    let submit = pb::SubmitBatch {
        requests: vec![pb::RequestSpec {
            prompt_token_ids: buffer(1, 0, tokens.len() as u64),
            ..Default::default()
        }],
    };
    link.send_input(wire::encode(
        pb::envelope::Body::Submit(submit),
        vec![tokens],
    ))
    .unwrap();
    let abort = pb::Abort { seq_ids: vec![7] };
    link.send_control(wire::encode(pb::envelope::Body::Abort(abort), vec![]))
        .unwrap();

    let [input, control] = core
        .join()
        .unwrap()
        .expect("the core received both messages");
    let pb::envelope::Body::Submit(batch) = &input.body else {
        panic!("input carried {:?}", input.body);
    };
    assert_eq!(
        input
            .tokens(batch.requests[0].prompt_token_ids.as_ref())
            .unwrap(),
        vec![1, 2, 3]
    );
    assert!(matches!(
        &control.body,
        pb::envelope::Body::Abort(a) if a.seq_ids == [7]
    ));
}

#[test]
fn a_link_gives_up_on_a_core_that_exits() {
    // No process has pid 0 in /proc, so the Core counts as exited.
    let dir = TempDir::new().unwrap();
    let ctx = zmq::Context::new();
    let err = CoreLink::connect(&ctx, core_spec(&dir, 0), &never_interrupted)
        .err()
        .unwrap();
    assert!(err.contains("exited"), "{err}");
}

#[test]
fn a_link_refuses_a_core_that_stops_before_it_is_ready() {
    let dir = TempDir::new().unwrap();
    let spec = core_spec(&dir, std::process::id());
    let core = fake_core(spec.clone(), pb::envelope::Body::Stopped(pb::Stopped {}));

    let ctx = zmq::Context::new();
    let err = CoreLink::connect(&ctx, spec, &never_interrupted)
        .err()
        .unwrap();
    assert!(err.contains("stopped before it was ready"), "{err}");
    // The link sent the Core nothing.
    assert!(core.join().unwrap().is_none());
}

#[test]
fn a_link_gives_up_when_interrupted() {
    // The Core is this process, so it never counts as exited, and nothing
    // connects: only the interruption ends the wait.
    let dir = TempDir::new().unwrap();
    let ctx = zmq::Context::new();
    let err = CoreLink::connect(&ctx, core_spec(&dir, std::process::id()), &|| true)
        .err()
        .unwrap();
    assert!(err.contains("interrupted"), "{err}");
}

/// A client connected to fake Cores, one per `pools` entry.
struct Harness {
    client: EngineClient,
    cores: Vec<FakeCore>,
    _dir: TempDir,
}

impl Harness {
    fn start(pools: &[Option<u64>]) -> Self {
        let dir = TempDir::new().unwrap();
        let spec = test_support::engine_spec(
            dir.path(),
            pools.len(),
            test_support::model_spec("fake", "fake"),
        );
        let (client, cores) = test_support::connect(&spec, pools);
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
            n: Some(n),
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
                rank: None,
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
        rank: None,
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

#[tokio::test]
async fn an_explicit_rank_places_every_sibling_there() {
    let harness = Harness::start(&[None, None]);
    let submit = |rank| {
        harness.client.submit(Submission {
            request_id: "req",
            prompt: &[1, 2],
            sampling: pb::SamplingParams {
                temperature: Some(1.0),
                n: Some(3),
                ..Default::default()
            },
            stop_token_sequences: vec![],
            n: 3,
            session: None,
            rank: Some(rank),
            stream: false,
        })
    };
    let _handles = submit(1).unwrap();
    let (specs, _) = harness.cores[1].submitted().expect("rank 1 gets them all");
    assert_eq!(specs.len(), 3);
    assert!(FakeCore::recv_on(&harness.cores[0].input, Duration::from_millis(200)).is_none());
    assert!(matches!(submit(2), Err(SeqFailure::Invalid(_))));
}

#[tokio::test]
async fn a_logprobs_request_reports_the_prefix_hit_not_the_computed_kv() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, true);
    let (_, specs, _) = harness.first_submission();
    let core = &harness.cores[0];
    // The stream says 3 prompt tokens hit the cache ...
    core.stream(&[(specs[0].seq_id, &[5], Some("max_tokens"))]);
    // ... while the Completion counts all the prompt's KV as computed.
    let completion = pb::Completion {
        seq_id: specs[0].seq_id,
        completion_token_ids: Some(pb::BufferRef {
            frame: 1,
            offset: 0,
            nbytes: 4,
        }),
        num_prompt_tokens: 3,
        num_cached_tokens: 3_000,
        leave_reason: "max_tokens".into(),
        ..Default::default()
    };
    core.send(
        pb::envelope::Body::Completions(pb::Completions {
            items: vec![completion],
        }),
        vec![token_frame(&[5])],
    );
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => assert_eq!(done.cached_tokens, 3),
        other => panic!("expected Finished, got {other:?}"),
    }
}

#[tokio::test]
async fn an_ended_handle_dropped_later_sends_no_abort() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, false);
    let (_, specs, _) = harness.first_submission();
    let core = &harness.cores[0];
    core.stream(&[(specs[0].seq_id, &[9], Some("eos"))]);
    assert!(matches!(
        next_event(&mut handles[0]).await,
        SeqEvent::Finished(_)
    ));
    drop(handles);
    assert!(FakeCore::recv_on(&core.control, Duration::from_millis(200)).is_none());
    assert_eq!(harness.client.rank_loads(), (vec![0], vec![0]));
}

#[tokio::test]
async fn shutdown_fails_what_is_still_in_flight() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, true, false);
    let _ = harness.first_submission();
    // The Core never answers the Shutdown.
    harness.client.shutdown(Duration::from_millis(100));
    assert!(matches!(
        next_event(&mut handles[0]).await,
        SeqEvent::Failed(SeqFailure::Unavailable(_))
    ));
    assert_eq!(harness.client.lost(), None);
}
