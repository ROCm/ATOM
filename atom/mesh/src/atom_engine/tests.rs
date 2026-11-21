//! The wire encoding, the launch spec, one Core's link and the client: a
//! message decodes back with its buffers and a malformed one is refused,
//! `EngineSpec` parses a hand-written spec, a `CoreLink` handshakes with a
//! fake Core on real ipc sockets and routes input and control to it, and gives
//! up on a Core that stops before it is ready, speaks another protocol version
//! or exits, or when interrupted, and `EngineClient` drives fake Cores
//! (`test_support`) through the protocol's every ending.
//!
//! The protocol across languages: the Core's messages as Python encodes them
//! (`tests/fixtures/atom_engine/core_*`, written by
//! `tests/test_engine_wire_cross_language.py`) drive the client here, and the
//! client's own messages are kept as `client_*` there for Python to decode.
//! `ATOM_UPDATE_FIXTURES=1` rewrites the `client_*` ones.

use std::{future::poll_fn, path::PathBuf, thread, time::Duration};

use serde_json::json;
use tempfile::TempDir;

use super::{
    client::{EngineClient, SeqEvent, SeqFailure, SeqHandle, Submission},
    link::{self, CoreLink},
    spec::{BalanceStrategy, CoreSpec, EngineSpec},
    test_support::{self, decoded, token_bytes, FakeCore, WAIT},
    wire::{self, kind, reject, Inbound, PROTOCOL_VERSION},
};

fn float_bytes(values: &[f64]) -> serde_bytes::ByteBuf {
    serde_bytes::ByteBuf::from(
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}

#[test]
fn a_completion_decodes_back_with_its_tokens_and_logprobs() {
    let completion = wire::Completion {
        seq_id: 4,
        completion_tokens: token_bytes(&[8, -1]),
        logprobs_bytes: Some(float_bytes(&[0.5, -1.25])),
        leave_reason: "stop".into(),
        ..Default::default()
    };
    let message = wire::encode(
        kind::ADD,
        &wire::Completions {
            items: vec![completion.clone()],
        },
    );
    let Inbound::Completions(done) = Inbound::decode(&message).unwrap() else {
        panic!("not Completions");
    };
    assert_eq!(done.items, [completion]);
    let item = &done.items[0];
    // u32::MAX crosses as int32 -1 and comes back as itself.
    assert_eq!(
        wire::tokens(&item.completion_tokens).unwrap(),
        [8, u32::MAX]
    );
    let logprobs = item.logprobs_bytes.as_ref().unwrap();
    assert_eq!(wire::floats(logprobs).unwrap(), [0.5, -1.25]);
}

#[test]
fn a_malformed_message_is_refused_and_an_unknown_kind_ignored() {
    assert!(Inbound::decode(&[]).is_err());
    assert!(Inbound::decode(b"not msgpack").is_err());
    // Three items, not [kind, body].
    let err = Inbound::decode(&rmp_serde::to_vec(&(6u8, 1u8, 2u8)).unwrap())
        .err()
        .unwrap();
    assert!(err.contains("[kind, body]"), "{err}");
    // A Stream whose body is not one.
    assert!(Inbound::decode(&rmp_serde::to_vec(&(kind::STREAM, "x")).unwrap()).is_err());
    // A kind only the Python client reads (METRICS) is no error.
    let metrics = rmp_serde::to_vec(&(9u8, (serde_bytes::Bytes::new(b"pickled"),))).unwrap();
    assert!(matches!(Inbound::decode(&metrics), Ok(Inbound::Ignored(9))));
    // Not a whole number of elements.
    assert!(wire::tokens(&[0; 6]).is_err());
    assert!(wire::floats(&[0; 12]).is_err());
}

#[test]
fn a_reader_fills_missing_trailing_fields_and_skips_unknown_ones() {
    // A Completion from an older Core: no fields past `leave_reason`.
    let old = (
        9u64,
        serde_bytes::Bytes::new(&[]),
        None::<()>,
        false,
        3u32,
        0u64,
        1.0f64,
        2.0f64,
        "eos",
    );
    let message = rmp_serde::to_vec(&(kind::ADD, (vec![old],))).unwrap();
    let Inbound::Completions(done) = Inbound::decode(&message).unwrap() else {
        panic!("not Completions");
    };
    assert_eq!(done.items[0].seq_id, 9);
    assert_eq!(done.items[0].leave_reason, "eos");
    assert_eq!(done.items[0].reject_code, reject::FINISHED);

    // A Ready from a newer Core: one field this client does not know.
    let newer = (PROTOCOL_VERSION, Some(64u64), json!({}), "a later field");
    let message = rmp_serde::to_vec(&(kind::READY, newer)).unwrap();
    let Inbound::Ready(ready) = Inbound::decode(&message).unwrap() else {
        panic!("not Ready");
    };
    assert_eq!(ready.max_pool_tokens, Some(64));
}

fn spec_json() -> serde_json::Value {
    json!({
        "model": {
            "served_name": "m", "tokenizer_path": "/m", "default_max_tokens": 16,
            "template_efforts": ["low", "high", "max"],
        },
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
/// back the next message on input and on control (kind and body), or none
/// if either does not come within the receive timeout.
/// A message's kind and body.
type Received = (u8, Vec<u8>);

fn fake_core(spec: CoreSpec, first: Vec<u8>) -> thread::JoinHandle<Option<[Received; 2]>> {
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
        output.send(first, 0).unwrap();
        let receive = |socket: &zmq::Socket| {
            let frames = link::recv_frames(socket).ok()?;
            let (kind, body) = wire::split(&frames[0]).ok()?;
            Some((kind, body.to_vec()))
        };
        Some([receive(&input)?, receive(&control)?])
    })
}

fn never_interrupted() -> bool {
    false
}

fn ready(max_pool_tokens: Option<u64>) -> Vec<u8> {
    wire::encode(
        kind::READY,
        &wire::Ready {
            max_pool_tokens,
            ..Default::default()
        },
    )
}

#[test]
fn a_link_handshakes_and_routes_input_and_control_to_their_core() {
    let dir = TempDir::new().unwrap();
    let spec = core_spec(&dir, std::process::id());
    let core = fake_core(spec.clone(), ready(Some(4096)));

    let ctx = zmq::Context::new();
    let (link, _output, max_pool_tokens) =
        CoreLink::connect(&ctx, spec, &never_interrupted).unwrap();
    assert_eq!(max_pool_tokens, Some(4096));

    let submit = wire::Submit {
        items: vec![wire::SubmitItem {
            prompt: token_bytes(&[1, 2, 3]),
            ..Default::default()
        }],
    };
    link.send_input(wire::encode(kind::ADD, &submit)).unwrap();
    let abort = wire::Abort { seq_ids: vec![7] };
    link.send_control(wire::encode(kind::ABORT, &abort))
        .unwrap();

    let [(input_kind, input), (control_kind, control)] = core
        .join()
        .unwrap()
        .expect("the core received both messages");
    assert_eq!(input_kind, kind::ADD);
    let batch: wire::Submit = decoded(&input);
    assert_eq!(wire::tokens(&batch.items[0].prompt).unwrap(), [1, 2, 3]);
    assert_eq!(control_kind, kind::ABORT);
    assert_eq!(decoded::<wire::Abort>(&control).seq_ids, [7]);
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
    let core = fake_core(spec.clone(), wire::encode(kind::SHUTDOWN, &wire::Empty {}));

    let ctx = zmq::Context::new();
    let err = CoreLink::connect(&ctx, spec, &never_interrupted)
        .err()
        .unwrap();
    assert!(err.contains("stopped before it was ready"), "{err}");
    // The link sent the Core nothing.
    assert!(core.join().unwrap().is_none());
}

#[test]
fn a_link_refuses_a_core_of_another_protocol_version() {
    let dir = TempDir::new().unwrap();
    let spec = core_spec(&dir, std::process::id());
    let newer = wire::Ready {
        protocol_version: PROTOCOL_VERSION + 1,
        ..Default::default()
    };
    let core = fake_core(spec.clone(), wire::encode(kind::READY, &newer));

    let ctx = zmq::Context::new();
    let err = CoreLink::connect(&ctx, spec, &never_interrupted)
        .err()
        .unwrap();
    assert!(err.contains("protocol"), "{err}");
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
        let sampling = wire::Sampling {
            temperature: 1.0,
            max_tokens: 8,
            n,
            logprobs: logprobs.then_some(true),
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
    fn first_submission(&self) -> (usize, Vec<wire::SubmitItem>) {
        for (rank, core) in self.cores.iter().enumerate() {
            if let Some((kind, body)) = FakeCore::recv_on(&core.input, Duration::from_millis(500)) {
                assert_eq!(kind, kind::ADD, "expected Submit");
                return (rank, decoded::<wire::Submit>(&body).items);
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
    let (rank, items) = harness.first_submission();
    let [item] = &items[..] else {
        panic!("expected one SubmitItem, got {items:?}")
    };
    assert_eq!(item.data_parallel_rank, Some(rank as i64));
    assert_eq!(item.external_request_id.as_deref(), Some("req"));
    assert!(!item.needs_independent_noise);
    assert_eq!(wire::tokens(&item.prompt).unwrap(), [11, 12, 13]);

    let core = &harness.cores[rank];
    core.stream(&[(item.seq_id, &[7], None)]);
    core.stream(&[(item.seq_id, &[8], Some("eos"))]);
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

fn completions(items: Vec<wire::Completion>) -> wire::Completions {
    wire::Completions { items }
}

#[tokio::test]
async fn a_logprobs_request_ends_on_its_completion() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, true);
    let (_, items) = harness.first_submission();
    let seq_id = items[0].seq_id;
    let core = &harness.cores[0];
    // Its finished stream entry does not end it: only the Completion has logprobs.
    core.stream(&[(seq_id, &[5, 6], Some("max_tokens"))]);
    let completion = wire::Completion {
        seq_id,
        completion_tokens: token_bytes(&[5, 6]),
        logprobs_bytes: Some(float_bytes(&[-0.5, -1.25])),
        num_prompt_tokens: 3,
        leave_reason: "max_tokens".into(),
        ..Default::default()
    };
    core.send(kind::ADD, &completions(vec![completion]));
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
async fn refusals_map_to_failures_by_code() {
    let harness = Harness::start(&[None]);
    for (code, expect_invalid) in [(reject::INVALID, true), (reject::OVERLOADED, false)] {
        let mut handles = harness.submit(1, false, false);
        let (_, items) = harness.first_submission();
        let refused = wire::Completion {
            seq_id: items[0].seq_id,
            leave_reason: "rejected: no".into(),
            reject_code: code,
            ..Default::default()
        };
        harness.cores[0].send(kind::ADD, &completions(vec![refused]));
        match next_event(&mut handles[0]).await {
            SeqEvent::Failed(SeqFailure::Invalid(reason)) if expect_invalid => {
                assert_eq!(reason, "no")
            }
            SeqEvent::Failed(SeqFailure::Unavailable(_)) if !expect_invalid => {}
            other => panic!("{code}: unexpected {other:?}"),
        }
    }
}

#[tokio::test]
async fn dropping_an_unfinished_handle_aborts_its_sequence() {
    let harness = Harness::start(&[None]);
    let handles = harness.submit(1, true, false);
    let (_, items) = harness.first_submission();
    drop(handles);
    let (kind, body) = harness.cores[0].control_message();
    assert_eq!(kind, kind::ABORT);
    assert_eq!(decoded::<wire::Abort>(&body).seq_ids, [items[0].seq_id]);
}

#[tokio::test]
async fn siblings_spread_over_ranks_each_with_the_prompt() {
    let harness = Harness::start(&[None, None]);
    let handles = harness.submit(4, false, false);
    assert_eq!(handles.len(), 4);
    let mut siblings = Vec::new();
    for core in &harness.cores {
        let items = core.submitted().expect("each rank gets its share");
        for item in items {
            assert!(item.needs_independent_noise);
            assert_eq!(item.external_request_id, None);
            assert_eq!(item.parent_request_id.as_deref(), Some("req"));
            assert_eq!(wire::tokens(&item.prompt).unwrap(), [11, 12, 13]);
            siblings.push(item.sibling_index);
        }
    }
    siblings.sort();
    assert_eq!(siblings, [0, 1, 2, 3]);
}

#[tokio::test]
async fn a_core_that_stops_unasked_loses_the_client() {
    let harness = Harness::start(&[None, None]);
    let mut handles = harness.submit(1, true, false);
    let (rank, _) = harness.first_submission();
    assert_eq!(harness.client.lost(), None);
    harness.cores[rank].send(kind::SHUTDOWN, &wire::Empty {});
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
async fn an_undecodable_output_loses_its_core() {
    // One message that is not msgpack, one Stream that is not a Stream:
    // either may have held a sequence's end, so the Core is lost with them.
    let bad_outputs = [
        b"not msgpack".to_vec(),
        rmp_serde::to_vec(&(kind::STREAM, "x")).unwrap(),
    ];
    for message in bad_outputs {
        let harness = Harness::start(&[None, None]);
        let mut handles = harness.submit(1, true, false);
        let (rank, _) = harness.first_submission();
        harness.cores[rank].send_raw(message);
        assert!(matches!(
            next_event(&mut handles[0]).await,
            SeqEvent::Failed(SeqFailure::Unavailable(_))
        ));
        let reason = tokio::time::timeout(WAIT, harness.client.wait_lost())
            .await
            .expect("the loss is reported");
        assert!(
            reason.starts_with(&format!(
                "EngineCore DP rank {rank} sent an undecodable output"
            )),
            "{reason}"
        );
        assert_eq!(harness.client.alive_core_count(), 1);
        assert_eq!(harness.client.rank_loads().0[rank], 0);
    }
}

#[tokio::test]
async fn shutdown_asks_every_core_to_stop_and_waits_for_them() {
    let harness = Harness::start(&[None, None]);
    // Answer each Shutdown with Stopped from a thread, as a Core would.
    let cores = harness.cores;
    let replies = thread::spawn(move || {
        for core in &cores {
            assert_eq!(core.control_message().0, kind::SHUTDOWN);
            core.send(kind::SHUTDOWN, &wire::Empty {});
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
            sampling: wire::Sampling {
                n: 3,
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
    let items = harness.cores[1].submitted().expect("rank 1 gets them all");
    assert_eq!(items.len(), 3);
    assert!(FakeCore::recv_on(&harness.cores[0].input, Duration::from_millis(200)).is_none());
    assert!(matches!(submit(2), Err(SeqFailure::Invalid(_))));
}

#[tokio::test]
async fn a_logprobs_request_reports_the_prefix_hit_not_the_computed_kv() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, true);
    let (_, items) = harness.first_submission();
    let core = &harness.cores[0];
    // The stream says 3 prompt tokens hit the cache ...
    core.stream(&[(items[0].seq_id, &[5], Some("max_tokens"))]);
    // ... while the Completion counts all the prompt's KV as computed.
    let completion = wire::Completion {
        seq_id: items[0].seq_id,
        completion_tokens: token_bytes(&[5]),
        num_prompt_tokens: 3,
        num_cached_tokens: 3_000,
        leave_reason: "max_tokens".into(),
        ..Default::default()
    };
    core.send(kind::ADD, &completions(vec![completion]));
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => assert_eq!(done.cached_tokens, 3),
        other => panic!("expected Finished, got {other:?}"),
    }
}

#[tokio::test]
async fn an_ended_handle_dropped_later_sends_no_abort() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, false);
    let (_, items) = harness.first_submission();
    let core = &harness.cores[0];
    core.stream(&[(items[0].seq_id, &[9], Some("eos"))]);
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

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/atom_engine")
}

fn fixture(name: &str) -> PathBuf {
    fixture_dir().join(format!("{name}.msgpack"))
}

/// A message as the Python test keeps it: its one frame's bytes.
fn read_fixture(name: &str) -> Vec<u8> {
    let path = fixture(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The next message on `socket`, as sent.
fn raw_message(socket: &zmq::Socket) -> Vec<u8> {
    assert!(
        socket.poll(zmq::POLLIN, WAIT.as_millis() as i64).unwrap() > 0,
        "no message within the timeout"
    );
    socket.recv_bytes(0).unwrap()
}

/// A Submit with its arrival times zeroed: they are the clock's.
fn timeless(message: &[u8]) -> wire::Submit {
    let (kind, body) = wire::split(message).unwrap();
    assert_eq!(kind, kind::ADD);
    let mut submit: wire::Submit = decoded(body);
    submit
        .items
        .iter_mut()
        .for_each(|item| item.arrive_time = 0.0);
    submit
}

/// Fail when any of the client's `messages` drifted from its fixture, which
/// Python decodes; under `ATOM_UPDATE_FIXTURES`, rewrite them all and fail (a
/// rewrite is not a pass).
fn check_client_fixtures(messages: &[(&str, Vec<u8>)]) {
    if std::env::var_os("ATOM_UPDATE_FIXTURES").is_some() {
        for (name, message) in messages {
            std::fs::write(fixture(name), message).unwrap();
        }
        panic!("rewrote the client_* fixtures; rerun without ATOM_UPDATE_FIXTURES");
    }
    for (name, message) in messages {
        let kept = read_fixture(name);
        let same = if *name == "client_submit" {
            timeless(message) == timeless(&kept)
        } else {
            *message == kept
        };
        assert!(
            same,
            "{name}.msgpack is stale: rerun with ATOM_UPDATE_FIXTURES=1, then \
             tests/test_engine_wire_cross_language.py against it"
        );
    }
}

#[tokio::test]
async fn the_clients_messages_match_what_python_decodes() {
    let harness = Harness::start(&[None]);
    let sampling = wire::Sampling {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.9,
        max_tokens: 64,
        stop_strings: Some(vec!["\n\n".into()]),
        n: 2,
        logprobs: Some(true),
        required_channel: Some(wire::RequiredChannel {
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
            stop_token_sequences: vec![vec![5, 6], vec![7]],
            n: 2,
            session: Some("sess"),
            rank: None,
            stream: false,
        })
        .unwrap();
    let core = &harness.cores[0];
    let submit = raw_message(&core.input);
    drop(handles);
    let abort = raw_message(&core.control);
    let _second_abort = raw_message(&core.control);
    harness.client.shutdown(Duration::from_millis(100));
    let shutdown = raw_message(&core.control);
    check_client_fixtures(&[
        ("client_submit", submit),
        ("client_abort", abort),
        ("client_shutdown", shutdown),
    ]);
}

#[tokio::test]
async fn python_core_messages_drive_the_client() {
    let decode =
        |name| Inbound::decode(&read_fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let Inbound::Ready(ready) = decode("core_ready") else {
        panic!("core_ready is not Ready");
    };
    assert_eq!(ready.protocol_version, PROTOCOL_VERSION);
    assert_eq!(ready.max_pool_tokens, Some(4096));
    assert!(matches!(decode("core_stopped"), Inbound::Stopped));
    // Only Python reads these.
    assert!(matches!(decode("core_metrics"), Inbound::Ignored(_)));

    // Each harness's first sequence is 0, the one the fixtures name.
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, true, false);
    harness.cores[0].send_raw(read_fixture("core_stream"));
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
    harness.cores[0].send_raw(read_fixture("core_completions"));
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => {
            assert_eq!(done.token_ids, [5, 6]);
            assert_eq!(done.logprobs.as_deref(), Some(&[-0.5, -1.25][..]));
            // The prefix hit comes from the stream, and this request had no
            // stream entry; the Completion's count is the KV computed.
            assert_eq!((done.prompt_tokens, done.cached_tokens), (3, 0));
            assert_eq!(done.leave_reason, "max_tokens");
        }
        other => panic!("expected Finished, got {other:?}"),
    }

    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, false);
    harness.cores[0].send_raw(read_fixture("core_rejected"));
    assert!(matches!(
        next_event(&mut handles[0]).await,
        SeqEvent::Failed(SeqFailure::Invalid(reason)) if reason == "ValueError: bad"
    ));
}

#[test]
fn the_spec_python_writes_parses() {
    // Written by tests/entrypoints/test_engine_spec.py (`build_engine_spec`).
    let path = fixture_dir().join("engine_spec.json");
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let spec = EngineSpec::from_json(&json).unwrap();
    let model = &spec.model;
    assert_eq!(model.served_name, "served");
    assert_eq!(model.tokenizer_path, "/models/m");
    assert_eq!(model.chat_template.as_deref(), Some("/t/chat.jinja"));
    assert_eq!(model.default_chat_template_kwargs["k"], json!(1));
    assert_eq!(
        (model.max_model_len, model.max_output_tokens),
        (Some(4096), Some(2048))
    );
    assert_eq!(model.model_card["context_window"], json!(2048));
    assert!(!model.synthetic_acceptance);
    assert_eq!(model.synthetic_token_text.as_deref(), Some("synthetic "));
    assert!(model.preserve_tool_call_arguments);
    let channel = model.required_channel.as_ref().unwrap();
    assert_eq!(
        (channel.after.as_slice(), channel.opening.as_slice()),
        (&[1][..], &[2][..])
    );
    assert_eq!(
        (channel.opens_at, channel.end_ids.as_slice()),
        (1, &[3, 7, 9][..])
    );
    let toggle = model.reasoning_toggle.as_ref().unwrap();
    assert_eq!(
        (toggle.name.as_str(), &toggle.off, &toggle.on),
        ("enable_thinking", &json!(false), &json!(true))
    );
    assert_eq!(spec.balance.strategy, BalanceStrategy::LeastRequests);
    assert_eq!(spec.engines.len(), 2);
    let core = &spec.engines[1];
    assert_eq!((core.dp_rank, core.pid), (1, 101));
    assert_eq!(core.input, "ipc:///tmp/core-1-input");
}
