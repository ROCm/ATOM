//! The wire framing, the launch spec and one Core's link: a message decodes
//! back with its buffers and a malformed one is refused, `EngineSpec` parses
//! a hand-written spec, and a `CoreLink` handshakes with a fake
//! Core on real ipc sockets and routes input and control to it.

use std::thread;

use prost::Message as _;
use serde_json::json;
use tempfile::TempDir;

use super::{
    link::{self, CoreLink},
    spec::{CoreSpec, EngineSpec},
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
            // A message the link sent elsewhere fails the test, not hangs it.
            socket.set_rcvtimeo(5000).unwrap();
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
