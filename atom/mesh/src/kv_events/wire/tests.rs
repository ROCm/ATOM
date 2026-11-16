use std::collections::HashMap;

use serde_json::Value as Json;

use super::*;

/// Batches as ATOM publishes them; ATOM's tests/test_kv_events_mesh_samples.py produces
/// them again from its block manager and publisher and checks this file.
const SAMPLES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/kv_events/samples.jsonl"
));

/// Each sample decoded for a source with blocks of 4 tokens: its rank, then its events.
const EXPECTED: &str = r#"
startup_clear         | rank 0: cleared
stored_root           | rank 0: stored 2
stored_child          | rank 0: stored 1
removed               | rank 0: removed 1; removed 1; removed 1
remote_kv_gpu         | rank 0: stored 2
multimodal_extra_keys | rank 0: drop namespace
dp_rank_2             | rank 2: stored 2
"#;

fn render(batch: &DecodedBatch) -> String {
    let events = batch.events.iter().map(|event| match event {
        KvEvent::Stored { hashes } => format!("stored {}", hashes.len()),
        KvEvent::Removed { hashes } => format!("removed {}", hashes.len()),
        KvEvent::Cleared => "cleared".into(),
    });
    let dropped = batch
        .dropped
        .iter()
        .map(|reason| format!("drop {reason:?}"));
    let rank = batch.dp_rank.map_or("-".into(), |rank| rank.to_string());
    let parts: Vec<String> = events.chain(dropped).collect();
    format!("rank {rank}: {}", parts.join("; ")).to_lowercase()
}

fn unhex(hex: &str) -> Vec<u8> {
    let byte = |i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
    (0..hex.len()).step_by(2).map(byte).collect()
}

/// The block hashes of each event.
type Keys = Vec<Vec<u64>>;

fn decoded_keys(batch: &DecodedBatch) -> Keys {
    let keys = |event: &KvEvent| match event {
        KvEvent::Stored { hashes } | KvEvent::Removed { hashes } => {
            hashes.iter().map(|hash| hash.0).collect()
        }
        KvEvent::Cleared => vec![],
    };
    batch.events.iter().map(keys).collect()
}

/// The same, from the sample's events written as JSON.
fn input_keys(events: &Json) -> Keys {
    let hash = |value: &Json| {
        value
            .as_u64()
            .unwrap_or_else(|| value.as_i64().unwrap() as u64)
    };
    let keys = |event: &Json| {
        let hashes = event["block_hashes"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        hashes.iter().map(hash).collect()
    };
    events.as_array().unwrap().iter().map(keys).collect()
}

#[test]
fn atom_samples_decode_as_specified() {
    let samples: HashMap<String, Json> = SAMPLES
        .lines()
        .map(|line| serde_json::from_str::<Json>(line).unwrap())
        .map(|sample| (sample["name"].as_str().unwrap().to_string(), sample))
        .collect();
    let expected: Vec<_> = EXPECTED
        .lines()
        .filter_map(|line| line.split_once('|'))
        .collect();
    assert_eq!(expected.len(), samples.len(), "unchecked samples");
    for (name, want) in expected {
        let sample = &samples[name.trim()];
        let batch = decode(&unhex(sample["payload"].as_str().unwrap()), 4).unwrap();
        assert_eq!(render(&batch), want.trim(), "{name}");
        if batch.dropped.is_empty() {
            assert_eq!(
                decoded_keys(&batch),
                input_keys(&sample["events"]),
                "{name}"
            );
        }
    }
}

/// A batch written as JSON, encoded as msgpack.
pub(crate) fn encode(batch: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    let batch = rmpv::ext::to_value(serde_json::from_str::<Json>(batch).unwrap()).unwrap();
    rmpv::encode::write_value(&mut payload, &batch).unwrap();
    payload
}

fn decode_json(batch: &str, block_size: u32) -> Result<DecodedBatch, DecodeError> {
    decode(&encode(batch), block_size)
}

#[test]
fn only_the_device_tier_of_the_full_attention_group_is_followed() {
    let stored = |fields: &str| {
        let event = r#"{"type": "BlockStored", "block_hashes": [1], "token_ids": [1, 2, 3, 4]"#;
        format!(r#"[1.0, [{event}, "block_size": 4{fields}}}], 0]"#)
    };
    for (fields, want) in [
        (r#", "medium": "GPU""#, "rank 0: stored 1"),
        (r#", "medium": "CPU""#, "rank 0: drop tier"),
        (r#", "group_idx": 1"#, "rank 0: drop group"),
        (
            r#", "kv_cache_spec_kind": "sliding_window""#,
            "rank 0: drop group",
        ),
        (r#", "lora_name": "adapter""#, "rank 0: drop namespace"),
        (
            r#", "extra_keys": [null], "lora_id": null"#,
            "rank 0: stored 1",
        ),
    ] {
        let batch = decode_json(&stored(fields), 4).unwrap();
        assert_eq!(render(&batch), want, "{fields}");
    }
    let other_size = decode_json(&stored(""), 8).unwrap();
    assert_eq!(render(&other_size), "rank 0: drop blocksize");
    let negative = r#"[1.0, [{"type": "BlockRemoved", "block_hashes": [-1]}]]"#;
    assert_eq!(
        decode_json(negative, 4).unwrap().events,
        vec![KvEvent::Removed {
            hashes: vec![EngineHash(u64::MAX)]
        }]
    );
    let cleared =
        r#"[1.0, [{"type": "AllBlocksCleared", "medium": "CPU"}, {"type": "AllBlocksCleared"}]]"#;
    assert_eq!(
        render(&decode_json(cleared, 4).unwrap()),
        "rank -: cleared; drop tier"
    );
}

#[test]
fn malformed_input_fails_the_batch_or_drops_the_event() {
    assert!(decode(&[0xc1], 4).is_err());
    for batch in ["1", "[1.0]", r#"{"ts": 1.0}"#, "[1.0, [], -1]"] {
        assert!(decode_json(batch, 4).is_err(), "{batch}");
    }
    let batch = decode_json(
        r#"[1.0, [{"type": "BlockStored", "block_hashes": [1, 2], "token_ids": [1, 2, 3, 4, 5, 6, 7], "block_size": 4},
            {"type": "BlockStored", "block_hashes": ["1"], "token_ids": [1, 2, 3, 4], "block_size": 4},
            {"type": "BlockRemoved", "block_hashes": [1], "medium": 1},
            {"type": "BlockTransferred", "block_hashes": [1]},
            ["BlockRemoved", [1], "GPU", 0], "not an event"]]"#,
        4,
    );
    let decode = ["drop decode"; 6].join("; ");
    assert_eq!(render(&batch.unwrap()), format!("rank -: {decode}"));
}

/// Decoding the batches ATOM publishes: a decode step's, one block of 16 tokens an event,
/// and a prefill step's, one event of many blocks. `KV_BENCH_EVENTS` (96) and
/// `KV_BENCH_BLOCKS` (512) size them.
/// `cargo test --release --lib kv_events::wire::tests::bench -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_decode() {
    let size = |name: &str, default: usize| {
        std::env::var(name).map_or(default, |value| value.parse().unwrap())
    };
    let stored = |first: u64, blocks: u64| {
        let hashes: Vec<String> = (first..first + blocks)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15).to_string())
            .collect();
        let tokens: Vec<String> = (first * 16..(first + blocks) * 16)
            .map(|token| (token % 150_000).to_string())
            .collect();
        format!(
            r#"{{"type": "BlockStored", "block_hashes": [{}], "parent_block_hash": {first}, "token_ids": [{}], "block_size": 16, "lora_id": null, "medium": "GPU", "lora_name": null, "token_offset": 0}}"#,
            hashes.join(", "),
            tokens.join(", ")
        )
    };
    for (step, events, blocks) in [
        ("decode", size("KV_BENCH_EVENTS", 96), 1),
        ("prefill", 1, size("KV_BENCH_BLOCKS", 512)),
    ] {
        let batch: Vec<String> = (0..events as u64)
            .map(|event| stored(event * blocks as u64, blocks as u64))
            .collect();
        let payload = encode(&format!("[1.0, [{}], 0]", batch.join(", ")));
        let mut samples: Vec<_> = (0..2000)
            .map(|_| {
                let started = std::time::Instant::now();
                assert_eq!(decode(&payload, 16).unwrap().events.len(), events);
                started.elapsed()
            })
            .collect();
        samples.sort();
        let (p50, p99) = (samples[1000], samples[1980]);
        println!(
            "{step} step, {events} events of {blocks} blocks, {} bytes: p50 {p50:?}, p99 {p99:?}; {:?} an event",
            payload.len(),
            p50 / events as u32
        );
    }
}
