use std::collections::HashMap;

use rmpv::Value;
use serde_json::Value as Json;

use super::*;

/// Payloads from the engines' own serializers (vLLM 3a104103, SGLang 0c44fbff, ATOM main
/// and ROCm/ATOM#3275), one JSON object per line with the events by field name.
const SAMPLES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/kv_events/samples.jsonl"
));

/// How every sample decodes under §3.3.4; a JSON object after the name is a `tier_map` to
/// decode it with.
const EXPECTED: &str = r#"
    vllm/stored_root                 | rank 0: stored device 2x4
    vllm/stored_child                | rank 0: stored device 1x4 child
    vllm/stored_bytes_hashes         | rank 0: stored device 2x4; stored device 1x4 child
    vllm/removed                     | rank 0: removed device 1; removed device 1
    vllm/all_cleared                 | rank 0: cleared
    vllm/dp_rank_1                   | rank 1: stored device 2x4
    vllm/salt_first_block            | rank 0: stored device 2x4 salt salt-abc; stored device 1x4 child
    vllm/lora_salt                   | rank 0: stored device 2x4 lora my-lora salt salt-abc
    vllm/multimodal                  | rank 0: stored device 1x4; drop namespace
    vllm/sliding_window_group        | rank 0: stored device 2x4; stored device 2x4 group 1:slidingwindow(128)
    vllm/lookahead_token_ids         | rank 0: stored device 2x4
    vllm/mooncake_cpu                | rank 0: drop unknowntier
    vllm/mooncake_cpu {"cpu":"drop"} | rank 0: drop storage
    vllm/umbp_cpu_local              | rank 0: stored host 1x4; removed host 1
    vllm/umbp_cpu_local {"CPU":"drop"} | rank 0: drop storage; drop storage
    vllm/ownership_kvcr              | rank 0: drop storage; drop storage
    vllm/locality_remote             | rank 0: drop storage; drop storage
    vllm/offloading_placeholder      | rank 0: stored host 1x0 untokened
    vllm/medium_nvme_unknown         | rank 0: drop unknowntier
    vllm/medium_storage              | rank 0: stored disk 2x4
    vllm/extra_unknown_field         | rank 0: stored device 2x4
    sglang/stored_root               | rank 0: stored device 2x4
    sglang/stored_child              | rank 0: stored device 1x4 child
    sglang/startup_all_cleared       | rank 0: cleared
    sglang/eagle_bigram              | rank 0: stored device 2x4
    sglang/cache_salt_every_page     | rank 0: stored device 2x4 salt salt-abc; stored device 1x4 child salt salt-abc
    sglang/hicache_cpu_pinned        | rank 0: stored device 2x4; stored host 2x4; removed device 2
    sglang/declared_disk_external    | rank 0: stored disk 1x4; drop storage
    sglang/dp_rank_1                 | rank 1: stored device 2x4
    atom/startup_clear               | rank 0: cleared
    atom/stored_root                 | rank 0: stored device 2x4
    atom/stored_child                | rank 0: stored device 1x4 child
    atom/removed                     | rank 0: removed device 1; removed device 1; removed device 1
    atom/remote_kv_gpu               | rank 0: stored device 2x4 child
    atom/multimodal_extra_keys       | rank 0: drop namespace
    atom/dp_rank_2                   | rank 2: stored device 2x4
    atom/main_stored_root            | rank 0: stored device 2x4
    atom/main_removed                | rank 0: removed device 1; removed device 1; removed device 1
"#;

fn render(batch: &DecodedBatch) -> String {
    let mut parts: Vec<String> = batch.events.iter().map(render_event).collect();
    parts.extend(
        batch
            .dropped
            .iter()
            .map(|reason| format!("drop {reason:?}")),
    );
    let rank = batch
        .dp_rank
        .map_or("-".to_string(), |rank| rank.to_string());
    format!("rank {rank}: {}", parts.join("; ")).to_lowercase()
}

fn render_event(event: &KvEvent) -> String {
    let KvEvent::Stored {
        parent,
        blocks,
        block_size,
        tier,
        group,
        ns,
    } = event
    else {
        return match event {
            KvEvent::Removed { hashes, tier, .. } => format!("removed {tier:?} {}", hashes.len()),
            _ => format!("{event:?}").replace(" { tier: None }", ""),
        };
    };
    let mut out = format!("stored {tier:?} {}x{block_size}", blocks.len());
    for (flag, text) in [
        (parent.is_some(), "child".to_string()),
        (
            blocks.iter().all(|block| block.tokens.is_none()),
            "untokened".to_string(),
        ),
        (
            *group != Group::default(),
            format!("group {}:{:?}", group.idx, group.kind),
        ),
    ] {
        if flag {
            out = format!("{out} {text}");
        }
    }
    for (key, value) in [("lora", &ns.lora), ("salt", &ns.salt)] {
        if let Some(value) = value {
            out = format!("{out} {key} {value}");
        }
    }
    out
}

fn unhex(hex: &str) -> Vec<u8> {
    let byte = |i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
    (0..hex.len()).step_by(2).map(byte).collect()
}

/// Hashes, parent and tokens of each event.
type Keys = Vec<(Vec<u64>, Option<u64>, Vec<u64>)>;

fn decoded_keys(batch: &DecodedBatch) -> Keys {
    let keys = |event: &KvEvent| match event {
        KvEvent::Stored { parent, blocks, .. } => {
            let tokens = blocks
                .iter()
                .flat_map(|block| block.tokens.as_deref().unwrap_or_default());
            let hashes = blocks.iter().map(|block| block.hash.0).collect();
            (
                hashes,
                parent.map(|hash| hash.0),
                tokens.map(|&t| t.into()).collect(),
            )
        }
        KvEvent::Removed { hashes, .. } => {
            (hashes.iter().map(|hash| hash.0).collect(), None, vec![])
        }
        KvEvent::Cleared { .. } => Default::default(),
    };
    batch.events.iter().map(keys).collect()
}

/// The same from the serialized events: integer hashes bit-cast, byte hashes by their low
/// 64 bits, EAGLE pairs by their first token.
fn input_keys(events: &Json) -> Keys {
    let hash = |value: &Json| match value.get("bytes_hex") {
        Some(hex) => unhex(hex.as_str().unwrap())
            .iter()
            .fold(0, |hash, &byte| hash << 8 | u64::from(byte)),
        None => value
            .as_u64()
            .unwrap_or_else(|| value.as_i64().unwrap() as u64),
    };
    let keys = |event: &Json| {
        let list = |field: &str| event[field].as_array().cloned().unwrap_or_default();
        let tokens = list("token_ids")
            .iter()
            .map(|token| token.get(0).unwrap_or(token).as_u64().unwrap())
            .collect();
        let parent = Some(&event["parent_block_hash"]).filter(|parent| !parent.is_null());
        (
            list("block_hashes").iter().map(hash).collect(),
            parent.map(hash),
            tokens,
        )
    };
    events.as_array().unwrap().iter().map(keys).collect()
}

#[test]
fn engine_samples_decode_as_specified() {
    let samples: HashMap<String, Json> = SAMPLES
        .lines()
        .map(|line| serde_json::from_str::<Json>(line).unwrap())
        .map(|sample| {
            (
                format!(
                    "{}/{}",
                    sample["engine"].as_str().unwrap(),
                    sample["name"].as_str().unwrap()
                ),
                sample,
            )
        })
        .collect();
    let mut decoded = HashMap::new();
    for (key, want) in EXPECTED.lines().filter_map(|line| line.split_once('|')) {
        let (name, tier_map) = key.trim().split_once(' ').unwrap_or((key.trim(), "{}"));
        let tier_map: TierMap = serde_json::from_str(tier_map).unwrap();
        let sample = &samples[name];
        let profile: WireProfile = serde_json::from_value(sample["profile"].clone()).unwrap();
        let payload = unhex(sample["payload"].as_str().unwrap());
        let batch = VllmWireDecoder::new(profile, tier_map.clone(), false)
            .decode(&payload)
            .unwrap();
        assert_eq!(render(&batch), want.trim(), "{key}");
        let strict = VllmWireDecoder::new(profile, tier_map, true).decode(&payload);
        let conforming = !batch
            .dropped
            .iter()
            .any(|reason| matches!(reason, DropReason::Decode | DropReason::UnknownTier));
        assert_eq!(strict.is_ok(), conforming, "{key}");
        if batch.dropped.is_empty() {
            assert_eq!(decoded_keys(&batch), input_keys(&sample["events"]), "{key}");
        }
        decoded.insert(name, batch);
    }
    assert_eq!(decoded.len(), samples.len(), "unchecked samples");
    let first = |name: &str| decoded_keys(&decoded[name]).remove(0).0;
    assert_eq!(first("vllm/stored_bytes_hashes"), first("vllm/stored_root"));
    for name in ["stored_root", "removed"] {
        let main = &decoded[&*format!("atom/main_{name}")];
        assert_eq!(main, &decoded[&*format!("atom/{name}")]);
    }
}

fn msgpack(json: Json) -> Value {
    match json {
        Json::Null => Value::Nil,
        Json::Bool(flag) => flag.into(),
        Json::Number(n) => n
            .as_u64()
            .map_or_else(|| n.as_f64().unwrap().into(), Value::from),
        Json::String(text) => text.into(),
        Json::Array(items) => Value::Array(items.into_iter().map(msgpack).collect()),
        Json::Object(fields) => Value::Map(
            fields
                .into_iter()
                .map(|(k, v)| (k.into(), msgpack(v)))
                .collect(),
        ),
    }
}

/// Decodes a batch written as JSON.
fn decode(batch: &str) -> Result<DecodedBatch, DecodeError> {
    let mut payload = Vec::new();
    rmpv::encode::write_value(&mut payload, &msgpack(serde_json::from_str(batch).unwrap()))
        .unwrap();
    VllmWireDecoder::default().decode(&payload)
}

#[test]
fn malformed_input_fails_the_batch_or_drops_the_event() {
    let malformed = VllmWireDecoder::default().decode(&[0xc1]);
    assert!(matches!(malformed, Err(DecodeError::Malformed(_))));
    for batch in ["1", "[1.0]", r#"{"ts": 1.0}"#] {
        assert!(matches!(decode(batch), Err(DecodeError::Malformed(_))));
    }
    let batch = decode(
        r#"[1.0, [{"type": "BlockStored", "block_hashes": [1, 2], "token_ids": [1, 2, 3, 4, 5, 6, 7], "block_size": 4},
            {"type": "BlockTransferred", "block_hashes": [1]}, "not an event"]]"#,
    );
    assert_eq!(
        render(&batch.unwrap()),
        "rank -: drop decode; drop decode; drop decode"
    );
}
