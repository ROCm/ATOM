use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use super::backend::sglang::SglangAdapter;
use super::backend::vllm::{VllmAdapter, VllmPrefillInfo};
use super::backend::BackendAdapter;
use super::test_support::*;
use super::types::AdapterError;

fn sglang_pair() -> (SglangAdapter, super::backend::PairCtx) {
    let prefill = make_prefill_http("http://prefill-1:8000", "m", Some(8998));
    let decode = make_decode_http("http://decode-1:8000", "m");
    let adapter = SglangAdapter;
    let ctx = adapter
        .prepare_pair(prefill.as_ref(), decode.as_ref())
        .expect("prepare_pair");
    (adapter, ctx)
}

#[test]
fn e01_sglang_inject_prefill_writes_three_keys() {
    let (adapter, ctx) = sglang_pair();
    let mut body = json!({"prompt": "hi"});
    adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
    let obj = body.as_object().unwrap();
    assert_eq!(obj["bootstrap_host"], json!("prefill-1"));
    assert_eq!(obj["bootstrap_port"], json!(8998));
    assert!(obj["bootstrap_room"].is_u64());
    assert_eq!(obj["prompt"], json!("hi"));
}

#[test]
fn e02_sglang_inject_prefill_port_none_writes_null() {
    let prefill = make_prefill_http("http://prefill-2:8000", "m", None);
    let decode = make_decode_http("http://decode-2:8000", "m");
    let adapter = SglangAdapter;
    let ctx = adapter
        .prepare_pair(prefill.as_ref(), decode.as_ref())
        .unwrap();
    let mut body = json!({});
    adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
    assert_eq!(body["bootstrap_port"], Value::Null);
    assert!(body.as_object().unwrap().contains_key("bootstrap_port"));
}

#[test]
fn e03_sglang_inject_decode_is_noop() {
    let (adapter, ctx) = sglang_pair();
    let mut body = json!({"prompt": "hi"});
    let before = body.clone();
    adapter.inject_decode_fields(&mut body, &ctx).unwrap();
    assert_eq!(body, before);
}

#[test]
fn e04_sglang_inject_batch_writes_three_arrays_of_size_n() {
    let (adapter, ctx) = sglang_pair();
    let mut body = json!({});
    adapter
        .inject_batch_prefill_fields(&mut body, &ctx, 3)
        .unwrap();
    let obj = body.as_object().unwrap();
    assert_eq!(obj["bootstrap_host"].as_array().unwrap().len(), 3);
    assert_eq!(obj["bootstrap_port"].as_array().unwrap().len(), 3);
    assert_eq!(obj["bootstrap_room"].as_array().unwrap().len(), 3);
    for v in obj["bootstrap_host"].as_array().unwrap() {
        assert_eq!(v, &json!("prefill-1"));
    }
    for v in obj["bootstrap_port"].as_array().unwrap() {
        assert_eq!(v, &json!(8998));
    }
}

#[test]
fn e05_sglang_inject_batch_room_ids_are_distinct() {
    let (adapter, ctx) = sglang_pair();
    let mut body = json!({});
    adapter
        .inject_batch_prefill_fields(&mut body, &ctx, 3)
        .unwrap();
    let rooms: Vec<u64> = body["bootstrap_room"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let unique: std::collections::HashSet<_> = rooms.iter().collect();
    assert_eq!(unique.len(), 3);
}

#[test]
fn e06_sglang_inject_on_non_object_returns_body_not_object() {
    let (adapter, ctx) = sglang_pair();
    let mut body = json!([1, 2, 3]);
    let before = body.clone();
    let err = adapter.inject_prefill_fields(&mut body, &ctx).unwrap_err();
    assert_eq!(err, AdapterError::BodyNotObject);
    assert_eq!(body, before);

    let err = adapter
        .inject_batch_prefill_fields(&mut body, &ctx, 2)
        .unwrap_err();
    assert_eq!(err, AdapterError::BodyNotObject);
    assert_eq!(body, before);
}

fn vllm_info_with(prefill_url: &str, dp_rank: usize) -> Arc<VllmPrefillInfo> {
    let mut bootstrap_addrs = HashMap::new();
    bootstrap_addrs.insert(prefill_url.to_string(), "10.0.0.1:9000".to_string());
    let mut engine_ids = HashMap::new();
    let mut per_rank = HashMap::new();
    per_rank.insert(dp_rank, "engine-abc".to_string());
    engine_ids.insert(prefill_url.to_string(), per_rank);
    Arc::new(VllmPrefillInfo {
        bootstrap_addrs,
        engine_ids,
    })
}

fn vllm_pair_for(
    prefill_url: &str,
    dp_rank: usize,
) -> (VllmAdapter, super::backend::PairCtx) {
    let prefill = make_prefill_http(prefill_url, "m", None);
    let decode = make_decode_http("http://decode:8000", "m");
    let adapter = VllmAdapter::new(vllm_info_with(prefill_url, dp_rank));
    let ctx = adapter
        .prepare_pair(prefill.as_ref(), decode.as_ref())
        .expect("prepare_pair");
    (adapter, ctx)
}

#[test]
fn e07_vllm_inject_prefill_kv_and_force_prefill_rewrites() {
    let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
    let mut body = json!({
        "prompt": "hi",
        "stream": true,
        "max_tokens": 256,
        "max_completion_tokens": 100,
    });
    adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
    let obj = body.as_object().unwrap();
    let kv = &obj["kv_transfer_params"];
    assert_eq!(kv["do_remote_decode"], json!(true));
    assert_eq!(kv["do_remote_prefill"], json!(false));
    assert!(kv["transfer_id"]
        .as_str()
        .unwrap()
        .starts_with("xfer-"));
    assert_eq!(obj["stream"], json!(false));
    assert_eq!(obj["max_tokens"], json!(1));
    assert_eq!(obj["max_completion_tokens"], json!(1));
}

#[test]
fn e08_vllm_inject_decode_lookups_and_shared_transfer_id() {
    let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
    let mut prefill_body = json!({});
    adapter
        .inject_prefill_fields(&mut prefill_body, &ctx)
        .unwrap();
    let prefill_xfer = prefill_body["kv_transfer_params"]["transfer_id"]
        .as_str()
        .unwrap()
        .to_string();

    let mut decode_body = json!({});
    adapter
        .inject_decode_fields(&mut decode_body, &ctx)
        .unwrap();
    let kv = &decode_body["kv_transfer_params"];
    assert_eq!(kv["do_remote_decode"], json!(false));
    assert_eq!(kv["do_remote_prefill"], json!(true));
    assert_eq!(kv["remote_bootstrap_addr"], json!("10.0.0.1:9000"));
    assert_eq!(kv["remote_engine_id"], json!("engine-abc"));
    assert_eq!(kv["transfer_id"].as_str().unwrap(), prefill_xfer);
}

#[test]
fn e09_vllm_force_prefill_max_completion_tokens_only_overwrites() {
    let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
    let mut body = json!({"prompt": "hi"});
    adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
    let obj = body.as_object().unwrap();
    assert!(!obj.contains_key("max_completion_tokens"));
    assert_eq!(obj["max_tokens"], json!(1));
    assert_eq!(obj["stream"], json!(false));
}

#[test]
fn e10_vllm_force_prefill_removes_stream_options() {
    let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
    let mut body = json!({
        "prompt": "hi",
        "stream_options": {"include_usage": true},
    });
    adapter.inject_prefill_fields(&mut body, &ctx).unwrap();
    assert!(!body
        .as_object()
        .unwrap()
        .contains_key("stream_options"));
}

#[test]
fn e11_vllm_prepare_pair_missing_bootstrap_addr() {
    let prefill = make_prefill_http("http://unknown:8000", "m", None);
    let decode = make_decode_http("http://decode:8000", "m");
    let adapter = VllmAdapter::new(vllm_info_with("http://other:8000", 0));
    let err = adapter
        .prepare_pair(prefill.as_ref(), decode.as_ref())
        .unwrap_err();
    assert_eq!(
        err,
        AdapterError::BootstrapAddrMissing {
            prefill_url: "http://unknown:8000".to_string()
        }
    );
}

#[test]
fn vllm_inject_prefill_on_non_object_returns_body_not_object() {
    let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
    let mut body = json!([1, 2, 3]);
    let before = body.clone();
    let err = adapter.inject_prefill_fields(&mut body, &ctx).unwrap_err();
    assert_eq!(err, AdapterError::BodyNotObject);
    assert_eq!(body, before);
}

#[test]
fn vllm_inject_decode_on_non_object_returns_body_not_object() {
    let (adapter, ctx) = vllm_pair_for("http://p:8000", 0);
    let mut body = json!("not-an-object");
    let before = body.clone();
    let err = adapter.inject_decode_fields(&mut body, &ctx).unwrap_err();
    assert_eq!(err, AdapterError::BodyNotObject);
    assert_eq!(body, before);
}

#[test]
fn e12_vllm_prepare_pair_missing_engine_id() {
    let prefill = make_prefill_http("http://p:8000", "m", None);
    let decode = make_decode_http("http://decode:8000", "m");
    let adapter = VllmAdapter::new(vllm_info_with("http://p:8000", 7));
    let err = adapter
        .prepare_pair(prefill.as_ref(), decode.as_ref())
        .unwrap_err();
    assert_eq!(
        err,
        AdapterError::EngineIdMissing {
            prefill_url: "http://p:8000".to_string(),
            dp_rank: 0,
        }
    );
}
