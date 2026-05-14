#![allow(unused_imports)]

use serde_json::{json, Value};

use super::backend::sglang::{SglangAdapter, SglangPairCtx};
use super::backend::vllm::{VllmAdapter, VllmPairCtx, VllmPrefillInfo};
use super::backend::BackendAdapter;
use super::test_support::*;
use super::types::AdapterError;

#[test]
fn e01_sglang_inject_prefill_writes_three_keys() {
    todo!()
}

#[test]
fn e02_sglang_inject_prefill_port_none_writes_null() {
    todo!()
}

#[test]
fn e03_sglang_inject_decode_is_noop() {
    todo!()
}

#[test]
fn e04_sglang_inject_batch_writes_three_arrays_of_size_n() {
    todo!()
}

#[test]
fn e05_sglang_inject_batch_room_ids_are_distinct() {
    todo!()
}

#[test]
fn e06_sglang_inject_on_non_object_returns_body_not_object() {
    todo!()
}

#[test]
fn e07_vllm_inject_prefill_kv_and_force_prefill_rewrites() {
    todo!()
}

#[test]
fn e08_vllm_inject_decode_lookups_and_shared_transfer_id() {
    todo!()
}

#[test]
fn e09_vllm_force_prefill_max_completion_tokens_only_overwrites() {
    todo!()
}

#[test]
fn e10_vllm_force_prefill_removes_stream_options() {
    todo!()
}

#[test]
fn e11_vllm_prepare_pair_missing_bootstrap_addr() {
    todo!()
}

#[test]
fn e12_vllm_prepare_pair_missing_engine_id() {
    todo!()
}
