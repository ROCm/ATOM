#![allow(unused_imports)]

use super::policy_apply::apply_policy;
use super::test_support::*;

#[tokio::test]
async fn b01_round_robin_invoked_once_idx_in_range() {
    todo!()
}

#[tokio::test]
async fn b02_random_invoked_once_returns_valid_idx() {
    todo!()
}

#[tokio::test]
async fn b03_power_of_two_invoked_once() {
    todo!()
}

#[tokio::test]
async fn b04_cache_aware_receives_request_text() {
    todo!()
}

#[tokio::test]
async fn b05_prefix_hash_receives_tokens() {
    todo!()
}

#[tokio::test]
async fn b06_hash_ring_keyed_by_real_model_id() {
    todo!()
}

#[tokio::test]
async fn b07_no_hash_ring_for_model_falls_back_gracefully() {
    todo!()
}

#[tokio::test]
async fn b08_request_text_passes_through() {
    todo!()
}

#[tokio::test]
async fn b09_tokens_pass_through() {
    todo!()
}

#[tokio::test]
async fn b10_headers_pass_through() {
    todo!()
}

#[tokio::test]
async fn b11_policy_returns_none_yields_typed_error() {
    todo!()
}

#[tokio::test]
async fn b12_empty_candidates_does_not_call_policy() {
    todo!()
}

#[test]
fn b13_pd_needs_request_text_aggregated_from_policies() {
    todo!()
}

#[tokio::test]
async fn b14_returned_worker_belongs_to_candidate_set() {
    todo!()
}
