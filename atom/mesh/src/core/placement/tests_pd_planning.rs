#![allow(unused_imports)]

use std::sync::Arc;

use super::planner::DefaultPlanner;
use super::test_support::*;
use super::traits::PdPlanner;
use super::types::{PlacementError, PlacementPlan};

#[tokio::test]
async fn d01_http_1p1d_pair() {
    todo!()
}

#[tokio::test]
async fn d02_grpc_1p1d_pair() {
    todo!()
}

#[tokio::test]
async fn d03_pd_cross_model_isolation() {
    todo!()
}

#[tokio::test]
async fn d04_pd_hash_ring_keyed_by_real_model_id() {
    todo!()
}

#[tokio::test]
async fn d05_zero_prefill_returns_no_prefill_workers() {
    todo!()
}

#[tokio::test]
async fn d06_zero_decode_returns_no_decode_workers() {
    todo!()
}

#[tokio::test]
async fn d07_grpc_pd_uses_separated_policies() {
    todo!()
}

#[tokio::test]
async fn d08_http_pd_uses_separated_policies() {
    todo!()
}

#[tokio::test]
async fn d09_prefill_none_short_circuits_decode() {
    todo!()
}

#[tokio::test]
async fn d10_decode_none_returns_policy_returned_none_with_prefill_in_trace() {
    todo!()
}

#[tokio::test]
async fn d11_tokens_pass_to_both_pd_policies() {
    todo!()
}

#[tokio::test]
async fn d12_text_passes_to_both_pd_policies() {
    todo!()
}

#[tokio::test]
async fn d13_headers_pass_to_both_pd_policies() {
    todo!()
}

#[tokio::test]
async fn d14_pair_preserves_prefill_bootstrap_port() {
    todo!()
}
