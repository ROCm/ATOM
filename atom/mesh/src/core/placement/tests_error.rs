#![allow(unused_imports)]

use super::test_support::*;
use super::types::{AdapterError, PlacementError};

#[tokio::test]
async fn g01_no_workers_triggered_by_empty_registry_and_no_model() {
    todo!()
}

#[tokio::test]
async fn g02_no_available_workers_triggered_by_all_unhealthy() {
    todo!()
}

#[tokio::test]
async fn g03_no_prefill_workers_triggered_in_pd_path() {
    todo!()
}

#[tokio::test]
async fn g04_no_decode_workers_triggered_in_pd_path() {
    todo!()
}

#[tokio::test]
async fn g05_policy_returned_none_triggered_by_always_none_policy() {
    todo!()
}

#[tokio::test]
async fn g06_model_not_found_triggered_when_model_id_unknown() {
    todo!()
}

#[tokio::test]
async fn g07_no_available_workers_in_pd_path_when_all_unhealthy() {
    todo!()
}

#[test]
fn g08_adapter_errors_map_to_5xx_response() {
    todo!()
}

#[test]
fn g09_error_display_includes_key_fields() {
    todo!()
}
