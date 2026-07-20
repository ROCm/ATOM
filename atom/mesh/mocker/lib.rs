#![allow(dead_code)]

//! Fixture-driven Atomesh mocker utilities.
//!
//! This crate contains reusable mocker primitives for Atomesh integration tests:
//! fixture schemas, virtual requests, replay stores, virtual backend workers,
//! golden response assertions, and full harness execution.

pub mod app_helpers;
pub mod test_harness;
pub mod virtual_workers;

pub use test_harness::{TestHarness, TestHarnessResult};
pub use virtual_workers::{
    any_json_contains, assert_any_json_contains, assert_json_contains, json_contains,
    BackendFixture, ConnectionModeFixture, GoldenAssert, MockCase, ReplayCaseStore,
    VirtualGrpcWorker, VirtualRequest, VirtualRequestEndpoint, VirtualRequestMode,
    VirtualRequestPipeline, VirtualRequestPipelineConfig, VirtualRequestPipelineResult,
    VirtualResponse, VirtualWorker, VirtualWorkerInstance, VirtualWorkerPool,
    VirtualWorkerPoolConfig, VirtualWorkerSpec, WorkerKindFixture,
};
