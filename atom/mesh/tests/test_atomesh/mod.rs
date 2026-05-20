#![allow(dead_code, unused_imports)]

//! Fixture-driven Atomesh test framework.
//!
//! The modules below model the test architecture as data fixtures, real
//! Atomesh requests, virtual backend workers, and a harness that connects them
//! through the production router/app stack.

pub mod golden_assert;
pub mod mock_test_case;
pub mod replay_case_store;
pub mod test_harness;
pub mod virtual_grpc_worker;
pub mod virtual_request;
pub mod virtual_worker;

pub use golden_assert::{assert_json_contains, GoldenAssert};
pub use mock_test_case::{ConnectionModeFixture, MockTestCase, WorkerKindFixture};
pub use replay_case_store::ReplayCaseStore;
pub use test_harness::{TestHarness, TestHarnessResult};
pub use virtual_grpc_worker::VirtualGrpcWorker;
pub use virtual_request::VirtualRequest;
pub use virtual_worker::VirtualWorker;
