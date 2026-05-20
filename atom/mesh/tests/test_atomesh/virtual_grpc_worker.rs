//! Reserved boundary for tonic-backed Atomesh virtual gRPC worker tests.
//!
//! The HTTP harness is fully executable today. gRPC fixtures should still be
//! loadable and classified through the same fixture schema, but the actual
//! tonic server is kept behind this type so future work has a clear extension
//! point instead of mixing gRPC transport details into the HTTP harness.

use super::mock_test_case::{ConnectionModeFixture, MockTestCase, WorkerKindFixture};

pub const GRPC_TRANSPORT_NOT_IMPLEMENTED: &str =
    "VirtualGrpcWorker tonic transport is not implemented yet";

#[derive(Debug)]
pub struct VirtualGrpcWorker {
    case_name: String,
    worker_kind: WorkerKindFixture,
}

impl VirtualGrpcWorker {
    pub fn new(case: MockTestCase) -> Result<Self, Box<dyn std::error::Error>> {
        if case.route.connection_mode != ConnectionModeFixture::Grpc {
            return Err("VirtualGrpcWorker requires a gRPC fixture".into());
        }

        Ok(Self {
            case_name: case.name.clone(),
            worker_kind: case.route.worker_kind.clone(),
        })
    }

    pub fn case_name(&self) -> &str {
        &self.case_name
    }

    pub fn worker_kind(&self) -> &WorkerKindFixture {
        &self.worker_kind
    }

    pub async fn start(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        Err(GRPC_TRANSPORT_NOT_IMPLEMENTED.into())
    }
}
