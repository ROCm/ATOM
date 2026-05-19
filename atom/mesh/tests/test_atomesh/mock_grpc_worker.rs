/// Placeholder for the future tonic-backed virtual worker.
///
/// The P0 refactor keeps gRPC out of the default test path while preserving the
/// module boundary described in the design doc.
#[derive(Debug, Default)]
pub struct MockGrpcWorker;
