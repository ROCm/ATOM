pub use super::legacy::StreamingMetricsParams;

/// Public metrics recording facade used by business paths.
///
/// Callers depend on this stable facade while the implementation still delegates
/// to the legacy recorder. The method bodies can move here later without
/// changing router, middleware, or worker call sites again.
pub type MeshMetrics = super::legacy::Metrics;

/// Temporary compatibility alias for call sites that have not migrated yet.
pub type Metrics = MeshMetrics;
