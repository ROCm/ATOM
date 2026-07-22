//! Metrics subsystem entry point.
//!
//! Business code should depend on the re-exported facade and helpers here rather
//! than reaching into individual modules for metric names or implementation
//! details. Engine metrics aggregation and Mesh self metrics exposure stay in
//! separate submodules to keep their scrape paths and failure semantics distinct.

pub mod config;
pub mod engine_metrics;
pub mod mesh_metrics;
pub mod recorder;
pub mod routes;
pub mod schema;

pub use config::PrometheusConfig;
pub use recorder::{MeshMetrics, StreamingMetricsParams};
pub use routes::MetricsRouteFactory;
pub use schema::{
    bool_to_static_str, labels as metrics_labels, method_to_static_str, normalize_path_for_metrics,
    status_code_to_cow, status_code_to_static_str, MetricKind, MetricSpec, MetricStatus,
    METRIC_INVENTORY,
};

pub(crate) use mesh_metrics::start_prometheus;
