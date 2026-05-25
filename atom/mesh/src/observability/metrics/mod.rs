pub mod config;
pub mod mesh_metrics;
pub mod recorder;
pub mod schema;

#[path = "../metrics.rs"]
mod legacy;

pub use config::PrometheusConfig;
pub use recorder::{MeshMetrics, Metrics, StreamingMetricsParams};
pub use schema::{
    METRIC_INVENTORY, MetricKind, MetricSpec, MetricStatus, bool_to_static_str,
    labels as metrics_labels, method_to_static_str, normalize_path, status_code_to_cow,
    status_code_to_static_str,
};

pub(crate) use mesh_metrics::start_prometheus;
