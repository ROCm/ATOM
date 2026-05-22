pub mod config;
pub mod schema;

#[path = "../metrics.rs"]
mod legacy;

pub use config::PrometheusConfig;
pub use legacy::{Metrics, StreamingMetricsParams};
pub use schema::{
    bool_to_static_str, labels as metrics_labels, method_to_static_str, normalize_path,
    status_code_to_cow, status_code_to_static_str, MetricKind, MetricSpec, MetricStatus,
    METRIC_INVENTORY,
};

pub(crate) use legacy::start_prometheus;
