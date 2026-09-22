//! Envoy external processing ingress, sharing placement and worker state with HTTP ingress.

mod admission;
mod config;
mod error;
mod executor;
mod lifecycle;
mod mutation;
mod request;
mod response;
mod routing;
mod runtime;
mod service;
mod session;

pub use config::ExtProcConfig;
pub use runtime::ExtProcRuntime;

/// Official Envoy v1.37.0 and gRPC health protocol bindings.
#[allow(clippy::all, rustdoc::broken_intra_doc_links, unused_qualifications)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/ext_proc.rs"));
}

use proto::envoy::{config::core::v3 as core, service::ext_proc::v3 as pb};
