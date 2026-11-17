//! The RL rollout control plane under `/v1/rl`, speaking SMG's RL protocol.
//! Mounted by `server::build_app` only with `--enable-rl`.

pub mod capability;
pub mod config;
pub mod discovery;
pub mod error;
pub mod protocol;

use std::sync::Arc;

use axum::{routing::get, Router};
pub use config::RlConfig;
pub use error::RlError;

use crate::server::AppState;

/// The `/v1/rl` routes.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/rl/workers", get(discovery::list_workers))
        .route("/v1/rl/workers/{id}", get(discovery::get_worker))
}
