//! The RL rollout control plane under `/v1/rl`, speaking SMG's RL protocol.
//! Mounted by `server::build_app` only with `--enable-rl`.

pub mod capability;
pub mod config;
pub mod discovery;
pub mod error;
pub mod fanout;
pub mod path;
pub mod protocol;
pub mod proxy;
pub mod selector;

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::header::AUTHORIZATION,
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
pub use config::RlConfig;
pub use error::RlError;

use crate::server::AppState;

/// The `/v1/rl` routes, each requiring `Authorization: Bearer <api_key>`
/// (scheme in any case). With no `api_key` every request is refused.
pub fn router(config: &RlConfig) -> Router<Arc<AppState>> {
    let token: Arc<str> = config.api_key.as_deref().unwrap_or_default().into();
    Router::new()
        .route("/v1/rl/workers", get(discovery::list_workers))
        .route("/v1/rl/workers/{id}", get(discovery::get_worker))
        .route(
            "/v1/rl/workers/{id}/engine/{*path}",
            get(proxy::proxy_handler).post(proxy::proxy_handler),
        )
        .route(
            "/v1/rl/engine/{*path}",
            get(fanout::fanout_handler).post(fanout::fanout_handler),
        )
        .route_layer(from_fn_with_state(token, require_bearer))
}

async fn require_bearer(State(token): State<Arc<str>>, request: Request, next: Next) -> Response {
    let presented = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map_or("", |(_, token)| token);
    if token.is_empty() || !constant_time_eq(presented.as_bytes(), token.as_bytes()) {
        return RlError::Unauthorized.into_response();
    }
    next.run(request).await
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0
}
