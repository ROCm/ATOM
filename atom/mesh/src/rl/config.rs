//! RL control-plane configuration, embedded in `RouterConfig` as `rl`.

use serde::{Deserialize, Serialize};

/// Default `--rl-control-timeout-secs`.
pub const DEFAULT_RL_CONTROL_TIMEOUT_SECS: u64 = 600;

/// Configuration for the `/v1/rl` routes. Inert unless `enabled`.
#[derive(Debug, Clone, PartialEq, Eq, clap::Args, Serialize, Deserialize)]
#[command(next_help_heading = "RL Control Plane")]
#[serde(default)]
pub struct RlConfig {
    /// Mount the RL control plane under /v1/rl.
    #[arg(id = "enable_rl", long = "enable-rl")]
    pub enabled: bool,
    /// Bearer token every /v1/rl request must carry as
    /// `Authorization: Bearer <token>`; required with --enable-rl.
    #[arg(id = "rl_api_key", long = "rl-api-key")]
    pub api_key: Option<String>,
    /// Total timeout of one engine control call (a refit can take minutes).
    #[arg(
        id = "rl_control_timeout_secs",
        long = "rl-control-timeout-secs",
        default_value_t = DEFAULT_RL_CONTROL_TIMEOUT_SECS
    )]
    pub control_timeout_secs: u64,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: None,
            control_timeout_secs: DEFAULT_RL_CONTROL_TIMEOUT_SECS,
        }
    }
}
