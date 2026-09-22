use std::{net::SocketAddr, path::PathBuf};

use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    config::{ConfigError, ConfigResult, RouterConfig},
    core::ConnectionMode,
};

#[derive(Debug, Clone, Args, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtProcConfig {
    /// Use Envoy external processing for inference; keep HTTP management and health endpoints.
    #[arg(long = "ext-proc", help_heading = "External Processing")]
    pub enabled: bool,
    #[arg(
        long = "ext-proc-listen",
        default_value = "127.0.0.1:9002",
        help_heading = "External Processing"
    )]
    pub listen: SocketAddr,
    #[arg(
        long = "ext-proc-max-message-bytes",
        default_value_t = 1048576,
        help_heading = "External Processing"
    )]
    pub max_message_bytes: usize,
    #[arg(
        long = "ext-proc-max-body-bytes",
        default_value_t = 8388608,
        help_heading = "External Processing"
    )]
    pub max_body_bytes: usize,
    #[arg(
        long = "ext-proc-max-streams",
        default_value_t = 256,
        help_heading = "External Processing"
    )]
    pub max_streams: usize,
    #[arg(
        long = "ext-proc-body-timeout-secs",
        default_value_t = 30,
        help_heading = "External Processing"
    )]
    pub body_timeout_secs: u64,
    #[arg(
        long = "ext-proc-decision-timeout-secs",
        default_value_t = 60,
        help_heading = "External Processing"
    )]
    pub decision_timeout_secs: u64,
    #[arg(
        long = "ext-proc-idle-timeout-secs",
        default_value_t = 300,
        help_heading = "External Processing"
    )]
    pub idle_timeout_secs: u64,
    #[arg(
        long = "ext-proc-drain-timeout-secs",
        default_value_t = 180,
        help_heading = "External Processing"
    )]
    pub drain_timeout_secs: u64,
    #[arg(
        long = "ext-proc-tls-cert",
        requires = "tls_key",
        help_heading = "External Processing"
    )]
    pub tls_cert: Option<PathBuf>,
    #[arg(
        long = "ext-proc-tls-key",
        requires = "tls_cert",
        help_heading = "External Processing"
    )]
    pub tls_key: Option<PathBuf>,
    /// Require client certificates signed by this CA.
    #[arg(
        long = "ext-proc-client-ca",
        requires = "tls_cert",
        help_heading = "External Processing"
    )]
    pub client_ca: Option<PathBuf>,
    /// HTTP executor listener used only for Prefill/Decode mode.
    #[arg(
        long = "ext-proc-executor-listen",
        default_value = "127.0.0.1:9003",
        help_heading = "External Processing"
    )]
    pub executor_listen: SocketAddr,
    /// Executor address reachable by Envoy; defaults to its bound address.
    #[arg(
        long = "ext-proc-executor-advertise",
        help_heading = "External Processing"
    )]
    pub executor_advertise: Option<SocketAddr>,
}

impl Default for ExtProcConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            listen: SocketAddr::from(([127, 0, 0, 1], 9002)),
            max_message_bytes: 1_048_576,
            max_body_bytes: 8_388_608,
            max_streams: 256,
            body_timeout_secs: 30,
            decision_timeout_secs: 60,
            idle_timeout_secs: 300,
            drain_timeout_secs: 180,
            tls_cert: None,
            tls_key: None,
            client_ca: None,
            executor_listen: SocketAddr::from(([127, 0, 0, 1], 9003)),
            executor_advertise: None,
        }
    }
}

impl ExtProcConfig {
    pub(crate) fn validate(&self, router: &RouterConfig) -> ConfigResult<()> {
        if !self.enabled {
            return Ok(());
        }
        let invalid = |reason: &str| ConfigError::IncompatibleConfig {
            reason: format!("ext-proc: {reason}"),
        };
        if router.atom_standalone || !matches!(router.connection_mode, ConnectionMode::Http) {
            return Err(invalid("requires external HTTP workers"));
        }
        if router.mode.is_pd_mode()
            && self.executor_listen.ip().is_unspecified()
            && self.executor_advertise.is_none()
        {
            return Err(invalid(
                "an executor bound to a wildcard requires an advertised address",
            ));
        }
        if self
            .executor_advertise
            .is_some_and(|address| address.ip().is_unspecified() || address.port() == 0)
        {
            return Err(invalid(
                "executor advertised address must be a concrete IP and nonzero port",
            ));
        }
        if self.max_message_bytes < 131_072 || self.max_body_bytes == 0 || self.max_streams == 0 {
            return Err(invalid(
                "message limit must be >= 128 KiB; body and stream limits must be positive",
            ));
        }
        if [
            self.body_timeout_secs,
            self.decision_timeout_secs,
            self.idle_timeout_secs,
            self.drain_timeout_secs,
        ]
        .contains(&0)
        {
            return Err(invalid("timeouts must be positive"));
        }
        if self.tls_cert.is_some() != self.tls_key.is_some()
            || (self.client_ca.is_some() && self.tls_cert.is_none())
        {
            return Err(invalid(
                "TLS requires both certificate and key; client CA requires TLS",
            ));
        }
        Ok(())
    }
}
