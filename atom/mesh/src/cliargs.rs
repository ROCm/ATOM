use std::sync::Arc;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};

use crate::{
    config::{
        BackendType, CircuitBreakerConfig, ConfigError, ConfigResult, HealthCheckConfig,
        MetricsConfig, PolicyConfig, RetryConfig, RouterConfig, RoutingMode, TokenizerCacheConfig,
    },
    core::ConnectionMode,
    observability::metrics::PrometheusConfig,
    routers::atom_standalone::AtomStandaloneRuntime,
    server::{ServerConfig, ServerTlsConfig},
};

pub fn parse_prefill_args() -> Vec<(String, Option<u16>)> {
    let args: Vec<String> = std::env::args().collect();
    parse_prefill_args_from(&args)
}

pub fn parse_prefill_args_from(args: &[String]) -> Vec<(String, Option<u16>)> {
    let mut prefill_entries = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--prefill" && i + 1 < args.len() {
            let url = args[i + 1].clone();
            let bootstrap_port = if i + 2 < args.len() && !args[i + 2].starts_with("--") {
                if let Ok(port) = args[i + 2].parse::<u16>() {
                    i += 1;
                    Some(port)
                } else if args[i + 2].to_lowercase() == "none" {
                    i += 1;
                    None
                } else {
                    None
                }
            } else {
                None
            };
            prefill_entries.push((url, bootstrap_port));
            i += 2;
        } else {
            i += 1;
        }
    }
    prefill_entries
}

pub fn filter_prefill_args_from(raw_args: &[String]) -> Vec<String> {
    let mut filtered_args = Vec::new();
    let mut i = 0;
    while i < raw_args.len() {
        if raw_args[i] == "--prefill" && i + 1 < raw_args.len() {
            i += 2;
            if i < raw_args.len()
                && !raw_args[i].starts_with("--")
                && (raw_args[i].parse::<u16>().is_ok() || raw_args[i].to_lowercase() == "none")
            {
                i += 1;
            }
        } else {
            filtered_args.push(raw_args[i].clone());
            i += 1;
        }
    }
    filtered_args
}

pub fn parse_decode_args() -> Vec<String> {
    let args: Vec<String> = std::env::args().collect();
    parse_decode_args_from(&args)
}

pub fn parse_decode_args_from(args: &[String]) -> Vec<String> {
    let mut decode_entries = Vec::new();
    let mut i = 0;

    while i < args.len() {
        if args[i] == "--decode" && i + 1 < args.len() {
            decode_entries.push(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }

    decode_entries
}

pub fn filter_decode_args_from(raw_args: &[String]) -> Vec<String> {
    let mut filtered_args = Vec::new();
    let mut i = 0;

    while i < raw_args.len() {
        if raw_args[i] == "--decode" && i + 1 < raw_args.len() {
            i += 2;
        } else {
            filtered_args.push(raw_args[i].clone());
            i += 1;
        }
    }

    filtered_args
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    #[value(name = "sglang")]
    Sglang,
    #[value(name = "vllm")]
    Vllm,
    #[value(name = "atom")]
    Atom,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Sglang => write!(f, "sglang"),
            Backend::Vllm => write!(f, "vllm"),
            Backend::Atom => write!(f, "atom"),
        }
    }
}

impl From<Backend> for BackendType {
    fn from(b: Backend) -> Self {
        match b {
            Backend::Sglang => BackendType::Sglang,
            Backend::Vllm => BackendType::Vllm,
            Backend::Atom => BackendType::Atom,
        }
    }
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "sglang" => Ok(Self::Sglang),
            "vllm" => Ok(Self::Vllm),
            "atom" => Ok(Self::Atom),
            other => Err(format!("unsupported backend: {other}")),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "mesh")]
#[command(about = "Atomesh - High-performance inference gateway")]
#[command(args_conflicts_with_subcommands = true)]
#[command(long_about = r#"
Atomesh - Rust-based inference gateway

Usage:
  mesh launch [OPTIONS]       Launch router (short command)
  atomesh launch [OPTIONS]  Launch router (full name)

Examples:
  # Regular mode
  mesh launch --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode
  mesh launch --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 9002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --policy cache_aware

  # With different policies
  mesh launch --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --prefill-policy cache_aware --decode-policy power_of_two

"#)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    #[command(flatten)]
    pub router_args: CliArgs,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Launch the router (same as running without subcommand)
    #[command(visible_alias = "start")]
    Launch {
        #[command(flatten)]
        args: CliArgs,
    },
}

#[derive(Parser, Debug, Clone)]
pub struct CliArgs {
    // ==================== Worker Configuration ====================
    /// Host address to bind the router server
    #[arg(long, default_value = "0.0.0.0", help_heading = "Worker Configuration")]
    pub host: String,

    /// Port number to bind the router server
    #[arg(long, default_value_t = 30000, help_heading = "Worker Configuration")]
    pub port: u16,

    /// List of worker URLs (supports IPv4 and IPv6)
    #[arg(long, num_args = 0.., help_heading = "Worker Configuration")]
    pub worker_urls: Vec<String>,

    // ==================== Routing Policy ====================
    /// Load balancing policy to use
    #[arg(long, default_value = "cache_aware", value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash"], help_heading = "Routing Policy")]
    pub policy: String,

    /// Cache threshold (0.0-1.0) for cache-aware routing
    #[arg(long, default_value_t = 0.3, help_heading = "Routing Policy")]
    pub cache_threshold: f32,

    /// Absolute threshold for load balancing trigger
    #[arg(long, default_value_t = 64, help_heading = "Routing Policy")]
    pub balance_abs_threshold: usize,

    /// Relative threshold for load balancing trigger
    #[arg(long, default_value_t = 1.5, help_heading = "Routing Policy")]
    pub balance_rel_threshold: f32,

    /// Interval in seconds between cache eviction operations
    #[arg(long, default_value_t = 120, help_heading = "Routing Policy")]
    pub eviction_interval: u64,

    /// Maximum size of the approximation tree for cache-aware routing
    #[arg(long, default_value_t = 67108864, help_heading = "Routing Policy")]
    pub max_tree_size: usize,

    /// Number of prefix tokens to use for prefix_hash policy
    #[arg(long, default_value_t = 256, help_heading = "Routing Policy")]
    pub prefix_token_count: usize,

    /// Load factor threshold for prefix_hash policy
    #[arg(long, default_value_t = 1.25, help_heading = "Routing Policy")]
    pub prefix_hash_load_factor: f64,

    /// Enable data parallelism aware scheduling
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    pub dp_aware: bool,

    // ==================== PD Disaggregation ====================
    /// Enable PD (Prefill-Decode) disaggregated mode
    #[arg(long, default_value_t = false, help_heading = "PD Disaggregation")]
    pub pd_disaggregation: bool,

    /// Decode server URLs (can be specified multiple times)
    #[arg(long, action = ArgAction::Append, help_heading = "PD Disaggregation")]
    pub decode: Vec<String>,

    /// Specific policy for prefill nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash"], help_heading = "PD Disaggregation")]
    pub prefill_policy: Option<String>,

    /// Specific policy for decode nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash"], help_heading = "PD Disaggregation")]
    pub decode_policy: Option<String>,

    /// Timeout in seconds for worker startup and registration
    #[arg(long, default_value_t = 1800, help_heading = "PD Disaggregation")]
    pub worker_startup_timeout_secs: u64,

    /// Interval in seconds between worker startup checks
    #[arg(long, default_value_t = 30, help_heading = "PD Disaggregation")]
    pub worker_startup_check_interval: u64,

    // ==================== Logging ====================
    /// Directory to store log files
    #[arg(long, help_heading = "Logging")]
    pub log_dir: Option<String>,

    /// Set the logging level
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"], help_heading = "Logging")]
    pub log_level: String,

    /// Enable structured JSON log output instead of plain text
    #[arg(long, default_value_t = false, help_heading = "Logging")]
    pub json_log: bool,

    // ==================== Prometheus Metrics ====================
    /// Port to expose Prometheus metrics
    #[arg(long, default_value_t = 29000, help_heading = "Prometheus Metrics")]
    pub prometheus_port: u16,

    /// Host address to bind the Prometheus metrics server
    #[arg(long, default_value = "0.0.0.0", help_heading = "Prometheus Metrics")]
    pub prometheus_host: String,

    /// Custom buckets for Prometheus duration metrics
    #[arg(long, num_args = 0.., help_heading = "Prometheus Metrics")]
    pub prometheus_duration_buckets: Vec<f64>,

    // ==================== Request Handling ====================
    /// Custom HTTP headers to check for request IDs
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    pub request_id_headers: Vec<String>,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 1800, help_heading = "Request Handling")]
    pub request_timeout_secs: u64,

    /// Grace period in seconds to wait for in-flight requests during shutdown
    #[arg(long, default_value_t = 180, help_heading = "Request Handling")]
    pub shutdown_grace_period_secs: u64,

    /// Maximum payload size in bytes
    #[arg(long, default_value_t = 536870912, help_heading = "Request Handling")]
    pub max_payload_size: usize,

    // ==================== Rate Limiting ====================
    /// Maximum concurrent requests (-1 to disable)
    #[arg(long, default_value_t = -1, help_heading = "Rate Limiting")]
    pub max_concurrent_requests: i32,

    /// Queue size for pending requests when limit reached
    #[arg(long, default_value_t = 100, help_heading = "Rate Limiting")]
    pub queue_size: usize,

    /// Maximum time in seconds a request can wait in queue
    #[arg(long, default_value_t = 60, help_heading = "Rate Limiting")]
    pub queue_timeout_secs: u64,

    /// Token bucket refill rate (tokens per second)
    #[arg(long, help_heading = "Rate Limiting")]
    pub rate_limit_tokens_per_second: Option<i32>,

    // ==================== Retry Configuration ====================
    /// Maximum number of retry attempts
    #[arg(long, default_value_t = 5, help_heading = "Retry Configuration")]
    pub retry_max_retries: u32,

    /// Initial backoff delay in milliseconds
    #[arg(long, default_value_t = 50, help_heading = "Retry Configuration")]
    pub retry_initial_backoff_ms: u64,

    /// Maximum backoff delay in milliseconds
    #[arg(long, default_value_t = 30000, help_heading = "Retry Configuration")]
    pub retry_max_backoff_ms: u64,

    /// Multiplier for exponential backoff
    #[arg(long, default_value_t = 1.5, help_heading = "Retry Configuration")]
    pub retry_backoff_multiplier: f32,

    /// Jitter factor (0.0-1.0) for retry delays
    #[arg(long, default_value_t = 0.2, help_heading = "Retry Configuration")]
    pub retry_jitter_factor: f32,

    /// Disable automatic retries
    #[arg(long, default_value_t = false, help_heading = "Retry Configuration")]
    pub disable_retries: bool,

    // ==================== Circuit Breaker ====================
    /// Number of failures before circuit opens
    #[arg(long, default_value_t = 10, help_heading = "Circuit Breaker")]
    pub cb_failure_threshold: u32,

    /// Successes needed in half-open state to close
    #[arg(long, default_value_t = 3, help_heading = "Circuit Breaker")]
    pub cb_success_threshold: u32,

    /// Seconds before attempting to close open circuit
    #[arg(long, default_value_t = 60, help_heading = "Circuit Breaker")]
    pub cb_timeout_duration_secs: u64,

    /// Sliding window duration for tracking failures
    #[arg(long, default_value_t = 120, help_heading = "Circuit Breaker")]
    pub cb_window_duration_secs: u64,

    /// Disable circuit breaker
    #[arg(long, default_value_t = false, help_heading = "Circuit Breaker")]
    pub disable_circuit_breaker: bool,

    // ==================== Health Checks ====================
    /// Failures before marking worker unhealthy
    #[arg(long, default_value_t = 3, help_heading = "Health Checks")]
    pub health_failure_threshold: u32,

    /// Successes before marking worker healthy
    #[arg(long, default_value_t = 2, help_heading = "Health Checks")]
    pub health_success_threshold: u32,

    /// Timeout in seconds for health check requests
    #[arg(long, default_value_t = 5, help_heading = "Health Checks")]
    pub health_check_timeout_secs: u64,

    /// Interval in seconds between health checks
    #[arg(long, default_value_t = 60, help_heading = "Health Checks")]
    pub health_check_interval_secs: u64,

    /// Health check endpoint path
    #[arg(long, default_value = "/health", help_heading = "Health Checks")]
    pub health_check_endpoint: String,

    /// Disable all worker health checks at startup
    #[arg(long, default_value_t = false, help_heading = "Health Checks")]
    pub disable_health_check: bool,

    // ==================== Tokenizer ====================
    /// Model path for loading tokenizer (HuggingFace ID or local path)
    #[arg(long, help_heading = "Tokenizer")]
    pub model_path: Option<String>,

    /// Explicit tokenizer path (overrides model_path)
    #[arg(long, help_heading = "Tokenizer")]
    pub tokenizer_path: Option<String>,

    /// Chat template path
    #[arg(long, help_heading = "Tokenizer")]
    pub chat_template: Option<String>,

    /// Enable L0 (exact match) tokenizer cache
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    pub tokenizer_cache_enable_l0: bool,

    /// Maximum entries in L0 tokenizer cache
    #[arg(long, default_value_t = 10000, help_heading = "Tokenizer")]
    pub tokenizer_cache_l0_max_entries: usize,

    /// Enable L1 (prefix matching) tokenizer cache
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    pub tokenizer_cache_enable_l1: bool,

    /// Maximum memory for L1 tokenizer cache in bytes
    #[arg(long, default_value_t = 52428800, help_heading = "Tokenizer")]
    pub tokenizer_cache_l1_max_memory: usize,

    // ==================== Parsers ====================
    /// Parser for reasoning models (e.g., deepseek-r1, qwen3)
    #[arg(long, help_heading = "Parsers")]
    pub reasoning_parser: Option<String>,

    /// Parser for tool-call interactions
    #[arg(long, help_heading = "Parsers")]
    pub tool_call_parser: Option<String>,

    // ==================== Backend ====================
    /// Backend runtime to use
    #[arg(long, value_enum, default_value_t = Backend::Sglang, alias = "runtime", help_heading = "Backend")]
    pub backend: Backend,

    // ==================== Control Plane Authentication ====================
    /// API key for worker connections
    #[arg(long, help_heading = "Control Plane Authentication")]
    pub api_key: Option<String>,

    // ==================== TLS ====================
    /// PEM certificate chain path for serving HTTPS
    #[arg(long, requires = "tls_key_path", help_heading = "TLS")]
    pub tls_cert_path: Option<std::path::PathBuf>,

    /// PEM private key path for serving HTTPS
    #[arg(long, requires = "tls_cert_path", help_heading = "TLS")]
    pub tls_key_path: Option<std::path::PathBuf>,
}

impl CliArgs {
    pub fn for_python(
        host: String,
        port: u16,
        worker_urls: Vec<String>,
        pd_disaggregation: bool,
        decode: Vec<String>,
        policy: String,
        backend: Backend,
    ) -> Self {
        Self {
            host,
            port,
            worker_urls,
            policy,
            backend,
            pd_disaggregation,
            decode,
            ..Self::default()
        }
    }

    pub fn determine_connection_mode(worker_urls: &[String]) -> ConnectionMode {
        for url in worker_urls {
            if url.starts_with("grpc://") || url.starts_with("grpcs://") {
                return ConnectionMode::Grpc { port: None };
            }
        }
        ConnectionMode::Http
    }

    pub fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
        match policy_str {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval,
                max_tree_size: self.max_tree_size,
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5,
            },
            "prefix_hash" => PolicyConfig::PrefixHash {
                prefix_token_count: self.prefix_token_count,
                load_factor: self.prefix_hash_load_factor,
            },
            _ => PolicyConfig::RoundRobin,
        }
    }

    fn validate_tls_args(&self) -> ConfigResult<()> {
        match (&self.tls_cert_path, &self.tls_key_path) {
            (Some(_), Some(_)) | (None, None) => Ok(()),
            (Some(_), None) => Err(ConfigError::MissingRequired {
                field: "tls_key_path".to_string(),
            }),
            (None, Some(_)) => Err(ConfigError::MissingRequired {
                field: "tls_cert_path".to_string(),
            }),
        }
    }

    pub fn to_router_config(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<RouterConfig> {
        self.validate_tls_args()?;

        // Determine routing mode based on PD disaggregation flag
        let mode = if self.pd_disaggregation {
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls: self.decode.clone(),
                prefill_policy: self.prefill_policy.as_ref().map(|p| self.parse_policy(p)),
                decode_policy: self.decode_policy.as_ref().map(|p| self.parse_policy(p)),
            }
        } else {
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        let policy = self.parse_policy(&self.policy);

        let metrics = Some(MetricsConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
        });

        let mut all_urls = Vec::new();
        match &mode {
            RoutingMode::Regular { worker_urls } => {
                all_urls.extend(worker_urls.clone());
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => {
                for (url, _) in prefill_urls {
                    all_urls.push(url.clone());
                }
                all_urls.extend(decode_urls.clone());
            }
        }
        let connection_mode = Self::determine_connection_mode(&all_urls);

        RouterConfig::builder()
            .mode(mode)
            .backend(self.backend.into())
            .policy(policy)
            .connection_mode(connection_mode)
            .host(&self.host)
            .port(self.port)
            .max_payload_size(self.max_payload_size)
            .request_timeout_secs(self.request_timeout_secs)
            .worker_startup_timeout_secs(self.worker_startup_timeout_secs)
            .worker_startup_check_interval_secs(self.worker_startup_check_interval)
            .max_concurrent_requests(self.max_concurrent_requests)
            .queue_size(self.queue_size)
            .queue_timeout_secs(self.queue_timeout_secs)
            .retry_config(RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            })
            .health_check_config(HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
                disable_health_check: self.disable_health_check,
            })
            .tokenizer_cache(TokenizerCacheConfig {
                enable_l0: self.tokenizer_cache_enable_l0,
                l0_max_entries: self.tokenizer_cache_l0_max_entries,
                enable_l1: self.tokenizer_cache_enable_l1,
                l1_max_memory: self.tokenizer_cache_l1_max_memory,
            })
            .log_level(&self.log_level)
            .maybe_api_key(self.api_key.as_ref())
            .maybe_metrics(metrics)
            .maybe_log_dir(self.log_dir.as_ref())
            .maybe_request_id_headers(
                (!self.request_id_headers.is_empty()).then(|| self.request_id_headers.clone()),
            )
            .maybe_rate_limit_tokens_per_second(self.rate_limit_tokens_per_second)
            .maybe_model_path(self.model_path.as_ref())
            .maybe_tokenizer_path(self.tokenizer_path.as_ref())
            .maybe_chat_template(self.chat_template.as_ref())
            .maybe_reasoning_parser(self.reasoning_parser.as_ref())
            .maybe_tool_call_parser(self.tool_call_parser.as_ref())
            .dp_aware(self.dp_aware)
            .retries(!self.disable_retries)
            .circuit_breaker(!self.disable_circuit_breaker)
            .build()
    }

    pub fn to_server_config(&self, router_config: RouterConfig) -> ServerConfig {
        self.to_server_config_with_runtime(router_config, None)
    }

    pub fn to_server_config_with_runtime(
        &self,
        router_config: RouterConfig,
        atom_standalone_runtime: Option<Arc<AtomStandaloneRuntime>>,
    ) -> ServerConfig {
        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
            duration_buckets: if self.prometheus_duration_buckets.is_empty() {
                None
            } else {
                Some(self.prometheus_duration_buckets.clone())
            },
        });

        ServerConfig {
            host: self.host.clone(),
            port: self.port,
            router_config,
            max_payload_size: self.max_payload_size,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            json_log: self.json_log,
            prometheus_config,
            request_timeout_secs: self.request_timeout_secs,
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
            shutdown_grace_period_secs: self.shutdown_grace_period_secs,
            tls: self
                .tls_cert_path
                .as_ref()
                .zip(self.tls_key_path.as_ref())
                .map(|(cert_path, key_path)| ServerTlsConfig {
                    cert_path: cert_path.clone(),
                    key_path: key_path.clone(),
                }),
            atom_standalone_runtime,
        }
    }
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 30000,
            worker_urls: Vec::new(),
            policy: "cache_aware".to_string(),
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            eviction_interval: 120,
            max_tree_size: 67_108_864,
            prefix_token_count: 256,
            prefix_hash_load_factor: 1.25,
            dp_aware: false,
            pd_disaggregation: false,
            decode: Vec::new(),
            prefill_policy: None,
            decode_policy: None,
            worker_startup_timeout_secs: 1800,
            worker_startup_check_interval: 30,
            log_dir: None,
            log_level: "info".to_string(),
            json_log: false,
            prometheus_port: 29_000,
            prometheus_host: "0.0.0.0".to_string(),
            prometheus_duration_buckets: Vec::new(),
            request_id_headers: Vec::new(),
            request_timeout_secs: 1800,
            shutdown_grace_period_secs: 180,
            max_payload_size: 536_870_912,
            max_concurrent_requests: -1,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            retry_max_retries: 5,
            retry_initial_backoff_ms: 50,
            retry_max_backoff_ms: 30_000,
            retry_backoff_multiplier: 1.5,
            retry_jitter_factor: 0.2,
            disable_retries: false,
            cb_failure_threshold: 10,
            cb_success_threshold: 3,
            cb_timeout_duration_secs: 60,
            cb_window_duration_secs: 120,
            disable_circuit_breaker: false,
            health_failure_threshold: 3,
            health_success_threshold: 2,
            health_check_timeout_secs: 5,
            health_check_interval_secs: 60,
            health_check_endpoint: "/health".to_string(),
            disable_health_check: false,
            model_path: None,
            tokenizer_path: None,
            chat_template: None,
            tokenizer_cache_enable_l0: false,
            tokenizer_cache_l0_max_entries: 10_000,
            tokenizer_cache_enable_l1: false,
            tokenizer_cache_l1_max_memory: 52_428_800,
            reasoning_parser: None,
            tool_call_parser: None,
            backend: Backend::Sglang,
            api_key: None,
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}
