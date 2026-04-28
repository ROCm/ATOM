use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use mesh::{
    config::{
        BackendType, CircuitBreakerConfig, ConfigResult, HealthCheckConfig, MetricsConfig,
        PolicyConfig, RetryConfig, RouterConfig, RoutingMode, TokenizerCacheConfig,
    },
    core::ConnectionMode,
    observability::metrics::PrometheusConfig,
    server::{self, ServerConfig},
    version,
};
fn parse_prefill_args() -> Vec<(String, Option<u16>)> {
    let args: Vec<String> = std::env::args().collect();
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    #[value(name = "sglang")]
    Sglang,
    #[value(name = "vllm")]
    Vllm,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Sglang => write!(f, "sglang"),
            Backend::Vllm => write!(f, "vllm"),
        }
    }
}

impl From<Backend> for BackendType {
    fn from(b: Backend) -> Self {
        match b {
            Backend::Sglang => BackendType::Sglang,
            Backend::Vllm => BackendType::Vllm,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "mesh")]
#[command(about = "ATOM Mesh - High-performance inference gateway")]
#[command(args_conflicts_with_subcommands = true)]
#[command(long_about = r#"
ATOM Mesh - Rust-based inference gateway

Usage:
  mesh launch [OPTIONS]       Launch router (short command)
  atom-mesh launch [OPTIONS]  Launch router (full name)

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
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    router_args: CliArgs,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch the router (same as running without subcommand)
    #[command(visible_alias = "start")]
    Launch {
        #[command(flatten)]
        args: CliArgs,
    },
}

#[derive(Parser, Debug)]
struct CliArgs {
    // ==================== Worker Configuration ====================
    /// Host address to bind the router server
    #[arg(long, default_value = "0.0.0.0", help_heading = "Worker Configuration")]
    host: String,

    /// Port number to bind the router server
    #[arg(long, default_value_t = 30000, help_heading = "Worker Configuration")]
    port: u16,

    /// List of worker URLs (supports IPv4 and IPv6)
    #[arg(long, num_args = 0.., help_heading = "Worker Configuration")]
    worker_urls: Vec<String>,

    // ==================== Routing Policy ====================
    /// Load balancing policy to use
    #[arg(long, default_value = "cache_aware", value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash"], help_heading = "Routing Policy")]
    policy: String,

    /// Cache threshold (0.0-1.0) for cache-aware routing
    #[arg(long, default_value_t = 0.3, help_heading = "Routing Policy")]
    cache_threshold: f32,

    /// Absolute threshold for load balancing trigger
    #[arg(long, default_value_t = 64, help_heading = "Routing Policy")]
    balance_abs_threshold: usize,

    /// Relative threshold for load balancing trigger
    #[arg(long, default_value_t = 1.5, help_heading = "Routing Policy")]
    balance_rel_threshold: f32,

    /// Interval in seconds between cache eviction operations
    #[arg(long, default_value_t = 120, help_heading = "Routing Policy")]
    eviction_interval: u64,

    /// Maximum size of the approximation tree for cache-aware routing
    #[arg(long, default_value_t = 67108864, help_heading = "Routing Policy")]
    max_tree_size: usize,

    /// Number of prefix tokens to use for prefix_hash policy
    #[arg(long, default_value_t = 256, help_heading = "Routing Policy")]
    prefix_token_count: usize,

    /// Load factor threshold for prefix_hash policy
    #[arg(long, default_value_t = 1.25, help_heading = "Routing Policy")]
    prefix_hash_load_factor: f64,

    /// Enable data parallelism aware scheduling
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    dp_aware: bool,

    // ==================== PD Disaggregation ====================
    /// Enable PD (Prefill-Decode) disaggregated mode
    #[arg(long, default_value_t = false, help_heading = "PD Disaggregation")]
    pd_disaggregation: bool,

    /// Decode server URLs (can be specified multiple times)
    #[arg(long, action = ArgAction::Append, help_heading = "PD Disaggregation")]
    decode: Vec<String>,

    /// Specific policy for prefill nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash"], help_heading = "PD Disaggregation")]
    prefill_policy: Option<String>,

    /// Specific policy for decode nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash"], help_heading = "PD Disaggregation")]
    decode_policy: Option<String>,

    /// Timeout in seconds for worker startup and registration
    #[arg(long, default_value_t = 1800, help_heading = "PD Disaggregation")]
    worker_startup_timeout_secs: u64,

    /// Interval in seconds between worker startup checks
    #[arg(long, default_value_t = 30, help_heading = "PD Disaggregation")]
    worker_startup_check_interval: u64,

    // ==================== Logging ====================
    /// Directory to store log files
    #[arg(long, help_heading = "Logging")]
    log_dir: Option<String>,

    /// Set the logging level
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"], help_heading = "Logging")]
    log_level: String,

    /// Enable structured JSON log output instead of plain text
    #[arg(long, default_value_t = false, help_heading = "Logging")]
    json_log: bool,

    // ==================== Prometheus Metrics ====================
    /// Port to expose Prometheus metrics
    #[arg(long, default_value_t = 29000, help_heading = "Prometheus Metrics")]
    prometheus_port: u16,

    /// Host address to bind the Prometheus metrics server
    #[arg(long, default_value = "0.0.0.0", help_heading = "Prometheus Metrics")]
    prometheus_host: String,

    /// Custom buckets for Prometheus duration metrics
    #[arg(long, num_args = 0.., help_heading = "Prometheus Metrics")]
    prometheus_duration_buckets: Vec<f64>,

    // ==================== Request Handling ====================
    /// Custom HTTP headers to check for request IDs
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    request_id_headers: Vec<String>,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 1800, help_heading = "Request Handling")]
    request_timeout_secs: u64,

    /// Grace period in seconds to wait for in-flight requests during shutdown
    #[arg(long, default_value_t = 180, help_heading = "Request Handling")]
    shutdown_grace_period_secs: u64,

    /// Maximum payload size in bytes
    #[arg(long, default_value_t = 536870912, help_heading = "Request Handling")]
    max_payload_size: usize,

    // ==================== Rate Limiting ====================
    /// Maximum concurrent requests (-1 to disable)
    #[arg(long, default_value_t = -1, help_heading = "Rate Limiting")]
    max_concurrent_requests: i32,

    /// Queue size for pending requests when limit reached
    #[arg(long, default_value_t = 100, help_heading = "Rate Limiting")]
    queue_size: usize,

    /// Maximum time in seconds a request can wait in queue
    #[arg(long, default_value_t = 60, help_heading = "Rate Limiting")]
    queue_timeout_secs: u64,

    /// Token bucket refill rate (tokens per second)
    #[arg(long, help_heading = "Rate Limiting")]
    rate_limit_tokens_per_second: Option<i32>,

    // ==================== Retry Configuration ====================
    /// Maximum number of retry attempts
    #[arg(long, default_value_t = 5, help_heading = "Retry Configuration")]
    retry_max_retries: u32,

    /// Initial backoff delay in milliseconds
    #[arg(long, default_value_t = 50, help_heading = "Retry Configuration")]
    retry_initial_backoff_ms: u64,

    /// Maximum backoff delay in milliseconds
    #[arg(long, default_value_t = 30000, help_heading = "Retry Configuration")]
    retry_max_backoff_ms: u64,

    /// Multiplier for exponential backoff
    #[arg(long, default_value_t = 1.5, help_heading = "Retry Configuration")]
    retry_backoff_multiplier: f32,

    /// Jitter factor (0.0-1.0) for retry delays
    #[arg(long, default_value_t = 0.2, help_heading = "Retry Configuration")]
    retry_jitter_factor: f32,

    /// Disable automatic retries
    #[arg(long, default_value_t = false, help_heading = "Retry Configuration")]
    disable_retries: bool,

    // ==================== Circuit Breaker ====================
    /// Number of failures before circuit opens
    #[arg(long, default_value_t = 10, help_heading = "Circuit Breaker")]
    cb_failure_threshold: u32,

    /// Successes needed in half-open state to close
    #[arg(long, default_value_t = 3, help_heading = "Circuit Breaker")]
    cb_success_threshold: u32,

    /// Seconds before attempting to close open circuit
    #[arg(long, default_value_t = 60, help_heading = "Circuit Breaker")]
    cb_timeout_duration_secs: u64,

    /// Sliding window duration for tracking failures
    #[arg(long, default_value_t = 120, help_heading = "Circuit Breaker")]
    cb_window_duration_secs: u64,

    /// Disable circuit breaker
    #[arg(long, default_value_t = false, help_heading = "Circuit Breaker")]
    disable_circuit_breaker: bool,

    // ==================== Health Checks ====================
    /// Failures before marking worker unhealthy
    #[arg(long, default_value_t = 3, help_heading = "Health Checks")]
    health_failure_threshold: u32,

    /// Successes before marking worker healthy
    #[arg(long, default_value_t = 2, help_heading = "Health Checks")]
    health_success_threshold: u32,

    /// Timeout in seconds for health check requests
    #[arg(long, default_value_t = 5, help_heading = "Health Checks")]
    health_check_timeout_secs: u64,

    /// Interval in seconds between health checks
    #[arg(long, default_value_t = 60, help_heading = "Health Checks")]
    health_check_interval_secs: u64,

    /// Health check endpoint path
    #[arg(long, default_value = "/health", help_heading = "Health Checks")]
    health_check_endpoint: String,

    /// Disable all worker health checks at startup
    #[arg(long, default_value_t = false, help_heading = "Health Checks")]
    disable_health_check: bool,

    // ==================== Tokenizer ====================
    /// Model path for loading tokenizer (HuggingFace ID or local path)
    #[arg(long, help_heading = "Tokenizer")]
    model_path: Option<String>,

    /// Explicit tokenizer path (overrides model_path)
    #[arg(long, help_heading = "Tokenizer")]
    tokenizer_path: Option<String>,

    /// Chat template path
    #[arg(long, help_heading = "Tokenizer")]
    chat_template: Option<String>,

    /// Enable L0 (exact match) tokenizer cache
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    tokenizer_cache_enable_l0: bool,

    /// Maximum entries in L0 tokenizer cache
    #[arg(long, default_value_t = 10000, help_heading = "Tokenizer")]
    tokenizer_cache_l0_max_entries: usize,

    /// Enable L1 (prefix matching) tokenizer cache
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    tokenizer_cache_enable_l1: bool,

    /// Maximum memory for L1 tokenizer cache in bytes
    #[arg(long, default_value_t = 52428800, help_heading = "Tokenizer")]
    tokenizer_cache_l1_max_memory: usize,

    // ==================== Parsers ====================
    /// Parser for reasoning models (e.g., deepseek-r1, qwen3)
    #[arg(long, help_heading = "Parsers")]
    reasoning_parser: Option<String>,

    /// Parser for tool-call interactions
    #[arg(long, help_heading = "Parsers")]
    tool_call_parser: Option<String>,

    // ==================== Backend ====================
    /// Backend runtime to use
    #[arg(long, value_enum, default_value_t = Backend::Sglang, alias = "runtime", help_heading = "Backend")]
    backend: Backend,

    // ==================== Control Plane Authentication ====================
    /// API key for worker connections
    #[arg(long, help_heading = "Control Plane Authentication")]
    api_key: Option<String>,
}

impl CliArgs {
    fn determine_connection_mode(worker_urls: &[String]) -> ConnectionMode {
        for url in worker_urls {
            if url.starts_with("grpc://") || url.starts_with("grpcs://") {
                return ConnectionMode::Grpc { port: None };
            }
        }
        ConnectionMode::Http
    }

    fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
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

    fn to_router_config(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<RouterConfig> {
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

        let builder = RouterConfig::builder()
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
            .circuit_breaker(!self.disable_circuit_breaker);

        builder.build()
    }

    fn to_server_config(&self, router_config: RouterConfig) -> ServerConfig {
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
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for version flags before parsing other args to avoid errors
    let args: Vec<String> = std::env::args().collect();
    for arg in &args {
        if arg == "--version" || arg == "-V" {
            println!("{}", version::get_version_string());
            return Ok(());
        }
        if arg == "--version-verbose" {
            println!("{}", version::get_verbose_version_string());
            return Ok(());
        }
    }

    let prefill_urls = parse_prefill_args();

    let mut filtered_args: Vec<String> = Vec::new();
    let raw_args: Vec<String> = std::env::args().collect();
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

    let cli = Cli::parse_from(filtered_args);

    // Handle subcommands or use direct args
    let cli_args = match cli.command {
        Some(Commands::Launch { args }) => args,
        None => cli.router_args,
    };

    println!("ATOM Mesh starting...");
    println!("Host: {}:{}", cli_args.host, cli_args.port);
    let mode_str = if cli_args.pd_disaggregation {
        "PD Disaggregated".to_string()
    } else {
        format!("Regular ({})", cli_args.backend)
    };
    println!("Mode: {}", mode_str);
    println!("Policy: {}", cli_args.policy);

    if cli_args.pd_disaggregation && !prefill_urls.is_empty() {
        println!("Prefill nodes: {:?}", prefill_urls);
        println!("Decode nodes: {:?}", cli_args.decode);
    }

    let router_config = cli_args.to_router_config(prefill_urls)?;
    router_config.validate()?;
    let server_config = cli_args.to_server_config(router_config);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move { server::startup(server_config).await })?;
    Ok(())
}
