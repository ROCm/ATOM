use std::{collections::BTreeMap, time::Duration};

use serde::{Deserialize, Serialize};

use super::{ConfigError, ConfigResult};
use crate::{
    core::{prepare_pool::PoolConfig, runtime::default_prepare_workers, ConnectionMode},
    kv_events::{source::TierMap, OffloadLoad, WireProfile},
    kv_storage::{KeyRule, RankProbe, ReaderRule},
};

/// Backend runtime type for inference workers.
/// Determines PD-disaggregation protocol (SGLang dual-dispatch+bootstrap vs vLLM Mooncake kv_transfer_params).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BackendType {
    #[default]
    #[serde(rename = "sglang")]
    Sglang,
    #[serde(rename = "vllm")]
    Vllm,
    #[serde(rename = "atom")]
    Atom,
}

/// Strategy used by the ATOM PD router to map a selected prefill DP rank to
/// the selected decode DP rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AtomPdRankMappingPolicy {
    /// Do not rewrite the planner-selected prefill/decode DP ranks.
    #[default]
    #[serde(rename = "none")]
    None,
    /// Map prefill rank N to decode rank N when both sides expose DP workers.
    #[serde(rename = "idx2idx")]
    Idx2Idx,
}

impl std::fmt::Display for AtomPdRankMappingPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Idx2Idx => write!(f, "idx2idx"),
        }
    }
}

impl std::str::FromStr for AtomPdRankMappingPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" | "disabled" | "off" => Ok(Self::None),
            "idx2idx" | "index_to_index" | "same_index" => Ok(Self::Idx2Idx),
            other => Err(format!("unsupported ATOM PD rank mapping policy: {other}")),
        }
    }
}

/// Main router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    #[cfg(feature = "ext-proc")]
    #[serde(default)]
    pub ext_proc: crate::ext_proc::ExtProcConfig,
    #[serde(default)]
    pub prepare_pool: PreparePoolConfig,
    pub mode: RoutingMode,
    #[serde(default)]
    pub backend: BackendType,
    #[serde(default)]
    pub connection_mode: ConnectionMode,
    #[serde(default)]
    pub atom_standalone: bool,
    #[serde(default)]
    pub atom_pd_rank_mapping_policy: AtomPdRankMappingPolicy,

    pub policy: PolicyConfig,
    pub host: String,
    pub port: u16,
    pub max_payload_size: usize,
    pub request_timeout_secs: u64,
    pub worker_startup_timeout_secs: u64,
    pub worker_startup_check_interval_secs: u64,
    pub dp_aware: bool,
    pub api_key: Option<String>,
    pub metrics: Option<MetricsConfig>,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub request_id_headers: Option<Vec<String>>,
    /// Set to -1 to disable rate limiting
    pub max_concurrent_requests: i32,
    pub queue_size: usize,
    pub queue_timeout_secs: u64,
    /// If not set, defaults to max_concurrent_requests
    pub rate_limit_tokens_per_second: Option<i32>,
    pub retry: RetryConfig,
    pub circuit_breaker: CircuitBreakerConfig,
    /// When true, overrides retry.max_retries to 1
    #[serde(default)]
    pub disable_retries: bool,
    /// When true, overrides circuit_breaker.failure_threshold to u32::MAX
    #[serde(default)]
    pub disable_circuit_breaker: bool,
    pub health_check: HealthCheckConfig,
    /// Can be a HuggingFace model ID or local path
    pub model_path: Option<String>,
    /// Overrides model_path tokenizer if provided
    pub tokenizer_path: Option<String>,
    pub chat_template: Option<String>,
    /// For reasoning models (e.g., deepseek-r1, qwen3)
    pub reasoning_parser: Option<String>,
    /// For tool-call interactions
    pub tool_call_parser: Option<String>,
    #[serde(default)]
    pub tokenizer_cache: TokenizerCacheConfig,
    #[serde(default)]
    pub kv_events: KvEventsConfig,
}

/// Default `--prepare-queue-timeout-ms`.
pub const DEFAULT_PREPARE_QUEUE_TIMEOUT_MS: u64 = 250;
/// Default `--prepare-timeout-ms`.
pub const DEFAULT_PREPARE_TIMEOUT_MS: u64 = 5000;
/// Default `--prepare-max-tokenize-bytes`.
pub const DEFAULT_MAX_TOKENIZE_BYTES: usize = 1024 * 1024;
/// Waiting jobs per preparation thread when `--prepare-queue-capacity` is unset.
pub const PREPARE_QUEUE_SLOTS_PER_WORKER: usize = 10;

/// Shared CPU preparation limits for HTTP, PD and external processing.
#[derive(Debug, Clone, clap::Args, Serialize, Deserialize, PartialEq, Eq)]
#[command(next_help_heading = "Request Preparation")]
#[serde(default)]
pub struct PreparePoolConfig {
    /// Preparation threads (default: half the available CPUs, at least 1).
    /// Overrides --ext-proc-parser-concurrency.
    #[arg(id = "prepare_workers", long = "prepare-workers")]
    pub workers: Option<usize>,
    /// Jobs waiting for a preparation thread (default: 10 per thread).
    #[arg(id = "prepare_queue_capacity", long = "prepare-queue-capacity")]
    pub queue_capacity: Option<usize>,
    /// Queue wait limit.
    #[arg(
        id = "prepare_queue_timeout_ms",
        long = "prepare-queue-timeout-ms",
        default_value_t = DEFAULT_PREPARE_QUEUE_TIMEOUT_MS
    )]
    pub queue_timeout_ms: u64,
    /// Deadline shared by parsing and tokenization.
    #[arg(
        id = "prepare_timeout_ms",
        long = "prepare-timeout-ms",
        default_value_t = DEFAULT_PREPARE_TIMEOUT_MS
    )]
    pub prepare_timeout_ms: u64,
    /// Tokenization byte limit (default: 1 MiB); overrides --ext-proc-max-tokenize-bytes.
    #[arg(id = "prepare_max_tokenize_bytes", long = "prepare-max-tokenize-bytes")]
    pub max_tokenize_bytes: Option<usize>,
}

impl Default for PreparePoolConfig {
    fn default() -> Self {
        Self {
            workers: None,
            queue_capacity: None,
            queue_timeout_ms: DEFAULT_PREPARE_QUEUE_TIMEOUT_MS,
            prepare_timeout_ms: DEFAULT_PREPARE_TIMEOUT_MS,
            max_tokenize_bytes: None,
        }
    }
}

/// Tokenizer cache configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerCacheConfig {
    /// Whole-string exact match cache
    #[serde(default = "default_enable_l0")]
    pub enable_l0: bool,
    #[serde(default = "default_l0_max_entries")]
    pub l0_max_entries: usize,
    /// Prefix matching at fixed boundaries
    #[serde(default = "default_enable_l1")]
    pub enable_l1: bool,
    #[serde(default = "default_l1_max_memory")]
    pub l1_max_memory: usize,
}

fn default_enable_l0() -> bool {
    false
}

fn default_l0_max_entries() -> usize {
    10_000
}

fn default_enable_l1() -> bool {
    false
}

fn default_l1_max_memory() -> usize {
    50 * 1024 * 1024 // 50MB
}

impl TokenizerCacheConfig {
    /// Returns Some(self) if any caching is enabled, None otherwise.
    /// Use this when passing cache config to tokenizer registration workflow.
    pub fn to_option(&self) -> Option<Self> {
        if self.enable_l0 || self.enable_l1 {
            Some(self.clone())
        } else {
            None
        }
    }
}

impl Default for TokenizerCacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: default_enable_l0(),
            l0_max_entries: default_l0_max_entries(),
            enable_l1: default_enable_l1(),
            l1_max_memory: default_l1_max_memory(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvEventsMode {
    /// Discover sources when a policy needs KV events; finding none is not an error.
    #[default]
    Auto,
    /// As `auto`, and a worker whose discovery fails is reported as `down`.
    On,
    Off,
}

impl std::str::FromStr for KvEventsMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            other => Err(format!("expected auto, on or off, got {other:?}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDiscoveryMethod {
    Static,
    Engine,
    ServerInfo,
}

/// Event endpoints of one worker, listed by DP rank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticKvEventSource {
    pub endpoints: Vec<String>,
    #[serde(default)]
    pub replay_endpoints: Vec<String>,
    #[serde(default)]
    pub topic: String,
    pub model: Option<String>,
    /// As in vLLM Router. Events of another block size are dropped; None takes each
    /// event's own.
    #[serde(default)]
    pub block_size: Option<u32>,
    #[serde(default)]
    pub profile: WireProfile,
    #[serde(default)]
    pub tier_map: TierMap,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvStorageBreakerConfig {
    pub failures: u32,
    pub open_secs: u64,
}

impl Default for KvStorageBreakerConfig {
    fn default() -> Self {
        Self {
            failures: 5,
            open_secs: 10,
        }
    }
}

/// Defaults for shared-storage lookups.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvStorageConfig {
    /// Mooncake master admin port (its `metrics_port`).
    pub mooncake_admin_port: u16,
    pub probe: RankProbe,
    pub deadline_ms: u64,
    pub breaker: KvStorageBreakerConfig,
    /// In-flight lookups per source; requests beyond it skip the source.
    pub max_inflight: u32,
}

impl Default for KvStorageConfig {
    fn default() -> Self {
        Self {
            mooncake_admin_port: 9003,
            probe: RankProbe::First,
            deadline_ms: 3,
            breaker: KvStorageBreakerConfig::default(),
            max_inflight: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageBackendConfig {
    MooncakeAdmin(Vec<String>),
    UmbpMaster(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageGateConfig {
    pub min_tokens: u32,
}

/// A storage source discovery cannot find. With the id of a generated source, only the
/// given fields override it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageSourceConfig {
    pub model: Option<String>,
    pub backend: Option<StorageBackendConfig>,
    pub key_rule: Option<KeyRule>,
    #[serde(default)]
    pub readers: ReaderRule,
    /// Defaults to the readers' `min_load_tokens`.
    pub gate: Option<StorageGateConfig>,
    pub deadline_ms: Option<u64>,
    pub probe: Option<RankProbe>,
}

/// KV event ingestion and shared-storage lookups. Unknown keys are errors: the section
/// is new, so no saved config carries removed ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvEventsConfig {
    pub enabled: KvEventsMode,
    /// Tried in order; the first non-empty result wins.
    pub discovery: Vec<KvDiscoveryMethod>,
    /// Keyed by worker base URL.
    pub static_sources: BTreeMap<String, StaticKvEventSource>,
    pub storage: KvStorageConfig,
    pub storage_sources: BTreeMap<String, StorageSourceConfig>,
    /// For `atom` sources whose discovery result carries no `offload`.
    pub offload_load: Option<OffloadLoad>,
    pub replay_timeout_ms: u64,
    pub stale_after_secs: u64,
    pub max_index_blocks: u64,
    /// Strict decoding, for conformance tests only.
    pub strict: bool,
}

impl Default for KvEventsConfig {
    fn default() -> Self {
        Self {
            enabled: KvEventsMode::Auto,
            discovery: vec![
                KvDiscoveryMethod::Static,
                KvDiscoveryMethod::Engine,
                KvDiscoveryMethod::ServerInfo,
            ],
            static_sources: BTreeMap::new(),
            storage: KvStorageConfig::default(),
            storage_sources: BTreeMap::new(),
            offload_load: None,
            replay_timeout_ms: 5000,
            stale_after_secs: 30,
            max_index_blocks: 4_000_000,
            strict: false,
        }
    }
}

/// Routing mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoutingMode {
    #[serde(rename = "regular")]
    Regular { worker_urls: Vec<String> },
    #[serde(rename = "prefill_decode")]
    PrefillDecode {
        /// With optional bootstrap ports
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_policy: Option<PolicyConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_policy: Option<PolicyConfig>,
    },
}

impl RoutingMode {
    pub fn is_pd_mode(&self) -> bool {
        matches!(self, RoutingMode::PrefillDecode { .. })
    }

    pub fn worker_count(&self) -> usize {
        match self {
            RoutingMode::Regular { worker_urls } => worker_urls.len(),
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => prefill_urls.len() + decode_urls.len(),
        }
    }

    /// Get the effective prefill policy for PD mode
    /// Falls back to the main policy if no specific prefill policy is set
    pub fn get_prefill_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { prefill_policy, .. } => {
                prefill_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }

    /// Get the effective decode policy for PD mode
    /// Falls back to the main policy if no specific decode policy is set
    pub fn get_decode_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { decode_policy, .. } => {
                decode_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }
}

/// Policy configuration for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PolicyConfig {
    #[serde(rename = "random")]
    Random,

    #[serde(rename = "round_robin")]
    RoundRobin,

    /// Session-affinity policy for data-parallel workers.
    /// Requests with the same `X-Session-ID` are routed to the same healthy worker.
    #[serde(rename = "dp_sticky")]
    DpSticky,

    #[serde(rename = "cache_aware")]
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    },

    #[serde(rename = "power_of_two")]
    PowerOfTwo { load_check_interval_secs: u64 },

    /// Prefix hash policy for KV cache-aware load balancing.
    /// A lightweight alternative to cache_aware radix tree.
    /// Routes requests based on prefix token hash for cache locality.
    /// - Uses consistent hash ring with bounded load balancing
    /// - Walks ring if worker is overloaded (load > avg * load_factor)
    /// - O(log n) lookup instead of O(prefix_len) radix tree traversal
    #[serde(rename = "prefix_hash")]
    PrefixHash {
        /// Number of prefix tokens to hash (default: 256)
        #[serde(default = "default_prefix_token_count")]
        prefix_token_count: usize,
        /// Load factor threshold - walk ring if load > avg * factor (default: 1.25)
        #[serde(default = "default_load_factor")]
        load_factor: f64,
    },
}

fn default_prefix_token_count() -> usize {
    256
}

fn default_load_factor() -> f64 {
    1.25
}

impl PolicyConfig {
    pub fn name(&self) -> &'static str {
        match self {
            PolicyConfig::Random => "random",
            PolicyConfig::RoundRobin => "round_robin",
            PolicyConfig::DpSticky => "dp_sticky",
            PolicyConfig::CacheAware { .. } => "cache_aware",
            PolicyConfig::PowerOfTwo { .. } => "power_of_two",
            PolicyConfig::PrefixHash { .. } => "prefix_hash",
        }
    }
}

/// Retry configuration for request handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f32,
    /// D' = D * (1 + U[-j, +j]) where j is jitter factor
    #[serde(default = "default_retry_jitter_factor")]
    pub jitter_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 50,
            max_backoff_ms: 30000,
            backoff_multiplier: 1.5,
            jitter_factor: 0.2,
        }
    }
}

fn default_retry_jitter_factor() -> f32 {
    0.2
}

/// Health check configuration for worker monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_secs: u64,
    pub check_interval_secs: u64,
    pub endpoint: String,
    pub disable_health_check: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_secs: 5,
            check_interval_secs: 60,
            endpoint: "/health".to_string(),
            disable_health_check: false,
        }
    }
}

/// Circuit breaker configuration for worker reliability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_duration_secs: u64,
    pub window_duration_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 3,
            timeout_duration_secs: 60,
            window_duration_secs: 120,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub port: u16,
    pub host: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
        }
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "ext-proc")]
            ext_proc: Default::default(),
            prepare_pool: PreparePoolConfig::default(),
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            backend: BackendType::Sglang,
            policy: PolicyConfig::Random,
            host: "0.0.0.0".to_string(),
            port: 3001,
            max_payload_size: 536_870_912,     // 512MB
            request_timeout_secs: 1800,        // 30 minutes
            worker_startup_timeout_secs: 1800, // 30 minutes for large model loading
            worker_startup_check_interval_secs: 30,
            dp_aware: false,
            api_key: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: -1,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
            connection_mode: ConnectionMode::Http,
            atom_standalone: false,
            atom_pd_rank_mapping_policy: AtomPdRankMappingPolicy::None,
            model_path: None,
            tokenizer_path: None,
            chat_template: None,
            reasoning_parser: None,
            tool_call_parser: None,
            tokenizer_cache: TokenizerCacheConfig::default(),
            kv_events: KvEventsConfig::default(),
        }
    }
}

impl RouterConfig {
    /// Preparation settings, with the legacy ext-proc options filling unset
    /// values when the external-processing transport is enabled.
    fn effective_prepare_pool(&self) -> PreparePoolConfig {
        let config = self.prepare_pool.clone();
        #[cfg(feature = "ext-proc")]
        if self.ext_proc.enabled {
            return PreparePoolConfig {
                workers: config.workers.or(self.ext_proc.parser_concurrency),
                max_tokenize_bytes: config
                    .max_tokenize_bytes
                    .or(self.ext_proc.max_tokenize_bytes),
                ..config
            };
        }
        config
    }

    pub fn resolved_prepare_pool(&self) -> ConfigResult<PoolConfig> {
        let config = self.effective_prepare_pool();
        let workers = config.workers.unwrap_or_else(default_prepare_workers);
        let queue_capacity = match config.queue_capacity {
            Some(capacity) => capacity,
            None => workers
                .checked_mul(PREPARE_QUEUE_SLOTS_PER_WORKER)
                .ok_or_else(|| ConfigError::InvalidValue {
                    field: "prepare_pool.queue_capacity".to_string(),
                    value: format!("{PREPARE_QUEUE_SLOTS_PER_WORKER} * {workers}"),
                    reason: "Automatic queue capacity exceeds the platform limit".to_string(),
                })?,
        };
        let pool = PoolConfig {
            workers,
            queue_capacity,
            queue_timeout: Duration::from_millis(config.queue_timeout_ms),
            prepare_timeout: Duration::from_millis(config.prepare_timeout_ms),
        };
        pool.validate()?;
        Ok(pool)
    }

    pub fn resolved_prepare_workers(&self) -> usize {
        self.effective_prepare_pool()
            .workers
            .unwrap_or_else(default_prepare_workers)
    }

    pub fn resolved_max_tokenize_bytes(&self) -> usize {
        self.effective_prepare_pool()
            .max_tokenize_bytes
            .unwrap_or(DEFAULT_MAX_TOKENIZE_BYTES)
    }

    /// Create a new configuration with mode and policy
    pub fn new(mode: RoutingMode, policy: PolicyConfig) -> Self {
        Self {
            mode,
            policy,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> ConfigResult<()> {
        crate::config::validation::ConfigValidator::validate(self)
    }

    /// Get the routing mode type as a string
    pub fn mode_type(&self) -> &'static str {
        match self.mode {
            RoutingMode::Regular { .. } => "regular",
            RoutingMode::PrefillDecode { .. } => "prefill_decode",
        }
    }

    /// Check if metrics are enabled
    pub fn has_metrics(&self) -> bool {
        self.metrics.is_some()
    }

    /// Compute the effective retry config considering disable flag
    pub fn effective_retry_config(&self) -> RetryConfig {
        let mut cfg = self.retry.clone();
        if self.disable_retries {
            cfg.max_retries = 1;
        }
        cfg
    }

    /// Compute the effective circuit breaker config considering disable flag
    pub fn effective_circuit_breaker_config(&self) -> CircuitBreakerConfig {
        let mut cfg = self.circuit_breaker.clone();
        if self.disable_circuit_breaker {
            cfg.failure_threshold = u32::MAX;
        }
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_events::model::TierTarget;

    #[test]
    fn kv_events_config_parses_documented_example_and_defaults_when_absent() {
        let config: KvEventsConfig = serde_json::from_str(
            r#"{"enabled": "auto", "discovery": ["static", "engine", "server_info"],
                "static_sources": {"http://10.0.0.7:8000": {
                    "endpoints": ["tcp://10.0.0.7:5557", "tcp://10.0.0.7:5558"],
                    "replay_endpoints": ["tcp://10.0.0.7:5657", "tcp://10.0.0.7:5658"],
                    "topic": "", "model": "zai-org/GLM-5.2", "profile": "vllm",
                    "tier_map": {"CPU": "drop"}}},
                "storage": {"mooncake_admin_port": 9003, "probe": "first", "deadline_ms": 3,
                    "breaker": {"failures": 5, "open_secs": 10}, "max_inflight": 64},
                "storage_sources": {"umbp-a": {"model": "zai-org/GLM-5.2",
                    "backend": {"umbp_master": "10.0.0.21:50051"},
                    "key_rule": {"sglang_page": {"page_size": 64, "layout": "mha_split_heads"}},
                    "readers": ["http://10.0.1.*:30000"], "gate": {"min_tokens": 4096}}},
                "offload_load": {"chunk_tokens": 256, "min_load_tokens": 8192},
                "replay_timeout_ms": 5000, "stale_after_secs": 30,
                "max_index_blocks": 4000000, "strict": false}"#,
        )
        .unwrap();
        let source = &config.static_sources["http://10.0.0.7:8000"];
        assert_eq!(source.tier_map["CPU"], TierTarget::Drop);
        assert_eq!(source.profile, WireProfile::Vllm);
        let umbp = &config.storage_sources["umbp-a"];
        assert_eq!(
            umbp.backend,
            Some(StorageBackendConfig::UmbpMaster("10.0.0.21:50051".into()))
        );
        assert_eq!(
            umbp.readers,
            ReaderRule::Workers(vec!["http://10.0.1.*:30000".into()])
        );
        assert!(matches!(
            umbp.key_rule,
            Some(KeyRule::SglangPage { page_size: 64, .. })
        ));

        let overrides: KvEventsConfig = serde_json::from_str(
            r#"{"enabled": "off", "storage_sources": {"atom-mooncake-5f0e3c9a":
                {"readers": "discovered", "deadline_ms": 5, "probe": "all"}}}"#,
        )
        .unwrap();
        assert_eq!(overrides.enabled, KvEventsMode::Off);
        assert_eq!(overrides.discovery, KvEventsConfig::default().discovery);
        let generated = &overrides.storage_sources["atom-mooncake-5f0e3c9a"];
        assert_eq!(generated.readers, ReaderRule::Discovered);
        assert_eq!(generated.probe, Some(RankProbe::All));
        assert!(serde_json::from_str::<StorageSourceConfig>(r#"{"readers": "all"}"#).is_err());
        for typo in [
            r#"{"static_sources": {"u": {"endpoints": [], "replay_endpoint": []}}}"#,
            r#"{"storage": {"deadline": 3}}"#,
            r#"{"offload_load": {"chunk_tokens": 256, "min_load_token": 0}}"#,
            r#"{"kv_events": {}}"#,
        ] {
            assert!(
                serde_json::from_str::<KvEventsConfig>(typo).is_err(),
                "{typo}"
            );
        }

        let mut legacy = serde_json::to_value(RouterConfig::default()).unwrap();
        legacy.as_object_mut().unwrap().remove("kv_events");
        let restored: RouterConfig = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.kv_events, KvEventsConfig::default());
    }

    #[test]
    fn prepare_pool_defaults_overrides_and_legacy_config_deserialization() {
        let config = RouterConfig::default();
        let workers = default_prepare_workers();
        assert_eq!(config.resolved_prepare_workers(), workers);
        let pool = config.resolved_prepare_pool().unwrap();
        assert_eq!(pool.queue_capacity, workers * 10);
        assert_eq!(pool.queue_timeout, Duration::from_millis(250));
        assert_eq!(pool.prepare_timeout, Duration::from_secs(5));
        assert_eq!(config.prepare_pool.queue_capacity, None);
        assert_eq!(config.resolved_max_tokenize_bytes(), 1024 * 1024);
        let mut legacy = serde_json::to_value(&config).unwrap();
        legacy.as_object_mut().unwrap().remove("prepare_pool");
        let restored: RouterConfig = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.prepare_pool, PreparePoolConfig::default());

        // Removed options in saved configs are ignored.
        let partial: PreparePoolConfig = serde_json::from_str(
            r#"{"workers":2,"shutdown_grace_ms":3000,"parse_inline_max_bytes":1024,
                "max_retained_input_bytes":1024}"#,
        )
        .unwrap();
        assert_eq!(partial.workers, Some(2));
        assert_eq!(partial.queue_capacity, None);
        let config = RouterConfig {
            prepare_pool: partial,
            ..Default::default()
        };
        assert_eq!(config.resolved_prepare_pool().unwrap().queue_capacity, 20);

        // An explicit queue capacity overrides the per-worker multiplier.
        let prepare_pool = serde_json::from_str(r#"{"workers":2,"queue_capacity":6}"#).unwrap();
        let config = RouterConfig {
            prepare_pool,
            ..Default::default()
        };
        config.validate().unwrap();
        assert_eq!(config.resolved_prepare_pool().unwrap().queue_capacity, 6);
    }

    #[cfg(feature = "ext-proc")]
    #[test]
    fn prepare_pool_resolves_enabled_legacy_options_and_common_overrides() {
        let mut config = RouterConfig::default();
        config.ext_proc.enabled = true;
        let workers = default_prepare_workers();
        assert_eq!(config.resolved_prepare_workers(), workers);
        config.ext_proc.enabled = false;
        config.ext_proc.parser_concurrency = Some(3);
        config.ext_proc.max_tokenize_bytes = Some(4096);
        // Merely compiling ext-proc must not change the ordinary HTTP limits.
        assert_eq!(config.resolved_prepare_workers(), workers);
        assert_eq!(config.resolved_max_tokenize_bytes(), 1024 * 1024);
        config.ext_proc.enabled = true;
        assert_eq!(config.resolved_prepare_workers(), 3);
        assert_eq!(config.resolved_prepare_pool().unwrap().queue_capacity, 30);
        assert_eq!(config.resolved_max_tokenize_bytes(), 4096);
        config.prepare_pool.workers = Some(1);
        config.prepare_pool.max_tokenize_bytes = Some(8192);
        assert_eq!(config.resolved_prepare_workers(), 1);
        assert_eq!(config.resolved_prepare_pool().unwrap().queue_capacity, 10);
        assert_eq!(config.resolved_max_tokenize_bytes(), 8192);
    }

    #[cfg(feature = "ext-proc")]
    #[test]
    fn legacy_ext_proc_config_file_still_sets_preparation_limits() {
        let mut legacy = serde_json::to_value(RouterConfig::default()).unwrap();
        legacy.as_object_mut().unwrap().remove("prepare_pool");
        legacy["ext_proc"] = serde_json::json!({
            "enabled": true,
            "parser_concurrency": 3,
            "max_tokenize_bytes": 4096,
        });
        let config: RouterConfig = serde_json::from_value(legacy).unwrap();
        config.validate().unwrap();
        assert_eq!(config.ext_proc.parser_concurrency, Some(3));
        assert_eq!(config.resolved_prepare_workers(), 3);
        assert_eq!(config.resolved_prepare_pool().unwrap().queue_capacity, 30);
        assert_eq!(config.resolved_max_tokenize_bytes(), 4096);
    }

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();

        assert!(
            matches!(config.mode, RoutingMode::Regular { worker_urls } if worker_urls.is_empty())
        );
        assert!(matches!(config.policy, PolicyConfig::Random));
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3001);
        assert_eq!(config.max_payload_size, 536_870_912);
        assert_eq!(config.request_timeout_secs, 1800);
        assert_eq!(config.worker_startup_timeout_secs, 1800);
        assert_eq!(config.worker_startup_check_interval_secs, 30);
        assert!(config.metrics.is_none());
        assert!(config.log_dir.is_none());
        assert!(config.log_level.is_none());
    }

    #[test]
    fn test_router_config_new() {
        let mode = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string(), "http://worker2".to_string()],
        };
        let policy = PolicyConfig::RoundRobin;

        let config = RouterConfig::new(mode, policy);

        match config.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 2);
                assert_eq!(worker_urls[0], "http://worker1");
                assert_eq!(worker_urls[1], "http://worker2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert!(matches!(config.policy, PolicyConfig::RoundRobin));
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3001);
    }

    #[test]
    fn test_router_config_serialization() {
        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker1".to_string()])
            .random_policy()
            .host("0.0.0.0")
            .port(8080)
            .log_dir("/var/log")
            .log_level("debug")
            .build_unchecked();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.host, deserialized.host);
        assert_eq!(config.port, deserialized.port);
        assert_eq!(config.max_payload_size, deserialized.max_payload_size);
        assert_eq!(config.log_dir, deserialized.log_dir);
        assert_eq!(config.log_level, deserialized.log_level);
        assert!(deserialized.metrics.is_none());
    }

    #[test]
    fn test_routing_mode_is_pd_mode() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        assert!(!regular.is_pd_mode());

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), Some(8001))],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };
        assert!(pd.is_pd_mode());
    }

    #[test]
    fn test_routing_mode_worker_count() {
        let regular = RoutingMode::Regular {
            worker_urls: vec![
                "http://worker1".to_string(),
                "http://worker2".to_string(),
                "http://worker3".to_string(),
            ],
        };
        assert_eq!(regular.worker_count(), 3);

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![
                ("http://prefill1".to_string(), Some(8001)),
                ("http://prefill2".to_string(), None),
            ],
            decode_urls: vec![
                "http://decode1".to_string(),
                "http://decode2".to_string(),
                "http://decode3".to_string(),
            ],
            prefill_policy: None,
            decode_policy: None,
        };
        assert_eq!(pd.worker_count(), 5);

        let empty_regular = RoutingMode::Regular {
            worker_urls: vec![],
        };
        assert_eq!(empty_regular.worker_count(), 0);
    }

    #[test]
    fn test_routing_mode_serialization() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        let json = serde_json::to_string(&regular).unwrap();
        assert!(json.contains("\"type\":\"regular\""));
        assert!(json.contains("\"worker_urls\""));

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), Some(8001))],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };
        let json = serde_json::to_string(&pd).unwrap();
        assert!(json.contains("\"type\":\"prefill_decode\""));
        assert!(json.contains("\"prefill_urls\""));
        assert!(json.contains("\"decode_urls\""));
    }

    #[test]
    fn test_policy_config_name() {
        assert_eq!(PolicyConfig::Random.name(), "random");
        assert_eq!(PolicyConfig::RoundRobin.name(), "round_robin");
        assert_eq!(PolicyConfig::DpSticky.name(), "dp_sticky");

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
        };
        assert_eq!(cache_aware.name(), "cache_aware");

        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        assert_eq!(power_of_two.name(), "power_of_two");
    }

    #[test]
    fn test_policy_config_serialization() {
        let random = PolicyConfig::Random;
        let json = serde_json::to_string(&random).unwrap();
        assert_eq!(json, r#"{"type":"random"}"#);

        let dp_sticky = PolicyConfig::DpSticky;
        let json = serde_json::to_string(&dp_sticky).unwrap();
        assert_eq!(json, r#"{"type":"dp_sticky"}"#);

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
        };
        let json = serde_json::to_string(&cache_aware).unwrap();
        assert!(json.contains("\"type\":\"cache_aware\""));
        assert!(json.contains("\"cache_threshold\":0.8"));
        assert!(json.contains("\"balance_abs_threshold\":10"));

        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        let json = serde_json::to_string(&power_of_two).unwrap();
        assert!(json.contains("\"type\":\"power_of_two\""));
        assert!(json.contains("\"load_check_interval_secs\":60"));
    }

    #[test]
    fn test_cache_aware_parameters() {
        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.75,
            balance_abs_threshold: 20,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 600,
            max_tree_size: 5000,
        };

        match cache_aware {
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                assert!((cache_threshold - 0.75).abs() < 0.0001);
                assert_eq!(balance_abs_threshold, 20);
                assert!((balance_rel_threshold - 2.0).abs() < 0.0001);
                assert_eq!(eviction_interval_secs, 600);
                assert_eq!(max_tree_size, 5000);
            }
            _ => panic!("Expected CacheAware"),
        }
    }

    #[test]
    fn test_power_of_two_parameters() {
        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 120,
        };

        match power_of_two {
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => {
                assert_eq!(load_check_interval_secs, 120);
            }
            _ => panic!("Expected PowerOfTwo"),
        }
    }

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();

        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_metrics_config_custom() {
        let config = MetricsConfig {
            port: 9090,
            host: "0.0.0.0".to_string(),
        };

        assert_eq!(config.port, 9090);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_mode_type() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .build_unchecked();
        assert_eq!(config.mode_type(), "regular");

        let config = RouterConfig::builder()
            .prefill_decode_mode(vec![], vec![])
            .build_unchecked();
        assert_eq!(config.mode_type(), "prefill_decode");
    }

    #[test]
    fn test_has_metrics() {
        let config = RouterConfig::default();
        assert!(!config.has_metrics());

        let config = RouterConfig::builder()
            .metrics_config(MetricsConfig::default())
            .build_unchecked();
        assert!(config.has_metrics());
    }

    #[test]
    fn test_large_worker_lists() {
        let large_urls: Vec<String> = (0..1000).map(|i| format!("http://worker{}", i)).collect();

        let config = RouterConfig::builder()
            .regular_mode(large_urls.clone())
            .build_unchecked();

        assert_eq!(config.mode.worker_count(), 1000);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 1000);
            }
            _ => panic!("Expected Regular mode"),
        }
    }

    #[test]
    fn test_unicode_in_config() {
        let config = RouterConfig::builder()
            .regular_mode(vec![
                "http://работник1".to_string(),
                "http://工作者2".to_string(),
            ])
            .log_dir("/日志/目录")
            .build_unchecked();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls[0], "http://работник1");
                assert_eq!(worker_urls[1], "http://工作者2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert_eq!(deserialized.log_dir, Some("/日志/目录".to_string()));
    }

    #[test]
    fn test_empty_string_fields() {
        let config = RouterConfig::builder()
            .host("")
            .log_dir("")
            .log_level("")
            .build_unchecked();

        assert_eq!(config.host, "");
        assert_eq!(config.log_dir, Some("".to_string()));
        assert_eq!(config.log_level, Some("".to_string()));
    }

    #[test]
    fn test_full_pd_mode_config() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://prefill1:8000".to_string(), Some(8001)),
                    ("http://prefill2:8000".to_string(), None),
                ],
                vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
            )
            .power_of_two_policy(30)
            .host("0.0.0.0")
            .port(3000)
            .max_payload_size(1048576)
            .request_timeout_secs(120)
            .worker_startup_timeout_secs(60)
            .worker_startup_check_interval_secs(5)
            .enable_metrics("0.0.0.0", 9090)
            .log_dir("/var/log/atomesh")
            .log_level("info")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 4);
        assert_eq!(config.policy.name(), "power_of_two");
        assert!(config.has_metrics());
    }

    #[test]
    fn test_full_regular_mode_config() {
        let config = RouterConfig::builder()
            .regular_mode(vec![
                "http://worker1:8000".to_string(),
                "http://worker2:8000".to_string(),
                "http://worker3:8000".to_string(),
            ])
            .cache_aware_policy(0.9, 5, 1.2, 600, 10000)
            .host("0.0.0.0")
            .port(3001)
            .max_payload_size(536870912)
            .request_timeout_secs(300)
            .worker_startup_timeout_secs(180)
            .worker_startup_check_interval_secs(15)
            .metrics_config(MetricsConfig::default())
            .log_level("debug")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(!config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 3);
        assert_eq!(config.policy.name(), "cache_aware");
        assert!(config.has_metrics());
    }

    #[test]
    fn test_config_with_all_options() {
        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker1".to_string()])
            .round_robin_policy()
            .host("::1") // IPv6
            .port(8888)
            .max_payload_size(1024 * 1024 * 512) // 512MB
            .request_timeout_secs(900)
            .worker_startup_timeout_secs(600)
            .worker_startup_check_interval_secs(20)
            .enable_metrics("::", 9999) // IPv6 any
            .log_dir("/opt/logs/atomesh")
            .log_level("trace")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(config.has_metrics());
        assert_eq!(config.mode_type(), "regular");

        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.host, "::1");
        assert_eq!(deserialized.port, 8888);
    }

    #[test]
    fn test_pd_policy_fallback_both_specified() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            }),
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
        };

        let main_policy = PolicyConfig::Random;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {}
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {}
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_prefill() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            }),
            decode_policy: None,
        };

        let main_policy = PolicyConfig::RoundRobin;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {}
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_decode() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
        };

        let main_policy = PolicyConfig::Random;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::Random => {}
            _ => panic!("Expected Random for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {}
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_none_specified() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };

        let main_policy = PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 20,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 2000,
        };

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for decode"),
        }
    }

    #[test]
    fn test_regular_mode_policy_fallback() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };

        let main_policy = PolicyConfig::RoundRobin;

        match regular.get_prefill_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for regular mode"),
        }

        match regular.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for regular mode"),
        }
    }
}
