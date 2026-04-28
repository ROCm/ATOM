use super::{
    BackendType, CircuitBreakerConfig, ConfigResult, HealthCheckConfig, MetricsConfig,
    PolicyConfig, RetryConfig, RouterConfig, RoutingMode, TokenizerCacheConfig,
};
use crate::core::ConnectionMode;

/// Builder for RouterConfig that wraps the config itself
/// This eliminates field duplication and stays in sync automatically
#[derive(Debug, Clone, Default)]
pub struct RouterConfigBuilder {
    config: RouterConfig,
}

impl RouterConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes ownership
    pub fn from_config(config: RouterConfig) -> Self {
        Self { config }
    }

    pub fn from_config_ref(config: &RouterConfig) -> Self {
        Self::from_config(config.clone())
    }

    // ==================== Backend ====================

    pub fn backend(mut self, backend: BackendType) -> Self {
        self.config.backend = backend;
        self
    }

    // ==================== Routing Mode ====================

    pub fn regular_mode(mut self, worker_urls: Vec<String>) -> Self {
        self.config.mode = RoutingMode::Regular { worker_urls };
        self
    }

    pub fn prefill_decode_mode(
        mut self,
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
    ) -> Self {
        self.config.mode = RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            prefill_policy: None,
            decode_policy: None,
        };
        self
    }

    /// With separate policies
    pub fn prefill_decode_mode_with_policies(
        mut self,
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        prefill_policy: Option<PolicyConfig>,
        decode_policy: Option<PolicyConfig>,
    ) -> Self {
        self.config.mode = RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            prefill_policy,
            decode_policy,
        };
        self
    }

    pub fn mode(mut self, mode: RoutingMode) -> Self {
        self.config.mode = mode;
        self
    }

    // ==================== Policy ====================

    pub fn policy(mut self, policy: PolicyConfig) -> Self {
        self.config.policy = policy;
        self
    }

    pub fn random_policy(mut self) -> Self {
        self.config.policy = PolicyConfig::Random;
        self
    }

    pub fn round_robin_policy(mut self) -> Self {
        self.config.policy = PolicyConfig::RoundRobin;
        self
    }

    pub fn cache_aware_policy(
        mut self,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    ) -> Self {
        self.config.policy = PolicyConfig::CacheAware {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
        };
        self
    }

    pub fn power_of_two_policy(mut self, load_check_interval_secs: u64) -> Self {
        self.config.policy = PolicyConfig::PowerOfTwo {
            load_check_interval_secs,
        };
        self
    }

    // ==================== Connection ====================

    pub fn connection_mode(mut self, mode: ConnectionMode) -> Self {
        self.config.connection_mode = mode;
        self
    }

    pub fn http_connection(mut self) -> Self {
        self.config.connection_mode = ConnectionMode::Http;
        self
    }

    pub fn grpc_connection(mut self, port: Option<u16>) -> Self {
        self.config.connection_mode = ConnectionMode::Grpc { port };
        self
    }

    pub fn grpc_connection_default(mut self) -> Self {
        self.config.connection_mode = ConnectionMode::Grpc { port: None };
        self
    }

    pub fn host<S: Into<String>>(mut self, host: S) -> Self {
        self.config.host = host.into();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    // ==================== Request ====================

    pub fn max_payload_size(mut self, size: usize) -> Self {
        self.config.max_payload_size = size;
        self
    }

    pub fn request_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.request_timeout_secs = timeout;
        self
    }

    pub fn worker_startup_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.worker_startup_timeout_secs = timeout;
        self
    }

    pub fn worker_startup_check_interval_secs(mut self, interval: u64) -> Self {
        self.config.worker_startup_check_interval_secs = interval;
        self
    }

    // ==================== Rate Limiting ====================

    pub fn max_concurrent_requests(mut self, max: i32) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    pub fn disable_rate_limiting(mut self) -> Self {
        self.config.max_concurrent_requests = -1;
        self
    }

    pub fn queue_size(mut self, size: usize) -> Self {
        self.config.queue_size = size;
        self
    }

    pub fn queue_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.queue_timeout_secs = timeout;
        self
    }

    pub fn rate_limit_tokens_per_second(mut self, tokens: i32) -> Self {
        self.config.rate_limit_tokens_per_second = Some(tokens);
        self
    }

    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    // ==================== Retry ====================

    pub fn retry_config(mut self, retry: RetryConfig) -> Self {
        self.config.retry = retry;
        self
    }

    pub fn disable_retries(mut self) -> Self {
        self.config.disable_retries = true;
        self
    }

    pub fn enable_retries(mut self) -> Self {
        self.config.disable_retries = false;
        self
    }

    // ==================== Circuit Breaker ====================

    pub fn circuit_breaker_config(mut self, circuit_breaker: CircuitBreakerConfig) -> Self {
        self.config.circuit_breaker = circuit_breaker;
        self
    }

    pub fn disable_circuit_breaker(mut self) -> Self {
        self.config.disable_circuit_breaker = true;
        self
    }

    pub fn enable_circuit_breaker(mut self) -> Self {
        self.config.disable_circuit_breaker = false;
        self
    }

    // ==================== Health Check ====================

    pub fn health_check_config(mut self, health_check: HealthCheckConfig) -> Self {
        self.config.health_check = health_check;
        self
    }

    // ==================== Metrics ====================

    pub fn metrics_config(mut self, metrics: MetricsConfig) -> Self {
        self.config.metrics = Some(metrics);
        self
    }

    pub fn enable_metrics<S: Into<String>>(mut self, host: S, port: u16) -> Self {
        self.config.metrics = Some(MetricsConfig {
            host: host.into(),
            port,
        });
        self
    }

    // ==================== Logging ====================

    pub fn log_dir<S: Into<String>>(mut self, dir: S) -> Self {
        self.config.log_dir = Some(dir.into());
        self
    }

    pub fn log_level<S: Into<String>>(mut self, level: S) -> Self {
        self.config.log_level = Some(level.into());
        self
    }

    pub fn request_id_headers(mut self, headers: Vec<String>) -> Self {
        self.config.request_id_headers = Some(headers);
        self
    }

    pub fn model_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.model_path = Some(path.into());
        self
    }

    /// Overrides model_path tokenizer
    pub fn tokenizer_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.tokenizer_path = Some(path.into());
        self
    }

    pub fn chat_template<S: Into<String>>(mut self, path: S) -> Self {
        self.config.chat_template = Some(path.into());
        self
    }

    // ==================== Parsers ====================

    pub fn reasoning_parser<S: Into<String>>(mut self, parser: S) -> Self {
        self.config.reasoning_parser = Some(parser.into());
        self
    }

    pub fn tool_call_parser<S: Into<String>>(mut self, parser: S) -> Self {
        self.config.tool_call_parser = Some(parser.into());
        self
    }

    // ==================== Tokenizer Cache ====================

    pub fn tokenizer_cache(mut self, cache: TokenizerCacheConfig) -> Self {
        self.config.tokenizer_cache = cache;
        self
    }

    pub fn enable_l0_cache(mut self, max_entries: usize) -> Self {
        self.config.tokenizer_cache.enable_l0 = true;
        self.config.tokenizer_cache.l0_max_entries = max_entries;
        self
    }

    pub fn enable_l1_cache(mut self, max_memory: usize) -> Self {
        self.config.tokenizer_cache.enable_l1 = true;
        self.config.tokenizer_cache.l1_max_memory = max_memory;
        self
    }

    // ==================== Data Parallelism ====================

    pub fn enable_dp_aware(mut self) -> Self {
        self.config.dp_aware = true;
        self
    }

    pub fn disable_dp_aware(mut self) -> Self {
        self.config.dp_aware = false;
        self
    }

    // ==================== Boolean Setters ====================
    // Accept bool parameters to conditionally set flags without if statements

    pub fn dp_aware(mut self, enable: bool) -> Self {
        self.config.dp_aware = enable;
        self
    }

    /// Inverse of disable_retries field
    pub fn retries(mut self, enable: bool) -> Self {
        self.config.disable_retries = !enable;
        self
    }

    /// Inverse of disable_circuit_breaker field
    pub fn circuit_breaker(mut self, enable: bool) -> Self {
        self.config.disable_circuit_breaker = !enable;
        self
    }

    // ==================== Option Setters ====================
    // Accept Option<T> and only set if Some

    pub fn maybe_api_key(mut self, key: Option<impl Into<String>>) -> Self {
        if let Some(k) = key {
            self.config.api_key = Some(k.into());
        }
        self
    }

    pub fn maybe_metrics(mut self, metrics: Option<MetricsConfig>) -> Self {
        self.config.metrics = metrics;
        self
    }

    pub fn maybe_log_dir(mut self, dir: Option<impl Into<String>>) -> Self {
        self.config.log_dir = dir.map(|d| d.into());
        self
    }

    pub fn maybe_log_level(mut self, level: Option<impl Into<String>>) -> Self {
        self.config.log_level = level.map(|l| l.into());
        self
    }

    pub fn maybe_request_id_headers(mut self, headers: Option<Vec<String>>) -> Self {
        self.config.request_id_headers = headers;
        self
    }

    pub fn maybe_rate_limit_tokens_per_second(mut self, tokens: Option<i32>) -> Self {
        self.config.rate_limit_tokens_per_second = tokens;
        self
    }

    pub fn maybe_model_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.config.model_path = path.map(|p| p.into());
        self
    }

    pub fn maybe_tokenizer_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.config.tokenizer_path = path.map(|p| p.into());
        self
    }

    pub fn maybe_chat_template(mut self, template: Option<impl Into<String>>) -> Self {
        self.config.chat_template = template.map(|t| t.into());
        self
    }

    pub fn maybe_reasoning_parser(mut self, parser: Option<impl Into<String>>) -> Self {
        self.config.reasoning_parser = parser.map(|p| p.into());
        self
    }

    pub fn maybe_tool_call_parser(mut self, parser: Option<impl Into<String>>) -> Self {
        self.config.tool_call_parser = parser.map(|p| p.into());
        self
    }

    // ==================== Build ====================

    pub fn build(self) -> ConfigResult<RouterConfig> {
        self.build_with_validation(true)
    }

    pub fn build_unchecked(self) -> RouterConfig {
        self.into()
    }

    pub fn build_with_validation(self, validate: bool) -> ConfigResult<RouterConfig> {
        let config: RouterConfig = self.into();
        if validate {
            config.validate()?;
        }
        Ok(config)
    }
}

impl From<RouterConfigBuilder> for RouterConfig {
    fn from(builder: RouterConfigBuilder) -> Self {
        builder.config
    }
}

impl RouterConfig {
    /// Create a builder for RouterConfig
    pub fn builder() -> RouterConfigBuilder {
        RouterConfigBuilder::new()
    }

    /// Create a builder from this configuration
    pub fn to_builder(&self) -> RouterConfigBuilder {
        RouterConfigBuilder::from_config_ref(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that .to_builder() round-trip conversion works correctly
    #[test]
    fn test_builder_from_existing_config() {
        let original = RouterConfigBuilder::new()
            .regular_mode(vec!["http://worker1:8000".to_string()])
            .port(3000)
            .build()
            .unwrap();

        let modified = original
            .to_builder()
            .port(4000)
            .enable_metrics("0.0.0.0", 29000)
            .build()
            .unwrap();

        assert_eq!(modified.port, 4000);
        assert!(modified.metrics.is_some());
    }

    /// Test complex routing mode helper method
    #[test]
    fn test_builder_prefill_decode_mode() {
        let config = RouterConfigBuilder::new()
            .prefill_decode_mode(
                vec![("http://prefill:8000".to_string(), Some(8001))],
                vec!["http://decode:8000".to_string()],
            )
            .power_of_two_policy(60)
            .build()
            .unwrap();

        assert!(config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 2);
    }

    /// Test complex policy helper method with multiple parameters
    #[test]
    fn test_builder_cache_aware_policy() {
        let config = RouterConfigBuilder::new()
            .regular_mode(vec!["http://worker1:8000".to_string()])
            .cache_aware_policy(0.8, 10, 1.5, 300, 1000)
            .build()
            .unwrap();

        match config.policy {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.8).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware policy"),
        }
    }

    #[test]
    fn test_builder_new_produces_defaults() {
        let config = RouterConfigBuilder::new().build_unchecked();
        let default = RouterConfig::default();
        assert_eq!(config.port, default.port);
        assert_eq!(config.host, default.host);
        assert!(matches!(config.policy, PolicyConfig::Random));
    }

    #[test]
    fn test_builder_from_config_preserves_all_fields() {
        let original = RouterConfig {
            port: 9999,
            host: "127.0.0.1".to_string(),
            dp_aware: true,
            api_key: Some("secret".to_string()),
            ..Default::default()
        };
        let rebuilt = RouterConfigBuilder::from_config(original.clone()).build_unchecked();
        assert_eq!(rebuilt.port, 9999);
        assert_eq!(rebuilt.host, "127.0.0.1");
        assert!(rebuilt.dp_aware);
        assert_eq!(rebuilt.api_key.as_deref(), Some("secret"));
    }

    #[test]
    fn test_builder_from_config_ref() {
        let original = RouterConfig {
            port: 7777,
            ..Default::default()
        };
        let rebuilt = RouterConfigBuilder::from_config_ref(&original).build_unchecked();
        assert_eq!(rebuilt.port, 7777);
    }

    #[test]
    fn test_builder_regular_mode() {
        let config = RouterConfigBuilder::new()
            .regular_mode(vec![
                "http://w1:8000".to_string(),
                "http://w2:8000".to_string(),
            ])
            .build()
            .unwrap();

        assert!(!config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 2);
    }

    #[test]
    fn test_builder_pd_mode_with_policies() {
        let config = RouterConfigBuilder::new()
            .prefill_decode_mode_with_policies(
                vec![("http://p1:8000".to_string(), Some(8001))],
                vec!["http://d1:8000".to_string()],
                Some(PolicyConfig::RoundRobin),
                Some(PolicyConfig::Random),
            )
            .build()
            .unwrap();

        assert!(config.mode.is_pd_mode());
        if let RoutingMode::PrefillDecode {
            prefill_policy,
            decode_policy,
            ..
        } = &config.mode
        {
            assert!(matches!(prefill_policy, Some(PolicyConfig::RoundRobin)));
            assert!(matches!(decode_policy, Some(PolicyConfig::Random)));
        } else {
            panic!("Expected PrefillDecode mode");
        }
    }

    #[test]
    fn test_builder_mode_direct() {
        let config = RouterConfigBuilder::new()
            .mode(RoutingMode::Regular {
                worker_urls: vec!["http://w:8000".to_string()],
            })
            .build()
            .unwrap();
        assert!(!config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 1);
    }

    #[test]
    fn test_builder_random_policy() {
        let config = RouterConfigBuilder::new()
            .regular_mode(vec!["http://w:8000".to_string()])
            .random_policy()
            .build()
            .unwrap();
        assert!(matches!(config.policy, PolicyConfig::Random));
    }

    #[test]
    fn test_builder_round_robin_policy() {
        let config = RouterConfigBuilder::new()
            .regular_mode(vec!["http://w:8000".to_string()])
            .round_robin_policy()
            .build()
            .unwrap();
        assert!(matches!(config.policy, PolicyConfig::RoundRobin));
    }

    #[test]
    fn test_builder_power_of_two_policy() {
        let config = RouterConfigBuilder::new()
            .regular_mode(vec![
                "http://w1:8000".to_string(),
                "http://w2:8000".to_string(),
            ])
            .power_of_two_policy(120)
            .build()
            .unwrap();
        match config.policy {
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => assert_eq!(load_check_interval_secs, 120),
            _ => panic!("Expected PowerOfTwo policy"),
        }
    }

    #[test]
    fn test_builder_policy_generic() {
        let config = RouterConfigBuilder::new()
            .regular_mode(vec!["http://w:8000".to_string()])
            .policy(PolicyConfig::RoundRobin)
            .build()
            .unwrap();
        assert!(matches!(config.policy, PolicyConfig::RoundRobin));
    }

    #[test]
    fn test_builder_http_connection() {
        let config = RouterConfigBuilder::new()
            .http_connection()
            .build_unchecked();
        assert!(matches!(config.connection_mode, ConnectionMode::Http));
    }

    #[test]
    fn test_builder_grpc_connection() {
        let config = RouterConfigBuilder::new()
            .grpc_connection(Some(50051))
            .build_unchecked();
        assert!(matches!(
            config.connection_mode,
            ConnectionMode::Grpc { port: Some(50051) }
        ));
    }

    #[test]
    fn test_builder_grpc_connection_default() {
        let config = RouterConfigBuilder::new()
            .grpc_connection_default()
            .build_unchecked();
        assert!(matches!(
            config.connection_mode,
            ConnectionMode::Grpc { port: None }
        ));
    }

    #[test]
    fn test_builder_connection_mode_direct() {
        let config = RouterConfigBuilder::new()
            .connection_mode(ConnectionMode::Grpc { port: Some(9000) })
            .build_unchecked();
        assert!(matches!(
            config.connection_mode,
            ConnectionMode::Grpc { port: Some(9000) }
        ));
    }

    #[test]
    fn test_builder_host_and_port() {
        let config = RouterConfigBuilder::new()
            .host("127.0.0.1")
            .port(8080)
            .build_unchecked();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_builder_request_settings() {
        let config = RouterConfigBuilder::new()
            .max_payload_size(1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(120)
            .worker_startup_check_interval_secs(5)
            .build_unchecked();
        assert_eq!(config.max_payload_size, 1024);
        assert_eq!(config.request_timeout_secs, 60);
        assert_eq!(config.worker_startup_timeout_secs, 120);
        assert_eq!(config.worker_startup_check_interval_secs, 5);
    }

    #[test]
    fn test_builder_rate_limiting() {
        let config = RouterConfigBuilder::new()
            .max_concurrent_requests(100)
            .queue_size(50)
            .queue_timeout_secs(30)
            .rate_limit_tokens_per_second(500)
            .build_unchecked();
        assert_eq!(config.max_concurrent_requests, 100);
        assert_eq!(config.queue_size, 50);
        assert_eq!(config.queue_timeout_secs, 30);
        assert_eq!(config.rate_limit_tokens_per_second, Some(500));
    }

    #[test]
    fn test_builder_disable_rate_limiting() {
        let config = RouterConfigBuilder::new()
            .disable_rate_limiting()
            .build_unchecked();
        assert_eq!(config.max_concurrent_requests, -1);
    }

    #[test]
    fn test_builder_api_key() {
        let config = RouterConfigBuilder::new()
            .api_key("my-key")
            .build_unchecked();
        assert_eq!(config.api_key.as_deref(), Some("my-key"));
    }

    #[test]
    fn test_builder_retry_config() {
        let retry = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 5000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        };
        let config = RouterConfigBuilder::new()
            .retry_config(retry)
            .build_unchecked();
        assert_eq!(config.retry.max_retries, 3);
        assert_eq!(config.retry.initial_backoff_ms, 100);
    }

    #[test]
    fn test_builder_disable_enable_retries() {
        let config = RouterConfigBuilder::new()
            .disable_retries()
            .build_unchecked();
        assert!(config.disable_retries);

        let config = RouterConfigBuilder::new()
            .disable_retries()
            .enable_retries()
            .build_unchecked();
        assert!(!config.disable_retries);
    }

    #[test]
    fn test_builder_circuit_breaker_config() {
        let cb = CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 2,
            timeout_duration_secs: 30,
            window_duration_secs: 60,
        };
        let config = RouterConfigBuilder::new()
            .circuit_breaker_config(cb)
            .build_unchecked();
        assert_eq!(config.circuit_breaker.failure_threshold, 5);
    }

    #[test]
    fn test_builder_disable_enable_circuit_breaker() {
        let config = RouterConfigBuilder::new()
            .disable_circuit_breaker()
            .build_unchecked();
        assert!(config.disable_circuit_breaker);

        let config = RouterConfigBuilder::new()
            .disable_circuit_breaker()
            .enable_circuit_breaker()
            .build_unchecked();
        assert!(!config.disable_circuit_breaker);
    }

    #[test]
    fn test_builder_health_check_config() {
        let hc = HealthCheckConfig {
            failure_threshold: 5,
            success_threshold: 1,
            timeout_secs: 10,
            check_interval_secs: 30,
            endpoint: "/ready".to_string(),
            disable_health_check: false,
        };
        let config = RouterConfigBuilder::new()
            .health_check_config(hc)
            .build_unchecked();
        assert_eq!(config.health_check.endpoint, "/ready");
        assert_eq!(config.health_check.failure_threshold, 5);
    }

    #[test]
    fn test_builder_metrics() {
        let config = RouterConfigBuilder::new()
            .enable_metrics("0.0.0.0", 9090)
            .build_unchecked();
        let m = config.metrics.unwrap();
        assert_eq!(m.host, "0.0.0.0");
        assert_eq!(m.port, 9090);
    }

    #[test]
    fn test_builder_metrics_config() {
        let mc = MetricsConfig {
            host: "localhost".to_string(),
            port: 8080,
        };
        let config = RouterConfigBuilder::new()
            .metrics_config(mc)
            .build_unchecked();
        assert_eq!(config.metrics.as_ref().unwrap().port, 8080);
    }

    #[test]
    fn test_builder_logging() {
        let config = RouterConfigBuilder::new()
            .log_dir("/var/log/mesh")
            .log_level("debug")
            .build_unchecked();
        assert_eq!(config.log_dir.as_deref(), Some("/var/log/mesh"));
        assert_eq!(config.log_level.as_deref(), Some("debug"));
    }

    #[test]
    fn test_builder_request_id_headers() {
        let config = RouterConfigBuilder::new()
            .request_id_headers(vec!["X-Request-ID".to_string()])
            .build_unchecked();
        let headers = config.request_id_headers.unwrap();
        assert_eq!(headers, vec!["X-Request-ID"]);
    }

    #[test]
    fn test_builder_model_and_tokenizer_paths() {
        let config = RouterConfigBuilder::new()
            .model_path("meta-llama/Llama-3-8B")
            .tokenizer_path("/models/tokenizer.json")
            .chat_template("/templates/chat.jinja")
            .build_unchecked();
        assert_eq!(config.model_path.as_deref(), Some("meta-llama/Llama-3-8B"));
        assert_eq!(
            config.tokenizer_path.as_deref(),
            Some("/models/tokenizer.json")
        );
        assert_eq!(
            config.chat_template.as_deref(),
            Some("/templates/chat.jinja")
        );
    }

    #[test]
    fn test_builder_parsers() {
        let config = RouterConfigBuilder::new()
            .reasoning_parser("deepseek-r1")
            .tool_call_parser("hermes")
            .build_unchecked();
        assert_eq!(config.reasoning_parser.as_deref(), Some("deepseek-r1"));
        assert_eq!(config.tool_call_parser.as_deref(), Some("hermes"));
    }

    #[test]
    fn test_builder_tokenizer_cache() {
        let config = RouterConfigBuilder::new()
            .enable_l0_cache(5000)
            .enable_l1_cache(100 * 1024 * 1024)
            .build_unchecked();
        assert!(config.tokenizer_cache.enable_l0);
        assert_eq!(config.tokenizer_cache.l0_max_entries, 5000);
        assert!(config.tokenizer_cache.enable_l1);
        assert_eq!(config.tokenizer_cache.l1_max_memory, 100 * 1024 * 1024);
    }

    #[test]
    fn test_builder_tokenizer_cache_config() {
        let cache = TokenizerCacheConfig {
            enable_l0: true,
            l0_max_entries: 1000,
            enable_l1: false,
            l1_max_memory: 0,
        };
        let config = RouterConfigBuilder::new()
            .tokenizer_cache(cache.clone())
            .build_unchecked();
        assert_eq!(config.tokenizer_cache, cache);
    }

    #[test]
    fn test_builder_dp_aware() {
        let config = RouterConfigBuilder::new()
            .enable_dp_aware()
            .build_unchecked();
        assert!(config.dp_aware);

        let config = RouterConfigBuilder::new()
            .enable_dp_aware()
            .disable_dp_aware()
            .build_unchecked();
        assert!(!config.dp_aware);
    }

    #[test]
    fn test_builder_bool_setters() {
        let config = RouterConfigBuilder::new()
            .dp_aware(true)
            .retries(false)
            .circuit_breaker(false)
            .build_unchecked();
        assert!(config.dp_aware);
        assert!(config.disable_retries);
        assert!(config.disable_circuit_breaker);

        let config = RouterConfigBuilder::new()
            .retries(true)
            .circuit_breaker(true)
            .build_unchecked();
        assert!(!config.disable_retries);
        assert!(!config.disable_circuit_breaker);
    }

    #[test]
    fn test_builder_maybe_api_key_some() {
        let config = RouterConfigBuilder::new()
            .maybe_api_key(Some("key123"))
            .build_unchecked();
        assert_eq!(config.api_key.as_deref(), Some("key123"));
    }

    #[test]
    fn test_builder_maybe_api_key_none() {
        let config = RouterConfigBuilder::new()
            .maybe_api_key(None::<String>)
            .build_unchecked();
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_builder_maybe_setters_with_none() {
        let config = RouterConfigBuilder::new()
            .maybe_metrics(None)
            .maybe_log_dir(None::<String>)
            .maybe_log_level(None::<String>)
            .maybe_request_id_headers(None)
            .maybe_rate_limit_tokens_per_second(None)
            .maybe_model_path(None::<String>)
            .maybe_tokenizer_path(None::<String>)
            .maybe_chat_template(None::<String>)
            .maybe_reasoning_parser(None::<String>)
            .maybe_tool_call_parser(None::<String>)
            .build_unchecked();
        assert!(config.metrics.is_none());
        assert!(config.log_dir.is_none());
        assert!(config.log_level.is_none());
        assert!(config.request_id_headers.is_none());
        assert!(config.rate_limit_tokens_per_second.is_none());
        assert!(config.model_path.is_none());
        assert!(config.tokenizer_path.is_none());
        assert!(config.chat_template.is_none());
        assert!(config.reasoning_parser.is_none());
        assert!(config.tool_call_parser.is_none());
    }

    #[test]
    fn test_builder_maybe_setters_with_some() {
        let config = RouterConfigBuilder::new()
            .maybe_log_dir(Some("/logs"))
            .maybe_log_level(Some("info"))
            .maybe_rate_limit_tokens_per_second(Some(100))
            .maybe_model_path(Some("model/path"))
            .maybe_tokenizer_path(Some("tok/path"))
            .maybe_chat_template(Some("tmpl"))
            .maybe_reasoning_parser(Some("rp"))
            .maybe_tool_call_parser(Some("tp"))
            .build_unchecked();
        assert_eq!(config.log_dir.as_deref(), Some("/logs"));
        assert_eq!(config.log_level.as_deref(), Some("info"));
        assert_eq!(config.rate_limit_tokens_per_second, Some(100));
        assert_eq!(config.model_path.as_deref(), Some("model/path"));
        assert_eq!(config.tokenizer_path.as_deref(), Some("tok/path"));
        assert_eq!(config.chat_template.as_deref(), Some("tmpl"));
        assert_eq!(config.reasoning_parser.as_deref(), Some("rp"));
        assert_eq!(config.tool_call_parser.as_deref(), Some("tp"));
    }

    #[test]
    fn test_builder_build_validates() {
        // Port 0 should fail validation
        let result = RouterConfigBuilder::new().port(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_build_unchecked_skips_validation() {
        // Port 0 is invalid but build_unchecked skips validation
        let config = RouterConfigBuilder::new().port(0).build_unchecked();
        assert_eq!(config.port, 0);
    }

    #[test]
    fn test_builder_build_with_validation_false() {
        let result = RouterConfigBuilder::new()
            .port(0)
            .build_with_validation(false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_into_router_config() {
        let builder = RouterConfigBuilder::new().port(5555);
        let config: RouterConfig = builder.into();
        assert_eq!(config.port, 5555);
    }

    #[test]
    fn test_router_config_builder_method() {
        let builder = RouterConfig::builder();
        let config = builder.port(6666).build_unchecked();
        assert_eq!(config.port, 6666);
    }

    #[test]
    fn test_builder_chaining_overwrites_last_wins() {
        let config = RouterConfigBuilder::new()
            .port(1000)
            .port(2000)
            .port(3000)
            .build_unchecked();
        assert_eq!(config.port, 3000);
    }

    #[test]
    fn test_builder_full_complex_config() {
        let config = RouterConfigBuilder::new()
            .prefill_decode_mode_with_policies(
                vec![
                    ("http://p1:8000".to_string(), Some(8001)),
                    ("http://p2:8000".to_string(), None),
                ],
                vec!["http://d1:8000".to_string(), "http://d2:8000".to_string()],
                Some(PolicyConfig::RoundRobin),
                Some(PolicyConfig::Random),
            )
            .cache_aware_policy(0.7, 5, 1.2, 60, 500)
            .host("10.0.0.1")
            .port(8080)
            .max_payload_size(1024 * 1024)
            .request_timeout_secs(300)
            .max_concurrent_requests(200)
            .api_key("secret-key")
            .enable_metrics("0.0.0.0", 9090)
            .log_dir("/var/log")
            .log_level("info")
            .enable_dp_aware()
            .enable_l0_cache(10000)
            .model_path("my-model")
            .reasoning_parser("deepseek-r1")
            .build()
            .unwrap();

        assert!(config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 4);
        assert_eq!(config.host, "10.0.0.1");
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_concurrent_requests, 200);
        assert_eq!(config.api_key.as_deref(), Some("secret-key"));
        assert!(config.metrics.is_some());
        assert!(config.dp_aware);
        assert!(config.tokenizer_cache.enable_l0);
        assert_eq!(config.model_path.as_deref(), Some("my-model"));
    }
}
