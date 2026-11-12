//! Factory for creating load balancing policies

use std::sync::Arc;

use super::{
    session_affinity::{SessionAffinityPolicy, SessionAffinityScope},
    CacheAwareConfig, CacheAwarePolicy, DpStickyPolicy, LoadBalancingPolicy, PowerOfTwoPolicy,
    PrefixHashConfig, PrefixHashPolicy, RandomPolicy, RoundRobinPolicy,
};
use crate::config::{PolicyConfig, SessionAffinityConfig};

/// Factory for creating policy instances
pub struct PolicyFactory;

impl PolicyFactory {
    /// Create a policy from configuration, using the default session-affinity settings.
    pub fn create_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        Self::create(config, &SessionAffinityConfig::default())
    }

    /// Create a policy and wrap it with a session table.
    ///
    /// Each call gets its own table, so prefill and decode do not share bindings.
    pub fn create(
        config: &PolicyConfig,
        affinity: &SessionAffinityConfig,
    ) -> Arc<dyn LoadBalancingPolicy> {
        Self::create_in(config, affinity, SessionAffinityScope::Regular)
    }

    pub(crate) fn create_in(
        config: &PolicyConfig,
        affinity: &SessionAffinityConfig,
        scope: SessionAffinityScope,
    ) -> Arc<dyn LoadBalancingPolicy> {
        if matches!(config, PolicyConfig::DpSticky) {
            return Arc::new(DpStickyPolicy::with_config(affinity.clone(), scope));
        }
        Arc::new(SessionAffinityPolicy::around_scoped(
            affinity.clone(),
            Self::create_bare(config),
            scope,
        ))
    }

    fn create_bare(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::DpSticky => Arc::new(DpStickyPolicy::new()),
            PolicyConfig::PowerOfTwo { .. } => Arc::new(PowerOfTwoPolicy::new()),
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                let config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                };
                Arc::new(CacheAwarePolicy::with_config(config))
            }
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
            } => {
                let config = PrefixHashConfig {
                    prefix_token_count: *prefix_token_count,
                    load_factor: *load_factor,
                };
                Arc::new(PrefixHashPolicy::new(config))
            }
        }
    }

    /// Create a policy by name (for dynamic loading)
    pub fn create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        Self::create_by_name_with(name, &SessionAffinityConfig::default())
    }

    pub fn create_by_name_with(
        name: &str,
        affinity: &SessionAffinityConfig,
    ) -> Option<Arc<dyn LoadBalancingPolicy>> {
        let bare: Arc<dyn LoadBalancingPolicy> = match name.to_lowercase().as_str() {
            "random" => Arc::new(RandomPolicy::new()),
            "round_robin" | "roundrobin" => Arc::new(RoundRobinPolicy::new()),
            "dp_sticky" | "dpsticky" => {
                return Some(Arc::new(DpStickyPolicy::with_config(
                    affinity.clone(),
                    SessionAffinityScope::Regular,
                )));
            }
            "power_of_two" | "poweroftwo" => Arc::new(PowerOfTwoPolicy::new()),
            "cache_aware" | "cacheaware" => Arc::new(CacheAwarePolicy::new()),
            "prefix_hash" | "prefixhash" => Arc::new(PrefixHashPolicy::with_defaults()),
            _ => return None,
        };
        Some(Arc::new(SessionAffinityPolicy::around_scoped(
            affinity.clone(),
            bare,
            SessionAffinityScope::Regular,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_from_config() {
        let policy = PolicyFactory::create_from_config(&PolicyConfig::Random);
        assert_eq!(policy.name(), "random");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
        assert_eq!(policy.name(), "round_robin");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::DpSticky);
        assert_eq!(policy.name(), "dp_sticky");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        });
        assert_eq!(policy.name(), "power_of_two");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 30,
            max_tree_size: 1000,
        });
        assert_eq!(policy.name(), "cache_aware");
    }

    #[tokio::test]
    async fn test_create_by_name() {
        assert!(PolicyFactory::create_by_name("random").is_some());
        assert!(PolicyFactory::create_by_name("RANDOM").is_some());
        assert!(PolicyFactory::create_by_name("round_robin").is_some());
        assert!(PolicyFactory::create_by_name("RoundRobin").is_some());
        assert!(PolicyFactory::create_by_name("dp_sticky").is_some());
        assert!(PolicyFactory::create_by_name("DpSticky").is_some());
        assert!(PolicyFactory::create_by_name("power_of_two").is_some());
        assert!(PolicyFactory::create_by_name("PowerOfTwo").is_some());
        assert!(PolicyFactory::create_by_name("cache_aware").is_some());
        assert!(PolicyFactory::create_by_name("CacheAware").is_some());
        assert!(PolicyFactory::create_by_name("unknown").is_none());
    }

    #[tokio::test]
    async fn test_create_prefix_hash_from_config() {
        let policy = PolicyFactory::create_from_config(&PolicyConfig::PrefixHash {
            prefix_token_count: 128,
            load_factor: 1.5,
        });
        assert_eq!(policy.name(), "prefix_hash");
    }

    #[tokio::test]
    async fn test_create_by_name_prefix_hash() {
        let p1 = PolicyFactory::create_by_name("prefix_hash");
        assert!(p1.is_some());
        assert_eq!(p1.unwrap().name(), "prefix_hash");

        let p2 = PolicyFactory::create_by_name("PrefixHash");
        assert!(p2.is_some());
        assert_eq!(p2.unwrap().name(), "prefix_hash");
    }

    #[tokio::test]
    async fn test_create_by_name_returns_none_for_empty() {
        assert!(PolicyFactory::create_by_name("").is_none());
    }

    #[tokio::test]
    async fn test_all_configs_produce_named_policies() {
        let configs = vec![
            ("random", PolicyConfig::Random),
            ("round_robin", PolicyConfig::RoundRobin),
            ("dp_sticky", PolicyConfig::DpSticky),
            (
                "power_of_two",
                PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 10,
                },
            ),
            (
                "cache_aware",
                PolicyConfig::CacheAware {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 5,
                    balance_rel_threshold: 1.2,
                    eviction_interval_secs: 60,
                    max_tree_size: 500,
                },
            ),
            (
                "prefix_hash",
                PolicyConfig::PrefixHash {
                    prefix_token_count: 256,
                    load_factor: 1.25,
                },
            ),
        ];

        for (expected_name, config) in configs {
            let policy = PolicyFactory::create_from_config(&config);
            assert_eq!(
                policy.name(),
                expected_name,
                "Config {:?} should produce policy named '{}'",
                config,
                expected_name
            );
        }
    }

    #[tokio::test]
    async fn test_create_by_name_case_insensitive() {
        // All supported names in various cases
        let cases = vec![
            ("random", "random"),
            ("RANDOM", "random"),
            ("Random", "random"),
            ("round_robin", "round_robin"),
            ("roundrobin", "round_robin"),
            ("dp_sticky", "dp_sticky"),
            ("dpsticky", "dp_sticky"),
            ("power_of_two", "power_of_two"),
            ("poweroftwo", "power_of_two"),
            ("cache_aware", "cache_aware"),
            ("cacheaware", "cache_aware"),
            ("prefix_hash", "prefix_hash"),
            ("prefixhash", "prefix_hash"),
        ];

        for (input, expected_name) in cases {
            let policy = PolicyFactory::create_by_name(input);
            assert!(
                policy.is_some(),
                "create_by_name('{}') should return Some",
                input
            );
            assert_eq!(
                policy.unwrap().name(),
                expected_name,
                "create_by_name('{}') should produce '{}'",
                input,
                expected_name
            );
        }
    }

    #[test]
    fn prefill_and_decode_tables_keep_their_scope() {
        let affinity = SessionAffinityConfig::default();
        let prefill = PolicyFactory::create_in(
            &PolicyConfig::Random,
            &affinity,
            SessionAffinityScope::Prefill,
        );
        let decode = PolicyFactory::create_in(
            &PolicyConfig::Random,
            &affinity,
            SessionAffinityScope::Decode,
        );
        assert_eq!(
            prefill
                .as_any()
                .downcast_ref::<SessionAffinityPolicy>()
                .unwrap()
                .scope_label(),
            "prefill"
        );
        assert_eq!(
            decode
                .as_any()
                .downcast_ref::<SessionAffinityPolicy>()
                .unwrap()
                .scope_label(),
            "decode"
        );
    }
}
