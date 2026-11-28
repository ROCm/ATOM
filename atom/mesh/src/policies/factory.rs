//! Factory for creating load balancing policies

use std::sync::Arc;

use super::{
    session_affinity::SessionAffinityScope, CacheAwareConfig, CacheAwarePolicy,
    LoadBalancingPolicy, PowerOfTwoPolicy, PrefixHashConfig, PrefixHashPolicy, RandomPolicy,
    RoundRobinPolicy, SessionAffinityPolicy,
};
use crate::config::{PolicyConfig, SessionAffinityConfig, SessionParentMode};

/// Factory for creating policy instances
pub struct PolicyFactory;

impl PolicyFactory {
    /// Create a policy from configuration
    pub fn create_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::DpSticky => Arc::new(SessionAffinityPolicy::dp_sticky()),
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

    /// Create a policy with a session table that uses `affinity` and reports
    /// `scope`, unless session affinity is off; `dp_sticky` always has one.
    /// Each call gets its own table, so prefill and decode do not share bindings.
    pub(crate) fn create_in(
        config: &PolicyConfig,
        affinity: &SessionAffinityConfig,
        scope: SessionAffinityScope,
    ) -> Arc<dyn LoadBalancingPolicy> {
        if matches!(config, PolicyConfig::DpSticky) {
            return Arc::new(SessionAffinityPolicy::dp_sticky_scoped(
                dp_sticky_affinity(affinity),
                scope,
            ));
        }
        Self::wrap(affinity, Self::create_from_config(config), scope)
    }

    fn wrap(
        affinity: &SessionAffinityConfig,
        bare: Arc<dyn LoadBalancingPolicy>,
        scope: SessionAffinityScope,
    ) -> Arc<dyn LoadBalancingPolicy> {
        if !affinity.wraps(scope.pool()) {
            return bare;
        }
        Arc::new(SessionAffinityPolicy::around_scoped(
            affinity.clone(),
            bare,
            scope,
        ))
    }

    /// Create a policy by name (for dynamic loading)
    pub fn create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        match name.to_lowercase().as_str() {
            "random" => Some(Arc::new(RandomPolicy::new())),
            "round_robin" | "roundrobin" => Some(Arc::new(RoundRobinPolicy::new())),
            "dp_sticky" | "dpsticky" => Some(Arc::new(SessionAffinityPolicy::dp_sticky())),
            "power_of_two" | "poweroftwo" => Some(Arc::new(PowerOfTwoPolicy::new())),
            "cache_aware" | "cacheaware" => Some(Arc::new(CacheAwarePolicy::new())),
            "prefix_hash" | "prefixhash" => Some(Arc::new(PrefixHashPolicy::with_defaults())),
            _ => None,
        }
    }

    /// [`Self::create_by_name`], with the session table [`Self::create_in`] adds.
    pub(crate) fn create_by_name_with(
        name: &str,
        affinity: &SessionAffinityConfig,
    ) -> Option<Arc<dyn LoadBalancingPolicy>> {
        match name.to_lowercase().as_str() {
            "dp_sticky" | "dpsticky" => Some(Arc::new(SessionAffinityPolicy::dp_sticky_scoped(
                dp_sticky_affinity(affinity),
                SessionAffinityScope::Regular,
            ))),
            _ => Self::create_by_name(name)
                .map(|bare| Self::wrap(affinity, bare, SessionAffinityScope::Regular)),
        }
    }
}

/// `dp_sticky` pins sessions whether or not session affinity is on. Off, it keys
/// on `x-session-id` only, without parent groups, as it did before session
/// affinity existed.
fn dp_sticky_affinity(affinity: &SessionAffinityConfig) -> SessionAffinityConfig {
    if affinity.is_enabled() {
        return affinity.clone();
    }
    SessionAffinityConfig {
        session_id_headers: vec!["x-session-id".to_string()],
        parent_mode: SessionParentMode::Independent,
        ..affinity.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SessionAffinityPool;

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

    #[tokio::test]
    async fn affinity_off_leaves_policies_bare_and_dp_sticky_on_x_session_id_only() {
        let off = SessionAffinityConfig {
            session_id_headers: Vec::new(),
            parent_mode: SessionParentMode::Group,
            ..SessionAffinityConfig::default()
        };
        let random =
            PolicyFactory::create_in(&PolicyConfig::Random, &off, SessionAffinityScope::Regular);
        assert!(random.as_any().downcast_ref::<RandomPolicy>().is_some());
        let named = PolicyFactory::create_by_name_with("round_robin", &off).unwrap();
        assert!(named.as_any().downcast_ref::<RoundRobinPolicy>().is_some());

        let dp_sticky =
            PolicyFactory::create_in(&PolicyConfig::DpSticky, &off, SessionAffinityScope::Regular);
        let workers: Vec<Arc<dyn crate::core::Worker>> = ["http://w1:8000", "http://w2:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(crate::core::BasicWorkerBuilder::new(url).build())
                    as Arc<dyn crate::core::Worker>
            })
            .collect();
        // Children of one parent do not share a binding: parent groups are off too.
        for (header, session, branch) in [
            ("x-session-id", "session-1", "new"),
            ("x-session-affinity", "session-2", "no_session"),
            ("x-session-id", "session-3", "new"),
        ] {
            let mut headers = http::HeaderMap::new();
            headers.insert(header, session.parse().unwrap());
            headers.insert("x-dynamo-parent-session-id", "root".parse().unwrap());
            let info = crate::policies::SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            let (_, decision) = dp_sticky.select_with_decision(&workers, &info).await;
            assert_eq!(decision.branch, branch, "{header}");
        }
    }

    #[test]
    fn other_policies_get_the_table_only_in_listed_pools() {
        let wrapped = |policy: &Arc<dyn LoadBalancingPolicy>| {
            policy
                .as_any()
                .downcast_ref::<SessionAffinityPolicy>()
                .is_some()
        };
        let prefill_only = SessionAffinityConfig {
            pools: vec![SessionAffinityPool::Prefill],
            ..SessionAffinityConfig::default()
        };
        for (affinity, scope, expected) in [
            (
                SessionAffinityConfig::default(),
                SessionAffinityScope::Regular,
                false,
            ),
            (prefill_only.clone(), SessionAffinityScope::Prefill, true),
            (prefill_only.clone(), SessionAffinityScope::Decode, false),
            (prefill_only, SessionAffinityScope::Regular, false),
        ] {
            let policy = PolicyFactory::create_in(&PolicyConfig::Random, &affinity, scope);
            assert_eq!(wrapped(&policy), expected, "{scope:?}");
            // dp_sticky always pins sessions.
            let dp_sticky = PolicyFactory::create_in(&PolicyConfig::DpSticky, &affinity, scope);
            assert!(wrapped(&dp_sticky), "{scope:?}");
        }
        let named = PolicyFactory::create_by_name_with("random", &SessionAffinityConfig::default())
            .unwrap();
        assert!(!wrapped(&named));
    }

    #[test]
    fn prefill_and_decode_tables_keep_their_scope() {
        let affinity = SessionAffinityConfig {
            pools: vec![SessionAffinityPool::Prefill, SessionAffinityPool::Decode],
            ..SessionAffinityConfig::default()
        };
        for (scope, label) in [
            (SessionAffinityScope::Prefill, "prefill"),
            (SessionAffinityScope::Decode, "decode"),
        ] {
            let policy = PolicyFactory::create_in(&PolicyConfig::DpSticky, &affinity, scope);
            let dp_sticky = policy
                .as_any()
                .downcast_ref::<SessionAffinityPolicy>()
                .unwrap();
            assert_eq!(dp_sticky.scope_label(), label);

            let policy = PolicyFactory::create_in(&PolicyConfig::Random, &affinity, scope);
            let wrapped = policy
                .as_any()
                .downcast_ref::<SessionAffinityPolicy>()
                .unwrap();
            assert_eq!(wrapped.scope_label(), label);
        }
    }
}
