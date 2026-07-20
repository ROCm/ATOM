use std::sync::{Arc, OnceLock};

use dashmap::DashMap;
use tracing::{debug, info, warn};

/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{CacheAwarePolicy, LoadBalancingPolicy, PolicyFactory};
use crate::{config::types::PolicyConfig, core::Worker};

/// Registry for managing model-to-policy mappings
#[derive(Clone)]
pub struct PolicyRegistry {
    /// Model ID -> Policy instance mapping (lock-free reads via DashMap)
    model_policies: Arc<DashMap<String, Arc<dyn LoadBalancingPolicy>>>,

    /// Model ID -> Worker count for cleanup tracking (lock-free reads via DashMap)
    model_worker_counts: Arc<DashMap<String, usize>>,

    /// Default policy instance (cached, immutable after creation)
    default_policy: Arc<dyn LoadBalancingPolicy>,

    /// Prefill policy for PD mode (set once at startup, lock-free reads via OnceLock)
    prefill_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Decode policy for PD mode (set once at startup, lock-free reads via OnceLock)
    decode_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,
}

impl PolicyRegistry {
    /// Create a new PolicyRegistry with a default policy
    pub fn new(default_policy_config: PolicyConfig) -> Self {
        let default_policy = Self::create_policy_from_config(&default_policy_config);

        Self {
            model_policies: Arc::new(DashMap::new()),
            model_worker_counts: Arc::new(DashMap::new()),
            default_policy,
            prefill_policy: Arc::new(OnceLock::new()),
            decode_policy: Arc::new(OnceLock::new()),
        }
    }

    /// Called when a worker is added
    /// Returns the policy that should be used for this worker's model
    pub fn on_worker_added(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // Increment worker count using DashMap entry API
        let count = self
            .model_worker_counts
            .entry(model_id.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
        debug!("Worker added for model {}, count: {}", model_id, *count);
        drop(count); // Release the entry lock

        // Check if model already has a policy (lock-free read via DashMap)
        if let Some(existing_policy) = self.model_policies.get(model_id) {
            debug!(
                "Model {} already has policy: {}",
                model_id,
                existing_policy.name()
            );
            return Arc::clone(&existing_policy);
        }

        // New model - determine policy
        let policy = self.determine_policy_for_model(model_id, policy_hint);

        info!(
            "Assigning policy {} to new model {}",
            policy.name(),
            model_id
        );

        // Store policy for this model (DashMap handles concurrent inserts)
        self.model_policies
            .insert(model_id.to_string(), Arc::clone(&policy));

        policy
    }

    /// Called when a worker is removed
    pub fn on_worker_removed(&self, model_id: &str) {
        // Decrement worker count and check if cleanup needed
        let should_cleanup = if let Some(mut count_ref) = self.model_worker_counts.get_mut(model_id)
        {
            *count_ref = count_ref.saturating_sub(1);
            debug!(
                "Worker removed for model {}, count: {}",
                model_id, *count_ref
            );
            if *count_ref == 0 {
                drop(count_ref); // Release before remove
                self.model_worker_counts.remove(model_id);
                true
            } else {
                false
            }
        } else {
            warn!(
                "Attempted to remove worker for model {} with no registered workers",
                model_id
            );
            false
        };

        // Clean up policy if this was the last worker
        if should_cleanup {
            if let Some((_, policy)) = self.model_policies.remove(model_id) {
                info!(
                    "Removed policy {} for model {} (last worker removed)",
                    policy.name(),
                    model_id
                );
            }
        }
    }

    /// Get the policy for a model (lock-free via DashMap)
    pub fn get_policy(&self, model_id: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        self.model_policies.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Get the default policy
    pub fn get_default_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        Arc::clone(&self.default_policy)
    }

    /// Get policy for a model, or default if not found
    pub fn get_policy_or_default(&self, model_id: &str) -> Arc<dyn LoadBalancingPolicy> {
        self.get_policy(model_id)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Determine policy for a new model
    fn determine_policy_for_model(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // 1. Check policy hint from worker
        if let Some(policy_type) = policy_hint {
            debug!("Using policy hint '{}' for model {}", policy_type, model_id);
            return self.create_policy_from_type(policy_type);
        }

        // 2. Use default policy
        debug!("Using default policy for model {}", model_id);
        Arc::clone(&self.default_policy)
    }

    /// Create a policy from a type string (delegates to PolicyFactory)
    fn create_policy_from_type(&self, policy_type: &str) -> Arc<dyn LoadBalancingPolicy> {
        PolicyFactory::create_by_name(policy_type).unwrap_or_else(|| {
            warn!("Unknown policy type '{}', using default", policy_type);
            Arc::clone(&self.default_policy)
        })
    }

    /// Create a policy from a PolicyConfig (delegates to PolicyFactory)
    fn create_policy_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        PolicyFactory::create_from_config(config)
    }

    /// Get current model->policy mappings (for debugging/monitoring)
    pub fn get_all_mappings(&self) -> std::collections::HashMap<String, String> {
        self.model_policies
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().name().to_string()))
            .collect()
    }

    /// Get worker counts per model
    pub fn get_worker_counts(&self) -> std::collections::HashMap<String, usize> {
        self.model_worker_counts
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    /// Clear all policies (useful for testing)
    pub fn clear(&self) {
        self.model_policies.clear();
        self.model_worker_counts.clear();
    }

    /// Set the prefill policy for PD mode (lock-free, set once at startup)
    pub fn set_prefill_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.prefill_policy.set(policy);
    }

    /// Set the decode policy for PD mode (lock-free, set once at startup)
    pub fn set_decode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.decode_policy.set(policy);
    }

    /// Get the prefill policy for PD mode, or default if not set (lock-free)
    pub fn get_prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.prefill_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get the decode policy for PD mode, or default if not set (lock-free)
    pub fn get_decode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.decode_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get all PowerOfTwo policies that need load updates (lock-free)
    pub fn get_all_power_of_two_policies(&self) -> Vec<Arc<dyn LoadBalancingPolicy>> {
        let mut power_of_two_policies = Vec::new();

        if self.default_policy.name() == "power_of_two" {
            power_of_two_policies.push(Arc::clone(&self.default_policy));
        }

        // Get prefill and decode policies (lock-free via OnceLock::get)
        let prefill_policy_opt = self.prefill_policy.get();
        let decode_policy_opt = self.decode_policy.get();

        if let Some(policy) = prefill_policy_opt {
            if policy.name() == "power_of_two" && !Arc::ptr_eq(policy, &self.default_policy) {
                power_of_two_policies.push(Arc::clone(policy));
            }
        }

        if let Some(policy) = decode_policy_opt {
            if policy.name() == "power_of_two"
                && !Arc::ptr_eq(policy, &self.default_policy)
                && !prefill_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                power_of_two_policies.push(Arc::clone(policy));
            }
        }

        for entry in self.model_policies.iter() {
            let policy = entry.value();
            if policy.name() == "power_of_two" {
                let already_added = power_of_two_policies.iter().any(|p| Arc::ptr_eq(p, policy));
                if !already_added {
                    power_of_two_policies.push(Arc::clone(policy));
                }
            }
        }

        power_of_two_policies
    }

    /// Initialize cache-aware policy with workers if applicable
    /// This should be called after workers are registered for a model
    pub fn init_cache_aware_policy(&self, model_id: &str, workers: &[Arc<dyn Worker>]) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    debug!(
                        "Initializing cache-aware policy with {} workers for model {}",
                        workers.len(),
                        model_id
                    );
                    cache_aware.init_workers(workers);
                }
            }
        }
    }

    /// Remove a worker from cache-aware policy if applicable
    /// This should be called when a worker is being removed
    pub fn remove_worker_from_cache_aware(&self, model_id: &str, worker_url: &str) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    cache_aware.remove_worker_by_url(worker_url);
                    debug!(
                        "Removed worker {} from cache-aware policy for model {}",
                        worker_url, model_id
                    );
                }
            }
        }
    }

    /// Initialize cache-aware policies for PD mode (prefill and decode) - lock-free
    pub fn init_pd_cache_aware_policies(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
    ) {
        // Initialize prefill policy if it's cache-aware (lock-free via OnceLock::get)
        if let Some(prefill_policy) = self.prefill_policy.get() {
            if prefill_policy.name() == "cache_aware" {
                if let Some(cache_aware) =
                    prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill cache-aware policy with {} workers",
                            prefill_workers.len()
                        );
                        cache_aware.init_workers(prefill_workers);
                    }
                }
            }
        }

        // Initialize decode policy if it's cache-aware (lock-free via OnceLock::get)
        if let Some(decode_policy) = self.decode_policy.get() {
            if decode_policy.name() == "cache_aware" {
                if let Some(cache_aware) = decode_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !decode_workers.is_empty() {
                        debug!(
                            "Initializing decode cache-aware policy with {} workers",
                            decode_workers.len()
                        );
                        cache_aware.init_workers(decode_workers);
                    }
                }
            }
        }
    }
}

impl std::fmt::Debug for PolicyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyRegistry")
            .field("model_policies", &self.model_policies)
            .field("model_worker_counts", &self.model_worker_counts)
            .field("default_policy", &self.default_policy.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_policy_registry_basic() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // First worker of a model sets the policy
        let policy1 = registry.on_worker_added("llama-3", Some("cache_aware"));
        assert_eq!(policy1.name(), "cache_aware");

        // Second worker of same model uses existing policy
        let policy2 = registry.on_worker_added("llama-3", Some("round_robin"));
        assert_eq!(policy2.name(), "cache_aware"); // Ignores hint, uses existing

        // Different model can have different policy
        let policy3 = registry.on_worker_added("gpt-4", Some("random"));
        assert_eq!(policy3.name(), "random");

        // Check mappings
        let mappings = registry.get_all_mappings();
        assert_eq!(mappings.get("llama-3").unwrap(), "cache_aware");
        assert_eq!(mappings.get("gpt-4").unwrap(), "random");

        // Check worker counts
        let counts = registry.get_worker_counts();
        assert_eq!(*counts.get("llama-3").unwrap(), 2);
        assert_eq!(*counts.get("gpt-4").unwrap(), 1);
    }

    #[tokio::test]
    async fn test_policy_registry_cleanup() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Add workers
        registry.on_worker_added("llama-3", Some("cache_aware"));
        registry.on_worker_added("llama-3", None);
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&2));

        // Remove one worker - policy should remain
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_some());
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&1));

        // Remove last worker - policy should be cleaned up
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_none());
        assert_eq!(registry.get_worker_counts().get("llama-3"), None);
    }

    #[tokio::test]
    async fn test_default_policy() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // No hint, no template - uses default
        let policy = registry.on_worker_added("unknown-model", None);
        assert_eq!(policy.name(), "round_robin");

        // Get default directly
        let default = registry.get_default_policy();
        assert_eq!(default.name(), "round_robin");
    }

    #[tokio::test]
    async fn test_get_policy_or_default() {
        let registry = PolicyRegistry::new(PolicyConfig::Random);

        // No policy for model -> returns default
        let policy = registry.get_policy_or_default("no-model");
        assert_eq!(policy.name(), "random");

        // Add worker with explicit policy -> returns that policy
        registry.on_worker_added("llama", Some("round_robin"));
        let policy = registry.get_policy_or_default("llama");
        assert_eq!(policy.name(), "round_robin");
    }

    #[tokio::test]
    async fn test_pd_mode_policies() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Before setting PD policies, should fallback to default
        let prefill = registry.get_prefill_policy();
        assert_eq!(prefill.name(), "round_robin");
        let decode = registry.get_decode_policy();
        assert_eq!(decode.name(), "round_robin");

        // Set PD policies
        registry.set_prefill_policy(Arc::new(crate::policies::RandomPolicy::new()));
        registry.set_decode_policy(Arc::new(crate::policies::PowerOfTwoPolicy::new()));

        assert_eq!(registry.get_prefill_policy().name(), "random");
        assert_eq!(registry.get_decode_policy().name(), "power_of_two");
    }

    #[tokio::test]
    async fn test_pd_policies_set_once() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // First set succeeds
        registry.set_prefill_policy(Arc::new(crate::policies::RandomPolicy::new()));
        assert_eq!(registry.get_prefill_policy().name(), "random");

        // Second set is ignored (OnceLock)
        registry.set_prefill_policy(Arc::new(crate::policies::PowerOfTwoPolicy::new()));
        assert_eq!(registry.get_prefill_policy().name(), "random"); // Still random
    }

    #[tokio::test]
    async fn test_get_all_power_of_two_policies_default() {
        let registry = PolicyRegistry::new(PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 10,
        });

        let p2_policies = registry.get_all_power_of_two_policies();
        assert_eq!(p2_policies.len(), 1);
        assert_eq!(p2_policies[0].name(), "power_of_two");
    }

    #[tokio::test]
    async fn test_get_all_power_of_two_policies_with_pd() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Set prefill to power_of_two
        registry.set_prefill_policy(Arc::new(crate::policies::PowerOfTwoPolicy::new()));

        let p2_policies = registry.get_all_power_of_two_policies();
        assert_eq!(p2_policies.len(), 1);

        // Also set decode to power_of_two
        registry.set_decode_policy(Arc::new(crate::policies::PowerOfTwoPolicy::new()));

        let p2_policies = registry.get_all_power_of_two_policies();
        assert_eq!(p2_policies.len(), 2); // prefill + decode (different instances)
    }

    #[tokio::test]
    async fn test_get_all_power_of_two_no_duplicates() {
        // Default is power_of_two, but it shouldn't appear twice
        let registry = PolicyRegistry::new(PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 10,
        });

        // Add a model-specific power_of_two via worker
        registry.on_worker_added("llama", Some("power_of_two"));

        let p2_policies = registry.get_all_power_of_two_policies();
        // default + model-specific = 2 distinct instances
        assert_eq!(p2_policies.len(), 2);
    }

    #[tokio::test]
    async fn test_clear() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        registry.on_worker_added("llama", Some("random"));
        registry.on_worker_added("gpt-4", Some("cache_aware"));
        assert_eq!(registry.get_all_mappings().len(), 2);

        registry.clear();
        assert_eq!(registry.get_all_mappings().len(), 0);
        assert_eq!(registry.get_worker_counts().len(), 0);
    }

    #[tokio::test]
    async fn test_on_worker_removed_nonexistent() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Removing from a model with no workers should not panic
        registry.on_worker_removed("nonexistent-model");
    }

    #[tokio::test]
    async fn test_multiple_models_independent() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        registry.on_worker_added("llama", Some("random"));
        registry.on_worker_added("gpt-4", Some("cache_aware"));

        // Remove all workers from llama
        registry.on_worker_removed("llama");

        // llama policy should be cleaned up, gpt-4 should remain
        assert!(registry.get_policy("llama").is_none());
        assert_eq!(registry.get_policy("gpt-4").unwrap().name(), "cache_aware");
    }

    #[tokio::test]
    async fn test_create_policy_from_type_unknown() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Unknown policy type should fall back to default
        let policy = registry.on_worker_added("model", Some("nonexistent_policy"));
        assert_eq!(policy.name(), "round_robin"); // Falls back to default
    }
}
