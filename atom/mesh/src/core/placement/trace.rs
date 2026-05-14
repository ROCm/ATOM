use super::types::PlacementTrace;

impl PlacementTrace {
    pub fn for_single(
        model_id: Option<&str>,
        candidate_count_before: usize,
        candidate_count_after: usize,
        selected_url: &str,
        policy_name: &'static str,
        hash_ring_key: Option<&str>,
    ) -> Self {
        Self {
            model_id: model_id.map(|s| s.to_string()),
            candidate_count_before,
            candidate_count_after,
            selected_urls: vec![selected_url.to_string()],
            policy_name: Some(policy_name),
            prefill_policy_name: None,
            decode_policy_name: None,
            hash_ring_key: hash_ring_key.map(|s| s.to_string()),
            notes: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn for_pair(
        model_id: Option<&str>,
        candidate_count_before: usize,
        candidate_count_after: usize,
        prefill_url: &str,
        decode_url: &str,
        prefill_policy: &'static str,
        decode_policy: &'static str,
        hash_ring_key: Option<&str>,
    ) -> Self {
        Self {
            model_id: model_id.map(|s| s.to_string()),
            candidate_count_before,
            candidate_count_after,
            selected_urls: vec![prefill_url.to_string(), decode_url.to_string()],
            policy_name: None,
            prefill_policy_name: Some(prefill_policy),
            decode_policy_name: Some(decode_policy),
            hash_ring_key: hash_ring_key.map(|s| s.to_string()),
            notes: Vec::new(),
        }
    }
}
