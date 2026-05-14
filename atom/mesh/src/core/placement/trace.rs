use super::types::PlacementTrace;

impl PlacementTrace {
    #[allow(clippy::too_many_arguments)]
    pub fn for_single(
        model_id: Option<&str>,
        candidate_count_before: usize,
        candidate_count_after: usize,
        selected_url: &str,
        policy_name: &'static str,
        hash_ring_key: Option<&str>,
    ) -> Self {
        let _ = (
            model_id,
            candidate_count_before,
            candidate_count_after,
            selected_url,
            policy_name,
            hash_ring_key,
        );
        todo!()
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
        let _ = (
            model_id,
            candidate_count_before,
            candidate_count_after,
            prefill_url,
            decode_url,
            prefill_policy,
            decode_policy,
            hash_ring_key,
        );
        todo!()
    }
}
