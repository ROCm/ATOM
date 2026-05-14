use super::types::PlacementTrace;

#[test]
fn f01_trace_model_id_matches_request() {
    let some_trace = PlacementTrace::for_single(Some("m1"), 3, 2, "http://w:8000", "rr", Some("m1"));
    assert_eq!(some_trace.model_id.as_deref(), Some("m1"));

    let none_trace = PlacementTrace::for_single(None, 3, 2, "http://w:8000", "rr", None);
    assert!(none_trace.model_id.is_none());
}

#[test]
fn f02_trace_candidate_counts_reflect_before_after_filter() {
    let trace = PlacementTrace::for_single(Some("m"), 10, 4, "http://w:8000", "rr", None);
    assert_eq!(trace.candidate_count_before, 10);
    assert_eq!(trace.candidate_count_after, 4);
}

#[test]
fn f03_trace_selected_urls_length_and_order() {
    let single = PlacementTrace::for_single(Some("m"), 1, 1, "http://only:8000", "rr", None);
    assert_eq!(single.selected_urls, vec!["http://only:8000".to_string()]);

    let pair = PlacementTrace::for_pair(
        Some("m"),
        4,
        2,
        "http://prefill:8000",
        "http://decode:8000",
        "rr",
        "rr",
        None,
    );
    assert_eq!(
        pair.selected_urls,
        vec![
            "http://prefill:8000".to_string(),
            "http://decode:8000".to_string(),
        ]
    );
}

#[test]
fn f04_single_mode_trace_policy_name_matches() {
    let trace = PlacementTrace::for_single(Some("m"), 2, 2, "http://w:8000", "round_robin", None);
    assert_eq!(trace.policy_name, Some("round_robin"));
    assert!(trace.prefill_policy_name.is_none());
    assert!(trace.decode_policy_name.is_none());
}

#[test]
fn f05_pair_mode_trace_has_separate_pd_policy_names() {
    let trace = PlacementTrace::for_pair(
        Some("m"),
        4,
        2,
        "http://p:8000",
        "http://d:8000",
        "round_robin",
        "random",
        None,
    );
    assert!(trace.policy_name.is_none());
    assert_eq!(trace.prefill_policy_name, Some("round_robin"));
    assert_eq!(trace.decode_policy_name, Some("random"));
    assert_ne!(trace.prefill_policy_name, trace.decode_policy_name);
}

#[test]
fn f06_trace_hash_ring_key_never_unknown_model_id() {
    let with_model = PlacementTrace::for_single(
        Some("m1"),
        2,
        2,
        "http://w:8000",
        "prefix_hash",
        Some("m1"),
    );
    assert_eq!(with_model.hash_ring_key.as_deref(), Some("m1"));

    let without_model =
        PlacementTrace::for_single(None, 2, 2, "http://w:8000", "round_robin", None);
    assert!(without_model.hash_ring_key.is_none());

    let pair_no_model = PlacementTrace::for_pair(
        None,
        4,
        2,
        "http://p:8000",
        "http://d:8000",
        "rr",
        "rr",
        None,
    );
    assert!(pair_no_model.hash_ring_key.is_none());
}
