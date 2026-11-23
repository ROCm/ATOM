//! The per-engine capability table and its `rl.*` label overrides.

use std::collections::HashMap;

use mesh::rl::{capability::capabilities_for, protocol::RlCapabilitySource};

fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn atom_has_no_static_capability_row() {
    let atom = capabilities_for("atom", &HashMap::new());
    assert_eq!(atom, capabilities_for("unknown-engine", &HashMap::new()));
    assert!(atom.pause_modes.is_empty() && atom.update_from.is_empty());
    assert!(!atom.flush_cache && !atom.reports_weight_version);
}

#[test]
fn rl_flag_labels_take_common_boolean_spellings() {
    for value in ["true", "1", "yes", "on", " TRUE ", "Yes", "ON"] {
        let caps = capabilities_for("vllm", &labels(&[("rl.abort", value)]));
        assert!(caps.abort, "{value:?} reads as true");
        assert_eq!(caps.source, RlCapabilitySource::Label);
    }
    for value in ["false", "0", "no", "off", "False", "NO", "Off"] {
        let caps = capabilities_for("sglang", &labels(&[("rl.flush_cache", value)]));
        assert!(!caps.flush_cache, "{value:?} reads as false");
        assert_eq!(caps.source, RlCapabilitySource::Label);
    }
}

#[test]
fn rl_flag_label_with_an_unknown_value_keeps_the_static_default() {
    let default = capabilities_for("sglang", &HashMap::new());
    for value in ["", "maybe", "2", "enabled"] {
        let caps = capabilities_for("sglang", &labels(&[("rl.flush_cache", value)]));
        assert_eq!(caps, default, "{value:?}");
    }
}

#[test]
fn rl_list_labels_trim_items_and_drop_empty_ones() {
    let caps = capabilities_for(
        "atom",
        &labels(&[
            ("rl.pause_modes", " abort , ,in_place,"),
            ("rl.update_from", "disk,, tensor "),
        ]),
    );
    assert_eq!(caps.pause_modes, ["abort", "in_place"]);
    assert_eq!(caps.update_from, ["disk", "tensor"]);
    assert_eq!(caps.source, RlCapabilitySource::Label);

    let caps = capabilities_for("sglang", &labels(&[("rl.update_from", " , ")]));
    assert!(
        caps.update_from.is_empty(),
        "an all-empty list clears the row"
    );
    assert_eq!(caps.source, RlCapabilitySource::Label);
}

#[test]
fn rl_sleep_wake_and_reports_weight_version_labels_override_the_row() {
    let caps = capabilities_for(
        "sglang",
        &labels(&[
            ("rl.sleep_wake", "false"),
            ("rl.reports_weight_version", "off"),
        ]),
    );
    assert!(!caps.sleep_wake && !caps.reports_weight_version);
    assert_eq!(caps.source, RlCapabilitySource::Label);

    let caps = capabilities_for(
        "atom",
        &labels(&[("rl.sleep_wake", "yes"), ("rl.reports_weight_version", "1")]),
    );
    assert!(caps.sleep_wake && caps.reports_weight_version);
}
