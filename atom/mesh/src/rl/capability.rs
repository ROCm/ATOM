//! Static per-engine capability table, overridable with `rl.*` worker labels.

use std::collections::HashMap;

use super::protocol::{RlCapabilities, RlCapabilitySource};

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn static_for(engine: &str) -> RlCapabilities {
    match engine {
        "sglang" => RlCapabilities {
            source: RlCapabilitySource::Static,
            pause_modes: strings(&["abort", "retract", "in_place"]),
            update_from: strings(&["disk", "tensor", "distributed"]),
            abort: true,
            flush_cache: true,
            sleep_wake: true,
            reports_weight_version: true,
        },
        "vllm" => RlCapabilities {
            source: RlCapabilitySource::Static,
            pause_modes: strings(&["abort", "wait", "keep"]),
            update_from: strings(&["disk", "distributed"]),
            abort: false,
            flush_cache: false,
            sleep_wake: true,
            reports_weight_version: false,
        },
        _ => RlCapabilities {
            source: RlCapabilitySource::Static,
            pause_modes: Vec::new(),
            update_from: Vec::new(),
            abort: false,
            flush_cache: false,
            sleep_wake: false,
            reports_weight_version: false,
        },
    }
}

fn list_label(labels: &HashMap<String, String>, key: &str) -> Option<Vec<String>> {
    labels.get(key).map(|v| {
        v.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    })
}

/// `true/false`, `1/0`, `yes/no` or `on/off`, any case; `None` for any other
/// value, which keeps the static default.
fn bool_label(labels: &HashMap<String, String>, key: &str) -> Option<bool> {
    let v = labels.get(key)?.trim().to_ascii_lowercase();
    match v.as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// The static row for `engine`, with any `rl.*` label overrides applied.
pub fn capabilities_for(engine: &str, labels: &HashMap<String, String>) -> RlCapabilities {
    let mut caps = static_for(engine);
    let mut overridden = false;
    if let Some(v) = list_label(labels, "rl.pause_modes") {
        caps.pause_modes = v;
        overridden = true;
    }
    if let Some(v) = list_label(labels, "rl.update_from") {
        caps.update_from = v;
        overridden = true;
    }
    for (key, slot) in [
        ("rl.abort", &mut caps.abort),
        ("rl.flush_cache", &mut caps.flush_cache),
        ("rl.sleep_wake", &mut caps.sleep_wake),
        (
            "rl.reports_weight_version",
            &mut caps.reports_weight_version,
        ),
    ] {
        if let Some(v) = bool_label(labels, key) {
            *slot = v;
            overridden = true;
        }
    }
    if overridden {
        caps.source = RlCapabilitySource::Label;
    }
    caps
}
