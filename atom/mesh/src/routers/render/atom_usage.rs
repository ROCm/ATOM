// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! ATOM's additions to an OpenAI `usage`, as `serving_chat.py` and `serving_completion.py`
//! write them; only an ATOM worker's sequences carry the timing (`WorkerMeta.timing`).

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::routers::token_handle::token_chunk::{SeqTiming, TokenChunk, WorkerMeta};

/// Which of ATOM's fields an endpoint's `usage` carries.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fields {
    pub cached_tokens: bool,
    pub timing: bool,
}

/// A chat completion, not streamed.
pub(crate) const CHAT: Fields = Fields {
    cached_tokens: true,
    timing: true,
};
/// A streamed chat's usage chunk.
pub(crate) const CHAT_STREAM: Fields = Fields {
    cached_tokens: true,
    timing: false,
};
/// A completion, not streamed.
pub(crate) const COMPLETION: Fields = Fields {
    cached_tokens: false,
    timing: true,
};

/// A sequence's timing under the `/generate` meta_info keys it travels in to the
/// completions adapter.
#[derive(Serialize, Deserialize)]
pub(crate) struct MetaInfoTiming {
    ttft_s: f64,
    tpot_s: f64,
    latency_s: f64,
}

impl From<SeqTiming> for MetaInfoTiming {
    fn from(timing: SeqTiming) -> Self {
        Self {
            ttft_s: timing.ttft,
            tpot_s: timing.tpot,
            latency_s: timing.latency,
        }
    }
}

pub(crate) struct AtomUsage {
    /// The prompt the siblings share, which ATOM counts once.
    prompt_tokens: Option<u32>,
    cached_tokens: u32,
    timing: SeqTiming,
    choices: usize,
}

impl AtomUsage {
    /// The request's, from its siblings' metas in index order: the first one's cached
    /// tokens and the slowest timing; none unless ATOM timed them.
    pub fn of<'a>(metas: impl IntoIterator<Item = &'a WorkerMeta>) -> Option<Self> {
        let mut usage: Option<Self> = None;
        for meta in metas {
            let timing = meta.timing?;
            match &mut usage {
                None => {
                    usage = Some(Self {
                        prompt_tokens: None,
                        cached_tokens: meta.cached_tokens,
                        timing,
                        choices: 1,
                    })
                }
                Some(u) => {
                    u.timing.ttft = u.timing.ttft.max(timing.ttft);
                    u.timing.tpot = u.timing.tpot.max(timing.tpot);
                    u.timing.latency = u.timing.latency.max(timing.latency);
                    u.choices += 1;
                }
            }
        }
        usage
    }

    pub fn of_completes(completes: &[TokenChunk]) -> Option<Self> {
        let mut usage = Self::of(completes.iter().filter_map(|c| match c {
            TokenChunk::Complete { meta, .. } => Some(meta),
            TokenChunk::Partial { .. } => None,
        }))?;
        usage.prompt_tokens = completes.iter().find_map(|c| match c {
            TokenChunk::Complete { usage, .. } => Some(usage.prompt_tokens),
            TokenChunk::Partial { .. } => None,
        });
        Some(usage)
    }

    /// The request's, from the `/generate` meta_infos carrying `MetaInfoTiming`.
    pub fn of_meta_infos(meta_infos: &[&Value]) -> Option<Self> {
        let metas = meta_infos
            .iter()
            .map(|info| {
                let timing = MetaInfoTiming::deserialize(*info).ok()?;
                let cached = info.get("cached_tokens").and_then(Value::as_f64);
                Some(WorkerMeta {
                    cached_tokens: cached.unwrap_or(0.0) as u32,
                    timing: Some(SeqTiming {
                        ttft: timing.ttft_s,
                        tpot: timing.tpot_s,
                        latency: timing.latency_s,
                    }),
                    ..Default::default()
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let mut usage = Self::of(&metas)?;
        usage.prompt_tokens = meta_infos
            .first()
            .and_then(|info| info.get("prompt_tokens")?.as_u64())
            .map(|tokens| tokens as u32);
        Some(usage)
    }

    /// Add this endpoint's `fields` to the `usage` object.
    pub fn extend(&self, usage: &mut Map<String, Value>, fields: Fields) {
        if let (Some(prompt), true) = (self.prompt_tokens, self.choices > 1) {
            // Summed over the siblings by the render; ATOM counts it once.
            let completion = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            usage.insert("prompt_tokens".into(), json!(prompt));
            usage.insert("total_tokens".into(), json!(u64::from(prompt) + completion));
        }
        if fields.cached_tokens {
            usage.insert(
                "prompt_tokens_details".into(),
                json!({"cached_tokens": self.cached_tokens}),
            );
        }
        if fields.timing {
            usage.insert("ttft_s".into(), json!(round4(self.timing.ttft)));
            usage.insert("tpot_s".into(), json!(round4(self.timing.tpot)));
            usage.insert("latency_s".into(), json!(round4(self.timing.latency)));
            if self.choices > 1 {
                usage.insert("num_choices".into(), json!(self.choices));
            }
        }
    }

    /// `response` serialized, its `usage` extended with `fields`.
    pub fn extend_response<T: Serialize>(
        &self,
        response: &T,
        fields: Fields,
    ) -> Result<Value, serde_json::Error> {
        let mut value = serde_json::to_value(response)?;
        if let Some(Value::Object(usage)) = value.get_mut("usage") {
            self.extend(usage, fields);
        }
        Ok(value)
    }
}

/// Python's `round(x, 4)`.
fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
