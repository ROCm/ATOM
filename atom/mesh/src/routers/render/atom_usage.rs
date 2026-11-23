//! ATOM's additions to an OpenAI `usage`, as `serving_chat.py` and `serving_completion.py`
//! write them; only an ATOM worker's sequences carry the timing (`WorkerMeta.timing`).

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

/// The `/generate` meta_info keys a sequence's timing travels under to the completions adapter.
const TIMING_KEYS: [&str; 3] = ["ttft_s", "tpot_s", "latency_s"];

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
                    u.timing.ttft_s = u.timing.ttft_s.max(timing.ttft_s);
                    u.timing.tpot_s = u.timing.tpot_s.max(timing.tpot_s);
                    u.timing.latency_s = u.timing.latency_s.max(timing.latency_s);
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

    /// `meta_info` with the timing of `meta`, if ATOM timed it.
    pub fn add_timing(meta_info: &mut Value, meta: &WorkerMeta) {
        let (Some(timing), Some(info)) = (meta.timing, meta_info.as_object_mut()) else {
            return;
        };
        let values = [timing.ttft_s, timing.tpot_s, timing.latency_s];
        for (key, value) in TIMING_KEYS.into_iter().zip(values) {
            info.insert(key.into(), json!(value));
        }
    }

    /// The request's, from the `/generate` meta_infos `add_timing` filled.
    pub fn of_meta_infos(meta_infos: &[&Value]) -> Option<Self> {
        let metas = meta_infos
            .iter()
            .map(|info| {
                let field = |key: &str| info.get(key).and_then(Value::as_f64);
                let [ttft_s, tpot_s, latency_s] = TIMING_KEYS.map(field);
                Some(WorkerMeta {
                    cached_tokens: field("cached_tokens").unwrap_or(0.0) as u32,
                    timing: Some(SeqTiming {
                        ttft_s: ttft_s?,
                        tpot_s: tpot_s?,
                        latency_s: latency_s?,
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
            usage.insert("ttft_s".into(), json!(round4(self.timing.ttft_s)));
            usage.insert("tpot_s".into(), json!(round4(self.timing.tpot_s)));
            usage.insert("latency_s".into(), json!(round4(self.timing.latency_s)));
            if self.choices > 1 {
                usage.insert("num_choices".into(), json!(self.choices));
            }
        }
    }

    /// `response` serialized, its `usage` extended with `fields`.
    pub fn extend_response<T: serde::Serialize>(
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
