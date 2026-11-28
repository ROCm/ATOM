//! ATOM's additions to an OpenAI `usage`, as `serving_chat.py` and
//! `serving_completion.py` write them: the prefix-cache hit
//! (`prompt_tokens_details.cached_tokens`, the first sibling's), the slowest
//! sibling's `ttft_s` / `tpot_s` / `latency_s`, and `num_choices` past one,
//! on a chat; a completion's are not carried yet.
//! Only ATOM workers, the standalone backend and gRPC workers, set
//! `WorkerMeta.timing`; for any other the usage is left as it is.

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

pub(crate) struct AtomUsage {
    /// The prompt the siblings share, which a chat counts once.
    prompt_tokens: Option<u32>,
    cached_tokens: u32,
    timing: SeqTiming,
    choices: usize,
}

impl AtomUsage {
    /// The request's, from its siblings' metas in index order; none unless
    /// the backend timed them.
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

    /// Add this endpoint's `fields` to the `usage` object.
    pub fn extend(&self, usage: &mut Map<String, Value>, fields: Fields) {
        if let (Some(prompt), true) = (self.prompt_tokens, self.choices > 1) {
            // Summed over the siblings by the render; the server counts it once.
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
