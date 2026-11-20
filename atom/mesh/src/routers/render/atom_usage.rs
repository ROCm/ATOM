//! ATOM's additions to an OpenAI `usage`, as `serving_chat.py` and
//! `serving_completion.py` write them: the prefix-cache hit
//! (`prompt_tokens_details.cached_tokens`, the first sibling's), the slowest
//! sibling's `ttft_s` / `tpot_s` / `latency_s`, and `num_choices` past one.
//! Only the ATOM standalone backend sets `WorkerMeta.timing`; for any other
//! the usage is left as it is.

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
/// A text completion, not streamed.
pub(crate) const COMPLETION: Fields = Fields {
    cached_tokens: false,
    timing: true,
};

pub(crate) struct AtomUsage {
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
        Self::of(completes.iter().filter_map(|c| match c {
            TokenChunk::Complete { meta, .. } => Some(meta),
            TokenChunk::Partial { .. } => None,
        }))
    }

    /// Add this endpoint's `fields` to the `usage` object.
    pub fn extend(&self, usage: &mut Map<String, Value>, fields: Fields) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(cached_tokens: u32, timing: Option<SeqTiming>) -> WorkerMeta {
        WorkerMeta {
            cached_tokens,
            timing,
            ..Default::default()
        }
    }

    fn timing(ttft: f64, tpot: f64, latency: f64) -> SeqTiming {
        SeqTiming {
            ttft,
            tpot,
            latency,
        }
    }

    #[test]
    fn an_untimed_backend_adds_nothing() {
        assert!(AtomUsage::of([&meta(3, None)]).is_none());
        assert!(AtomUsage::of(std::iter::empty()).is_none());
    }

    #[test]
    fn siblings_report_the_first_hit_and_the_slowest_times() {
        let metas = [
            meta(8, Some(timing(0.1, 0.02, 1.0))),
            meta(0, Some(timing(0.3, 0.01, 0.5))),
        ];
        let usage = AtomUsage::of(&metas).unwrap();
        let mut chat = Map::new();
        usage.extend(&mut chat, CHAT);
        assert_eq!(
            Value::Object(chat),
            json!({
                "prompt_tokens_details": {"cached_tokens": 8},
                "ttft_s": 0.3, "tpot_s": 0.02, "latency_s": 1.0, "num_choices": 2,
            })
        );
        let mut stream = Map::new();
        usage.extend(&mut stream, CHAT_STREAM);
        assert_eq!(
            Value::Object(stream),
            json!({"prompt_tokens_details": {"cached_tokens": 8}})
        );
        let mut completion = Map::new();
        AtomUsage::of([&meta(8, Some(timing(0.123456, 0.0, 2.0)))])
            .unwrap()
            .extend(&mut completion, COMPLETION);
        assert_eq!(
            Value::Object(completion),
            json!({"ttft_s": 0.1235, "tpot_s": 0.0, "latency_s": 2.0})
        );
    }
}
