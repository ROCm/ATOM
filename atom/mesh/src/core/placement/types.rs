use std::sync::Arc;

use http::HeaderMap;
use thiserror::Error;

use crate::core::{steps::tokenizer_registration::served_template, RuntimeType, Worker};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Http,
    Grpc,
}

#[derive(Debug, Clone, Default)]
pub struct RequestDescriptor<'a> {
    pub model_id: Option<&'a str>,
    pub protocol: Option<Protocol>,
    pub text: Option<&'a str>,
    pub tokens: Option<&'a [u32]>,
    pub headers: Option<&'a HeaderMap>,
    pub stream: bool,
    /// Media items in the request, as ingress counted them.
    pub media_count: usize,
    /// The template a chat's prompt rendered with: the chat goes only to the ATOM
    /// workers that serve it.
    pub chat_template: Option<&'a RenderedTemplate>,
}

impl RequestDescriptor<'_> {
    pub fn carries_media(&self) -> bool {
        self.media_count > 0
    }
}

/// The chat template a chat's prompt rendered with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedTemplate {
    /// `template_fingerprint` of the template.
    pub fingerprint: u64,
    /// The file the tokenizer was loaded with, else `--chat-template`: what an ATOM
    /// worker without its own file serves, as the fingerprint ranks them.
    pub fallback_file: Option<String>,
}

impl RenderedTemplate {
    /// Whether `worker` reads this chat's prompt ids as rendered: a worker other than
    /// ATOM is sent the text as well; an ATOM worker must serve the same template.
    pub fn served_by(&self, worker: &dyn Worker) -> bool {
        worker.metadata().runtime_type != RuntimeType::Atom
            || served_template(worker, self.fallback_file.as_deref()) == self.fingerprint
    }
}

#[derive(Debug)]
pub enum PlacementPlan {
    Single {
        worker: Arc<dyn Worker>,
        policy_name: &'static str,
        /// Prompt tokens charged to `worker` until its prefill finishes.
        prefill_tokens: usize,
    },
    Pair {
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        prefill_policy: &'static str,
        decode_policy: &'static str,
        /// Prompt tokens charged to `prefill` until its prefill finishes.
        prefill_tokens: usize,
    },
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum PlacementError {
    #[error("no workers in registry")]
    NoWorkers,
    #[error("no available (healthy) workers")]
    NoAvailableWorkers,
    #[error("no prefill workers available")]
    NoPrefillWorkers,
    #[error("no decode workers available")]
    NoDecodeWorkers,
    #[error("policy returned None")]
    PolicyReturnedNone,
    #[error("model not found: {model_id}")]
    ModelNotFound { model_id: String },
    #[error("no worker serves the chat template the prompt rendered with")]
    ChatTemplateChanged,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum AdapterError {
    #[error("request body is not a JSON object")]
    BodyNotObject,
    #[error("bootstrap_addr missing for prefill {prefill_url}")]
    BootstrapAddrMissing { prefill_url: String },
    #[error("engine_id missing for prefill {prefill_url} dp_rank {dp_rank}")]
    EngineIdMissing { prefill_url: String, dp_rank: usize },
    #[error("tp_size missing for prefill {prefill_url}")]
    TpSizeMissing { prefill_url: String },
    #[error("PairCtx type does not match adapter")]
    CtxTypeMismatch,
}
