//! The token bridge: requests go to Python's `TokenBridge`
//! (`atom/entrypoints/atomesh/token_bridge.py`), which hands their token ids
//! to LLMEngine / CoreManager; each engine step comes back in one
//! `StepSink.push`.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
use tracing::{debug, warn};

use super::{
    events::{
        submitted, Aborter, Admission, EventRegistry, SeqEvent, SeqEventSource, SeqFinish, Stage,
    },
    RequestExtras,
};
use crate::{
    core::placement::types::PlacementPlan,
    protocols::common::StringOrArray,
    routers::{
        grpc::engine::Dispatcher,
        prepare::generation_payload::GenerationPayload,
        token_handle::{engine_error::EngineError, token_handle::TokenHandle},
    },
};

/// One sequence's share of an engine step, as `TokenBridge` pushes it:
/// `(key, sibling_index, new_token_ids, finished, finish_reason,
/// num_cached_tokens)`.
type StepOutput = (u64, u32, Vec<u32>, bool, Option<String>, u32);

/// Where the engine's steps arrive. Python builds one, gives it to its
/// `TokenBridge`, and Rust finds it there (`AtomBridgeRuntime`).
#[pyclass(name = "StepSink", frozen)]
pub struct StepSink {
    pub(crate) registry: Arc<EventRegistry>,
}

#[pymethods]
impl StepSink {
    #[new]
    pub(crate) fn new() -> Self {
        Self {
            registry: Arc::new(EventRegistry::default()),
        }
    }

    /// One engine step: a `StepOutput` per request that moved.
    fn push(&self, outputs: Vec<StepOutput>) {
        for (key, index, token_ids, finished, reason, cached_tokens) in outputs {
            let finish = finished.then_some(SeqFinish {
                reason,
                cached_tokens,
            });
            self.registry.deliver(
                key,
                SeqEvent::Step {
                    index,
                    token_ids,
                    finish,
                },
            );
        }
    }

    /// End every open stream with an error (the engine is gone).
    fn fail_all(&self, message: &str) {
        self.registry.fail_all(message);
    }

    /// End these requests' streams with an error: a step of theirs was lost.
    fn fail(&self, keys: Vec<u64>, message: &str) {
        for key in keys {
            self.registry
                .deliver(key, SeqEvent::Failed(message.to_string()));
        }
    }
}

/// What the bridge submits besides the prompt's token ids. Fields the engine
/// has no use for (penalties, `min_p`) are dropped, as the OpenAI server
/// drops them; ones it cannot honour yet are refused (`from_payload`).
#[derive(Debug, Clone)]
pub(crate) struct BridgeRequest {
    pub request_id: String,
    pub token_ids: Vec<u32>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub n: i32,
    pub max_tokens: Option<i32>,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    pub extras: RequestExtras,
}

impl BridgeRequest {
    pub(crate) fn from_payload(
        payload: &GenerationPayload,
        extras: &RequestExtras,
    ) -> Result<Self, String> {
        let refused = [
            (payload.token_ids.is_empty(), "the prompt is empty"),
            (
                payload.stream && payload.sampling.n > 1,
                "streaming with n > 1 is not supported by the ATOM Rust frontend yet",
            ),
            (
                payload.logprob.return_logprob || payload.logprob.logprob_start_len >= 0,
                "logprobs are not supported by the ATOM Rust frontend yet",
            ),
            (
                payload
                    .stop
                    .stop_token_ids
                    .as_ref()
                    .is_some_and(|ids| !ids.is_empty()),
                "stop_token_ids is not supported by the ATOM engine",
            ),
            (
                payload.tool_constraints.is_some(),
                "tool_choice that constrains the output is not supported by the ATOM Rust \
                 frontend yet",
            ),
            (
                payload.return_hidden_states,
                "return_hidden_states is not supported by the ATOM engine",
            ),
            (
                payload.sampling.presence_penalty != 0.0,
                "presence_penalty is not supported by the ATOM Rust frontend yet",
            ),
        ];
        if let Some((_, reason)) = refused.iter().find(|(refuse, _)| *refuse) {
            return Err(reason.to_string());
        }
        let stop = match &payload.stop.stop {
            None => Vec::new(),
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(v)) => v.clone(),
        };
        Ok(Self {
            request_id: payload.request_id.clone(),
            token_ids: payload.token_ids.clone(),
            temperature: payload.sampling.temperature,
            top_p: payload.sampling.top_p,
            top_k: payload.sampling.top_k,
            n: payload.sampling.n,
            max_tokens: payload.sampling.max_new_tokens,
            stop,
            ignore_eos: payload.sampling.ignore_eos,
            extras: extras.clone(),
        })
    }

    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let params = PyDict::new(py);
        let extras = &self.extras;
        params.set_item("request_id", &self.request_id)?;
        params.set_item("temperature", self.temperature)?;
        params.set_item("top_p", self.top_p)?;
        params.set_item("top_k", self.top_k)?;
        params.set_item("n", self.n)?;
        params.set_item("max_tokens", self.max_tokens)?;
        params.set_item("stop", PyList::new(py, &self.stop)?)?;
        params.set_item("ignore_eos", self.ignore_eos)?;
        params.set_item("data_parallel_rank", extras.data_parallel_rank)?;
        params.set_item("tool_choice_required", extras.tool_choice_required)?;
        params.set_item("dp_session_id", &extras.dp_session_id)?;
        params.set_item("dp_parent_session_id", &extras.dp_parent_session_id)?;
        params.set_item("chat", &extras.chat_json)?;
        Ok(params)
    }
}

/// Stops a request through `TokenBridge.abort`, off the async workers.
struct PyAborter {
    bridge: Arc<Py<PyAny>>,
}

impl Aborter for PyAborter {
    fn abort(&self, key: u64) {
        let bridge = Arc::clone(&self.bridge);
        let abort = move || call_abort(&bridge, key);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn_blocking(abort);
            }
            Err(_) => abort(),
        }
    }
}

fn call_abort(bridge: &Py<PyAny>, key: u64) {
    Python::attach(|py| {
        if let Err(e) = bridge.bind(py).call_method1("abort", (vec![key],)) {
            warn!(key, error = %e, "TokenBridge.abort failed");
        }
    });
}

/// `Dispatcher` over the token bridge. Placement is ignored: there is one
/// engine, and CoreManager places requests on its DP ranks itself.
pub(crate) struct BridgeDispatcher {
    bridge: Arc<Py<PyAny>>,
    registry: Arc<EventRegistry>,
}

impl BridgeDispatcher {
    pub(crate) fn new(bridge: Arc<Py<PyAny>>, registry: Arc<EventRegistry>) -> Self {
        Self { bridge, registry }
    }
}

#[async_trait]
impl Dispatcher for BridgeDispatcher {
    async fn dispatch(
        &self,
        _placement: &PlacementPlan,
        payload: &mut GenerationPayload,
    ) -> Result<TokenHandle, EngineError> {
        let started_at = std::time::Instant::now();
        let request = BridgeRequest::from_payload(payload, &RequestExtras::current())
            .map_err(EngineError::RequestBuildFailed)?;
        let (key, events) = self.registry.register();
        let admission = Admission::new(
            key,
            Arc::clone(&self.registry),
            Arc::new(PyAborter {
                bridge: Arc::clone(&self.bridge),
            }),
        );
        let stage = admission.stage();
        let bridge = Arc::clone(&self.bridge);
        let request_id = request.request_id.clone();
        let prompt_tokens = request.token_ids.len() as u32;

        let siblings = tokio::task::spawn_blocking(move || submit(&bridge, key, &request, &stage))
            .await
            .map_err(|e| {
                EngineError::Transport(tonic::Status::internal(format!("submit task: {e}")))
            })??;

        debug!(key, %request_id, prompt_tokens, siblings, "Submitted to the token bridge");
        Ok(TokenHandle::new(SeqEventSource::new(
            events,
            admission,
            request_id,
            prompt_tokens,
            siblings,
            started_at,
        )))
    }
}

/// Hand the request to Python; how many sequences it made of it (`n`, or 1
/// where greedy sampling collapsed `n`).
fn submit(
    bridge: &Py<PyAny>,
    key: u64,
    request: &BridgeRequest,
    stage: &Mutex<Stage>,
) -> Result<usize, EngineError> {
    let siblings = Python::attach(|py| {
        request
            .params(py)
            .and_then(|params| {
                bridge
                    .bind(py)
                    .call_method1("submit", (key, request.token_ids.as_slice(), params))?
                    .extract::<usize>()
            })
            .map_err(|e| py_err_to_engine(py, &e))
    })?;
    if submitted(stage) {
        // The client left while Python was admitting it.
        call_abort(bridge, key);
    }
    Ok(siblings)
}

/// A `ValueError` is the request's fault (400); anything else, the server's.
/// Called holding the GIL, on a blocking thread.
fn py_err_to_engine(py: Python<'_>, error: &PyErr) -> EngineError {
    let message = error.value(py).to_string();
    if error.is_instance_of::<PyValueError>(py) {
        EngineError::RequestBuildFailed(message)
    } else {
        EngineError::Transport(tonic::Status::unavailable(message))
    }
}

#[cfg(test)]
mod tests {
    use super::{BridgeRequest, RequestExtras};
    use crate::routers::prepare::generation_payload::{
        GenerationPayload, LogprobConfig, SamplingParams, StopConfig,
    };

    fn payload(presence_penalty: f32) -> GenerationPayload {
        GenerationPayload {
            request_id: "req-1".to_string(),
            text: String::new(),
            token_ids: vec![1, 2, 3],
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: -1,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty,
                repetition_penalty: 1.0,
                max_new_tokens: Some(16),
                min_new_tokens: 0,
                n: 1,
                ignore_eos: false,
            },
            stop: StopConfig {
                stop: None,
                stop_token_ids: None,
                skip_special_tokens: true,
                no_stop_trim: false,
            },
            logprob: LogprobConfig {
                return_logprob: false,
                top_logprobs_num: 0,
                logprob_start_len: -1,
                token_ids_logprob: Vec::new(),
                input_logprobs: false,
            },
            tool_constraints: None,
            pd_metadata: None,
            stream: false,
            return_hidden_states: false,
            log_metrics: false,
        }
    }

    #[test]
    fn a_presence_penalty_is_refused_not_dropped() {
        let extras = RequestExtras::default();
        assert!(BridgeRequest::from_payload(&payload(0.0), &extras).is_ok());
        let err = BridgeRequest::from_payload(&payload(1.5), &extras).err();
        assert!(err.is_some_and(|e| e.contains("presence_penalty")));
    }
}
