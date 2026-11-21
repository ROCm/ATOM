//! ATOM workers over `atom.grpc.v1`: the client, a payload as a GenerateRequest sent
//! on a Generate call or the Session, its stream as `TokenChunk`s, and ATOM's P/D flow.

pub(crate) mod pair;
pub(crate) mod request;
pub(crate) mod session;
pub(crate) mod stream;

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;
use http::StatusCode;
use tonic::{transport::Channel, Status, Streaming};

use self::session::AtomSessions;
use self::stream::http_error;
use crate::atom_engine::grpc::{
    atom_engine_client::AtomEngineClient, prepare_structured_output_response::Response as Prepared,
    Frames, GenerateRequest, GenerateResponse, GetModelInfoRequest, GetModelInfoResponse,
    GetServerInfoRequest, GetServerInfoResponse, HealthCheckRequest, HealthCheckResponse,
    PrepareStructuredOutputRequest, PrepareStructuredOutputResponse,
};
use crate::core::Worker;
use crate::routers::token_handle::engine_error::EngineError;

/// A request's messages from ATOM, as its Generate call sends them.
pub(crate) type Responses = Pin<Box<dyn Stream<Item = Result<GenerateResponse, Status>> + Send>>;

/// How each request reaches its ATOM instance: a Generate call of its own, or the
/// instance's Session (`--atom-grpc-session`).
#[derive(Clone, Default)]
pub(crate) enum AtomCalls {
    #[default]
    Generate,
    Session(Arc<AtomSessions>),
}

impl AtomCalls {
    /// Starts `request` on `worker`'s instance, which `client` reaches. Dropping the
    /// stream before its last message ends the request at ATOM.
    pub(crate) async fn start(
        &self,
        worker: &Arc<dyn Worker>,
        client: &AtomClient,
        request: GenerateRequest,
    ) -> Result<Responses, EngineError> {
        match self {
            Self::Generate => {
                let call = client.generate(request).await;
                Ok(Box::pin(call.map_err(EngineError::Transport)?))
            }
            Self::Session(sessions) => {
                let request = prepare_structured_output(client, request).await?;
                let stream = request.stream;
                let (spec, prompt) = request::request_spec(request)?;
                let submitted = sessions.submit(worker, spec, prompt, stream).await?;
                Ok(Box::pin(submitted))
            }
        }
    }
}

/// `request` with the fields ATOM's PrepareStructuredOutput derives from its chat's
/// StructuredOutput in its sampling; a request without one, as it is.
async fn prepare_structured_output(
    client: &AtomClient,
    mut request: GenerateRequest,
) -> Result<GenerateRequest, EngineError> {
    let Some(structured_output) = request.structured_output.take() else {
        return Ok(request);
    };
    let asked = PrepareStructuredOutputRequest {
        structured_output: Some(structured_output),
        sampling: request.sampling.clone(),
        stop_token_sequences: request.stop_token_sequences.clone(),
        kv_transfer_params_json: request.kv_transfer_params_json.clone(),
    };
    let answer = client.prepare_structured_output(asked).await;
    let derived = match answer.map_err(EngineError::Transport)?.response {
        Some(Prepared::Sampling(derived)) => derived,
        Some(Prepared::Error(error)) => return Err(http_error(error)),
        None => {
            return Err(EngineError::Http {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: "PrepareStructuredOutput answered neither sampling nor error".into(),
            })
        }
    };
    let sampling = request.sampling.get_or_insert_default();
    sampling.required_channel = derived.required_channel;
    sampling.json_schema = derived.json_schema;
    sampling.structured_tool_format = derived.structured_tool_format;
    sampling.structured_reasoning_end = derived.structured_reasoning_end;
    Ok(request)
}

/// One ATOM instance's `AtomEngine` service.
#[derive(Clone)]
pub struct AtomClient {
    client: AtomEngineClient<Channel>,
}

impl AtomClient {
    /// Connects as the SGLang and vLLM clients do; `grpc://` is read as `http://`.
    pub async fn connect(url: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let endpoint = match url.strip_prefix("grpc://") {
            Some(addr) => format!("http://{addr}"),
            None => url.to_string(),
        };
        let channel = Channel::from_shared(endpoint)?
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10))
            .keep_alive_while_idle(true)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .tcp_nodelay(true)
            .http2_adaptive_window(true)
            .initial_stream_window_size(Some(16 * 1024 * 1024))
            .initial_connection_window_size(Some(32 * 1024 * 1024))
            .connect()
            .await?;
        Ok(Self {
            client: AtomEngineClient::new(channel),
        })
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, Status> {
        let mut client = self.client.clone();
        Ok(client
            .health_check(HealthCheckRequest {})
            .await?
            .into_inner())
    }

    pub async fn get_model_info(&self) -> Result<GetModelInfoResponse, Status> {
        let mut client = self.client.clone();
        Ok(client
            .get_model_info(GetModelInfoRequest {})
            .await?
            .into_inner())
    }

    pub async fn get_server_info(&self) -> Result<GetServerInfoResponse, Status> {
        let mut client = self.client.clone();
        Ok(client
            .get_server_info(GetServerInfoRequest {})
            .await?
            .into_inner())
    }

    /// Dropping the returned stream before its last message cancels the call,
    /// which ends the request at ATOM.
    pub async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<Streaming<GenerateResponse>, Status> {
        let mut client = self.client.clone();
        Ok(client.generate(request).await?.into_inner())
    }

    /// The Session call: ATOM reads `frames` as they come and answers on the
    /// returned stream.
    pub async fn session(
        &self,
        frames: impl Stream<Item = Frames> + Send + 'static,
    ) -> Result<Streaming<Frames>, Status> {
        let mut client = self.client.clone();
        Ok(client.session(frames).await?.into_inner())
    }

    pub async fn prepare_structured_output(
        &self,
        request: PrepareStructuredOutputRequest,
    ) -> Result<PrepareStructuredOutputResponse, Status> {
        let mut client = self.client.clone();
        Ok(client
            .prepare_structured_output(request)
            .await?
            .into_inner())
    }
}

#[cfg(test)]
mod tests;
