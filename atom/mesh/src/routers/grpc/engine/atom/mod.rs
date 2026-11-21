//! ATOM workers over `atom.grpc.v1`: the client, a payload as a GenerateRequest,
//! and a Generate stream as `TokenChunk`s.

pub(crate) mod request;
pub(crate) mod stream;

use std::time::Duration;

use tonic::{transport::Channel, Status, Streaming};

use crate::atom_engine::grpc::{
    atom_engine_client::AtomEngineClient, GenerateRequest, GenerateResponse, GetModelInfoRequest,
    GetModelInfoResponse, GetServerInfoRequest, GetServerInfoResponse, HealthCheckRequest,
    HealthCheckResponse,
};

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
}

#[cfg(test)]
mod tests;
