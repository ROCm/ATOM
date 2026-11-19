//! gRPC client for the ATOM token engine.
//!
//! The wire format matches `proto/atom/inference/v1/generate.proto`. The codec
//! is hand-written so the default build does not run prost codegen.

use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::{Buf, BufMut};
use dashmap::DashMap;
use futures::Stream;
use http::uri::PathAndQuery;
use tonic::client::Grpc;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status, Streaming};

use super::types::{TokenGenerateEvent, TokenGenerateRequest, TokenUsage};
use super::TokenPathError;
use crate::core::Worker;

pub type EngineStream =
    Pin<Box<dyn Stream<Item = Result<TokenGenerateEvent, TokenPathError>> + Send>>;

const GENERATE: &str = "/atom.inference.v1.AtomInference/Generate";
const ABORT: &str = "/atom.inference.v1.AtomInference/AbortRequests";

#[async_trait]
pub trait AtomEngine: Send + Sync {
    async fn generate(
        &self,
        worker: &dyn Worker,
        request: TokenGenerateRequest,
    ) -> Result<EngineStream, TokenPathError>;

    async fn abort(
        &self,
        worker: &dyn Worker,
        request_ids: &[String],
    ) -> Result<(), TokenPathError>;
}

/// Endpoint of the engine process. DP rank travels in the request, not the URL.
pub fn grpc_endpoint(worker: &dyn Worker) -> String {
    let raw = worker.base_url().trim_end_matches('/');
    let without_rank = raw.split('@').next().unwrap_or(raw);
    if let Some(rest) = without_rank.strip_prefix("grpc://") {
        format!("http://{rest}")
    } else if let Some(rest) = without_rank.strip_prefix("grpcs://") {
        format!("https://{rest}")
    } else if without_rank.starts_with("http://") || without_rank.starts_with("https://") {
        without_rank.to_string()
    } else {
        format!("http://{without_rank}")
    }
}

#[derive(Clone)]
struct WireRequest {
    request_id: String,
    model: String,
    token_ids: Vec<u32>,
    sampling_params_json: String,
    stream: bool,
    kv_transfer_params_json: String,
    data_parallel_rank: Option<u32>,
    session_id: String,
    max_tokens: Option<u32>,
}

#[derive(Clone, Default)]
struct WireEvent {
    token_ids: Vec<u32>,
    finish_reason: String,
    kv_transfer_params_json: String,
    error_json: String,
    usage: Option<WireUsage>,
}

#[derive(Clone, Default)]
struct WireUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
}

#[derive(Clone, Default)]
struct WireAbort {
    request_ids: Vec<String>,
}

fn to_wire(request: &TokenGenerateRequest) -> WireRequest {
    WireRequest {
        request_id: request.request_id.clone(),
        model: request.model.clone(),
        token_ids: request.token_ids.clone(),
        sampling_params_json: request.sampling_params.to_string(),
        stream: request.stream,
        kv_transfer_params_json: request
            .kv_transfer_params
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
        data_parallel_rank: request.data_parallel_rank,
        session_id: request.session_id.clone().unwrap_or_default(),
        max_tokens: request.max_tokens,
    }
}

fn event_from_wire(event: WireEvent) -> Result<TokenGenerateEvent, TokenPathError> {
    let kv_transfer_params = if event.kv_transfer_params_json.is_empty() {
        None
    } else {
        Some(
            serde_json::from_str(&event.kv_transfer_params_json).map_err(|error| {
                TokenPathError::Engine {
                    role: "engine",
                    message: format!("kv_transfer_params is not JSON: {error}"),
                }
            })?,
        )
    };
    let error = if event.error_json.is_empty() {
        None
    } else {
        Some(
            serde_json::from_str(&event.error_json)
                .unwrap_or_else(|_| serde_json::json!({"message": event.error_json})),
        )
    };
    Ok(TokenGenerateEvent {
        token_ids: event.token_ids,
        finish_reason: if event.finish_reason.is_empty() {
            None
        } else {
            Some(event.finish_reason)
        },
        kv_transfer_params,
        usage: event.usage.map(|usage| TokenUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cached_tokens: usage.cached_tokens,
        }),
        error,
    })
}

#[derive(Default)]
pub struct GrpcAtomEngine {
    channels: DashMap<String, Channel>,
}

impl GrpcAtomEngine {
    pub fn new() -> Self {
        Self::default()
    }

    async fn channel(&self, endpoint: &str) -> Result<Channel, TokenPathError> {
        if let Some(channel) = self.channels.get(endpoint) {
            return Ok(channel.clone());
        }
        let channel = Endpoint::from_shared(endpoint.to_string())
            .map_err(|error| TokenPathError::Engine {
                role: "engine",
                message: format!("bad endpoint {endpoint}: {error}"),
            })?
            .connect()
            .await
            .map_err(|error| TokenPathError::Engine {
                role: "engine",
                message: format!("connect {endpoint}: {error}"),
            })?;
        self.channels.insert(endpoint.to_string(), channel.clone());
        Ok(channel)
    }
}

#[async_trait]
impl AtomEngine for GrpcAtomEngine {
    async fn generate(
        &self,
        worker: &dyn Worker,
        request: TokenGenerateRequest,
    ) -> Result<EngineStream, TokenPathError> {
        let endpoint = grpc_endpoint(worker);
        let channel = self.channel(&endpoint).await?;
        let mut grpc = Grpc::new(channel);
        grpc.ready().await.map_err(|error| TokenPathError::Engine {
            role: "engine",
            message: format!("channel not ready: {error}"),
        })?;
        let path = PathAndQuery::from_static(GENERATE);
        let response = grpc
            .server_streaming(
                Request::new(to_wire(&request)),
                path,
                ProtoCodec::<WireRequest, WireEvent>::default(),
            )
            .await
            .map_err(|status| TokenPathError::Engine {
                role: "engine",
                message: status.to_string(),
            })?;
        Ok(Box::pin(ProtoEventStream {
            inner: response.into_inner(),
            failed: false,
        }))
    }

    async fn abort(
        &self,
        worker: &dyn Worker,
        request_ids: &[String],
    ) -> Result<(), TokenPathError> {
        let endpoint = grpc_endpoint(worker);
        let channel = self.channel(&endpoint).await?;
        let mut grpc = Grpc::new(channel);
        grpc.ready().await.map_err(|error| TokenPathError::Engine {
            role: "engine",
            message: format!("channel not ready: {error}"),
        })?;
        let path = PathAndQuery::from_static(ABORT);
        grpc.unary(
            Request::new(WireAbort {
                request_ids: request_ids.to_vec(),
            }),
            path,
            ProtoCodec::<WireAbort, WireAbort>::default(),
        )
        .await
        .map_err(|status| TokenPathError::Engine {
            role: "engine",
            message: status.to_string(),
        })?;
        Ok(())
    }
}

struct ProtoEventStream {
    inner: Streaming<WireEvent>,
    failed: bool,
}

impl Stream for ProtoEventStream {
    type Item = Result<TokenGenerateEvent, TokenPathError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.failed {
            return Poll::Ready(None);
        }
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => Poll::Ready(Some(event_from_wire(event))),
            Poll::Ready(Some(Err(status))) => {
                this.failed = true;
                Poll::Ready(Some(Err(TokenPathError::Engine {
                    role: "engine",
                    message: status.to_string(),
                })))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

struct ProtoCodec<E, D>(PhantomData<(E, D)>);

impl<E, D> Default for ProtoCodec<E, D> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<E, D> Codec for ProtoCodec<E, D>
where
    E: Wire + Send + 'static,
    D: Wire + Send + 'static,
{
    type Encode = E;
    type Decode = D;
    type Encoder = WireEncoder<E>;
    type Decoder = WireDecoder<D>;

    fn encoder(&mut self) -> Self::Encoder {
        WireEncoder(PhantomData)
    }

    fn decoder(&mut self) -> Self::Decoder {
        WireDecoder(PhantomData)
    }
}

struct WireEncoder<T>(PhantomData<T>);
struct WireDecoder<T>(PhantomData<T>);

impl<T: Wire> Encoder for WireEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, dst: &mut EncodeBuf<'_>) -> Result<(), Self::Error> {
        item.write(dst);
        Ok(())
    }
}

impl<T: Wire> Decoder for WireDecoder<T> {
    type Item = T;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Self::Error> {
        let mut bytes = vec![0u8; src.remaining()];
        src.copy_to_slice(&mut bytes);
        T::read(&bytes)
            .map(Some)
            .map_err(|error| Status::internal(error))
    }
}

trait Wire: Sized {
    fn encode_bytes(&self) -> Vec<u8>;
    fn read(bytes: &[u8]) -> Result<Self, String>;

    fn write(&self, dst: &mut EncodeBuf<'_>) {
        dst.put_slice(&self.encode_bytes());
    }
}

impl Wire for WireRequest {
    fn encode_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_string(&mut buf, 1, &self.request_id);
        write_string(&mut buf, 2, &self.model);
        write_packed_u32(&mut buf, 3, &self.token_ids);
        write_string(&mut buf, 4, &self.sampling_params_json);
        if self.stream {
            write_varint_field(&mut buf, 5, 1);
        }
        write_string(&mut buf, 6, &self.kv_transfer_params_json);
        if let Some(rank) = self.data_parallel_rank {
            write_varint_field(&mut buf, 7, rank as u64);
        }
        write_string(&mut buf, 8, &self.session_id);
        if let Some(max_tokens) = self.max_tokens {
            write_varint_field(&mut buf, 9, max_tokens as u64);
        }
        buf
    }

    fn read(bytes: &[u8]) -> Result<Self, String> {
        let mut out = WireRequest {
            request_id: String::new(),
            model: String::new(),
            token_ids: Vec::new(),
            sampling_params_json: String::new(),
            stream: false,
            kv_transfer_params_json: String::new(),
            data_parallel_rank: None,
            session_id: String::new(),
            max_tokens: None,
        };
        for field in fields(bytes)? {
            match field {
                Field::Bytes(1, bytes) => out.request_id = utf8(bytes)?,
                Field::Bytes(2, bytes) => out.model = utf8(bytes)?,
                Field::Bytes(3, bytes) => out.token_ids.extend(unpack_u32(bytes)?),
                Field::Varint(3, value) => out.token_ids.push(value as u32),
                Field::Bytes(4, bytes) => out.sampling_params_json = utf8(bytes)?,
                Field::Varint(5, value) => out.stream = value != 0,
                Field::Bytes(6, bytes) => out.kv_transfer_params_json = utf8(bytes)?,
                Field::Varint(7, value) => out.data_parallel_rank = Some(value as u32),
                Field::Bytes(8, bytes) => out.session_id = utf8(bytes)?,
                Field::Varint(9, value) => out.max_tokens = Some(value as u32),
                _ => {}
            }
        }
        Ok(out)
    }
}

impl Wire for WireEvent {
    fn encode_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_packed_u32(&mut buf, 1, &self.token_ids);
        write_string(&mut buf, 2, &self.finish_reason);
        write_string(&mut buf, 3, &self.kv_transfer_params_json);
        write_string(&mut buf, 4, &self.error_json);
        if let Some(usage) = &self.usage {
            let mut nested = Vec::new();
            if usage.prompt_tokens != 0 {
                write_varint_field(&mut nested, 1, usage.prompt_tokens as u64);
            }
            if usage.completion_tokens != 0 {
                write_varint_field(&mut nested, 2, usage.completion_tokens as u64);
            }
            if usage.cached_tokens != 0 {
                write_varint_field(&mut nested, 3, usage.cached_tokens as u64);
            }
            write_bytes(&mut buf, 5, &nested);
        }
        buf
    }

    fn read(bytes: &[u8]) -> Result<Self, String> {
        let mut out = WireEvent::default();
        for field in fields(bytes)? {
            match field {
                Field::Bytes(1, bytes) => out.token_ids.extend(unpack_u32(bytes)?),
                Field::Varint(1, value) => out.token_ids.push(value as u32),
                Field::Bytes(2, bytes) => out.finish_reason = utf8(bytes)?,
                Field::Bytes(3, bytes) => out.kv_transfer_params_json = utf8(bytes)?,
                Field::Bytes(4, bytes) => out.error_json = utf8(bytes)?,
                Field::Bytes(5, bytes) => {
                    let mut usage = WireUsage::default();
                    for nested in fields(bytes)? {
                        match nested {
                            Field::Varint(1, value) => usage.prompt_tokens = value as u32,
                            Field::Varint(2, value) => usage.completion_tokens = value as u32,
                            Field::Varint(3, value) => usage.cached_tokens = value as u32,
                            _ => {}
                        }
                    }
                    out.usage = Some(usage);
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

impl Wire for WireAbort {
    fn encode_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for id in &self.request_ids {
            write_string(&mut buf, 1, id);
        }
        buf
    }

    fn read(bytes: &[u8]) -> Result<Self, String> {
        let mut request_ids = Vec::new();
        for field in fields(bytes)? {
            if let Field::Bytes(1, bytes) = field {
                request_ids.push(utf8(bytes)?);
            }
        }
        Ok(WireAbort { request_ids })
    }
}

enum Field<'a> {
    Varint(u32, u64),
    Bytes(u32, &'a [u8]),
}

fn fields(bytes: &[u8]) -> Result<Vec<Field<'_>>, String> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let key = read_varint(bytes, &mut index)?;
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u8;
        match wire {
            0 => out.push(Field::Varint(field, read_varint(bytes, &mut index)?)),
            2 => {
                let len = read_varint(bytes, &mut index)? as usize;
                let end = index.checked_add(len).ok_or("length overflow")?;
                if end > bytes.len() {
                    return Err("truncated bytes field".into());
                }
                out.push(Field::Bytes(field, &bytes[index..end]));
                index = end;
            }
            1 | 5 => {
                let width = if wire == 1 { 8 } else { 4 };
                let end = index.checked_add(width).ok_or("length overflow")?;
                if end > bytes.len() {
                    return Err("truncated fixed field".into());
                }
                index = end;
            }
            _ => return Err(format!("unsupported wire type {wire}")),
        }
    }
    Ok(out)
}

fn read_varint(bytes: &[u8], index: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        if *index >= bytes.len() {
            return Err("truncated varint".into());
        }
        let byte = bytes[*index];
        *index += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            return Err("varint too long".into());
        }
    }
}

fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn write_key(buf: &mut Vec<u8>, field: u32, wire: u8) {
    write_varint(buf, ((field as u64) << 3) | wire as u64);
}

fn write_varint_field(buf: &mut Vec<u8>, field: u32, value: u64) {
    write_key(buf, field, 0);
    write_varint(buf, value);
}

fn write_bytes(buf: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    write_key(buf, field, 2);
    write_varint(buf, bytes.len() as u64);
    buf.extend_from_slice(bytes);
}

fn write_string(buf: &mut Vec<u8>, field: u32, value: &str) {
    if !value.is_empty() {
        write_bytes(buf, field, value.as_bytes());
    }
}

fn write_packed_u32(buf: &mut Vec<u8>, field: u32, values: &[u32]) {
    if values.is_empty() {
        return;
    }
    let mut packed = Vec::new();
    for value in values {
        write_varint(&mut packed, *value as u64);
    }
    write_bytes(buf, field, &packed);
}

fn unpack_u32(bytes: &[u8]) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        out.push(read_varint(bytes, &mut index)? as u32);
    }
    Ok(out)
}

fn utf8(bytes: &[u8]) -> Result<String, String> {
    String::from_utf8(bytes.to_vec()).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_roundtrip_keeps_token_ids_and_rank() {
        let request = WireRequest {
            request_id: "req-1".into(),
            model: "m".into(),
            token_ids: vec![1, 128, 300],
            sampling_params_json: "{\"temperature\":0}".into(),
            stream: true,
            kv_transfer_params_json: "{\"do_remote_decode\":true}".into(),
            data_parallel_rank: Some(3),
            session_id: "sess".into(),
            max_tokens: Some(1),
        };
        let decoded = WireRequest::read(&request.encode_bytes()).unwrap();
        assert_eq!(decoded.token_ids, vec![1, 128, 300]);
        assert_eq!(decoded.data_parallel_rank, Some(3));
        assert!(decoded.stream);
        assert_eq!(decoded.max_tokens, Some(1));
        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.model, "m");
        assert_eq!(decoded.session_id, "sess");
    }

    #[test]
    fn unpacked_repeated_token_ids_are_kept() {
        let mut raw = Vec::new();
        write_varint_field(&mut raw, 3, 7);
        write_varint_field(&mut raw, 3, 8);
        let decoded = WireRequest::read(&raw).unwrap();
        assert_eq!(decoded.token_ids, vec![7, 8]);

        let mut event = Vec::new();
        write_varint_field(&mut event, 1, 11);
        write_varint_field(&mut event, 1, 12);
        let decoded = WireEvent::read(&event).unwrap();
        assert_eq!(decoded.token_ids, vec![11, 12]);
    }
}
