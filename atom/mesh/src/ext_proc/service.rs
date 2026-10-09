use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use tokio::sync::{mpsc, watch, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::app_context::AppContext;

use super::{
    admission::Admission, executor::PdExecutor, pb, request::RequestParser, session::Session,
};

pub(super) struct ExtProcService {
    app: Arc<AppContext>,
    admission: Arc<Admission>,
    streams: Arc<Semaphore>,
    parser: Arc<RequestParser>,
    draining: Arc<AtomicBool>,
    force_stop: watch::Receiver<bool>,
    executor: Option<Arc<PdExecutor>>,
}

impl ExtProcService {
    pub fn new(
        app: Arc<AppContext>,
        draining: Arc<AtomicBool>,
        force_stop: watch::Receiver<bool>,
        executor: Option<Arc<PdExecutor>>,
    ) -> Self {
        Self {
            admission: Arc::new(Admission::new(&app)),
            streams: Arc::new(Semaphore::new(app.router_config.ext_proc.max_streams)),
            parser: Arc::new(RequestParser::new(app.clone())),
            app,
            draining,
            force_stop,
            executor,
        }
    }
}

#[tonic::async_trait]
impl pb::external_processor_server::ExternalProcessor for ExtProcService {
    type ProcessStream = ReceiverStream<Result<pb::ProcessingResponse, Status>>;

    async fn process(
        &self,
        request: Request<tonic::Streaming<pb::ProcessingRequest>>,
    ) -> Result<Response<Self::ProcessStream>, Status> {
        if self.draining.load(Ordering::Acquire) {
            return Err(Status::unavailable("Mesh is draining"));
        }
        let permit = self
            .streams
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("ext-proc stream limit exceeded"))?;
        let (tx, rx) = mpsc::channel(1);
        let session = Session::new(
            self.app.clone(),
            self.admission.clone(),
            tx,
            self.executor.clone(),
            self.parser.clone(),
        );
        let force_stop = self.force_stop.clone();
        tokio::spawn(async move {
            let _permit = permit;
            session.run(request.into_inner(), force_stop).await;
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
