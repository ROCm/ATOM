use std::path::PathBuf;

use atomesh_mocker::{
    MockCase, VirtualRequestMode, VirtualRequestPipeline, VirtualRequestPipelineConfig,
    VirtualWorkerPool, VirtualWorkerPoolConfig, VirtualWorkerSpec,
};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(about = "Run Atomesh mocker tools")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the virtual request producer-consumer benchmark process.
    BenchmarkRequest(BenchmarkRequestArgs),
    /// Run virtual backend workers in this process.
    VirtualWorkers(VirtualWorkersArgs),
}

#[derive(Debug, Parser)]
struct BenchmarkRequestArgs {
    /// Target base URL, for example http://127.0.0.1:3000 or grpc://127.0.0.1:30010.
    #[arg(long)]
    base_url: String,

    /// Optional HTTP Host header value. Used only in http mode.
    #[arg(long)]
    host: Option<String>,

    /// PEM CA certificate path to trust for HTTPS benchmark requests.
    #[arg(long, help_heading = "TLS")]
    tls_ca_cert_path: Option<PathBuf>,

    /// Accept invalid TLS certificates for local HTTPS testing.
    #[arg(long, default_value_t = false, help_heading = "TLS")]
    tls_accept_invalid_certs: bool,

    /// Number of producer tasks that build VirtualRequest values.
    #[arg(long, default_value_t = 1)]
    producer_threads: usize,

    /// Number of consumer tasks that POST requests and validate responses.
    #[arg(long, default_value_t = 1)]
    consumer_threads: usize,

    /// Bounded queue capacity between producers and consumers.
    #[arg(long, default_value_t = 4096)]
    queue_capacity: usize,

    /// Fixture file paths to run.
    #[arg(required = true)]
    fixtures: Vec<PathBuf>,
}

#[derive(Debug, Parser)]
struct VirtualWorkersArgs {
    /// IP address or host to bind virtual workers on.
    #[arg(long, default_value = "127.0.0.1")]
    ip: String,

    /// First port used by this process. Workers bind to base_port + worker_index.
    #[arg(long)]
    base_port: u16,

    /// Number of virtual workers to start in this process.
    #[arg(long, default_value_t = 1)]
    workers: usize,

    /// Fixture file paths. Workers cycle through these cases.
    #[arg(required = true)]
    fixtures: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.command {
        Command::BenchmarkRequest(args) => run_benchmark_request(args).await,
        Command::VirtualWorkers(args) => run_virtual_workers(args).await,
    }
}

async fn run_benchmark_request(
    args: BenchmarkRequestArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let cases = args
        .fixtures
        .iter()
        .map(MockCase::from_fixture)
        .collect::<Result<Vec<_>, _>>()?;
    let mode = request_mode_from_base_url(&args.base_url);
    let pipeline = VirtualRequestPipeline::try_new(
        VirtualRequestPipelineConfig::new(args.base_url)
            .mode(mode)
            .host_header(args.host)
            .tls_ca_cert_path(args.tls_ca_cert_path)
            .tls_accept_invalid_certs(args.tls_accept_invalid_certs)
            .producer_threads(args.producer_threads)
            .consumer_threads(args.consumer_threads)
            .queue_capacity(args.queue_capacity),
    )
    .map_err(|error| -> Box<dyn std::error::Error> { error })?;
    let results = pipeline
        .run_cases(cases)
        .await
        .map_err(|error| -> Box<dyn std::error::Error> { error })?;

    for result in results {
        log_case_result(&result);
    }

    Ok(())
}

fn request_mode_from_base_url(base_url: &str) -> VirtualRequestMode {
    if base_url.starts_with("grpc://") {
        VirtualRequestMode::Grpc
    } else {
        VirtualRequestMode::Http
    }
}

async fn run_virtual_workers(args: VirtualWorkersArgs) -> Result<(), Box<dyn std::error::Error>> {
    let cases = args
        .fixtures
        .iter()
        .map(MockCase::from_fixture)
        .collect::<Result<Vec<_>, _>>()?;
    let worker_count = args.workers.max(1);
    let specs = (0..worker_count)
        .map(|index| VirtualWorkerSpec::from_case(cases[index % cases.len()].clone()))
        .collect();

    let mut workers = VirtualWorkerPool::start(
        VirtualWorkerPoolConfig::new(args.ip.clone(), args.base_port),
        specs,
    )
    .await
    .map_err(|error| -> Box<dyn std::error::Error> { error })?;

    println!(
        "virtual_workers_started ip={} base_port={} workers={} urls={:?}",
        args.ip,
        args.base_port,
        worker_count,
        workers.urls()
    );
    tokio::signal::ctrl_c().await?;
    workers.stop().await;
    println!("virtual_workers_stopped");
    Ok(())
}

fn log_case_result(result: &atomesh_mocker::VirtualRequestPipelineResult) {
    println!(
        concat!(
            "endpoint={} status={} ",
            "stream_events={} body={}"
        ),
        result.endpoint.path(),
        result.response.status,
        result.response.stream_events.len(),
        result.response.body
    );
}
