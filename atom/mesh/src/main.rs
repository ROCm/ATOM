use clap::Parser;
use mesh::{
    cliargs::{filter_prefill_args_from, parse_prefill_args, Cli, Commands},
    server, version,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    for arg in &args {
        if arg == "--version" || arg == "-V" {
            println!("{}", version::get_version_string());
            return Ok(());
        }
        if arg == "--version-verbose" {
            println!("{}", version::get_verbose_version_string());
            return Ok(());
        }
    }

    let prefill_urls = parse_prefill_args();
    let filtered_args = filter_prefill_args_from(&args);
    let cli = Cli::parse_from(filtered_args);
    let cli_args = match cli.command {
        Some(Commands::Launch { args }) => args,
        None => cli.router_args,
    };

    println!("Atomesh starting...");
    println!("Host: {}:{}", cli_args.host, cli_args.port);
    let mode_str = if cli_args.pd_disaggregation {
        "PD Disaggregated".to_string()
    } else {
        format!("Regular ({})", cli_args.backend)
    };
    println!("Mode: {}", mode_str);
    println!("Policy: {}", cli_args.policy);

    if cli_args.pd_disaggregation && !prefill_urls.is_empty() {
        println!("Prefill nodes: {:?}", prefill_urls);
        println!("Decode nodes: {:?}", cli_args.decode);
    }

    let router_config = cli_args.to_router_config(prefill_urls)?;
    router_config.validate()?;
    let server_config = cli_args.to_server_config(router_config);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move { server::startup(server_config).await })?;
    Ok(())
}
