use anyhow::Result;
use clap::Parser as _;

/// `Pipeline::from_config` spawns a config watcher and needs a live Tokio
/// handle, so every startup step has to happen inside the runtime.
#[tokio::main]
async fn main() -> Result<()> {
    birelate_server::run(birelate_server::Cli::parse()).await
}
