//! Interactive mode command

use crate::interactive::app::run_rich_interactive;
use anyhow::Result;
use std::path::PathBuf;
use tracing::debug;

/// Start interactive mode
pub async fn interactive_command(
    config_loader: crate::config::CliConfigLoader,
    trajectory_file: Option<PathBuf>,
    debug_output: bool,
) -> Result<()> {
    if debug_output {
        debug!("Debug output enabled");
    }

    if let Some(trajectory_file) = &trajectory_file {
        debug!("Trajectory file: {}", trajectory_file.display());
    }

    // Load LLM configuration
    let llm_config = config_loader.load().await?;
    if debug_output {
        debug!("Using protocol: {}", llm_config.protocol.as_str());
        debug!("Using model: {}", llm_config.model);
    }

    // Get current working directory
    let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project_path = dunce::canonicalize(&current_dir).unwrap_or(current_dir);

    if debug_output {
        debug!("Project path: {}", project_path.display());
    }

    // Run the interactive mode (always use rich mode)
    run_rich_interactive(llm_config, project_path, debug_output).await
}
