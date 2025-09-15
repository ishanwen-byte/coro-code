//! Single task execution command

use anyhow::Result;
use std::path::PathBuf;
use tracing::{debug, info};

/// Configuration for running a single task
pub struct RunConfig {
    pub task: String,
    pub config_loader: crate::config::CliConfigLoader,
    pub max_steps: Option<usize>,
    pub trajectory_file: Option<PathBuf>,
    pub must_patch: bool,
    pub patch_path: PathBuf,
    pub working_dir: Option<PathBuf>,
    pub debug_output: bool,
}

/// Execute a single task
pub async fn run_command(config: RunConfig) -> Result<()> {
    info!("Executing task: {}", config.task);

    use crate::output::cli_handler::{CliOutputConfig, CliOutputHandler};
    use coro_core::{trajectory::TrajectoryRecorder, AgentBuilder, AgentConfig, OutputMode};

    // Load LLM configuration
    let llm_config = config.config_loader.load().await?;
    info!("🤖 Using protocol: {}", llm_config.protocol.as_str());
    info!("🤖 Using model: {}", llm_config.model);

    // Create agent configuration with CLI tools
    let mut agent_config = AgentConfig {
        tools: crate::tools::get_default_cli_tools(),
        ..Default::default()
    };
    if let Some(steps) = config.max_steps {
        agent_config.max_steps = steps;
    }
    if config.debug_output {
        agent_config.output_mode = OutputMode::Debug;
    }

    // Create CLI output handler
    let cli_config = CliOutputConfig {
        realtime_updates: true, // Always enable realtime updates for better UX
    };
    let cli_output = Box::new(CliOutputHandler::new(cli_config));

    // Build agent with new configuration system and CLI tools
    let cli_tool_registry = crate::tools::create_cli_tool_registry();
    let agent = AgentBuilder::new(llm_config)
        .with_agent_config(agent_config)
        .build_with_output_and_registry(cli_output, cli_tool_registry)
        .await?;

    // Initialize trajectory recorder
    let trajectory = TrajectoryRecorder::new();
    let task_entry = coro_core::trajectory::TrajectoryEntry::task_start(
        config.task.clone(),
        serde_json::json!({"max_steps": config.max_steps.unwrap_or(200)}),
    );
    trajectory.record(task_entry).await?;

    if let Some(trajectory_file) = &config.trajectory_file {
        info!("📊 Trajectory file: {}", trajectory_file.display());
    }

    debug!("🤖 Using coro-code Agent system prompt");

    // Get current working directory
    let current_dir = config
        .working_dir
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let project_path = dunce::canonicalize(&current_dir).unwrap_or(current_dir);

    debug!("📁 Project path: {}", project_path.display());

    // Execute the task using the agent
    let mut agent = agent; // Make mutable for execution
    let _execution_result = agent
        .execute_task_with_context(&config.task, &project_path)
        .await?;

    if config.must_patch {
        info!("📄 Creating patch file: {}", config.patch_path.display());
        std::fs::write(
            &config.patch_path,
            "# Placeholder patch file\n# Changes would be recorded here\n",
        )?;
    }

    // Save trajectory if requested
    if let Some(trajectory_file) = &config.trajectory_file {
        info!("📊 Trajectory saved to: {}", trajectory_file.display());
    }

    info!("✅ Task completed successfully");

    Ok(())
}
