//! Cross-platform shell execution tool

use async_trait::async_trait;
use coro_core::error::Result;
use coro_core::impl_tool_factory;
use coro_core::tools::utils::maybe_truncate;
use coro_core::tools::{Tool, ToolCall, ToolExample, ToolResult};
use serde_json::json;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout, Duration};

/// Shell configuration for different operating systems
#[derive(Debug, Clone)]
struct ShellConfig {
    command: String,
    args: Vec<String>,
    sentinel: String,
    is_windows: bool,
}

impl ShellConfig {
    fn new() -> Self {
        if cfg!(target_os = "windows") {
            Self {
                command: "pwsh".to_string(),
                args: vec![
                    "-NoProfile".to_string(),
                    "-NoLogo".to_string(),
                    "-ExecutionPolicy".to_string(),
                    "Bypass".to_string(),
                    "-Command".to_string(),
                    "-".to_string(),
                ],
                sentinel: ",,,,shell-command-exit-__ERROR_CODE__-banner,,,,".to_string(),
                is_windows: true,
            }
        } else {
            Self {
                command: "/bin/bash".to_string(),
                args: vec![],
                sentinel: ",,,,shell-command-exit-__ERROR_CODE__-banner,,,,".to_string(),
                is_windows: false,
            }
        }
    }
}

/// A session of a cross-platform shell
struct ShellSession {
    process: Option<Child>,
    started: bool,
    timed_out: bool,
    config: ShellConfig,
    timeout: Duration,
}

impl ShellSession {
    fn new() -> Self {
        Self {
            process: None,
            started: false,
            timed_out: false,
            config: ShellConfig::new(),
            timeout: Duration::from_secs(120),
        }
    }

    async fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }

        let mut cmd = Command::new(&self.config.command);

        if !self.config.args.is_empty() {
            cmd.args(&self.config.args);
        }

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(unix)]
        {
            #[allow(unused_imports)]
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        self.process = Some(cmd.spawn()?);
        self.started = true;
        Ok(())
    }

    fn stop(&mut self) {
        if !self.started {
            return;
        }

        if let Some(mut process) = self.process.take() {
            if process.try_wait().unwrap_or(None).is_none() {
                let _ = process.kill();
            }
        }
        self.started = false;
    }

    async fn run(&mut self, command: &str) -> Result<(i32, String, String)> {
        if !self.started || self.process.is_none() {
            return Err("Session has not started.".into());
        }

        if self.timed_out {
            return Err(format!(
                "timed out: shell has not returned in {} seconds and must be restarted",
                self.timeout.as_secs()
            )
            .into());
        }

        let process = self.process.as_mut().unwrap();

        if let Ok(Some(status)) = process.try_wait() {
            return Err(format!(
                "shell has exited with returncode {}. tool must be restarted.",
                status.code().unwrap_or(-1)
            )
            .into());
        }

        let (sentinel_before, sentinel_after) = self
            .config
            .sentinel
            .split_once("__ERROR_CODE__")
            .ok_or("Invalid sentinel format")?;

        let full_command = if self.config.is_windows {
            // PowerShell syntax
            let errcode_retriever = "$LASTEXITCODE";
            format!(
                "try {{ {} }} catch {{ Write-Error $_ }}; echo '{}'\r\n",
                command,
                self.config
                    .sentinel
                    .replace("__ERROR_CODE__", errcode_retriever)
            )
        } else {
            // Bash syntax
            let errcode_retriever = "$?";
            format!(
                "({}); echo '{}'\n",
                command,
                self.config
                    .sentinel
                    .replace("__ERROR_CODE__", errcode_retriever)
            )
        };

        if let Some(stdin) = process.stdin.as_mut() {
            stdin.write_all(full_command.as_bytes()).await?;
            stdin.flush().await?;
        } else {
            return Err("No stdin available".into());
        }

        let result = timeout(self.timeout, async {
            let mut output = String::new();
            let mut error_code = 0;

            if let Some(stdout) = process.stdout.as_mut() {
                let mut reader = BufReader::new(stdout);
                let mut buffer = Vec::new();

                loop {
                    match reader.read_until(b'\n', &mut buffer).await {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            let line = String::from_utf8_lossy(&buffer);
                            output.push_str(&line);
                            buffer.clear();

                            if let Some(pos) = output.rfind(sentinel_before) {
                                let content = output[..pos].to_string();
                                let rest = &output[pos..];

                                if let Some(code_start) = rest.find(sentinel_before) {
                                    let code_part = &rest[code_start + sentinel_before.len()..];
                                    if let Some(code_end) = code_part.find(sentinel_after) {
                                        let code_str = &code_part[..code_end];
                                        error_code = code_str.trim().parse().unwrap_or(-1);
                                    }
                                }

                                output = content;
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            }

            Ok::<(i32, String, String), coro_core::error::Error>((
                error_code,
                output,
                String::new(),
            ))
        })
        .await;

        match result {
            Ok(Ok((exit_code, stdout, stderr))) => {
                let stdout_clean = stdout.trim_end_matches('\n').to_string();
                Ok((exit_code, stdout_clean, stderr))
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                self.timed_out = true;
                Err(format!(
                    "timed out: shell has not returned in {} seconds and must be restarted",
                    self.timeout.as_secs()
                )
                .into())
            }
        }
    }
}

/// Tool for executing shell commands with session management
pub struct ShellTool {
    session: Arc<Mutex<Option<ShellSession>>>,
}

impl ShellTool {
    /// Create a new shell tool
    pub fn new() -> Self {
        Self {
            session: Arc::new(Mutex::new(None)),
        }
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        if cfg!(target_os = "windows") {
            "Run commands in Windows PowerShell\n* State is persistent across command calls."
        } else {
            "Run commands in a bash shell\n* State is persistent across command calls."
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let command_description = if cfg!(target_os = "windows") {
            "The Windows PowerShell command to run."
        } else {
            "The bash command to run."
        };

        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": command_description
                },
                "restart": {
                    "type": "boolean",
                    "description": "Set to true to restart the shell session."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        let restart: bool = call.get_parameter_or("restart", false);

        if restart {
            let mut session_guard = self.session.lock().await;
            if let Some(ref mut session) = *session_guard {
                session.stop();
            }
            *session_guard = Some(ShellSession::new());
            if let Some(ref mut session) = *session_guard {
                session.start().await?;
            }

            return Ok(ToolResult::success(
                &call.id,
                &"tool has been restarted.".to_string(),
            ));
        }

        let command: String = call.get_parameter("command")?;

        let needs_start = {
            let mut session_guard = self.session.lock().await;
            if session_guard.is_none() {
                *session_guard = Some(ShellSession::new());
                true
            } else {
                !session_guard.as_ref().unwrap().started
            }
        };

        if needs_start {
            let mut session_guard = self.session.lock().await;
            if let Some(ref mut session) = *session_guard {
                session.start().await?;
            }
        }

        let result = {
            let mut session_guard = self.session.lock().await;
            session_guard.as_mut().unwrap().run(&command).await
        };

        match result {
            Ok((exit_code, stdout, stderr)) => {
                let mut output = String::new();

                if !stdout.is_empty() {
                    output.push_str(&maybe_truncate(&stdout, None));
                }

                if !stderr.is_empty() {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&maybe_truncate(&stderr, None));
                }

                if output.is_empty() {
                    output = format!("Command completed with exit code: {}", exit_code);
                }

                Ok(ToolResult::success(&call.id, &output).with_data(json!({
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr
                })))
            }
            Err(e) => Ok(ToolResult::error(
                &call.id,
                &format!("Error running shell command: {}", e),
            )),
        }
    }

    fn requires_confirmation(&self) -> bool {
        true
    }

    fn examples(&self) -> Vec<ToolExample> {
        if cfg!(target_os = "windows") {
            vec![
                ToolExample {
                    description: "List files in current directory".to_string(),
                    parameters: json!({"command": "Get-ChildItem"}),
                    expected_result: "Directory listing".to_string(),
                },
                ToolExample {
                    description: "Recursively find all .rs files".to_string(),
                    parameters: json!({"command": "Get-ChildItem -Recurse -Filter *.rs"}),
                    expected_result: "A list of all Rust files in the project".to_string(),
                },
            ]
        } else {
            vec![
                ToolExample {
                    description: "List files in current directory".to_string(),
                    parameters: json!({"command": "ls -la"}),
                    expected_result: "Directory listing".to_string(),
                },
            ]
        }
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new()
    }
}

impl_tool_factory!(
    ShellToolFactory,
    ShellTool,
    "shell",
    if cfg!(target_os = "windows") {
        "Execute Windows commands using PowerShell"
    } else {
        "Execute bash commands on Unix-like systems"
    }
);
