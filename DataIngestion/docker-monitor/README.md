# Docker Monitoring Tool

A Bun-based tool for monitoring Docker Compose environments, specifically designed to extract relevant information for LLM analysis. The tool focuses on capturing errors, warnings, and other important events from Docker Compose services.

## Features

- **Automated Docker Compose Management**: Starts, monitors, and manages Docker Compose environments
- **Intelligent Log Filtering**: Focuses on capturing errors, warnings, and their context
- **Health Status Monitoring**: Tracks container health status over time
- **System Stats Collection**: Periodically collects resource usage statistics
- **Markdown-formatted Output**: Generates a comprehensive report in a format optimized for LLM consumption
- **Cross-Platform**: Uses Bun's shell execution capabilities to run on any platform

## Requirements

- [Bun](https://bun.sh/) JavaScript runtime
- Docker and Docker Compose
- Any platform supported by Bun (Windows, macOS, Linux)

## Installation

```bash
# Clone the repository (if not already done)
# Navigate to the docker-monitor directory
cd DataIngestion/docker-monitor

# Install dependencies
bun install
```

## Usage

### Using Bun scripts (cross-platform)

```bash
# Basic usage (monitors all services)
bun run monitor

# Monitor specific services
bun run monitor --services=elixir_ingestion,timescaledb

# Specify a custom output file
bun run monitor --output=my-analysis.txt

# Shortcut to monitor Elixir services
bun run monitor:elixir

# Shortcut to monitor Redis services
bun run monitor:redis

# Shortcut to monitor both Elixir and Redis together
bun run monitor:elixir-redis

# Show help
bun run help
```

### Using Convenience Scripts

For Unix-like systems (Linux, macOS):

```bash
# Make the script executable (first time only)
chmod +x ./monitor.sh

# Run the monitor
./monitor.sh
```

For Windows Command Prompt:

```cmd
monitor.bat
```

For Windows PowerShell:

```powershell
.\start-monitor.js
```

### Predefined Scripts

The following scripts are available for convenience:

- `bun run monitor:all` - Monitor all services in the docker-compose.yml
- `bun run monitor:elixir` - Monitor only the Elixir ingestion service
- `bun run monitor:db` - Monitor only the TimescaleDB service

## How It Works

1. The tool starts by bringing down any running Docker Compose services and then starting them fresh
2. It begins monitoring logs in real-time, focusing on lines that contain important keywords like "error", "warning", "failed", etc.
3. When it detects an important event, it captures not just the event itself but also several lines of context before it
4. Every 5 minutes, it collects system statistics and container health information
5. All information is saved to a Markdown-formatted file for easy analysis by an LLM

## Output Format

The tool generates a Markdown file with the following sections:

- **Environment Startup**: Output from starting the Docker Compose environment
- **Service Status**: Status of all services
- **Issues Detected**: Detailed information about errors and warnings, including context
- **System Stats**: Resource usage statistics
- **Container Health**: Health status of all containers
- **Summary**: A concise summary of all issues detected, grouped by service and type

## Cross-Platform Implementation

This tool uses Bun's shell execution feature (`$` function) to run shell commands, making it compatible with all platforms supported by Bun including Windows, macOS, and Linux. The implementation eliminates platform-specific scripts and uses JavaScript/TypeScript for all functionality.

Example of using Bun's shell execution:

```typescript
import { $ } from "bun";

// Run a shell command
const output = await $`docker-compose ps`.text();
console.log(output);

// With arguments
const service = "elixir_ingestion";
await $`docker-compose logs ${service}`;
```

## Integrating with LLMs

The output file is specifically formatted to be easily consumed by Large Language Models. You can feed the generated file directly to an LLM with a prompt like:

```
Analyze the following Docker Compose monitoring report and identify:
1. Key issues and their potential causes
2. Recommendations for fixing the problems
3. Any patterns in the errors that might indicate deeper issues

Here's the report:
[CONTENT OF THE MONITORING FILE]
```

## Customization

You can customize the monitoring behavior by modifying the constants at the top of the `index.ts` file:

- `LOG_LEVELS`: The keywords to look for when identifying important log entries
- `CONTEXT_LINES`: The number of lines of context to capture before an error
- `LINE_BUFFER_SIZE`: The maximum number of recent lines to keep in memory
