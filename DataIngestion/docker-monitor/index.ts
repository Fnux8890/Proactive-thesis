#!/usr/bin/env bun

import { spawn } from "bun";
import fs from "fs";
import path from "path";
import { parseArgs } from "util";

// Define the log levels to capture
const LOG_LEVELS = ["error", "warn", "warning", "critical", "exception", "fail", "failed", "failure"];
const CONTEXT_LINES = 5; // Number of lines of context to capture before an error

// Redis-specific monitoring
const REDIS_METRICS = ["memory", "clients", "stats", "keyspace"];
let redisMetrics: Record<string, any> = {};

// Parse command line arguments
const args = parseArgs({
    options: {
        output: {
            type: "string",
            short: "o",
            default: "docker-insights.txt"
        },
        dir: {
            type: "string",
            short: "d",
            default: ".."
        },
        services: {
            type: "string",
            short: "s",
            default: ""
        },
        help: {
            type: "boolean",
            short: "h",
            default: false
        }
    }
}).values;

// Show help information
if (args.help) {
    console.log(`
  Docker Compose Monitor
  
  Usage:
    bun run index.ts [options]
  
  Options:
    -o, --output    Output file for captured logs (default: docker-insights.txt)
    -d, --dir       Directory containing docker-compose.yml (default: parent directory)
    -s, --services  Specific services to monitor (comma-separated, default: all)
    -h, --help      Show this help information
  `);
    process.exit(0);
}

const outputFile = args.output as string;
const dockerComposeDir = path.resolve(args.dir as string);
const specificServices = args.services ? (args.services as string).split(",") : [];

console.log(`Starting Docker Compose monitoring in ${dockerComposeDir}`);
console.log(`Results will be saved to ${outputFile}`);
if (specificServices.length > 0) {
    console.log(`Monitoring only these services: ${specificServices.join(", ")}`);
}

const LINE_BUFFER_SIZE = 1000;
let recentLines: string[] = [];
let issues: { service: string, level: string, message: string, context: string[] }[] = [];

// Initialize output file
fs.writeFileSync(outputFile, `# Docker Compose Monitoring Results\n\nGenerated: ${new Date().toISOString()}\n\n`);

// Start Docker Compose
async function startDockerCompose() {
    console.log("Starting Docker Compose...");

    fs.appendFileSync(outputFile, "## Environment Startup\n\n");

    // Check if Docker Compose is running and stop it if it is
    const stopProcess = spawn({
        cmd: ["docker", "compose", "-f", path.join(dockerComposeDir, "docker-compose.yml"), "down"],
        cwd: dockerComposeDir,
        stdout: "pipe",
        stderr: "pipe",
    });

    await stopProcess.exited;

    // Start Docker Compose
    const startProcess = spawn({
        cmd: ["docker", "compose", "-f", path.join(dockerComposeDir, "docker-compose.yml"), "up", "-d"],
        cwd: dockerComposeDir,
        stdout: "pipe",
        stderr: "pipe",
    });

    // Capture startup output
    let startupOutput = "";
    startProcess.stdout.pipeTo(new WritableStream({
        write(chunk) {
            startupOutput += chunk;
            process.stdout.write(chunk);
        }
    }));

    startProcess.stderr.pipeTo(new WritableStream({
        write(chunk) {
            startupOutput += chunk;
            process.stderr.write(chunk);
        }
    }));

    await startProcess.exited;
    fs.appendFileSync(outputFile, "```\n" + startupOutput + "\n```\n\n");

    // Check service status
    const statusProcess = spawn({
        cmd: ["docker", "compose", "-f", path.join(dockerComposeDir, "docker-compose.yml"), "ps"],
        cwd: dockerComposeDir,
        stdout: "pipe",
    });

    const statusOutput = await new Response(statusProcess.stdout).text();
    console.log("\nService Status:");
    console.log(statusOutput);

    fs.appendFileSync(outputFile, "## Service Status\n\n```\n" + statusOutput + "\n```\n\n");

    return statusProcess.exited;
}

// Monitor logs
async function monitorLogs() {
    console.log("\nMonitoring logs for issues...");
    fs.appendFileSync(outputFile, "## Issues Detected\n\n");

    const cmd = ["docker", "compose", "-f", path.join(dockerComposeDir, "docker-compose.yml"), "logs", "--follow"];

    // Add specific services if provided
    if (specificServices.length > 0) {
        cmd.push(...specificServices);
    }

    const logsProcess = spawn({
        cmd,
        cwd: dockerComposeDir,
        stdout: "pipe",
        stderr: "pipe",
    });

    // Process stdout
    const processLogStream = async (stream: ReadableStream<Uint8Array>, isError = false) => {
        const textDecoder = new TextDecoder();
        const reader = stream.getReader();

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                if (value) {
                    const text = textDecoder.decode(value);
                    const lines = text.split("\n");

                    for (const line of lines) {
                        if (line.trim() === "") continue;

                        // Add to the recent lines buffer
                        recentLines.push(line);
                        if (recentLines.length > LINE_BUFFER_SIZE) {
                            recentLines.shift();
                        }

                        // Check if this line contains important information
                        const lowerCaseLine = line.toLowerCase();
                        const matchedLevel = LOG_LEVELS.find(level => lowerCaseLine.includes(level));

                        if (matchedLevel) {
                            // Extract service name
                            const serviceMatch = line.match(/^([^|]+)\s*\|/);
                            const service = serviceMatch ? serviceMatch[1].trim() : "unknown";

                            // Get context (previous lines)
                            const contextStartIndex = Math.max(0, recentLines.length - CONTEXT_LINES - 1);
                            const context = recentLines.slice(contextStartIndex, recentLines.length - 1);

                            issues.push({
                                service,
                                level: matchedLevel,
                                message: line,
                                context
                            });

                            // Log the issue
                            console.log(`\n[Issue detected in ${service}] - ${matchedLevel}`);
                            console.log(`  ${line}`);

                            // Save to file immediately
                            fs.appendFileSync(outputFile, `### Issue in ${service} (${matchedLevel})\n\n`);
                            fs.appendFileSync(outputFile, "Context:\n```\n" + context.join("\n") + "\n```\n\n");
                            fs.appendFileSync(outputFile, "Message:\n```\n" + line + "\n```\n\n");
                        }

                        // Echo the line to console
                        if (isError) {
                            process.stderr.write(line + "\n");
                        } else {
                            process.stdout.write(line + "\n");
                        }
                    }
                }
            }
        } catch (err) {
            console.error("Error reading log stream:", err);
        } finally {
            reader.releaseLock();
        }
    };

    // Process both stdout and stderr
    await Promise.all([
        processLogStream(logsProcess.stdout),
        processLogStream(logsProcess.stderr, true)
    ]);

    return logsProcess.exited;
}

// Function to collect system stats periodically
async function collectSystemStats() {
    // Collect container stats
    const statsProcess = spawn({
        cmd: ["docker", "stats", "--no-stream"],
        stdout: "pipe",
    });

    const statsOutput = await new Response(statsProcess.stdout).text();
    fs.appendFileSync(outputFile, "## System Stats\n\n```\n" + statsOutput + "\n```\n\n");

    // If we're monitoring Redis, collect Redis-specific metrics
    if (specificServices.length === 0 || specificServices.includes("redis")) {
        await collectRedisMetrics();
    }

    return statsProcess.exited;
}

// Function to collect Redis-specific metrics
async function collectRedisMetrics() {
    console.log("Collecting Redis metrics...");
    
    // Commands to run inside Redis container
    const redisCommands = [
        { name: "info", args: [] },
        { name: "client", args: ["list"] },
        { name: "config", args: ["get", "maxmemory"] },
        { name: "info", args: ["keyspace"] }
    ];
    
    fs.appendFileSync(outputFile, "## Redis Metrics\n\n");
    
    for (const cmd of redisCommands) {
        try {
            const dockerCmd = [
                "docker", "exec", "data-ingestion-redis-1", 
                "redis-cli", cmd.name, ...cmd.args
            ];
            
            const redisProcess = spawn({
                cmd: dockerCmd,
                stdout: "pipe",
                stderr: "pipe",
            });
            
            const redisOutput = await new Response(redisProcess.stdout).text();
            const section = cmd.name + (cmd.args.length > 0 ? " " + cmd.args.join(" ") : "");
            
            fs.appendFileSync(outputFile, `### Redis ${section}\n\n`);
            fs.appendFileSync(outputFile, "```\n" + redisOutput + "\n```\n\n");
            
            // Store metrics for analysis
            redisMetrics[section] = redisOutput;
            
            await redisProcess.exited;
        } catch (error) {
            console.error(`Error collecting Redis metrics for ${cmd.name}:`, error);
            fs.appendFileSync(outputFile, `Error collecting Redis metrics for ${cmd.name}\n\n`);
        }
    }
    
    // Analyze Redis metrics
    analyzeRedisMetrics();
}

// Function to analyze Redis metrics and provide insights
function analyzeRedisMetrics() {
    if (Object.keys(redisMetrics).length === 0) {
        return;
    }
    
    fs.appendFileSync(outputFile, "## Redis Analysis\n\n");
    
    try {
        // Extract memory usage
        const memoryMatch = redisMetrics["info"]?.match(/used_memory_human:(\S+)/);
        if (memoryMatch) {
            fs.appendFileSync(outputFile, `- Memory usage: ${memoryMatch[1]}\n`);
        }
        
        // Extract client connections
        const connectedClientsMatch = redisMetrics["info"]?.match(/connected_clients:(\d+)/);
        if (connectedClientsMatch) {
            fs.appendFileSync(outputFile, `- Connected clients: ${connectedClientsMatch[1]}\n`);
        }
        
        // Extract keyspace info
        const keyspaceInfo = redisMetrics["info keyspace"];
        if (keyspaceInfo && !keyspaceInfo.includes("empty")) {
            fs.appendFileSync(outputFile, "- Key statistics:\n");
            const keyspaceLines = keyspaceInfo.split("\n").filter(line => line.trim().length > 0);
            for (const line of keyspaceLines) {
                if (line.startsWith("db")) {
                    fs.appendFileSync(outputFile, `  - ${line}\n`);
                }
            }
        } else {
            fs.appendFileSync(outputFile, "- Keyspace: No keys found in database\n");
        }
        
        // Check if maxmemory is set
        const maxmemoryMatch = redisMetrics["config get maxmemory"]?.match(/maxmemory\s+(\d+)/);
        if (maxmemoryMatch && maxmemoryMatch[1] !== "0") {
            fs.appendFileSync(outputFile, `- Max memory limit: ${maxmemoryMatch[1]} bytes\n`);
        } else {
            fs.appendFileSync(outputFile, "- Max memory limit: Not set (unlimited)\n");
        }
        
        // Add general recommendations
        fs.appendFileSync(outputFile, "\n### Recommendations\n\n");
        fs.appendFileSync(outputFile, "- Ensure Redis has enough memory allocated for caching needs\n");
        fs.appendFileSync(outputFile, "- Monitor keyspace size to prevent unbounded growth\n");
        fs.appendFileSync(outputFile, "- Consider setting appropriate eviction policies if not already configured\n");
        fs.appendFileSync(outputFile, "- For production, consider enabling persistence with RDB snapshots or AOF logs\n\n");
    } catch (error) {
        console.error("Error analyzing Redis metrics:", error);
        fs.appendFileSync(outputFile, "Error analyzing Redis metrics\n\n");
    }
}

// Function to get container health status
async function getContainerHealth() {
    const healthCmd = [
        "docker", "compose", "-f",
        path.join(dockerComposeDir, "docker-compose.yml"),
        "ps", "--format", "json"
    ];

    const healthProcess = spawn({
        cmd: healthCmd,
        cwd: dockerComposeDir,
        stdout: "pipe",
    });

    const healthOutput = await new Response(healthProcess.stdout).text();

    try {
        const containers = JSON.parse(healthOutput);
        fs.appendFileSync(outputFile, "## Container Health\n\n");

        for (const container of containers) {
            fs.appendFileSync(outputFile, `- **${container.Service}**: ${container.State} (Health: ${container.Health || 'N/A'})\n`);
        }

        fs.appendFileSync(outputFile, "\n");
    } catch (err) {
        console.error("Error parsing container health:", err);
        fs.appendFileSync(outputFile, "## Container Health\n\nError parsing health information\n\n");
    }

    return healthProcess.exited;
}

// Function to generate a summary report
function generateSummary() {
    if (issues.length === 0) {
        fs.appendFileSync(outputFile, "## Summary\n\nNo issues detected during monitoring.\n\n");
        return;
    }

    // Group issues by service
    const issuesByService: Record<string, typeof issues> = {};
    for (const issue of issues) {
        if (!issuesByService[issue.service]) {
            issuesByService[issue.service] = [];
        }
        issuesByService[issue.service].push(issue);
    }

    fs.appendFileSync(outputFile, "## Summary\n\n");
    fs.appendFileSync(outputFile, `Total issues detected: ${issues.length}\n\n`);

    for (const [service, serviceIssues] of Object.entries(issuesByService)) {
        fs.appendFileSync(outputFile, `### ${service}: ${serviceIssues.length} issues\n\n`);

        // Group by error type
        const issueTypes: Record<string, number> = {};
        for (const issue of serviceIssues) {
            issueTypes[issue.level] = (issueTypes[issue.level] || 0) + 1;
        }

        for (const [type, count] of Object.entries(issueTypes)) {
            fs.appendFileSync(outputFile, `- ${type}: ${count}\n`);
        }

        fs.appendFileSync(outputFile, "\n");
    }
}

// Main function
async function main() {
    try {
        // Start Docker Compose
        await startDockerCompose();

        // Setup monitoring in the background
        const monitorPromise = monitorLogs();

        // Give services time to start before checking health
        await new Promise(resolve => setTimeout(resolve, 30000));

        // Collect initial system stats
        await collectSystemStats();

        // Get container health
        await getContainerHealth();

        // Setup periodic system stats collection
        const statsInterval = setInterval(async () => {
            await collectSystemStats();
            await getContainerHealth();
        }, 300000); // Every 5 minutes

        // Listen for Ctrl+C to gracefully exit
        process.on('SIGINT', async () => {
            console.log("\nShutting down...");
            clearInterval(statsInterval);

            // Generate summary
            generateSummary();

            console.log(`\nResults saved to ${outputFile}`);
            process.exit(0);
        });

        console.log("\nMonitoring in progress. Press Ctrl+C to generate report and exit.");

        // Wait indefinitely
        await monitorPromise;

    } catch (error) {
        console.error("Error:", error);
        fs.appendFileSync(outputFile, `## Error\n\n${error}\n\n`);
    }
}

main().catch(console.error);