#!/usr/bin/env bun
import { $ } from "bun";
import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";

// Parse command line arguments
const { values } = parseArgs({
    options: {
        services: { type: "string", short: "s", default: "" },
        output: { type: "string", short: "o", default: "docker-insights.txt" },
        help: { type: "boolean", short: "h", default: false }
    }
});

// Show ASCII art banner
console.log(`
 _____             _                ___  ___            _ _             
|  __ \\           | |               |  \\/  |           (_) |            
| |  | | ___   ___| | _____ _ __    | .  . | ___  _ __  _| |_ ___  _ __ 
| |  | |/ _ \\ / __| |/ / _ \\ '__|   | |\\/| |/ _ \\| '_ \\| | __/ _ \\| '__|
| |__| | (_) | (__|   <  __/ |      | |  | | (_) | | | | | || (_) | |   
|_____/ \\___/ \\___|_|\\_\\___|_|      \\_|  |_/\\___/|_| |_|_|\\__\\___/|_|   
                                                                         
`);

// Display version info
const getBunVersion = async () => {
    const bunVersion = await $`bun -v`.text();
    return bunVersion.trim();
};

const main = async () => {
    const bunVersion = await getBunVersion();
    console.log(`Bun Version: ${bunVersion}`);
    console.log("Docker Monitor v1.0.0");
    console.log("");

    // Check if Bun is installed - we're already running in Bun, so this check is redundant
    // but keeping for similarity with the original script logic

    // Check if Docker is running
    try {
        await $`docker info`.quiet();
    } catch (error) {
        console.error("\x1b[31mError: Docker is not running or not accessible.");
        console.error("Please start Docker Desktop or Docker Engine before running this script.\x1b[0m");
        process.exit(1);
    }

    // Check if docker-compose.yml exists in parent directory
    const parentDir = path.resolve(process.cwd(), "..");
    const dockerComposeFile = path.join(parentDir, "docker-compose.yml");

    if (!fs.existsSync(dockerComposeFile)) {
        console.error("\x1b[31mError: docker-compose.yml not found in parent directory.");
        console.error("Please run this script from the docker-monitor directory within DataIngestion.\x1b[0m");
        process.exit(1);
    }

    // Build command arguments
    const args = [];

    if (values.help) {
        args.push("--help");
    } else {
        args.push("--dir=..");
        args.push(`--output=${values.output}`);

        if (values.services) {
            args.push(`--services=${values.services}`);
        }
    }

    // Display what we're about to do
    console.log("\x1b[32mStarting Docker Compose monitoring:\x1b[0m");
    if (values.services) {
        console.log(`\x1b[33m  - Monitoring services: ${values.services}\x1b[0m`);
    } else {
        console.log("\x1b[33m  - Monitoring all services\x1b[0m");
    }
    console.log(`\x1b[33m  - Output file: ${values.output}\x1b[0m`);
    console.log("");
    console.log("\x1b[35mPress Ctrl+C to stop monitoring and generate report.\x1b[0m");
    console.log("");

    // Run the monitor
    try {
        await $`bun run index.ts ${args}`;
    } catch (error) {
        console.error("\x1b[31mAn error occurred while running the monitor:");
        console.error(error.message);
        process.exit(1);
    }
};

main(); 