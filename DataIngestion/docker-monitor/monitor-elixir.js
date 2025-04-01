#!/usr/bin/env bun
import { $ } from "bun";
import path from "node:path";

// Default output file is elixir-insights.txt
const outputFile = "elixir-insights.txt";

// Show banner
console.log(`
 _____ _ _      _         __  __             _ _            
|  ___| (_)_  _(_)_ __   |  \\/  | ___  _ __ (_) |_ ___  _ __
| |_  | | \\ \\/ / | '__|  | |\\/| |/ _ \\| '_ \\| | __/ _ \\| '__|
|  _| | | |>  <| | |     | |  | | (_) | | | | | || (_) | |   
|_|   |_|_/_/\\_\\_|_|     |_|  |_|\\___/|_| |_|_|\\__\\___/|_|   
`);

console.log("\x1b[33mSpecialized monitoring script for the Elixir ingestion service\x1b[0m");
console.log("\x1b[33mThis will focus only on the Elixir-related services and collect detailed information\x1b[0m");
console.log("");

// Run the monitor with Elixir-specific service focus
const servicesToMonitor = "elixir_ingestion,timescaledb,redis";

console.log(`\x1b[32mMonitoring services: ${servicesToMonitor}\x1b[0m`);
console.log(`\x1b[32mOutput will be saved to: ${outputFile}\x1b[0m`);
console.log("");
console.log("\x1b[35mPress Ctrl+C to stop monitoring and generate the final report.\x1b[0m");
console.log("");

// Call the main script with our arguments
const main = async () => {
    try {
        await $`bun run start-monitor.js --services=${servicesToMonitor} --output=${outputFile}`;
    } catch (error) {
        console.error("\x1b[31mAn error occurred while running the monitor:");
        console.error(error.message);
        process.exit(1);
    }
};

main(); 