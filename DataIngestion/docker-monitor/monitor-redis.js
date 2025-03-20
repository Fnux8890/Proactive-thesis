#!/usr/bin/env bun
import { $ } from "bun";
import path from "node:path";

// Default output file
const outputFile = "redis-insights.txt";

// Show banner
console.log(`
 _____          _ _       __  __             _ _            
|  __ \\        | (_)     |  \\/  |           (_) |           
| |__) |___  __| |_ ___  | \\  / | ___  _ __  _| |_ ___  ___ 
|  _  // _ \\/ _\` | / __| | |\\/| |/ _ \\| '_ \\| | __/ _ \\/ __|
| | \\ \\  __/ (_| | \\__ \\ | |  | | (_) | | | | | || (_) \\__ \\
|_|  \\_\\___|\\__,_|_|___/ |_|  |_|\\___/|_| |_|_|\\__\\___/|___/
`);

console.log("\x1b[33mSpecialized monitoring script for Redis cache and message broker\x1b[0m");
console.log("\x1b[33mThis will focus on Redis and connected services, collecting detailed Redis metrics\x1b[0m");
console.log("");

// Run the monitor with Redis-specific service focus
const servicesToMonitor = "redis,elixir_ingestion";

console.log(`\x1b[32mMonitoring services: ${servicesToMonitor}\x1b[0m`);
console.log(`\x1b[32mOutput will be saved to: ${outputFile}\x1b[0m`);
console.log("");
console.log("\x1b[35mPress Ctrl+C to stop monitoring and generate the final report.\x1b[0m");
console.log("");

// Call the main script with our arguments
const main = async () => {
    try {
        await $`bun run index.ts --services=${servicesToMonitor} --output=${outputFile}`;
    } catch (error) {
        console.error("\x1b[31mAn error occurred while running Redis monitoring:");
        console.error(error);
        process.exit(1);
    }
};

main();
