#!/usr/bin/env bun
import { $ } from "bun";
import path from "node:path";

// Default output file
const outputFile = "elixir-redis-insights.txt";

// Show banner
console.log(`
 _____  _      ______          _ _        _____ ______    _ _     
|  ___|| |     |  _  \\        | (_)      |  _  || ___ \\  | (_)    
| |__  | |     | | | |__  __ _| |_ _ __  | | | || |_/ /__| |_ ___ 
|  __| | |     | | | |\\ \\/ /| | | | '__| | | | ||    // _\` | / __|
| |___ | |____ | |/ /  >  < | | | | |    \\ \\_/ /| |\\ \\ (_| | \\__ \\
\\____/ \\_____/ |___/  /_/\\_\\|_|_|_|_|     \\___/ \\_| \\_\\__,_|_|___/                                                                        
`);

console.log("\x1b[33mComprehensive monitoring script for Elixir ingestion and Redis services\x1b[0m");
console.log("\x1b[33mThis will monitor both the Elixir ingestion service and Redis, collecting detailed metrics\x1b[0m");
console.log("");

// Run the monitor with Elixir and Redis service focus
const servicesToMonitor = "elixir_ingestion,redis,timescaledb";

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
        console.error("\x1b[31mAn error occurred while running Elixir/Redis monitoring:");
        console.error(error);
        process.exit(1);
    }
};

main();
