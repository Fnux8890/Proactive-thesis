use std::env;
use std::fs;
use std::path::Path;
use serde::Deserialize;
use walkdir::WalkDir;

#[derive(Deserialize, Debug)]
struct DataFileEntry {
    workspace_path: String,
    container_path: String,
    status: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Starting Rust Pipeline ---");

    // --- Get Data Source Path from Environment Variable ---
    let data_source_env = env::var("DATA_SOURCE_PATH").unwrap_or_else(|_|
        {
            eprintln!("Warning: DATA_SOURCE_PATH environment variable not set. Defaulting to './data' relative to executable.");
            "./data".to_string()
        }
    );
    let data_source_path = Path::new(&data_source_env);
    println!("Data source directory (inside container): {:?}", data_source_path);

    // --- Option 1: Read file list from data_files.json (like Python setup) ---
    // This assumes data_files.json is available relative to the executable's context
    // In Docker, this might need adjustment depending on where things are copied/mounted.
    // Let's assume for now it's accessible relative to the workdir where the binary runs
    // Or perhaps better, relative to the *host* workspace mapped into the container.

    // Let's try finding it relative to the *parent* of the executable's directory first
    // (assuming executable is in /usr/local/bin and source/config might be mounted elsewhere)
    // THIS IS A GUESS and might need refinement based on actual Docker setup.

    let possible_json_path = Path::new("/app/data_files.json"); // A common pattern if app code is in /app

    // --- Option 2: Discover files by walking the DATA_SOURCE_PATH directory ---
    println!("Discovering files via directory walk:");
    let mut discovered_files_count = 0;
    for entry in WalkDir::new(data_source_path)
        .into_iter()
        .filter_map(|e| e.ok()) // Ignore errors
        .filter(|e| e.file_type().is_file()) // Only consider files
        .filter(|e| {
            // Basic filter for CSV or JSON
            e.path().extension().map_or(false, |ext| ext == "csv" || ext == "json")
        })
    {
        println!("  Found: {:?}", entry.path());
        discovered_files_count += 1;
    }
    println!("Total files found via walk: {}", discovered_files_count);


    // --- Attempt to read data_files.json (if needed and found) ---
    // This part is less reliable in Docker without knowing exact mount/copy locations.
    // The directory walk (Option 2) is generally more robust if all data is under DATA_SOURCE_PATH.
    /*
    if possible_json_path.exists() {
        println!("Reading file list from {:?}", possible_json_path);
        let json_content = fs::read_to_string(possible_json_path)?;
        let files: Vec<DataFileEntry> = serde_json::from_str(&json_content)?;

        println!("Files listed in data_files.json:");
        for file_entry in files {
            // We'd use file_entry.container_path within the Docker context
            println!("  - {}", file_entry.container_path);
        }
    } else {
        eprintln!("Warning: data_files.json not found at expected location ({:?}). Relying on directory walk.", possible_json_path);
    }
    */

    println!("--- Rust Pipeline Finished (Initial Setup) ---");
    Ok(())
} 