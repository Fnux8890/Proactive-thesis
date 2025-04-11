
Okay, let's outline a detailed, DS and SE-oriented plan for the Rust pipeline, focusing on robustness and starting incrementally.

**Core Principles:**

1. **Configuration-Driven:** Parsing rules, file types, paths, and metadata will be driven by an enhanced `data_files.json` (integrating `meta_data.md` info).
2. **Modularity (SE):** Separate Rust modules for configuration, file processing logic, different parsers (CSV, JSON), data models, and error handling.
3. **Early & Typed Parsing (DS & SE):** Convert data to appropriate Rust types (`chrono::DateTime`, `f64`, `Option<T>`, etc.) as soon as possible during parsing, not later.
4. **Explicit NULL/Error Handling (DS & SE):** Clearly define how missing values (`""`, `#NUM!`, `-999`) and parsing errors are handled (e.g., map to `NULL`/`None`, skip row, log error).
5. **Target Schema Focus (DS):** Define the target data structure early (even if DB insertion is later) to guide parsing.
6. **Incremental Development (SE):** Build and verify functionality piece by piece, starting with one file type.

---

**Implementation Plan:**

**Phase 0: Project Setup & Enhanced Configuration**

1. **Define Enhanced `data_files.json` Structure:**
    * **Goal:** Create a single source of truth for file paths and parsing instructions.
    * **Action:** Define a new JSON structure (or augment the existing one) for each file entry. Include fields like:
        * `workspace_path`: Original path.
        * `container_path`: Path inside container.
        * `status`: (e.g., "pending", "processing", "done", "skipped", "error").
        * `format_type`: A unique identifier for the file format/source (e.g., "KnudJepsenNO3Lamp", "AarslevCelleCSV", "AarslevCelleJSON", "AarslevWeather").
        * `source_system`: (e.g., "KnudJepsen", "Aarslev").
        * `delimiter`: Character (e.g., ";", ",").
        * `quoting`: Boolean or specific quote character (`"`).
        * `header_rows`: Number of rows to skip.
        * `timestamp_info`: Object detailing column index/name(s) for timestamp parts (e.g., `{"date_col": 0, "time_col": 1, "format": "%Y-%m-%d %H:%M:%S"}` or `{"unix_ms_col": 0}`).
        * `column_map`: Array of objects mapping source column index/name to target canonical name and expected type (`{"source": "Celle 5: Lufttemperatur", "target": "air_temp_c", "type": "float"}`). *Crucial for handling varied headers.*
        * `null_markers`: List of strings to treat as NULL (e.g., `["", "#NUM!"]`).
    * **Output:** A documented new structure for `data_files.json`.
    * **(Manual Step):** Populate the *first few entries* of the *actual* `data_files.json` file with this detailed structure based on `meta_data.md`.

2. **Rust Configuration Module (`config.rs`):**
    * **Goal:** Load and manage the pipeline configuration.
    * **Action:**
        * Define Rust structs mirroring the new `data_files.json` structure using `serde`. Use `Option<T>` for fields that might not apply to all formats.
        * Implement a function `load_config(path: &Path) -> Result<Vec<FileConfig>, ConfigError>` to read and deserialize `data_files.json`.
        * Define `ConfigError` enum in `error.rs`.
    * **Output:** `config.rs`, `error.rs` (basic).

3. **Basic Orchestration (`main.rs`):**
    * **Goal:** Load config and prepare for processing the first file.
    * **Action:**
        * Modify `main.rs` to call `load_config`.
        * Get the *first* `FileConfig` entry from the loaded vector.
        * Log the path and `format_type` of this first file.
    * **Output:** Modified `main.rs`.

**Phase 1: Implement First Parser (Knud Jepsen CSV)**

1. **Target Data Model (`data_models.rs`):**
    * **Goal:** Define the unified structure for parsed data rows before potential DB insertion.
    * **Action:** Create a struct `ParsedRecord` (or similar). Initially, include fields relevant to `KnudJepsenNO3Lamp` (e.g., `timestamp: DateTime<Utc>`, `air_temp_middle_c: Option<f64>`, `co2_ppm: Option<f64>`, etc.). Use `Option<T>` extensively.
    * **Output:** `data_models.rs`.

2. **CSV Parser Module (`parsers/csv_parser.rs` & `parsers/mod.rs`):**
    * **Goal:** Create a reusable CSV parser driven by the configuration.
    * **Action:**
        * Create the `parsers` module (`mod.rs`).
        * In `csv_parser.rs`, implement `parse_csv(config: &FileConfig, file_path: &Path) -> Result<Vec<ParsedRecord>, ParseError>`.
        * Use the `csv` crate. Configure the reader with `delimiter` and `quoting` from `config`.
        * Skip `config.header_rows`.
        * Iterate through rows:
            * Combine date/time columns or parse Unix timestamp using `config.timestamp_info` and `chrono`. Handle errors -> `ParseError::Timestamp`.
            * For each column defined in `config.column_map`:
                * Get the string value from the CSV record by index/name.
                * Check if it's in `config.null_markers` -> map to `None`.
                * If not null, attempt parsing/conversion based on `column_map[...].type`. Use `trim()`, handle potential decimal comma (`replace(',', '.')`), use `parse::<T>()`. Handle errors -> `ParseError::TypeConversion`.
                * Store the parsed `Option<T>` value.
            * Construct a `ParsedRecord` struct.
            * Collect successfully parsed records. Log errors for failed rows (include row number, file path, error details).
    * **Output:** `parsers/mod.rs`, `parsers/csv_parser.rs`, extended `error.rs`.

3. **File Processor Logic (`file_processor.rs`):**
    * **Goal:** Select and invoke the correct parser based on config.
    * **Action:**
        * Create `file_processor.rs`.
        * Add `process_file(config_entry: &FileConfig) -> Result<Vec<ParsedRecord>, PipelineError>`.
        * Use a `match` statement on `config_entry.format_type`:
            * For `"KnudJepsenNO3Lamp"` (and similar CSV types later): Call `parsers::csv_parser::parse_csv`.
            * Add placeholders for other types (`"AarslevCelleJSON"`, etc.).
            * Handle unknown `format_type` as an error.
        * Map `ParseError` to `PipelineError`.
    * **Output:** `file_processor.rs`.

4. **Update `main.rs`:**
    * **Goal:** Call the file processor for the first file and log results.
    * **Action:**
        * Call `file_processor::process_file` with the first config entry.
        * Log the number of records parsed successfully.
        * Log any errors returned.
    * **Output:** Updated `main.rs`.

**Phase 2: Handling Aarslev JSON & Skipping Logic**

1. **JSON Parser Module (`parsers/json_parser.rs`):**
    * **Goal:** Parse the specific nested structure of `output-*.csv.json` files.
    * **Action:**
        * Add `parse_aarslev_celle_json(config: &FileConfig, file_path: &Path) -> Result<Vec<ParsedRecord>, ParseError>`.
        * Use `serde_json` to deserialize the top-level JSON object (`Map<String, SensorStream>`).
        * Define helper structs (`SensorStream`, `Properties`, `Metadata`) to match the JSON structure.
        * Iterate through the map's key-value pairs (sensor path, stream data).
        * For each sensor's `Readings` array:
            * Iterate through `[timestamp_ms, value]` pairs.
            * Convert timestamp using `chrono::NaiveDateTime::from_timestamp_millis` then to `DateTime<Utc>`. Handle errors.
            * Determine the target field in `ParsedRecord` based on the sensor path key (e.g., `"/Cell5/air_temperature"` maps to `air_temp_c`). *This mapping might need to be part of the config or hardcoded initially.*
            * Parse the `value` based on `stream_data.Properties.ReadingType` (likely always "double" based on example, but check). Handle errors.
            * Create/update `ParsedRecord` instances. *Challenge: JSON is columnar, need to aggregate records by timestamp.* A `HashMap<i64, ParsedRecord>` might be useful here, keyed by timestamp millis.
    * **Output:** `parsers/json_parser.rs`.

2. **Refine `file_processor.rs`:**
    * **Goal:** Implement the logic to prioritize JSON and call the correct parser.
    * **Action:**
        * Modify `process_file` (or add a higher-level orchestrator):
            * *Before* calling a parser for an `AarslevCelleCSV` file from `celle5` or `celle6`, check the *entire* loaded config list: Does a corresponding `.csv.json` file exist with `status == "pending"`?
            * If yes: Log skipping the `.csv` file, update its status in the config (in memory for now) to `"skipped_superseded"`, and return `Ok(vec![])`.
            * If no: Proceed to parse the CSV using `csv_parser`.
            * Add a `match` arm for `format_type == "AarslevCelleJSON"` to call `json_parser`.
    * **Output:** Updated `file_processor.rs`.

**Phase 3: Iteration and Generalization**

1. **Update `main.rs` Loop:** Modify `main.rs` to iterate through *all* entries in the loaded config, calling `file_processor::process_file` for each.
2. **Add Configurations:** Populate `data_files.json` with detailed configurations for *all* file types identified in `meta_data.md`.
3. **Implement Other Parsers:** Add `match` arms in `file_processor.rs` and potentially new functions in `csv_parser.rs` (or new parser files if needed) to handle the remaining `format_type` variations (Aarslev Weather CSV, other Knud Jepsen CSVs, etc.), using their specific configurations.
4. **Refine Error Handling:** Improve logging, potentially aggregate errors per file, decide on pipeline failure strategy (fail fast vs. process remaining).
5. **Database Insertion (`db_inserter.rs`):** Implement logic to connect to TimescaleDB (using `sqlx`) and insert the `Vec<ParsedRecord>` data, mapping struct fields to table columns. Handle potential DB errors.

---

This plan starts small (config + first file), ensures core parsing logic is robust and type-safe, implements the specific JSON prioritization, and then scales to handle all files using the configuration-driven approach. This balances DS needs (correct types, nulls, metadata) with SE principles (modularity, configurability, incremental development).
