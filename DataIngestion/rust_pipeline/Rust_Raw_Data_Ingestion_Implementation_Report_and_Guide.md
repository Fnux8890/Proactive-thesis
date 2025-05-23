# Rust Raw-Data Ingestion — Implementation Report & Guide

This document focuses exclusively on the Rust codebase that ingests heterogeneous CSV/JSON sensor logs into the `sensor_data` database schema.  It supersedes earlier planning notes that covered Python feature extraction and broader project management.

---

## 1 | Current ingestion pipeline at a glance

* **Parsers** – `csv_parser.rs`, `json_parser.rs`, `aarslev_morten_sdu_parser.rs`; each converts raw rows into `ParsedRecord` structs.
* **Orchestrator** – `file_processor.rs` chooses the correct parser based on `FileConfig.format`.
* **Validation** – `validation.rs` enforces schema and basic range checks before DB insert.
* **Upsampling** – `perform_upsampling` regularises data to a 1-minute grid and exports `sensor_data_upsampled.csv`.

Architecture already separates I/O (CSV/JSON), conversion, validation, and DB I/O, but several gaps remain in correctness, extensibility, and performance.

---

## 2 | Outstanding issues (must-fix)

1. **Timezone handling** – CSV parsers assume local timestamps are UTC, shifting winter data by +1 h.
2. **`set_field!` macro brittleness** – Adding a new column requires macro edit; silent drops otherwise.
3. **Duplicate comma-decimal helpers** – Inconsistent parsing rules across modules.
4. **Loose integer coercion** – Accepts floats like `1.9999` as `2`.
5. **Validation gaps** – Only RH range checked; temperature, CO₂, radiation missing.
6. **Skip-row accounting** – Individual rows logged but no aggregated reason counts.
7. **Single-threaded ingest** – Inefficient on multi-core hosts.

---

## 3 | Change checklist by file

### 3.1 `config.rs`

• Add `timezone: Option<String>` to `TimestampInfo` struct.  Example: `{ "strategy": "UnixMillis", "timezone": "Europe/Copenhagen" }`.

### 3.2 `csv_parser.rs`

• Use `chrono_tz::Tz` to convert local time → UTC using new `timezone` field.
• Import `parse_locale_float` from `utils.rs` (see below).
• Tighten integer coercion: `abs(frac) < 1e-6` else `PipelineError::NonIntegralFloat`.
• Optional: wrap `reader.records()` with `par_bridge()` for record-level parallelism.

### 3.3 `json_parser.rs`

• Reuse the same timezone conversion helper so JSON and CSV timestamps align.

### 3.4 `aarslev_morten_sdu_parser.rs`

• Remove inline decimal parser; call `utils::parse_locale_float`.

### 3.5 `models.rs`

• Delete `set_field!` macro.
• Create `lazy_static!` map `FIELD_SETTERS` that pairs header names with setter fns.
• Expose a `set_field(&mut self, header: &str, value: &str)` that looks up and applies the setter; returns error if unknown.

### 3.6 `utils.rs` (new)

• `pub fn parse_locale_float(s: &str) -> Result<f64, ParseFloatError>` – replaces commas with dots, then `parse()`.
• `pub fn parse_with_tz(ts: &str, tz: &str) -> Result<DateTime<Utc>, ChronoError>` – helper wrapping `chrono_tz` lookup.

### 3.7 `validation.rs`

• Extend `DATA_INTEGRITY_SQL` with three additional clauses: temperature (<-40 OR >60), CO₂ (<200 OR >3000), radiation (<0).
• Re-run unit tests to expect new failure modes.

### 3.8 `file_processor.rs` & `errors.rs`

• Introduce enum `SkipReason` and increment counts in a `HashMap<SkipReason, usize>` during parsing.
• At end of `process_one_file`, log a one-line summary: `info!(?path, ?counts, "row_skip_summary")`.

### 3.9 `main.rs`

• Add `rayon = "1.8"` to `Cargo.toml`.
• Collect all paths, then: `paths.par_iter().try_for_each(process_one_file)` for file-level parallelism.
• Ensure each worker thread gets its own connection from the pool (or create a new client per file).

### 3.10 `Cargo.toml`

• Add deps: `chrono-tz`, `rayon`, `lazy_static`.
• Enable `features = ["serde"]` if you switch to Serde for `ParsedRecord`.

### 3.11 `tests/` (new folder)

• Five-row fixtures per parser module.
• Assert: correct UTC conversion, field counts, range validation, and skip-reason totals.

---

## 4 | Acceptance criteria

* End-to-end ingest of one month completes with:
  – runtime ≤ 50 % of current baseline (thanks to Rayon),
  – no `DataIntegrityError`,
  – header of `sensor_data_upsampled.csv` identical to `TARGET_COLUMNS`,
  – row count ≥ 98 % of theoretical (days × 1440),
  – skip-reason summary logged.

---

## 5 | Next steps

1. Apply changes in the listed order (timezone first, macro removal second).
2. Run unit tests locally (`cargo test`), then an end-to-end month ingest.
3. If acceptance criteria pass, tag the crate `v0.9.0` and push the container image.

Once these steps are complete, the loader will be timezone-correct, schema-robust, and parallelised, ready for the tsfresh feature-extraction stage.
