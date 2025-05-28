# SenmaticDataTools-master Overview

The **SenmaticDataTools-master** suite is a collection of Java-based tools designed for processing CSV files derived from sensor readings and weather forecasts. These utilities provide functionalities such as cleaning, merging, downsampling, and reformatting data to prepare it for ingestion into systems like sMAP or other analytics pipelines. This document explains the key components, how they interact, and details significant code functionalities.

---

## 1. SenmaticTosMAPCleaner.java

**Purpose:**  
Converts raw CSV files exported from Excel into a cleaned format suitable for sMAP ingestion.

**Key Functionalities:**

- **Input Handling:**
  - Supports both single file and batch processing when the input ends with `"*.csv"`. In the batch mode, the tool lists all CSV files in the specified directory.
- **Data Cleaning Steps:**
  - Reads each line using the `CSVReader` with a semicolon (`;`) delimiter.
  - For each field:
    - Replaces empty entries (after trimming) with `"0.0"`.
    - Converts commas to periods to standardize decimal points.
    - Skips lines that contain error tokens like `#NUM!` or values above a configured threshold (e.g., >1.0E+30).
- **Output Generation:**
  - Writes the cleaned data to an output CSV file, preserving the header and appending each valid cleaned line.

---

## 2. MultipleColumnToOneCsv.java

**Purpose:**  
Consolidates CSV files that contain multiple repeated sets of columns (due to column grouping) into a single-column format.

**Key Functionalities:**

- **Directory Processing:**
  - When provided with an argument ending in `"*.csv"`, it processes all CSV files within the directory.
  - Recursively calls its main routine with individual file arguments.
- **Chunking Mechanism:**
  - Reads the first line to determine the splitting point, detecting empty fields that signal the start of a new column group.
  - Uses a generic `chunk` method to divide the line (and subsequent lines) into multiple arrays, each representing a group of columns.
- **Output Generation:**
  - Rewrites the data by flattening the grouped columns into separate CSV entries and writes to the output file.

---

## 3. DataCombiner.java

**Purpose:**  
Merges two CSV files based on a common timestamp, typically using a weather file as the target and data from sMAP as the secondary input.

**Key Functionalities:**

- **File Validation:**
  - Expects at least three arguments: target file, secondary file, and the output file.
  - Validates file existence; gracefully exits if a file is not found.
- **Merging Logic:**
  - **Secondary File Mapping:** Uses a `CSVReader` (with `;` delimiter) to read the secondary file. The first column is assumed to be the timestamp. The remaining columns are stored in a map keyed by the timestamp.
  - **Target File Processing:** Reads the target CSV, and for each row obtains the timestamp, then looks up corresponding data from the map.
  - If any timestamp in the target file is missing in the secondary data, it logs an error and halts execution.
- **Output Generation:**
  - Combines each matching row using a helper `combine` method (joining tokens with `;`) and writes them line by line to the output file.

---

## 4. Downsampler.java

**Purpose:**  
Reduces the temporal resolution of time series data by aggregating or selecting representative readings over uniform intervals (typically hourly).

**Key Functionalities:**

- **Parameter Parsing:**
  - Expects five arguments: input CSV file, start timestamp, end timestamp, interval (in milliseconds), and output filename.
- **Target Interval Mapping:**
  - Constructs two `TreeMap` objects (`readingsOne` and `readingsTwo`) that pre-populate keys for every interval between the start and end timestamps.
- **Reading and Assignment:**
  - Reads the CSV file using `CSVReader` (with `;` as delimiter) and parses each timestamp.
  - For each timestamp, it uses a rounding mechanism to determine the closest interval (calculating differences to the nearest lower and higher hour using the `Calendar` class).
  - If the new reading is closer (or if no previous reading exists), it overwrites the stored value for that interval.
- **Output Generation:**
  - Writes the header and then iterates over the defined intervals, outputting each timestamp alongside the corresponding downsampled values formatted to three decimal places.

---

## 5. WeatherForecastImporter.java

**Purpose:**  
Fetches weather forecast data from a MySQL database and compiles it into a unified CSV format for subsequent analysis or ingestion.

**Key Functionalities:**

- **Database Connectivity:**
  - Establishes a connection to a MySQL database using JDBC. Connection details (URL, username, and password) are hardcoded.
- **Data Extraction:**
  - Executes SQL queries to extract two sets of data:
    - Temperature readings from the table `aarslev_uc55_d0`.
    - Sun radiation values from the table `aarslev_uc56_d0`.
  - Uses timestamps as the key to merge the two data sets.
- **Data Cleaning:**
  - For temperature data: Converts the special value `-999` to `null` and formats valid readings to three decimal places.
  - For sun radiation values: Checks for negative values and assigns `null` if detected, otherwise formats to three decimal places.
- **Output Generation:**
  - Aggregates all data into a `TreeMap` keyed by timestamp, converts it into a list of string arrays, and finally writes each row to an output CSV file using a helper method that joins array elements with a semicolon.

---

## 6. CSVReader.java

**Purpose:**  
A lightweight CSV parser utility that supports custom delimiters and quote characters.

**Key Functionalities:**

- **Customization:**
  - Provides several constructors to set the separator (default is `,`), quote character (default is `"`), and optional lines to skip.
- **Parsing Mechanism:**
  - Reads lines from a given `Reader` via a `BufferedReader` and splits them into tokens.
  - Handles quoted fields, including multi-line quoted fields and escaped quotes (by checking for duplicate quote characters).
- **Utility Methods:**
  - `readNext()`: Reads and parses the next line into an array of strings.
  - `readAll()`: Reads the entire CSV file into a list of string arrays.
- **Resource Management:**
  - Provides a `close()` method to terminate the underlying stream.

---

## 7. Reading.java

**Purpose:**  
A simple data-holder (POJO) for storing a time-series reading.

**Key Functionalities:**

- **Attributes:**
  - Stores a timestamp (as a double) and its corresponding value.
- **Access Methods:**
  - Getter methods for timestamp and value.
- **Overridden toString():**
  - Formats the reading as `<timestamp in long>: <value>`, facilitating logging and debugging.

---

## 8. README.md

**Content Overview:**  
The README file provides usage instructions and an overview of the workflow for converting thesis PDFs to Markdown. Although not directly related to CSV processing, it reflects the broader context in which these tools might be employed. The instructions cover:

- **Docker-based Execution:**  
  Steps for building a Docker image, setting up directories, and running converters.
- **Local Installation:**  
  Directions to set up a Python virtual environment and install dependencies using `uv venv` and `uv pip install`.
- **Feature Summary:**  
  Highlights features such as structure preservation, progress tracking, and error handling.

---

## Usage Context and Workflow

The tools in the **SenmaticDataTools-master** suite are typically orchestrated in the following workflow:

1. **Export and Conversion:**  
   - Initially, Excel sheets are exported as CSV files.
   - The `MultipleColumnToOneCsv` tool is used to reformat CSV files that have multiple column groups.

2. **Data Cleaning:**  
   - The `SenmaticTosMAPCleaner` cleans the CSV files by handling missing values, formatting numbers, and filtering out invalid rows.

3. **Data Merging:**  
   - The `DataCombiner` merges cleaned data with target datasets (typically with timestamps) to create a unified dataset.

4. **Downsampling:**  
   - The `Downsampler` reduces the dosages of high-frequency time series data into aggregated hourly readings.

5. **Weather Data Import:**  
   - The `WeatherForecastImporter` pulls weather forecasts from a database, cleans them, and exports to CSV, providing supplementary data for further analysis or modeling.

6. **Parsing and Utility:**  
   - The `CSVReader` and `Reading` classes support all file parsing and in-memory data handling across the processing tools.

Together, these tools automate the transformation of raw, heterogeneous CSV data into a structured, clean format suitable for climate control systems, analysis, and forecasting.

---
