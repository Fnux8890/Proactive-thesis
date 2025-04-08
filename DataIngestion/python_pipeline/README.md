# Python Data Cleaning Pipeline

This directory contains a Python-based pipeline for cleaning and processing data, primarily using the Pandas library.

## Conceptual Pipeline Structure

A typical approach involves writing Python scripts that leverage the Pandas library for data manipulation, along with standard libraries for file system interaction.

### 1. Core Structure

```python
import pandas as pd
import json
import glob # For finding files
import os   # For path manipulation (consider using pathlib for modern Python)
from pathlib import Path
# Potentially libraries for database connection like sqlalchemy or psycopg2
# import sqlalchemy
# import psycopg2

# --- Configuration ---
DATA_DIRECTORY = Path(os.getenv('DATA_SOURCE_PATH', 'path/to/your/data/')) # Use env var or default
OUTPUT_DIRECTORY = Path('path/to/cleaned_data/')
DB_CONNECTION_STRING = os.getenv('DATABASE_URL') # Example for DB connection

# --- File Discovery ---
def find_data_files(data_dir):
    \"\"\"Finds CSV and JSON files in the specified directory.\"\"\"
    csv_files = list(data_dir.glob('*.csv'))
    json_files = list(data_dir.glob('*.json'))
    print(f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files.")
    return csv_files, json_files

# --- Loading Functions ---
def load_csv(file_path):
    \"\"\"Loads a CSV file into a pandas DataFrame, attempting to sniff the separator.\"\"\"
    try:
        # Leverage pandas' sniffing capabilities
        # Refer: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='warn')
        print(f"Successfully loaded {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_json(file_path):
    \"\"\"Loads a JSON file (assuming JSON Lines format for this example).\"\"\"
    try:
        # For JSON Lines format
        df = pd.read_json(file_path, lines=True)
        # For complex/nested JSON, load with json lib and normalize
        # See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html
        # with open(file_path, 'r') as f:
        #     data = json.load(f)
        # df = pd.json_normalize(data) # Adjust based on JSON structure
        print(f"Successfully loaded {file_path}")
        return df
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# --- Cleaning Function(s) ---
def clean_data(df, file_path):
    \"\"\"Applies various data cleaning steps to the DataFrame.\"\"\"
    print(f"Cleaning data from {file_path}...")
    original_rows = len(df)

    # Make a copy to avoid modifying the original DataFrame in place unexpectedly
    df_cleaned = df.copy()

    # 1. Drop Unnecessary Columns (Example)
    # Based on Real Python article: https://realpython.com/python-data-cleaning-numpy-pandas/
    # columns_to_drop = ['Edition Statement', 'Corporate Author'] # Example columns
    # df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors='ignore')

    # 2. Handle Missing Values (example: fill numerical with median, categorical with mode)
    for col in df_cleaned.select_dtypes(include=['number']).columns:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_val)
    for col in df_cleaned.select_dtypes(include=['object', 'category']).columns:
        mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
        df_cleaned[col] = df_cleaned[col].fillna(mode_val)

    # 3. Correct Data Types (example: convert column to datetime)
    # Using errors='coerce' will turn unparseable dates into NaT (Not a Time)
    if 'timestamp_col' in df_cleaned.columns:
         df_cleaned['timestamp_col'] = pd.to_datetime(df_cleaned['timestamp_col'], errors='coerce')
         # Optionally drop rows where conversion failed:
         # df_cleaned = df_cleaned.dropna(subset=['timestamp_col'])

    # Example: Clean publication date using regex (from Real Python)
    # Assumes a 'Date of Publication' column exists
    if 'Date of Publication' in df_cleaned.columns:
        # Ensure the column is string first
        df_cleaned['Date of Publication'] = df_cleaned['Date of Publication'].astype(str)
        # Extract first 4 digits using regex: ^(\d{4})
        # See: https://realpython.com/python-data-cleaning-numpy-pandas/#tidying-up-fields-in-the-data
        extr = df_cleaned['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
        df_cleaned['Date of Publication'] = pd.to_numeric(extr, errors='coerce') # Convert to numeric, NaNs on failure


    # 4. Handle Outliers (example: clipping based on IQR)
    # ... implementation needed ...

    # 5. Remove Duplicates (example: based on a subset of columns)
    # See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
    # subset_cols = ['timestamp_col', 'sensor_id'] # Example
    # if all(col in df_cleaned.columns for col in subset_cols):
    #     df_cleaned = df_cleaned.drop_duplicates(subset=subset_cols, keep='first')

    # 6. Data Formatting / Standardization (example: strip whitespace from string columns)
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        df_cleaned[col] = df_cleaned[col].str.strip()

    print(f"Cleaning finished. Original rows: {original_rows}, Cleaned rows: {len(df_cleaned)}")
    return df_cleaned

# --- Saving Function ---
def save_data(df, output_dir, original_filename):
    \"\"\"Saves the cleaned DataFrame to a Parquet file.\"\"\"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"cleaned_{original_filename}.parquet"
    try:
        # Parquet is often more efficient for storage and downstream processing
        df.to_parquet(output_filename, index=False)
        print(f"Saved cleaned data to {output_filename}")
    except Exception as e:
        print(f"Error saving data to {output_filename}: {e}")

# --- Main Execution Logic ---
def main():
    csv_files, json_files = find_data_files(DATA_DIRECTORY)
    all_files = csv_files + json_files
    processed_files = 0

    for file_path in all_files:
        print(f"--- Processing {file_path.name} ---")
        df = None
        if file_path.suffix == '.csv':
            df = load_csv(file_path)
        elif file_path.suffix == '.json':
            df = load_json(file_path)
        # Add elif for other file types like .xlsx using pd.read_excel

        if df is not None and not df.empty:
            cleaned_df = clean_data(df, file_path.name)
            if cleaned_df is not None and not cleaned_df.empty:
                 save_data(cleaned_df, OUTPUT_DIRECTORY, file_path.stem)
                 processed_files += 1
            else:
                print(f"Skipping saving for {file_path.name} due to empty DataFrame after cleaning.")
        else:
            print(f"Skipping processing for {file_path.name} due to loading error or empty file.")

    print(f"--- Pipeline finished. Processed {processed_files}/{len(all_files)} files. ---")

if __name__ == "__main__":
    main()

```

This structure provides modularity. You would create a main Python script (e.g., `pipeline.py`) containing this logic.

### 2. Handling CSV/JSON Specifics

**CSV Handling (`pandas.read_csv`)**

*   **Delimiter Detection**: `sep=None, engine='python'` is a good starting point for auto-detection. Specify `sep=','` or `delimiter=';'` if needed.
*   **Common Parameters**:
    *   `header=0`: First row is header. `None` if no header.
    *   `names=['col1', 'col2']`: Provide names if no header.
    *   `encoding='utf-8'`: Specify file encoding if not standard (e.g., 'latin1').
    *   `on_bad_lines='warn'`: How to handle rows with incorrect field counts ('error', 'skip').
    *   `skiprows=N`: Skip initial rows.
    *   `comment='#':` Character indicating comment lines.
    *   `parse_dates=['col_name']`: Attempt date parsing during load.
    *   `dtype={'col_id': str}`: Force column types (e.g., preserve leading zeros in IDs).
    *   `low_memory=False`: Can help with mixed-type inference issues in large files, but uses more memory.

**JSON Handling (`json` module, `pandas.read_json`, `pandas.json_normalize`)**

*   **Loading**:
    *   **JSON Lines**: `pd.read_json(path, lines=True)` where each line is a valid JSON object.
    *   **Standard/Nested JSON**: Load with `json.load(file_object)` into Python dicts/lists first.
*   **Flattening**: Use `pandas.json_normalize(python_dict_or_list)` to convert nested structures into a flat DataFrame. This is very powerful for complex JSON.
*   **Cleaning**:
    *   Handle `KeyError` during manual parsing or `NaN` columns after `json_normalize`.
    *   Wrap loading in `try...except json.JSONDecodeError`.
    *   Check `dtypes` and use `astype()` for corrections.

### 3. Running in Docker

(See `Dockerfile` and `docker-compose.yml` in this directory). The setup uses `uv` for faster dependency management within the container.

### References

*   **Pandas `read_csv`**: [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
*   **Pandas `read_json`**: [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html)
*   **Pandas `json_normalize`**: [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html)
*   **Real Python Data Cleaning**: [https://realpython.com/python-data-cleaning-numpy-pandas/](https://realpython.com/python-data-cleaning-numpy-pandas/)
*   **UV (Package Manager)**: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

---

*This file provides an overview. The actual implementation would be in a `.py` script.* 