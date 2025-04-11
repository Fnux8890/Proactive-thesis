# Data Metadata Extraction

This script extracts metadata from CSV files located in the `../../Data` directory (relative to this script) and its subdirectories (`aarslev`, `knudjepsen`).

## Setup

1. **Create a virtual environment:**

    ```powershell
    uv venv
    ```

2. **Activate the virtual environment:**
    - Windows (Powershell):

        ```powershell
        .venv\Scripts\Activate.ps1
        ```

    - macOS/Linux (Bash/Zsh):

        ```bash
        source .venv/bin/activate
        ```

    *(Note: No packages are needed for this script as it uses standard Python libraries, but activating the environment is good practice.)*

## Usage

Run the script from within the `Jupyter/data_analysis` directory:

```powershell
python extract_metadata.py
```

The script will:

- Scan `../../Data/aarslev/` and `../../Data/knudjepsen/` for `.csv` files.
- Extract metadata like file path, row count, column count, headers, and min/max timestamps (if found).
- Save the extracted metadata to a file named `metadata_output.txt` in the current directory (`Jupyter/data_analysis`).
