# Thesis PDF to Markdown Converter

This script converts thesis PDF files to Markdown format using Microsoft's MarkItDown tool.

## Usage Options

### 1. Using Docker (Recommended)

The easiest way to run the converter is using Docker, which handles all dependencies automatically.

1. Build the Docker image:

```bash
docker compose build
```

2. Create a data directory for your PDFs:

```bash
mkdir -p data
```

3. Copy your PDF file to the data directory:

```bash
cp your-thesis.pdf data/
```

4. Run the converter:

```bash
docker compose run converter your-thesis.pdf
```

Or specify an output file:

```bash
docker compose run converter your-thesis.pdf --output-file output.md
```

### 2. Local Installation

If you prefer to run the script locally, you'll need:

#### Prerequisites

- Python 3.8 or higher
- UV package manager (already installed)

#### Installation

1. Create a virtual environment:

```bash
uv venv
```

2. Install the required packages:

```bash
uv pip install -r requirements.txt
```

#### Usage

Run the script using UV:

```bash
uv run python pdf_to_markdown.py thesis.pdf
```

Specify output file:

```bash
uv run python pdf_to_markdown.py thesis.pdf --output-file output.md
```

## Features

- Converts PDF documents to markdown format using Microsoft's MarkItDown
- Preserves document structure and formatting
- Progress indicators for conversion process
- Rich console output with color-coded status messages

## Output

The script will generate a markdown file containing:

- Converted text content
- Preserved document structure
- Formatted headings and sections

## Error Handling

The script includes comprehensive error handling and will:

- Display clear error messages for common issues
- Provide progress feedback during conversion
- Gracefully handle exceptions

## Dependencies

- markitdown: Microsoft's PDF to Markdown conversion tool
- typer: Command-line interface
- rich: Console output formatting

## Notes

- The conversion quality depends on the structure and formatting of the input PDF
- The script uses Microsoft's MarkItDown tool which handles the conversion internally
- When using Docker, place your PDF files in the `data` directory
- The Docker setup handles all dependencies automatically
