#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "pandas",
#     "xlrd",
#     "typer",
#     "rich",
#     "pathlib"
# ]
# ///

import pandas as pd
from pathlib import Path
import typer
from rich.console import Console
from typing import Optional

def convert_file(file_path: Path) -> None:
    """
    Convert an Excel (.xls) file to CSV.
    The CSV file will be saved in the same directory with the same basename.
    
    Args:
        file_path (Path): The path to the Excel (.xls) file.
    """
    console = Console()
    try:
        # Read the Excel file using the xlrd engine (required for .xls files)
        df = pd.read_excel(file_path, engine="xlrd")
    except Exception as e:
        console.print(f"[red]Error reading file {file_path}: {e}[/red]")
        raise

    # Generate the output CSV file path with the same base name and in the same directory
    csv_file: Path = file_path.with_suffix('.csv')
    try:
        df.to_csv(csv_file, index=False)
        console.print(f"[green]Converted {file_path} -> {csv_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error writing CSV file {csv_file}: {e}[/red]")
        raise

def main(
    input_path: Path = typer.Argument(
        ...,
        help="Path to an Excel (.xls) file or directory to search recursively for .xls files."
    )
) -> None:
    """
    Convert Excel (.xls) files to CSV format.
    
    If a file path is provided, it is converted directly.
    If a directory is provided, all .xls files in the directory (including subdirectories)
    will be converted.
    
    Args:
        input_path (Path): Path to an Excel file or a directory containing Excel files.
    """
    console = Console()

    # Attempt to resolve the input path relative to current working directory.
    resolved_input = input_path.resolve()
    if not resolved_input.exists():
        # If not found, attempt to resolve relative to the repository root (one parent above the script)
        alt_input = (Path(__file__).parent.parent / input_path).resolve()
        if alt_input.exists():
            resolved_input = alt_input
            console.print(f"[yellow]Input path not found in current directory; using alternate path: {resolved_input}[/yellow]")
    input_path = resolved_input

    if input_path.is_file():
        if input_path.suffix.lower() != ".xls":
            console.print(f"[yellow]{input_path} doesn't have a .xls suffix. Skipping conversion.[/yellow]")
            raise typer.Exit(code=1)
        convert_file(input_path)
    elif input_path.is_dir():
        # Use rglob to search recursively for all .xls files under the given directory.
        xls_files = list(input_path.rglob("*.xls"))
        if not xls_files:
            console.print(f"[red]No .xls files found in {input_path}.[/red]")
            raise typer.Exit(code=1)
        for file in xls_files:
            try:
                convert_file(file)
            except Exception as e:
                console.print(f"[red]Failed converting {file}: {e}[/red]")
    else:
        console.print(f"[red]Invalid input path: {input_path}[/red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    typer.run(main)
