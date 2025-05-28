#!/usr/bin/env python3
"""
PDF to Markdown Converter using Microsoft's MarkItDown.
This script converts PDF files — by default located in the 'data' folder — to markdown format.
It supports processing a single PDF file or all PDF files in a directory,
either sequentially or concurrently.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Any
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from markitdown import MarkItDown


class PDFToMarkdownConverter:
    def __init__(self, input_file: Path, output_file: Optional[Path] = None) -> None:
        """Initialize the PDF converter.

        Args:
            input_file (Path): Path to the input PDF file.
            output_file (Optional[Path]): Path for the output markdown file.
                Defaults to "output/<inputfile>.md" if not provided.
        """
        self.input_file: Path = input_file.resolve(strict=True)
        if output_file is None:
            self.output_file: Path = Path("output") / self.input_file.with_suffix('.md').name
        else:
            self.output_file = output_file
        self.console: Console = Console()

    def convert_to_markdown(self, use_progress: bool = True) -> None:
        """Convert PDF content to markdown format using MarkItDown.

        Args:
            use_progress (bool): Whether to show an individual progress indicator.
                Use False when running multiple conversions concurrently.
        """
        try:
            md_converter = MarkItDown()
            if use_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    task = progress.add_task("[cyan]Converting PDF to Markdown...", total=None)
                    result: Any = md_converter.convert(str(self.input_file))
                    self.output_file.write_text(result.text_content, encoding="utf-8")
                    progress.update(task, completed=True)
            else:
                result: Any = md_converter.convert(str(self.input_file))
                self.output_file.write_text(result.text_content, encoding="utf-8")
            self.console.print(
                f"[green]Successfully converted {self.input_file} to {self.output_file}"
            )
        except Exception as e:
            self.console.print(f"[red]Error during conversion: {e}")
            raise

def process_single_pdf(file_path: Path, dest_dir: Path, use_progress: bool = False) -> None:
    """Process a single PDF file and convert it to Markdown.

    Args:
        file_path (Path): The PDF file to convert.
        dest_dir (Path): Destination directory for the converted Markdown file.
        use_progress (bool): Whether to show an individual progress indicator.
            Defaults to False.
    """
    output_file: Path = dest_dir / file_path.with_suffix(".md").name
    converter = PDFToMarkdownConverter(file_path, output_file)
    converter.convert_to_markdown(use_progress=use_progress)

def main(
    input_path: Path = typer.Argument(
        Path("data"),
        help="Path to a PDF file or a directory containing PDF files (default: 'data' folder)"
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output-path",
        "--output-file",
        help="Path for the output markdown file or directory. For a directory input, outputs are saved here."
    ),
    parallel: bool = typer.Option(
        False,
        help="Process PDF files concurrently if a directory is provided."
    )
) -> None:
    """Convert PDF files to Markdown format using MarkItDown.

    This function handles both single PDF file conversion and batch conversion
    for all PDF files in a directory. Batch processing can be done sequentially or concurrently.

    Args:
        input_path (Path): Path to a PDF file or directory containing PDF files. Defaults to the 'data' folder.
        output_path (Optional[Path]): Optional output file (for single file) or directory (for batch conversion).
        parallel (bool): Flag to process multiple PDF files concurrently.
    """
    console: Console = Console()
    if input_path.is_file():
        # Single file conversion: use the provided output_path or default location.
        converter = PDFToMarkdownConverter(input_path, output_path)
        converter.convert_to_markdown()
    elif input_path.is_dir():
        # Determine the destination directory.
        dest_dir: Path = output_path if output_path is not None else Path("output")
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Use rglob to search recursively through subdirectories for PDF files.
        pdf_files: List[Path] = sorted(input_path.rglob("*.pdf"))
        if not pdf_files:
            console.print(f"[red]No PDF files found in the directory: {input_path}")
            raise typer.Exit(1)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Converting {len(pdf_files)} PDF file(s)...", total=len(pdf_files)
            )
            if parallel:
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(process_single_pdf, pdf, dest_dir): pdf for pdf in pdf_files}
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            console.print(f"[red]Error processing {futures[future]}: {e}")
                        progress.advance(task)
            else:
                for pdf in pdf_files:
                    try:
                        process_single_pdf(pdf, dest_dir, use_progress=False)
                    except Exception as e:
                        console.print(f"[red]Error processing {pdf}: {e}")
                    progress.advance(task)
        console.print(f"[green]Successfully processed {len(pdf_files)} PDF file(s).")
    else:
        console.print(f"[red]The specified input path is neither a file nor a directory: {input_path}")
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)