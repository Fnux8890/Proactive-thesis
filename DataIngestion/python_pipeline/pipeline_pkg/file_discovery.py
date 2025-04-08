"""Handles discovering data files in the source directory."""

import logging
from pathlib import Path
from typing import List, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileFinder:
    """Scans a directory for files of specified types."""

    def __init__(self, supported_types: List[str]):
        """Initializes the FileFinder.

        Args:
            supported_types: A list of file extensions (e.g., ['.csv', '.json'])
                           to search for.
        """
        if not supported_types:
            raise ValueError("supported_types list cannot be empty.")
        self.supported_types = [t.lower() for t in supported_types]
        logger.info(f"FileFinder initialized for types: {self.supported_types}")

    def find_files(self, data_dir: Path) -> List[Path]:
        """Finds all files matching the supported types in the directory.

        Args:
            data_dir: The directory Path object to search in.

        Returns:
            A list of Path objects for the found files.
        """
        if not data_dir.is_dir():
            logger.error(f"Data directory not found or is not a directory: {data_dir}")
            return []

        found_files = []
        for file_type in self.supported_types:
            pattern = f"**/*{file_type}"
            files = list(data_dir.rglob(pattern))
            logger.info(f"Found {len(files)} files matching '{pattern}' recursively in {data_dir}")
            found_files.extend(files)

        logger.info(f"Total relevant files found: {len(found_files)}")
        return found_files 