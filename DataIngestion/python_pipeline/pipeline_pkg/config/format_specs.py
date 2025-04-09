# DataIngestion/python_pipeline/pipeline_pkg/config/format_specs.py
import fnmatch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import logging
import os # Import os for getenv

logger = logging.getLogger(__name__)

# Define type aliases for clarity
LoaderParams = Dict[str, Any]
TimestampInfo = Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str]]] # (col, format) or ((date_col, date_fmt), (time_col, time_fmt))
ValueColSpec = Union[List[str], Dict[str, str], str, Dict[str, Tuple[str, str]], List[Union[str, Tuple[str, ...]]]]

# NOTE: This configuration is based on meta_data.md and requires verification
#       by inspecting the actual files, especially header structures and encodings.
FILE_FORMAT_SPECS: Dict[str, Dict[str, Any]] = {
    # --- Knudjepsen CSVs ---
    "Data/knudjepsen/NO3_LAMPEGRP_1.csv": {
        "type": "csv",
        "loader_params": {
            "sep": ";",
            "header": [0, 1, 2], # Multi-index header (Names, Sources, Units)
            "quoting": 0, # csv.QUOTE_MINIMAL (default for pandas)
            "decimal": ",",
            "encoding": "iso-8859-1", # Changed from utf-8
            "low_memory": False,
            "skipinitialspace": True,
        },
        "timestamp_handling": "knudjepsen_manual", # Special flag for custom timestamp parsing
        "cleaner": "KnudjepsenSourceUnitCleaner",
        "timestamp_info": ('', 'dd-mm-yyyy HH:MM:SS'), # First column, specific format
        "value_info": { # Define level meanings for the new cleaner
             'level_map': {0: 'measurement', 1: 'source', 2: 'unit'},
             'source_location_regex': r"Afd\s*(\d+)", # Regex to extract location number from source string
        },
        "base_source_id": "knudjepsen",
        "location": "NO3",
        "measurement_group": "LAMPEGRP_1",
    },
    "Data/knudjepsen/NO3NO4.extra.csv": {
        "type": "csv",
        "loader_params": {
            "sep": ";", "header": [0, 1, 2], "quoting": 0, "decimal": ",",
            "encoding": "iso-8859-1", # Changed from utf-8
            "low_memory": False, "skipinitialspace": True,
        },
        "timestamp_handling": "knudjepsen_manual", # Special flag for custom timestamp parsing
        "cleaner": "KnudjepsenExtraCsvCleaner", # Special cleaner for Afd 3/4 structure
        "timestamp_info": ('', 'dd-mm-yyyy HH:MM:SS'),
        # Values need special handling due to '[Afd 3]' in header
        "value_info": { # Define how to parse complex headers
             'level_map': {0: 'measurement', 1: 'unit', 2: 'location_suffix'},
             'location_prefix': 'Afd',
        },
         "base_source_id": "knudjepsen",
         "location": "NO3NO4",
         "measurement_group": "extra",
    },
    "Data/knudjepsen/NO4_LAMPEGRP_1.csv": { # Assume similar format to NO3_LAMPEGRP_1
        "type": "csv",
        "loader_params": {
            "sep": ";", "header": [0, 1, 2], "quoting": 0, "decimal": ",",
            "encoding": "iso-8859-1", # Changed from utf-8
            "low_memory": False, "skipinitialspace": True,
        },
        "timestamp_handling": "knudjepsen_manual",
        "cleaner": "KnudjepsenSourceUnitCleaner", # Use new cleaner
        "timestamp_info": ('', 'dd-mm-yyyy HH:MM:SS'),
        "value_info": { # Define level meanings for the new cleaner
             'level_map': {0: 'measurement', 1: 'source', 2: 'unit'},
             'source_location_regex': r"Afd\s*(\d+)", # Regex to extract location number from source string
        },
        "base_source_id": "knudjepsen",
        "location": "NO4", # Base location for the file
        "measurement_group": "LAMPEGRP_1",
    },
    "Data/knudjepsen/NO3-NO4_belysningsgrp.csv": {
        "type": "csv",
        "loader_params": {
            "sep": ";", "header": [0, 1], # Names, Sources (Units empty)
             "quoting": 0, "decimal": ",", # Decimal likely not needed but set for consistency
             "encoding": "iso-8859-1", # Changed from utf-8
             "low_memory": False, "skipinitialspace": True,
        },
        "timestamp_handling": "knudjepsen_manual", # Special flag for custom timestamp parsing
        "cleaner": "KnudjepsenBelysningCsvCleaner", # Needs specific logic for LAMPGRP columns
        "timestamp_info": ('', 'dd-mm-yyyy HH:MM:SS'),
         # Special handling for columns like 'målt status [LAMPGRP 1]'
         "value_info": {
             'level_map': {0: 'measurement_base', 1: 'group'},
             'group_prefix': 'LAMPGRP',
         },
         "base_source_id": "knudjepsen",
         "location": "NO3-NO4",
         "measurement_group": "belysningsgrp",
    },

    # --- Aarslev CSVs ---
    # Pattern for MortenSDUData files
    "Data/aarslev/*/MortenSDUData.csv": {
        "type": "csv",
        "loader_params": {
            "sep": ",", "header": 0, "quoting": 3, # QUOTE_NONE
            "decimal": ".", "encoding": "utf-8", # Verify encoding
            "low_memory": False, "skipinitialspace": True,
        },
        "cleaner": "AarslevMortenCsvCleaner",
        "timestamp_info": ('Start', '%Y-%m-%d %H:%M'), # Note format code difference
        # 'End' column might indicate aggregation period, ignore for now
        "value_info": {
            "regex_pattern": r"^(Celle \d+\..*) \[(.*?)\]$", # Capture (Name) [Unit]
            "column_name_group": 1,
            "unit_group": 2,
        },
        "source_id_pattern": r"Data/aarslev/([^/]+)/MortenSDUData.csv", # Capture month/folder name
        "base_source_id": "aarslev_morten",
        "location_pattern": r"(Celle \d+)\..*", # Extract 'Celle X' from column name
    },
    "Data/aarslev/weather_jan_feb_2014.csv": {
        "type": "csv",
        "loader_params": {
            "sep": ";", "header": 0, "quoting": 3, "decimal": ".",
            "encoding": "utf-8", "low_memory": False, "skipinitialspace": True,
        },
        "cleaner": "AarslevSimpleCsvCleaner",
        "timestamp_info": ('timestamp', 'unix_ms'),
        "value_columns": ['temperature', 'sun_radiation'],
        "units": {'temperature': 'C', 'sun_radiation': 'W/m2'}, # Provided separately
        "base_source_id": "aarslev",
        "location": "weather_station", # Assume general location
        "measurement_group": "weather_jan_feb_2014",
    },
    # Pattern for celle output files
    "Data/aarslev/celle*/output-*.csv": {
        "type": "csv",
        "loader_params": {
            "sep": ";", "header": 0, "quoting": 0, # Headers are quoted, pandas handles this
            "decimal": ",", # Verify decimal separator (meta says quoted, check actual data)
            "encoding": "utf-8", # Often used with Danish data, verify
            "low_memory": False, "skipinitialspace": True,
        },
        "cleaner": "AarslevCelleCsvCleaner",
        "timestamp_info": (('Date', '%Y-%m-%d'), ('Time', '%H:%M:%S')), # Separate Date/Time
        # Column names are like '"Celle 5: Lufttemperatur"' - need cleaning
        "value_info": {
            "regex_pattern": r'"(Celle \d+: .*)"+', # Extract meaningful part
            "column_name_group": 1,
            # Units might need a separate map based on the extracted name
        },
        "unit_map_pattern": { # Map extracted name patterns to units
             r'Celle \d+: Lufttemperatur': 'C',
             r'Celle \d+:.*[Ee]ndelig.*[Vv]arme.*': 'C', # More flexible for 'Endelig fælles varme sætpunkt'
             r'Celle \d+:.*[Vv]arme.*[Ss]ætpunkt': 'C', # Alternative pattern for varme sætpunkt
             r'Celle \d+:.*[Vv]ent.*position.*\d': '%', # More flexible for ventilation positions
             r'Celle \d+:.*RH%': '%RH', # Luftfugtighed RH%
             r'Celle \d+:.*VPD': 'kPa', # Luftfugtighed VPD
             r'Celle \d+:.*CO2$': 'ppm', # Just CO2
             r'Celle \d+:.*CO2.*krav': 'ppm', # CO2 krav
             r'Celle \d+:.*CO2.*dosering': '%', # CO2 dosering
             r'Celle \d+:.*[Ll]ys.*[Ii]ntensitet': 'µmol/m²/s', # Lys intensitet
             r'Celle \d+:.*[Gg]ardin.*position': '%', # Gardin position
             r'Celle \d+:.*[Ss]olindstråling': 'W/m2', # Solindstråling
             r'Celle \d+:.*[Uu]detemperatur': 'C', # Udetemperatur
             r'Celle \d+:.*[Ff]low.*[Tt]emperatur': 'C', # Flow temperatur
             # Catch-all for any temperature-related columns
             r'Celle \d+:.*[Tt]emp.*': 'C',
             # Catch-all with cell number, for everything else
             r'Celle \d+:.*': 'Unknown', # Default unit for anything with Celle prefix
        },
        "source_id_pattern": r"Data/aarslev/(celle\d+)/output-.*\.csv", # Extract 'celleX'
        "base_source_id": "aarslev",
        "location_pattern": r"(Celle \d+):.*", # Extract 'Celle X' from cleaned column name
    },
    "Data/aarslev/data_jan_feb_2014.csv": { # Similar structure to weather
        "type": "csv",
        "loader_params": {
            "sep": ";", "header": 0, "quoting": 3, "decimal": ".",
            "encoding": "utf-8", "low_memory": False, "skipinitialspace": True,
        },
        "cleaner": "AarslevSimpleCsvCleaner",
        "timestamp_info": ('timestamp', 'unix_ms'),
        "value_columns": ['temperature_forecast', 'sun_radiation_forecast', 'sun_radiation', 'temperature'],
        "units": {'temperature': 'C', 'sun_radiation': 'W/m2', 'temperature_forecast': 'C', 'sun_radiation_forecast': 'W/m2'},
        "base_source_id": "aarslev",
        "location": "forecast_observation", # Indicate source type
        "measurement_group": "data_jan_feb_2014",
    },
    "Data/aarslev/winter2014.csv": { # Assuming same format as data_jan_feb_2014
        "type": "csv",
        "loader_params": {
            "sep": ";", "header": 0, "quoting": 3, "decimal": ".",
            "encoding": "utf-8", "low_memory": False, "skipinitialspace": True,
        },
        "cleaner": "AarslevSimpleCsvCleaner",
        "timestamp_info": ('timestamp', 'unix_ms'), # Assumption
        "value_columns": ['temperature_forecast', 'sun_radiation_forecast', 'sun_radiation', 'temperature'], # Assumption
        "units": {'temperature': 'C', 'sun_radiation': 'W/m2', 'temperature_forecast': 'C', 'sun_radiation_forecast': 'W/m2'}, # Assumption
        "base_source_id": "aarslev",
        "location": "forecast_observation", # Assumption
        "measurement_group": "winter2014",
    },
    "Data/aarslev/temperature_sunradiation_jan_feb_2014.json.csv": {
        "type": "csv",
        "loader_params": {
            "sep": ";", "header": 0, "quoting": 3, "decimal": ".",
            "encoding": "utf-8", "low_memory": False, "skipinitialspace": True,
        },
        "cleaner": "AarslevUuidHeaderCsvCleaner", # Special cleaner for UUID headers
        "timestamp_info": ('timestamp', 'unix_ms'),
        "uuid_map": { # Map UUIDs from header to measurement names/units
            '5f893ebd-002c-3708-8e76-bb519336f210': ('sun_radiation', 'W/m2'), # Verify units
            'be3ddbb8-5210-3be3-84d8-904df2da1754': ('temperature', 'C'), # Verify units
        },
        "base_source_id": "aarslev",
        "location": "tempsun_json_export", # Indicate source
        "measurement_group": "jan_feb_2014",
    },

    # --- Aarslev JSONs ---
    "Data/aarslev/temperature_sunradiation_jan_feb_2014.json": {
        "type": "json",
        "loader_params": {
            # Standard JSON list loading, cleaner handles structure
            "orient": None, # Let cleaner handle parsing
        },
        "cleaner": "AarslevStreamListJsonCleaner",
        "structure": "list_of_streams",
         # Cleaner extracts timestamp, value, uuid, units, source name from structure
         "base_source_id": "aarslev",
         "location": "tempsun_json_import", # Indicate source
         "measurement_group": "jan_feb_2014",
    },
    # Pattern for celle JSON files
    "Data/aarslev/celle*/output-*.csv.json": {
        "type": "json",
        "loader_params": {
            # Standard JSON dict loading, cleaner handles structure
            "orient": None,
        },
        "cleaner": "AarslevStreamDictJsonCleaner",
        "structure": "dict_of_streams",
         # Cleaner extracts timestamp, value, key (sensor path), uuid, units, source name
         "base_source_id": "aarslev",
         "source_id_pattern": r"Data/aarslev/(celle\d+)/output-.*\.csv\.json", # Extract 'celleX'
         "location_pattern": r"/(Cell\d+)/.*", # Extract 'CellX' from key path
    }
}


def get_format_spec(file_path: Path) -> Optional[Dict[str, Any]]:
    """Finds the matching format specification for a given file path.

    Handles absolute paths (like those within Docker) by attempting to
    make them relative to a potential known base directory pattern ('Data/').
    Matches based on exact relative path first, then patterns using pathlib.Path.match.
    Paths are converted to POSIX format for consistent matching.

    Args:
        file_path: The Path object for the file.

    Returns:
        A dictionary containing the format specification, including the
        matched pattern and full path, or None if no match is found.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    # Define the known base path inside the container
    container_base_path_str = os.getenv('DATA_SOURCE_PATH', '/app/data') # Get from env var used by pipeline
    container_base_path = Path(container_base_path_str).resolve()
    absolute_file_path = file_path.resolve()
    logger.debug(f"Attempting to find format spec for absolute path: {absolute_file_path}")
    logger.debug(f"Using container base path for comparison: {container_base_path}")

    # Try to make the file path relative to the container base path
    try:
        relative_path = absolute_file_path.relative_to(container_base_path)
        # Construct the path string expected by the spec keys (e.g., "Data/aarslev/...")
        relative_path_str = Path("Data").joinpath(relative_path).as_posix()
        logger.debug(f"Generated relative path string for matching: {relative_path_str}")
    except ValueError:
        # If the path is not relative to the base path, use its original form
        relative_path_str = file_path.as_posix()
        logger.warning(f"Path {absolute_file_path} is not relative to base {container_base_path}. Using original: {relative_path_str}")


    # --- Match against FILE_FORMAT_SPECS using relative_path_str ---

    # Convert the relative path string to a Path object for matching
    relative_path_obj = Path(relative_path_str)

    # Check for exact matches first (using the string representation)
    if relative_path_str in FILE_FORMAT_SPECS:
        spec = FILE_FORMAT_SPECS[relative_path_str].copy()
        spec['matched_pattern'] = relative_path_str
        spec['full_path'] = file_path # Store original absolute path
        logger.debug(f"Found exact match: {relative_path_str}")
        return spec

    # Check for pattern matches using Path.match
    sorted_patterns = sorted(FILE_FORMAT_SPECS.keys(), key=len, reverse=True)
    logger.debug(f"Checking patterns for relative path object: {relative_path_obj}")

    for pattern_str in sorted_patterns:
        logger.debug(f"Comparing path '{relative_path_obj}' against pattern string: {pattern_str}")
        if isinstance(pattern_str, str) and ("*" in pattern_str or "?" in pattern_str):
             # Use Path.match for comparison
             try:
                 match_result = relative_path_obj.match(pattern_str)
                 logger.debug(f"Path.match result for pattern '{pattern_str}': {match_result}")
                 if match_result:
                     spec = FILE_FORMAT_SPECS[pattern_str].copy()
                     spec['matched_pattern'] = pattern_str
                     spec['full_path'] = file_path # Store original absolute path
                     # Log match success clearly
                     logger.info(f"Found pattern match using Path.match: Pattern='{pattern_str}', Path='{relative_path_obj}'")
                     return spec
             except Exception as e:
                 # Log potential errors during matching (e.g., invalid pattern?)
                 logger.error(f"Error during Path.match for pattern '{pattern_str}' and path '{relative_path_obj}': {e}")
        else:
             logger.debug(f"Skipping pattern (not string or no wildcard): {pattern_str}")


    logger.warning(f"No format specification found for file: {file_path} (tried matching path: {relative_path_obj} against defined specs)")
    return None 