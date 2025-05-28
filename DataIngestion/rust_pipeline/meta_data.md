### File: Data/knudjepsen/NO3_LAMPEGRP_1.csv

*   **Delimiter:** Semicolon (`;`)
*   **Quoting:** Double quotes (`"`)
*   **Header Rows:** 3 (Names, Sources, Units)
*   **Columns & Assumed Data Format:**
    *   `""` (Timestamp): `dd-mm-yyyy HH:MM:SS` string
    *   `mål temp afd  Mid` (°C): Decimal (comma separator)
    *   `udetemp` (°C): Decimal (comma separator)
    *   `stråling` (W/m²): Integer
    *   `CO2 målt` (ppm): Integer
    *   `mål gard 1` (%): Decimal (comma separator)
    *   `mål gard 2` (%): Decimal (comma separator)
    *   `målt status` (""): Integer (likely boolean 0/1)
    *   `mål FD` (g/m3): Decimal (comma separator)
    *   `mål rør 1` (°C): Decimal (comma separator)
    *   `mål rør 2` (°C): Decimal (comma separator)

### File: Data/knudjepsen/NO3NO4.extra.csv

*   **Delimiter:** Semicolon (`;`)
*   **Quoting:** Double quotes (`"`)
*   **Header Rows:** 3 (Names, Sources, Units)
*   **Columns & Assumed Data Format:**
    *   `""` (Timestamp): `dd-mm-yyyy HH:MM:SS` string
    *   `mål FD` (g/m3) [Afd 3]: Decimal (comma separator)
    *   `mål RF` (%) [Afd 3]: Integer or Decimal (comma separator)
    *   `mål læ` (%) [Afd 3]: Integer
    *   `mål vind` (%) [Afd 3]: Integer
    *   `mål FD` (g/m3) [Afd 4]: Decimal (comma separator)
    *   `mål RF` (%) [Afd 4]: Decimal (comma separator)
    *   `mål læ` (%) [Afd 4]: Integer
    *   `mål vind` (%) [Afd 4]: Integer

### File: Data/knudjepsen/NO4_LAMPEGRP_1.csv

*   **Delimiter:** Semicolon (`;`)
*   **Quoting:** Double quotes (`"`)
*   **Header Rows:** 3 (Names, Sources, Units)
*   **Columns & Assumed Data Format:**
    *   `""` (Timestamp): `dd-mm-yyyy HH:MM:SS` string
    *   `mål temp afd  Mid` (°C): Decimal (comma separator)
    *   `udetemp` (°C): Decimal (comma separator)
    *   `stråling` (W/m²): Integer
    *   `CO2 målt` (ppm): Decimal (comma separator)
    *   `mål gard 1` (%): Decimal (comma separator)
    *   `mål gard 2` (%): Decimal (comma separator)
    *   `målt status` (""): Integer (likely boolean 0/1)
    *   `mål FD` (g/m3): Integer or Decimal (comma separator)
    *   `mål rør 1` (°C): Integer or Decimal (comma separator)
    *   `mål rør 2` (°C): Decimal (comma separator)

### File: Data/knudjepsen/NO3-NO4_belysningsgrp.csv

*   **Delimiter:** Semicolon (`;`)
*   **Quoting:** Double quotes (`"`)
*   **Header Rows:** 3 (Names, Sources, Units)
*   **Columns & Assumed Data Format:**
    *   `""` (Timestamp): `dd-mm-yyyy HH:MM:SS` string
    *   `målt status` [LAMPGRP 1]: Integer (likely boolean 0/1)
    *   `målt status` [LAMPGRP 2]: Integer (likely boolean 0/1)
    *   `målt status` [LAMPGRP 3]: Integer (likely boolean 0/1)
    *   `målt status` [LAMPGRP 4]: Integer (likely boolean 0/1)
    *   `målt status` [LAMPGRP 1]: Integer (likely boolean 0/1)
    *   `målt status` [LAMPGRP 2]: Integer (likely boolean 0/1)

### File: Data/aarslev/september2014/MortenSDUData.csv (and likely similar files in `Data/aarslev/*/`)

*   **Delimiter:** Comma (`,`)
*   **Quoting:** None observed
*   **Header Rows:** 1
*   **Columns & Assumed Data Format:**
    *   `Start`: Timestamp string (`yyyy-mm-dd HH:MM`)
    *   `End`: Timestamp string (`yyyy-mm-dd HH:MM`)
    *   `Celle 6.RH Zone1 (Celle 6) [RH%]`: Decimal
    *   `Celle 6.CO2 (Celle 6) [ppm]`: Decimal
    *   `Celle 6.CO2 status (Celle 6) []`: Decimal (likely 0.0)
    *   `Celle 6.Luft temp (Celle 6) [°C]`: Decimal
    *   `Celle 6.Flow1 (Celle 6) [°C]`: Decimal
    *   *(Note: Column names might vary slightly, e.g., `Celle 5` instead of `Celle 6`)*

### File: Data/aarslev/weather_jan_feb_2014.csv

*   **Delimiter:** Semicolon (`;`)
*   **Quoting:** None observed
*   **Header Rows:** 1
*   **Columns & Assumed Data Format:**
    *   `timestamp`: Integer (Unix timestamp, likely milliseconds)
    *   `temperature`: Decimal
    *   `sun_radiation`: Decimal

### File: Data/aarslev/celle5/output-2014-01-01-00-00.csv (and likely similar files in `Data/aarslev/celle*/`)

*   **Delimiter:** Semicolon (`;`)
*   **Quoting:** Double quotes (`"`) around column names in header
*   **Header Rows:** 1
*   **Columns & Assumed Data Format:**
    *   `Date`: Date string (`yyyy-mm-dd`)
    *   `Time`: Time string (`HH:MM:SS`)
    *   `"Celle 5: Lufttemperatur"`: Decimal (quoted)
    *   `"Celle 5: Endelig fælles varme sætpunkt"`: Integer (quoted)
    *   `"Celle 5: Vent. position 1"`: Integer (quoted)
    *   `"Celle 5: Vent. position 2"`: Integer (quoted)
    *   `"Celle 5: Luftfugtighed RH%"`: Decimal (quoted)
    *   `"Celle 5: Luftfugtighed VPD"`: Decimal (quoted)
    *   `"Celle 5: CO2"`: Decimal (quoted)
    *   `"Celle 5: CO2 krav"`: Integer (quoted)
    *   `"Celle 5: CO2 dosering"`: Integer (quoted)
    *   `"Celle 5: "`: Integer (quoted, likely status/boolean)
    *   `"Celle 5: "`: Integer (quoted, likely status/boolean)
    *   `"Celle 5: Lys intensitet"`: Integer (quoted)
    *   `"Celle 5: Gardin 1 position"`: Integer (quoted)
    *   `"Celle 5: Solindstråling"`: Decimal (quoted)
    *   `"Celle 5: Udetemperatur"`: Decimal (quoted)
    *   `"Celle 5: Flow temperatur 1"`: Decimal (quoted)
    *   `"Celle 5: Flow temperatur 2"`: Decimal (quoted)
    *   *(Note: Column names might vary, e.g., `Celle 6` instead of `Celle 5`. Data might be empty or use a different format further down)*

### File: Data/aarslev/data_jan_feb_2014.csv

*   **Delimiter:** Semicolon (`;`)
*   **Quoting:** None observed
*   **Header Rows:** 1
*   **Columns & Assumed Data Format:**
    *   `timestamp`: Integer (Unix timestamp, likely milliseconds)
    *   `temperature_forecast`: Decimal
    *   `sun_radiation_forecast`: Decimal
    *   `sun_radiation`: Decimal
    *   `temperature`: Decimal

### File: Data/aarslev/winter2014.csv

*   *(Assuming similar format to `Data/aarslev/data_jan_feb_2014.csv` based on name, needs confirmation)*

### File: Data/aarslev/temperature_sunradiation_jan_feb_2014.json.csv

*   **Delimiter:** Semicolon (`;`)
*   **Quoting:** None observed
*   **Header Rows:** 1 (UUIDs: `5f893ebd-002c-3708-8e76-bb519336f210` and `be3ddbb8-5210-3be3-84d8-904df2da1754`)
*   **Columns & Assumed Data Format (based on filename & context):**
    *   `timestamp`: Integer (Unix timestamp, likely milliseconds)
    *   Column 2 (UUID: `5f893ebd-...`): Decimal (Assumed Sun Radiation)
    *   Column 3 (UUID: `be3ddbb8-...`): Decimal (Assumed Temperature)

### File: Data/aarslev/temperature_sunradiation_jan_feb_2014.json

*   **Format:** JSON List of Data Streams
*   **Structure:**
    *   List `[` ... `]` containing multiple stream dictionaries `{...}`.
    *   Each stream dictionary contains:
        *   `uuid` (string): Unique identifier for the stream.
        *   `Readings` (list): List of `[timestamp, value]` pairs.
            *   `timestamp`: Integer (Unix timestamp, likely milliseconds)
            *   `value`: Decimal (Temperature or Sun Radiation based on context)
        *   `Properties` (dictionary):
            *   `Timezone` (string): e.g., "Europe/Copenhagen"
            *   `UnitofMeasure` (string): e.g., "W/m2" or "C"
            *   `ReadingType` (string): e.g., "double"
        *   `Metadata` (dictionary):
            *   `SourceName` (string): e.g., "Aarslev"

### File: Data/aarslev/celle5/output-*.csv.json (and likely similar files in `Data/aarslev/celle*/`)

*   **Format:** JSON Dictionary of Data Streams
*   **Structure:**
    *   Dictionary `{` ... `}` where keys are path-like strings representing sensor names (e.g., `"/Cell5/air_temperature"`, `"/Cell5/co2"`).
    *   Each value associated with a key is a stream dictionary containing:
        *   `uuid` (string): Unique identifier for the stream.
        *   `Readings` (list): List of `[timestamp, value]` pairs.
            *   `timestamp`: Integer (Unix timestamp, likely milliseconds)
            *   `value`: Decimal or Integer (depends on the specific sensor, units in `Properties`)
        *   `Properties` (dictionary):
            *   `Timezone` (string): e.g., "Europe/Copenhagen"
            *   `UnitofMeasure` (string): e.g., "C", "ppm", "%*100"
            *   `ReadingType` (string): e.g., "double"
        *   `Metadata` (dictionary):
            *   `SourceName` (string): e.g., "Aarslev"
