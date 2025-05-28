// Allow dead code for data models that are primarily used for deserialization
#![allow(dead_code)]

use serde::{Deserialize, Deserializer, Serialize};
use serde::de::{self, Visitor};
use std::collections::HashMap;
use std::fmt;
use serde_json;
use chrono;

// Helper function to parse comma-decimal strings
// Moved here as it's primarily used by SensorValue deserializer
pub fn parse_comma_decimal(s: &str) -> Result<f64, std::num::ParseFloatError> {
    s.replace(',', ".").parse::<f64>()
}

// --- Deserialization Helpers ---

// Helper to deserialize Option<String> into Option<i32>
pub(crate) fn deserialize_optional_int<'de, D>(deserializer: D) -> Result<Option<i32>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt_s: Option<String> = Option::deserialize(deserializer)?;
    match opt_s {
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                match trimmed.parse::<i32>() {
                    Ok(val) => Ok(Some(val)),
                    Err(_) => Ok(None), // Treat parse errors as None for optional int
                }
            }
        }
        None => Ok(None),
    }
}

// Helper to deserialize Option<String> (with comma decimal) into Option<f64>
pub(crate) fn deserialize_optional_string_to_float<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt_s: Option<String> = Option::deserialize(deserializer)?;
    match opt_s {
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                match parse_comma_decimal(trimmed) { // Reuse our existing float parser
                    Ok(val) => Ok(Some(val)),
                    Err(_) => Ok(None), // Treat parse errors as None for optional float
                }
            }
        }
        None => Ok(None),
    }
}

// Helper Function for Deserializing Option<f64> with Comma Decimals
pub(crate) fn deserialize_optional_comma_decimal<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    // Deserialize as an Option<String> first to handle empty/null fields gracefully
    let opt_s: Option<String> = Option::deserialize(deserializer)?;

    match opt_s {
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                // Treat empty string as None
                Ok(None)
            } else {
                // Attempt to parse using the existing helper
                match parse_comma_decimal(trimmed) {
                    Ok(val) => Ok(Some(val)),
                    Err(_) => {
                        // If parsing fails, treat it as None for Option<f64>.
                        // Alternatively, you could return a custom error here:
                        // Err(de::Error::custom(format!("Invalid number format: {}", trimmed)))
                        // But for optional fields, returning None is often preferred.
                        Ok(None)
                    }
                }
            }
        }
        None => {
            // If the field is explicitly null or missing, it's None
            Ok(None)
        }
    }
}
// --- End Deserialization Helpers ---

// Enum to represent parsed sensor values more explicitly
#[derive(Debug, Clone)] // Add Debug and Clone
#[allow(dead_code)]
pub enum SensorValue {
    Number(f64),
    Empty,
    Text(String),
}

// Implement Deserialize for SensorValue to handle quotes and parsing
impl<'de> Deserialize<'de> for SensorValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SensorValueVisitor;

        impl<'de> Visitor<'de> for SensorValueVisitor {
            type Value = SensorValue;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string (potentially quoted, comma-decimal), number, or empty field representing a sensor value")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let trimmed = value.trim_matches('"');
                if trimmed.is_empty() {
                    Ok(SensorValue::Empty)
                } else {
                    // Use the parse_comma_decimal function from this module
                    match parse_comma_decimal(trimmed) {
                        Ok(num) => Ok(SensorValue::Number(num)),
                        Err(_) => Ok(SensorValue::Text(trimmed.to_string())),
                    }
                }
            }

            // Handle cases where the CSV reader directly provides a float
            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Number(value))
            }

            // Also handle potential integers if the CSV reader provides them
            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Number(value as f64))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Number(value as f64))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Empty)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Empty)
            }
        }

        // Use deserialize_any to let the visitor handle the type determined by serde_csv
        deserializer.deserialize_any(SensorValueVisitor)
    }
}


// Placeholder struct for initial testing or fallback
#[derive(Debug, Deserialize)]
pub struct GenericRecord {
    #[serde(flatten)]
    pub other_fields: HashMap<String, String>,
}

// Final Struct for aarslev/*MortenSDUData*.csv
#[derive(Debug)] // No Deserialize derive - implementing manually
pub struct AarslevMortenRecord {
    pub start: Option<String>,
    pub end: Option<String>,
    pub celle_data: HashMap<String, Option<f64>>,
}

// Struct for knudjepsen/NO{3,4}_LAMPEGRP_1.csv format
#[derive(Debug, Deserialize)]
pub struct KnudjepsenLampRecord {
    #[serde(rename = "")]
    pub timestamp_str: String,
    #[serde(rename = "mål temp afd  Mid")]
    pub maal_temp_afd_mid_str: Option<String>,
    pub udetemp_str: Option<String>,
    pub stråling: Option<String>,
    #[serde(rename = "CO2 målt")]
    pub co2_maalt_str: Option<String>,
    #[serde(rename = "mål gard 1")]
    pub maal_gard_1_str: Option<String>,
    #[serde(rename = "mål gard 2")]
    pub maal_gard_2_str: Option<String>,
    // Read maalt_status as String initially to handle potential non-integer values
    #[serde(rename = "målt status")]
    pub maalt_status: Option<String>,
    #[serde(rename = "mål FD")]
    pub maal_fd_str: Option<String>,
    #[serde(rename = "mål rør 1")]
    pub maal_roer_1_str: Option<String>,
    #[serde(rename = "mål rør 2")]
    pub maal_roer_2_str: Option<String>,
}

// Struct for aarslev/celle*/output-*.csv
#[derive(Debug, Deserialize)] // ADD Deserialize back
pub struct AarslevCelleOutputRecord {
    // We will parse Date and Time manually using config
    #[serde(skip_deserializing)] // Skip automatic deserialization for this field
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>, 
    // Store the combined format string derived from config
    #[serde(skip)] // Don't try to deserialize this from CSV
    pub timestamp_format: Option<String>, 
    #[serde(flatten)]
    pub sensor_data: HashMap<String, SensorValue>,
}

// Struct for aarslev/{weather, data_jan_feb, winter}*.csv types
// Might need refinement based on actual file contents / merging strategy
#[derive(Debug, Deserialize, Clone)]
pub struct AarslevWeatherRecord {
    #[serde(rename = "timestamp")]
    pub timestamp: serde_json::Value,
    pub temperature: Option<f64>,
    pub sun_radiation: Option<f64>,
    pub actual_temperature: Option<f64>,
    pub actual_radiation: Option<f64>,
}

// Struct for aarslev/temperature_sunradiation*.json.csv
#[derive(Debug, Deserialize)]
pub struct AarslevTempSunRecord {
    pub timestamp: Option<i64>, // Assuming epoch (timestamp parsing later)
    #[serde(flatten)]
    pub uuid_data: HashMap<String, Option<f64>>,
}

// Struct specifically for knudjepsen/NO3-NO4_belysningsgrp.csv
// Using a tuple struct to handle duplicate headers by position.
#[derive(Debug, Deserialize)]
pub struct KnudjepsenBelysningGrpRecord(
    // No rename needed for the first field (timestamp)
    pub Option<String>,

    // Use position for the subsequent 'målt status' fields
    #[serde(deserialize_with = "deserialize_optional_int")]
    pub Option<i32>, // Corresponds to first 'målt status'
    #[serde(deserialize_with = "deserialize_optional_int")]
    pub Option<i32>, // Corresponds to second 'målt status'
    #[serde(deserialize_with = "deserialize_optional_int")]
    pub Option<i32>, // Corresponds to third 'målt status'
    #[serde(deserialize_with = "deserialize_optional_int")]
    pub Option<i32>, // Corresponds to fourth 'målt status'
    #[serde(deserialize_with = "deserialize_optional_int")]
    pub Option<i32>, // Corresponds to fifth 'målt status'
    #[serde(deserialize_with = "deserialize_optional_int")]
    pub Option<i32>, // Corresponds to sixth 'målt status'
);

// Struct for knudjepsen/NO?_EkstraLysStyring.csv
#[derive(Debug, Deserialize)]
pub struct KnudjepsenEkstraLysRecord {
    #[serde(rename = "")]
    pub timestamp_str: Option<String>, // Keep as string for now (timestamp parsing later)
    #[serde(rename = "Ønske om ekstra lys")]
    pub oenske_ekstra_lys: Option<String>, // Could be bool? Check data.
    #[serde(rename = "lampe timer on")]
    pub lampe_timer_on: Option<i64>,
    #[serde(rename = "lampe timer off")]
    pub lampe_timer_off: Option<i64>,
}

// Struct for knudjepsen/NO?_BelysningsStyring.csv
#[derive(Debug, Deserialize)]
pub struct KnudjepsenBelysningRecord {
    #[serde(rename = "")]
    pub timestamp_str: Option<String>,
    #[serde(deserialize_with = "deserialize_optional_int")]
    pub behov: Option<i32>,
    pub status: Option<String>,
    #[serde(rename = "timer tænd", deserialize_with = "deserialize_optional_int")]
    pub timer_on: Option<i32>,
    #[serde(rename = "timer sluk", deserialize_with = "deserialize_optional_int")]
    pub timer_off: Option<i32>,
    #[serde(rename = "DLI sum", deserialize_with = "deserialize_optional_string_to_float")]
    pub dli_sum: Option<f64>, // Changed type to Option<f64> and renamed field
}

// Struct for knudjepsen/NO3NO4.extra.csv
// Using a tuple struct to handle duplicate headers by position.
#[derive(Debug, Deserialize)]
pub struct KnudjepsenExtraRecord(
    // Timestamp column
    pub Option<String>,

    // Afd 3 columns (positional)
    #[serde(deserialize_with = "deserialize_optional_comma_decimal")]
    pub Option<f64>, // afd3_maal_fd
    #[serde(deserialize_with = "deserialize_optional_comma_decimal")]
    pub Option<f64>, // afd3_maal_rf
    #[serde(deserialize_with = "deserialize_optional_comma_decimal")]
    pub Option<f64>, // afd3_maal_lae
    #[serde(deserialize_with = "deserialize_optional_comma_decimal")]
    pub Option<f64>, // afd3_maal_vind

    // Afd 4 columns (positional)
    #[serde(deserialize_with = "deserialize_optional_comma_decimal")]
    pub Option<f64>, // afd4_maal_fd
    #[serde(deserialize_with = "deserialize_optional_comma_decimal")]
    pub Option<f64>, // afd4_maal_rf
    #[serde(deserialize_with = "deserialize_optional_comma_decimal")]
    pub Option<f64>, // afd4_maal_lae
    #[serde(deserialize_with = "deserialize_optional_comma_decimal")]
    pub Option<f64>, // afd4_maal_vind
);

// Struct for aarslev/celle*/output-*.csv.json files (Configuration)
#[derive(Debug, Deserialize, Clone)]
pub struct AarslevCelleJsonConfig {
    #[serde(rename = "Configuration")]
    pub configuration: Option<AarslevCelleJsonConfigDetails>,
    #[serde(rename = "Filename")]
    pub filename: Option<String>,
    #[serde(rename = "Variables")]
    pub variables: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AarslevCelleJsonConfigDetails {
    pub date_format: Option<String>,
    pub delimiter: Option<String>,
    pub time_format: Option<String>,
}

// ADDED: Struct for Aarslev JSON data files (e.g., temperature_sunradiation...json)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AarslevJsonReading {
    pub uuid: String,
    #[serde(rename = "Readings")]
    pub readings: Vec<(f64, f64)>, // Vec of (timestamp_ms_f64, value_f64)
}

// ADDED: Wrapper struct because the JSON is an array of AarslevJsonReading
// We might deserialize directly into Vec<AarslevJsonReading> instead,
// but having this struct allows potential future expansion if the outer structure changes.
// For now, we'll likely deserialize into Vec<AarslevJsonReading>.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AarslevJsonDataFile {
    pub data: Vec<AarslevJsonReading>,
}

// --- Structs for parsing sensor data from .csv.json files ---

// Represents the entire structure of a .csv.json file
// pub type SensorDataFile = HashMap<String, SensorEntry>; // REMOVED unused type alias

// Represents a single sensor's data within the JSON
#[derive(Debug, Clone, Deserialize, Serialize)] // Ensure derives are present
pub struct SensorEntry {
    // Readings are arrays of [timestamp (i64), value (f64)]
    #[serde(rename = "Readings")]
    pub readings: Vec<(i64, f64)>,
    #[serde(rename = "uuid")] // Add rename for exact JSON field match
    pub uuid: String,
    #[serde(rename = "Properties")] // Add rename for exact JSON field match
    pub properties: Properties,
    #[serde(rename = "Metadata")] // Add rename for exact JSON field match
    pub metadata: Metadata,
}

// Represents the Properties object within a SensorEntry
#[derive(Debug, Clone, Deserialize, Serialize)] // Ensure derives are present
pub struct Properties {
    #[serde(rename = "Timezone")] // Add rename for exact JSON field match
    pub timezone: Option<String>,
    #[serde(rename = "UnitofMeasure")] // Add rename for exact JSON field match
    pub unit_of_measure: Option<String>,
    #[serde(rename = "ReadingType")] // Add rename for exact JSON field match
    pub reading_type: Option<String>,
}

// Represents the Metadata object within a SensorEntry
#[derive(Debug, Clone, Deserialize, Serialize)] // Ensure derives are present
pub struct Metadata {
    #[serde(rename = "SourceName")] // Add rename for exact JSON field match
    pub source_name: Option<String>,
}
// --- End of structs for .csv.json sensor data --- 