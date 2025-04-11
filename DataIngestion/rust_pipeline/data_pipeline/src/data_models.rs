use chrono::{DateTime, Utc};
// use std::collections::HashMap; // Remove unused import

/// Represents a single, unified row of parsed data from any source file.
/// Fields are optional to accommodate different file formats.
#[derive(Debug, Clone, Default)] // Add Default trait for easier construction
pub struct ParsedRecord {
    // Core Identification & Timing
    pub source_file: Option<String>, // Path of the original file
    pub source_system: Option<String>, // e.g., "KnudJepsen", "Aarslev"
    pub format_type: Option<String>,   // e.g., "KnudJepsenNO3Lamp"
    pub timestamp_utc: Option<DateTime<Utc>>,

    // Common Environmental Measurements
    pub air_temp_c: Option<f64>,             // Generic air temp
    pub air_temp_middle_c: Option<f64>,      // Specific to KnudJepsen
    pub outside_temp_c: Option<f64>,
    pub relative_humidity_percent: Option<f64>,
    pub humidity_deficit_g_m3: Option<f64>,
    pub radiation_w_m2: Option<f64>,
    pub light_intensity_lux: Option<f64>,   // More specific light unit
    pub light_intensity_umol: Option<f64>,  // PAR light unit
    pub outside_light_w_m2: Option<f64>,
    pub co2_measured_ppm: Option<f64>,
    pub co2_required_ppm: Option<f64>,
    pub co2_dosing_status: Option<f64>,     // Or maybe bool/enum?
    pub co2_status: Option<f64>,            // Another CO2 status?
    pub rain_status: Option<bool>,          // From MortenSDUData

    // Control System State
    pub vent_pos_1_percent: Option<f64>,      // Aarslev Celle
    pub vent_pos_2_percent: Option<f64>,      // Aarslev Celle
    pub vent_lee_afd3_percent: Option<f64>,   // KnudJepsen Extra
    pub vent_wind_afd3_percent: Option<f64>,  // KnudJepsen Extra
    pub vent_lee_afd4_percent: Option<f64>,   // KnudJepsen Extra
    pub vent_wind_afd4_percent: Option<f64>,  // KnudJepsen Extra
    pub curtain_1_percent: Option<f64>,
    pub curtain_2_percent: Option<f64>,
    pub curtain_3_percent: Option<f64>,
    pub curtain_4_percent: Option<f64>,
    pub window_1_percent: Option<f64>,       // From MortenSDUData
    pub window_2_percent: Option<f64>,       // From MortenSDUData
    pub lamp_grp1_no3_status: Option<bool>, // KnudJepsen Lighting
    pub lamp_grp2_no3_status: Option<bool>,
    pub lamp_grp3_no3_status: Option<bool>,
    pub lamp_grp4_no3_status: Option<bool>,
    pub lamp_grp1_no4_status: Option<bool>,
    pub lamp_grp2_no4_status: Option<bool>,
    pub measured_status_bool: Option<bool>,  // KnudJepsen Lamp

    // Heating & Flow
    pub heating_setpoint_c: Option<f64>,    // Aarslev Celle
    pub pipe_temp_1_c: Option<f64>,
    pub pipe_temp_2_c: Option<f64>,
    pub flow_temp_1_c: Option<f64>,         // From MortenSDUData
    pub flow_temp_2_c: Option<f64>,         // From MortenSDUData

    // Forecasts (from specific Aarslev files)
    pub temperature_forecast_c: Option<f64>,
    pub sun_radiation_forecast_w_m2: Option<f64>,
    pub temperature_actual_c: Option<f64>, // Separate 'actual' if distinct from 'measured'
    pub sun_radiation_actual_w_m2: Option<f64>,

    // Knudjepsen Specific (Less Common)
    pub vpd_hpa: Option<f64>,              // Aarslev Celle VPD
    pub humidity_deficit_afd3_g_m3: Option<f64>, // KnudJepsen Extra
    pub relative_humidity_afd3_percent: Option<f64>, // KnudJepsen Extra
    pub humidity_deficit_afd4_g_m3: Option<f64>, // KnudJepsen Extra
    pub relative_humidity_afd4_percent: Option<f64>, // KnudJepsen Extra
    pub behov: Option<i32>,                // KnudJepsen Belysning
    pub status_str: Option<String>,        // KnudJepsen Belysning (original string status)
    pub timer_on: Option<i32>,             // KnudJepsen Belysning
    pub timer_off: Option<i32>,            // KnudJepsen Belysning
    pub dli_sum: Option<f64>,              // KnudJepsen Belysning
    pub oenske_ekstra_lys: Option<String>, // Knudjepsen EkstraLys
    pub lampe_timer_on: Option<i64>,       // KnudJepsen EkstraLys
    pub lampe_timer_off: Option<i64>,      // KnudJepsen EkstraLys

    // Potential field for JSON UUIDs where data structure is column-oriented
    pub uuid: Option<String>,
    pub value: Option<f64>, // Generic value used in some JSON structures

    // Add a generic map for any columns not explicitly mapped?
    // pub extra_data: Option<HashMap<String, String>>,
} 