# Description of `public.sensor_data_merged` Table and Derived Horticultural Features

This document provides a description of the `public.sensor_data_merged` table from the MCP Postgres database and a list of potential horticulture-specific features that can be derived from it. The insights from the "Data Availability Analysis Report" (`report.md`, found in `Jupyter/classes/data_availability_analyzer/`) are heavily integrated into this description.

## 1. `sensor_data_merged` Table Structure

The `sensor_data_merged` table aggregates sensor readings from various sources and periods. Its utility is significantly impacted by data availability, which varies across distinct "Eras" as identified in the availability report.

### 1.1. Column Overview

| Column Name                        | Data Type                | Notes / Potential Horticultural Meaning (Availability from report.md)                                  |
|------------------------------------|--------------------------|--------------------------------------------------------------------------------------------------------|
| `time`                             | timestamp with time zone | Primary timestamp for sensor readings.                                                                 |
| `air_temp_c`                       | double precision         | Air Temperature (°C). (Good in Era 1, mostly missing Era 2)                                            |
| `air_temp_middle_c`                | double precision         | Middle Air Temperature (°C). (Availability not explicitly detailed, likely similar to `air_temp_c`)        |
| `behov`                            | integer                  | Unknown meaning. (Largely missing/dead column)                                                           |
| `co2_dosing_status`                | double precision         | Status of CO2 dosing. (Availability not explicitly detailed, likely Era 2 if active)                   |
| `co2_measured_ppm`                 | double precision         | Measured CO2 concentration (ppm). (Patchy Era 1, Intermittent Era 2)                                   |
| `co2_required_ppm`                 | double precision         | Required/Setpoint CO2 (ppm). (Availability not detailed)                                                 |
| `co2_status`                       | double precision         | General CO2 system status. (Sample data shows 0, possibly related to dosing status)                        |
| `curtain_1_percent`                | double precision         | Position of Curtain 1 (%). (Sample shows value in Era 1 start)                                         |
| `curtain_2_percent`                | double precision         | Position of Curtain 2 (%).                                                                               |
| `curtain_3_percent`                | double precision         | Position of Curtain 3 (%). (Sample shows value in Era 1 start)                                         |
| `curtain_4_percent`                | double precision         | Position of Curtain 4 (%). (Sample shows value in Era 1 start)                                         |
| `dli_sum`                          | double precision         | Daily Light Integral sum (mol/m²/day). (Sample shows constant value, likely daily aggregated)            |
| `flow_temp_1_c`                    | double precision         | Flow Temperature for Pipe/Circuit 1 (°C). (Sample shows value in Era 1 start)                            |
| `flow_temp_2_c`                    | double precision         | Flow Temperature for Pipe/Circuit 2 (°C). (Sample shows value in Era 1 start)                            |
| `format_type`                      | text                     | Internal data format identifier.                                                                       |
| `heating_setpoint_c`               | double precision         | Target heating temperature (°C). (Extremely limited: June-Aug 2014 only)                               |
| `humidity_deficit_afd3_g_m3`       | double precision         | Humidity Deficit for AFD3 (g/m³). (Good in Era 2)                                                      |
| `humidity_deficit_afd4_g_m3`       | double precision         | Humidity Deficit for AFD4 (g/m³). (Good in Era 2)                                                      |
| `humidity_deficit_g_m3`            | double precision         | General Humidity Deficit (g/m³). (Availability not detailed, potentially Era 1 if related to core RH)    |
| `lamp_group`                       | character varying        | Identifier for a lamp group.                                                                           |
| `lamp_grp1_no3_status`             | boolean                  | Status of Lamp Group 1 (Zone NO3). (Intermittent Era 2)                                                |
| `lamp_grp1_no4_status`             | boolean                  | Status of Lamp Group 1 (Zone NO4). (Intermittent Era 2)                                                |
| `lamp_grp2_no3_status`             | boolean                  | Status of Lamp Group 2 (Zone NO3). (Intermittent Era 2)                                                |
| `lamp_grp2_no4_status`             | boolean                  | Status of Lamp Group 2 (Zone NO4). (Intermittent Era 2)                                                |
| `lamp_grp3_no3_status`             | boolean                  | Status of Lamp Group 3 (Zone NO3). (Intermittent Era 2)                                                |
| `lamp_grp4_no3_status`             | boolean                  | Status of Lamp Group 4 (Zone NO3). (Intermittent Era 2)                                                |
| `lampe_timer_off`                  | bigint                   | Timer setting for lamps OFF. (Largely missing/dead column)                                               |
| `lampe_timer_on`                   | bigint                   | Timer setting for lamps ON. (Largely missing/dead column)                                                |
| `light_intensity_lux`              | double precision         | Light Intensity (Lux). (Availability not detailed, likely Era 1 if present)                            |
| `light_intensity_umol`             | double precision         | Photosynthetically Active Radiation (µmol/m²/s). (Good in Era 1, mostly missing Era 2)                   |
| `measured_status_bool`             | boolean                  | General measured status. (Largely missing/dead column)                                                 |
| `oenske_ekstra_lys`                | text                     | Request for extra light (Danish). (Largely missing/dead column)                                        |
| `outside_light_w_m2`               | double precision         | External/Outdoor light intensity (W/m²). (Sample shows 0, availability likely tracks `radiation_w_m2`) |
| `outside_temp_c`                   | double precision         | External/Outdoor Air Temperature (°C). (Sample shows value, likely good in Era 1 with `air_temp_c`)    |
| `pipe_temp_1_c`                    | double precision         | Pipe Temperature 1 (°C). (Intermittent Era 2)                                                          |
| `pipe_temp_2_c`                    | double precision         | Pipe Temperature 2 (°C). (Intermittent Era 2)                                                          |
| `radiation_w_m2`                   | double precision         | Global Solar Radiation (W/m²). (Good in Era 1 & fairly continuous in Era 2)                            |
| `rain_status`                      | boolean                  | Boolean status indicating rain. (Good in Era 1)                                                        |
| `relative_humidity_afd3_percent`   | double precision         | Relative Humidity for AFD3 (%). (Good in Era 2)                                                        |
| `relative_humidity_afd4_percent`   | double precision         | Relative Humidity for AFD4 (%). (Good in Era 2)                                                        |
| `relative_humidity_percent`        | double precision         | Relative Humidity (%). (Good in Era 1, mostly missing Era 2)                                           |
| `source_file`                      | text                     | Original source file path for the data.                                                                |
| `source_system`                    | text                     | Original source system identifier (e.g., "Aarslev").                                                     |
| `status_str`                       | text                     | General status string. (Largely missing/dead column)                                                   |
| `sun_radiation_actual_w_m2`        | double precision         | Actual solar radiation (W/m²), potentially distinct from `radiation_w_m2`.                               |
| `sun_radiation_forecast_w_m2`      | double precision         | Forecasted solar radiation (W/m²).                                                                     |
| `temperature_actual_c`             | double precision         | Actual temperature (°C), potentially distinct from `air_temp_c`.                                         |
| `temperature_forecast_c`           | double precision         | Forecasted temperature (°C).                                                                           |
| `timer_off`                        | integer                  | General timer OFF setting. (Largely missing/dead column)                                                 |
| `timer_on`                         | integer                  | General timer ON setting. (Largely missing/dead column)                                                  |
| `uuid`                             | text                     | Unique identifier, likely for specific records or devices.                                             |
| `value`                            | double precision         | Generic value column. (Largely missing/dead column as per report)                                      |
| `vent_lee_afd3_percent`            | double precision         | Lee-side Vent Position for AFD3 (%). (Intermittent Era 2)                                                |
| `vent_lee_afd4_percent`            | double precision         | Lee-side Vent Position for AFD4 (%). (Intermittent Era 2)                                                |
| `vent_pos_1_percent`               | double precision         | Vent Position 1 (%). (Intermittent Era 2)                                                                |
| `vent_pos_2_percent`               | double precision         | Vent Position 2 (%). (Intermittent Era 2)                                                                |
| `vent_wind_afd3_percent`           | double precision         | Wind-side Vent Position for AFD3 (%). (Intermittent Era 2)                                               |
| `vent_wind_afd4_percent`           | double precision         | Wind-side Vent Position for AFD4 (%). (Intermittent Era 2)                                               |
| `vpd_hpa`                          | double precision         | Vapor Pressure Deficit (hPa). (Availability not detailed, likely calculated if Temp/RH present)        |
| `window_1_percent`                 | double precision         | Window Position 1 (%). (Largely missing/dead column)                                                     |
| `window_2_percent`                 | double precision         | Window Position 2 (%). (Largely missing/dead column)                                                     |

*Data types are based on `information_schema`. Availability notes are primarily from `report.md`.*

### 1.2. General Data Characteristics (Summary from `report.md` and Sample Query)

* **Temporal Coverage:** Late 2013 to late 2016.
* **Two Key Eras & Major Gap:**
  * **Era 1 (Approx. Early 2014 - Mid/Late August 2014):**
    * **Rich in:** Core environmental sensors like `air_temp_c`, `relative_humidity_percent`, `light_intensity_umol` (PAR), `radiation_w_m2` (global solar radiation), `rain_status`. Curtain positions (`curtain_1_percent`, etc.) and some flow temperatures (`flow_temp_1_c`) also show in the sample.
    * **Patchy/Limited:** `co2_measured_ppm`, `heating_setpoint_c` (very short window).
    * **Mostly Missing:** Actuators (vents, lamps), pipe temperatures.
  * **Major Gap (Approx. Late August 2014 - Mid/Late Oct 2015):** Most sensors inactive.
  * **Era 2 (Approx. Mid/Late October 2015 - Late 2016):**
    * **More Active:** Actuator statuses (`vent_..._percent`, `lamp_grpX_status`), `pipe_temp_X_c`, AFD-specific humidity (`humidity_deficit_afdX_g_m3`, `relative_humidity_afdX_percent`). `radiation_w_m2` is available.
    * **Intermittent:** Many active sensors in this era show high intra-day intermittency.
    * **Mostly Missing:** Core environmental sensors from Era 1 (e.g., `air_temp_c`, `relative_humidity_percent`, `light_intensity_umol`).
* **Data Sparsity:** Many columns can be `null`, as seen in the sample query. The `dli_sum` appears as a daily aggregated value.
* **"Dead" Columns:** Several columns identified in `report.md` (e.g., `behov`, `status_str`, `window_1_percent`, `oenske_ekstra_lys`, `value`) have minimal to no data and should likely be excluded from feature engineering.
* **Provenance:** `source_system`, `source_file`, `format_type` track data origin.

## 2. Potential Horticulture-Specific Features for Analysis and GANs

Given the data structure, distinct availability Eras, and the goal of potentially generating synthetic data with time-series GANs, feature engineering needs to be strategic and era-aware. Features useful for GANs aim to capture temporal dynamics, inter-variable relationships, and realistic data distributions.

**General Note:** For all time-aggregated features (daily, hourly), calculations must respect the data availability within each Era to avoid misleading results due to gaps or periods of high intermittency.

### 2.1. Light Environment Features

* **Daily Light Integral (DLI) from PAR sensor (Era 1):**
  * **Description:** Total photosynthetic light (mol/m²/day).
  * **Source Columns:** `time`, `light_intensity_umol`.
  * **Reasoning:** Critical for plant growth. GANs should learn typical DLI patterns and its relation to other variables. `dli_sum` column might be a pre-calculated version.
* **Daily Light Integral (DLI) from Global Radiation (Era 1 & 2):**
  * **Description:** Estimated DLI from `radiation_w_m2` (requires conversion factor, e.g., W/m² to PAR µmol/m²/s, then integrate).
  * **Source Columns:** `time`, `radiation_w_m2`.
  * **Reasoning:** Provides DLI estimate when direct PAR is missing (Era 2).
* **Artificial Light Status & Duration (Era 2):**
  * **Description:** Binary status (ON/OFF), duration ON per day for each lamp group.
  * **Source Columns:** `time`, `lamp_grp1_no3_status`, `lamp_grp1_no4_status`, etc. (all boolean `lamp_grpX_status` columns).
  * **Reasoning:** Captures artificial lighting strategy. Important for GANs to model energy use and light supplementation logic.
* **Combined Artificial Light Activity (Era 2):**
  * **Description:** Number of active lamp groups, total lamp ON-hours per day.
  * **Source Columns:** `time`, all `lamp_grpX_status` columns.
  * **Reasoning:** Overall artificial light usage.
* **Photoperiod Length:**
  * **Description:** Calculated or defined (e.g., based on natural sunrise/sunset or fixed lighting schedule).
  * **Source Columns:** `time`.
  * **Reasoning:** Fundamental for many plant responses.
* **Light Sum/Average Metrics:**
  * **Description:** Hourly/daily average/sum of `light_intensity_umol` (Era 1), `radiation_w_m2` (Era 1 & 2). Peak values.
  * **Reasoning:** Characterizes light conditions at finer granularities.

### 2.2. Temperature Management Features

* **Air Temperature Metrics (Era 1 focus for `air_temp_c`, potentially `air_temp_middle_c`):**
  * **Description:** Average, min, max (daily, nightly, hourly) internal (`air_temp_c`) and external (`outside_temp_c`) temperatures.
  * **Reasoning:** Core environmental control variable.
* **Temperature Difference (ΔT) (Era 1):**
  * **Description:** `air_temp_c` - `outside_temp_c`.
  * **Reasoning:** Indicates heat loss/gain dynamics.
* **Growing Degree Days (GDD) (Era 1):**
  * **Description:** Accumulated heat units above a base temperature.
  * **Source Columns:** `time`, `air_temp_c`.
  * **Reasoning:** Tracks thermal time for plant development.
* **Heating System Activity (Era 2 focus for pipe temps; Era 1 for flow temps & very limited `heating_setpoint_c`):**
  * **Description:**
    * Average pipe temperatures (`pipe_temp_1_c`, `pipe_temp_2_c`).
    * Average flow temperatures (`flow_temp_1_c`, `flow_temp_2_c`).
    * Binary "heating ON" flag (e.g., if `pipe_temp_1_c` > `air_temp_c` + threshold, or if `heating_setpoint_c` is high and `air_temp_c` is low).
    * Duration of active heating.
  * **Reasoning:** Quantifies heating system operation. GANs need to learn relationship between heating and air temperature. The scarcity of `heating_setpoint_c` makes direct control modeling hard.
* **Temperature within Optimal/Stress Ranges:**
  * **Description:** Hours per day `air_temp_c` is within/outside target bands.
  * **Reasoning:** Identifies periods of optimal growth or stress.

### 2.3. CO2 Environment Features (Era 1 & 2, mind intermittency)

* **CO2 Concentration Metrics:**
  * **Description:** Average, min, max (daily, hourly) `co2_measured_ppm`.
  * Average `co2_measured_ppm` during daylight/lighting periods.
  * **Reasoning:** CO2 is key for photosynthesis.
* **CO2 Dosing Indicators:**
  * **Description:** Features from `co2_dosing_status`, `co2_required_ppm`, `co2_status`.
  * **Reasoning:** Captures active CO2 management.
* **Duration in Optimal/Target CO2 Range:**
  * **Description:** Time spent within desired CO2 levels.
  * **Reasoning:** Measures effectiveness of CO2 control.

### 2.4. Humidity & Airflow Features

* **Relative Humidity Metrics (Era 1 for `relative_humidity_percent`; Era 2 for AFD specific):**
  * **Description:** Average, min, max (daily, hourly) RH.
  * **Source Columns:** `relative_humidity_percent` (Era 1), `relative_humidity_afd3_percent`, `relative_humidity_afd4_percent` (Era 2).
  * **Reasoning:** Impacts transpiration and disease risk.
* **Humidity Deficit / VPD (Era 1 for general; Era 2 for AFD specific; `vpd_hpa` if available):**
  * **Description:**
    * Directly use `humidity_deficit_g_m3` (Era 1), `humidity_deficit_afd3_g_m3`, `humidity_deficit_afd4_g_m3` (Era 2).
    * Calculate VPD if Temp & RH co-occur: `VPD = SaturationVaporPressure(Temp) * (1 - RH/100)`.
    * Use `vpd_hpa` if populated.
  * **Reasoning:** VPD is a key driver of transpiration. GANs should model its dynamics.
* **Ventilation System Activity (Era 2 focus):**
  * **Description:** Average positions, duration open, rate of change for `vent_lee_afd3_percent`, `vent_wind_afd3_percent`, `vent_lee_afd4_percent`, `vent_wind_afd4_percent`, `vent_pos_1_percent`, `vent_pos_2_percent`.
  * **Reasoning:** Ventilation is crucial for temperature, humidity, and CO2 control.
* **Rain Status (Era 1):**
  * **Description:** Binary `rain_status`.
  * **Reasoning:** Often linked to vent closure.

### 2.5. Curtain/Screen Features (Primarily from Era 1 sample, check full availability)

* **Curtain Position Metrics:**
  * **Description:** Average position, duration closed/at specific positions for `curtain_1_percent`, `curtain_2_percent`, `curtain_3_percent`, `curtain_4_percent`.
  * **Reasoning:** Screens manage light and energy.

### 2.6. Temporal Features (Consistent Across Eras)

* **Cyclical Time Features:** Time of day (sin/cos encoded), day of year (sin/cos encoded).
* **Categorical Time Features:** Hour of day, day of week, month.
* **Reasoning:** Capture diurnal and seasonal patterns critical for environmental variables and plant responses.

### 2.7. Lagged and Rolling Aggregate Features

* **Description:** Apply to key continuous variables within each Era, respecting data availability.
  * Lagged values (e.g., `air_temp_c` 1h, 3h, 6h ago).
  * Rolling means, medians, std dev, min, max (e.g., 1-hour, 6-hour, 24-hour windows).
* **Source Columns:** `air_temp_c`, `relative_humidity_percent`, `co2_measured_ppm`, `radiation_w_m2`, `pipe_temp_X_c`, vent positions, etc.
* **Reasoning:** Crucial for time-series GANs to learn temporal dependencies and smoothed trends. Must handle NaNs carefully during calculation (e.g., `min_periods` for rolling functions).

### 2.8. Interaction & Complex Derived Features

* **Temperature & Humidity:** VPD (if not direct), Dew Point.
* **Light & CO2:** CO2 levels during high light periods.
* **Energy Proxies:**
  * Combined lamp ON-time multiplied by estimated wattage.
  * Heating energy proxy: (Pipe Temp - Air Temp) * duration, if pipe temp > air temp.
* **Control Logic Proxies:**
  * Difference between setpoint and measured value (e.g., `heating_setpoint_c - air_temp_c`, where `heating_setpoint_c` is available).
  * Frequency of actuator changes (e.g., number of vent movements per hour).

### 2.9. Features for GANs (especially GAN-doppelganger) & Handling Data Issues

* **Era Identifier:** A categorical feature indicating "Era 1", "Gap", or "Era 2". This can be an "attribute" in GAN-doppelganger.
* **Data Availability Flags:**
  * Binary flags: `[column_name]_is_valid` (0 if NaN, 1 if present) for key intermittent sensors.
  * `time_since_last_valid_[column_name]` for intermittent sensors.
  * **Reasoning:** Can help the GAN learn patterns of missingness or generate more realistic data by understanding when sensors are typically offline.
* **Normalized/Standardized Values:** All numerical features should be scaled appropriately before feeding into a GAN.
* **Time Delta Feature:** `delta_t` = time difference between consecutive records. Useful if sampling rate is irregular, though `sensor_data_merged` seems to be at fixed intervals when data is present.

### 2.10. Source & Metadata Features

* `source_system`: Can be used as a categorical attribute if different systems have distinct behaviors.
* `format_type`: If different formats imply different sensor sets or reliability.

**Key Considerations Based on `report.md`:**

* **Prioritize Era 1 (Early 2014 - Aug 2014):** This period has the best co-availability of core environmental sensors (`air_temp_c`, `relative_humidity_percent`, `light_intensity_umol`, `radiation_w_m2`). It's a strong candidate for initial, more comprehensive feature engineering and model training.
* **Handle Era 2 (Oct 2015 - Late 2016) Carefully:** Focus on available actuator data, pipe temps, AFD humidity, and `radiation_w_m2`. Be mindful of high intra-day intermittency when creating rolling/lagged features.
* **The Major Gap:** GANs will struggle to learn across this. Treat data as separate sequences or use an indicator.
* **Exclude "Dead" Columns:** Columns like `behov`, `status_str`, `window_1_percent`, `oenske_ekstra_lys`, `value` should be dropped early.

This list should serve as a robust starting point for your feature engineering. The actual implementation will require careful handling of data availability and potential imputation strategies, guided by the `report.md`.
