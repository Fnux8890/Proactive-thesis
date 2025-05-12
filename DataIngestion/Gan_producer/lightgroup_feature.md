# Potential Horticulture-Specific Features

This document outlines potential features that can be engineered from the available sensor data, primarily focusing on insights gained from the processed CSV files. The `source_group` column (derived from filenames like `NO3-NO4_belysningsgrp`, `NO3NO4.extra`, `NO3_LAMPEGRP_1`, `NO4_LAMPEGRP_1`) is crucial for comparative analysis as column availability varies significantly between these groups.

**Note:** Knowledge of the `sensor_data_table` in the MCP Postgres database would allow for a more comprehensive list, potentially including direct measurements not available or clearly identifiable in the current CSVs.

## I. Light Environment Features

Light is a primary driver for photosynthesis and plant development.

1. **Individual Light System "ON" Duration & Proportion**
    * **Description:** Total time (e.g., hours per day) or proportion of a period (e.g., photoperiod) that a specific artificial light system is active.
    * **Source Columns:** DatetimeIndex, `målt status` (for `NO3_LAMPEGRP_1`, `NO4_LAMPEGRP_1` groups), `målt status_1` through `målt status_6` (primarily for `NO3-NO4_belysningsgrp` group).
    * **Calculation:** Sum `status == 1` over a time window (e.g., daily) and multiply by the sampling interval. For proportion, divide by the window duration.
    * **Reasoning:** Quantifies the contribution of specific supplemental light sources. Essential for understanding artificial DLI contribution, energy consumption per light group, and evaluating lighting strategies. Useful for comparing different lighting zones if these status columns represent them.

2. **Combined Artificial Light System "ON" Metrics**
    * **Description:** Metrics representing the overall artificial lighting activity within a `source_group`.
        * Total summed light-hours from all active systems per day (as seen in `daily_total_light_activity_per_group.png`).
        * Average number of distinct light systems active concurrently.
        * Maximum number of distinct light systems active concurrently.
    * **Source Columns:** DatetimeIndex, all relevant `målt status` and `målt status_X` columns for a given `source_group`.
    * **Reasoning:** Provides an overview of the total supplemental lighting load. Key for overall energy consumption estimates related to lighting and assessing the intensity of the artificial light period.

3. **Natural Light Intensity Metrics (from `stråling`)**
    * **Description:** Metrics derived from the `stråling` sensor, which likely measures solar radiation (e.g., W/m² or µmol/m²/s PAR – units need confirmation).
        * **Daily Light Integral (DLI):** Total photosynthetic light received per day (mol/m²/day). Requires conversion if `stråling` is not in µmol/m²/s.
        * Average hourly `stråling`.
        * Peak `stråling` value per day.
        * Hours per day `stråling` exceeds a specific threshold (e.g., indicating very bright conditions).
        * Cumulative `stråling` over a week or other period.
    * **Source Columns:** DatetimeIndex, `stråling` (present in `NO3_LAMPEGRP_1`, `NO4_LAMPEGRP_1` groups).
    * **Reasoning:** DLI is a critical factor for plant growth, yield, and quality. Understanding natural light contribution helps optimize supplemental lighting and screen usage.

4. **Screening/Shading System Features (from `mål gard 1`, `mål gard 2`)**
    * **Description:** Metrics related to the operation of climate screens, assuming these columns represent screen status (e.g., 0 for open, 1 for closed, or a percentage).
        * Duration screens (`mål gard 1`, `mål gard 2`) are closed/active per day.
        * Proportion of daylight hours screens are closed.
        * Number of screen movements (transitions from open to close or vice-versa) per day.
        * Correlation of screen status with `stråling` (e.g., are screens used during high solar radiation?).
        * Average screen position if the data is analog (percentage).
    * **Source Columns:** DatetimeIndex, `mål gard 1`, `mål gard 2` (present in `NO3_LAMPEGRP_1`, `NO4_LAMPEGRP_1` groups), `stråling`.
    * **Reasoning:** Screens are used to manage temperature, light intensity, and humidity. Analyzing their usage pattern is key to understanding climate control strategy and energy saving.

## II. Temperature Management Features

Temperature directly influences plant metabolic rates, growth speed, and can induce stress.

1. **Air Temperature Features (Internal & External)**
    * **Description:** Metrics describing the air temperature conditions.
        * Average, min, max daily internal air temperature (`mål temp afd Mid`).
        * Average, min, max nightly internal air temperature.
        * Average, min, max daily external air temperature (`udetemp`).
        * **Temperature Difference (ΔT):** Internal (`mål temp afd Mid`) - External (`udetemp`).
        * **Growing Degree Days (GDD):** Calculated daily using `mål temp afd Mid` and a crop-specific base temperature. (e.g., `max(0, AvgTemp - BaseTemp)`).
        * Hours per day/night internal temperature is within an optimal range for the crop.
        * Hours per day/night internal temperature is above stress thresholds (e.g., >30°C) or below chill thresholds.
        * Rate of temperature change (e.g., °C per hour) for `mål temp afd Mid`.
        * Average day temperature (e.g., 6 AM - 6 PM) vs. average night temperature.
    * **Source Columns:** DatetimeIndex, `mål temp afd Mid`, `udetemp` (present in `NO3_LAMPEGRP_1`, `NO4_LAMPEGRP_1` groups).
    * **Reasoning:** Essential for crop scheduling, predicting developmental stages (GDD), identifying stress periods, and evaluating greenhouse insulation/climate control effectiveness (ΔT).

2. **Heating System Activity Features (from `mål rør 1`, `mål rør 2`)**
    * **Description:** Metrics derived from pipe temperatures, assuming these indicate heating system operation.
        * Average pipe temperatures (`mål rør 1`, `mål rør 2`).
        * Duration pipe temperatures are above a defined setpoint or significantly above air temperature (indicating active heating).
        * Difference between average pipe temperature and average air temperature (`mål temp afd Mid`).
        * Cumulative "heating degree hours" or a proxy for heating energy based on pipe temperature elevation and duration.
    * **Source Columns:** DatetimeIndex, `mål rør 1`, `mål rør 2`, `mål temp afd Mid` (present in `NO3_LAMPEGRP_1`, `NO4_LAMPEGRP_1` groups).
    * **Reasoning:** Indicates when and how intensively the heating system is used, crucial for energy consumption analysis and maintaining optimal temperatures.

## III. CO2 Environment Features

CO2 is a vital input for photosynthesis.

1. **CO2 Concentration Metrics (from `CO2 målt`)**
    * **Description:** Metrics describing CO2 levels in the greenhouse.
        * Average, min, max daily/hourly CO2 concentration.
        * Average CO2 concentration during daylight hours (when photosynthesis occurs).
        * Duration (hours per day) CO2 is within an optimal range (e.g., 700-1200 ppm for enrichment, or near ambient ~400 ppm if not enriched).
        * Duration CO2 is above a certain high threshold (e.g., >1500 ppm, indicating active enrichment or poor ventilation).
        * Correlation of CO2 levels with ventilation activity (if available) or light status.
    * **Source Columns:** DatetimeIndex, `CO2 målt` (present in `NO3_LAMPEGRP_1`, `NO4_LAMPEGRP_1` groups).
    * **Reasoning:** Managing CO2 can significantly boost crop yield and quality. These features help assess the effectiveness of CO2 management (enrichment or ventilation).

## IV. Humidity & Airflow Related Features

Humidity and airflow impact plant transpiration, nutrient uptake, and disease risk.

1. **Relative Humidity (RH) & Vapor Pressure Deficit (VPD) (if `mål RF_X` is RH)**
    * **Description:** Metrics related to air moisture content.
        * Average, min, max daily/hourly RH (from `mål RF_1`, `mål RF_2`).
        * Hours RH is within optimal/suboptimal ranges (e.g., too high >85%, too low <40%).
        * **Vapor Pressure Deficit (VPD):** Calculated using air temperature (e.g., `mål temp afd Mid` if available in the same `source_group` as `mål RF_X`) and RH. VPD = SaturationVaporPressure(Temp) * (1 - RH/100).
        * Average, min, max daily/hourly VPD.
        * Hours VPD is within optimal ranges (e.g., 0.5-1.2 kPa for many crops).
    * **Source Columns:** DatetimeIndex, `mål RF_1`, `mål RF_2` (present in `NO3NO4.extra` group). `mål temp afd Mid` (from `LAMPEGRP` files) would be needed for VPD calculations if these sensors were in the same physical location/group. *Cross-group VPD calculation requires careful consideration of data alignment.*
    * **Reasoning:** RH directly affects plant transpiration and disease likelihood. VPD is a more accurate measure of the "drying power" of the air and its impact on plant water stress and nutrient uptake.

2. **Generic Airflow/Ventilation Proxy Features (from `mål FD_X`, `mål læ_X`, `mål vind_X`)**
    * **Description:** Basic statistical features if the exact nature of these sensors is unknown. `FD` might relate to flow/deficit, `læ` to leeward/shelter, `vind` to wind.
        * Average, min, max, std dev of `mål FD_1`, `mål FD_2`, `mål læ_1`, `mål læ_2`, `mål vind_1`, `mål vind_2` per hour/day.
    * **Source Columns:** DatetimeIndex, `mål FD_1`, `mål FD_2`, `mål læ_1`, `mål læ_2`, `mål vind_1`, `mål vind_2` (present in `NO3NO4.extra` group).
    * **Reasoning:** These might capture aspects of air movement, ventilation status, or pressure differences. Even without precise understanding, their variance or average levels might correlate with other important greenhouse states or control actions. *Further information on these sensors is needed for more specific features.*

## V. Energy Use Proxy Features

Direct energy consumption data is ideal, but proxies can be derived.

1. **Lighting Energy Proxy:**
    * **Description:** Sum of "ON" durations for all identified light systems.
    * **Source Columns:** DatetimeIndex, all `målt status` and `målt status_X` columns.
    * **Reasoning:** Directly proportional to energy used by lights, assuming constant power per light system.

2. **Heating Energy Proxy:**
    * **Description:** Sum of "active heating" durations derived from pipe temperatures.
    * **Source Columns:** DatetimeIndex, `mål rør 1`, `mål rør 2`, `mål temp afd Mid`.
    * **Reasoning:** Higher/longer pipe temperatures relative to air temperature suggest more heating energy.

## VI. Temporal Features

Incorporating time-based features can capture cyclical patterns.

1. **Cyclical Time Features:**
    * **Description:** Representations of time that respect their cyclical nature.
        * Time of day: e.g., `sin(2 * pi * hour / 24)`, `cos(2 * pi * hour / 24)`.
        * Day of year: e.g., `sin(2 * pi * day_of_year / 365.25)`, `cos(2 * pi * day_of_year / 365.25)`.
    * **Source Columns:** DatetimeIndex.
    * **Reasoning:** Helps models understand diurnal and annual cycles that influence plant responses and environmental conditions.

2. **Categorical Time Features:**
    * **Description:** Discrete time units.
        * Hour of day (as integer).
        * Day of week.
        * Month.
        * Season.
        * Is_weekend / Is_weekday.
    * **Source Columns:** DatetimeIndex.
    * **Reasoning:** May capture operational patterns or environmental trends associated with these periods.

## VII. Lagged and Rolling Aggregate Features

These features capture historical context and trends.

1. **Lagged Sensor Values:**
    * **Description:** The value of a sensor from a previous time step(s).
    * **Example:** `mål temp afd Mid` from 1 hour ago, `CO2 målt` from 30 minutes ago, DLI from yesterday.
    * **Source Columns:** Any numerical sensor column, DatetimeIndex.
    * **Reasoning:** Current conditions are often influenced by recent past conditions. Plant responses can also have lags.

2. **Rolling Aggregates:**
    * **Description:** Statistical aggregates (mean, median, sum, std dev, min, max) over a moving time window.
    * **Example:** Rolling 24-hour average for `mål temp afd Mid`, rolling 7-day sum for DLI (if calculated daily).
    * **Source Columns:** Any numerical sensor column, DatetimeIndex.
    * **Reasoning:** Smooths out short-term fluctuations, highlights trends, and can define baseline conditions.

## VIII. Cross-Variable & Interaction Features (Examples)

These combine information from multiple sensors.

1. **Temperature-Humidity Interactions:**
    * VPD (already listed under Humidity).
    * Dew point temperature (calculated from Temp & RH).
2. **Light-CO2 Interaction:**
    * CO2 concentration specifically during periods of high light (`stråling` or artificial lights ON).
3. **Control System Interactions:**
    * Temperature (`mål temp afd Mid`) when heating pipes (`mål rør 1/2`) are active vs. inactive.
    * Internal temperature/RH when screens (`mål gard 1/2`) are active vs. inactive.

---

This list provides a strong starting point. The next steps would involve:

* **Confirming units and exact meanings** of all sensor columns (especially `stråling`, `mål FD_X`, `mål RF_X`, `mål læ_X`, `mål vind_X`).
* **Integrating knowledge of the `sensor_data_table` from Postgres** to add features based on potentially richer or more direct measurements.
* **Prioritizing features** based on the specific horticultural questions or model objectives.
* **Implementing the feature engineering pipeline.**
