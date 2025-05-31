# Sensor Data Merged Table Analysis Results

## Critical Findings

### 1. **Massive Data Sparsity**
- **Total rows**: 1,457,281 (covering ~2.8 years from 2013-12-01 to 2016-09-08)
- **Completely empty rows**: 1,330,176 (91.3% of all data!)
- **Rows with actual sensor data**: ~127,105 (8.7%)

### 2. **NULL Percentages by Column**
| Column | NULL % | Non-NULL Rows |
|--------|--------|---------------|
| light_intensity_umol | 99.90% | 1,520 |
| heating_setpoint_c | 99.08% | 13,418 |
| relative_humidity_percent | 98.97% | 14,991 |
| air_temp_c | 98.69% | 19,134 |
| vent_wind_afd3_percent | 98.54% | 21,328 |
| vent_lee_afd3_percent | 98.54% | 21,328 |
| pipe_temp_1_c | 96.37% | 52,835 |
| co2_measured_ppm | 95.35% | 67,825 |
| radiation_w_m2 | 95.12% | 71,069 |

### 3. **Data Sources and Coverage**
The data comes from multiple sources with very different sensor coverage:

**Aarslev Sources:**
- `celle5/celle6` CSV files: Provide air_temp, humidity, CO2, radiation, heating_setpoint
- `temperature_sunradiation` JSON files: Only air_temp and radiation
- `MortenSDUData` files: Varied sensors (some only light_intensity)

**KnudJepsen Sources:**
- `NO3_LAMPEGRP_1.csv` / `NO4_LAMPEGRP_1.csv`: CO2, radiation, pipe_temp
- `NO3-NO4_belysningsgrp.csv`: Only lamp status data

### 4. **Temporal Data Distribution**
- Data is stored at 1-minute intervals with NO gaps in the time series
- However, most minutes have completely empty sensor readings
- Daily data coverage ranges from 0.07% to 10% of the day having any sensor data
- Best coverage period: December 2013 (up to 10% daily coverage)
- Worst coverage: Most days have <2% coverage

### 5. **Data Availability Patterns**

**2013-2014 Period:**
- Some air temperature and humidity data
- Limited CO2 measurements
- Very sparse light intensity data (only ~1,520 total readings!)

**2015-2016 Period:**
- NO air temperature or humidity data
- Only CO2 measurements from KnudJepsen
- No light intensity data

### 6. **Key Issues for Pipeline**

1. **Extreme Sparsity**: With 91% empty rows and most sensors having >95% NULL values, traditional time-series analysis will be challenging

2. **Sensor Islands**: Data appears in small "islands" - typically single minutes of data separated by hours of empty readings

3. **Missing Critical Sensors**: Light intensity (crucial for plant growth) has only 1,520 readings out of 1.45M rows (0.1% coverage)

4. **Inconsistent Sources**: Different files provide different subsets of sensors with no overlap in many cases

5. **Time Period Gaps**: Complete absence of certain sensor types for entire years

## Additional Critical Issues

### 7. **Data Quality Problems**
- **Light Intensity Corruption**: Light intensity values contain extreme outliers (up to 10^39 μmol)
  - Only 934 out of 1,520 values are in reasonable range (0-10,000 μmol)
  - 47 values are extreme outliers
  - 539 values are exactly 0
- **Sensor Value Ranges**:
  - Air Temperature: -9.4°C to 54.8°C (reasonable)
  - CO2: 0 to 1,232 ppm (reasonable)
  - Radiation: 0 to 1,091 W/m² (reasonable)

### 8. **Best Available Data Periods**
Analysis shows the best continuous data coverage occurs in:
- **June 2014**: Up to 144 readings/day across multiple sensors
- **February-March 2014**: Good coverage with 130-140 readings/day
- **December 2013**: Moderate coverage with ~138 readings/day

Even in the best periods, data is collected only ~6 times per hour (every 10 minutes) rather than continuously.

### 9. **Multi-Sensor Availability**
- Only 4,786 hours (out of 24,289 total) have temperature, humidity, AND CO2 data
- Only 3,772 hours have all basic sensors including radiation
- Light intensity data almost never overlaps with other sensors

## Recommendations

1. **Data Cleaning**: 
   - Remove the 1.3M empty rows before processing
   - Filter out corrupted light intensity values (>10,000 μmol)
   
2. **Focus Period**: Target June 2014 or February-March 2014 for initial pipeline testing

3. **Resampling Strategy**: 
   - Aggregate to hourly intervals minimum
   - Consider daily aggregation for most analyses

4. **Imputation Strategy**:
   - Cannot use traditional time-series imputation due to massive gaps
   - Consider seasonal/daily pattern-based imputation
   - May need to treat as sparse event data rather than continuous time series

5. **Feature Limitations**:
   - Light intensity data is too sparse and corrupted to be reliable
   - Focus on temperature, humidity, CO2, and radiation as primary features

6. **Pipeline Redesign**:
   - Current pipeline assumes dense, continuous data - needs complete rethinking
   - Consider event-based or sparse data processing approaches
   - May need to abandon certain analyses that require continuous data