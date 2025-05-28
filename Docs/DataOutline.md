# Data Outline - Aarslev Greenhouse Data

## Folder Structure

```
Data/
├── aarslev/
│   ├── december2013/
│   │   └── MortenSDUData.csv
│   ├── januar2014/
│   │   └── MortenSDUData.csv
│   ├── februar2014/
│   │   └── MortenSDUData.csv
│   ├── marts2014/
│   │   └── MortenSDUData.csv
│   ├── april2014/
│   │   └── MortenSDUData.csv
│   ├── maj2014/
│   │   └── MortenSDUData.csv
│   ├── juni2014/
│   │   └── MortenSDUData.csv
│   ├── august2014/
│   │   └── MortenSDUData.csv
│   ├── september2014/
│   │   └── MortenSDUData.csv
│   └── data_jan_feb_2014.csv
```

## CSV Files

### Monthly Greenhouse Condition Data (MortenSDUData.csv)

These files contain hourly measurements of greenhouse conditions, arranged chronologically.

**Location**: In each month-specific folder (december2013, januar2014, etc.)

**Format**:

```
Start,End,Celle 6.RH Zone1 (Celle 6) [RH%],Celle 6.CO2 (Celle 6) [ppm],Celle 6.CO2 status (Celle 6) [],Celle 6.Luft temp (Celle 6) [°C],Celle 6.Flow1 (Celle 6) [°C]
2014-04-01 00:00,2014-04-01 00:59,52.5,667.5,0.0,21.2,40.7
...
```

**Columns**:

1. `Start` - Start timestamp (YYYY-MM-DD HH:MM)
2. `End` - End timestamp (YYYY-MM-DD HH:MM)
3. `Celle 6.RH Zone1 (Celle 6) [RH%]` - Relative humidity percentage
4. `Celle 6.CO2 (Celle 6) [ppm]` - Carbon dioxide level in ppm
5. `Celle 6.CO2 status (Celle 6) []` - CO2 status indicator
6. `Celle 6.Luft temp (Celle 6) [°C]` - Air temperature in Celsius
7. `Celle 6.Flow1 (Celle 6) [°C]` - Flow temperature in Celsius

**Available Months**:

- December 2013
- January 2014
- February 2014
- March 2014 (Note: Contains anomalous readings around March 6)
- April 2014
- May 2014
- June 2014 (Note: Contains repeated values from June 22-30)
- August 2014
- September 2014

**Notes**:

- Each file contains hourly data for the entire month
- Files typically contain between 670-750 records
- Data is comma-separated
- Each record represents one hour of measurements

### Weather Forecast Data (data_jan_feb_2014.csv)

Contains weather forecasts and actual measurements for January-February 2014.

**Location**: `Data/aarslev/data_jan_feb_2014.csv`

**Format**:

```
timestamp;temperature_forecast;sun_radiation_forecast;sun_radiation;temperature
1390518000000;-4.250;0.000;0.000;-3.864
...
```

**Columns**:

1. `timestamp` - Unix timestamp in milliseconds
2. `temperature_forecast` - Forecasted temperature
3. `sun_radiation_forecast` - Forecasted solar radiation
4. `sun_radiation` - Actual solar radiation
5. `temperature` - Actual temperature

**Notes**:

- Contains approximately 530 records
- Data is semicolon-separated
- Timestamps are in Unix milliseconds format
- Data spans January-February 2014

## Data Overview

This collection provides detailed climate monitoring of greenhouse conditions ("Celle 6") throughout most of 2014, with hourly measurements of:

- Relative humidity
- CO2 concentration
- Temperature (air and flow)
- CO2 status indicators

The weather forecast data provides additional external weather context for January-February 2014.
