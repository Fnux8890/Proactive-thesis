# Data Exploration for Aarslev Sensor & Forecast Data

This document explains the structure of the Aarslev dataset available for January–February 2014 and outlines how you can use a Python Notebook to explore and analyze the data. The dataset consists of several files in different formats, providing both forecasted and measured weather parameters.

---

## Overview of Data Files

### 1. `data_jan_feb_2014.csv`

- **Format:** CSV (semicolon-delimited)
- **Columns:**
  - `timestamp`: Epoch time in milliseconds.
  - `temperature_forecast`: Forecasted temperature values.
  - `sun_radiation_forecast`: Forecasted sun radiation values.
  - `sun_radiation`: Measured sun radiation values.
  - `temperature`: Measured temperature values.
- **Description:**  
  This file contains time series data representing both forecasted and observed weather parameters. It is useful for comparing model forecasts with actual measurements.

---

### 2. `temperature_sunradiation_jan_feb_2014.json`

- **Format:** JSON
- **Structure:**
  - Each JSON object includes:
    - `uuid`: A unique identifier for a sensor or data source.
    - `Readings`: A list of pairs `[timestamp, value]` representing the sensor readings.
- **Description:**  
  This file holds raw time series readings (likely for temperature or sun radiation) for one or more sensors identified by their unique identifiers.

---

### 3. `temperature_sunradiation_jan_feb_2014.json.csv`

- **Format:** CSV (semicolon-delimited)
- **Columns:**
  - `timestamp`: Epoch time in milliseconds.
  - One or more columns named after sensor UUIDs (e.g., `5f893ebd-002c-3708-8e76-bb519336f210`, `be3ddbb8-5210-3be3-84d8-904df2da1754`), each containing the corresponding sensor's readings.
- **Description:**  
  This file is a CSV representation of the JSON data, which is more convenient for tabular analysis. It allows you to easily compare sensor values across time.

---

## Goals and Potential Analyses

Using a Python Notebook, you can explore and analyze the Aarslev dataset to:

- **Visualize Temporal Trends:**  
  Plot temperature and sun radiation over time to inspect patterns, trends, or seasonal effects.
- **Compare Forecast vs. Measured Values:**  
  Evaluate how forecasted values compare with the actual sensor measurements.
- **Merge and Integrate Data:**  
  Combine data from different sources (e.g., the CSV file and JSON CSV file) to create a comprehensive view of the weather conditions.
- **Anomaly Detection:**  
  Identify missing, unexpected, or outlier readings in the data.
- **Statistical Analysis and Summaries:**  
  Compute summary statistics (mean, median, standard deviation) and analyze correlations between different weather parameters.

---

## Setting Up Your Python Environment

To begin your analysis in a Python Notebook, follow these steps on your Windows 11 machine using Powershell and `uv` for package management:

1. **Create a Virtual Environment:**

    ```powershell
    uv venv
    ```

2. **Install Necessary Packages:**

    ```powershell
    uv pip install pandas matplotlib seaborn jupyterlab
    ```

3. **Launch Jupyter Lab:**

    ```powershell
    jupyter lab
    ```

---

## Sample Python Notebook Workflow

Below is a sample workflow to guide your analysis:

```python
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Read the Main CSV Data File
csv_file = Path("Data/aarslev/data_jan_feb_2014.csv")
df = pd.read_csv(csv_file, delimiter=';')

# Convert the timestamp from milliseconds to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

# Display basic information and the first few rows
print("DataFrame Shape:", df.shape)
print(df.head())

# Visualize Temperature Trends
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['temperature'], label='Measured Temperature')
plt.plot(df['datetime'], df['temperature_forecast'], label='Forecast Temperature', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Trend - January to February 2014')
plt.legend()
plt.show()

# Visualize Sun Radiation Trends
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['sun_radiation'], label='Measured Sun Radiation')
plt.plot(df['datetime'], df['sun_radiation_forecast'], label='Forecast Sun Radiation', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Sun Radiation')
plt.title('Sun Radiation Trend - January to February 2014')
plt.legend()
plt.show()

# Exploring the Sensor Data (JSON CSV)
json_csv_file = Path("Data/aarslev/temperature_sunradiation_jan_feb_2014.json.csv")
df_sensors = pd.read_csv(json_csv_file, delimiter=';')

# Convert the timestamp to datetime
df_sensors['datetime'] = pd.to_datetime(df_sensors['timestamp'], unit='ms')

# Display a preview of the sensor data
print("Sensor DataFrame Shape:", df_sensors.shape)
print(df_sensors.head())

# Optionally, visualize one of the sensor readings
sensor_id = df_sensors.columns[1]  # e.g., the first sensor column after timestamp
plt.figure(figsize=(12, 6))
plt.plot(df_sensors['datetime'], df_sensors[sensor_id], label=f'Sensor {sensor_id}')
plt.xlabel('Time')
plt.ylabel('Sensor Reading')
plt.title(f'Sensor {sensor_id} Readings - January to February 2014')
plt.legend()
plt.show()
```

---

## Extending Your Analysis

Once you have loaded and visualized the basic trends, consider the following additional analyses:

- **Data Cleaning:**  
  Identify missing or anomalous values and apply necessary corrections.
- **Resampling:**  
  If the data is too granular, try resampling (e.g., hourly or daily averages) to observe long-term trends.
- **Merging Datasets:**  
  Join the sensor data with forecast data based on timestamps to analyze discrepancies between actual measurements and predictions.
- **Correlation Analysis:**  
  Compute correlations between temperature, sun radiation, and their forecasted values to assess model accuracy.

---

## Conclusion

This guide provides a starting point for exploring the Aarslev dataset using a Python Notebook. With the rich combination of forecast and measured data available in multiple formats, you have ample opportunity to perform comprehensive exploratory data analysis and prepare the data for further modeling or integration into larger data processing pipelines.
