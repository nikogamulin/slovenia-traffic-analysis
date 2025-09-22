# Traffic Data Dictionary
**Generated on:** 2025-09-06 15:40:47
**Total Features:** 22
**Categories:** 11

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Data Quality Summary](#data-quality-summary)
- [Feature Categories](#feature-categories)
  - [Station](#station)
  - [Temporal](#temporal)
  - [Temporal Boolean](#temporal-boolean)
  - [Temporal Cyclical](#temporal-cyclical)
  - [Temporal Lag](#temporal-lag)
  - [Temporal Smoothed](#temporal-smoothed)
  - [Traffic Boolean](#traffic-boolean)
  - [Traffic Derived](#traffic-derived)
  - [Traffic Raw](#traffic-raw)
  - [Weather](#weather)
  - [Weather Boolean](#weather-boolean)
- [Feature Relationships](#feature-relationships)
- [Data Transformations](#data-transformations)

## Dataset Overview

- **Total Records:** 2,000
- **Total Features:** 44
- **Memory Usage:** 0.75 MB
- **Date Range:** 2024-01-01T00:00:00 to 2024-03-24T07:00:00

## Data Quality Summary

- **Duplicate Records:** 0
- **Features with Missing Values:** 5
- **Features with >5% Outliers:** 6

## Feature Categories

### Station

**Count:** 1 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `station_id` | Station ID | object | N/A | Identifier for the traffic measurement station |

### Temporal

**Count:** 4 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `day_of_week` | Day of Week | int64 | day | Day of the week (0=Monday, 6=Sunday) |
| `hour` | Hour of Day | int64 | hour | Hour component of the timestamp (0-23) |
| `month` | Month | int64 | month | Month component of the timestamp (1-12) |
| `timestamp` | Timestamp | datetime64[ns] | datetime | Date and time of the measurement |

### Temporal Boolean

**Count:** 2 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `is_rush_hour` | Is Rush Hour | int64 | N/A | Binary flag indicating rush hour periods (7-9 AM, 4-6 PM) |
| `is_weekend` | Is Weekend | int64 | N/A | Binary flag indicating weekend (Saturday/Sunday) |

### Temporal Cyclical

**Count:** 2 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `hour_cos` | Hour Cosine | float64 | dimensionless | Cosine transformation of hour for cyclical encoding |
| `hour_sin` | Hour Sine | float64 | dimensionless | Sine transformation of hour for cyclical encoding |

### Temporal Lag

**Count:** 1 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `vehicle_count_lag1` | Vehicle Count (Previous Hour) | float64 | vehicles | Vehicle count from the previous hour (lag-1) |

### Temporal Smoothed

**Count:** 1 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `vehicle_count_ma3` | Vehicle Count (3-Hour Moving Average) | float64 | vehicles | 3-hour moving average of vehicle count |

### Traffic Boolean

**Count:** 1 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `is_congested` | Is Congested | int64 | N/A | Binary flag indicating congested conditions (occupancy > 60%) |

### Traffic Derived

**Count:** 3 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `congestion_score` | Congestion Score | float64 | score (0-100) | Composite score indicating congestion level |
| `flow_efficiency` | Flow Efficiency | float64 | (vehicles×km/h)/1000 | Product of speed and volume (throughput measure) |
| `traffic_density` | Traffic Density | float64 | vehicles/(km/h) | Vehicle count divided by average speed (density measure) |

### Traffic Raw

**Count:** 3 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `avg_speed` | Average Speed | float64 | km/h | Average speed of vehicles in km/h |
| `occupancy` | Occupancy Rate | float64 | % | Percentage of time the sensor detects vehicles |
| `vehicle_count` | Vehicle Count | float64 | vehicles | Number of vehicles detected in the time period |

### Weather

**Count:** 3 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `precipitation` | Precipitation | float64 | mm | Amount of rainfall in millimeters |
| `temperature` | Temperature | float64 | °C | Air temperature in Celsius |
| `visibility` | Visibility | float64 | m | Visibility distance in meters |

### Weather Boolean

**Count:** 1 features

| Feature | Display Name | Type | Unit | Description |
|---------|--------------|------|------|-------------|
| `is_rainy` | Is Rainy | int64 | N/A | Binary flag indicating rainy conditions (precipitation > 1mm) |

## Detailed Feature Descriptions

### Station Features

#### `station_id`

**Display Name:** Station ID

**Description:** Identifier for the traffic measurement station

**Data Type:** `object`

**Example Values:** ['A1_KM_125', 'A2_KM_89', 'A3_KM_203']

**Source:** traffic_sensors

**Business Meaning:** Different stations may have distinct traffic patterns

**Related Features:** `station_A1KM125`, `station_A2KM89`, `station_A3KM203`

**Importance Score:** 0.6/1.0

---

### Temporal Features

#### `day_of_week`

**Display Name:** Day of Week

**Description:** Day of the week (0=Monday, 6=Sunday)

**Data Type:** `int64`

**Unit:** day

**Valid Range:** 0 to 6

**Example Values:** [0, 1, 2, 3, 4, 5, 6]

**Transformation:** `extract_dayofweek_from_timestamp`

**Source:** derived_from_timestamp

**Business Meaning:** Distinguishes weekday vs weekend traffic patterns

**Related Features:** `is_weekend`

**Importance Score:** 0.8/1.0

---

#### `hour`

**Display Name:** Hour of Day

**Description:** Hour component of the timestamp (0-23)

**Data Type:** `int64`

**Unit:** hour

**Valid Range:** 0 to 23

**Example Values:** [0, 6, 12, 18, 23]

**Transformation:** `extract_hour_from_timestamp`

**Source:** derived_from_timestamp

**Business Meaning:** Used to identify daily traffic patterns and peak hours

**Importance Score:** 0.9/1.0

---

#### `month`

**Display Name:** Month

**Description:** Month component of the timestamp (1-12)

**Data Type:** `int64`

**Unit:** month

**Valid Range:** 1 to 12

**Example Values:** [1, 3, 6, 9, 12]

**Transformation:** `extract_month_from_timestamp`

**Source:** derived_from_timestamp

**Business Meaning:** Captures seasonal traffic variations

**Importance Score:** 0.6/1.0

---

#### `timestamp`

**Display Name:** Timestamp

**Description:** Date and time of the measurement

**Data Type:** `datetime64[ns]`

**Unit:** datetime

**Source:** traffic_sensors

**Business Meaning:** When the traffic measurement was taken

**Quality Checks:** not_null, chronological_order

**Importance Score:** 1.0/1.0

---

### Temporal Boolean Features

#### `is_rush_hour`

**Display Name:** Is Rush Hour

**Description:** Binary flag indicating rush hour periods (7-9 AM, 4-6 PM)

**Data Type:** `int64`

**Valid Range:** 0 to 1

**Example Values:** [0, 1]

**Transformation:** `1 if (7 <= hour < 9) or (16 <= hour < 18) else 0`

**Source:** derived_from_hour

**Business Meaning:** Rush hours typically show highest traffic volumes

**Related Features:** `hour`, `is_morning_rush`, `is_evening_rush`

**Importance Score:** 0.9/1.0

---

#### `is_weekend`

**Display Name:** Is Weekend

**Description:** Binary flag indicating weekend (Saturday/Sunday)

**Data Type:** `int64`

**Valid Range:** 0 to 1

**Example Values:** [0, 1]

**Transformation:** `1 if day_of_week >= 5 else 0`

**Source:** derived_from_day_of_week

**Business Meaning:** Weekend traffic patterns differ significantly from weekdays

**Related Features:** `day_of_week`

**Importance Score:** 0.8/1.0

---

### Temporal Cyclical Features

#### `hour_cos`

**Display Name:** Hour Cosine

**Description:** Cosine transformation of hour for cyclical encoding

**Data Type:** `float64`

**Unit:** dimensionless

**Valid Range:** -1 to 1

**Transformation:** `cos(2*pi*hour/24)`

**Source:** derived_from_hour

**Business Meaning:** Captures cyclical nature of daily patterns

**Related Features:** `hour_sin`, `hour`

**Importance Score:** 0.7/1.0

---

#### `hour_sin`

**Display Name:** Hour Sine

**Description:** Sine transformation of hour for cyclical encoding

**Data Type:** `float64`

**Unit:** dimensionless

**Valid Range:** -1 to 1

**Transformation:** `sin(2*pi*hour/24)`

**Source:** derived_from_hour

**Business Meaning:** Captures cyclical nature of daily patterns

**Related Features:** `hour_cos`, `hour`

**Importance Score:** 0.7/1.0

---

### Temporal Lag Features

#### `vehicle_count_lag1`

**Display Name:** Vehicle Count (Previous Hour)

**Description:** Vehicle count from the previous hour (lag-1)

**Data Type:** `float64`

**Unit:** vehicles

**Transformation:** `lag(vehicle_count, 1)`

**Source:** derived_from_vehicle_count

**Business Meaning:** Previous traffic conditions influence current state

**Related Features:** `vehicle_count`

**Importance Score:** 0.6/1.0

**Missing Value Handling:** forward_fill_or_drop

---

### Temporal Smoothed Features

#### `vehicle_count_ma3`

**Display Name:** Vehicle Count (3-Hour Moving Average)

**Description:** 3-hour moving average of vehicle count

**Data Type:** `float64`

**Unit:** vehicles

**Transformation:** `rolling_mean(vehicle_count, window=3)`

**Source:** derived_from_vehicle_count

**Business Meaning:** Smoothed traffic trend reduces noise in predictions

**Related Features:** `vehicle_count`

**Importance Score:** 0.5/1.0

**Missing Value Handling:** requires_minimum_window

---

### Traffic Boolean Features

#### `is_congested`

**Display Name:** Is Congested

**Description:** Binary flag indicating congested conditions (occupancy > 60%)

**Data Type:** `int64`

**Valid Range:** 0 to 1

**Example Values:** [0, 1]

**Transformation:** `1 if occupancy / 100 > 0.6 else 0`

**Source:** derived_from_occupancy

**Business Meaning:** Indicates when traffic conditions are significantly impaired

**Related Features:** `occupancy`, `congestion_score`

**Importance Score:** 0.8/1.0

---

### Traffic Derived Features

#### `congestion_score`

**Display Name:** Congestion Score

**Description:** Composite score indicating congestion level

**Data Type:** `float64`

**Unit:** score (0-100)

**Valid Range:** 0 to 100

**Transformation:** `clip(occupancy / avg_speed * 100, 0, 100)`

**Source:** derived_from_occupancy_and_avg_speed

**Business Meaning:** 0-30: free flow, 30-60: moderate, 60+: congested

**Related Features:** `occupancy`, `avg_speed`, `is_congested`

**Importance Score:** 0.8/1.0

---

#### `flow_efficiency`

**Display Name:** Flow Efficiency

**Description:** Product of speed and volume (throughput measure)

**Data Type:** `float64`

**Unit:** (vehicles×km/h)/1000

**Transformation:** `avg_speed * vehicle_count / 1000`

**Source:** derived_from_vehicle_count_and_avg_speed

**Business Meaning:** Higher values indicate better traffic throughput

**Related Features:** `vehicle_count`, `avg_speed`

**Importance Score:** 0.7/1.0

---

#### `traffic_density`

**Display Name:** Traffic Density

**Description:** Vehicle count divided by average speed (density measure)

**Data Type:** `float64`

**Unit:** vehicles/(km/h)

**Transformation:** `vehicle_count / (avg_speed + 1)`

**Source:** derived_from_vehicle_count_and_avg_speed

**Business Meaning:** Higher values indicate more congested conditions

**Related Features:** `vehicle_count`, `avg_speed`, `is_congested`

**Importance Score:** 0.8/1.0

---

### Traffic Raw Features

#### `avg_speed`

**Display Name:** Average Speed

**Description:** Average speed of vehicles in km/h

**Data Type:** `float64`

**Unit:** km/h

**Valid Range:** 0 to 150

**Example Values:** [30, 50, 80, 100, 120]

**Source:** traffic_sensors

**Business Meaning:** Indicates traffic flow efficiency and congestion level

**Related Features:** `traffic_density`, `congestion_score`, `is_congested`

**Quality Checks:** non_negative, speed_limit_check

**Importance Score:** 1.0/1.0

---

#### `occupancy`

**Display Name:** Occupancy Rate

**Description:** Percentage of time the sensor detects vehicles

**Data Type:** `float64`

**Unit:** %

**Valid Range:** 0 to 100

**Example Values:** [5, 15, 30, 60, 85]

**Source:** traffic_sensors

**Business Meaning:** Direct measure of roadway utilization

**Related Features:** `is_congested`, `congestion_score`

**Quality Checks:** percentage_range, logical_consistency

**Importance Score:** 0.9/1.0

---

#### `vehicle_count`

**Display Name:** Vehicle Count

**Description:** Number of vehicles detected in the time period

**Data Type:** `float64`

**Unit:** vehicles

**Valid Range:** 0 to 500

**Example Values:** [20, 50, 100, 150, 200]

**Source:** traffic_sensors

**Business Meaning:** Primary measure of traffic volume

**Related Features:** `traffic_density`, `flow_efficiency`

**Quality Checks:** non_negative, reasonable_range

**Importance Score:** 1.0/1.0

---

### Weather Features

#### `precipitation`

**Display Name:** Precipitation

**Description:** Amount of rainfall in millimeters

**Data Type:** `float64`

**Unit:** mm

**Valid Range:** 0 to 100

**Example Values:** [0, 1, 5, 10, 20]

**Source:** weather_api

**Business Meaning:** Rain significantly impacts traffic speed and congestion

**Related Features:** `is_rainy`, `weather_severity`

**Quality Checks:** non_negative, outlier_detection

**Importance Score:** 0.7/1.0

---

#### `temperature`

**Display Name:** Temperature

**Description:** Air temperature in Celsius

**Data Type:** `float64`

**Unit:** °C

**Valid Range:** -30 to 50

**Example Values:** [-5, 0, 10, 20, 30]

**Source:** weather_api

**Business Meaning:** Temperature affects driving behavior and traffic patterns

**Related Features:** `is_cold`, `is_hot`

**Quality Checks:** range_check, outlier_detection

**Importance Score:** 0.5/1.0

---

#### `visibility`

**Display Name:** Visibility

**Description:** Visibility distance in meters

**Data Type:** `float64`

**Unit:** m

**Valid Range:** 0 to 15000

**Example Values:** [100, 500, 1000, 5000, 10000]

**Source:** weather_api

**Business Meaning:** Low visibility conditions reduce traffic speed for safety

**Related Features:** `is_foggy`, `weather_severity`

**Quality Checks:** non_negative, range_check

**Importance Score:** 0.6/1.0

---

### Weather Boolean Features

#### `is_rainy`

**Display Name:** Is Rainy

**Description:** Binary flag indicating rainy conditions (precipitation > 1mm)

**Data Type:** `int64`

**Valid Range:** 0 to 1

**Example Values:** [0, 1]

**Transformation:** `1 if precipitation > 1.0 else 0`

**Source:** derived_from_precipitation

**Business Meaning:** Rain conditions significantly impact traffic behavior

**Related Features:** `precipitation`, `weather_severity`

**Importance Score:** 0.7/1.0

---

## Feature Relationships

### Derivation Chain

| Feature | Related Features |
|---------|------------------|
| `avg_speed` | `traffic_density`, `congestion_score`, `is_congested` |
| `congestion_score` | `occupancy`, `avg_speed`, `is_congested` |
| `day_of_week` | `is_weekend` |
| `flow_efficiency` | `vehicle_count`, `avg_speed` |
| `hour_cos` | `hour_sin`, `hour` |
| `hour_sin` | `hour_cos`, `hour` |
| `is_congested` | `occupancy`, `congestion_score` |
| `is_rainy` | `precipitation`, `weather_severity` |
| `is_rush_hour` | `hour`, `is_morning_rush`, `is_evening_rush` |
| `is_weekend` | `day_of_week` |
| `occupancy` | `is_congested`, `congestion_score` |
| `precipitation` | `is_rainy`, `weather_severity` |
| `station_id` | `station_A1KM125`, `station_A2KM89`, `station_A3KM203` |
| `temperature` | `is_cold`, `is_hot` |
| `traffic_density` | `vehicle_count`, `avg_speed`, `is_congested` |
| `vehicle_count` | `traffic_density`, `flow_efficiency` |
| `vehicle_count_lag1` | `vehicle_count` |
| `vehicle_count_ma3` | `vehicle_count` |
| `visibility` | `is_foggy`, `weather_severity` |

## Data Transformations

| Feature | Transformation Formula | Source |
|---------|------------------------|--------|
| `congestion_score` | `clip(occupancy / avg_speed * 100, 0, 100)` | derived_from_occupancy_and_avg_speed |
| `day_of_week` | `extract_dayofweek_from_timestamp` | derived_from_timestamp |
| `flow_efficiency` | `avg_speed * vehicle_count / 1000` | derived_from_vehicle_count_and_avg_speed |
| `hour` | `extract_hour_from_timestamp` | derived_from_timestamp |
| `hour_cos` | `cos(2*pi*hour/24)` | derived_from_hour |
| `hour_sin` | `sin(2*pi*hour/24)` | derived_from_hour |
| `is_congested` | `1 if occupancy / 100 > 0.6 else 0` | derived_from_occupancy |
| `is_rainy` | `1 if precipitation > 1.0 else 0` | derived_from_precipitation |
| `is_rush_hour` | `1 if (7 <= hour < 9) or (16 <= hour < 18) else 0` | derived_from_hour |
| `is_weekend` | `1 if day_of_week >= 5 else 0` | derived_from_day_of_week |
| `month` | `extract_month_from_timestamp` | derived_from_timestamp |
| `traffic_density` | `vehicle_count / (avg_speed + 1)` | derived_from_vehicle_count_and_avg_speed |
| `vehicle_count_lag1` | `lag(vehicle_count, 1)` | derived_from_vehicle_count |
| `vehicle_count_ma3` | `rolling_mean(vehicle_count, window=3)` | derived_from_vehicle_count |

## Data Quality Issues

### Missing Values

| Feature | Missing Count | Missing % |
|---------|---------------|-----------|
| `vehicle_count_ma3` | 2 | 0.1% |
| `avg_speed_ma3` | 2 | 0.1% |
| `vehicle_count_lag1` | 1 | 0.05% |
| `avg_speed_lag1` | 1 | 0.05% |
| `occupancy_lag1` | 1 | 0.05% |

### Outliers (>1%)

| Feature | Outlier Count | Outlier % |
|---------|---------------|-----------|
| `is_rush_hour` | 333 | 16.65% |
| `is_windy` | 300 | 15.0% |
| `is_cold` | 296 | 14.8% |
| `is_hot` | 294 | 14.7% |
| `is_morning_rush` | 167 | 8.35% |
| `is_evening_rush` | 166 | 8.3% |
| `wind_speed` | 93 | 4.65% |
| `is_holiday` | 90 | 4.5% |
| `is_congested` | 84 | 4.2% |
| `precipitation` | 79 | 3.95% |
| `traffic_density` | 42 | 2.1% |
| `is_foggy` | 39 | 1.95% |
| `congestion_score` | 30 | 1.5% |
| `avg_speed_ma3` | 21 | 1.05% |

## Usage Guidelines

### Best Practices

1. **Temporal Features**: Use cyclical encoding (sin/cos) for hour, day, and month features to capture periodic patterns.
2. **Weather Impact**: Combine weather boolean flags for comprehensive weather condition assessment.
3. **Traffic Metrics**: Use derived metrics (density, flow_efficiency) for better model performance.
4. **Missing Values**: Handle lag and rolling features appropriately - they naturally have missing values at the beginning.
5. **Outliers**: Review outliers in traffic metrics as they may indicate incidents or sensor malfunctions.

### Feature Selection Recommendations

**High Importance Features (Score ≥ 0.8):**

- `timestamp` (1.0): When the traffic measurement was taken
- `vehicle_count` (1.0): Primary measure of traffic volume
- `avg_speed` (1.0): Indicates traffic flow efficiency and congestion level
- `hour` (0.9): Used to identify daily traffic patterns and peak hours
- `occupancy` (0.9): Direct measure of roadway utilization
- `is_rush_hour` (0.9): Rush hours typically show highest traffic volumes
- `day_of_week` (0.8): Distinguishes weekday vs weekend traffic patterns
- `traffic_density` (0.8): Higher values indicate more congested conditions
- `congestion_score` (0.8): 0-30: free flow, 30-60: moderate, 60+: congested
- `is_weekend` (0.8): Weekend traffic patterns differ significantly from weekdays
- `is_congested` (0.8): Indicates when traffic conditions are significantly impaired

---

*Documentation generated automatically on 2025-09-06 15:40:47*
