# Slovenian Traffic Data Legend - English Documentation

## Data Collection Infrastructure

### Monitoring Systems
- **PIC** - Traffic Information Center for State Roads in Slovenia
- **ACB** - Highway Database System
- **ŠM** - Counting Station
- **AŠP** - Automatic Traffic Counter
- **TA** - Traffic Agent

### Counting Configuration
- **(1)** - Counting Lane 1
- **(2)** - Counting Lane 2
- **(A)** - Traffic Section A (Direction A)
- **(B)** - Traffic Section B (Direction B)

## Geographic Codes

### Major Cities
| Code | City |
|------|------|
| LJ | Ljubljana |
| MB | Maribor |
| CE | Celje |
| KR | Kranj |
| MS | Murska Sobota |
| KP | Koper |
| NG | Nova Gorica |

### Country Codes
| Code | Country |
|------|---------|
| SVN/SLO | Slovenia |
| ITA | Italy |
| HUN | Hungary |
| HRV | Croatia |
| AUT | Austria |

## Vehicle Classification System

### Basic Vehicle Classes
| Code | Vehicle Type | Description |
|------|--------------|-------------|
| **A0** | Motorcycles | Two-wheeled motor vehicles |
| **A1** | Passenger Cars | Cars, cars with trailers |
| **A2** | Vans | Combined/multipurpose vehicles |
| **B1** | Light Commercial | Vehicles up to 3.5t |
| **B2** | Medium Trucks | Trucks 3.5t - 7t |
| **B3** | Heavy Trucks | Trucks over 7t |
| **B4** | Trucks with Trailer | Truck combinations |
| **B5** | Semi-trailers | Articulated trucks |
| **C1** | Buses | Standard buses |
| **C2** | City Buses with Trailer | Articulated buses |
| **XX** | Unrecognized | Unclassified vehicles |

### Aggregated Vehicle Categories
| Code | Category | Composition |
|------|----------|-------------|
| **MO** | Motorcycles | A0 |
| **OA** | Passenger Vehicles | A1 + XX (unrecognized) |
| **LT** | Light Commercial | A2 + B1 (up to 3.5t) |
| **ST** | Medium Trucks | B2 (3.5t - 7t) |
| **TT** | Heavy Trucks | B3 (over 7t) |
| **TP** | Trucks with Trailer | B4 |
| **TTP** | Semi-trailers | B5 (articulated) |
| **BUS** | All Buses | C1 + C2 |

### Toll Classification Groups
| Code | Group | Description | Composition |
|------|-------|-------------|-------------|
| **VV** | Vignette Vehicles | Vehicles requiring vignette | A0 + A1 + A2 + B1 + XX |
| **TV** | Toll Vehicles | Vehicles paying distance-based tolls | B2 + B3 + B4 + B5 + C1 + C2 |
| **HV** | Heavy Vehicles | Trucks over 7.5t | B3 + B4 + B5 |
| **LTB** | Light + Buses | Combined category | (A2 + B1) + (C1 + C2) |

## System Status Codes

### Error and Maintenance Codes
| Code | Status | Description |
|------|--------|-------------|
| **X** | Missing Records | Data gaps in recording |
| **E** | Loop Error | Inductive loop malfunction |
| **B** | Loop Break | Physical loop interruption |
| **F** | Frequency Module Error | Signal processing fault |
| **M** | SD Card Error | Storage write failure |
| **P** | Power Outage | Main power loss |
| **L** | Low Battery | Battery depletion |
| **C** | Charging | Battery charging in progress |
| **O** | Cabinet Open | Maintenance access detected |
| **R** | Device Reset | System restart occurred |
| **D** | Daylight Saving | Time adjustment for reports |

### System Measurements
- **Temp** - Internal temperature (°C)
- **UBat** - Battery voltage at terminals (V)
- **USol** - Solar panel charging voltage (V)

## Traffic Flow Metrics

### Speed Measurements
| Code | Metric | Description |
|------|--------|-------------|
| **VA** | Average Speed A1 | Mean speed of passenger cars |
| **VMin** | Minimum Speed | Lowest speed in interval (km/h) |
| **VAvg** | Average Speed | Mean speed all vehicles (km/h) |
| **VMax** | Maximum Speed | Highest speed in interval (km/h) |

### Speed Percentiles
- **v(15%)** - 15th percentile: 15% of vehicles traveled slower than this speed
- **v(50%)** - Median speed: 50% of vehicles traveled slower than this speed
- **v(85%)** - 85th percentile: 85% of vehicles traveled slower than this speed

### Traffic Density Metrics
| Code | Metric | Unit | Description |
|------|--------|------|-------------|
| **GAP** | Average Gap | ms | Time between vehicles |
| **OCC** | Occupancy | ‰ | Road space utilization (per mille) |

## Traffic Volume Indicators

### Daily Traffic Measures
| Code | Indicator | Description |
|------|-----------|-------------|
| **PDP** | Average Daily Traffic | Daily vehicle count average |
| **PLDP** | Annual Average Daily Traffic | Yearly average of daily counts |
| **PQDP** | Quarterly Average Daily Traffic | Quarterly average of daily counts |

## Data Quality Indicators

### Directional Data Status
| Symbol | Meaning |
|--------|---------|
| **(A)(B)!** | Incomplete data both directions |
| **(A)!** | Incomplete data direction A |
| **(B)!** | Incomplete data direction B |
| **(A)→(B)** | Direction A using data from A |
| **(A)←(B)** | Direction A using data from B |

### Counter Configuration Types
| Code | Configuration |
|------|--------------|
| **L1** | Single lane counter (Lane 1 only) |
| **L2** | Single lane counter (Lane 2 only) |
| **L3** | Dual lane counter (Sum of L1 + L2) |

---

*Data specifications by Mikrobit Senzorika d.o.o., Slovenia (2024)*  
*System deployed across Slovenian highway network for continuous traffic monitoring*