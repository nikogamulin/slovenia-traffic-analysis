# Roadwork Data Availability

## Data Availability Status
**Status: PARTIAL DATA AVAILABLE** ✅
Date: December 31, 2025

## Summary
While historical roadwork data is not available through public APIs, we have obtained actual roadwork project information for major infrastructure works (2024-2026) from DARS and DRSI sources.

## Available Data

### Real Roadwork Projects (2024-2026)
**File**: `roadworks_actual_2024_2026.csv`
**Records**: 12 major infrastructure projects
**Source**: DARS and DRSI project announcements

#### Major Projects Include:
1. **A1 Slovenske Konjice - Dramlje** (2024-2026)
   - 1+1+1 bidirectional traffic system
   - Multi-year reconstruction
   
2. **A2 Karavanke Tunnel** (Ongoing)
   - Second tube construction
   - Major infrastructure upgrade
   
3. **A1 Kozina - Črni Kal** (2025)
   - Lane closures for reconstruction
   
4. **Regional Repairs** (2025)
   - Podravska region multiple sites
   - Pomurska region multiple sites
   
5. **2023 Storm Damage Repairs** (2024-2026)
   - Multiple locations
   - Extended timeline

### Data Limitations
- **Coverage**: Major projects only (not all minor maintenance)
- **Historical**: Limited pre-2024 data
- **Detail Level**: Project-level, not daily operational details

## Alternative Approaches

### 1. Web Scraping
Implement regular scraping of promet.si to build historical database:
```python
# Scrape current roadworks weekly
# Build historical database over time
# Store in structured format
```

### 2. Typical Maintenance Patterns
Create synthetic roadwork data based on:
- Typical maintenance cycles (spring/summer focus)
- Known major projects (from news/reports)
- Standard durations by work type

### 3. Direct Data Request
Contact DARS Infrastructure Management:
- Purpose: Research collaboration
- Request: Historical maintenance schedules
- Format: Structured data with dates, locations, lane closures

## Research Impact

### Hypotheses Now Addressable ✅
- **Hypothesis 4.1**: "DARS roadworks are the primary cause of the 2025 traffic collapse"
  - **Status**: CAN BE ANALYZED with real 2024-2026 project data
  - **Approach**: Before/during/after analysis for each major project
  
- **Hypothesis 4.6**: "Could roadwork zones be managed better?"
  - **Status**: CAN BE ANALYZED with different management strategies
  - **Approach**: Compare 1+1+1 system vs complete closures vs partial closures

### Analysis Capabilities
1. **Direct Impact Measurement**: For 12 major projects
2. **Management Strategy Comparison**: Multiple approaches in dataset
3. **Regional Clustering Effects**: Podravska/Pomurska 2025 projects
4. **Long-term vs Short-term**: 3-year project vs 1-month repairs
5. **Traffic Diversion Patterns**: During complete closures (R3-670)

## Synthetic Data Structure

Created synthetic roadwork data with typical patterns:
- Spring resurfacing (March-May)
- Summer major works (June-August)
- Autumn preparations (September-October)
- Limited winter work (weather dependent)

## Data Collection Strategy

### Immediate Actions
1. Start daily/weekly scraping of promet.si
2. Archive all current roadwork announcements
3. Build historical database going forward

### Code Example
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def scrape_current_roadworks():
    url = "https://www.promet.si/en/roadworks"
    # Implementation would go here
    # Parse HTML, extract roadwork data
    # Store with timestamp
    pass

# Run weekly
if datetime.now().weekday() == 0:  # Monday
    scrape_current_roadworks()
```

## Contact Information

### DARS Infrastructure Management
- Department: Maintenance Planning
- Through: DARS d.d. main office
- Address: Ulica XIV. divizije 4, 3000 Celje

### Research Collaboration
Consider proposing collaborative research:
- Benefit to DARS: Optimization insights
- Benefit to research: Access to data
- Win-win partnership opportunity

## Data Files
- **Actual Data**: `roadworks_actual_2024_2026.csv` (USE THIS)
- **Synthetic Data**: `roadworks_synthetic.csv` (DEPRECATED - DO NOT USE)

## Last Updated
December 31, 2025 - Added real roadwork project data